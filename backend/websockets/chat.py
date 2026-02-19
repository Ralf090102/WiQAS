"""
WebSocket Chat Handler

Real-time bidirectional chat via WebSocket with robust token streaming.
Compatible with HuggingFace chat-ui and other WebSocket clients.
"""

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect, status

from src.generation.generator import WiQASGenerator
from src.utilities.config import WiQASConfig

logger = logging.getLogger(__name__)


class ChatWebSocketHandler:
    """Handler for WebSocket chat connections with robust token streaming."""
    
    def __init__(
        self,
        websocket: WebSocket,
        session_id: str,
        generator: WiQASGenerator,
        config: WiQASConfig,
    ):
        """
        Initialize WebSocket handler.
        
        Args:
            websocket: WebSocket connection
            session_id: Chat session identifier
            session_manager: Session manager instance
            generator: Answer generator instance
            config: Configuration instance
        """
        self.websocket = websocket
        self.session_id = session_id
        self.generator = generator
        self.config = config
        self.connected = False
        
        # Token streaming queue (fixes fire-and-forget issues)
        self.token_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._queue_task: asyncio.Task | None = None
        self._is_processing = False
    
    async def _process_token_queue(self):
        """
        Process tokens from the queue and send them to the client.
        
        This runs in a dedicated task to ensure tokens are sent in order
        without race conditions or dropped messages.
        """
        logger.debug(f"Token queue processor started for session {self.session_id}")
        
        try:
            while self.connected or not self.token_queue.empty():
                try:
                    # Get token with timeout to allow checking connected status
                    token = await asyncio.wait_for(
                        self.token_queue.get(),
                        timeout=0.1
                    )
                    
                    # None is sentinel value for "done streaming"
                    if token is None:
                        logger.debug("Received end-of-stream signal")
                        break
                    
                    # Send token to client
                    await self.send_message(message_type="token", content=token)
                    self.token_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # No token available, loop again to check connection status
                    continue
                except Exception as e:
                    logger.error(f"Error processing token: {e}", exc_info=True)
                    break
        
        finally:
            logger.debug(f"Token queue processor stopped for session {self.session_id}")
            self._is_processing = False
    
    def queue_token(self, token: str):
        """
        Queue a token for sending (thread-safe).
        
        This is called from the sync streaming callback and safely
        adds tokens to the async queue.
        
        Args:
            token: Token content to send
        """
        try:
            # Put token in queue (non-blocking)
            self.token_queue.put_nowait(token)
        except asyncio.QueueFull:
            logger.warning(f"Token queue full, dropping token: {token[:20]}...")
        except Exception as e:
            logger.error(f"Failed to queue token: {e}")
    
    async def start_token_streaming(self):
        """Start the token queue processor task."""
        if not self._is_processing:
            self._is_processing = True
            self._queue_task = asyncio.create_task(self._process_token_queue())
            logger.debug("Started token streaming task")
    
    async def stop_token_streaming(self):
        """Stop the token queue processor and signal end of stream."""
        # Signal end of stream
        await self.token_queue.put(None)
        
        # Wait for queue to finish processing
        if self._queue_task and not self._queue_task.done():
            try:
                await asyncio.wait_for(self._queue_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Token queue processor did not finish in time")
                if not self._queue_task.done():
                    self._queue_task.cancel()
        
        logger.debug("Stopped token streaming task")
    
    async def connect(self) -> bool:
        """
        Accept WebSocket connection and verify session.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            await self.websocket.accept()
            self.connected = True
            
            # Verify session exists
            session = self.session_manager.get_session(self.session_id)
            if not session:
                logger.warning(f"Session not found: {self.session_id}")
                await self.send_error(
                    f"Session not found: {self.session_id}",
                    code=404,
                )
                return False
            
            logger.info(f"WebSocket connected for session: {self.session_id}")
            
            # Send connection success
            await self.send_message(
                message_type="connected",
                data={
                    "session_id": self.session_id,
                    "message": "WebSocket connection established",
                },
            )
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}", exc_info=True)
            self.connected = False
            return False
    
    async def disconnect(self):
        """Close WebSocket connection and cleanup resources."""
        if self.connected:
            try:
                # Stop token streaming first
                if self._is_processing:
                    await self.stop_token_streaming()
                
                # Close WebSocket
                await self.websocket.close()
                logger.info(f"WebSocket disconnected for session: {self.session_id}")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            finally:
                self.connected = False
    
    async def send_message(
        self,
        message_type: str,
        content: str | None = None,
        data: dict[str, Any] | None = None,
    ):
        """
        Send a message to the client.
        
        Args:
            message_type: Message type (token, sources, metadata, done, error)
            content: Message content (optional)
            data: Additional data payload (optional)
        """
        # TODO
    
    async def send_error(self, error_message: str, code: int = 500):
        """
        Send an error message to the client.
        
        Args:
            error_message: Error description
            code: Error code
        """
        await self.send_message(
            message_type="error",
            content=error_message,
            data={"code": code},
        )
    
    async def _maybe_generate_title(self, first_message: str):
        """
        Generate a session title if this is the first user message.
        
        Args:
            first_message: The user's first message
        """
        try:
            # Reload session from database to get latest messages
            session = self.session_manager.get_session(self.session_id, reload_from_db=True)
            if not session:
                return
            
            # Only generate if no title or title is default "New Chat"
            existing_title = session.metadata.get("title", "")
            if existing_title and existing_title != "New Chat":
                logger.debug(f"Session {self.session_id} already has title: '{existing_title}'")
                return
            
            # Only generate for the first exchange (2 messages: 1 user + 1 assistant)
            # Note: We just added both messages, so should have exactly 2
            message_count = len(session.messages)
            logger.debug(f"Session {self.session_id} has {message_count} messages")
            
            if message_count != 2:
                logger.debug(f"Skipping title generation - need exactly 2 messages, have {message_count}")
                return
            
            logger.info(f"Auto-generating title for session {self.session_id}")
            
            # Use LLM to generate a concise title
            title_prompt = f"""Generate a very short, concise title (maximum 6 words) for a conversation that starts with this message:

"{first_message}"

Reply with ONLY the title, nothing else. No quotes, no explanations."""

            messages = [
                {"role": "user", "content": title_prompt}
            ]
            
            # Generate title using LLM (non-streaming)
            from src.core.llm import OllamaClient
            llm = OllamaClient(timeout=10)
            
            response = llm.generate(
                messages=messages,
                model=self.config.rag.llm.model,
                temperature=0.7,
                top_p=0.9,
                max_tokens=20,  # Keep it short
                stream=False,
            )
            
            generated_title = response.get("message", {}).get("content", "").strip()
            
            # Clean up the title (remove quotes if present)
            generated_title = generated_title.strip('"\'').strip()
            
            # Limit length
            if len(generated_title) > 60:
                generated_title = generated_title[:57] + "..."
            
            if generated_title and generated_title != first_message:
                # Update session metadata with generated title
                self.session_manager.update_session_metadata(
                    self.session_id,
                    {"title": generated_title}
                )
                
                logger.info(f"Generated title for session {self.session_id}: '{generated_title}'")
                
                # Optionally send title update to client
                await self.send_message(
                    message_type="title",
                    content=generated_title,
                    data={"session_id": self.session_id}
                )
            
        except Exception as e:
            logger.warning(f"Failed to generate title for session {self.session_id}: {e}")
            # Don't fail the entire request if title generation fails

async def chat_websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    generator: WiQASGenerator,
    config: WiQASConfig,
):
    """
    WebSocket endpoint for real-time chat.
    
    Compatible with HuggingFace chat-ui and standard WebSocket clients.
    
    Args:
        websocket: WebSocket connection
        session_id: Chat session identifier
        session_manager: Session manager instance
        generator: Answer generator instance
        config: Configuration instance
    
    Message format (client → server):
        {
            "type": "message",
            "content": "What is machine learning?",
            "data": {
                "rag_mode": "auto",
                "include_sources": true,
                "temperature": 0.7
            }
        }
    
    Message format (server → client):
        {
            "type": "token",         # or "sources", "metadata", "done", "error"
            "content": "Machine",     # token content (for type="token")
            "data": {...}             # additional data
        }
    """
    handler = ChatWebSocketHandler(
        websocket=websocket,
        session_id=session_id,
        generator=generator,
        config=config,
    )
    
    # Connect and verify session
    if not await handler.connect():
        return
    
    # Start message loop
    await handler.listen()
