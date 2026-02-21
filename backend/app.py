"""
WiQAS Backend API - FastAPI Application

Main FastAPI application with CORS, lifespan events, and route registration.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.dependencies import cleanup_resources, initialize_resources

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ========== LIFESPAN EVENTS ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.
    
    Startup:
        - Initialize configuration
        - Warm up retriever and generator (optional)
        - Check Ollama connection
        
    Shutdown:
        - Cleanup resources
        - Close database connections
    """
    logger.info("🚀 Starting WiQAS Backend API...")
    
    try:
        # Initialize shared resources
        initialize_resources()
        logger.info("✅ Resources initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize resources: {e}")
        raise
    
    # Application is running
    yield
    
    # Shutdown: cleanup resources
    logger.info("🛑 Shutting down WiQAS Backend API...")
    cleanup_resources()
    logger.info("✅ Cleanup complete")


# ========== FASTAPI APP ==========
app = FastAPI(
    title="WiQAS RAG Assistant API",
    description="Backend API for WiQAS - Local RAG Assistant with chat capabilities",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ========== CORS MIDDLEWARE ==========
# Allow frontend to communicate with backend
# For GCP/Cloud deployment: Set CORS_ORIGINS environment variable
# Example: CORS_ORIGINS=http://your-vm-ip:3000,http://your-vm-ip:5173
import os

cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env:
    # Parse comma-separated origins from environment variable
    cors_origins = [origin.strip() for origin in cors_origins_env.split(",")]
    logger.info(f"Using CORS origins from environment: {cors_origins}")
else:
    # Default to localhost for development
    cors_origins = [
        "http://localhost:3000",  # Svelte dev server
        "http://34.124.143.216:3000",  # Vite dev server
        "http://localhost:8080",  # Alternative frontend port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ]
    logger.info("Using default CORS origins for local development")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["X-Total-Count", "X-Page-Size"],  # Custom headers for pagination
)


# ========== EXCEPTION HANDLERS ==========
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for uncaught errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "type": type(exc).__name__,
        },
    )


# ========== ROOT ENDPOINT ==========
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information.
    """
    return {
        "name": "WiQAS RAG Assistant API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "health": "/health",
            "status": "/api/status",
            "config": "/api/config",
            "rag": "/api/ask, /api/query",
            "ingestion": "/api/ingest/*",
        },
    }


# ========== ROUTE REGISTRATION ==========
# Import and include routers
from backend.api import health, ingestion, models, rag, settings

app.include_router(settings.router)
app.include_router(ingestion.router)
app.include_router(rag.router)
app.include_router(models.router)
app.include_router(health.router)


# ========== WEBSOCKET ROUTES ==========
from fastapi import WebSocket

from backend.dependencies import (
    get_config_dependency,
    get_generator_dependency,
)
from backend.websockets.chat import chat_websocket_endpoint


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat.
    
    Compatible with HuggingFace chat-ui and standard WebSocket clients.
    
    Args:
        websocket: WebSocket connection
        session_id: Chat session identifier
    
    Example usage (JavaScript):
        const ws = new WebSocket('ws://localhost:8000/ws/chat/abc123');
        
        ws.onopen = () => {
            ws.send(JSON.stringify({
                type: 'message',
                content: 'What is machine learning?',
                data: { rag_mode: 'auto', include_sources: true }
            }));
        };
        
        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'token') {
                console.log(msg.content);
            }
        };
    """
    # Get dependencies manually (WebSocket doesn't support Depends)
    generator = get_generator_dependency(get_config_dependency())
    config = get_config_dependency()
    
    await chat_websocket_endpoint(
        websocket=websocket,
        session_id=session_id,
        generator=generator,
        config=config,
    )


# ========== STARTUP MESSAGE ==========
@app.on_event("startup")
async def startup_message():
    """Print startup information."""
    logger.info("=" * 60)
    logger.info("WiQAS Backend API is ready!")
    logger.info("Docs: http://localhost:8000/docs")
    logger.info("=" * 60)


if __name__ == "__main__":
    import uvicorn
    
    # Run with: python -m backend.app
    # Or: uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
