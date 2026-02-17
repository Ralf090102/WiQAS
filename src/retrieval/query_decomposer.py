"""
Query Decomposition Module for WiQAS

Decomposes complex queries into simpler sub-queries to improve retrieval accuracy.
Handles multi-part questions, comparative queries, and compound requests.
"""

import time
from typing import Any

from langchain_community.llms import Ollama

from src.utilities.config import WiQASConfig
from src.utilities.gpu_utils import get_gpu_manager, detect_gpu_info


class QueryDecomposer:
    """
    Decomposes complex queries into multiple sub-queries for better retrieval.
    
    Uses LLM-based decomposition to intelligently split queries while maintaining
    context and intent. Supports GPU acceleration for faster inference.
    """

    def __init__(self, config: WiQASConfig):
        """
        Initialize the query decomposer.

        Args:
            config: WiQASConfig instance with query decomposition settings
        """
        self.config = config
        self.decomposition_config = config.rag.query_decomposition
        self.llm_config = config.rag.llm
        self.gpu_config = config.gpu
        
        # Initialize LLM for decomposition
        self._llm = None
        self._device_info = None
        
        # Performance metrics
        self.total_decompositions = 0
        self.total_time = 0.0
        
        if self.decomposition_config.enabled:
            self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize the LLM for query decomposition with GPU support."""
        try:
            # Get GPU device info
            if self.gpu_config.enabled:
                gpu_available, device_name, gpu_info = detect_gpu_info()
                self._device_info = {
                    'gpu_available': gpu_available,
                    'device_name': device_name,
                    'gpu_info': gpu_info
                }
                if gpu_available:
                    print(f"✅ Query Decomposer using GPU: {device_name}")
                else:
                    print("ℹ️ Query Decomposer using CPU (GPU not available)")
            
            # Initialize Ollama LLM
            self._llm = Ollama(
                model=self.decomposition_config.model,
                base_url=self.llm_config.base_url,
                temperature=self.decomposition_config.temperature,
                num_ctx=self.decomposition_config.context_window,
                num_gpu=1 if self.gpu_config.enabled and self._device_info and self._device_info['gpu_available'] else 0,
            )
            
            print(f"✅ Query Decomposer initialized with model: {self.decomposition_config.model}")
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize Query Decomposer LLM: {e}")
            print("ℹ️ Query decomposition will be disabled")
            self.decomposition_config.enabled = False

    def should_decompose(self, query: str) -> bool:
        """
        Determine if a query should be decomposed.

        Args:
            query: Input query string

        Returns:
            True if query should be decomposed, False otherwise
        """
        if not self.decomposition_config.enabled:
            return False
        
        if len(query.split()) < self.decomposition_config.min_query_length:
            return False
        
        # Check for multi-part indicators
        multi_part_indicators = [
            ' and ', ' or ', '?', 'compare', 'difference', 'versus', 'vs',
            'how many', 'what are', 'list', 'different', 'both', 'either',
            'as well as', 'also', 'additionally', 'furthermore'
        ]
        
        query_lower = query.lower()
        indicator_count = sum(1 for indicator in multi_part_indicators if indicator in query_lower)
        
        # Decompose if query has multiple indicators or is sufficiently long
        return indicator_count >= 2 or len(query.split()) > 20

    def decompose(self, query: str) -> tuple[list[str], float]:
        """
        Decompose a complex query into sub-queries.

        Args:
            query: Input query string

        Returns:
            Tuple of (list of sub-queries, decomposition time in seconds)
        """
        start_time = time.time()
        
        # Check if decomposition is needed
        if not self.should_decompose(query):
            decomposition_time = time.time() - start_time
            return [query], decomposition_time
        
        try:
            # Build decomposition prompt
            prompt = self._build_decomposition_prompt(query)
            
            # Call LLM for decomposition
            response = self._llm.invoke(prompt)
            
            # Parse sub-queries from response
            sub_queries = self._parse_sub_queries(response, query)
            
            # Limit number of sub-queries
            if len(sub_queries) > self.decomposition_config.max_sub_queries:
                sub_queries = sub_queries[:self.decomposition_config.max_sub_queries]
            
            decomposition_time = time.time() - start_time
            
            # Update metrics
            self.total_decompositions += 1
            self.total_time += decomposition_time
            
            if self.config.logging.verbose:
                print(f"\n🔍 Query Decomposition ({decomposition_time:.3f}s):")
                print(f"   Original: {query}")
                for i, sq in enumerate(sub_queries, 1):
                    print(f"   Sub-query {i}: {sq}")
            
            return sub_queries, decomposition_time
            
        except Exception as e:
            print(f"⚠️ Query decomposition failed: {e}")
            print(f"ℹ️ Falling back to original query")
            decomposition_time = time.time() - start_time
            return [query], decomposition_time

    def _build_decomposition_prompt(self, query: str) -> str:
        """
        Build the prompt for query decomposition.

        Args:
            query: Original query

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a query decomposition assistant for a Filipino culture and history knowledge base.

Your task is to break down complex questions into simpler sub-questions that can be answered independently.

Guidelines:
1. Preserve the original intent and context
2. Create focused, atomic sub-questions
3. Maintain cultural and historical context
4. Each sub-question should be self-contained
5. Limit to {self.decomposition_config.max_sub_queries} sub-questions maximum
6. If the question is already simple, return it as-is

Original Question: {query}

Decompose this into focused sub-questions. Output ONLY the sub-questions, one per line, without numbering or explanations.

Sub-questions:"""

        return prompt

    def _parse_sub_queries(self, response: str, original_query: str) -> list[str]:
        """
        Parse sub-queries from LLM response.

        Args:
            response: LLM response text
            original_query: Original query (fallback)

        Returns:
            List of sub-queries
        """
        # Split by newlines and clean up
        lines = response.strip().split('\n')
        sub_queries = []
        
        for line in lines:
            # Clean the line
            line = line.strip()
            
            # Skip empty lines, headers, or explanatory text
            if not line or line.lower().startswith(('sub-question', 'question', 'note:', '#')):
                continue
            
            # Remove numbering (1., 2), -, *, etc.)
            import re
            line = re.sub(r'^[\d\-\*\.\)\]]+\s*', '', line)
            line = line.strip()
            
            # Add if it looks like a question
            if line and len(line) > 10:
                sub_queries.append(line)
        
        # Fallback to original query if parsing failed
        if not sub_queries:
            sub_queries = [original_query]
        
        return sub_queries

    def get_stats(self) -> dict[str, Any]:
        """
        Get decomposition statistics.

        Returns:
            Dictionary with performance metrics
        """
        avg_time = self.total_time / self.total_decompositions if self.total_decompositions > 0 else 0.0
        
        return {
            "enabled": self.decomposition_config.enabled,
            "total_decompositions": self.total_decompositions,
            "total_time": self.total_time,
            "average_time": avg_time,
            "gpu_enabled": self.gpu_config.enabled,
            "device_info": self._device_info,
        }

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.total_decompositions = 0
        self.total_time = 0.0
