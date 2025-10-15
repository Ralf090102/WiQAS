"""
WiQAS Document Retriever

A simple, high-level interface for document retrieval that encapsulates
the entire WiQAS search pipeline into a single query method.

Usage:
    from src.retrieval.retriever import WiQASRetriever

    retriever = WiQASRetriever()
    results = retriever.query("What is bayanihan?")
    print(results)
"""

import logging
import time

from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.reranker import Document, RerankerManager
from src.retrieval.search import HybridSearcher, KeywordSearcher, MMRSearcher, SearchResult, SemanticSearcher
from src.retrieval.vector_store import ChromaVectorStore
from src.utilities.config import TimingBreakdown, WiQASConfig
from src.utilities.utils import log_error, log_info, log_warning

# Set up logging
logger = logging.getLogger(__name__)


class WiQASRetriever:
    """
    High-level document retriever for WiQAS.

    Provides a simple interface to query the knowledge base using the full
    WiQAS retrieval pipeline: search → rerank → MMR → format results.
    """

    def __init__(self, config: WiQASConfig | None = None):
        """
        Initialize the WiQAS retriever.

        Args:
            config: Optional WiQAS configuration. If None, loads default config.
        """
        self.config = config or WiQASConfig()
        self._vector_store = None
        self._embedding_manager = None
        self._reranker = None
        self._mmr_searcher = None
        self._initialized = False

    def _initialize_components(self):
        """Lazy initialization of retrieval components."""
        if self._initialized:
            return

        try:
            log_info("Initializing WiQAS retriever components...")

            # Initialize core components
            self._vector_store = ChromaVectorStore(self.config)
            self._embedding_manager = EmbeddingManager(self.config)

            # Initialize reranker
            self._reranker = RerankerManager(self.config.rag.reranker)

            # Initialize MMR searcher
            self._mmr_searcher = MMRSearcher(self._embedding_manager, self.config)

            self._initialized = True
            log_info("WiQAS retriever components initialized successfully")

        except Exception as e:
            log_error(f"Failed to initialize retriever components: {e}")
            raise

    def _check_knowledge_base(self) -> int:
        """
        Check if the knowledge base has documents.

        Returns:
            Number of documents in the knowledge base.

        Raises:
            ValueError: If no documents are found.
        """
        stats = self._vector_store.get_collection_stats()
        doc_count = stats.get("document_count", 0)

        if doc_count == 0:
            raise ValueError("No documents found in knowledge base. " "Please run ingestion first using 'python run_retrieval.py ingest <path>'")

        log_info(f"Found {doc_count} documents in knowledge base")
        return doc_count

    def _perform_search(self, query: str, k: int, search_type: str) -> list[SearchResult]:
        """
        Perform the initial search (semantic or hybrid).

        Args:
            query: Search query
            k: Number of results to retrieve
            search_type: Type of search ('semantic' or 'hybrid')

        Returns:
            List of search results
        """
        if search_type == "semantic":
            searcher = SemanticSearcher(self._embedding_manager, self._vector_store, self.config)
            return searcher.search(query, k=k)

        elif search_type == "hybrid":
            # Create semantic and keyword searchers
            semantic_searcher = SemanticSearcher(self._embedding_manager, self._vector_store, self.config)
            keyword_searcher = KeywordSearcher(self.config)

            # Get documents from vector store for keyword indexing
            doc_count = self._check_knowledge_base()
            all_results = semantic_searcher.search(query, k=doc_count)

            # Prepare data for keyword search
            documents = [r.content for r in all_results]
            document_ids = [r.document_id for r in all_results]
            metadatas = [r.metadata for r in all_results]

            # Index documents for keyword search
            keyword_searcher.index_documents(documents, document_ids, metadatas)

            # Create hybrid searcher and search
            hybrid_searcher = HybridSearcher(semantic_searcher, keyword_searcher, self.config)
            return hybrid_searcher.search(query, k=k)

        else:
            raise ValueError(f"Unsupported search type: {search_type}. Use 'semantic' or 'hybrid'.")

    def _apply_reranking(self, query: str, results: list[SearchResult], k: int, llm_analysis: bool = True) -> list[SearchResult]:
        """
        Apply reranking to search results.

        Args:
            query: Original search query
            results: Initial search results
            k: Number of top results to return
            llm_analysis: Whether to use LLM-based cultural analysis

        Returns:
            Reranked search results
        """
        if not results:
            return results

        # Create a copy of reranker config if we need to disable LLM analysis
        reranker_config = self.config.rag.reranker
        if not llm_analysis:
            from dataclasses import replace

            reranker_config = replace(reranker_config, use_llm_cultural_analysis=False, score_threshold=0.0)
            reranker = RerankerManager(reranker_config)
        else:
            reranker = self._reranker

        # Convert SearchResult objects to Document objects for reranking
        docs_to_rerank = []
        for result in results:
            doc = Document(content=result.content, metadata=result.metadata, score=result.score, doc_id=result.document_id)
            docs_to_rerank.append(doc)

        # Rerank documents
        reranked_docs = reranker.rerank_documents(query, docs_to_rerank, top_k=k)

        # Convert back to SearchResult objects
        reranked_results = []
        for doc in reranked_docs:
            result = SearchResult(
                document_id=doc.doc_id,
                content=doc.content,
                metadata=doc.metadata,
                score=doc.score,
                search_type=f"{results[0].search_type}_reranked" if results else "reranked",
            )
            reranked_results.append(result)

        return reranked_results

    def _apply_mmr(self, query: str, results: list[SearchResult], k: int) -> list[SearchResult]:
        """
        Apply MMR diversity search to results.

        Args:
            query: Original search query
            results: Search results to diversify
            k: Number of diverse results to return

        Returns:
            Diversified search results
        """
        if not results or len(results) <= 1:
            return results

        # Apply MMR to get diverse subset
        mmr_results = self._mmr_searcher.search(query, candidate_results=results, k=k)

        # Update search_type to indicate MMR was applied
        for result in mmr_results:
            if result.search_type and not result.search_type.endswith("_mmr"):
                result.search_type += "_mmr"

        return mmr_results

    def _format_results(self, results: list[SearchResult]) -> str:
        """
        Format search results into a readable string.

        Args:
            results: Search results to format

        Returns:
            Formatted results string
        """
        if not results:
            return "No results found for your query."

        formatted_parts = [f"Found {len(results)} relevant results:\n"]

        for i, result in enumerate(results, 1):
            # Basic result info
            source = result.metadata.get("source_file", "Unknown")
            file_type = result.metadata.get("file_type", "Unknown")
            score = result.score

            # Chunk information
            chunk_info = ""
            if "chunk_index" in result.metadata:
                chunk_idx = result.metadata["chunk_index"] + 1
                chunk_total = result.metadata.get("chunk_total", "Unknown")
                chunk_info = f" (Chunk {chunk_idx}/{chunk_total})"

            # Format header
            formatted_parts.append(f"\n--- Result {i} (Score: {score:.4f}) ---")
            formatted_parts.append(f"Source: {source}{chunk_info}")
            formatted_parts.append(f"Type: {file_type}")

            # Content preview (limit to 300 characters for readability)
            content = result.content.strip()
            if len(content) > 300:
                content = content[:300] + "..."

            formatted_parts.append(f"Content: {content}")
            formatted_parts.append("")

        return "\n".join(formatted_parts)

    def query(
        self,
        query_text: str,
        k: int = 5,
        search_type: str = "hybrid",
        enable_reranking: bool = True,
        enable_mmr: bool = True,
        llm_analysis: bool = True,
        formatted: bool = True,
        include_timing: bool = False,
    ) -> str:
        """
        Query the knowledge base and return formatted results.

        Args:
            query_text: The search query string
            k: Number of results to return (default: 5)
            search_type: Type of search - 'semantic' or 'hybrid' (default: 'hybrid')
            enable_reranking: Whether to apply reranking (default: True)
            enable_mmr: Whether to apply MMR diversity (default: True)
            llm_analysis: Whether to use LLM-based cultural analysis (default: True)
            formatted: Whether to return formatted string or raw results (default: True)
            include_timing: Whether to include timing breakdown in results (default: False)

        Returns:
            Formatted string containing search results and optionally timing breakdown

        Raises:
            ValueError: If knowledge base is empty or search type is invalid
            RuntimeError: If retrieval components fail to initialize
        """
        try:
            timing = TimingBreakdown()

            self._initialize_components()

            # Check knowledge base
            doc_count = self._check_knowledge_base()

            log_info(f"Knowledge base contains {doc_count} documents")
            log_info(f"Querying: '{query_text}' (type: {search_type}, k: {k})")

            # Track embedding time
            embedding_start = time.time()
            if search_type in ["semantic", "hybrid"]:
                # Time the query embedding generation
                self._embedding_manager.encode_single(query_text)
            else:
                # For non-semantic searches, embedding time is 0
                pass
            timing.embedding_time = time.time() - embedding_start

            # Perform initial search (includes vector similarity computation)
            search_start = time.time()
            results = self._perform_search(query_text, k=k, search_type=search_type)
            timing.search_time = time.time() - search_start

            if not results:
                log_warning(f"No initial results found for query: {query_text}")
                return "No results found for your query."

            log_info(f"Initial search returned {len(results)} results")

            # Apply reranking if enabled
            if enable_reranking and results:
                log_info("Applying reranking...")
                rerank_start = time.time()
                results = self._apply_reranking(query_text, results, k=k, llm_analysis=llm_analysis)
                timing.reranking_time = time.time() - rerank_start
                log_info(f"Reranking returned {len(results)} results")

            # Apply MMR diversity if enabled
            if enable_mmr and results and len(results) > 1:
                log_info("Applying MMR diversity...")
                mmr_start = time.time()
                results = self._apply_mmr(query_text, results, k=k)
                timing.mmr_time = time.time() - mmr_start
                log_info(f"MMR returned {len(results)} diverse results")

            # Total time
            timing.total_time = timing.embedding_time + timing.search_time + timing.reranking_time + timing.mmr_time

            if formatted:
                # Format and return results
                formatted_results = self._format_results(results)

                if include_timing:
                    # Add timing breakdown to the results
                    formatted_results += f"\n\n{timing.format_timing_summary()}"

                return formatted_results

            log_info(f"Query completed successfully, returning {len(results)} results")

            if include_timing:
                return {"results": results, "timing": timing}

            return results

        except ValueError as e:
            # User-facing errors (empty knowledge base, invalid search type)
            log_warning(f"Query failed: {e}")
            return f"Error: {e}"

        except Exception as e:
            # System errors
            log_error(f"Unexpected error during query: {e}")
            return f"An error occurred while processing your query: {e}"

    def get_status(self) -> dict:
        """
        Get the current status of the retriever and knowledge base.

        Returns:
            Dictionary containing status information
        """
        try:
            self._initialize_components()
            stats = self._vector_store.get_collection_stats()

            return {
                "initialized": self._initialized,
                "document_count": stats.get("document_count", 0),
                "collection_stats": stats,
                "config": {
                    "embedding_model": self.config.rag.embedding.model,
                    "llm_model": self.config.rag.llm.model,
                    "reranking_enabled": self.config.rag.retrieval.enable_reranking,
                    "chunk_size": self.config.rag.chunking.chunk_size,
                },
            }
        except Exception as e:
            return {"initialized": False, "error": str(e), "document_count": 0}


# Convenience function for simple usage
def query_knowledge_base(query: str, **kwargs) -> str:
    """
    Convenience function to query the knowledge base with default settings.

    Args:
        query: Search query string
        **kwargs: Additional arguments passed to WiQASRetriever.query()

    Returns:
        Formatted search results string
    """
    retriever = WiQASRetriever()
    return retriever.query(query, **kwargs)
