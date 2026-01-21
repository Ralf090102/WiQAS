"""
Reranker module for WiQAS (With Query Answering System).

This module provides document reranking capabilities using the BAAI/bge-reranker-v2-m3 model,
which pairs well with BGE-M3 embeddings.
"""

import logging
import time
from typing import Any

import torch
from sentence_transformers import CrossEncoder

from src.utilities.config import RerankerConfig
from src.utilities.gpu_utils import get_gpu_manager

logger = logging.getLogger(__name__)


class Document:
    """Document class for reranking operations"""

    def __init__(self, content: str, metadata: dict[str, Any] | None = None, score: float = 0.0, doc_id: str | None = None):
        self.content = content
        self.metadata = metadata or {}
        self.score = score
        self.doc_id = doc_id or ""

    def __repr__(self) -> str:
        return f"Document(content='{self.content[:50]}...', score={self.score:.3f})"


class RerankerManager:
    """
    Manages document reranking using BAAI/bge-reranker-v2-m3 model.

    This reranker provides cross-encoder reranking for semantic relevance scoring.
    """

    def __init__(self, config: RerankerConfig | None = None):
        """
        Initialize the reranker manager.

        Args:
            config: Reranker configuration. If None, uses default configuration.
        """
        self.config = config or RerankerConfig()
        self.model: CrossEncoder | None = None

        # Initialize GPU manager for optimal device selection and memory management
        try:
            from src.utilities.config import WiQASConfig

            wiqas_config = WiQASConfig()
            self.gpu_manager = get_gpu_manager(wiqas_config)
        except Exception:
            self.gpu_manager = get_gpu_manager()

        self._device = str(self.gpu_manager.get_device())

        # Optimize batch size based on available GPU memory
        base_batch_size = self.config.batch_size
        self.batch_size = self.gpu_manager.get_optimal_batch_size(base_batch_size)

        logger.info(f"Initializing RerankerManager with model: {self.config.model}")
        logger.info(f"Using device: {self._device}")
        logger.info(f"Optimized batch size: {self.batch_size} (base: {base_batch_size})")

    def _initialize_model(self) -> None:
        """Initialize the reranker model if not already loaded"""
        if self.model is None:
            try:
                logger.info(f"Loading reranker model: {self.config.model}")
                self.model = CrossEncoder(self.config.model, device=self._device, max_length=512)

                # Enable mixed precision
                if self.gpu_manager.is_nvidia_gpu:
                    logger.info("Mixed precision enabled for GPU inference")
                else:
                    logger.info("Using CPU for reranking")

                logger.info("Reranker model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load reranker model: {e}")
                raise RuntimeError(f"Could not initialize reranker model: {e}")

    def rerank_documents(self, query: str, documents: list[Document], top_k: int | None = None) -> list[Document]:
        """
        Rerank documents based on semantic relevance to the query.

        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top documents to return. If None, uses config.top_k

        Returns:
            List of reranked documents sorted by relevance score
        """
        if not documents:
            logger.warning("No documents provided for reranking")
            return []

        if not query.strip():
            logger.warning("Empty query provided for reranking")
            return documents

        self._initialize_model()

        if self.model is None:
            logger.error("Reranker model not available")
            return documents

        top_k = top_k or self.config.top_k
        logger.info(f"Reranking {len(documents)} documents for query: '{query[:50]}...'")

        try:
            pairs = [(query, doc.content) for doc in documents]

            batch_size = self.batch_size  # Use optimized batch size
            all_scores = []

            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i : i + batch_size]

                # Use mixed precision for GPU inference
                if self.gpu_manager.is_nvidia_gpu:
                    with torch.no_grad():
                        with self.gpu_manager.enable_mixed_precision():
                            batch_scores = self.model.predict(batch_pairs)
                else:
                    with torch.no_grad():
                        batch_scores = self.model.predict(batch_pairs)

                # Ensure each score is a float
                if hasattr(batch_scores, "__iter__") and not isinstance(batch_scores, str):
                    batch_scores = [float(score) for score in batch_scores]
                else:
                    batch_scores = [float(batch_scores)]
                all_scores.extend(batch_scores)

                # Clear GPU cache periodically for large batches
                if self.gpu_manager.is_nvidia_gpu and i > 0 and i % (batch_size * 5) == 0:
                    self.gpu_manager.clear_cache()

            # Create reranked documents with updated scores
            reranked_docs = []
            for doc, rerank_score in zip(documents, all_scores):
                reranked_doc = Document(
                    content=doc.content,
                    metadata=doc.metadata.copy() if doc.metadata else {},
                    score=rerank_score,
                    doc_id=doc.doc_id,
                )

                if reranked_doc.metadata is None:
                    reranked_doc.metadata = {}
                reranked_doc.metadata.update(
                    {
                        "original_score": doc.score,
                        "rerank_score": float(rerank_score),
                        "final_score": rerank_score,
                    }
                )

                reranked_docs.append(reranked_doc)

            # Filter by score threshold
            filtered_docs = [doc for doc in reranked_docs if doc.score >= self.config.score_threshold]

            # Sort by score
            filtered_docs.sort(key=lambda x: x.score, reverse=True)

            # Return top_k results
            result = filtered_docs[:top_k]

            logger.info(f"Reranking complete: {len(result)} documents returned (filtered from {len(documents)} original)")

            if result and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Top reranked documents:")
                for i, doc in enumerate(result[:3]):
                    logger.debug(f"  {i+1}. Score: {doc.score:.3f} - Content: {doc.content[:100]}...")

            return result

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fallback: return original documents sorted by original score
            return sorted(documents, key=lambda x: x.score, reverse=True)[:top_k]

    def rerank_search_results(self, query: str, search_results: list[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Rerank search results from the retrieval pipeline.

        Args:
            query: Search query
            search_results: List of search result dictionaries
            top_k: Number of top results to return

        Returns:
            List of reranked search results
        """
        if not search_results:
            return []

        # Convert search results to Document objects
        documents = []
        for result in search_results:
            doc = Document(
                content=result.get("content", ""),
                metadata=result.get("metadata", {}),
                score=result.get("score", 0.0),
                doc_id=result.get("id", ""),
            )
            documents.append(doc)

        # Rerank documents
        reranked_docs = self.rerank_documents(query, documents, top_k)

        # Convert back to search result format
        reranked_results = []
        for doc in reranked_docs:
            result = {"id": doc.doc_id, "content": doc.content, "metadata": doc.metadata, "score": doc.score}
            reranked_results.append(result)

        return reranked_results

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the reranker model.

        Returns:
            Dictionary containing model information
        """
        gpu_info = self.gpu_manager.get_memory_info() if self.gpu_manager else {}

        info = {
            "model_name": self.config.model,
            "device": self._device,
            "batch_size": self.config.batch_size,
            "optimized_batch_size": self.batch_size,
            "base_batch_size": self.config.batch_size,
            "score_threshold": self.config.score_threshold,
            "model_loaded": self.model is not None,
            "gpu_info": gpu_info,
            "nvidia_gpu_detected": self.gpu_manager.is_nvidia_gpu if self.gpu_manager else False,
        }

        if self.model is not None:
            info["max_length"] = getattr(self.model, "max_length", "unknown")

        return info

    def cleanup(self) -> None:
        """Cleanup resources and GPU memory."""
        try:
            if self.gpu_manager:
                self.gpu_manager.clear_cache()

            if self.model is not None:
                del self.model
                self.model = None

            logger.debug("RerankerManager cleanup completed")

        except Exception as e:
            logger.warning(f"Error during reranker cleanup: {e}")


def create_reranker(config: RerankerConfig | None = None) -> RerankerManager:
    """
    Factory function to create a reranker manager.

    Args:
        config: Reranker configuration. If None, uses default configuration.

    Returns:
        Configured RerankerManager instance
    """
    return RerankerManager(config)


def rerank_documents(query: str, documents: list[Document], config: RerankerConfig | None = None, top_k: int | None = None) -> list[Document]:
    """
    Convenience function to rerank documents.

    Args:
        query: Search query
        documents: List of documents to rerank
        config: Reranker configuration
        top_k: Number of top documents to return

    Returns:
        List of reranked documents
    """
    reranker = create_reranker(config)
    return reranker.rerank_documents(query, documents, top_k)


def rerank_search_results(query: str, search_results: list[dict[str, Any]], config: RerankerConfig | None = None, top_k: int | None = None) -> list[dict[str, Any]]:
    """
    Convenience function to rerank search results.

    Args:
        query: Search query
        search_results: List of search result dictionaries
        config: Reranker configuration
        top_k: Number of top results to return

    Returns:
        List of reranked search results
    """
    reranker = create_reranker(config)
    return reranker.rerank_search_results(query, search_results, top_k)
