"""
Reranker module for WiQAS (With Query Answering System).

This module provides document reranking capabilities using the BAAI/bge-reranker-v2-m3 model,
which pairs well with BGE-M3 embeddings. It includes intelligent LLM-based cultural analysis
for Filipino content with caching and batch processing.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import requests
import torch
from sentence_transformers import CrossEncoder

from src.utilities.config import RerankerConfig
from src.utilities.gpu_utils import get_gpu_manager

logger = logging.getLogger(__name__)


@dataclass
class CulturalAnalysis:
    """Results of LLM cultural content analysis"""

    confidence: float
    explanation: str
    boost_factor: float
    analyzed_at: float

    def is_cultural(self, threshold: float = 0.6) -> bool:
        """Check if content is considered cultural based on threshold"""
        return self.confidence >= threshold


class Document:
    """Document class for reranking operations"""

    def __init__(self, content: str, metadata: dict[str, Any] | None = None, score: float = 0.0, doc_id: str | None = None):
        self.content = content
        self.metadata = metadata or {}
        self.score = score
        self.doc_id = doc_id or ""

    def __repr__(self) -> str:
        return f"Document(content='{self.content[:50]}...', score={self.score:.3f})"


@dataclass
class RerankResult:
    """Result of document reranking with detailed scoring information"""

    document: Document
    relevance_score: float
    cultural_score: float
    final_score: float
    cultural_explanation: str


class RerankerManager:
    """
    Manages document reranking using BAAI/bge-reranker-v2-m3 model.

    This reranker specializes in Filipino cultural content and provides:
    - Cross-encoder reranking for semantic relevance
    - Intelligent LLM-based cultural content analysis with confidence scoring
    - Content hash-based caching for efficiency
    - Batch processing for multiple documents
    - Configurable confidence thresholds and boost factors
    - Integration with WiQAS retrieval pipeline
    """

    def __init__(self, config: RerankerConfig | None = None):
        """
        Initialize the reranker manager.

        Args:
            config: Reranker configuration. If None, uses default configuration.
        """
        self.config = config or RerankerConfig()
        self.model: CrossEncoder | None = None
        self._cultural_analysis_cache: dict[str, CulturalAnalysis] = {}

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
        logger.info(f"Cultural analysis: {'LLM-based' if self.config.use_llm_cultural_analysis else 'disabled'}")
        if self.config.use_llm_cultural_analysis:
            logger.info(f"Confidence threshold: {self.config.cultural_confidence_threshold}")
            logger.info(f"Batch processing: {'enabled' if self.config.enable_batch_processing else 'disabled'}")

    def _get_content_hash(self, content: str) -> str:
        """
        Generate hash for content to use as cache key.

        Args:
            content: Document content

        Returns:
            SHA-256 hash of content
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _is_cache_valid(self, analysis: CulturalAnalysis) -> bool:
        """
        Check if cached cultural analysis is still valid.

        Args:
            analysis: Cached cultural analysis

        Returns:
            True if cache is still valid
        """
        if not self.config.cache_cultural_analysis:
            return False

        cache_age = time.time() - analysis.analyzed_at
        return cache_age < self.config.cultural_cache_ttl

    def _calculate_boost_factor(self, confidence: float) -> float:
        """
        Calculate cultural boost factor based on confidence level.

        Args:
            confidence: Cultural confidence score (0.0 to 1.0)

        Returns:
            Boost factor to apply
        """
        if not self.config.enable_cultural_boost:
            return 1.0

        if confidence < self.config.cultural_confidence_threshold:
            return 1.0
        elif confidence >= self.config.high_confidence_threshold:
            return self.config.high_confidence_boost
        else:
            # Linear interpolation between low and high confidence boosts
            range_size = self.config.high_confidence_threshold - self.config.cultural_confidence_threshold
            confidence_in_range = confidence - self.config.cultural_confidence_threshold
            boost_range = self.config.high_confidence_boost - self.config.low_confidence_boost

            return self.config.low_confidence_boost + (confidence_in_range / range_size) * boost_range

    def _analyze_single_document_cultural_content(self, content: str) -> CulturalAnalysis | None:
        """
        Analyze a single document for Filipino cultural content using LLM.

        Args:
            content: Document content to analyze

        Returns:
            CulturalAnalysis if successful, None if failed
        """
        if not self.config.use_llm_cultural_analysis:
            return None

        try:
            prompt = f"""
            Analyze this text for Filipino cultural content.
            Rate from 0.0 to 1.0 how much this content relates to Filipino culture,
            language, traditions, food, places, values, or history.

Text: "{content[:500]}..."

Respond with ONLY valid JSON in this exact format:
{{"confidence": 0.0, "explanation": "Brief explanation"}}

Examples:
- "Bayanihan spirit in communities" → {{"confidence": 0.9, "explanation": "Directly discusses Filipino cultural value"}}
- "Adobo recipe ingredients" → {{"confidence": 0.85, "explanation": "Traditional Filipino food"}}
- "Python programming tutorial" → {{"confidence": 0.0, "explanation": "No Filipino cultural content"}}
- "Asian community cooperation" → {{"confidence": 0.2, "explanation": "General Asian concept, not specifically Filipino"}}"""

            payload = {
                "model": self.config.llm_model,
                "prompt": prompt,
                "temperature": self.config.llm_temperature,
                "max_tokens": 100,
                "stream": False,
                "options": {"top_k": 5, "top_p": 0.3, "num_predict": 100},
            }

            response = requests.post(f"{self.config.llm_base_url.rstrip('/')}/api/generate", json=payload, timeout=self.config.llm_timeout)

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()

                try:
                    start_idx = generated_text.find("{")
                    end_idx = generated_text.rfind("}") + 1

                    if start_idx != -1 and end_idx > start_idx:
                        json_str = generated_text[start_idx:end_idx]
                        analysis_data = json.loads(json_str)

                        confidence = float(analysis_data.get("confidence", 0.0))
                        explanation = analysis_data.get("explanation", "No explanation provided")

                        # Validate confidence is in valid range
                        confidence = max(0.0, min(1.0, confidence))

                        # Calculate boost factor
                        boost_factor = self._calculate_boost_factor(confidence)

                        return CulturalAnalysis(confidence=confidence, explanation=explanation, boost_factor=boost_factor, analyzed_at=time.time())

                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse LLM cultural analysis response: {e}")
                    logger.debug(f"Raw response: {generated_text[:100]}...")

                logger.warning("LLM cultural analysis response was not in expected JSON format")
                return None

            else:
                logger.warning(f"LLM API returned status {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.warning(f"LLM cultural analysis request timed out after {self.config.llm_timeout}s")
            return None
        except requests.exceptions.ConnectionError:
            logger.warning("Failed to connect to LLM API for cultural analysis")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error in cultural analysis: {e}")
            return None

    def _analyze_batch_cultural_content(self, documents: list[Document]) -> dict[str, CulturalAnalysis]:
        """
        Analyze multiple documents for cultural content in a single LLM call.

        Args:
            documents: List of documents to analyze

        Returns:
            Dictionary mapping document content hashes to cultural analysis
        """
        if not self.config.use_llm_cultural_analysis or not self.config.enable_batch_processing:
            return {}

        try:
            doc_texts = []
            for i, doc in enumerate(documents):
                content_preview = doc.content[:300].replace("\n", " ")
                doc_texts.append(f'Document {i+1}: "{content_preview}..."')

            batch_prompt = f"""
            Analyze these {len(documents)} texts for Filipino cultural content.
            Rate each from 0.0 to 1.0 for Filipino culture relevance.

{chr(10).join(doc_texts)}

Respond with ONLY valid JSON array in this exact format:
[
  {{"confidence": 0.0, "explanation": "Brief explanation"}},
  {{"confidence": 0.0, "explanation": "Brief explanation"}}
]

Rate based on Filipino culture, language, traditions, food, places, values, or history."""

            payload = {
                "model": self.config.llm_model,
                "prompt": batch_prompt,
                "temperature": self.config.llm_temperature,
                "max_tokens": 300,
                "stream": False,
                "options": {"top_k": 5, "top_p": 0.3, "num_predict": 300},
            }

            response = requests.post(f"{self.config.llm_base_url.rstrip('/')}/api/generate", json=payload, timeout=self.config.llm_timeout)

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()

                try:
                    start_idx = generated_text.find("[")
                    end_idx = generated_text.rfind("]") + 1

                    if start_idx != -1 and end_idx > start_idx:
                        json_str = generated_text[start_idx:end_idx]
                        analysis_list = json.loads(json_str)

                        if isinstance(analysis_list, list) and len(analysis_list) == len(documents):
                            results = {}
                            current_time = time.time()

                            for doc, analysis_data in zip(documents, analysis_list):
                                try:
                                    confidence = float(analysis_data.get("confidence", 0.0))
                                    explanation = analysis_data.get("explanation", "No explanation provided")

                                    confidence = max(0.0, min(1.0, confidence))
                                    boost_factor = self._calculate_boost_factor(confidence)

                                    content_hash = self._get_content_hash(doc.content)
                                    results[content_hash] = CulturalAnalysis(
                                        confidence=confidence,
                                        explanation=explanation,
                                        boost_factor=boost_factor,
                                        analyzed_at=current_time,
                                    )

                                except (ValueError, KeyError) as e:
                                    logger.warning(f"Invalid analysis data for document: {e}")
                                    continue

                            logger.info(f"Batch analyzed {len(results)} documents for cultural content")
                            return results

                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse batch cultural analysis response: {e}")
                    logger.debug(f"Raw response: {generated_text[:200]}...")

                logger.warning("Batch cultural analysis response was not in expected format")
                return {}

            else:
                logger.warning(f"LLM API returned status {response.status_code} for batch analysis")
                return {}

        except Exception as e:
            logger.warning(f"Error in batch cultural analysis: {e}")
            return {}

    def _get_cultural_analysis(self, document: Document) -> CulturalAnalysis:
        """
        Get cultural analysis for a document (with caching).

        Args:
            document: Document to analyze

        Returns:
            Cultural analysis result
        """
        content_hash = self._get_content_hash(document.content)

        if content_hash in self._cultural_analysis_cache:
            cached_analysis = self._cultural_analysis_cache[content_hash]
            if self._is_cache_valid(cached_analysis):
                logger.debug(f"Using cached cultural analysis for document {content_hash}")
                return cached_analysis
            else:
                del self._cultural_analysis_cache[content_hash]

        # Try to analyze document with LLM
        analysis = None
        if self.config.use_llm_cultural_analysis:
            analysis = self._analyze_single_document_cultural_content(document.content)

        if analysis is None:
            # Create default analysis if LLM fails or disabled
            content_lower = document.content.lower()
            basic_keywords = [
                "filipino",
                "philippines",
                "adobo",
                "sinigang",
                "bayanihan",
                "kapamilya",
                "jeepney",
                "tagalog",
                "cebuano",
                "luzon",
                "visayas",
                "mindanao",
            ]

            has_cultural_content = any(keyword in content_lower for keyword in basic_keywords)
            confidence = 0.7 if has_cultural_content else 0.0
            boost_factor = self._calculate_boost_factor(confidence)

            analysis = CulturalAnalysis(
                confidence=confidence,
                explanation=("LLM analysis failed, using basic keyword detection" if has_cultural_content else "No cultural keywords detected"),
                boost_factor=boost_factor,
                analyzed_at=time.time(),
            )

        # Cache the result
        if self.config.cache_cultural_analysis:
            self._cultural_analysis_cache[content_hash] = analysis

        return analysis

    def _batch_analyze_documents(self, documents: list[Document]) -> None:
        """
        Batch analyze documents for cultural content and update cache.

        Args:
            documents: Documents to analyze
        """
        if not self.config.enable_batch_processing or not self.config.use_llm_cultural_analysis:
            return

        documents_to_analyze = []
        for doc in documents:
            content_hash = self._get_content_hash(doc.content)
            if content_hash not in self._cultural_analysis_cache:
                documents_to_analyze.append(doc)
            elif not self._is_cache_valid(self._cultural_analysis_cache[content_hash]):
                # Remove expired entry and mark for re-analysis
                del self._cultural_analysis_cache[content_hash]
                documents_to_analyze.append(doc)

        if not documents_to_analyze:
            logger.debug("All documents already have valid cultural analysis cache")
            return

        # Process in batches
        batch_size = self.config.batch_analysis_size
        for i in range(0, len(documents_to_analyze), batch_size):
            batch = documents_to_analyze[i : i + batch_size]
            logger.info(f"Batch analyzing {len(batch)} documents for cultural content")

            batch_results = self._analyze_batch_cultural_content(batch)

            # Update cache with batch results
            if self.config.cache_cultural_analysis:
                self._cultural_analysis_cache.update(batch_results)

    def _initialize_model(self) -> None:
        """Initialize the reranker model if not already loaded"""
        if self.model is None:
            try:
                logger.info(f"Loading reranker model: {self.config.model}")
                self.model = CrossEncoder(self.config.model, device=self._device, max_length=512)

                # Enable mixed precision
                if self.gpu_manager.is_nvidia_gpu:
                    try:
                        self.model.model.eval()

                        # Enable half precision on newer GPUs for faster inference
                        if torch.cuda.get_device_capability(self.gpu_manager.device)[0] >= 7:  # Volta and newer
                            self.model.model.half()
                            logger.info("Enabled half precision for reranker model")

                        logger.info(f"Using NVIDIA GPU for reranking: {torch.cuda.get_device_name(self.gpu_manager.device.index or 0)}")
                    except Exception as e:
                        logger.warning(f"Could not enable GPU optimizations for reranker: {e}")
                else:
                    logger.info("Using CPU for reranking (no NVIDIA GPU detected)")

                logger.info("Reranker model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load reranker model: {e}")
                raise RuntimeError(f"Could not initialize reranker model: {e}")

    def rerank_documents(self, query: str, documents: list[Document], top_k: int | None = None) -> list[Document]:
        """
        Rerank documents based on relevance and cultural content using intelligent analysis.

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

        # Batch analyze cultural content for all documents if enabled
        if self.config.use_llm_cultural_analysis and self.config.enable_batch_processing:
            self._batch_analyze_documents(documents)

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

            reranked_docs = []
            cultural_boost_count = 0

            for doc, rerank_score in zip(documents, all_scores):
                # Get cultural analysis
                cultural_analysis = self._get_cultural_analysis(doc)

                # Apply cultural boost
                final_score = rerank_score * cultural_analysis.boost_factor

                if cultural_analysis.boost_factor > 1.0:
                    cultural_boost_count += 1
                    logger.debug(f"Cultural boost applied: {rerank_score:.3f} -> {final_score:.3f} " f"(confidence: {cultural_analysis.confidence:.2f})")

                reranked_doc = Document(
                    content=doc.content,
                    metadata=doc.metadata.copy() if doc.metadata else {},
                    score=final_score,
                    doc_id=doc.doc_id,
                )

                if reranked_doc.metadata is None:
                    reranked_doc.metadata = {}
                reranked_doc.metadata.update(
                    {
                        "original_score": doc.score,
                        "rerank_score": float(rerank_score),
                        "cultural_confidence": cultural_analysis.confidence,
                        "cultural_boost": cultural_analysis.boost_factor,
                        "cultural_explanation": cultural_analysis.explanation,
                        "final_score": final_score,
                    }
                )

                reranked_docs.append(reranked_doc)

            filtered_docs = [doc for doc in reranked_docs if doc.score >= self.config.score_threshold]

            filtered_docs.sort(key=lambda x: x.score, reverse=True)

            result = filtered_docs[:top_k]

            logger.info(f"Reranking complete: {len(result)} documents returned " f"(filtered from {len(documents)} original, {cultural_boost_count} culturally boosted)")

            if result and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Top reranked documents:")
                for i, doc in enumerate(result[:3]):
                    cultural_info = ""
                    if doc.metadata and "cultural_confidence" in doc.metadata:
                        cultural_info = f" [cultural: {doc.metadata['cultural_confidence']:.2f}]"
                    logger.debug(f"  {i+1}. Score: {doc.score:.3f}{cultural_info} - Content: {doc.content[:100]}...")

            return result

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fallback: return original documents sorted by original score
            return sorted(documents, key=lambda x: x.score, reverse=True)[:top_k]

    def rerank(self, query: str, documents: list[Document], top_k: int | None = None) -> list[RerankResult]:
        """
        Rerank documents and return detailed results with cultural analysis.

        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of top results to return

        Returns:
            List of reranked documents with detailed scoring information
        """
        if not documents:
            logger.warning("No documents provided for reranking")
            return []

        if not query.strip():
            logger.warning("Empty query provided for reranking")
            return []

        self._initialize_model()

        if self.model is None:
            logger.error("Reranker model not available")
            return []

        if top_k is None:
            top_k = self.config.default_top_k

        logger.info(f"Reranking {len(documents)} documents for query: {query[:50]}...")

        # Batch analyze cultural content for all documents if enabled
        if self.config.use_llm_cultural_analysis and self.config.enable_batch_processing:
            self._batch_analyze_documents(documents)

        # Calculate cross-encoder scores
        query_doc_pairs = [(query, doc.content) for doc in documents]

        if not query_doc_pairs:
            logger.warning("No valid query-document pairs for cross-encoder")
            return []

        # Get cross-encoder scores
        try:
            start_time = time.time()

            # Use mixed precision
            if self.gpu_manager.is_nvidia_gpu:
                with torch.no_grad():
                    with self.gpu_manager.enable_mixed_precision():
                        scores = self.model.predict(query_doc_pairs)
            else:
                with torch.no_grad():
                    scores = self.model.predict(query_doc_pairs)

            inference_time = time.time() - start_time

            # Ensure scores is a list of floats
            if hasattr(scores, "__iter__") and not isinstance(scores, str):
                scores = [float(score) for score in scores]
            else:
                # Single score case
                scores = [float(scores)]

            device_info = f" on {self._device}" if self.gpu_manager.is_nvidia_gpu else " on CPU"
            logger.info(f"Cross-encoder scored {len(scores)} documents in {inference_time:.2f}s{device_info}")

        except Exception as e:
            logger.error(f"Cross-encoder scoring failed: {e}")
            # Fallback to neutral scores
            scores = [0.5] * len(documents)

        # Create results with cultural boost
        results = []
        cultural_boost_count = 0

        for doc, relevance_score in zip(documents, scores):
            try:
                # Get cultural analysis
                cultural_analysis = self._get_cultural_analysis(doc)

                # Apply cultural boost
                boosted_score = relevance_score * cultural_analysis.boost_factor

                if cultural_analysis.boost_factor > 1.0:
                    cultural_boost_count += 1
                    logger.debug(f"Cultural boost applied: {relevance_score:.3f} -> {boosted_score:.3f} " f"(confidence: {cultural_analysis.confidence:.2f})")

                result = RerankResult(
                    document=doc,
                    relevance_score=relevance_score,
                    cultural_score=cultural_analysis.confidence,
                    final_score=boosted_score,
                    cultural_explanation=cultural_analysis.explanation,
                )
                results.append(result)

            except Exception as e:
                logger.warning(f"Error processing document for reranking: {e}")
                # Create result without cultural boost
                result = RerankResult(
                    document=doc,
                    relevance_score=relevance_score,
                    cultural_score=0.0,
                    final_score=relevance_score,
                    cultural_explanation="Cultural analysis failed",
                )
                results.append(result)

        results.sort(key=lambda x: x.final_score, reverse=True)

        top_results = results[:top_k]

        logger.info(f"Reranking complete: returned {len(top_results)} documents " f"({cultural_boost_count} received cultural boost)")

        return top_results

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
            "cultural_boost_enabled": self.config.enable_cultural_boost,
            "cultural_boost_factor": self.config.cultural_boost_factor,
            "model_loaded": self.model is not None,
            "use_llm_cultural_analysis": self.config.use_llm_cultural_analysis,
            "llm_model": self.config.llm_model,
            "llm_base_url": self.config.llm_base_url,
            "cultural_cache_enabled": self.config.cache_cultural_analysis,
            "cultural_cache_ttl": self.config.cultural_cache_ttl,
            "batch_processing_enabled": self.config.enable_batch_processing,
            "batch_analysis_size": self.config.batch_analysis_size,
            "cultural_confidence_threshold": self.config.cultural_confidence_threshold,
            "cached_analyses_count": len(self._cultural_analysis_cache),
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

            self._cultural_analysis_cache.clear()
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
