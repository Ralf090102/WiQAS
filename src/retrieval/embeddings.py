"""
Embedding generation and management for WiQAS using Hugging Face models.
Supports BGE-M3 and other multilingual models for Filipino cultural content.
"""

import hashlib
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from src.utilities.utils import (
    ensure_config,
    log_debug,
    log_error,
    log_info,
    log_success,
    log_warning,
    timer,
)

if TYPE_CHECKING:
    from src.utilities.config import WiQASConfig


class EmbeddingManager:
    """
    Manages embedding generation using Hugging Face models.

    Supports BGE-M3 and other multilingual models optimized for
    Filipino cultural content and cross-language understanding.
    """

    def __init__(self, config: Optional["WiQASConfig"] = None):
        """
        Initialize embedding manager.

        Args:
            config: WiQAS configuration object
        """
        self.config = ensure_config(config)
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self.cache_dir = self._setup_cache_directory()
        self.embedding_cache = {}

        self._load_model()

    def _get_device(self) -> str:
        """Determine device for embedding computation."""
        if self.config.gpu.enabled and torch.cuda.is_available():
            device = "cuda"
            log_info(f"Using GPU for embeddings: {torch.cuda.get_device_name()}", config=self.config)
        else:
            device = "cpu"
            log_info("Using CPU for embeddings", config=self.config)

        return device

    def _setup_cache_directory(self) -> Path | None:
        """Setup embedding cache directory if caching is enabled."""
        if not self.config.rag.embedding.cache_embeddings:
            return None

        try:
            cache_dir = Path(self.config.system.storage.temp_directory) / "embedding_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            log_debug(f"Embedding cache directory: {cache_dir}", self.config)
            return cache_dir
        except Exception as e:
            log_warning(f"Failed to create cache directory: {e}", config=self.config)
            return None

    def _load_model(self) -> None:
        """Load the embedding model based on configuration."""
        model_name = self.config.rag.embedding.model

        try:
            log_info(f"Loading embedding model: {model_name}", config=self.config)

            # BGE-M3
            if "bge-m3" in model_name.lower() or model_name == "BAAI/bge-m3":
                self._load_bge_m3_model(model_name)

            # Other sentence-transformers models
            elif model_name in ["nomic-embed-text", "all-MiniLM-L6-v2", "all-mpnet-base-v2"]:
                self._load_sentence_transformer(model_name)

            # Generic Hugging Face model
            else:
                self._load_huggingface_model(model_name)

            log_success(f"Successfully loaded model: {model_name}", config=self.config)

        except Exception as e:
            log_error(f"Failed to load embedding model {model_name}: {e}", config=self.config)
            self._fallback_model()

    def _load_bge_m3_model(self, model_name: str = "BAAI/bge-m3") -> None:
        """Load BGE-M3 model with optimized settings."""
        try:
            self.model = SentenceTransformer(model_name, device=self.device)

            # Set maximum sequence length for BGE-M3 (supports up to 8192 tokens)
            if hasattr(self.model, "max_seq_length"):
                self.model.max_seq_length = 8192
                log_info(f"Set BGE-M3 max sequence length to {self.model.max_seq_length}", config=self.config)

            # Set to evaluation mode
            self.model.eval()

            log_info("BGE-M3 model loaded successfully", config=self.config)

        except Exception as e:
            log_error(f"Failed to load BGE-M3: {e}", config=self.config)
            raise

    def _load_sentence_transformer(self, model_name: str) -> None:
        """Load a sentence-transformers model."""
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model.eval()

        except Exception as e:
            log_error(f"Failed to load sentence-transformer {model_name}: {e}", config=self.config)
            raise

    def _load_huggingface_model(self, model_name: str) -> None:
        """Load a generic Hugging Face model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()

        except Exception as e:
            log_error(f"Failed to load Hugging Face model {model_name}: {e}", config=self.config)
            raise

    def _fallback_model(self) -> None:
        """Load a fallback model if primary model fails."""
        fallback_models = ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"]

        for fallback in fallback_models:
            try:
                log_warning(f"Trying fallback model: {fallback}", config=self.config)
                self.model = SentenceTransformer(fallback, device=self.device)
                self.model.eval()
                log_success(f"Loaded fallback model: {fallback}", config=self.config)
                return
            except Exception as e:
                log_warning(f"Fallback model {fallback} also failed: {e}", config=self.config)
                continue

        raise RuntimeError("All embedding models failed to load")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _load_from_cache(self, cache_key: str) -> list[float] | None:
        """Load embedding from cache if available."""
        if not self.config.rag.embedding.cache_embeddings or not self.cache_dir:
            return None

        # Check memory cache
        if cache_key in self.embedding_cache:
            log_debug("Embedding found in memory cache", self.config)
            return self.embedding_cache[cache_key]

        # Check disk cache
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    embedding = pickle.load(f)

                    # Store in memory cache
                    self.embedding_cache[cache_key] = embedding

                    log_debug("Embedding found in disk cache", self.config)
                    return embedding
        except Exception as e:
            log_warning(f"Failed to load from cache: {e}", config=self.config)

        return None

    def _save_to_cache(self, cache_key: str, embedding: list[float]) -> None:
        """Save embedding to cache."""
        if not self.config.rag.embedding.cache_embeddings:
            return

        # Save to memory cache
        self.embedding_cache[cache_key] = embedding

        # Save to disk cache
        if self.cache_dir:
            try:
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                with open(cache_file, "wb") as f:
                    pickle.dump(embedding, f)
                log_debug("Embedding saved to cache", self.config)
            except Exception as e:
                log_warning(f"Failed to save to cache: {e}", config=self.config)

    @timer
    def encode_single(self, text: str) -> list[float]:
        """
        Encode a single text into embedding.

        Args:
            text: Input text to encode

        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            log_warning("Empty text provided for encoding", config=self.config)
            return []

        cache_key = self._get_cache_key(text)
        cached_embedding = self._load_from_cache(cache_key)
        if cached_embedding is not None:
            return cached_embedding

        try:
            # Generate embedding
            if isinstance(self.model, SentenceTransformer):
                with torch.no_grad():
                    embedding = self.model.encode(
                        text, convert_to_tensor=False, normalize_embeddings=True, show_progress_bar=False
                    )
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()

            else:
                # For generic Hugging Face models
                with torch.no_grad():
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                        self.device
                    )

                    outputs = self.model(**inputs)
                    # Mean pooling
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                    embedding = embedding.tolist()

            self._save_to_cache(cache_key, embedding)

            log_debug(f"Generated embedding of dimension {len(embedding)}", self.config)
            return embedding

        except Exception as e:
            log_error(f"Failed to encode text: {e}", config=self.config)
            return []

    @timer
    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Encode multiple texts into embeddings efficiently.

        Args:
            texts: List of input texts to encode

        Returns:
            List of embedding vectors
        """
        if not texts:
            log_warning("Empty text list provided for batch encoding", config=self.config)
            return []

        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)

        if not valid_texts:
            log_warning("No valid texts found for encoding", config=self.config)
            return [[] for _ in texts]

        try:
            batch_size = self.config.rag.embedding.batch_size
            all_embeddings = []

            # Batch Processing
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i : i + batch_size]

                log_debug(f"Processing batch {i//batch_size + 1} with {len(batch_texts)} texts", self.config)

                if isinstance(self.model, SentenceTransformer):
                    with torch.no_grad():
                        # For BGE-M3, add explicit truncation and show_progress_bar
                        batch_embeddings = self.model.encode(
                            batch_texts,
                            convert_to_tensor=False,
                            normalize_embeddings=True,
                            batch_size=len(batch_texts),
                            show_progress_bar=False,
                        )

                        if isinstance(batch_embeddings, np.ndarray):
                            batch_embeddings = batch_embeddings.tolist()

                        all_embeddings.extend(batch_embeddings)
                else:
                    # Process individual texts for generic models
                    for text in batch_texts:
                        embedding = self.encode_single(text)
                        all_embeddings.append(embedding)

            result_embeddings = [[] for _ in texts]
            for i, embedding in enumerate(all_embeddings):
                original_index = valid_indices[i]
                result_embeddings[original_index] = embedding

            log_success(f"Generated {len(all_embeddings)} embeddings", config=self.config)
            return result_embeddings

        except Exception as e:
            log_error(f"Failed to encode batch: {e}", config=self.config)
            return [[] for _ in texts]

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.

        Returns:
            Embedding dimension
        """
        try:
            if isinstance(self.model, SentenceTransformer):
                return self.model.get_sentence_embedding_dimension()
            else:
                test_embedding = self.encode_single("test")
                return len(test_embedding) if test_embedding else 0

        except Exception as e:
            log_error(f"Failed to get embedding dimension: {e}", config=self.config)
            return 768  # Default fallback

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        try:
            self.embedding_cache.clear()

            if self.cache_dir and self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()

            log_info("Embedding cache cleared", config=self.config)

        except Exception as e:
            log_error(f"Failed to clear cache: {e}", config=self.config)

    def get_model_info(self) -> dict[str, any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.config.rag.embedding.model,
            "device": self.device,
            "embedding_dimension": self.get_embedding_dimension(),
            "cache_enabled": self.config.rag.embedding.cache_embeddings,
            "batch_size": self.config.rag.embedding.batch_size,
        }


def create_embedding_manager(config: Optional["WiQASConfig"] = None) -> EmbeddingManager:
    """
    Factory function to create an EmbeddingManager instance.

    Args:
        config: WiQAS configuration object

    Returns:
        Initialized EmbeddingManager instance
    """
    return EmbeddingManager(config)
