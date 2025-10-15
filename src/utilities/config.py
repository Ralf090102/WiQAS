import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


# ========== ENVIRONMENT VARIABLE HELPERS ==========
def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean value from environment variable"""
    value = os.getenv(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    elif value in ("false", "0", "no", "off"):
        return False
    return default


def get_env_int(key: str, default: int = 0) -> int:
    """Get integer value from environment variable"""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float value from environment variable"""
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def get_env_str(key: str, default: str = "") -> str:
    """Get string value from environment variable"""
    return os.getenv(key, default)


def get_env_enum(key: str, enum_class: type, default: Any) -> Any:
    """Get enum value from environment variable"""
    value = os.getenv(key, "").lower()
    for enum_val in enum_class:
        if enum_val.value.lower() == value:
            return enum_val
    return default


# ========== ENUMS ==========
class LogLevel(str, Enum):
    """Logging levels"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ChunkerType(str, Enum):
    """Text chunking strategies"""

    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SMART = "smart"


# ========== TIMING BREAKDOWN ==========
@dataclass
class TimingBreakdown:
    """Container for component timing information."""

    # Retrieval components
    embedding_time: float = 0.0
    search_time: float = 0.0
    reranking_time: float = 0.0
    mmr_time: float = 0.0

    # Generation components
    context_preparation_time: float = 0.0
    prompt_building_time: float = 0.0
    llm_generation_time: float = 0.0

    # Multilingual components
    translation_time: float = 0.0
    language_detection_time: float = 0.0

    total_time: float = 0.0

    def get_percentages(self) -> dict[str, float]:
        """Calculate percentage breakdown of timing."""
        if self.total_time == 0:
            return {
                "embedding_percent": 0.0,
                "search_percent": 0.0,
                "reranking_percent": 0.0,
                "mmr_percent": 0.0,
                "context_preparation_percent": 0.0,
                "prompt_building_percent": 0.0,
                "llm_generation_percent": 0.0,
                "translation_percent": 0.0,
                "language_detection_percent": 0.0,
            }

        return {
            "embedding_percent": (self.embedding_time / self.total_time) * 100,
            "search_percent": (self.search_time / self.total_time) * 100,
            "reranking_percent": (self.reranking_time / self.total_time) * 100,
            "mmr_percent": (self.mmr_time / self.total_time) * 100,
            "context_preparation_percent": (self.context_preparation_time / self.total_time) * 100,
            "prompt_building_percent": (self.prompt_building_time / self.total_time) * 100,
            "llm_generation_percent": (self.llm_generation_time / self.total_time) * 100,
            "translation_percent": (self.translation_time / self.total_time) * 100,
            "language_detection_percent": (self.language_detection_time / self.total_time) * 100,
        }

    def format_timing_summary(self) -> str:
        """Format timing breakdown as a readable string."""
        retrieval_sum = self.embedding_time + self.search_time + self.reranking_time + self.mmr_time
        generation_sum = self.context_preparation_time + self.prompt_building_time + self.llm_generation_time
        multilingual_sum = self.translation_time + self.language_detection_time
        component_sum = retrieval_sum + generation_sum + multilingual_sum

        percentages = self.get_percentages_from_components()

        lines = [
            "Timing Breakdown",
            f"embedding time = {self.embedding_time:.2f} s",
            f"search time = {self.search_time:.2f} s",
            f"reranking time = {self.reranking_time:.2f} s",
        ]

        if self.mmr_time > 0:
            lines.append(f"mmr time = {self.mmr_time:.2f} s")

        if generation_sum > 0:
            lines.extend(
                [
                    f"context preparation time = {self.context_preparation_time:.2f} s",
                    f"prompt building time = {self.prompt_building_time:.2f} s",
                    f"llm generation time = {self.llm_generation_time:.2f} s",
                ]
            )

        if multilingual_sum > 0:
            lines.extend(
                [
                    f"translation time = {self.translation_time:.2f} s",
                    f"language detection time = {self.language_detection_time:.2f} s",
                ]
            )

        lines.extend(
            [
                f"total time = {component_sum:.2f} s",
                "",
                "Time % Breakdown",
                f"embedding time = {percentages['embedding_percent']:.2f}%",
                f"search time = {percentages['search_percent']:.2f}%",
                f"reranking time = {percentages['reranking_percent']:.2f}%",
            ]
        )

        if self.mmr_time > 0:
            lines.append(f"mmr time = {percentages['mmr_percent']:.2f}%")

        if generation_sum > 0:
            lines.extend(
                [
                    f"context preparation time = {percentages['context_preparation_percent']:.2f}%",
                    f"prompt building time = {percentages['prompt_building_percent']:.2f}%",
                    f"llm generation time = {percentages['llm_generation_percent']:.2f}%",
                ]
            )

        if multilingual_sum > 0:
            lines.extend(
                [
                    f"translation time = {percentages['translation_percent']:.2f}%",
                    f"language detection time = {percentages['language_detection_percent']:.2f}%",
                ]
            )

        return "\n".join(lines)

    def get_percentages_from_components(self) -> dict[str, float]:
        """Calculate percentage breakdown based on component times only."""
        retrieval_sum = self.embedding_time + self.search_time + self.reranking_time + self.mmr_time
        generation_sum = self.context_preparation_time + self.prompt_building_time + self.llm_generation_time
        multilingual_sum = self.translation_time + self.language_detection_time
        component_sum = retrieval_sum + generation_sum + multilingual_sum

        if component_sum == 0:
            return {
                "embedding_percent": 0.0,
                "search_percent": 0.0,
                "reranking_percent": 0.0,
                "mmr_percent": 0.0,
                "context_preparation_percent": 0.0,
                "prompt_building_percent": 0.0,
                "llm_generation_percent": 0.0,
                "translation_percent": 0.0,
                "language_detection_percent": 0.0,
            }

        return {
            "embedding_percent": (self.embedding_time / component_sum) * 100,
            "search_percent": (self.search_time / component_sum) * 100,
            "reranking_percent": (self.reranking_time / component_sum) * 100,
            "mmr_percent": (self.mmr_time / component_sum) * 100,
            "context_preparation_percent": (self.context_preparation_time / component_sum) * 100,
            "prompt_building_percent": (self.prompt_building_time / component_sum) * 100,
            "llm_generation_percent": (self.llm_generation_time / component_sum) * 100,
            "translation_percent": (self.translation_time / component_sum) * 100,
            "language_detection_percent": (self.language_detection_time / component_sum) * 100,
        }


# ========== BASE CONFIGURATION CLASS ==========
@dataclass
class BaseConfig:
    """Base configuration class"""

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def validate(self) -> None:
        """Validate configuration values"""
        pass


# ========== RAG CONFIGURATION ==========
@dataclass
class EmbeddingConfig(BaseConfig):
    """Embedding model configuration"""

    model: str = "BAAI/bge-m3"
    batch_size: int = 32
    timeout: int = 30
    cache_embeddings: bool = True

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Load embedding configuration from environment variables"""
        return cls(
            model=get_env_str("WIQAS_EMBEDDING_MODEL", "BAAI/bge-m3"),
            batch_size=get_env_int("WIQAS_EMBEDDING_BATCH_SIZE", 32),
            timeout=get_env_int("WIQAS_EMBEDDING_TIMEOUT", 30),
            cache_embeddings=get_env_bool("WIQAS_EMBEDDING_CACHE", True),
        )


@dataclass
class ChunkingConfig(BaseConfig):
    """Document chunking configuration"""

    strategy: ChunkerType = ChunkerType.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 256
    max_chunk_size: int = 512
    min_chunk_size: int = 256

    @classmethod
    def from_env(cls) -> "ChunkingConfig":
        """Load chunking configuration from environment variables"""
        return cls(
            strategy=get_env_enum("WIQAS_CHUNKING_STRATEGY", ChunkerType, ChunkerType.RECURSIVE),
            chunk_size=get_env_int("WIQAS_CHUNK_SIZE", 512),
            chunk_overlap=get_env_int("WIQAS_CHUNK_OVERLAP", 256),
            max_chunk_size=get_env_int("WIQAS_MAX_CHUNK_SIZE", 512),
            min_chunk_size=get_env_int("WIQAS_MIN_CHUNK_SIZE", 256),
        )

    def validate(self) -> None:
        """Validate chunking configuration"""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if self.max_chunk_size < self.min_chunk_size:
            raise ValueError("max_chunk_size must be >= min_chunk_size")
        if self.chunk_size > self.max_chunk_size:
            raise ValueError("chunk_size must be <= max_chunk_size")


@dataclass
class PreprocessingConfig(BaseConfig):
    """Text preprocessing configuration"""

    similarity_threshold: float = 0.85
    min_text_length: int = 50
    enable_deduplication: bool = True
    enable_normalization: bool = True

    @classmethod
    def from_env(cls) -> "PreprocessingConfig":
        """Load preprocessing configuration from environment variables"""
        return cls(
            similarity_threshold=get_env_float("WIQAS_PREPROCESSING_SIMILARITY_THRESHOLD", 0.85),
            min_text_length=get_env_int("WIQAS_PREPROCESSING_MIN_TEXT_LENGTH", 50),
            enable_deduplication=get_env_bool("WIQAS_PREPROCESSING_ENABLE_DEDUPLICATION", True),
            enable_normalization=get_env_bool("WIQAS_PREPROCESSING_ENABLE_NORMALIZATION", True),
        )

    def validate(self) -> None:
        """Validate preprocessing configuration"""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.min_text_length <= 0:
            raise ValueError("min_text_length must be positive")


@dataclass
class RetrievalConfig(BaseConfig):
    """Document retrieval configuration"""

    default_k: int = 5
    max_k: int = 20
    similarity_threshold: float = 0.2
    enable_reranking: bool = True

    enable_hybrid_search: bool = True
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3

    # MMR (Maximal Marginal Relevance)
    enable_mmr: bool = True
    mmr_diversity_bias: float = 0.5
    mmr_fetch_k: int = 20
    mmr_threshold: float = 0.475

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        """Load retrieval configuration from environment variables"""
        return cls(
            default_k=get_env_int("WIQAS_RETRIEVAL_DEFAULT_K", 5),
            max_k=get_env_int("WIQAS_RETRIEVAL_MAX_K", 20),
            similarity_threshold=get_env_float("WIQAS_RETRIEVAL_SIMILARITY_THRESHOLD", 0.2),
            enable_reranking=get_env_bool("WIQAS_RETRIEVAL_ENABLE_RERANKING", True),
            enable_hybrid_search=get_env_bool("WIQAS_RETRIEVAL_ENABLE_HYBRID_SEARCH", True),
            semantic_weight=get_env_float("WIQAS_RETRIEVAL_SEMANTIC_WEIGHT", 0.7),
            keyword_weight=get_env_float("WIQAS_RETRIEVAL_KEYWORD_WEIGHT", 0.3),
            enable_mmr=get_env_bool("WIQAS_RETRIEVAL_ENABLE_MMR", True),
            mmr_diversity_bias=get_env_float("WIQAS_RETRIEVAL_MMR_DIVERSITY_BIAS", 0.5),
            mmr_fetch_k=get_env_int("WIQAS_RETRIEVAL_MMR_FETCH_K", 20),
            mmr_threshold=get_env_float("WIQAS_RETRIEVAL_MMR_THRESHOLD", 0.475),
        )

    def validate(self) -> None:
        """Validate retrieval configuration"""
        if self.default_k <= 0:
            raise ValueError("default_k must be positive")
        if self.max_k < self.default_k:
            raise ValueError("max_k must be >= default_k")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.enable_hybrid_search:
            if not 0.0 <= self.semantic_weight <= 1.0:
                raise ValueError("semantic_weight must be between 0.0 and 1.0")
            if not 0.0 <= self.keyword_weight <= 1.0:
                raise ValueError("keyword_weight must be between 0.0 and 1.0")
            if abs(self.semantic_weight + self.keyword_weight - 1.0) > 0.001:
                raise ValueError("semantic_weight + keyword_weight must equal 1.0")
        if self.enable_mmr:
            if not 0.0 <= self.mmr_diversity_bias <= 1.0:
                raise ValueError("mmr_diversity_bias must be between 0.0 and 1.0")
            if self.mmr_fetch_k < self.default_k:
                raise ValueError("mmr_fetch_k must be >= default_k")


@dataclass
class RerankerConfig(BaseConfig):
    """Document reranking configuration"""

    model: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 16
    timeout: int = 30
    top_k: int = 10
    score_threshold: float = 0.5
    enable_cultural_boost: bool = True
    cultural_boost_factor: float = 1.2

    # LLM-based cultural content analysis
    use_llm_cultural_analysis: bool = True
    llm_model: str = "mistral:latest"
    llm_base_url: str = "http://localhost:11434"
    llm_timeout: int = 90
    llm_temperature: float = 0.1

    # Cultural analysis thresholds
    cultural_confidence_threshold: float = 0.6
    high_confidence_threshold: float = 0.8
    low_confidence_boost: float = 1.1
    high_confidence_boost: float = 1.5

    # Caching and batch processing
    cache_cultural_analysis: bool = True
    cultural_cache_ttl: int = 7200
    batch_analysis_size: int = 10
    enable_batch_processing: bool = True

    @classmethod
    def from_env(cls) -> "RerankerConfig":
        """Load reranker configuration from environment variables"""
        return cls(
            model=get_env_str("WIQAS_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            batch_size=get_env_int("WIQAS_RERANKER_BATCH_SIZE", 16),
            timeout=get_env_int("WIQAS_RERANKER_TIMEOUT", 30),
            top_k=get_env_int("WIQAS_RERANKER_TOP_K", 10),
            score_threshold=get_env_float("WIQAS_RERANKER_SCORE_THRESHOLD", 0.5),
            enable_cultural_boost=get_env_bool("WIQAS_RERANKER_ENABLE_CULTURAL_BOOST", True),
            cultural_boost_factor=get_env_float("WIQAS_RERANKER_CULTURAL_BOOST_FACTOR", 1.2),
            use_llm_cultural_analysis=get_env_bool("WIQAS_RERANKER_USE_LLM_CULTURAL_ANALYSIS", True),
            llm_model=get_env_str("WIQAS_RERANKER_LLM_MODEL", "mistral:latest"),
            llm_base_url=get_env_str("WIQAS_RERANKER_LLM_BASE_URL", "http://localhost:11434"),
            llm_timeout=get_env_int("WIQAS_RERANKER_LLM_TIMEOUT", 90),
            llm_temperature=get_env_float("WIQAS_RERANKER_LLM_TEMPERATURE", 0.1),
            cultural_confidence_threshold=get_env_float("WIQAS_RERANKER_CULTURAL_CONFIDENCE_THRESHOLD", 0.6),
            high_confidence_threshold=get_env_float("WIQAS_RERANKER_HIGH_CONFIDENCE_THRESHOLD", 0.8),
            low_confidence_boost=get_env_float("WIQAS_RERANKER_LOW_CONFIDENCE_BOOST", 1.1),
            high_confidence_boost=get_env_float("WIQAS_RERANKER_HIGH_CONFIDENCE_BOOST", 1.5),
            cache_cultural_analysis=get_env_bool("WIQAS_RERANKER_CACHE_CULTURAL_ANALYSIS", True),
            cultural_cache_ttl=get_env_int("WIQAS_RERANKER_CULTURAL_CACHE_TTL", 7200),
            batch_analysis_size=get_env_int("WIQAS_RERANKER_BATCH_ANALYSIS_SIZE", 10),
            enable_batch_processing=get_env_bool("WIQAS_RERANKER_ENABLE_BATCH_PROCESSING", True),
        )

    def validate(self) -> None:
        """Validate reranker configuration"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0.0 and 1.0")
        if self.cultural_boost_factor < 0.0:
            raise ValueError("cultural_boost_factor must be non-negative")
        if self.llm_timeout <= 0:
            raise ValueError("llm_timeout must be positive")
        if not 0.0 <= self.cultural_confidence_threshold <= 1.0:
            raise ValueError("cultural_confidence_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.high_confidence_threshold <= 1.0:
            raise ValueError("high_confidence_threshold must be between 0.0 and 1.0")
        if self.cultural_confidence_threshold > self.high_confidence_threshold:
            raise ValueError("cultural_confidence_threshold must be <= high_confidence_threshold")
        if self.batch_analysis_size <= 0:
            raise ValueError("batch_analysis_size must be positive")
        if self.cultural_cache_ttl <= 0:
            raise ValueError("cultural_cache_ttl must be positive")


@dataclass
class LLMConfig(BaseConfig):
    """Large Language Model configuration"""

    model: str = "mistral:latest"
    base_url: str = "http://localhost:11434"
    timeout: int = 90

    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int | None = None

    # RAG prompt
    system_prompt: str = "You are WiQAS, a helpful AI assistant with access to a knowledge base. " "Use the provided context to answer questions accurately and cite sources when appropriate."

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load LLM configuration from environment variables"""
        max_tokens_str = get_env_str("WIQAS_LLM_MAX_TOKENS", "")
        max_tokens = int(max_tokens_str) if max_tokens_str.isdigit() else None

        return cls(
            model=get_env_str("WIQAS_LLM_MODEL", "mistral:latest"),
            base_url=get_env_str("WIQAS_LLM_BASE_URL", "http://localhost:11434"),
            timeout=get_env_int("WIQAS_LLM_TIMEOUT", 90),
            temperature=get_env_float("WIQAS_LLM_TEMPERATURE", 0.7),
            top_p=get_env_float("WIQAS_LLM_TOP_P", 0.9),
            max_tokens=max_tokens,
            system_prompt=get_env_str(
                "WIQAS_LLM_SYSTEM_PROMPT",
                "You are WiQAS, a helpful AI assistant with access to a knowledge base. " "Use the provided context to answer questions accurately and cite sources when appropriate.",
            ),
        )


@dataclass
class VectorStoreConfig(BaseConfig):
    """Vector store configuration"""

    index_type: str = "chroma"
    collection_name: str = "wiqas_knowledge_base"
    persist_immediately: bool = True
    persist_directory: str = "./data/chroma-data"
    distance_metric: str = "cosine"
    use_gpu: bool = False
    batch_size: int = 64

    @classmethod
    def from_env(cls) -> "VectorStoreConfig":
        """Load vector store configuration from environment variables"""
        return cls(
            index_type=get_env_str("WIQAS_VECTORSTORE_INDEX_TYPE", "chroma"),
            collection_name=get_env_str("WIQAS_VECTORSTORE_COLLECTION_NAME", "wiqas_knowledge_base"),
            persist_immediately=get_env_bool("WIQAS_VECTORSTORE_PERSIST_IMMEDIATELY", True),
            persist_directory=get_env_str("WIQAS_VECTORSTORE_PERSIST_DIRECTORY", "./data/chroma-data"),
            distance_metric=get_env_str("WIQAS_VECTORSTORE_DISTANCE_METRIC", "cosine"),
            use_gpu=get_env_bool("WIQAS_VECTORSTORE_USE_GPU", False),
            batch_size=get_env_int("WIQAS_VECTORSTORE_BATCH_SIZE", 64),
        )


@dataclass
class AnswerGeneratorConfig(BaseConfig):
    """Answer generation configuration"""

    model: str = "aisingapore/Gemma-SEA-LION-v3-9B"
    base_url: str = "http://localhost:11434"
    timeout: int = 120
    backend: str = "hf"  # ollama | hf

    # init gen params
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int | None = 1024

    @classmethod
    def from_env(cls) -> "AnswerGeneratorConfig":
        """Load answer generator configuration from environment variables"""
        return cls(
            model=get_env_str("WIQAS_ANSWER_GENERATOR_MODEL", "aisingapore/Gemma-SEA-LION-v3-9B"),
            base_url=get_env_str("WIQAS_ANSWER_GENERATOR_BASE_URL", "http://localhost:11434"),
            timeout=get_env_int("WIQAS_ANSWER_GENERATOR_TIMEOUT", 120),
            backend=get_env_str("WIQAS_BACKEND", "hf"),  # ollama | hf
            temperature=get_env_float("WIQAS_ANSWER_GENERATOR_TEMPERATURE", 0.7),
            top_p=get_env_float("WIQAS_ANSWER_GENERATOR_TOP_P", 0.9),
            max_tokens=get_env_int("WIQAS_ANSWER_GENERATOR_MAX_TOKENS", 1024),
        )


# ========== MULTILINGUAL CONFIGURATION ==========
@dataclass
class MultilingualConfig(BaseConfig):
    """Multilingual retrieval configuration"""
    
    enable_cross_lingual: bool = True
    auto_translate_queries: bool = True
    supported_languages: list[str] = field(default_factory=lambda: ["en", "tl"])
    
    enable_language_detection: bool = True
    language_boost_same: float = 1.0
    language_boost_cross: float = 1.0
    
    # Translation settings
    translation_service: str = "deep_translator"
    translation_cache_ttl: int = 1800
    max_translation_length: int = 500
    enable_translation_cache: bool = True
    
    # Multi-query approach
    enable_multi_query: bool = True
    max_queries_per_request: int = 2  # Original + translated
    query_weight_original: float = 1.0
    query_weight_translated: float = 0.8
    
    @classmethod
    def from_env(cls) -> "MultilingualConfig":
        """Load multilingual configuration from environment variables"""
        return cls(
            enable_cross_lingual=get_env_bool("WIQAS_MULTILINGUAL_ENABLE_CROSS_LINGUAL", True),
            auto_translate_queries=get_env_bool("WIQAS_MULTILINGUAL_AUTO_TRANSLATE", True),
            enable_language_detection=get_env_bool("WIQAS_MULTILINGUAL_LANGUAGE_DETECTION", True),
            language_boost_same=get_env_float("WIQAS_MULTILINGUAL_BOOST_SAME", 1.2),
            language_boost_cross=get_env_float("WIQAS_MULTILINGUAL_BOOST_CROSS", 1.0),
            translation_service=get_env_str("WIQAS_MULTILINGUAL_TRANSLATION_SERVICE", "deep_translator"),
            translation_cache_ttl=get_env_int("WIQAS_MULTILINGUAL_TRANSLATION_CACHE_TTL", 3600),
            max_translation_length=get_env_int("WIQAS_MULTILINGUAL_MAX_TRANSLATION_LENGTH", 500),
            enable_translation_cache=get_env_bool("WIQAS_MULTILINGUAL_ENABLE_CACHE", True),
            enable_multi_query=get_env_bool("WIQAS_MULTILINGUAL_ENABLE_MULTI_QUERY", True),
            max_queries_per_request=get_env_int("WIQAS_MULTILINGUAL_MAX_QUERIES", 2),
            query_weight_original=get_env_float("WIQAS_MULTILINGUAL_WEIGHT_ORIGINAL", 1.0),
            query_weight_translated=get_env_float("WIQAS_MULTILINGUAL_WEIGHT_TRANSLATED", 0.8),
        )

    def validate(self) -> None:
        """Validate multilingual configuration"""
        if self.language_boost_same < 0.0:
            raise ValueError("language_boost_same must be non-negative")
        if self.language_boost_cross < 0.0:
            raise ValueError("language_boost_cross must be non-negative")
        if self.translation_cache_ttl <= 0:
            raise ValueError("translation_cache_ttl must be positive")
        if self.max_translation_length <= 0:
            raise ValueError("max_translation_length must be positive")
        if self.max_queries_per_request <= 0:
            raise ValueError("max_queries_per_request must be positive")
        if not self.supported_languages:
            raise ValueError("supported_languages cannot be empty")


# ========= EVALUATION CONFIGURATION ==========
@dataclass
class EvaluationConfig(BaseConfig):
    """Evaluation configuration"""

    dataset_path: str = "./src/evaluation/evaluation_dataset.json"
    limit: int | None = None
    randomize: bool = False
    disable_cultural_llm_analysis: bool = False

    # Retrieval settings
    search_type: str = "hybrid"
    k_results: int = 5
    enable_reranking: bool = True
    enable_mmr: bool = True

    similarity_threshold: float = 0.5

    @classmethod
    def from_env(cls) -> "EvaluationConfig":
        """Load evaluation configuration from environment variables"""
        limit_str = get_env_str("WIQAS_EVALUATION_LIMIT", "")
        limit = int(limit_str) if limit_str.isdigit() else None

        return cls(
            dataset_path=get_env_str("WIQAS_EVALUATION_DATASET_PATH", "./src/evaluation/evaluation_dataset.json"),
            limit=limit,
            randomize=get_env_bool("WIQAS_EVALUATION_RANDOMIZE", False),
            disable_cultural_llm_analysis=get_env_bool("WIQAS_EVALUATION_DISABLE_CULTURAL_LLM", False),
            search_type=get_env_str("WIQAS_EVALUATION_SEARCH_TYPE", "hybrid"),
            k_results=get_env_int("WIQAS_EVALUATION_K_RESULTS", 5),
            enable_reranking=get_env_bool("WIQAS_EVALUATION_ENABLE_RERANKING", True),
            enable_mmr=get_env_bool("WIQAS_EVALUATION_ENABLE_MMR", True),
            similarity_threshold=get_env_float("WIQAS_EVALUATION_SIMILARITY_THRESHOLD", 0.5),
        )

    def validate(self) -> None:
        """Validate evaluation configuration"""
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when specified")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.k_results <= 0:
            raise ValueError("k_results must be positive")


# ========== MAIN RAG CONFIGURATION ==========
@dataclass
class RAGConfig(BaseConfig):
    """Complete RAG pipeline configuration"""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    generator: AnswerGeneratorConfig = field(default_factory=AnswerGeneratorConfig)
    multilingual: MultilingualConfig = field(default_factory=MultilingualConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Load RAG configuration from environment variables"""
        return cls(
            embedding=EmbeddingConfig.from_env(),
            preprocessing=PreprocessingConfig.from_env(),
            chunking=ChunkingConfig.from_env(),
            retrieval=RetrievalConfig.from_env(),
            reranker=RerankerConfig.from_env(),
            llm=LLMConfig.from_env(),
            vectorstore=VectorStoreConfig.from_env(),
            generator=AnswerGeneratorConfig.from_env(),
            evaluation=EvaluationConfig.from_env(),
            multilingual=MultilingualConfig.from_env(),
        )


# ========== SYSTEM CONFIGURATION ==========
@dataclass
class StorageConfig(BaseConfig):
    """Data storage configuration"""

    data_directory: str = "./wiqas-data"
    temp_directory: str = "./temp"
    knowledge_base_directory: str = "./data/knowledge_base"

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Load storage configuration from environment variables"""
        return cls(
            data_directory=get_env_str("WIQAS_STORAGE_DATA_DIRECTORY", "./wiqas-data"),
            temp_directory=get_env_str("WIQAS_STORAGE_TEMP_DIRECTORY", "./temp"),
            knowledge_base_directory=get_env_str("WIQAS_STORAGE_KNOWLEDGE_BASE_DIRECTORY", "./data/knowledge_base"),
        )


@dataclass
class SystemConfig(BaseConfig):
    """System configuration"""

    storage: StorageConfig = field(default_factory=StorageConfig)

    # Ollama integration
    require_ollama: bool = True
    ollama_health_check: bool = True

    @classmethod
    def from_env(cls) -> "SystemConfig":
        """Load system configuration from environment variables"""
        return cls(
            storage=StorageConfig.from_env(),
            require_ollama=get_env_bool("WIQAS_SYSTEM_REQUIRE_OLLAMA", True),
            ollama_health_check=get_env_bool("WIQAS_SYSTEM_OLLAMA_HEALTH_CHECK", True),
        )


# ========== GPU CONFIGURATION ==========
@dataclass
class GPUConfig(BaseConfig):
    """GPU acceleration configuration"""

    enabled: bool = True
    auto_detect: bool = True
    preferred_device: str = "auto"  # "auto", "cpu", "cuda:0"
    fallback_to_cpu: bool = True
    memory_fraction: float = 0.9  # Fraction of GPU memory to use (0.1-1.0)
    clear_cache_frequency: int = 10
    enable_mixed_precision: bool = True  # Enable mixed precision training for faster inference
    batch_size_multiplier: float = 2.0

    @classmethod
    def from_env(cls) -> "GPUConfig":
        """Load GPU configuration from environment variables"""
        return cls(
            enabled=get_env_bool("WIQAS_GPU_ENABLED", True),
            auto_detect=get_env_bool("WIQAS_GPU_AUTO_DETECT", True),
            preferred_device=get_env_str("WIQAS_GPU_PREFERRED_DEVICE", "auto"),
            fallback_to_cpu=get_env_bool("WIQAS_GPU_FALLBACK_TO_CPU", True),
            memory_fraction=get_env_float("WIQAS_GPU_MEMORY_FRACTION", 0.9),
            clear_cache_frequency=get_env_int("WIQAS_GPU_CLEAR_CACHE_FREQUENCY", 10),
            enable_mixed_precision=get_env_bool("WIQAS_GPU_ENABLE_MIXED_PRECISION", True),
            batch_size_multiplier=get_env_float("WIQAS_GPU_BATCH_SIZE_MULTIPLIER", 2.0),
        )

    def validate(self) -> None:
        """Validate GPU configuration values"""
        if self.memory_fraction <= 0 or self.memory_fraction > 1.0:
            raise ValueError("memory_fraction must be between 0.1 and 1.0")
        if self.clear_cache_frequency <= 0:
            raise ValueError("clear_cache_frequency must be positive")
        if self.batch_size_multiplier <= 0:
            raise ValueError("batch_size_multiplier must be positive")


# ========== LOGGING CONFIGURATION ==========
@dataclass
class LoggingConfig(BaseConfig):
    """Logging configuration"""

    level: LogLevel = LogLevel.INFO
    log_to_file: bool = True
    log_file_path: str = "./logs/wiqas.log"
    log_to_console: bool = True
    verbose: bool = False

    @classmethod
    def from_env(cls) -> "LoggingConfig":
        """Load logging configuration from environment variables"""
        return cls(
            level=get_env_enum("WIQAS_LOGGING_LEVEL", LogLevel, LogLevel.INFO),
            log_to_file=get_env_bool("WIQAS_LOGGING_LOG_TO_FILE", True),
            log_file_path=get_env_str("WIQAS_LOGGING_LOG_FILE_PATH", "./logs/wiqas.log"),
            log_to_console=get_env_bool("WIQAS_LOGGING_LOG_TO_CONSOLE", True),
            verbose=get_env_bool("WIQAS_LOGGING_VERBOSE", False),
        )


# ========== MAIN CONFIGURATION CLASS ==========
@dataclass
class WiQASConfig(BaseConfig):
    """Complete WiQAS configuration"""

    rag: RAGConfig = field(default_factory=RAGConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    version: str = "1.0.0"

    @classmethod
    def from_env(cls) -> "WiQASConfig":
        """Load configuration from environment variables"""
        config = cls(
            rag=RAGConfig.from_env(),
            system=SystemConfig.from_env(),
            gpu=GPUConfig.from_env(),
            logging=LoggingConfig.from_env(),
            version=get_env_str("WIQAS_VERSION", "1.0.0"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate entire configuration"""
        self.rag.preprocessing.validate()
        self.rag.chunking.validate()
        self.rag.retrieval.validate()
        # Add more validations as needed


def get_config(from_env: bool = False) -> WiQASConfig:
    """
    Get a configuration instance.

    Args:
        from_env: If True, load configuration from environment variables.
                 If False, use default values.

    Returns:
        WiQASConfig instance with specified settings

    Example:
        # Use default configuration
        config = get_config()

        # Load from environment variables
        config = get_config(from_env=True)

        # Or directly
        config = WiQASConfig.from_env()

        Environment Variables:
        RAG Configuration:
            WIQAS_EMBEDDING_MODEL="nomic-embed-text"
            WIQAS_EMBEDDING_BATCH_SIZE=32
            WIQAS_CHUNK_SIZE=128
            WIQAS_CHUNK_OVERLAP=0
            WIQAS_CHUNKING_STRATEGY="recursive"
            WIQAS_LLM_MODEL="mistral:latest"
            WIQAS_LLM_BASE_URL="http://localhost:11434"
            WIQAS_LLM_TEMPERATURE=0.7
            WIQAS_VECTORSTORE_COLLECTION_NAME="wiqas_knowledge_base"
            WIQAS_VECTORSTORE_PERSIST_DIRECTORY="./data/chroma-data"
            WIQAS_RETRIEVAL_DEFAULT_K=5        System Configuration:
            WIQAS_STORAGE_DATA_DIRECTORY="./wiqas-data"
            WIQAS_SYSTEM_REQUIRE_OLLAMA=true
            WIQAS_LOGGING_LEVEL="info"

        GPU Configuration:
            WIQAS_GPU_ENABLED=true
            WIQAS_GPU_AUTO_DETECT=true
            WIQAS_GPU_PREFERRED_DEVICE="auto"
            WIQAS_GPU_FALLBACK_TO_CPU=true
            WIQAS_GPU_MEMORY_FRACTION=0.9
            WIQAS_GPU_CLEAR_CACHE_FREQUENCY=10
            WIQAS_GPU_ENABLE_MIXED_PRECISION=true
            WIQAS_GPU_BATCH_SIZE_MULTIPLIER=2.0
    """
    if from_env:
        return WiQASConfig.from_env()
    return WiQASConfig()
