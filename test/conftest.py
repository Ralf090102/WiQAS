"""
Shared pytest fixtures and configuration for WiQAS tests.
Provides common mock objects and test data across all test modules.
"""

import pytest
from unittest.mock import Mock, patch
import time
import numpy as np


@pytest.fixture
def mock_config():
    """Mock WiQAS configuration object with common settings."""
    config = Mock()

    # LLM settings
    config.llm.model = "mistral:latest"
    config.llm.base_url = "http://localhost:11434"
    config.llm.timeout = 90
    config.llm.temperature = 0.3
    config.llm.system_message = "You are a helpful AI assistant."

    # GPU settings
    config.gpu.enabled = False

    # RAG settings
    config.rag.embedding.model = "all-MiniLM-L6-v2"
    config.rag.embedding.cache_embeddings = False
    config.rag.embedding.batch_size = 32

    # Vector store settings
    config.rag.vectorstore.persist_directory = "/tmp/test_chroma"
    config.rag.vectorstore.distance_metric = "cosine"
    config.rag.vectorstore.persist_immediately = False

    # Retrieval settings
    config.rag.retrieval.max_k = 10
    config.rag.retrieval.similarity_threshold = 0.7
    config.rag.retrieval.semantic_weight = 0.7
    config.rag.retrieval.keyword_weight = 0.3
    config.rag.retrieval.mmr_diversity_bias = 0.3
    config.rag.retrieval.mmr_threshold = 0.5

    # System settings
    config.system.storage.temp_directory = "/tmp"

    return config


@pytest.fixture
def mock_reranker_config():
    """Mock reranker configuration object."""
    config = Mock()
    config.model = "BAAI/bge-reranker-v2-m3"
    config.use_llm_cultural_analysis = False
    config.enable_batch_processing = False
    config.cache_cultural_analysis = False
    config.enable_cultural_boost = True
    config.cultural_boost_factor = 1.2
    config.score_threshold = 0.0
    config.top_k = 5
    config.default_top_k = 5
    config.batch_size = 8
    config.cultural_confidence_threshold = 0.6
    config.high_confidence_threshold = 0.8
    config.low_confidence_boost = 1.1
    config.high_confidence_boost = 1.3
    config.cultural_cache_ttl = 3600
    config.batch_analysis_size = 10
    config.llm_model = "mistral:latest"
    config.llm_base_url = "http://localhost:11434"
    config.llm_timeout = 15
    config.llm_temperature = 0.1
    return config


@pytest.fixture
def sample_documents():
    """Sample document data for testing."""
    return [
        {
            "id": "doc1",
            "content": "This is the first test document about Filipino culture and traditions.",
            "metadata": {"source": "wiki", "category": "culture"},
            "score": 0.8,
        },
        {
            "id": "doc2",
            "content": "Second document discusses adobo recipe and cooking methods.",
            "metadata": {"source": "cookbook", "category": "food"},
            "score": 0.7,
        },
        {
            "id": "doc3",
            "content": "Third document covers general programming concepts and algorithms.",
            "metadata": {"source": "textbook", "category": "technology"},
            "score": 0.6,
        },
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embedding vectors for testing."""
    return [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0], [0.2, 0.4, 0.6, 0.8, 1.0]]


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response."""
    return {
        "model": "mistral:latest",
        "response": "This is a test response from the LLM.",
        "done": True,
        "context": [1, 2, 3, 4, 5],
        "total_duration": 1000000000,
        "load_duration": 100000000,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 200000000,
        "eval_count": 20,
        "eval_duration": 700000000,
    }


@pytest.fixture
def mock_chat_messages():
    """Mock chat messages for LLM testing."""
    return [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of the Philippines?"},
    ]


@pytest.fixture
def mock_vector_store_results():
    """Mock vector store query results."""
    return {
        "ids": [["doc1", "doc2", "doc3"]],
        "distances": [[0.1, 0.2, 0.3]],
        "documents": [["First document", "Second document", "Third document"]],
        "metadatas": [
            [
                {"source": "test1", "category": "culture"},
                {"source": "test2", "category": "food"},
                {"source": "test3", "category": "general"},
            ]
        ],
    }


@pytest.fixture
def mock_embedding_manager():
    """Mock embedding manager with common methods."""
    manager = Mock()
    manager.encode_single.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    manager.encode_batch.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0], [0.2, 0.4, 0.6, 0.8, 1.0]]
    manager.get_embedding_dimension.return_value = 5
    return manager


@pytest.fixture
def mock_vector_store():
    """Mock vector store with common methods."""
    store = Mock()
    store.add_documents.return_value = True
    store.query.return_value = {
        "ids": [["doc1", "doc2"]],
        "distances": [[0.1, 0.2]],
        "documents": [["First document", "Second document"]],
        "metadatas": [[{"source": "test1"}, {"source": "test2"}]],
    }
    store.get_collection_stats.return_value = {"document_count": 42, "collection_name": "test_collection"}
    return store


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer model."""
    with patch("src.retrieval.embeddings.SentenceTransformer") as mock_st:
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        mock_model.eval.return_value = None
        mock_model.get_sentence_embedding_dimension.return_value = 4
        mock_st.return_value = mock_model
        yield mock_st


@pytest.fixture
def mock_cross_encoder():
    """Mock CrossEncoder model for reranking."""
    with patch("src.retrieval.reranker.CrossEncoder") as mock_ce:
        mock_model = Mock()
        mock_model.predict.return_value = [0.8, 0.6, 0.9]
        mock_model.max_length = 512
        mock_ce.return_value = mock_model
        yield mock_ce


@pytest.fixture
def mock_bm25():
    """Mock BM25 for keyword search."""
    with patch("src.retrieval.search.BM25Okapi") as mock_bm25_class:
        mock_bm25_instance = Mock()
        mock_bm25_instance.get_scores.return_value = [0.8, 0.3, 0.9]
        mock_bm25_class.return_value = mock_bm25_instance
        yield mock_bm25_class


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client."""
    with patch("src.retrieval.vector_store.chromadb.PersistentClient") as mock_client_class:
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.add.return_value = None
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "distances": [[0.1, 0.2]],
            "documents": [["First document", "Second document"]],
            "metadatas": [[{"source": "test1"}, {"source": "test2"}]],
        }
        mock_collection.count.return_value = 10

        mock_client.get_collection.return_value = mock_collection
        mock_client.create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        yield mock_client_class


@pytest.fixture
def mock_cuda_unavailable():
    """Mock CUDA as unavailable for consistent CPU testing."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture
def mock_ollama_api():
    """Mock Ollama API requests."""
    with patch("ollama.list") as mock_list, patch("ollama.show") as mock_show, patch("ollama.chat") as mock_chat:

        # Mock model list
        mock_list.return_value = {
            "models": [{"name": "mistral:latest", "size": 2000000000}, {"name": "mistral:7b", "size": 4000000000}]
        }

        # Mock model info
        mock_show.return_value = {
            "modelfile": "FROM mistral:latest",
            "parameters": {"temperature": 0.3},
            "template": "{{ .Prompt }}",
            "details": {
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "3B",
                "quantization_level": "Q4_K_M",
            },
        }

        # Mock chat response
        mock_chat.return_value = {
            "message": {"role": "assistant", "content": "This is a test response from Ollama."},
            "done": True,
            "total_duration": 1000000000,
            "load_duration": 100000000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 200000000,
            "eval_count": 20,
            "eval_duration": 700000000,
        }

        yield {"list": mock_list, "show": mock_show, "chat": mock_chat}


@pytest.fixture
def cultural_analysis_data():
    """Sample cultural analysis data for testing."""
    return [
        {
            "content": "Adobo is a popular Filipino dish",
            "confidence": 0.9,
            "explanation": "Directly mentions Filipino food",
            "boost_factor": 1.3,
        },
        {"content": "Programming in Python", "confidence": 0.0, "explanation": "No cultural content", "boost_factor": 1.0},
        {
            "content": "Bayanihan spirit in communities",
            "confidence": 0.85,
            "explanation": "Filipino cultural value",
            "boost_factor": 1.25,
        },
    ]


@pytest.fixture(autouse=True)
def mock_file_operations():
    """Mock file system operations to avoid actual file creation."""
    with patch("pathlib.Path.mkdir") as mock_mkdir, patch("pathlib.Path.exists", return_value=True) as mock_exists:
        yield {"mkdir": mock_mkdir, "exists": mock_exists}


@pytest.fixture
def mock_time():
    """Mock time.time() for consistent timestamp testing."""
    fixed_time = 1234567890.0
    with patch("time.time", return_value=fixed_time):
        yield fixed_time


# Pytest configuration
def pytest_configure(config):
    """Configure pytest settings."""
    # Add custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark tests by location
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "test_" in item.nodeid:
            item.add_marker(pytest.mark.unit)

        # Mark slow tests
        if any(keyword in item.name.lower() for keyword in ["batch", "large", "slow"]):
            item.add_marker(pytest.mark.slow)
