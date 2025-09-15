"""
Tests for the LLM module (src.core.llm).
Tests the most important features: connection checking, model validation, and response generation.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import ollama

from src.core.llm import (
    clear_caches,
    model_exists,
    check_ollama_connection,
    get_available_models,
    check_model_availability,
    generate_response,
    _validate_response,
    _cache_lock,
    _connection_cache,
    _model_existence_cache,
)


class TestCacheManagement:
    """Test cache management functionality."""

    def test_clear_caches(self, isolated_caches):
        """Test that clear_caches properly resets all cache structures."""
        # Populate caches with test data
        global _connection_cache, _model_existence_cache
        _connection_cache["last_check_result"] = True
        _connection_cache["last_check_time"] = time.time()
        _model_existence_cache["test_model"] = True

        # Clear caches
        clear_caches()

        # Verify caches are reset
        assert _connection_cache["last_check_result"] is False
        assert _connection_cache["last_check_time"] == 0
        assert len(_model_existence_cache) == 0


class TestModelExistence:
    """Test model existence checking functionality."""

    @patch("src.core.llm.ollama.show")
    def test_model_exists_success(self, mock_show, mock_config):
        """Test successful model existence check."""
        mock_show.return_value = {"model": "test_model"}

        result = model_exists("test_model", mock_config)

        assert result is True
        mock_show.assert_called_once_with("test_model")
        # Verify caching
        assert _model_existence_cache["test_model"] is True

    @patch("src.core.llm.ollama.show")
    def test_model_exists_failure(self, mock_show, mock_config):
        """Test model existence check when model doesn't exist."""
        mock_show.side_effect = Exception("Model not found")

        result = model_exists("nonexistent_model", mock_config)

        assert result is False
        assert _model_existence_cache["nonexistent_model"] is False

    def test_model_exists_cached_result(self, mock_config):
        """Test that cached results are used without making API calls."""
        _model_existence_cache["cached_model"] = True

        with patch("src.core.llm.ollama.show") as mock_show:
            result = model_exists("cached_model", mock_config)

            assert result is True
            mock_show.assert_not_called()


class TestOllamaConnection:
    """Test Ollama connection checking functionality."""

    @patch("src.core.llm.ollama.list")
    def test_check_ollama_connection_success(self, mock_list, mock_config):
        """Test successful Ollama connection check."""
        mock_list.return_value = {"models": []}

        result = check_ollama_connection(mock_config)

        assert result is True
        mock_list.assert_called_once()
        assert _connection_cache["last_check_result"] is True

    @patch("src.core.llm.ollama.list")
    def test_check_ollama_connection_failure(self, mock_list, mock_config, isolated_caches):
        """Test Ollama connection check when service is unavailable."""
        mock_list.side_effect = Exception("Connection refused")

        result = check_ollama_connection(mock_config)

        assert result is False
        assert _connection_cache["last_check_result"] is False

    @patch("time.time")
    @patch("src.core.llm.ollama.list")
    def test_check_ollama_connection_uses_cache(self, mock_list, mock_time, mock_config):
        """Test that connection check uses cached results when still valid."""
        # Set up time progression
        mock_time.side_effect = [1000, 1015]  # 15 seconds later (within 30s cache duration)

        # First call
        mock_list.return_value = {"models": []}
        result1 = check_ollama_connection(mock_config)

        # Second call should use cache
        result2 = check_ollama_connection(mock_config)

        assert result1 is True
        assert result2 is True
        mock_list.assert_called_once()  # Only called once due to caching


class TestAvailableModels:
    """Test available models retrieval functionality."""

    @patch("src.core.llm.ollama.list")
    def test_get_available_models_success(self, mock_list, mock_config):
        """Test successful retrieval of available models."""
        # Mock response with models attribute
        mock_response = Mock()
        mock_model1 = Mock()
        mock_model1.model = "mistral:latest"
        mock_model1.name = None

        mock_model2 = Mock()
        mock_model2.model = "mistral:7b"
        mock_model2.name = None

        mock_response.models = [mock_model1, mock_model2, {"name": "gemma:2b"}]  # Dict format
        mock_list.return_value = mock_response

        result = get_available_models(mock_config)

        assert "mistral:latest" in result
        assert "mistral:7b" in result
        assert "gemma:2b" in result
        mock_list.assert_called_once()

    @patch("src.core.llm.ollama.list")
    def test_get_available_models_failure(self, mock_list, mock_config):
        """Test get_available_models when API call fails."""
        mock_list.side_effect = Exception("API Error")

        result = get_available_models(mock_config)

        assert result == []

    @patch("src.core.llm.ollama.list")
    def test_get_available_models_empty_response(self, mock_list, mock_config):
        """Test handling of empty or malformed response."""
        mock_list.return_value = None

        result = get_available_models(mock_config)

        assert result == []


class TestModelAvailability:
    """Test comprehensive model availability checking."""

    @patch("src.core.llm.check_ollama_connection")
    @patch("src.core.llm.model_exists")
    def test_check_model_availability_success(self, mock_model_exists, mock_connection, mock_config):
        """Test successful model availability check."""
        mock_connection.return_value = True
        mock_model_exists.return_value = True

        result = check_model_availability("mistral:latest", mock_config)

        assert result is True
        mock_connection.assert_called_once_with(mock_config)
        mock_model_exists.assert_called_once_with("mistral:latest", mock_config)

    @patch("src.core.llm.check_ollama_connection")
    def test_check_model_availability_no_connection(self, mock_connection, mock_config):
        """Test model availability check when Ollama is not connected."""
        mock_connection.return_value = False

        result = check_model_availability("mistral:latest", mock_config)

        assert result is False
        mock_connection.assert_called_once_with(mock_config)

    @patch("src.core.llm.check_ollama_connection")
    @patch("src.core.llm.model_exists")
    @patch("src.core.llm.get_available_models")
    def test_check_model_availability_model_not_found(self, mock_get_models, mock_model_exists, mock_connection, mock_config):
        """Test model availability check when model doesn't exist."""
        mock_connection.return_value = True
        mock_model_exists.return_value = False
        mock_get_models.return_value = ["mistral:latest", "mistral:7b"]

        result = check_model_availability("nonexistent_model", mock_config)

        assert result is False
        mock_get_models.assert_called_once_with(mock_config)


class TestResponseValidation:
    """Test response validation functionality."""

    def test_validate_response_normal(self, mock_config):
        """Test validation of normal response."""
        response = "This is a normal response with sufficient content."

        result = _validate_response(response, mock_config)

        assert result == response.strip()

    def test_validate_response_empty(self, mock_config):
        """Test validation of empty response."""
        result = _validate_response("", mock_config)

        assert result == "[Warning: Empty response generated]"

    def test_validate_response_whitespace_only(self, mock_config):
        """Test validation of whitespace-only response."""
        result = _validate_response("   \n\t  ", mock_config)

        assert result == "[Warning: Empty response generated]"

    def test_validate_response_very_short(self, mock_config):
        """Test validation of very short response."""
        result = _validate_response("Hi", mock_config)

        assert result == "Hi"  # Still valid, just logged as warning

    def test_validate_response_with_extra_whitespace(self, mock_config):
        """Test that extra whitespace is properly stripped."""
        response = "   \n  Valid response with extra whitespace  \n  "

        result = _validate_response(response, mock_config)

        assert result == "Valid response with extra whitespace"


class TestResponseGeneration:
    """Test LLM response generation functionality."""

    @patch("src.core.llm.check_model_availability")
    @patch("src.core.llm.ollama.chat")
    def test_generate_response_success(self, mock_chat, mock_availability, mock_config):
        """Test successful response generation."""
        mock_availability.return_value = True
        mock_chat.return_value = {"message": {"content": "This is a test response"}, "done": True}

        result = generate_response("Test prompt", mock_config)

        assert result == "This is a test response"
        mock_availability.assert_called_once()
        mock_chat.assert_called_once()

        # Verify chat was called with correct parameters
        call_args = mock_chat.call_args
        assert "model" in call_args.kwargs
        assert "messages" in call_args.kwargs
        assert call_args.kwargs["messages"][0]["role"] == "system"
        assert call_args.kwargs["messages"][1]["role"] == "user"
        assert call_args.kwargs["messages"][1]["content"] == "Test prompt"

    @patch("src.core.llm.check_model_availability")
    def test_generate_response_empty_prompt(self, mock_availability, mock_config):
        """Test response generation with empty prompt."""
        result = generate_response("", mock_config)

        assert result == "[Error: Empty prompt]"
        mock_availability.assert_not_called()

    @patch("src.core.llm.check_model_availability")
    def test_generate_response_model_unavailable(self, mock_availability, mock_config):
        """Test response generation when model is unavailable."""
        mock_availability.return_value = False

        result = generate_response("Test prompt", mock_config)

        assert "[Error: Model" in result
        assert "not available]" in result

    @patch("src.core.llm.check_model_availability")
    @patch("src.core.llm.ollama.chat")
    def test_generate_response_with_overrides(self, mock_chat, mock_availability, mock_config):
        """Test response generation with parameter overrides."""
        mock_availability.return_value = True
        mock_chat.return_value = {"message": {"content": "Override response"}, "done": True}

        result = generate_response(
            prompt="Test prompt",
            config=mock_config,
            model="custom_model",
            temperature=0.8,
            max_tokens=100,
            system_message="Custom system message",
        )

        assert result == "Override response"

        # Verify overrides were used
        call_args = mock_chat.call_args
        assert call_args.kwargs["model"] == "custom_model"
        assert call_args.kwargs["options"]["temperature"] == 0.8
        assert call_args.kwargs["options"]["num_predict"] == 100
        assert call_args.kwargs["messages"][0]["content"] == "Custom system message"

    @patch("src.core.llm.check_model_availability")
    @patch("src.core.llm.ollama.chat")
    def test_generate_response_streaming(self, mock_chat, mock_availability, mock_config):
        """Test streaming response generation."""
        mock_availability.return_value = True

        # Mock streaming response
        mock_chat.return_value = iter(
            [
                {"message": {"content": "This "}},
                {"message": {"content": "is "}},
                {"message": {"content": "streaming"}},
            ]
        )

        tokens_received = []

        def token_callback(token):
            tokens_received.append(token)

        result = generate_response("Test prompt", mock_config, stream=True, on_token=token_callback)

        assert result == "This is streaming"
        assert tokens_received == ["This ", "is ", "streaming"]

    @patch("src.core.llm.check_model_availability")
    @patch("src.core.llm.ollama.chat")
    def test_generate_response_ollama_error(self, mock_chat, mock_availability, mock_config):
        """Test response generation with Ollama API error."""
        mock_availability.return_value = True
        mock_chat.side_effect = ollama.ResponseError("API Error")

        result = generate_response("Test prompt", mock_config)

        assert "[Error: Ollama API error" in result

    @patch("src.core.llm.check_model_availability")
    @patch("src.core.llm.ollama.chat")
    def test_generate_response_general_exception(self, mock_chat, mock_availability, mock_config):
        """Test response generation with general exception."""
        mock_availability.return_value = True
        mock_chat.side_effect = Exception("General error")

        result = generate_response("Test prompt", mock_config)

        assert result == "[Error: Failed to generate response]"


@pytest.mark.integration
@pytest.mark.requires_ollama
class TestLLMIntegration:
    """Integration tests that require actual Ollama service."""

    def test_full_llm_workflow_integration(self):
        """
        Integration test for complete LLM workflow.
        Tests actual Ollama connection, model availability, and response generation.

        Requires Ollama to be running with at least one model available.
        """
        # Import here to get a real config, not mocked
        from src.utilities.config import WiQASConfig

        real_config = WiQASConfig()

        # Test connection
        connection_result = check_ollama_connection(real_config)
        if not connection_result:
            pytest.skip("Ollama service not available for integration test")

        # Get available models
        available_models = get_available_models(real_config)
        if not available_models:
            pytest.skip("No models available in Ollama for integration test")

        # Use first available model
        test_model = available_models[0]

        # Test model availability
        model_available = check_model_availability(test_model, real_config)
        assert model_available is True

        # Test response generation
        response = generate_response(
            prompt="Hello, please respond with exactly 'Integration test successful'",
            config=real_config,
            model=test_model,
            temperature=0.1,
        )

        # Verify response is valid and not an error
        assert not response.startswith("[Error:")
        assert not response.startswith("[Warning:")
        assert len(response.strip()) > 0

        print(f"Integration test completed with model: {test_model}")
        print(f"Response: {response}")


# Helper fixtures for this module
@pytest.fixture
def isolated_caches():
    """Provide completely isolated cache state for testing."""
    # Save current cache state
    old_connection_cache = _connection_cache.copy()
    old_model_cache = _model_existence_cache.copy()

    # Reset caches to known clean state
    clear_caches()

    yield

    # Restore original cache state
    _connection_cache.clear()
    _connection_cache.update(old_connection_cache)
    _model_existence_cache.clear()
    _model_existence_cache.update(old_model_cache)


@pytest.fixture(autouse=True)
def clean_caches():
    """Automatically clean caches before each test."""
    clear_caches()
    yield
