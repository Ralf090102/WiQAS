"""
Handles interaction with Ollama LLM models.
"""

import time
from threading import Lock
from typing import TYPE_CHECKING, Optional

import ollama

from src.utilities.utils import ensure_config, log_debug, log_error, log_info, log_warning

if TYPE_CHECKING:
    from src.utilities.config import WiQASConfig

# Thread-safe cache management
_cache_lock = Lock()
_connection_cache = {"last_check_time": 0, "last_check_result": False, "cache_duration": 90}
_model_existence_cache = {}


def _extract_model_names(models_response) -> list[str]:
    """Extract model names from Ollama list response (object or dict format)."""
    model_names: list[str] = []

    models = []
    if hasattr(models_response, "models"):
        models = models_response.models
    elif isinstance(models_response, dict):
        models = models_response.get("models", [])

    for model in models:
        name = None
        if hasattr(model, "model"):
            name = model.model
        elif hasattr(model, "name"):
            name = model.name
        elif isinstance(model, dict):
            name = model.get("name") or model.get("model")

        if name:
            model_names.append(name)

    return model_names


def clear_caches():
    """Clear all internal caches."""
    with _cache_lock:
        # Update the existing dictionary instead of creating a new one
        _connection_cache.clear()
        _connection_cache.update({"last_check_time": 0, "last_check_result": False, "cache_duration": 90})
        _model_existence_cache.clear()


def model_exists(model: str, config: Optional["WiQASConfig"] = None) -> bool:
    """
    Check if the specified model exists in Ollama.

    Args:
        model: The name of the model to check
        config: Configuration object for logging

    Returns:
        True if the model exists, False otherwise
    """
    config = ensure_config(config)

    with _cache_lock:
        if model in _model_existence_cache:
            result = _model_existence_cache[model]
            log_debug(f"Model '{model}' existence from cache: {result}", config)
            return result

    try:
        ollama.show(model)
        with _cache_lock:
            _model_existence_cache[model] = True
        log_debug(f"Model '{model}' exists in Ollama", config)
        return True
    except Exception as e:
        with _cache_lock:
            _model_existence_cache[model] = False
        log_debug(f"Model '{model}' not found: {e}", config)
        return False


def check_ollama_connection(config: Optional["WiQASConfig"] = None) -> bool:
    """
    Check if Ollama service is running and accessible (with caching).

    Args:
        config: Configuration object for logging and caching settings

    Returns:
        True if Ollama is accessible, False otherwise
    """
    config = ensure_config(config)

    now = time.time()

    # Use cached result if recent
    with _cache_lock:
        if now - _connection_cache["last_check_time"] < _connection_cache["cache_duration"]:
            log_debug(f"Using cached connection result: {_connection_cache['last_check_result']}", config)
            return _connection_cache["last_check_result"]

    try:
        ollama.list()
        with _cache_lock:
            _connection_cache["last_check_result"] = True
            _connection_cache["last_check_time"] = now
        log_debug("Ollama connection successful", config)
    except Exception as e:
        with _cache_lock:
            _connection_cache["last_check_result"] = False
            _connection_cache["last_check_time"] = now
        log_error(f"Cannot connect to Ollama: {e}", config=config)
        log_info("Make sure Ollama is running (ollama serve)", config=config)

    return _connection_cache["last_check_result"]


def get_available_models(config: Optional["WiQASConfig"] = None) -> list:
    """
    Get list of available Ollama models.

    Args:
        config: Configuration object for logging

    Returns:
        List of available model names
    """
    config = ensure_config(config)

    try:
        models_response = ollama.list()
        log_debug(f"Ollama models response type: {type(models_response)}", config)

        model_names = []

        # Handle ListResponse object
        if hasattr(models_response, "models"):
            models = models_response.models
            log_debug(f"Found {len(models)} models in response", config)

            for model in models:
                name = None
                if hasattr(model, "model"):
                    name = model.model
                elif hasattr(model, "name"):
                    name = model.name
                elif isinstance(model, dict):
                    name = model.get("name") or model.get("model") or model.get("id")

                if name:
                    model_names.append(name)
                    log_debug(f"Found model: {name}", config)
                else:
                    log_debug(f"Unknown model format: {model}", config)

        return model_names
    except Exception as e:
        log_error(f"Failed to get available models: {e}", config=config)
        return []


def check_model_availability(model: str, config: Optional["WiQASConfig"] = None) -> bool:
    """
    Check if the specified model is available in Ollama.

    Args:
        model: Model name to check
        config: Configuration object for logging

    Returns:
        True if model is available, False otherwise
    """
    config = ensure_config(config)

    # Check Ollama connection
    if not check_ollama_connection(config):
        return False

    # Check model exists
    if model_exists(model, config):
        return True

    # Model not found
    try:
        available_models = get_available_models(config)
        log_warning(f"Model '{model}' not found. Available: {available_models}", config=config)
        log_info(f"To install: ollama pull {model}", config=config)
    except Exception as e:
        log_error(f"Failed to list available models: {e}", config=config)

    return False


def switch_active_ollama_model(
    new_model: str,
    current_model: str | None = None,
    base_url: str = "http://localhost:11434",
    config: Optional["WiQASConfig"] = None,
    auxiliary_models_to_unload: list[str] | None = None,
    load_keep_alive_seconds: int = 300,
    max_retries: int = 2,
) -> dict[str, str | bool | None]:
    """
    Switch active Ollama generation model safely.

    Steps:
    1. Verify target model exists in Ollama.
    2. Unload current model (if any and different).
    3. Pre-load new model and keep it alive in memory.
    """
    config = ensure_config(config)

    if not new_model or not new_model.strip():
        raise ValueError("Model name cannot be empty")

    new_model = new_model.strip()
    client = ollama.Client(host=base_url)

    try:
        models_response = client.list()
        available_models = _extract_model_names(models_response)
    except Exception as e:
        log_error(f"Failed to list models from Ollama: {e}", config=config)
        raise RuntimeError(f"Failed to communicate with Ollama: {e}") from e

    if new_model not in available_models:
        raise ValueError(f"Model '{new_model}' not found in Ollama")

    if current_model == new_model:
        return {
            "status": "success",
            "message": f"Model '{new_model}' is already active",
            "previous_model": current_model,
            "active_model": new_model,
            "already_active": True,
        }

    models_to_unload: list[str] = []
    if current_model:
        models_to_unload.append(current_model)
    if auxiliary_models_to_unload:
        models_to_unload.extend(auxiliary_models_to_unload)

    # Deduplicate and never unload the target model itself
    models_to_unload = list(dict.fromkeys([m for m in models_to_unload if m and m != new_model]))

    for model_name in models_to_unload:
        try:
            client.generate(model=model_name, prompt=".", keep_alive=0, stream=False)
            if model_name == current_model:
                log_info(f"Unloaded previous model: {model_name}", config=config)
            else:
                log_info(f"Unloaded auxiliary model: {model_name}", config=config)
        except Exception as e:
            log_warning(f"Could not explicitly unload model '{model_name}': {e}", config=config)

    load_error = None
    for attempt in range(1, max_retries + 1):
        try:
            client.generate(
                model=new_model,
                prompt="pre-load",
                keep_alive=load_keep_alive_seconds,
                stream=False,
            )
            log_info(
                f"Loaded and activated model: {new_model} (keep_alive={load_keep_alive_seconds}s)",
                config=config,
            )
            load_error = None
            break
        except Exception as e:
            load_error = e
            error_text = str(e).lower()
            is_oom = "out of memory" in error_text or "cudamalloc" in error_text or "cuda" in error_text

            if attempt < max_retries and is_oom:
                log_warning(
                    f"OOM while loading '{new_model}' (attempt {attempt}/{max_retries}). Retrying after cleanup...",
                    config=config,
                )
                for model_name in models_to_unload:
                    try:
                        client.generate(model=model_name, prompt=".", keep_alive=0, stream=False)
                    except Exception:
                        pass
                time.sleep(2 * attempt)
                continue

            break

    if load_error is not None:
        log_error(f"Failed to load model '{new_model}': {load_error}", config=config)
        raise RuntimeError(f"Failed to load model '{new_model}' in Ollama: {load_error}") from load_error

    with _cache_lock:
        _model_existence_cache[new_model] = True

    return {
        "status": "success",
        "message": f"Switched active model from '{current_model or 'none'}' to '{new_model}'",
        "previous_model": current_model,
        "active_model": new_model,
        "already_active": False,
    }


def _validate_response(response: str, config: Optional["WiQASConfig"] = None) -> str:
    """Validate and clean response content."""
    config = ensure_config(config)

    if not response or not response.strip():
        log_warning("Empty response generated", config=config)
        return "[Warning: Empty response generated]"

    cleaned = response.strip()

    if len(cleaned) < 3:
        log_warning(f"Very short response: '{cleaned}'", config=config)

    log_debug(f"Response generated: {len(cleaned)} chars, {len(cleaned.split())} words", config)

    return cleaned


def generate_response(
    prompt: str,
    config: Optional["WiQASConfig"] = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_message: str | None = None,
    stream: bool = False,
    on_token=None,
) -> str:
    """
    Generate a response using Ollama LLM.

    Args:
        prompt: The input prompt for the model
        config: Configuration object with LLM settings (uses default if None)
        model: Override model (uses config default if None)
        temperature: Override temperature (uses config default if None)
        max_tokens: Maximum tokens to generate
        system_message: Optional system message
        stream: Whether to stream the response
        on_token: Callback for streaming tokens

    Returns:
        Generated response text or error message
    """
    config = ensure_config(config)

    if not prompt or not prompt.strip():
        log_error("Empty prompt provided", config=config)
        return "[Error: Empty prompt]"

    # Use config defaults
    model = model or config.rag.llm.model
    temperature = temperature if temperature is not None else config.rag.llm.temperature
    system_message = system_message or config.rag.llm.system_prompt

    # Validate model availability
    if not check_model_availability(model, config):
        return f"[Error: Model '{model}' not available]"

    log_debug(f"Generating response with model '{model}' (temp={temperature})", config)

    try:
        # Prepare messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        # Prepare options
        options = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens
        if config.rag.llm.timeout:
            options["timeout"] = config.rag.llm.timeout

        # Generate response
        if stream:
            content = ""
            for chunk in ollama.chat(model=model, messages=messages, options=options, stream=True):
                part = chunk.get("message", {}).get("content", "")
                if part:
                    if on_token:
                        on_token(part)
                    content += part
            result = _validate_response(content, config)
        else:
            response = ollama.chat(model=model, messages=messages, options=options)
            content = response.get("message", {}).get("content", "")
            result = _validate_response(content, config)

        return result

    except ollama.ResponseError as e:
        log_error(f"Ollama API error: {e}", config=config)
        return f"[Error: Ollama API error - {e}]"
    except Exception as e:
        log_error(f"LLM generation failed: {e}", config=config)
        return "[Error: Failed to generate response]"
