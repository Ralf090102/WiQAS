"""
Models API endpoints for managing LLM models
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
import ollama

from backend.models.models import (
    ModelConfig,
    UpdateModelRequest,
    UpdateLLMConfigRequest,
    ModelsListResponse,
    OllamaModelInfo,
)
from backend.dependencies import get_config_dependency
from src.utilities.config import WiQASConfig
from src.core.llm import check_ollama_connection

logger = logging.getLogger(__name__)
router = APIRouter()


# ========== GET CURRENT MODEL CONFIGURATION ==========
@router.get(
    "/api/models/config",
    summary="Get current LLM configuration",
    description="Returns the current active LLM model configuration",
    tags=["Models"],
    response_model=ModelConfig,
)
async def get_model_config(config: WiQASConfig = Depends(get_config_dependency)):
    """
    Get the current LLM model configuration.
    
    Args:
        config: Singleton config instance (injected)
    
    Returns:
        ModelConfig with current model settings from WiQASConfig
        
    Raises:
        HTTPException: If configuration cannot be loaded
    """
    try:
        return ModelConfig(
            model=config.rag.llm.model,
            base_url=config.rag.llm.base_url,
            temperature=config.rag.llm.temperature,
            top_p=config.rag.llm.top_p,
            max_tokens=config.rag.llm.max_tokens,
            timeout=config.rag.llm.timeout,
            system_prompt=config.rag.llm.system_prompt,
        )
        
    except Exception as e:
        logger.error(f"Failed to get model configuration: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model configuration: {str(e)}",
        )


# ========== UPDATE CURRENT MODEL ==========
@router.patch(
    "/api/models/config",
    summary="Update active LLM model",
    description="Update the active LLM model used for generation",
    tags=["Models"],
    response_model=ModelConfig,
)
async def update_model_config(
    request: UpdateModelRequest,
    config: WiQASConfig = Depends(get_config_dependency)
):
    """
    Update the active LLM model.
    
    Args:
        request: UpdateModelRequest with new model name
        config: Singleton config instance (injected)
        
    Returns:
        Updated ModelConfig
        
    Raises:
        HTTPException: If update fails or Ollama is not available
    """
    try:
        # Check if Ollama is running
        if not check_ollama_connection():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama service is not running. Please start Ollama first.",
            )
        
        # Verify the model exists in Ollama
        try:
            models_response = ollama.list()
            available_models = [m["name"] for m in models_response.get("models", [])]
            if request.model not in available_models:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{request.model}' not found in Ollama.",
                )
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to communicate with Ollama: {str(e)}",
            )

        # Update the config
        config.rag.llm.model = request.model
        config.save()
        
        logger.info(f"Active LLM model updated to: {request.model}")
        
        return ModelConfig(
            model=config.rag.llm.model,
            base_url=config.rag.llm.base_url,
            temperature=config.rag.llm.temperature,
            top_p=config.rag.llm.top_p,
            max_tokens=config.rag.llm.max_tokens,
            timeout=config.rag.llm.timeout,
            system_prompt=config.rag.llm.system_prompt,
        )

    except HTTPException:
        raise  # Re-raise HTTPException to preserve status code and detail
    except Exception as e:
        logger.error(f"Failed to update model configuration: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


# ========== SET ACTIVE MODEL (UNLOAD/LOAD) ==========
class SetActiveModelRequest(UpdateModelRequest):
    pass

@router.post(
    "/api/models/set-active",
    summary="Set the active generation model",
    description="Safely switches the active LLM by unloading the current model before loading the new one to prevent memory overload.",
    tags=["Models"],
    status_code=status.HTTP_204_NO_CONTENT,
)
async def set_active_model(
    request: SetActiveModelRequest,
    config: WiQASConfig = Depends(get_config_dependency)
):
    """
    Sets the active generation model, ensuring only one is loaded at a time.
    
    This endpoint prevents GPU out-of-memory errors by:
    1. Identifying the currently loaded generation model.
    2. Sending a request to Ollama to unload it (`keep_alive`: 0).
    3. Sending a request to load the new model and keep it in memory (`keep_alive`: -1).
    4. Updating the application's configuration to reflect the change.
    
    Args:
        request: Request containing the name of the model to activate.
        config: Singleton config instance (injected).
    """
    new_model = request.model
    current_model = config.rag.llm.model
    
    if new_model == current_model:
        logger.info(f"Model '{new_model}' is already active. No change needed.")
        return

    try:
        client = ollama.Client(host=config.rag.llm.base_url)
        
        # 1. Unload the current model if it's different from the new one
        if current_model and new_model != current_model:
            logger.info(f"Unloading current model: {current_model}")
            try:
                # Setting keep_alive to 0 unloads the model after the request
                client.generate(model=current_model, prompt=".", keep_alive=0, stream=False)
            except Exception as e:
                # This might fail if the model wasn't loaded, which is fine.
                logger.warning(f"Could not explicitly unload model {current_model}: {e}")

        # 2. Load the new model and keep it in memory
        logger.info(f"Loading and activating new model: {new_model}")
        client.generate(model=new_model, prompt="pre-load", keep_alive=-1, stream=False)

        # 3. Update the configuration
        config.rag.llm.model = new_model
        config.save()
        
        logger.info(f"Successfully activated new model: {new_model}")

    except Exception as e:
        logger.error(f"Failed to set active model to '{new_model}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to switch model in Ollama: {str(e)}",
        )


# ========== UPDATE LLM CONFIGURATION PARAMETERS ==========
@router.patch(
    "/api/models/parameters",
    summary="Update LLM configuration parameters",
    description="Update temperature, top_p, max_tokens, timeout, and system_prompt (does not change model name)",
    tags=["Models"],
    response_model=ModelConfig,
)
async def update_llm_parameters(
    request: UpdateLLMConfigRequest,
    config: WiQASConfig = Depends(get_config_dependency)
):
    """
    Update LLM configuration parameters.
    
    Allows updating:
    - temperature: Text generation randomness (0.0-2.0)
    - top_p: Nucleus sampling parameter (0.0-1.0)
    - max_tokens: Maximum tokens to generate (null for unlimited)
    - timeout: Request timeout in seconds
    - system_prompt: System prompt for the LLM
    
    Does NOT update:
    - model: Use PATCH /api/models/config instead
    - base_url: Should remain localhost:11434
    
    Args:
        request: UpdateLLMConfigRequest with optional parameters to update
        config: Singleton config instance (injected)
        
    Returns:
        Updated ModelConfig with all current settings
        
    Raises:
        HTTPException: If validation fails or update fails
    """
    try:
        # Update only the provided fields
        if request.temperature is not None:
            if not 0.0 <= request.temperature <= 2.0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Temperature must be between 0.0 and 2.0"
                )
            config.rag.llm.temperature = request.temperature
            logger.info(f"Updated temperature to: {request.temperature}")
        
        if request.top_p is not None:
            if not 0.0 <= request.top_p <= 1.0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="top_p must be between 0.0 and 1.0"
                )
            config.rag.llm.top_p = request.top_p
            logger.info(f"Updated top_p to: {request.top_p}")
        
        if request.max_tokens is not None:
            if request.max_tokens < 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="max_tokens must be at least 1 (or null for unlimited)"
                )
            config.rag.llm.max_tokens = request.max_tokens
            logger.info(f"Updated max_tokens to: {request.max_tokens}")
        
        if request.timeout is not None:
            if request.timeout < 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="timeout must be at least 1 second"
                )
            config.rag.llm.timeout = request.timeout
            logger.info(f"Updated timeout to: {request.timeout}")
        
        if request.system_prompt is not None:
            if not request.system_prompt.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="system_prompt cannot be empty"
                )
            config.rag.llm.system_prompt = request.system_prompt
            logger.info(f"Updated system_prompt")
        
        logger.info("✅ LLM configuration parameters updated successfully")
        
        # Return complete updated configuration
        return ModelConfig(
            model=config.rag.llm.model,
            base_url=config.rag.llm.base_url,
            temperature=config.rag.llm.temperature,
            top_p=config.rag.llm.top_p,
            max_tokens=config.rag.llm.max_tokens,
            timeout=config.rag.llm.timeout,
            system_prompt=config.rag.llm.system_prompt,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update LLM parameters: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update LLM parameters: {str(e)}",
        )


# ========== LIST OLLAMA MODELS ==========
@router.get(
    "/api/models",
    summary="List available Ollama models",
    description="Get list of all available Ollama models with current active model",
    tags=["Models"],
    response_model=ModelsListResponse,
)
async def list_models(config: WiQASConfig = Depends(get_config_dependency)):
    """
    List all available Ollama models and show which one is currently active.
    
    Args:
        config: Singleton config instance (injected)
    
    Returns detailed information about each installed model including:
    - Model name
    - Model ID (digest)
    - Size
    - Modified date
    - Current active model indicator
    
    Returns:
        ModelsListResponse with models list and current active model
        
    Raises:
        HTTPException: If Ollama is not available or request fails
    """
    try:
        # Check if Ollama is running
        if not check_ollama_connection():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama service is not running. Please start Ollama first.",
            )
        
        # Get current active model from singleton config
        current_model = config.rag.llm.model
        
        # Get models from Ollama
        models_response = ollama.list()
        
        models_list = []
        
        # Parse the response
        if hasattr(models_response, 'models'):
            for model in models_response.models:
                # Get modified_at and convert datetime to string if needed
                modified_at = getattr(model, 'modified_at', None)
                if modified_at and hasattr(modified_at, 'isoformat'):
                    # It's a datetime object, convert to ISO string
                    modified_str = modified_at.isoformat()
                else:
                    # Already a string or None
                    modified_str = modified_at
                
                model_info = OllamaModelInfo(
                    name=getattr(model, 'model', getattr(model, 'name', 'unknown')),
                    id=getattr(model, 'digest', 'unknown')[:12],  # Short digest like CLI
                    size=_format_size(getattr(model, 'size', 0)),
                    size_bytes=getattr(model, 'size', 0),
                    modified=modified_str,
                    details={
                        "format": getattr(model, 'details', {}).get('format') if hasattr(model, 'details') else None,
                        "family": getattr(model, 'details', {}).get('family') if hasattr(model, 'details') else None,
                        "parameter_size": getattr(model, 'details', {}).get('parameter_size') if hasattr(model, 'details') else None,
                    }
                )
                models_list.append(model_info)
        
        return ModelsListResponse(
            status="success",
            current_model=current_model,
            total_models=len(models_list),
            models=models_list,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list Ollama models: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve Ollama models: {str(e)}",
        )


def _format_size(size_bytes: int) -> str:
    """
    Format byte size to human-readable format (matches Ollama CLI output).
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "4.4 GB", "274 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.0f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.1f} GB"
