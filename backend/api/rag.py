"""
RAG API Endpoints

Endpoints for Retrieval-Augmented Generation:
- Semantic search (query only)
- RAG ask (query + LLM generation)
- Streaming responses
"""

import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from backend.dependencies import (
    get_config_dependency,
    get_generator_dependency,
    get_retriever_dependency,
)
from backend.models.rag import (
    AskRequest,
    AskResponse,
    QueryRequest,
    QueryResponse,
    SearchResult,
    Source,
    StreamChunk,
    TimingBreakdown,
)
from src.generation.generator import WiQASGenerator
from src.retrieval.retriever import WiQASRetriever
from src.utilities.config import WiQASConfig

logger = logging.getLogger(__name__)

router = APIRouter()


# ========== SEMANTIC QUERY (RETRIEVAL ONLY) ==========
@router.post(
    "/api/query",
    response_model=QueryResponse,
    summary="Semantic search",
    description="Search knowledge base using semantic similarity (no LLM)",
    tags=["RAG"],
)
async def query_knowledge_base(
    request: QueryRequest,
    retriever: WiQASRetriever = Depends(get_retriever_dependency),
    config: WiQASConfig = Depends(get_config_dependency),
):
    """
    Perform semantic search on the knowledge base.
    
    Separates required argument (query) from optional settings.
    Matches run.py query command pattern:
    - Required: query string
    - Optional: k, enable_reranking, similarity_threshold, verbose
    
    Optional settings override config defaults when provided.
    
    Args:
        request: Query request with search parameters
        retriever: Retriever instance (injected)
        config: Configuration instance (injected)
        
    Returns:
        QueryResponse with search results and metadata
        
    Raises:
        HTTPException: If knowledge base is empty or search fails
    """
    try:
        start_time = time.time()
        
        # ===== EXTRACT REQUIRED ARGUMENT =====
        query = request.query
        
        # ===== APPLY OPTIONAL SETTINGS (override config defaults) =====
        k = request.k if request.k is not None else config.rag.retrieval.default_k
        enable_reranking = (
            request.enable_reranking
            if request.enable_reranking is not None
            else config.rag.retrieval.enable_reranking
        )
        similarity_threshold = (
            request.similarity_threshold
            if request.similarity_threshold is not None
            else config.rag.retrieval.similarity_threshold
        )
        verbose = request.verbose
        
        logger.info(
            f"Query request: '{query}' (k={k}, reranking={enable_reranking}, "
            f"threshold={similarity_threshold}, verbose={verbose})"
        )
        
        # Perform retrieval (matches run.py pattern)
        # Note: WiQAS retriever.query() can return formatted string or dict with results
        retrieval_result = retriever.query(
            query_text=query,
            k=k,
            search_type="hybrid",
            enable_reranking=enable_reranking,
            enable_mmr=True,
            formatted=False,  # Get raw results, not formatted string
            include_timing=verbose,
        )
        
        # Handle both dict and list returns
        if isinstance(retrieval_result, dict):
            results = retrieval_result.get("results", [])
        else:
            results = retrieval_result
        
        # Filter by similarity threshold
        original_count = len(results)
        results = [r for r in results if r.score >= similarity_threshold]
        filtered_count = original_count - len(results)
        
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} results below threshold {similarity_threshold}")
        
        processing_time = time.time() - start_time
        
        # Convert to response model
        search_results = [
            SearchResult(
                content=r.content,
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ]
        
        logger.info(
            f"Query completed: {len(search_results)} results in {processing_time:.3f}s"
        )
        
        # Build response metadata
        response_metadata = {
            "k_requested": k,
            "k_returned": len(search_results),
            "filtered_count": filtered_count,
            "reranking_enabled": enable_reranking,
            "hybrid_search": config.rag.retrieval.enable_hybrid_search,
            "mmr_enabled": config.rag.retrieval.enable_mmr,
            "similarity_threshold": similarity_threshold,
        }
        
        # Add verbose timing if requested
        if verbose:
            response_metadata["timing"] = {
                "total_time": processing_time,
                "results_per_second": len(search_results) / processing_time if processing_time > 0 else 0,
            }
        
        return QueryResponse(
            results=search_results,
            total_results=len(search_results),
            query=query,
            processing_time=processing_time,
            metadata=response_metadata,
        )
        
    except ValueError as e:
        # Knowledge base empty or validation error
        logger.warning(f"Query validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )


# ========== RAG ASK (RETRIEVAL + LLM) ==========
@router.post(
    "/api/ask",
    response_model=AskResponse,
    summary="RAG question answering",
    description="Answer questions using RAG (retrieval + LLM generation)",
    tags=["RAG"],
)
async def ask_question(
    request: AskRequest,
    generator: WiQASGenerator = Depends(get_generator_dependency),
    config: WiQASConfig = Depends(get_config_dependency),
):
    """
    Answer a question using RAG pipeline.
    
    Separates required argument (query) from optional settings.
    Matches run.py ask command pattern:
    - Required: query string
    - Optional: k, include_sources, temperature, max_tokens, verbose
    
    When verbose=True, includes detailed timing breakdown (matches run.py --verbose flag).
    
    Args:
        request: Ask request with question and parameters
        generator: Generator instance (injected)
        config: Configuration instance (injected)
        
    Returns:
        AskResponse with answer, sources, and optional timing breakdown
        
    Raises:
        HTTPException: If generation fails or knowledge base is empty
    """
    try:
        start_time = time.time()
        
        # ===== EXTRACT REQUIRED ARGUMENT =====
        query = request.query
        
        # ===== APPLY OPTIONAL SETTINGS (override config defaults) =====
        k = request.k if request.k is not None else config.rag.retrieval.default_k
        include_sources = request.include_sources
        verbose = request.verbose
        
        logger.info(
            f"Ask request: '{query}' (k={k}, sources={include_sources}, verbose={verbose})"
        )
        
        # Generate RAG response using WiQAS generator.generate()
        result = generator.generate(
            query=query,
            k=k,
            query_type=None,  # Auto-detected
            language=None,  # Auto-detected
            show_contexts=include_sources,
            include_timing=verbose,
            include_classification=False,
            enable_query_decomposition=False,
        )
        
        processing_time = time.time() - start_time
        
        # Extract answer and contexts from result
        answer = result["answer"]
        query_type = result.get("query_type", "general")
        contexts = result.get("contexts", [])
        
        # Convert contexts to sources (matches run.py source handling)
        sources = []
        if include_sources and contexts:
            for i, ctx in enumerate(contexts):
                # Handle both dict and SearchResult objects
                if isinstance(ctx, dict):
                    content = ctx.get("content", "")
                    metadata = ctx.get("metadata", {})
                    score = ctx.get("final_score", 0.0)
                else:
                    content = getattr(ctx, "content", "")
                    metadata = getattr(ctx, "metadata", {}) if hasattr(ctx, "metadata") else {}
                    score = getattr(ctx, "final_score", 0.0)
                
                source_file = metadata.get("source_file", metadata.get("source", "Unknown"))
                page = metadata.get("page")
                citation = source_file
                if page:
                    citation = f"{source_file} (page {page})"
                
                sources.append(
                    Source(
                        index=i + 1,
                        content=content[:500],  # Truncate long content
                        citation=citation,
                        score=score,
                        metadata=metadata,
                    )
                )
        
        # Extract timing breakdown if verbose mode (matches run.py --verbose)
        timing_breakdown = None
        if verbose and result.get("timing"):
            timing_obj = result["timing"]
            timing_breakdown = TimingBreakdown(
                embedding_time=getattr(timing_obj, "embedding_time", 0.0),
                search_time=getattr(timing_obj, "search_time", 0.0),
                reranking_time=getattr(timing_obj, "reranking_time", 0.0),
                mmr_time=getattr(timing_obj, "mmr_time", 0.0),
                context_preparation_time=getattr(timing_obj, "context_preparation_time", 0.0),
                prompt_building_time=getattr(timing_obj, "prompt_building_time", 0.0),
                llm_generation_time=getattr(timing_obj, "llm_generation_time", 0.0),
                total_time=getattr(timing_obj, "total_time", 0.0),
            )
        
        logger.info(
            f"Ask completed: {len(answer)} chars, "
            f"{len(sources)} sources in {processing_time:.3f}s"
        )
        
        return AskResponse(
            answer=answer,
            sources=sources,
            query=query,
            query_type=query_type,
            processing_time=processing_time,
            metadata={
                "model": config.rag.llm.model,
                "temperature": request.temperature or config.rag.llm.temperature,
                "k": k,
                "num_contexts_used": len(contexts),
            },
            timing=timing_breakdown,  # Only included when verbose=True
        )
        
    except ValueError as e:
        logger.warning(f"Ask validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Ask failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ask failed: {str(e)}",
        )





# ========== STREAMING NOT YET IMPLEMENTED ==========
# Streaming requires additional LLM callback implementation in WiQAS core
# TODO: Implement streaming support in src/core/llm.py

# @router.post("/api/ask/stream")
# async def ask_stream(...)
#     Placeholder for future streaming implementation

