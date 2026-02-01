"""
Result Fusion Module for WiQAS

Implements Reciprocal Rank Fusion (RRF) for merging results from multiple queries.
Used primarily for query decomposition and multilingual retrieval.
"""

from typing import Any


def reciprocal_rank_fusion(
    results_list: list[list[dict[str, Any]]],
    k: int = 60,
    deduplicate: bool = True,
) -> list[dict[str, Any]]:
    """
    Merge multiple result lists using Reciprocal Rank Fusion.

    RRF Formula: score(d) = sum(1 / (k + rank(d, q)) for each query q)
    where k is a constant (default 60) and rank is the position in the result list.

    Args:
        results_list: List of result lists from different queries
        k: RRF constant (default 60, from original paper)
        deduplicate: Remove duplicate documents (by page_content)

    Returns:
        Merged and re-ranked list of documents with RRF scores

    Example:
        >>> query1_results = [doc1, doc2, doc3]
        >>> query2_results = [doc2, doc4, doc1]
        >>> fused = reciprocal_rank_fusion([query1_results, query2_results])
        >>> # doc2 gets highest score (appears in both top positions)
    """
    if not results_list:
        return []
    
    if len(results_list) == 1:
        return results_list[0]
    
    # Calculate RRF scores for each document
    doc_scores: dict[str, dict[str, Any]] = {}
    
    for query_results in results_list:
        for rank, doc in enumerate(query_results):
            # Use page_content as document identifier
            doc_id = doc.get("page_content", "") or str(doc.get("metadata", {}))
            
            if not doc_id:
                continue
            
            # Calculate RRF score contribution from this ranking
            rrf_score = 1.0 / (k + rank + 1)  # +1 because rank is 0-indexed
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "doc": doc,
                    "rrf_score": 0.0,
                    "appearances": 0,
                }
            
            doc_scores[doc_id]["rrf_score"] += rrf_score
            doc_scores[doc_id]["appearances"] += 1
    
    # Sort by RRF score (descending) and appearance count (descending for ties)
    fused_docs = sorted(
        doc_scores.values(),
        key=lambda x: (x["rrf_score"], x["appearances"]),
        reverse=True,
    )
    
    # Extract documents and add RRF metadata
    result = []
    for item in fused_docs:
        doc = item["doc"].copy()
        
        # Add fusion metadata
        if "metadata" not in doc:
            doc["metadata"] = {}
        
        doc["metadata"]["rrf_score"] = item["rrf_score"]
        doc["metadata"]["query_appearances"] = item["appearances"]
        doc["metadata"]["fusion_method"] = "reciprocal_rank_fusion"
        
        result.append(doc)
    
    # Deduplicate if requested
    if deduplicate:
        seen_content = set()
        deduplicated = []
        
        for doc in result:
            content = doc.get("page_content", "")
            if content and content not in seen_content:
                seen_content.add(content)
                deduplicated.append(doc)
        
        return deduplicated
    
    return result


def weighted_fusion(
    results_list: list[list[dict[str, Any]]],
    weights: list[float] | None = None,
    deduplicate: bool = True,
) -> list[dict[str, Any]]:
    """
    Merge multiple result lists using weighted score fusion.

    Args:
        results_list: List of result lists from different queries
        weights: Weight for each result list (default: equal weights)
        deduplicate: Remove duplicate documents

    Returns:
        Merged and re-ranked list of documents with weighted scores
    """
    if not results_list:
        return []
    
    if len(results_list) == 1:
        return results_list[0]
    
    # Default to equal weights
    if weights is None:
        weights = [1.0] * len(results_list)
    
    if len(weights) != len(results_list):
        raise ValueError("Number of weights must match number of result lists")
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Calculate weighted scores
    doc_scores: dict[str, dict[str, Any]] = {}
    
    for results, weight in zip(results_list, weights):
        for doc in results:
            doc_id = doc.get("page_content", "") or str(doc.get("metadata", {}))
            
            if not doc_id:
                continue
            
            # Get existing score or use default
            doc_score = doc.get("metadata", {}).get("score", 0.5)
            weighted_score = doc_score * weight
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "doc": doc,
                    "weighted_score": 0.0,
                    "appearances": 0,
                }
            
            doc_scores[doc_id]["weighted_score"] += weighted_score
            doc_scores[doc_id]["appearances"] += 1
    
    # Sort by weighted score
    fused_docs = sorted(
        doc_scores.values(),
        key=lambda x: x["weighted_score"],
        reverse=True,
    )
    
    # Extract and annotate documents
    result = []
    for item in fused_docs:
        doc = item["doc"].copy()
        
        if "metadata" not in doc:
            doc["metadata"] = {}
        
        doc["metadata"]["weighted_score"] = item["weighted_score"]
        doc["metadata"]["query_appearances"] = item["appearances"]
        doc["metadata"]["fusion_method"] = "weighted_fusion"
        
        result.append(doc)
    
    # Deduplicate if requested
    if deduplicate:
        seen_content = set()
        deduplicated = []
        
        for doc in result:
            content = doc.get("page_content", "")
            if content and content not in seen_content:
                seen_content.add(content)
                deduplicated.append(doc)
        
        return deduplicated
    
    return result
