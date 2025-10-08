"""
WiQAS Retrieval Evaluator

Evaluates the performance of the WiQAS retrieval system using cosine similarity
between retrieved content and ground truth context.
"""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.retriever import WiQASRetriever
from src.utilities.config import WiQASConfig
from src.utilities.utils import log_error, log_info, log_warning


class RetrievalEvaluator:
    """
    Evaluates retrieval performance using cosine similarity between 
    retrieved content and ground truth context.
    """

    def __init__(self, config: WiQASConfig | None = None):
        """
        Initialize the evaluator.
        
        Args:
            config: WiQAS configuration object
        """
        self.config = config or WiQASConfig()
        self.eval_config = self.config.rag.evaluation
        self.embedding_manager = EmbeddingManager(self.config)
        self.retriever = WiQASRetriever(self.config)
        
    def load_evaluation_dataset(self) -> list[dict[str, Any]]:
        """
        Load the evaluation dataset from JSON file.
        
        Returns:
            List of evaluation items
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            json.JSONDecodeError: If dataset file is invalid JSON
        """
        dataset_path = Path(self.eval_config.dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")
            
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                
            log_info(f"Loaded {len(dataset)} items from evaluation dataset")
            
            # Apply limit and randomization
            if self.eval_config.randomize:
                random.shuffle(dataset)
                log_info("Dataset randomized")
                
            if self.eval_config.limit is not None:
                dataset = dataset[:self.eval_config.limit]
                log_info(f"Limited dataset to {len(dataset)} items")
                
            return dataset
            
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in dataset file: {e}")
            
    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts using embeddings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Generate embeddings for both texts
            embedding1 = self.embedding_manager.encode_single(text1)
            embedding2 = self.embedding_manager.encode_single(text2)
            
            # Reshape for sklearn
            embedding1 = np.array(embedding1).reshape(1, -1)
            embedding2 = np.array(embedding2).reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            
            return float(similarity)
            
        except Exception as e:
            log_error(f"Error calculating cosine similarity: {e}")
            return 0.0
            
    def find_best_match(self, retrieved_results: list, ground_truth_context: str) -> tuple[float, str, int]:
        """
        Find the best matching retrieved result against ground truth.
        
        Args:
            retrieved_results: List of retrieved search results
            ground_truth_context: Ground truth text to match against
            
        Returns:
            Tuple of (best_similarity_score, best_matching_content, result_index)
        """
        if not retrieved_results:
            return 0.0, "", -1
            
        best_similarity = 0.0
        best_content = ""
        best_index = -1
        
        for i, result in enumerate(retrieved_results):
            similarity = self.calculate_cosine_similarity(result.content, ground_truth_context)
            if similarity > best_similarity:
                best_similarity = similarity
                best_content = result.content
                best_index = i
                
        return best_similarity, best_content, best_index
        
    def evaluate_single_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate a single item from the dataset.
        
        Args:
            item: Evaluation item containing question and ground_truth_context
            
        Returns:
            Dictionary with evaluation results
        """
        question = item.get("question", "")
        metadata = item.get("metadata", {})
        ground_truth_context = metadata.get("ground_truth_context", "")
        
        if not question or not ground_truth_context:
            log_warning(f"Skipping invalid item: missing question or ground_truth_context")
            return {
                "question": question,
                "ground_truth_context": ground_truth_context,
                "retrieved_content": "",
                "similarity_score": 0.0,
                "result_index": -1,
                "total_results": 0,
                "error": "Missing question or ground_truth_context"
            }
            
        try:
            results = self.retriever.query(
                query_text=question,
                k=self.eval_config.k_results,
                search_type=self.eval_config.search_type,
                enable_reranking=self.eval_config.enable_reranking,
                enable_mmr=self.eval_config.enable_mmr,
                llm_analysis=not self.eval_config.disable_cultural_llm_analysis,
                formatted=False  # Get raw results, not formatted string
            )
            
            if isinstance(results, str):
                # If we got an error string instead of results
                return {
                    "question": question,
                    "ground_truth_context": ground_truth_context,
                    "retrieved_content": "",
                    "similarity_score": 0.0,
                    "result_index": -1,
                    "total_results": 0,
                    "error": results
                }
            
            best_similarity, best_content, best_index = self.find_best_match(results, ground_truth_context)
            
            return {
                "question": question,
                "ground_truth_context": ground_truth_context,
                "retrieved_content": best_content,
                "similarity_score": best_similarity,
                "result_index": best_index,
                "total_results": len(results)
            }
            
        except Exception as e:
            log_error(f"Error evaluating item: {e}")
            return {
                "question": question,
                "ground_truth_context": ground_truth_context,
                "retrieved_content": "",
                "similarity_score": 0.0,
                "result_index": -1,
                "total_results": 0,
                "error": str(e)
            }
            
    def evaluate(self) -> dict[str, Any]:
        """
        Run full evaluation on the dataset.
        
        Returns:
            Dictionary containing evaluation results and statistics
        """
        log_info("Starting retrieval evaluation...")
        
        # Load dataset
        try:
            dataset = self.load_evaluation_dataset()
        except Exception as e:
            log_error(f"Failed to load evaluation dataset: {e}")
            return {"error": str(e)}
            
        if not dataset:
            log_warning("Empty evaluation dataset")
            return {"error": "Empty evaluation dataset"}
            
        # Evaluate each item
        results = []
        similarities = []
        above_threshold_count = 0
        error_count = 0
        
        log_info(f"Evaluating {len(dataset)} items...")
        
        for i, item in enumerate(dataset):
            log_info(f"Evaluating item {i+1}/{len(dataset)}: {item.get('question', 'No question')[:50]}...")
            
            result = self.evaluate_single_item(item)
            results.append(result)
            
            if "error" in result:
                error_count += 1
            else:
                similarities.append(result["similarity_score"])
                if result["similarity_score"] >= self.eval_config.similarity_threshold:
                    above_threshold_count += 1
                    
        # Calculate statistics
        if similarities:
            avg_similarity = np.mean(similarities)
            median_similarity = np.median(similarities)
            std_similarity = np.std(similarities)
            min_similarity = np.min(similarities)
            max_similarity = np.max(similarities)
        else:
            avg_similarity = median_similarity = std_similarity = min_similarity = max_similarity = 0.0
            
        success_rate = (len(similarities) / len(dataset)) * 100 if dataset else 0.0
        threshold_rate = (above_threshold_count / len(similarities)) * 100 if similarities else 0.0
        
        evaluation_summary = {
            "dataset_info": {
                "total_items": len(dataset),
                "successful_evaluations": len(similarities),
                "errors": error_count,
                "success_rate": f"{success_rate:.1f}%"
            },
            "similarity_statistics": {
                "average": f"{avg_similarity:.4f}",
                "median": f"{median_similarity:.4f}",
                "std_deviation": f"{std_similarity:.4f}",
                "min": f"{min_similarity:.4f}",
                "max": f"{max_similarity:.4f}"
            },
            "threshold_analysis": {
                "threshold": self.eval_config.similarity_threshold,
                "above_threshold": above_threshold_count,
                "above_threshold_rate": f"{threshold_rate:.1f}%"
            },
            "configuration": {
                "search_type": self.eval_config.search_type,
                "k_results": self.eval_config.k_results,
                "enable_reranking": self.eval_config.enable_reranking,
                "enable_mmr": self.eval_config.enable_mmr,
                "cultural_llm_disabled": self.eval_config.disable_cultural_llm_analysis,
                "randomized": self.eval_config.randomize,
                "limit": self.eval_config.limit
            },
            "detailed_results": results
        }
        
        log_info(f"Evaluation completed! Average similarity: {avg_similarity:.4f}")
        log_info(f"Items above threshold ({self.eval_config.similarity_threshold}): {above_threshold_count}/{len(similarities)} ({threshold_rate:.1f}%)")
        
        return evaluation_summary
        
    def save_results(self, results: dict[str, Any], output_path: str | Path) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save the results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            log_info(f"Evaluation results saved to: {output_path}")
        except Exception as e:
            log_error(f"Failed to save results: {e}")


def run_evaluation(config: WiQASConfig | None = None, output_path: str | None = None) -> dict[str, Any]:
    """
    Convenience function to run evaluation.
    
    Args:
        config: Optional WiQAS configuration
        output_path: Optional path to save results
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = RetrievalEvaluator(config)
    results = evaluator.evaluate()
    
    if output_path:
        evaluator.save_results(results, output_path)
        
    return results