"""
WiQAS Retrieval Evaluator

Evaluates the performance of the WiQAS retrieval system using cosine similarity
between retrieved content and ground truth context.
"""

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.retriever import WiQASRetriever
from src.utilities.config import WiQASConfig
from src.utilities.gpu_utils import get_gpu_manager
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

        # Initialize GPU manager for optimized evaluation
        self.gpu_manager = get_gpu_manager(self.config)
        self.device = self.gpu_manager.get_device()

        # Optimize batch size for similarity calculations
        self.batch_size = self.gpu_manager.get_optimal_batch_size(32)

        log_info(f"RetrievalEvaluator initialized with device: {self.device}", config=self.config)
        if self.gpu_manager.is_nvidia_gpu:
            log_info(f"GPU acceleration enabled for evaluation with batch size: {self.batch_size}", config=self.config)

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
            with open(dataset_path, encoding="utf-8") as f:
                dataset = json.load(f)

            log_info(f"Loaded {len(dataset)} items from evaluation dataset")

            # Apply limit and randomization
            if self.eval_config.randomize:
                random.shuffle(dataset)
                log_info("Dataset randomized")

            if self.eval_config.limit is not None:
                dataset = dataset[: self.eval_config.limit]
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
            # Generate embeddings for both texts with GPU acceleration
            if self.gpu_manager.is_nvidia_gpu:
                with torch.no_grad():
                    with self.gpu_manager.enable_mixed_precision():
                        embeddings = self.embedding_manager.encode_batch([text1, text2])
            else:
                embeddings = self.embedding_manager.encode_batch([text1, text2])

            embedding1, embedding2 = embeddings[0], embeddings[1]

            if not embedding1 or not embedding2:
                log_warning("One or both embeddings are empty", config=self.config)
                return 0.0

            # Reshape for sklearn
            embedding1 = np.array(embedding1).reshape(1, -1)
            embedding2 = np.array(embedding2).reshape(1, -1)

            # Calculate cosine similarity
            similarity = cosine_similarity(embedding1, embedding2)[0][0]

            return float(similarity)

        except Exception as e:
            log_error(f"Error calculating cosine similarity: {e}", config=self.config)
            return 0.0

    def calculate_batch_similarities(self, texts: list[str], ground_truth_context: str) -> list[float]:
        """
        Calculate cosine similarities between multiple texts and ground truth using batch processing.

        Args:
            texts: List of texts to compare against ground truth
            ground_truth_context: Ground truth text

        Returns:
            List of similarity scores
        """
        if not texts or not ground_truth_context:
            return [0.0] * len(texts)

        try:
            all_texts = texts + [ground_truth_context]

            start_time = time.time()

            if self.gpu_manager.is_nvidia_gpu:
                with torch.no_grad():
                    with self.gpu_manager.enable_mixed_precision():
                        all_embeddings = self.embedding_manager.encode_batch(all_texts)
            else:
                all_embeddings = self.embedding_manager.encode_batch(all_texts)

            encoding_time = time.time() - start_time

            # Split embeddings: retrieved results vs ground truth
            text_embeddings = all_embeddings[:-1]
            ground_truth_embedding = all_embeddings[-1]

            if not ground_truth_embedding:
                log_warning("Ground truth embedding is empty", config=self.config)
                return [0.0] * len(texts)

            similarities = []
            ground_truth_array = np.array(ground_truth_embedding).reshape(1, -1)

            for embedding in text_embeddings:
                if not embedding:
                    similarities.append(0.0)
                else:
                    text_array = np.array(embedding).reshape(1, -1)
                    similarity = cosine_similarity(text_array, ground_truth_array)[0][0]
                    similarities.append(float(similarity))

            device_info = f" on {self.device}" if self.gpu_manager.is_nvidia_gpu else " on CPU"
            log_info(
                f"Batch similarity calculation for {len(texts)} texts completed in {encoding_time:.2f}s{device_info}",
                config=self.config,
            )

            return similarities

        except Exception as e:
            log_error(f"Error in batch similarity calculation: {e}", config=self.config)
            return [0.0] * len(texts)

    def find_best_match(self, retrieved_results: list, ground_truth_context: str) -> tuple[float, str, int]:
        """
        Find the best matching retrieved result against ground truth using batch processing.

        Args:
            retrieved_results: List of retrieved search results
            ground_truth_context: Ground truth text to match against

        Returns:
            Tuple of (best_similarity_score, best_matching_content, result_index)
        """
        if not retrieved_results:
            return 0.0, "", -1

        # Extract content from results
        contents = [result.content for result in retrieved_results]

        # Use batch processing for better GPU utilization
        similarities = self.calculate_batch_similarities(contents, ground_truth_context)

        # Find the best match
        best_index = -1
        best_similarity = 0.0
        best_content = ""

        for i, similarity in enumerate(similarities):
            if similarity > best_similarity:
                best_similarity = similarity
                best_content = contents[i]
                best_index = i

        return best_similarity, best_content, best_index

    def calculate_classification_metrics(self, results: list[dict[str, Any]]) -> dict[str, float]:
        """
        Calculate classification metrics (accuracy, precision, recall, F1) for retrieval evaluation.

        Args:
            results: List of evaluation results from evaluate_single_item

        Returns:
            Dictionary containing classification metrics
        """
        # Filter out results with errors
        valid_results = [r for r in results if "error" not in r]

        if not valid_results:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "true_positives": 0, "false_positives": 0, "true_negatives": 0, "false_negatives": 0, "total_queries": 0}

        threshold = self.eval_config.similarity_threshold

        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for result in valid_results:
            similarity = result["similarity_score"]

            has_ground_truth = bool(result["ground_truth_context"].strip())

            if has_ground_truth:
                # Ground truth exists, so we expect to find a relevant document
                if similarity >= threshold:
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                # No ground truth (edge case), treat as negative case
                if similarity >= threshold:
                    false_positives += 1
                else:
                    true_negatives += 1

        total_queries = len(valid_results)

        accuracy = (true_positives + true_negatives) / total_queries if total_queries > 0 else 0.0

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "total_queries": total_queries,
        }

    def calculate_retrieval_metrics_at_k(self, results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        """
        Calculate retrieval metrics at different k values (Precision@K, Recall@K).

        Args:
            results: List of evaluation results from evaluate_single_item

        Returns:
            Dictionary containing metrics at different k values
        """
        valid_results = [r for r in results if "error" not in r]

        if not valid_results:
            return {}

        k_values = [1, 3, 5, 10]
        threshold = self.eval_config.similarity_threshold
        metrics_at_k = {}

        for k in k_values:
            if k > self.eval_config.k_results:
                continue

            relevant_found_at_k = 0
            total_relevant = 0
            total_retrieved_at_k = 0

            for result in valid_results:
                has_ground_truth = bool(result["ground_truth_context"].strip())

                if has_ground_truth:
                    total_relevant += 1

                    # Best match among top-k results
                    if result["result_index"] != -1 and result["result_index"] < k:
                        if result["similarity_score"] >= threshold:
                            relevant_found_at_k += 1

                    total_retrieved_at_k += min(k, result["total_results"])

            precision_at_k = relevant_found_at_k / (len(valid_results) * k) if len(valid_results) > 0 else 0.0
            recall_at_k = relevant_found_at_k / total_relevant if total_relevant > 0 else 0.0

            metrics_at_k[f"k_{k}"] = {"precision": precision_at_k, "recall": recall_at_k, "relevant_found": relevant_found_at_k, "total_relevant": total_relevant}

        return metrics_at_k

    def evaluate_single_item(self, item: dict[str, Any], item_number: int) -> dict[str, Any]:
        """
        Evaluate a single item from the dataset.

        Args:
            item: Evaluation item containing question and ground_truth_context
            item_number: Sequential item number for tracking

        Returns:
            Dictionary with evaluation results
        """
        question = item.get("question", "")
        metadata = item.get("metadata", {})
        ground_truth_context = metadata.get("ground_truth_context", "")

        if not question or not ground_truth_context:
            log_warning(f"Skipping invalid item {item_number}: missing question or ground_truth_context")
            return {
                "item": item_number,
                "question": question,
                "ground_truth_context": ground_truth_context,
                "retrieved_content": "",
                "similarity_score": 0.0,
                "result_index": -1,
                "total_results": 0,
                "error": "Missing question or ground_truth_context",
            }

        try:
            results = self.retriever.query(
                query_text=question,
                k=self.eval_config.k_results,
                search_type=self.eval_config.search_type,
                enable_reranking=self.eval_config.enable_reranking,
                enable_mmr=self.eval_config.enable_mmr,
                llm_analysis=not self.eval_config.disable_cultural_llm_analysis,
                formatted=False,  # Get raw results, not formatted string
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
                    "error": results,
                }

            best_similarity, best_content, best_index = self.find_best_match(results, ground_truth_context)

            return {
                "item": item_number,
                "question": question,
                "ground_truth_context": ground_truth_context,
                "retrieved_content": best_content,
                "similarity_score": best_similarity,
                "result_index": best_index,
                "total_results": len(results),
            }

        except Exception as e:
            log_error(f"Error evaluating item {item_number}: {e}")
            return {
                "item": item_number,
                "question": question,
                "ground_truth_context": ground_truth_context,
                "retrieved_content": "",
                "similarity_score": 0.0,
                "result_index": -1,
                "total_results": 0,
                "error": str(e),
            }

    def evaluate(self) -> dict[str, Any]:
        """
        Run full evaluation on the dataset with GPU acceleration.

        Returns:
            Dictionary containing evaluation results and statistics
        """
        start_time = time.time()

        log_info("Starting GPU-accelerated retrieval evaluation...", config=self.config)
        if self.gpu_manager.is_nvidia_gpu:
            log_info(f"Using GPU acceleration with batch size: {self.batch_size}", config=self.config)
        else:
            log_info("Using CPU processing", config=self.config)

        try:
            dataset = self.load_evaluation_dataset()
        except Exception as e:
            log_error(f"Failed to load evaluation dataset: {e}", config=self.config)
            return {"error": str(e)}

        if not dataset:
            log_warning("Empty evaluation dataset", config=self.config)
            return {"error": "Empty evaluation dataset"}

        # Evaluate each item with GPU memory management
        results = []
        similarities = []
        above_threshold_count = 0
        error_count = 0

        evaluation_start = time.time()
        log_info(f"Evaluating {len(dataset)} items with GPU acceleration...", config=self.config)

        for i, item in enumerate(dataset):
            item_number = i + 1
            item_question = item.get("question", "No question")[:50]
            log_info(f"Evaluating item {item_number}/{len(dataset)}: {item_question}...", config=self.config)

            result = self.evaluate_single_item(item, item_number)
            results.append(result)

            if "error" in result:
                error_count += 1
            else:
                similarities.append(result["similarity_score"])
                if result["similarity_score"] >= self.eval_config.similarity_threshold:
                    above_threshold_count += 1

            # Periodic GPU memory cleanup to prevent OOM
            if self.gpu_manager.is_nvidia_gpu and (i + 1) % 10 == 0:
                self.gpu_manager.clear_cache()
                log_info(f"GPU memory cleared after {i+1} evaluations", config=self.config)

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

        # Calculate performance metrics
        total_time = time.time() - start_time
        evaluation_time = time.time() - evaluation_start
        avg_time_per_item = evaluation_time / len(dataset) if dataset else 0.0

        log_info("Calculating classification metrics...", config=self.config)
        classification_metrics = self.calculate_classification_metrics(results)

        log_info("Calculating retrieval metrics at different k values...", config=self.config)
        retrieval_metrics_at_k = self.calculate_retrieval_metrics_at_k(results)

        evaluation_summary = {
            "dataset_info": {
                "total_items": len(dataset),
                "successful_evaluations": len(similarities),
                "errors": error_count,
                "success_rate": f"{success_rate:.1f}%",
            },
            "similarity_statistics": {
                "average": f"{avg_similarity:.4f}",
                "median": f"{median_similarity:.4f}",
                "std_deviation": f"{std_similarity:.4f}",
                "min": f"{min_similarity:.4f}",
                "max": f"{max_similarity:.4f}",
            },
            "classification_metrics": {
                "accuracy": f"{classification_metrics['accuracy']:.4f}",
                "precision": f"{classification_metrics['precision']:.4f}",
                "recall": f"{classification_metrics['recall']:.4f}",
                "f1_score": f"{classification_metrics['f1_score']:.4f}",
                "confusion_matrix": {
                    "true_positives": classification_metrics["true_positives"],
                    "false_positives": classification_metrics["false_positives"],
                    "true_negatives": classification_metrics["true_negatives"],
                    "false_negatives": classification_metrics["false_negatives"],
                    "total_queries": classification_metrics["total_queries"],
                },
            },
            "retrieval_metrics_at_k": retrieval_metrics_at_k,
            "threshold_analysis": {
                "threshold": self.eval_config.similarity_threshold,
                "above_threshold": above_threshold_count,
                "above_threshold_rate": f"{threshold_rate:.1f}%",
            },
            "configuration": {
                "search_type": self.eval_config.search_type,
                "k_results": self.eval_config.k_results,
                "enable_reranking": self.eval_config.enable_reranking,
                "enable_mmr": self.eval_config.enable_mmr,
                "cultural_llm_disabled": self.eval_config.disable_cultural_llm_analysis,
                "randomized": self.eval_config.randomize,
                "limit": self.eval_config.limit,
            },
            "performance_metrics": {
                "total_time_seconds": f"{total_time:.2f}",
                "evaluation_time_seconds": f"{evaluation_time:.2f}",
                "average_time_per_item_seconds": f"{avg_time_per_item:.2f}",
                "items_per_second": f"{len(dataset) / evaluation_time:.2f}" if evaluation_time > 0 else "0.00",
                "device_info": self.get_performance_info(),
            },
            "detailed_results": results,
        }

        if self.gpu_manager.is_nvidia_gpu:
            self.gpu_manager.clear_cache()

        # Log completion with performance metrics
        device_info = f" on {self.device}" if self.gpu_manager.is_nvidia_gpu else " on CPU"
        log_info(f"GPU-accelerated evaluation completed{device_info}!", config=self.config)
        log_info(f"Total time: {total_time:.2f}s, Evaluation time: {evaluation_time:.2f}s", config=self.config)
        log_info(
            f"Average time per item: {avg_time_per_item:.2f}s, Items per second: {len(dataset) / evaluation_time:.2f}",
            config=self.config,
        )
        log_info(f"Average similarity: {avg_similarity:.4f}", config=self.config)
        log_info(
            f"Items above threshold ({self.eval_config.similarity_threshold}): {above_threshold_count}/{len(similarities)} ({threshold_rate:.1f}%)",
            config=self.config,
        )

        # Log classification metrics
        log_info("Classification Metrics:", config=self.config)
        log_info(f"  Accuracy: {classification_metrics['accuracy']:.4f}", config=self.config)
        log_info(f"  Precision: {classification_metrics['precision']:.4f}", config=self.config)
        log_info(f"  Recall: {classification_metrics['recall']:.4f}", config=self.config)
        log_info(f"  F1-Score: {classification_metrics['f1_score']:.4f}", config=self.config)

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
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            log_info(f"Evaluation results saved to: {output_path}")
        except Exception as e:
            log_error(f"Failed to save results: {e}", config=self.config)

    def cleanup(self) -> None:
        """Cleanup GPU memory and resources."""
        try:
            if self.gpu_manager:
                self.gpu_manager.clear_cache()

            if hasattr(self.embedding_manager, "cleanup"):
                self.embedding_manager.cleanup()

            log_info("RetrievalEvaluator cleanup completed", config=self.config)

        except Exception as e:
            log_warning(f"Error during evaluator cleanup: {e}", config=self.config)

    def get_performance_info(self) -> dict[str, any]:
        """Get performance and device information."""
        gpu_info = self.gpu_manager.get_memory_info() if self.gpu_manager else {}
        embedding_info = self.embedding_manager.get_model_info() if hasattr(self.embedding_manager, "get_model_info") else {}

        return {
            "device": str(self.device),
            "batch_size": self.batch_size,
            "gpu_acceleration": self.gpu_manager.is_nvidia_gpu if self.gpu_manager else False,
            "gpu_info": gpu_info,
            "embedding_model_info": embedding_info,
        }


def run_evaluation(config: WiQASConfig | None = None, output_path: str | None = None) -> dict[str, Any]:
    """
    Convenience function to run GPU-accelerated evaluation with proper cleanup.

    Args:
        config: Optional WiQAS configuration
        output_path: Optional path to save results

    Returns:
        Evaluation results dictionary
    """
    evaluator = RetrievalEvaluator(config)

    try:
        results = evaluator.evaluate()

        if output_path:
            evaluator.save_results(results, output_path)

        return results

    finally:
        evaluator.cleanup()
