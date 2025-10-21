#!/usr/bin/env python3
"""
RAGAS Evaluation Pipeline with Ollama Backend

This script uses RAGAS library's built-in metrics with Ollama as the LLM provider.
Ollama is integrated through LangChain's ChatOllama wrapper for compatibility.

Dependencies:
    pip install ragas datasets langchain langchain-community pandas

Usage:
    # Using command line argument
    python ragas_eval.py --input sample.json --model llama3.1

    # Using environment variable
    export OLLAMA_MODEL=llama3.1
    python ragas_eval.py --input sample.json

    # Evaluate multiple files in a folder
    python ragas_eval.py --input ./samples/ --model llama3.1 --output ./results/

Environment Variables:
    OLLAMA_MODEL: Default model to use (default: llama3.1)
    OLLAMA_BASE_URL: Ollama API URL (default: http://localhost:11434)

Output:
    - Console: Per-item and summary statistics
    - ragas_results.json: Detailed results in JSON format
    - ragas_results.csv: Results in CSV format for analysis

How it works:
    - RAGAS handles ALL metric computations
    - Ollama is plugged in via LangChain's ChatOllama for LLM calls
    - Ollama embeddings are used via LangChain's OllamaEmbeddings
    - RAGAS calls Ollama internally for reasoning steps in metrics like faithfulness
"""

import logging,os
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")


# RAGAS imports
try:
    from datasets import Dataset
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings
    from ragas import RunConfig
except ImportError as e:
    print("Error: Required packages not installed.")
    print("Install with: pip install ragas datasets langchain langchain-community pandas")
    print(f"Details: {e}")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("ragas").setLevel(logging.DEBUG)
logging.getLogger("langchain").setLevel(logging.DEBUG)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAGAS_PARALLEL"] = "false"
os.environ["RAGAS_DEBUG"] = "true"

# ============================================================================
# Data Models
# ============================================================================


@dataclass
class EvaluationInput:
    """Structure for evaluation input data"""

    question: str
    ground_truth: str
    answer: str  # RAGAS uses 'answer' not 'model_answer'
    contexts: list[str]
    metadata: dict[str, Any] | None = None


@dataclass
class EvaluationResult:
    """Structure for evaluation results"""

    question: str
    faithfulness: float
    answer_correctness: float
    context_recall: float
    context_precision: float
    answer_relevancy: float
    answer_similarity: float
    overall_score: float
    metadata: dict[str, Any] | None = None
    evaluation_time: float = 0.0
    model_used: str = ""


# ============================================================================
# Ollama Integration with RAGAS
# ============================================================================


def setup_ollama_for_ragas(model_name: str = DEFAULT_MODEL):
    """
    Setup Ollama as LLM and embeddings provider for RAGAS.

    Returns:
        tuple: (llm, embeddings) ready for RAGAS evaluation
    """
    print("Initializing Ollama integration...")
    print(f"  Model: {model_name}")
    print(f"  Base URL: {OLLAMA_BASE_URL}")

    try:
        # Create ChatOllama instance for LLM calls
        # RAGAS will use this for reasoning steps in metrics
        llm = ChatOllama(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=0,  # Deterministic for evaluation
        )

        # Create OllamaEmbeddings for semantic similarity
        # RAGAS uses embeddings for metrics like answer_similarity
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=OLLAMA_BASE_URL,
        )

        # Test connection
        print("  Testing connection...", end=" ", flush=True)
        llm.invoke("test")
        print("✓")

        return llm, embeddings

    except Exception as e:
        print(f"\n✗ Error connecting to Ollama: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure Ollama is running: ollama serve")
        print("  2. Check if model is available: ollama list")
        print(f"  3. Pull model if needed: ollama pull {model_name}")
        print(f"  4. Verify URL: {OLLAMA_BASE_URL}")
        sys.exit(1)


# ============================================================================
# Input/Output Handling
# ============================================================================


def load_evaluation_data(input_path: str) -> list[EvaluationInput]:
    """Load evaluation data from JSON file or directory"""
    path = Path(input_path)
    data_items = []

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = list(path.glob("*.json"))
        if not files:
            raise ValueError(f"No JSON files found in directory: {input_path}")
    else:
        raise ValueError(f"Invalid path: {input_path}")

    for file_path in files:
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

                # Handle both single object and array
                if isinstance(data, list):
                    items = data
                else:
                    items = [data]

                for item in items:
                    # Validate required fields
                    # RAGAS expects: question, answer, contexts, ground_truth
                    required_fields = ["question", "ground_truth", "contexts"]

                    # Accept either 'model_answer' or 'answer'
                    if "model_answer" not in item and "answer" not in item:
                        print(f"Warning: Skipping item in {file_path} - missing 'answer' or 'model_answer'")
                        continue

                    missing_fields = [f for f in required_fields if f not in item]
                    if missing_fields:
                        print(f"Warning: Skipping item in {file_path} - missing fields: {missing_fields}")
                        continue

                    # Normalize field names
                    answer = item.get("answer") or item.get("model_answer")

                    data_items.append(
                        EvaluationInput(
                            question=item["question"],
                            ground_truth=item["ground_truth"],
                            answer=answer,
                            contexts=item["contexts"],
                            metadata=item.get("metadata", {}),
                        )
                    )

        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {file_path}: {e}")
            continue
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    if not data_items:
        raise ValueError("No valid evaluation data found")

    return data_items


def save_results(results: list[EvaluationResult], output_dir: str = "."):
    """Save results to JSON and CSV files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    json_path = output_path / "ragas_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json_data = [asdict(r) for r in results]
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to {json_path}")

    # Save to CSV
    csv_path = output_path / "ragas_results.csv"
    df = pd.DataFrame([asdict(r) for r in results])

    # Flatten metadata if present
    if "metadata" in df.columns:
        metadata_df = pd.json_normalize(df["metadata"])
        metadata_df.columns = [f"metadata_{col}" for col in metadata_df.columns]
        df = pd.concat([df.drop("metadata", axis=1), metadata_df], axis=1)

    df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to {csv_path}")


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================


def evaluate_with_ragas(items: list[EvaluationInput], llm, embeddings, model_name: str) -> list[EvaluationResult]:
    """
    Evaluate items using RAGAS with Ollama backend.

    This function:
    1. Converts items to RAGAS Dataset format
    2. Runs RAGAS evaluate() with all metrics
    3. RAGAS internally calls Ollama via LangChain for:
       - LLM reasoning (faithfulness, context_precision, etc.)
       - Embeddings (answer_similarity)
    """

    # Convert to RAGAS dataset format
    dataset_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for item in items:
        dataset_dict["question"].append(item.question)
        dataset_dict["answer"].append(item.answer)
        dataset_dict["contexts"].append(item.contexts)
        dataset_dict["ground_truth"].append(item.ground_truth)

    dataset = Dataset.from_dict(dataset_dict)

    print(f"\nRunning RAGAS evaluation on {len(items)} items...")
    print("RAGAS will call Ollama for:")
    print("  - LLM reasoning: faithfulness, context_precision, answer_relevancy")
    print("  - Embeddings: answer_similarity, answer_correctness")
    print("\nThis may take several minutes...\n")

    start_time = time.time()

    # Define metrics to evaluate
    # RAGAS handles all the computation internally
    metrics = [
        faithfulness,  # Measures if answer is grounded in contexts
        answer_correctness,  # F1-based correctness vs ground truth
        context_recall,  # How much of ground_truth is in contexts
        context_precision,  # Precision of retrieved contexts
        answer_relevancy,  # Relevance of answer to question
        answer_similarity,  # Semantic similarity to ground_truth
    ]

    run_config = RunConfig(timeout=120, log_tenacity=True)

    
    try:
        # Run RAGAS evaluation
        # RAGAS will use Ollama (via llm and embeddings) for all metric computations
        evaluation_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=run_config
        )

        eval_time = time.time() - start_time

        # Convert RAGAS results to our format
        results = []
        for i, item in enumerate(items):
            # RAGAS returns a DataFrame-like object
            row_data = evaluation_result.to_pandas().iloc[i]

            # Calculate overall score (average of all metrics)
            metric_scores = {
                "faithfulness": float(row_data.get("faithfulness", 0)),
                "answer_correctness": float(row_data.get("answer_correctness", 0)),
                "context_recall": float(row_data.get("context_recall", 0)),
                "context_precision": float(row_data.get("context_precision", 0)),
                "answer_relevancy": float(row_data.get("answer_relevancy", 0)),
                "answer_similarity": float(row_data.get("answer_similarity", 0)),
            }

            overall = sum(metric_scores.values()) / len(metric_scores)

            result = EvaluationResult(
                question=item.question,
                faithfulness=metric_scores["faithfulness"],
                answer_correctness=metric_scores["answer_correctness"],
                context_recall=metric_scores["context_recall"],
                context_precision=metric_scores["context_precision"],
                answer_relevancy=metric_scores["answer_relevancy"],
                answer_similarity=metric_scores["answer_similarity"],
                overall_score=overall,
                metadata=item.metadata,
                evaluation_time=eval_time / len(items),
                model_used=model_name,
            )
            results.append(result)

            # Print per-item results
            print(f"\n{'='*80}")
            print(f"Item {i+1}/{len(items)}")
            print(f"Question: {item.question}...")
            print(f"{'='*80}")
            print(f"  Faithfulness:        {result.faithfulness:.2f}")
            print(f"  Answer Correctness:  {result.answer_correctness:.2f}")
            print(f"  Context Recall:      {result.context_recall:.2f}")
            print(f"  Context Precision:   {result.context_precision:.2f}")
            print(f"  Answer Relevancy:    {result.answer_relevancy:.2f}")
            print(f"  Answer Similarity:   {result.answer_similarity:.2f}")
            print(f"  {'─'*78}")
            print(f"  Overall Score:       {result.overall_score:.2f}")

        return results

    except Exception as e:
        print(f"\n✗ Error during RAGAS evaluation: {e}")
        print("\nThis could be due to:")
        print("  - Ollama connection issues")
        print("  - Model context length limitations")
        print("  - Invalid input data format")
        raise


def run_evaluation(input_path: str, model_name: str | None = None, output_dir: str = ".") -> list[EvaluationResult]:
    """Main evaluation pipeline"""

    print(f"\n{'='*80}")
    print("RAGAS Evaluation Pipeline with Ollama")
    print(f"{'='*80}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup Ollama for RAGAS
    model = model_name or DEFAULT_MODEL
    llm, embeddings = setup_ollama_for_ragas(model)

    # Load data
    print(f"\nLoading evaluation data from: {input_path}")
    try:
        items = load_evaluation_data(input_path)
        print(f"✓ Loaded {len(items)} items")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)

    # Run RAGAS evaluation
    try:
        results = evaluate_with_ragas(items, llm, embeddings, model)
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        sys.exit(1)

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total Items: {len(results)}")

    if results:
        avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
        avg_correctness = sum(r.answer_correctness for r in results) / len(results)
        avg_recall = sum(r.context_recall for r in results) / len(results)
        avg_precision = sum(r.context_precision for r in results) / len(results)
        avg_relevancy = sum(r.answer_relevancy for r in results) / len(results)
        avg_similarity = sum(r.answer_similarity for r in results) / len(results)
        avg_overall = sum(r.overall_score for r in results) / len(results)
        total_time = sum(r.evaluation_time for r in results)

        print("\nAverage Scores:")
        print(f"  Faithfulness:        {avg_faithfulness:.2f}")
        print(f"  Answer Correctness:  {avg_correctness:.2f}")
        print(f"  Context Recall:      {avg_recall:.2f}")
        print(f"  Context Precision:   {avg_precision:.2f}")
        print(f"  Answer Relevancy:    {avg_relevancy:.2f}")
        print(f"  Answer Similarity:   {avg_similarity:.2f}")
        print(f"  {'─'*78}")
        print(f"  Overall Score:       {avg_overall:.2f}")
        print(f"\nTotal Time: {total_time:.2f}s")
        print(f"Avg Time per Item: {total_time/len(results):.2f}s")

    # Save results
    save_results(results, output_dir)

    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"{'='*80}\n")

    return results


# ============================================================================
# CLI Interface
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="RAGAS Evaluation Pipeline with Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ragas_eval.py --input sample.json --model llama3.1
  python ragas_eval.py --input ./samples/ --output ./results/
  OLLAMA_MODEL=mistral python ragas_eval.py --input sample.json

How Ollama is Used:
  - Ollama provides the LLM via LangChain's ChatOllama wrapper
  - RAGAS calls Ollama internally for all metric computations:
    * Faithfulness: LLM checks if answer claims are in contexts
    * Answer Correctness: F1 score + semantic similarity via embeddings
    * Context Recall: LLM checks if ground_truth is in contexts
    * Context Precision: LLM ranks context relevance
    * Answer Relevancy: LLM checks answer relevance to question
    * Answer Similarity: Cosine similarity using Ollama embeddings
  - All prompting and scoring logic is handled by RAGAS

Integration with Live RAG Pipeline:
  To integrate this evaluator into a live RAG pipeline:

  1. Import the evaluation functions:
     from ragas_eval import setup_ollama_for_ragas, evaluate_with_ragas
     from ragas_eval import EvaluationInput

  2. Setup Ollama once at startup:
     llm, embeddings = setup_ollama_for_ragas("llama3.1")

  3. After your RAG generates responses, create evaluation batch:
     eval_items = [
         EvaluationInput(
             question=query,
             ground_truth=expected,  # From test set
             answer=rag_response,
             contexts=retrieved_docs
         )
         for query, expected, rag_response, retrieved_docs in test_set
     ]

  4. Run evaluation:
     results = evaluate_with_ragas(eval_items, llm, embeddings, "llama3.1")

  5. Monitor metrics in production:
     - Track faithfulness to detect hallucinations
     - Monitor answer_relevancy for off-topic responses
     - Use context_precision to optimize retrieval
        """,
    )

    parser.add_argument("--input", "-i", required=True, help="Path to JSON file or directory containing evaluation data")

    parser.add_argument("--model", "-m", default=None, help=f"Ollama model to use (default: {DEFAULT_MODEL} or OLLAMA_MODEL env var)")

    parser.add_argument("--output", "-o", default=".", help="Output directory for results (default: current directory)")

    args = parser.parse_args()

    try:
        run_evaluation(input_path=args.input, model_name=args.model, output_dir=args.output)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
