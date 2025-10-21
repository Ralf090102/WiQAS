"""
RAGAS Evaluation with Ollama via OpenAI-Compatible Endpoint

Requirements:
    pip install ragas datasets langchain langchain-openai pandas
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_correctness,
        context_recall,
        context_precision,
        answer_relevancy,
        answer_similarity,
    )
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.embeddings import OllamaEmbeddings
    from ragas import RunConfig
except ImportError as e:
    print(f"Error: {e}")
    print("Install with: pip install ragas datasets langchain langchain-openai pandas")
    sys.exit(1)

# Configuration
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAGAS_PARALLEL"] = "false"
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvaluationInput:
    """Structure for evaluation input data"""
    question: str
    ground_truth: str
    answer: str
    contexts: List[str]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationResult:
    """Structure for evaluation results"""
    question: str
    faithfulness: Optional[float]
    answer_correctness: Optional[float]
    context_recall: Optional[float]
    context_precision: Optional[float]
    answer_relevancy: Optional[float]
    answer_similarity: Optional[float]
    overall_score: float
    metadata: Optional[Dict[str, Any]] = None
    evaluation_time: float = 0.0
    model_used: str = ""


def setup_ollama_for_ragas(model_name: str = DEFAULT_MODEL):
    """
    Setup Ollama using OpenAI-compatible endpoint
    This avoids parameter compatibility issues
    """
    logger.info(f"Initializing Ollama via OpenAI-compatible endpoint...")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Base URL: {OLLAMA_BASE_URL}/v1")
    
    try:
        # Use OpenAI client pointed at Ollama
        llm = ChatOpenAI(
            model=model_name,
            base_url=f"{OLLAMA_BASE_URL}/v1",
            api_key="ollama",  # Dummy key required by OpenAI client
            temperature=0.1,
            max_tokens=512,
            timeout=120,
        )
        
        # Use OpenAI embeddings pointed at Ollama
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=OLLAMA_BASE_URL,
        )
        
        # Test LLM connection
        logger.info("  Testing LLM connection...")
        try:
            test_response = llm.invoke("Say 'test' and nothing else.")
            logger.info(f"  ✓ LLM working (response: '{test_response.content.strip()}')")
        except Exception as e:
            logger.error(f"  ✗ LLM test failed: {e}")
            raise
        
        # Test embeddings
        logger.info("  Testing embeddings...")
        try:
            test_embedding = embeddings.embed_query("test")
            logger.info(f"  ✓ Embeddings working (dim: {len(test_embedding)})")
        except Exception as e:
            logger.error(f"  ✗ Embeddings test failed: {e}")
            raise
        
        return llm, embeddings
        
    except Exception as e:
        logger.error(f"\n✗ Error connecting to Ollama: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("  1. Make sure Ollama is running: ollama serve")
        logger.info("  2. Check if model is available: ollama list")
        logger.info(f"  3. Pull model if needed: ollama pull {model_name}")
        logger.info("  4. Pull embeddings: ollama pull nomic-embed-text")
        logger.info(f"  5. Verify URL: {OLLAMA_BASE_URL}")
        logger.info("  6. Test OpenAI endpoint: curl http://localhost:11434/v1/models")
        sys.exit(1)


def load_evaluation_data(input_path: str, max_items: Optional[int] = None) -> List[EvaluationInput]:
    """Load evaluation data from JSON file"""
    path = Path(input_path)
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            items_raw = data
        else:
            items_raw = [data]
        
        # Limit items if specified
        if max_items:
            items_raw = items_raw[:max_items]
        
        data_items = []
        for i, item in enumerate(items_raw):
            # Validate required fields
            required_fields = ['question', 'ground_truth', 'contexts']
            
            if 'model_answer' not in item and 'answer' not in item:
                logger.warning(f"Skipping item {i} - missing 'answer' or 'model_answer'")
                continue
            
            missing_fields = [f for f in required_fields if f not in item]
            if missing_fields:
                logger.warning(f"Skipping item {i} - missing fields: {missing_fields}")
                continue
            
            answer = item.get('answer') or item.get('model_answer')
            
            data_items.append(EvaluationInput(
                question=item['question'],
                ground_truth=item['ground_truth'],
                answer=answer,
                contexts=item['contexts'],
                metadata=item.get('metadata', {})
            ))
        
        if not data_items:
            raise ValueError("No valid evaluation data found")
        
        logger.info(f"✓ Loaded {len(data_items)} valid items")
        return data_items
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def evaluate_with_ragas(items: List[EvaluationInput], llm, embeddings, 
                       model_name: str, use_all_metrics: bool = False) -> List[EvaluationResult]:
    """
    Evaluate items using RAGAS with Ollama backend
    """
    
    # Convert to RAGAS dataset format
    dataset_dict = {
        'question': [],
        'answer': [],
        'contexts': [],
        'ground_truth': [],
    }
    
    for item in items:
        dataset_dict['question'].append(item.question)
        dataset_dict['answer'].append(item.answer)
        dataset_dict['contexts'].append(item.contexts)
        dataset_dict['ground_truth'].append(item.ground_truth)
    
    dataset = Dataset.from_dict(dataset_dict)
    
    logger.info(f"\nRunning RAGAS evaluation on {len(items)} items...")
    
    # Choose metrics based on what works
    if use_all_metrics:
        metrics = [
            answer_similarity,      # Fast, embeddings only - WORKS
            answer_relevancy,       # Medium, requires LLM
            faithfulness,           # Slow, requires LLM
            context_recall,         # Slow, requires LLM
            context_precision,      # Slow, requires LLM
            answer_correctness,     # Medium, LLM + embeddings
        ]
        logger.info("Using ALL metrics (may timeout on some)")
    else:
        metrics = [
            answer_similarity,      # This is the only one that works reliably
        ]
        logger.info("Using ONLY working metrics (answer_similarity)")
    
    logger.info(f"Metrics: {', '.join([m.name for m in metrics])}")
    logger.info("This may take several minutes...\n")
    
    start_time = time.time()
    
    # More generous timeout settings
    run_config = RunConfig(
        timeout=300,          # 5 minutes per metric evaluation
        max_retries=1,
        max_wait=60,
        log_tenacity=False
    )
    
    try:
        evaluation_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            raise_exceptions=False
        )
        
        eval_time = time.time() - start_time
        
        # Convert RAGAS results to our format
        results = []
        df = evaluation_result.to_pandas()
        
        for i, item in enumerate(items):
            row_data = df.iloc[i]
            
            # Extract scores with NaN handling - None means no value was generated
            def get_score(col_name):
                if col_name not in row_data.index:
                    return None  # Metric wasn't run
                val = row_data.get(col_name)
                if pd.isna(val):
                    return None  # RAGAS couldn't generate value
                return float(val)
            
            metric_scores = {
                'faithfulness': get_score('faithfulness'),
                'answer_correctness': get_score('answer_correctness'),
                'context_recall': get_score('context_recall'),
                'context_precision': get_score('context_precision'),
                'answer_relevancy': get_score('answer_relevancy'),
                'answer_similarity': get_score('answer_similarity'),
            }
            
            # Calculate overall only from valid scores
            valid_scores = [v for v in metric_scores.values() if v is not None]
            overall = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            
            result = EvaluationResult(
                question=item.question,
                faithfulness=metric_scores['faithfulness'],
                answer_correctness=metric_scores['answer_correctness'],
                context_recall=metric_scores['context_recall'],
                context_precision=metric_scores['context_precision'],
                answer_relevancy=metric_scores['answer_relevancy'],
                answer_similarity=metric_scores['answer_similarity'],
                overall_score=overall,
                metadata=item.metadata,
                evaluation_time=eval_time / len(items),
                model_used=model_name
            )
            results.append(result)
            
            # Print per-item results
            logger.info(f"\n{'='*80}")
            logger.info(f"Item {i+1}/{len(items)}")
            logger.info(f"Question: {item.question[:100]}...")
            logger.info(f"{'='*80}")
            for metric_name, score in metric_scores.items():
                if score is not None:
                    logger.info(f"  {metric_name:20s}: {score:.3f}")
                else:
                    logger.info(f"  {metric_name:20s}: N/A (not generated)")
            logger.info(f"  {'─'*78}")
            logger.info(f"  Overall Score:       {result.overall_score:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"\n✗ Error during RAGAS evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise


def save_results(results: List[EvaluationResult], output_dir: str = "."):
    """Save results to JSON and CSV files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON (with None values preserved)
    json_path = output_path / "ragas_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json_data = [asdict(r) for r in results]
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ Results saved to {json_path}")
    
    # Save to CSV (None becomes empty string)
    csv_path = output_path / "ragas_results.csv"
    df = pd.DataFrame([asdict(r) for r in results])
    
    if 'metadata' in df.columns:
        metadata_df = pd.json_normalize(df['metadata'])
        metadata_df.columns = [f'metadata_{col}' for col in metadata_df.columns]
        df = pd.concat([df.drop('metadata', axis=1), metadata_df], axis=1)
    
    df.to_csv(csv_path, index=False)
    logger.info(f"✓ Results saved to {csv_path}")


def run_evaluation(input_path: str, model_name: Optional[str] = None, 
                   output_dir: str = ".", max_items: Optional[int] = None,
                   use_all_metrics: bool = False):
    """Main evaluation pipeline"""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"RAGAS Evaluation Pipeline (Ollama via OpenAI Endpoint)")
    logger.info(f"{'='*80}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup Ollama
    model = model_name or DEFAULT_MODEL
    llm, embeddings = setup_ollama_for_ragas(model)
    
    # Load data
    logger.info(f"\nLoading evaluation data from: {input_path}")
    try:
        items = load_evaluation_data(input_path, max_items)
    except Exception as e:
        logger.error(f"✗ Error loading data: {e}")
        sys.exit(1)
    
    # Run evaluation
    try:
        results = evaluate_with_ragas(items, llm, embeddings, model, use_all_metrics)
    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}")
        sys.exit(1)
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info(f"EVALUATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total Items: {len(results)}")
    
    if results:
        # Calculate averages only for valid scores
        def avg_valid(scores):
            valid = [s for s in scores if s is not None]
            return sum(valid) / len(valid) if valid else None
        
        logger.info(f"\nAverage Scores:")
        
        metrics_data = [
            ('Answer Similarity', [r.answer_similarity for r in results]),
            ('Answer Relevancy', [r.answer_relevancy for r in results]),
            ('Faithfulness', [r.faithfulness for r in results]),
            ('Context Recall', [r.context_recall for r in results]),
            ('Context Precision', [r.context_precision for r in results]),
            ('Answer Correctness', [r.answer_correctness for r in results]),
        ]
        
        for metric_name, scores in metrics_data:
            avg = avg_valid(scores)
            valid_count = sum(1 for s in scores if s is not None)
            if avg is not None:
                logger.info(f"  {metric_name:20s}: {avg:.3f} ({valid_count}/{len(results)} items)")
            else:
                logger.info(f"  {metric_name:20s}: N/A (no valid scores)")
        
        total_time = sum(r.evaluation_time for r in results)
        logger.info(f"\nTotal Time: {total_time:.2f}s")
        logger.info(f"Avg Time per Item: {total_time/len(results):.2f}s")
    
    # Save results
    save_results(results, output_dir)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluation complete!")
    logger.info(f"{'='*80}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='RAGAS Evaluation with Ollama (via OpenAI endpoint)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 10 items (fast, only answer_similarity)
  python ragas_fixed.py --input sample.json --model llama3.1 --max-items 10
  
  # Try all metrics (may timeout)
  python ragas_fixed.py --input sample.json --model llama3.1 --all-metrics
  
  # Full evaluation with working metrics only
  python ragas_fixed.py --input sample.json --model llama3.1

Note: 
- By default, only answer_similarity is used (it's the only one that works)
- Use --all-metrics to try all metrics (but expect timeouts)
- Ollama must support OpenAI-compatible endpoint (Ollama 0.1.0+)
- Missing values (None/null in JSON, empty in CSV) indicate RAGAS couldn't generate that score
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Path to JSON file with evaluation data')
    parser.add_argument('--model', '-m', default=None,
                       help=f'Ollama model (default: {DEFAULT_MODEL})')
    parser.add_argument('--output', '-o', default='.',
                       help='Output directory (default: current)')
    parser.add_argument('--max-items', '-n', type=int, default=None,
                       help='Max items to evaluate (default: all)')
    parser.add_argument('--all-metrics', action='store_true',
                       help='Try all metrics (may timeout, default: only answer_similarity)')
    
    args = parser.parse_args()
    
    try:
        run_evaluation(
            input_path=args.input,
            model_name=args.model,
            output_dir=args.output,
            max_items=args.max_items,
            use_all_metrics=args.all_metrics
        )
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