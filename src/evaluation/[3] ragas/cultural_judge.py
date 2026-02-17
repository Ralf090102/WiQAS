"""
Cultural LLM Judge - Multi-Model Evaluation System

A robust CLI tool for evaluating cultural answers using multiple LLM judges through Ollama.
Features outlier detection, statistical analysis, and comprehensive reporting.

Dependencies:
    pip install pandas numpy scipy langchain langchain-community

Usage:
    # Basic usage with default models
    python cultural_judge.py --input cultural_qa.json

    # Specify custom models
    python cultural_judge.py --input cultural_qa.json --models llama3.1,mistral,qwen2.5

    # Specify output directory
    python cultural_judge.py --input cultural_qa.json --output ./results/

    # Enable verbose logging
    python cultural_judge.py --input cultural_qa.json --verbose

Input JSON Format:
    [
        {
            "question": "What is the significance of...",
            "cultural_golden_answer": "The cultural significance is...",
            "model_answer": "According to tradition...",
            "metadata": {
                "culture": "Filipino",
                "category": "tradition"
            }
        }
    ]

Environment Variables:
    OLLAMA_BASE_URL: Ollama API URL (default: http://localhost:11434)
"""

import argparse
import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================================
# Configuration
# ============================================================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODELS = ["llama3.1", "mistral", "qwen2.5"]

# Outlier detection thresholds
OUTLIER_Z_THRESHOLD = 2.0  # Z-score threshold for outlier detection
MIN_MODELS_FOR_OUTLIER = 3  # Minimum models needed for outlier detection


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class CulturalQuestion:
    """Input data structure for cultural evaluation"""

    question: str
    cultural_golden_answer: str
    model_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeScore:
    """Score from a single judge model"""

    judge_model: str
    accuracy_score: float  # 0-10: How accurate is the answer
    cultural_sensitivity_score: float  # 0-10: Cultural appropriateness
    completeness_score: float  # 0-10: How complete is the answer
    overall_score: float  # Average of above scores
    reasoning: str
    evaluation_time: float


@dataclass
class EvaluationResult:
    """Complete evaluation result with outlier detection"""

    question: str
    cultural_golden_answer: str
    model_answer: str
    judge_scores: list[JudgeScore]
    mean_score: float
    median_score: float
    std_deviation: float
    outliers_detected: list[str]  # List of judge models that were outliers
    final_score: float  # Score after removing outliers
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class StatisticalSummary:
    """Statistical summary of all evaluations"""

    total_evaluations: int
    mean_final_score: float
    median_final_score: float
    std_deviation: float
    min_score: float
    max_score: float
    score_distribution: dict[str, int]  # Score ranges -> count
    outlier_statistics: dict[str, int]  # Model -> outlier count
    judge_agreement_rate: float  # How often judges agree (within 1 point)
    evaluation_metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# LLM Interface
# ============================================================================


class LLMJudge(ABC):
    """Abstract base class for LLM judges"""

    @abstractmethod
    def evaluate(
        self, question: str, cultural_golden_answer: str, model_answer: str
    ) -> JudgeScore:
        """Evaluate a cultural answer and return scores"""
        pass


class OllamaJudge(LLMJudge):
    """Ollama-based LLM judge implementation"""

    EVALUATION_PROMPT = """You are an expert cultural evaluator specializing in factual cultural knowledge. Your task is to evaluate how well a model's answer compares to the golden (reference) answer for a cultural factoid question.

Question: {question}

Golden Answer (Reference): {cultural_golden_answer}

Model Answer (To Evaluate): {model_answer}

Please evaluate the Model Answer on three dimensions using a 1-5 scale:

1. FACTUAL ACCURACY (1-5): How factually correct is the answer compared to the golden answer?
   - 5 (Excellent): All core facts are correct and precise
   - 4 (Good): Core facts correct, minor details missing or slightly imprecise
   - 3 (Fair): Main facts present but contains some errors or missing important details
   - 2 (Poor): Multiple factual errors or significant misunderstandings
   - 1 (Very Poor): Completely wrong or irrelevant information

2. CULTURAL APPROPRIATENESS (1-5): Is the answer culturally respectful and contextually appropriate?
   NOTE: For factual questions, this primarily checks for respectful terminology and tone.
   
   Be LENIENT for purely factual content - if the answer states facts neutrally without disrespect, score 4-5.
   
   - 5 (Excellent): Neutral, factual, respectful tone - appropriate for factual content
   - 4 (Good): Appropriate with very minor phrasing issues, still respectful
   - 3 (Fair): Some awkward phrasing or generalizations but not offensive
   - 2 (Poor): Problematic generalizations or inappropriate framing
   - 1 (Very Poor): Offensive language, harmful stereotypes, or disrespectful content

3. COMPLETENESS (1-5): How thoroughly does the answer cover the key information?
    
    Be LENIENT since this is only a factoid and is expected to be concise.

   - 5 (Excellent): Covers all critical elements from the golden answer
   - 4 (Good): Provides the main correct information; minor contextual or secondary details may be missing.
   - 3 (Fair): Answers the question but lacks several relevant or clarifying details.
   - 2 (Poor): Very incomplete, missing most key information
   - 1 (Very Poor): Does not adequately address the question

Provide your evaluation in the following JSON format with detailed reasoning:
{{
    "accuracy_score": <integer 1-5>,
    "cultural_sensitivity_score": <integer 1-5>,
    "completeness_score": <integer 1-5>,
    "reasoning": "<Provide detailed explanation covering:
                  1) What the model got right/wrong factually
                  2) Any cultural sensitivity concerns (or note if neutral/appropriate)
                  3) What information is missing or incomplete
                  4) Overall assessment>"
}}

IMPORTANT: 
- Respond ONLY with valid JSON. No additional text before or after the JSON.
- Use INTEGER scores 1-5 only (no decimals).
- Be thorough in your reasoning - explain specific strengths and weaknesses.
- For factual questions, cultural_sensitivity should usually be 4-5 unless there are clear issues."""

    def __init__(self, model_name: str, base_url: str = OLLAMA_BASE_URL):
        self.model_name = model_name
        self.base_url = base_url
        self.logger = logging.getLogger(f"OllamaJudge.{model_name}")

        try:
            from langchain_community.chat_models import ChatOllama

            self.llm = ChatOllama(
                model=model_name, base_url=base_url, temperature=0.3, format="json"
            )
            self._test_connection()
        except ImportError:
            self.logger.error("langchain-community not installed")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            raise

    def _test_connection(self):
        """Test connection to Ollama"""
        try:
            self.llm.invoke("test")
            self.logger.info(f"Connected to {self.model_name}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running and {self.model_name} is available. "
                f"Error: {e}"
            )

    def evaluate(
        self, question: str, cultural_golden_answer: str, model_answer: str
    ) -> JudgeScore:
        """Evaluate using Ollama LLM"""
        start_time = time.time()

        prompt = self.EVALUATION_PROMPT.format(
            question=question, cultural_golden_answer=cultural_golden_answer, model_answer=model_answer
        )

        try:
            self.logger.debug(f"Evaluating with {self.model_name}")
            response = self.llm.invoke(prompt)

            # Extract content from response
            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            # Parse JSON response
            result = json.loads(content)

            # Calculate overall score
            overall = (
                result["accuracy_score"]
                + result["cultural_sensitivity_score"]
                + result["completeness_score"]
            ) / 3

            eval_time = time.time() - start_time

            return JudgeScore(
                judge_model=self.model_name,
                accuracy_score=float(result["accuracy_score"]),
                cultural_sensitivity_score=float(result["cultural_sensitivity_score"]),
                completeness_score=float(result["completeness_score"]),
                overall_score=overall,
                reasoning=result["reasoning"],
                evaluation_time=eval_time,
            )

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON from {self.model_name}: {e}")
            raise
        except KeyError as e:
            self.logger.error(f"Missing key in response: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise


# ============================================================================
# Outlier Detection
# ============================================================================


class OutlierDetector:
    """Detect outliers in judge scores using statistical methods"""

    @staticmethod
    def detect_outliers(
        scores: list[JudgeScore], z_threshold: float = OUTLIER_Z_THRESHOLD
    ) -> tuple[list[str], list[JudgeScore]]:
        """
        Detect outliers using Z-score method.

        Returns:
            tuple: (outlier_judge_names, non_outlier_scores)
        """
        if len(scores) < MIN_MODELS_FOR_OUTLIER:
            return [], scores

        overall_scores = [s.overall_score for s in scores]
        mean = np.mean(overall_scores)
        std = np.std(overall_scores)

        # Avoid division by zero
        if std == 0:
            return [], scores

        outliers = []
        non_outliers = []

        for score in scores:
            z_score = abs((score.overall_score - mean) / std)
            if z_score > z_threshold:
                outliers.append(score.judge_model)
            else:
                non_outliers.append(score)

        return outliers, non_outliers

    @staticmethod
    def calculate_statistics(
        scores: list[JudgeScore],
    ) -> tuple[float, float, float]:
        """Calculate mean, median, std for scores"""
        if not scores:
            return 0.0, 0.0, 0.0

        overall_scores = [s.overall_score for s in scores]
        return (
            float(np.mean(overall_scores)),
            float(np.median(overall_scores)),
            float(np.std(overall_scores)),
        )


# ============================================================================
# Main Evaluator
# ============================================================================


class CulturalEvaluator:
    """Main evaluator orchestrating multiple judge models"""

    def __init__(self, judge_models: list[str], base_url: str = OLLAMA_BASE_URL):
        self.logger = logging.getLogger("CulturalEvaluator")
        self.judges = []

        self.logger.info(f"Initializing {len(judge_models)} judge models...")

        for model_name in judge_models:
            try:
                judge = OllamaJudge(model_name, base_url)
                self.judges.append(judge)
                self.logger.info(f"  ✓ {model_name}")
            except Exception as e:
                self.logger.error(f"  ✗ {model_name}: {e}")
                raise

        if not self.judges:
            raise ValueError("No judges successfully initialized")

    def evaluate_single(self, question_data: CulturalQuestion) -> EvaluationResult:
        """Evaluate a single question with all judges"""
        self.logger.info(f"Evaluating: {question_data.question[:60]}...")

        judge_scores = []

        # Collect scores from all judges
        for judge in self.judges:
            try:
                score = judge.evaluate(
                    question_data.question,
                    question_data.cultural_golden_answer,
                    question_data.model_answer,
                )
                judge_scores.append(score)
                self.logger.info(
                    f"  {judge.model_name}: {score.overall_score:.2f}/5 "
                    f"(Acc: {score.accuracy_score:.1f}, Cult: {score.cultural_sensitivity_score:.1f}, Comp: {score.completeness_score:.1f})"
                )
                self.logger.debug(f"  Reasoning: {score.reasoning}")
            except Exception as e:
                self.logger.error(f"  {judge.model_name} failed: {e}")
                continue

        if not judge_scores:
            raise RuntimeError("All judges failed to evaluate")

        # Calculate initial statistics
        mean, median, std = OutlierDetector.calculate_statistics(judge_scores)

        # Detect outliers
        outliers, non_outlier_scores = OutlierDetector.detect_outliers(judge_scores)

        # Calculate final score (without outliers)
        final_mean, _, _ = OutlierDetector.calculate_statistics(non_outlier_scores)

        if outliers:
            self.logger.warning(f"  Outliers detected: {', '.join(outliers)}")
            self.logger.info(
                f"  Final score (outliers removed): {final_mean:.2f}/5"
            )

        return EvaluationResult(
            question=question_data.question,
            cultural_golden_answer=question_data.cultural_golden_answer,
            model_answer=question_data.model_answer,
            judge_scores=judge_scores,
            mean_score=mean,
            median_score=median,
            std_deviation=std,
            outliers_detected=outliers,
            final_score=final_mean,
            metadata=question_data.metadata,
        )

    def evaluate_batch(
        self, questions: list[CulturalQuestion]
    ) -> list[EvaluationResult]:
        """Evaluate multiple questions"""
        results = []

        for i, question in enumerate(questions, 1):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Question {i}/{len(questions)}")
            self.logger.info(f"{'='*80}")

            try:
                result = self.evaluate_single(question)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to evaluate question {i}: {e}")
                continue

        return results


# ============================================================================
# Statistical Analysis
# ============================================================================


class StatisticalAnalyzer:
    """Analyze evaluation results and generate summary statistics"""

    @staticmethod
    def analyze(results: list[EvaluationResult]) -> StatisticalSummary:
        """Generate comprehensive statistical summary"""
        if not results:
            raise ValueError("No results to analyze")

        final_scores = [r.final_score for r in results]

        # Score distribution (0-3, 3-5, 5-7, 7-9, 9-10)
        score_dist = {
            "0-3 (Poor)": sum(1 for s in final_scores if s < 3),
            "3-5 (Fair)": sum(1 for s in final_scores if 3 <= s < 5),
            "5-7 (Good)": sum(1 for s in final_scores if 5 <= s < 7),
            "7-9 (Very Good)": sum(1 for s in final_scores if 7 <= s < 9),
            "9-10 (Excellent)": sum(1 for s in final_scores if 9 <= s <= 10),
        }

        # Outlier statistics by model
        outlier_counts = {}
        for result in results:
            for outlier_model in result.outliers_detected:
                outlier_counts[outlier_model] = (
                    outlier_counts.get(outlier_model, 0) + 1
                )

        # Judge agreement rate
        agreement_count = 0
        total_comparisons = 0

        for result in results:
            scores = [s.overall_score for s in result.judge_scores]
            if len(scores) >= 2:
                # Check if all scores are within 1 point of each other
                score_range = max(scores) - min(scores)
                if score_range <= 1.0:
                    agreement_count += 1
                total_comparisons += 1

        agreement_rate = (
            agreement_count / total_comparisons if total_comparisons > 0 else 0
        )

        return StatisticalSummary(
            total_evaluations=len(results),
            mean_final_score=float(np.mean(final_scores)),
            median_final_score=float(np.median(final_scores)),
            std_deviation=float(np.std(final_scores)),
            min_score=float(min(final_scores)),
            max_score=float(max(final_scores)),
            score_distribution=score_dist,
            outlier_statistics=outlier_counts,
            judge_agreement_rate=agreement_rate,
        )


# ============================================================================
# I/O Handlers
# ============================================================================


class DataLoader:
    """Load and validate input data"""

    @staticmethod
    def load_from_json(filepath: str) -> list[CulturalQuestion]:
        """Load cultural questions from JSON file.

        Accepts multiple input shapes:
          - { "question", "cultural_golden_answer", "model_answer", "metadata": {...} }
          - { "question", "ground_truth", "model_answer", "metadata": {"cultural_golden_answer": ...} }
          - Top-level or metadata can contain the golden answer.
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        questions: list[CulturalQuestion] = []
        for i, item in enumerate(data):
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}

            cultural_golden_answer = metadata.get("cultural_golden_answer")

            model_answer = item.get("model_answer") 

            missing = []
            if not item.get("question"):
                missing.append("question")
            if not cultural_golden_answer:
                missing.append("cultural_golden_answer (or ground_truth / metadata.cultural_golden_answer)")
            if not model_answer:
                missing.append("model_answer")

            if missing:
                logging.warning(f"Skipping item {i}: missing fields {missing}")
                continue

            questions.append(
                CulturalQuestion(
                    question=item["question"],
                    cultural_golden_answer=cultural_golden_answer,
                    model_answer=model_answer,
                    metadata=metadata,
                )
            )

        if not questions:
            raise ValueError("No valid questions found in input file")

        return questions



class ResultsExporter:
    """Export results to various formats"""

    @staticmethod
    def export_json(
        results: list[EvaluationResult], summary: StatisticalSummary, filepath: str
    ):
        """Export to JSON format"""
        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_evaluations": len(results),
            },
            "statistical_summary": asdict(summary),
            "results": [asdict(r) for r in results],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    @staticmethod
    def export_csv(results: list[EvaluationResult], filepath: str):
        """Export to CSV format with reasoning"""
        rows = []

        for result in results:
            base_row = {
                "question": result.question,
                "cultural_golden_answer": result.cultural_golden_answer,
                "model_answer": result.model_answer,
                "mean_score": result.mean_score,
                "median_score": result.median_score,
                "std_deviation": result.std_deviation,
                "final_score": result.final_score,
                "outliers_detected": ",".join(result.outliers_detected),
            }

            # Add individual judge scores with reasoning
            for score in result.judge_scores:
                base_row[f"{score.judge_model}_overall"] = score.overall_score
                base_row[f"{score.judge_model}_accuracy"] = score.accuracy_score
                base_row[
                    f"{score.judge_model}_cultural"
                ] = score.cultural_sensitivity_score
                base_row[f"{score.judge_model}_completeness"] = score.completeness_score
                base_row[f"{score.judge_model}_reasoning"] = score.reasoning
                base_row[
                    f"{score.judge_model}_is_outlier"
                ] = score.judge_model in result.outliers_detected

            # Add metadata
            for key, value in result.metadata.items():
                base_row[f"metadata_{key}"] = value

            rows.append(base_row)

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, encoding="utf-8")

    @staticmethod
    def export_detailed_report(results: list[EvaluationResult], filepath: str):
        """Export detailed report with all reasoning"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("CULTURAL LLM JUDGE - DETAILED EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            for i, result in enumerate(results, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"EVALUATION {i}/{len(results)}\n")
                f.write(f"{'='*80}\n\n")

                f.write(f"QUESTION:\n{result.question}\n\n")
                f.write(f"GOLDEN ANSWER:\n{result.cultural_golden_answer}\n\n")
                f.write(f"MODEL ANSWER:\n{result.model_answer}\n\n")

                f.write(f"{'─'*80}\n")
                f.write(f"JUDGE EVALUATIONS:\n")
                f.write(f"{'─'*80}\n\n")

                for score in result.judge_scores:
                    is_outlier = score.judge_model in result.outliers_detected
                    outlier_marker = " [OUTLIER]" if is_outlier else ""
                    
                    f.write(f"Judge: {score.judge_model}{outlier_marker}\n")
                    f.write(f"  Overall Score:           {score.overall_score:.2f}/5\n")
                    f.write(f"  - Factual Accuracy:      {score.accuracy_score:.2f}/5\n")
                    f.write(f"  - Cultural Sensitivity:  {score.cultural_sensitivity_score:.2f}/5\n")
                    f.write(f"  - Completeness:          {score.completeness_score:.2f}/5\n")
                    f.write(f"  \n  Reasoning:\n")
                    # Indent reasoning for readability
                    reasoning_lines = score.reasoning.split('\n')
                    for line in reasoning_lines:
                        f.write(f"  {line}\n")
                    f.write(f"\n")

                f.write(f"{'─'*80}\n")
                f.write(f"STATISTICAL SUMMARY:\n")
                f.write(f"{'─'*80}\n")
                f.write(f"Mean Score (all judges):     {result.mean_score:.2f}/5\n")
                f.write(f"Median Score:                {result.median_score:.2f}/5\n")
                f.write(f"Standard Deviation:          {result.std_deviation:.2f}\n")
                if result.outliers_detected:
                    f.write(f"Outliers Detected:           {', '.join(result.outliers_detected)}\n")
                    f.write(f"Final Score (outliers removed): {result.final_score:.2f}/5\n")
                else:
                    f.write(f"Outliers Detected:           None\n")
                    f.write(f"Final Score:                 {result.final_score:.2f}/5\n")
                f.write(f"\n")

    @staticmethod
    def export_summary_report(summary: StatisticalSummary, filepath: str):
        """Export human-readable summary report"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("CULTURAL LLM JUDGE - STATISTICAL SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total Evaluations: {summary.total_evaluations}\n\n")

            f.write("SCORE STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean Final Score:   {summary.mean_final_score:.2f}/5\n")
            f.write(f"Median Final Score: {summary.median_final_score:.2f}/5\n")
            f.write(f"Std Deviation:      {summary.std_deviation:.2f}\n")
            f.write(f"Min Score:          {summary.min_score:.2f}/5\n")
            f.write(f"Max Score:          {summary.max_score:.2f}/5\n\n")

            f.write("SCORE DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            for range_name, count in summary.score_distribution.items():
                pct = (count / summary.total_evaluations) * 100
                f.write(f"{range_name:20} {count:3} ({pct:5.1f}%)\n")

            f.write("\nOUTLIER STATISTICS\n")
            f.write("-" * 80 + "\n")
            if summary.outlier_statistics:
                for model, count in summary.outlier_statistics.items():
                    pct = (count / summary.total_evaluations) * 100
                    f.write(f"{model:20} {count:3} outliers ({pct:5.1f}%)\n")
            else:
                f.write("No outliers detected\n")

            f.write(f"\nJUDGE AGREEMENT RATE\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"Judges agree (within 1 point): {summary.judge_agreement_rate:.1%}\n"
            )


# ============================================================================
# CLI Interface
# ============================================================================


def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Cultural LLM Judge - Multi-Model Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cultural_judge.py --input cultural_qa.json
  python cultural_judge.py --input data.json --models llama3.1,mistral,qwen2.5
  python cultural_judge.py --input data.json --output ./results/ --verbose

Input Format:
  JSON file with array of objects containing:
  - question: The cultural question
  - cultural_golden_answer: The ideal/reference answer
  - model_answer: The answer to evaluate

Output Files:
  - results.json: Complete results with all scores and reasoning
  - results.csv: Tabular format with reasoning for analysis
  - detailed_report.txt: Full evaluation details with all judge reasoning
  - summary.txt: Human-readable statistical summary
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input JSON file",
    )

    parser.add_argument(
        "--models",
        "-m",
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated list of Ollama models (default: {','.join(DEFAULT_MODELS)})",
    )

    parser.add_argument(
        "--output",
        "-o",
        default=".",
        help="Output directory (default: current directory)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.verbose)
    logger = logging.getLogger("main")

    # Parse models
    models = [m.strip() for m in args.models.split(",")]

    logger.info("=" * 80)
    logger.info("CULTURAL LLM JUDGE - Multi-Model Evaluation System")
    logger.info("=" * 80)
    logger.info(f"Input: {args.input}")
    logger.info(f"Models: {', '.join(models)}")
    logger.info(f"Output: {args.output}")

    try:
        # Load data
        logger.info("\nLoading input data...")
        questions = DataLoader.load_from_json(args.input)
        logger.info(f"✓ Loaded {len(questions)} questions")

        # Initialize evaluator
        evaluator = CulturalEvaluator(models)

        # Run evaluation
        logger.info("\nStarting evaluation...")
        results = evaluator.evaluate_batch(questions)

        if not results:
            logger.error("No successful evaluations")
            sys.exit(1)

        # Analyze results
        logger.info("\nGenerating statistical summary...")
        summary = StatisticalAnalyzer.analyze(results)

        # Export results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("\nExporting results...")

        json_path = output_dir / "results.json"
        ResultsExporter.export_json(results, summary, str(json_path))
        logger.info(f"✓ JSON: {json_path}")

        csv_path = output_dir / "results.csv"
        ResultsExporter.export_csv(results, str(csv_path))
        logger.info(f"✓ CSV: {csv_path}")

        detailed_path = output_dir / "detailed_report.txt"
        ResultsExporter.export_detailed_report(results, str(detailed_path))
        logger.info(f"✓ Detailed Report: {detailed_path}")

        summary_path = output_dir / "summary.txt"
        ResultsExporter.export_summary_report(summary, str(summary_path))
        logger.info(f"✓ Summary: {summary_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        print(f"\nTotal Evaluations: {summary.total_evaluations}")
        print(f"Mean Score: {summary.mean_final_score:.2f}/5")
        print(f"Judge Agreement Rate: {summary.judge_agreement_rate:.1%}")
        print(
            f"\nOutliers Detected: {sum(summary.outlier_statistics.values())} total"
        )
        for model, count in summary.outlier_statistics.items():
            print(f"  - {model}: {count}")

        print("\n" + "=" * 80 + "\n")

    except KeyboardInterrupt:
        logger.info("\n\nEvaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\nError: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()