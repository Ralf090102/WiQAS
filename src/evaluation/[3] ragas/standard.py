import json
import re
import argparse
import os
from typing import List, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import statistics
import warnings

# Suppress some huggingface warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    from bert_score import score
except ImportError:
    print("Error: bert_score library not found. Please run 'pip install bert-score'")
    exit(1)

@dataclass
class QAMetrics:
    """Store metrics for a single Q&A pair"""
    bert_score: float  # Replaced F1
    f1_score: float    # Kept for reference (optional)
    is_grounded: bool
    cultural_relevance: float
    has_context: bool

@dataclass
class SystemMetrics:
    """Aggregate system-level metrics"""
    total_questions: int = 0
    avg_bert_score: float = 0.0
    avg_f1_score: float = 0.0
    grounded_answers: int = 0
    questions_with_context: int = 0
    cultural_relevance_scores: List[float] = field(default_factory=list)
    
    # Semantic similarity tiers (Adjusted for BERTScore distribution)
    # BERTScore usually ranges 0.85-1.0, so thresholds must be higher than F1
    excellent_answers: int = 0  # BERTScore > 0.93
    good_answers: int = 0       # BERTScore > 0.90
    acceptable_answers: int = 0 # BERTScore > 0.70
    poor_answers: int = 0       # BERTScore <= 0.70
    
    # Accuracy metrics
    excellent_rate: float = 0.0
    good_rate: float = 0.0
    acceptable_rate: float = 0.0
    overall_quality_rate: float = 0.0
    
    # Reliability
    grounding_rate: float = 0.0
    context_availability_rate: float = 0.0

class TextNormalizer:
    @staticmethod
    def normalize(text: str) -> str:
        if not text: return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        return TextNormalizer.normalize(text).split()

class BERTScoreCalculator:
    """Calculate Semantic Similarity using BERT embeddings"""
    
    def __init__(self, model_type='distilbert-base-uncased'):
        print(f"Loading BERTScore model ({model_type})... this may take a moment.")
        self.model_type = model_type
        # We don't initialize the model here directly to allow batch processing logic if needed,
        # but bert_score library handles caching automatically.

    def calculate_batch(self, predictions: List[str], references: List[str]) -> List[float]:
        """
        Calculate BERTScore for a batch of items (much faster than loop)
        Returns the F1 component of BERTScore (Precision/Recall harmonic mean)
        """
        if not predictions or not references:
            return []
            
        # P, R, F1 = score(cands, refs, model_type=...)
        # We verify inputs are not empty strings to avoid errors
        clean_preds = [p if p else " " for p in predictions]
        clean_refs = [r if r else " " for r in references]
        
        P, R, F1 = score(clean_preds, clean_refs, lang="en", model_type=self.model_type, verbose=False)
        return F1.tolist()

class F1Calculator:
    """Legacy lexical metric (kept for comparison)"""
    @staticmethod
    def calculate(prediction: str, ground_truth: str) -> float:
        pred_tokens = set(TextNormalizer.tokenize(prediction))
        truth_tokens = set(TextNormalizer.tokenize(ground_truth))
        if not pred_tokens and not truth_tokens: return 1.0
        if not pred_tokens or not truth_tokens: return 0.0
        common_tokens = pred_tokens & truth_tokens
        precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(common_tokens) / len(truth_tokens) if truth_tokens else 0.0
        if precision + recall == 0: return 0.0
        return 2 * (precision * recall) / (precision + recall)

class GroundingChecker:
    @staticmethod
    def is_grounded(answer: str, contexts: List[str], threshold: float = 0.5) -> bool:
        if not answer or not contexts: return False
        answer_tokens = set(TextNormalizer.tokenize(answer))
        
        no_info_phrases = ['no information', 'not provided', 'unknown', 'hindi malinaw']
        if any(p in answer.lower() for p in no_info_phrases):
            return len(contexts) == 0
            
        for context in contexts:
            context_tokens = set(TextNormalizer.tokenize(context))
            if not answer_tokens: continue
            overlap = len(answer_tokens & context_tokens)
            if (overlap / len(answer_tokens)) >= threshold:
                return True
        return False

class QAEvaluator:
    def __init__(self, grounding_threshold: float = 0.5):
        self.grounding_threshold = grounding_threshold
        self.bert_calc = BERTScoreCalculator()
        self.f1_calc = F1Calculator()
        self.grounding_checker = GroundingChecker()
        
    def evaluate_dataset(self, qa_data: List[Dict[str, Any]]) -> SystemMetrics:
        metrics = SystemMetrics()
        metrics.total_questions = len(qa_data)
        
        # Prepare batches for BERTScore (it's faster this way)
        all_preds = [item.get('model_answer', '') for item in qa_data]
        all_refs = [item.get('ground_truth', '') for item in qa_data]
        
        print(f"Computing semantic similarity for {len(qa_data)} items...")
        bert_scores = self.bert_calc.calculate_batch(all_preds, all_refs)
        
        f1_scores = []

        for idx, qa_item in enumerate(qa_data):
            model_answer = qa_item.get('model_answer', '')
            ground_truth = qa_item.get('ground_truth', '')
            contexts = qa_item.get('contexts', [])
            
            # Legacy F1 (Lexical)
            lexical_f1 = self.f1_calc.calculate(model_answer, ground_truth)
            f1_scores.append(lexical_f1)
            
            # BERTScore (Semantic)
            semantic_score = bert_scores[idx]
            
            # Grounding
            is_grounded = self.grounding_checker.is_grounded(
                model_answer, contexts, self.grounding_threshold
            )
            
            has_context = len(contexts) > 0
            
            # Tally scores
            # Note: BERTScore baselines are high. 0.70 is often "okay".
            if semantic_score > 0.93:
                metrics.excellent_answers += 1
            elif semantic_score > 0.90:
                metrics.good_answers += 1
            elif semantic_score > 0.70:
                metrics.acceptable_answers += 1
            else:
                metrics.poor_answers += 1
                
            if is_grounded: metrics.grounded_answers += 1
            if has_context: metrics.questions_with_context += 1
            
            if 'metadata' in qa_item and 'cultural_relevance' in qa_item['metadata']:
                metrics.cultural_relevance_scores.append(qa_item['metadata']['cultural_relevance'])

        # Aggregate
        if metrics.total_questions > 0:
            metrics.excellent_rate = metrics.excellent_answers / metrics.total_questions
            metrics.good_rate = metrics.good_answers / metrics.total_questions
            metrics.acceptable_rate = metrics.acceptable_answers / metrics.total_questions
            metrics.overall_quality_rate = (metrics.excellent_answers + metrics.good_answers + metrics.acceptable_answers) / metrics.total_questions
            
            metrics.avg_bert_score = statistics.mean(bert_scores)
            metrics.avg_f1_score = statistics.mean(f1_scores)
            
            metrics.grounding_rate = metrics.grounded_answers / metrics.total_questions
            metrics.context_availability_rate = metrics.questions_with_context / metrics.total_questions
            
        return metrics

class ReportGenerator:
    @staticmethod
    def generate_summary(metrics: SystemMetrics) -> str:
        report = []
        report.append("=" * 70)
        report.append("Q&A SYSTEM EDUCATIONAL SUITABILITY (SEMANTIC EVALUATION)")
        report.append("=" * 70)
        report.append(f"Total Questions Evaluated: {metrics.total_questions}")
        report.append("")
        
        report.append("-" * 70)
        report.append("SEMANTIC ACCURACY (BERTScore)")
        report.append("-" * 70)
        report.append(f"Average BERTScore:       {metrics.avg_bert_score:.4f}")
        report.append(f"(Legacy Lexical F1):     {metrics.avg_f1_score:.4f}")
        report.append("")
        report.append("Semantic Quality Distribution:")
        report.append(f"  Excellent (>0.93):     {metrics.excellent_rate:.2%} ({metrics.excellent_answers})")
        report.append(f"  Good (>0.90):          {metrics.good_rate:.2%} ({metrics.good_answers})")
        report.append(f"  Acceptable (>0.70):    {metrics.acceptable_rate:.2%} ({metrics.acceptable_answers})")
        report.append(f"  Poor (<=0.70):         {(metrics.poor_answers/metrics.total_questions):.2%} ({metrics.poor_answers})")
        report.append("")
        report.append(f"Overall Semantic Pass Rate: {metrics.overall_quality_rate:.2%}")
        report.append("")
        
        report.append("-" * 70)
        report.append("RELIABILITY (Grounding)")
        report.append("-" * 70)
        report.append(f"Grounding Rate:          {metrics.grounding_rate:.2%}")
        report.append(f"Context Availability:    {metrics.context_availability_rate:.2%}")
        
        return "\n".join(report)

    @staticmethod
    def save_reports(metrics: SystemMetrics, output_dir: str, input_filename: str):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(input_filename))[0]
        
        txt_path = os.path.join(output_dir, f"{base_name}_BERTSCORE_{timestamp}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(ReportGenerator.generate_summary(metrics))
            
        json_path = os.path.join(output_dir, f"{base_name}_BERTSCORE_{timestamp}.json")
        json_data = {
            "metrics": {
                "avg_bert_score": metrics.avg_bert_score,
                "avg_lexical_f1": metrics.avg_f1_score,
                "grounding_rate": metrics.grounding_rate
            }
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
            
        return txt_path, json_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Input JSON file')
    parser.add_argument('--output', '-o', default='.', help='Output directory')
    parser.add_argument('--max-items', '-n', type=int, default=None, help='Testing limit')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
        return
        
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if args.max_items: data = data[:args.max_items]
        
    evaluator = QAEvaluator()
    metrics = evaluator.evaluate_dataset(data)
    
    print("\n" + ReportGenerator.generate_summary(metrics))
    ReportGenerator.save_reports(metrics, args.output, args.input)

if __name__ == "__main__":
    main()