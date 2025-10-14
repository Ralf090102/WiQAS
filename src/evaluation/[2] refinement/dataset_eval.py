import json
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
import ollama
from pathlib import Path


class ValidationStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass
class ValidationResult:
    question: str
    ground_truth: str
    question_based_on_context: ValidationStatus
    question_reasoning: str
    answer_directly_answers: ValidationStatus
    answer_reasoning: str
    context_snippet: str
    metadata: Dict[str, Any]


class DatasetEvaluator:
    def __init__(self, model: str = "llama3.1"):
        self.model = model
        
    def check_question_validity(self, question: str, context: str) -> tuple[ValidationStatus, str]:
        """Check if the question can be answered based on the context."""
        prompt = f"""Given the following context, determine if the question can be answered based SOLELY on the information provided in the context.

Context:
{context}

Question:
{question}

Respond in the following format:
STATUS: [VALID/INVALID/PARTIAL]
REASONING: [Brief explanation of why the question is or isn't answerable from the context]

Be strict - the question should be answerable directly from the context without external knowledge."""

        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt}
            ])
            
            result = response['message']['content']
            
            # Parse response
            status_line = [line for line in result.split('\n') if line.startswith('STATUS:')]
            reasoning_line = [line for line in result.split('\n') if line.startswith('REASONING:')]
            
            if status_line and reasoning_line:
                status_text = status_line[0].replace('STATUS:', '').strip().lower()
                reasoning = reasoning_line[0].replace('REASONING:', '').strip()
                
                if 'valid' in status_text and 'invalid' not in status_text:
                    return ValidationStatus.VALID, reasoning
                elif 'partial' in status_text:
                    return ValidationStatus.PARTIAL, reasoning
                else:
                    return ValidationStatus.INVALID, reasoning
            
            return ValidationStatus.ERROR, "Could not parse response"
            
        except Exception as e:
            return ValidationStatus.ERROR, f"Error: {str(e)}"
    
    def check_answer_validity(self, question: str, answer: str, context: str) -> tuple[ValidationStatus, str]:
        """Check if the answer directly answers the question based on the context."""
        prompt = f"""Given the following context, question, and answer, determine if the answer DIRECTLY answers the question based on the context.

Context:
{context}

Question:
{question}

Provided Answer:
{answer}

Respond in the following format:
STATUS: [VALID/INVALID/PARTIAL]
REASONING: [Brief explanation of whether the answer correctly and directly answers the question based on the context]

Be strict - the answer should directly address the question using information from the context."""

        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt}
            ])
            
            result = response['message']['content']
            
            # Parse response
            status_line = [line for line in result.split('\n') if line.startswith('STATUS:')]
            reasoning_line = [line for line in result.split('\n') if line.startswith('REASONING:')]
            
            if status_line and reasoning_line:
                status_text = status_line[0].replace('STATUS:', '').strip().lower()
                reasoning = reasoning_line[0].replace('REASONING:', '').strip()
                
                if 'valid' in status_text and 'invalid' not in status_text:
                    return ValidationStatus.VALID, reasoning
                elif 'partial' in status_text:
                    return ValidationStatus.PARTIAL, reasoning
                else:
                    return ValidationStatus.INVALID, reasoning
            
            return ValidationStatus.ERROR, "Could not parse response"
            
        except Exception as e:
            return ValidationStatus.ERROR, f"Error: {str(e)}"


class DatasetValidator:
    def __init__(self, model: str = "llama3.1", verbose: bool = False):
        self.evaluator = DatasetEvaluator(model)
        self.verbose = verbose
        self.results: List[ValidationResult] = []
    
    def validate_item(self, item: Dict[str, Any]) -> ValidationResult:
        """Validate a single dataset item."""
        question = item.get('question', '')
        ground_truth = item.get('ground_truth', '')
        context = item.get('metadata', {}).get('ground_truth_context', '')
        metadata = item.get('metadata', {})
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Validating: {question[:50]}...")
        
        # Check if question is based on context
        q_status, q_reasoning = self.evaluator.check_question_validity(question, context)
        if self.verbose:
            print(f"Question validity: {q_status.value}")
        
        # Check if answer directly answers the question
        a_status, a_reasoning = self.evaluator.check_answer_validity(question, ground_truth, context)
        if self.verbose:
            print(f"Answer validity: {a_status.value}")
        
        return ValidationResult(
            question=question,
            ground_truth=ground_truth,
            question_based_on_context=q_status,
            question_reasoning=q_reasoning,
            answer_directly_answers=a_status,
            answer_reasoning=a_reasoning,
            context_snippet=context[:200] + "..." if len(context) > 200 else context,
            metadata=metadata
        )
    
    def validate_dataset(self, dataset: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Validate entire dataset."""
        print(f"Starting validation of {len(dataset)} items...")
        
        for i, item in enumerate(dataset, 1):
            print(f"\n[{i}/{len(dataset)}] Processing...")
            result = self.validate_item(item)
            self.results.append(result)
        
        return self.results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate summary report of validation results."""
        total = len(self.results)
        
        q_valid = sum(1 for r in self.results if r.question_based_on_context == ValidationStatus.VALID)
        q_invalid = sum(1 for r in self.results if r.question_based_on_context == ValidationStatus.INVALID)
        q_partial = sum(1 for r in self.results if r.question_based_on_context == ValidationStatus.PARTIAL)
        
        a_valid = sum(1 for r in self.results if r.answer_directly_answers == ValidationStatus.VALID)
        a_invalid = sum(1 for r in self.results if r.answer_directly_answers == ValidationStatus.INVALID)
        a_partial = sum(1 for r in self.results if r.answer_directly_answers == ValidationStatus.PARTIAL)
        
        return {
            "total_items": total,
            "question_validation": {
                "valid": q_valid,
                "invalid": q_invalid,
                "partial": q_partial,
                "valid_percentage": (q_valid / total * 100) if total > 0 else 0
            },
            "answer_validation": {
                "valid": a_valid,
                "invalid": a_invalid,
                "partial": a_partial,
                "valid_percentage": (a_valid / total * 100) if total > 0 else 0
            },
            "overall_quality": {
                "both_valid": sum(1 for r in self.results 
                                 if r.question_based_on_context == ValidationStatus.VALID 
                                 and r.answer_directly_answers == ValidationStatus.VALID),
                "both_valid_percentage": (sum(1 for r in self.results 
                                             if r.question_based_on_context == ValidationStatus.VALID 
                                             and r.answer_directly_answers == ValidationStatus.VALID) / total * 100) if total > 0 else 0
            }
        }
    
    def save_results(self, output_path: str):
        """Save validation results to JSON file."""
        output = {
            "report": self.generate_report(),
            "detailed_results": [
                {
                    "question": r.question,
                    "ground_truth": r.ground_truth,
                    "question_validation": {
                        "status": r.question_based_on_context.value,
                        "reasoning": r.question_reasoning
                    },
                    "answer_validation": {
                        "status": r.answer_directly_answers.value,
                        "reasoning": r.answer_reasoning
                    },
                    "context_snippet": r.context_snippet,
                    "metadata": r.metadata
                }
                for r in self.results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate evaluation dataset using Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validator.py dataset.json -o results.json
  python validator.py dataset.json -m llama3.1 -v
  python validator.py dataset.json --model mistral -o validation_results.json --verbose
        """
    )
    
    parser.add_argument('input_file', type=str, help='Path to input JSON dataset file')
    parser.add_argument('-o', '--output', type=str, default='validation_results.json',
                       help='Path to output JSON results file (default: validation_results.json)')
    parser.add_argument('-m', '--model', type=str, default='llama3.1',
                       help='Ollama model to use (default: llama3.1)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Load dataset
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} items from {args.input_file}")
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{args.input_file}': {e}")
        return
    
    # Validate dataset
    validator = DatasetValidator(model=args.model, verbose=args.verbose)
    validator.validate_dataset(dataset)
    
    # Generate and display report
    report = validator.generate_report()
    
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    print(f"\nTotal Items: {report['total_items']}")
    print(f"\nQuestion Validation:")
    print(f"  Valid: {report['question_validation']['valid']} ({report['question_validation']['valid_percentage']:.1f}%)")
    print(f"  Invalid: {report['question_validation']['invalid']}")
    print(f"  Partial: {report['question_validation']['partial']}")
    print(f"\nAnswer Validation:")
    print(f"  Valid: {report['answer_validation']['valid']} ({report['answer_validation']['valid_percentage']:.1f}%)")
    print(f"  Invalid: {report['answer_validation']['invalid']}")
    print(f"  Partial: {report['answer_validation']['partial']}")
    print(f"\nOverall Quality:")
    print(f"  Both Valid: {report['overall_quality']['both_valid']} ({report['overall_quality']['both_valid_percentage']:.1f}%)")
    
    # Save results
    validator.save_results(args.output)
    print(f"\nValidation complete!")


if __name__ == "__main__":
    main()