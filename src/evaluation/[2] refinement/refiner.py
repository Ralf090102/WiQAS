import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import ollama


class RefinementAction(Enum):
    NONE = "none"
    REFINE_QUESTION = "refine_question"
    REFINE_ANSWER = "refine_answer"
    REFINE_BOTH = "refine_both"


@dataclass
class RefinementItem:
    question: str
    ground_truth: str
    action: RefinementAction
    judge1_question_status: str
    judge1_answer_status: str
    judge1_question_reasoning: str
    judge1_answer_reasoning: str
    judge2_question_status: str
    judge2_answer_status: str
    judge2_question_reasoning: str
    judge2_answer_reasoning: str
    context_snippet: str
    metadata: Dict[str, Any]
    refined_question: Optional[str] = None
    refined_answer: Optional[str] = None
    refinement_reasoning: Optional[str] = None


class LLMRefiner:
    def __init__(self, model: str = "llama3.2"):
        self.model = model
    
    def refine_question(self, question: str, context: str, judge_feedback: str) -> Tuple[str, str]:
        """Use LLM to refine a question based on context and judge feedback."""
        prompt = f"""You are refining a question for a QA dataset.  Keep the **same language** (Filipino, English, or mixed) as the original answer. Do NOT translate. The question must be answerable SOLELY from the provided context.

Context (THIS IS YOUR ONLY SOURCE OF TRUTH):
{context}

Original Question (NOT ANSWERABLE or PARTIALLY ANSWERABLE from context):
{question}

Why This Question Failed:
{judge_feedback}

CRITICAL REQUIREMENTS:
1. Read the context carefully and identify what information IS actually present
2. The question MUST ask about information that EXPLICITLY EXISTS in the context above
3. DO NOT ask about information that requires outside knowledge, inference, or is not stated in the context
4. If the original question asks about X but the context only discusses Y, change the question to ask about Y
5. Focus on concrete facts, dates, names, events, or statements that are directly written in the 
6. Questions should start with  why, who, when, what, where (factoid)
7. Avoid generic lead-ins like "According to the text," "Based on the passage," etc.
8. If the question is in Filipino, then it must stay in Filipino. Keep the same original language.


Example of GOOD refinement:
- Context mentions: "The document was written in 900 CE"
- Bad question: "Why was the document important?" (requires interpretation)
- Good question: "When was the document written?" (directly stated)

Example of addressing judge feedback:
- Feedback: "The context doesn't explain the significance, only describes the event"
- Solution: Change question from asking "why" to asking "what happened" or "when did it occur"

Now refine the question to be DIRECTLY and COMPLETELY answerable from the context alone.

Respond in the following format:
REFINED_QUESTION: [Your refined question here]
REASONING: [Explain specifically how this question is now answerable from the context, citing what information you're asking about]"""


        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt}
            ])
            
            result = response['message']['content']
            
            # Parse response
            question_line = [line for line in result.split('\n') if line.startswith('REFINED_QUESTION:')]
            reasoning_line = [line for line in result.split('\n') if line.startswith('REASONING:')]
            
            if question_line:
                refined_q = question_line[0].replace('REFINED_QUESTION:', '').strip()
                reasoning = reasoning_line[0].replace('REASONING:', '').strip() if reasoning_line else "No reasoning provided"
                return refined_q, reasoning
            
            return question, "Error: Could not parse LLM response"
            
        except Exception as e:
            return question, f"Error during refinement: {str(e)}"
    
    def refine_answer(self, question: str, answer: str, context: str, judge_feedback: str) -> Tuple[str, str]:
        """Use LLM to refine an answer based on question, context, and judge feedback."""
        prompt = f"""You are refining an answer for a QA dataset.  Keep the **same language** (Filipino, English, or mixed) as the original answer. Do NOT translate.  The answer must directly answer the question using ONLY information from the provided context.

Context (YOUR ONLY SOURCE):
{context}

Question:
{question}

Original Answer (INADEQUATE):
{answer}

Why This Answer Failed:
{judge_feedback}

CRITICAL REQUIREMENTS:
1. Your answer MUST directly address what the question is asking
2. Use ONLY information explicitly stated in the context - no interpretation, no outside knowledge
3. If the question asks "how", explain the process/method mentioned in context
4. If the question asks "why", state the reason/cause given in context
5. If the question asks "what", describe exactly what is stated in context
6. If the question asks "when/where", provide the specific time/location from context
7. Quote or paraphrase the relevant part of the context that answers the question
8. Avoid generic lead-ins like "According to the text," "Based on the passage," etc.
9. If the question is in Filipino, then it must stay in Filipino. Keep the same original language.

Example of GOOD refinement:
- Question: "When was the document created?"
- Context: "The Laguna Copperplate Inscription dates to 900 CE"
- Bad answer: "It was created a long time ago" (too vague)
- Good answer: "The document was created in 900 CE" (specific, from context)

Your answer should be:
- Complete (fully answers the question)
- Specific (uses exact details from context)
- Direct (no unnecessary information)
- Grounded (only uses context information)

Respond in the following format:
REFINED_ANSWER: [Your refined answer here]
REASONING: [Explain how this answer now directly addresses the question using context information]"""


        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt}
            ])
            
            result = response['message']['content']
            
            # Parse response
            answer_line = [line for line in result.split('\n') if line.startswith('REFINED_ANSWER:')]
            reasoning_line = [line for line in result.split('\n') if line.startswith('REASONING:')]
            
            if answer_line:
                refined_a = answer_line[0].replace('REFINED_ANSWER:', '').strip()
                reasoning = reasoning_line[0].replace('REASONING:', '').strip() if reasoning_line else "No reasoning provided"
                return refined_a, reasoning
            
            return answer, "Error: Could not parse LLM response"
            
        except Exception as e:
            return answer, f"Error during refinement: {str(e)}"
    
    def refine_both(self, question: str, answer: str, context: str, 
                   q_feedback: str, a_feedback: str) -> Tuple[str, str, str]:
        """Use LLM to refine both question and answer."""
        prompt = f"""You are refining a question-answer pair for a QA dataset. Keep the **same language** (Filipino, English, or mixed) as the original answer. Do NOT translate. Both must be based SOLELY on the provided context.

Context:
{context}

Original Question:
{question}

Original Answer:
{answer}

Question Feedback:
{q_feedback}

Answer Feedback:
{a_feedback}

Task: Refine both the question and answer so that:
1. The question is clearly answerable from the context
2. The answer directly addresses the refined question
3. Both are based solely on information in the context
4. Both use natural language
5. Avoid generic lead-ins like "According to the text," "Based on the passage," etc.
6. concise but complete.


Respond in the following format:
REFINED_QUESTION: [Your refined question here]
REFINED_ANSWER: [Your refined answer here]
REASONING: [Brief explanation of what you changed and why]"""

        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt}
            ])
            
            result = response['message']['content']
            
            # Parse response
            question_line = [line for line in result.split('\n') if line.startswith('REFINED_QUESTION:')]
            answer_line = [line for line in result.split('\n') if line.startswith('REFINED_ANSWER:')]
            reasoning_line = [line for line in result.split('\n') if line.startswith('REASONING:')]
            
            if question_line and answer_line:
                refined_q = question_line[0].replace('REFINED_QUESTION:', '').strip()
                refined_a = answer_line[0].replace('REFINED_ANSWER:', '').strip()
                reasoning = reasoning_line[0].replace('REASONING:', '').strip() if reasoning_line else "No reasoning provided"
                return refined_q, refined_a, reasoning
            
            return question, answer, "Error: Could not parse LLM response"
            
        except Exception as e:
            return question, answer, f"Error during refinement: {str(e)}"


class DatasetRefiner:
    def __init__(self, model: str = "llama3.2", verbose: bool = False, interactive: bool = False):
        self.llm_refiner = LLMRefiner(model)
        self.verbose = verbose
        self.interactive = interactive
        self.refinement_items: List[RefinementItem] = []
        
    def is_valid(self, status: str) -> bool:
        """Check if a status is considered valid."""
        return status.lower() == "valid"
    
    def determine_refinement_action(
        self, 
        j1_q_status: str, 
        j1_a_status: str,
        j2_q_status: str,
        j2_a_status: str
    ) -> RefinementAction:
        """Determine what refinement action is needed based on two judges' assessments."""
        j1_q_valid = self.is_valid(j1_q_status)
        j1_a_valid = self.is_valid(j1_a_status)
        j2_q_valid = self.is_valid(j2_q_status)
        j2_a_valid = self.is_valid(j2_a_status)
        
        # Both judges agree both are valid - no refinement needed
        if j1_q_valid and j1_a_valid and j2_q_valid and j2_a_valid:
            return RefinementAction.NONE
        
        # Determine what needs refinement based on disagreement
        question_needs_refinement = not (j1_q_valid and j2_q_valid)
        answer_needs_refinement = not (j1_a_valid and j2_a_valid)
        
        if question_needs_refinement and answer_needs_refinement:
            return RefinementAction.REFINE_BOTH
        elif question_needs_refinement:
            return RefinementAction.REFINE_QUESTION
        elif answer_needs_refinement:
            return RefinementAction.REFINE_ANSWER
        else:
            return RefinementAction.NONE
    
    def get_user_approval(self, original: str, refined: str, item_type: str) -> str:
        """Interactive CLI to approve or edit refinements."""
        print(f"\n{'='*60}")
        print(f"Original {item_type}:")
        print(f"  {original}")
        print(f"\nRefined {item_type}:")
        print(f"  {refined}")
        print(f"\n{'='*60}")
        
        while True:
            choice = input("\nOptions: [a]ccept / [e]dit / [k]eep original: ").lower().strip()
            
            if choice == 'a':
                return refined
            elif choice == 'k':
                return original
            elif choice == 'e':
                print(f"\nEnter your edited {item_type} (press Enter twice to finish):")
                lines = []
                while True:
                    line = input()
                    if line == "" and lines and lines[-1] == "":
                        break
                    lines.append(line)
                edited = " ".join(lines[:-1]).strip()
                return edited if edited else refined
            else:
                print("Invalid choice. Please enter 'a', 'e', or 'k'.")
    
    def refine_item(self, item: RefinementItem, index: int, total: int) -> RefinementItem:
        """Refine a single item based on its action."""
        if item.action == RefinementAction.NONE:
            return item
        
        print(f"\n{'='*60}")
        print(f"[{index}/{total}] Refining: {item.action.value}")
        print(f"{'='*60}")
        print(f"Question: {item.question[:80]}...")
        
        context = item.metadata.get('ground_truth_context', item.context_snippet)
        
        # Combine judge feedback
        q_feedback = f"Judge 1 ({item.judge1_question_status}): {item.judge1_question_reasoning}\nJudge 2 ({item.judge2_question_status}): {item.judge2_question_reasoning}"
        a_feedback = f"Judge 1 ({item.judge1_answer_status}): {item.judge1_answer_reasoning}\nJudge 2 ({item.judge2_answer_status}): {item.judge2_answer_reasoning}"
        
        print(q_feedback)
        print(a_feedback)
                
        if item.action == RefinementAction.REFINE_QUESTION:
            print("Refining question...")
            # refined_q, reasoning = self.llm_refiner.refine_question(
            #     item.question, context, q_feedback
            # )
  
            # if self.interactive:
            #     refined_q = self.get_user_approval(item.question, refined_q, "Question")
            
            # item.refined_question = refined_q
            # item.refined_answer = item.ground_truth
            # item.refinement_reasoning = reasoning
            
            # if self.verbose:
            #     print(f"  Original: {item.question}")
            #     print(f"  Refined:  {refined_q}")
        
        elif item.action == RefinementAction.REFINE_ANSWER or item.action == RefinementAction.REFINE_BOTH:
            print("Refining answer...")
            refined_a, reasoning = self.llm_refiner.refine_answer(
                item.question, item.ground_truth, context, a_feedback
            )
            
            if self.interactive:
                refined_a = self.get_user_approval(item.ground_truth, refined_a, "Answer")
            
            item.refined_question = item.question
            item.refined_answer = refined_a
            item.refinement_reasoning = reasoning
            
            if self.verbose:
                print(f"  Original: {item.ground_truth}")
                print(f"  Refined:  {refined_a}")
        
        elif item.action == RefinementAction.REFINE_BOTH:
            print("Refining both question and answer...")
            # refined_q, refined_a, reasoning = self.llm_refiner.refine_both(
            #     item.question, item.ground_truth, context, q_feedback, a_feedback
            # )
            
            # if self.interactive:
            #     refined_q = self.get_user_approval(item.question, refined_q, "Question")
            #     refined_a = self.get_user_approval(item.ground_truth, refined_a, "Answer")
            
            # item.refined_question = refined_q
            # item.refined_answer = refined_a
            # item.refinement_reasoning = reasoning
            
            # if self.verbose:
            #     print(f"  Original Q: {item.question}")
            #     print(f"  Refined Q:  {refined_q}")
            #     print(f"  Original A: {item.ground_truth}")
            #     print(f"  Refined A:  {refined_a}")
        
        return item
    
    def analyze_datasets(
        self, 
        judge1_data: List[Dict[str, Any]], 
        judge2_data: List[Dict[str, Any]]
    ) -> List[RefinementItem]:
        """Analyze two judge datasets and determine refinement needs."""
        if len(judge1_data) != len(judge2_data):
            print(f"Warning: Dataset sizes don't match (Judge1: {len(judge1_data)}, Judge2: {len(judge2_data)})")
            print("Processing only matching items...")
        
        min_len = min(len(judge1_data), len(judge2_data))
        
        for i in range(min_len):
            j1_item = judge1_data[i]
            j2_item = judge2_data[i]
            
            # Extract status values
            j1_q_status = j1_item.get('question_validation', {}).get('status', 'error')
            j1_a_status = j1_item.get('answer_validation', {}).get('status', 'error')
            j2_q_status = j2_item.get('question_validation', {}).get('status', 'error')
            j2_a_status = j2_item.get('answer_validation', {}).get('status', 'error')
            
            # Determine action
            action = self.determine_refinement_action(
                j1_q_status, j1_a_status, j2_q_status, j2_a_status
            )
            
            refinement_item = RefinementItem(
                question=j1_item.get('question', ''),
                ground_truth=j1_item.get('ground_truth', ''),
                action=action,
                judge1_question_status=j1_q_status,
                judge1_answer_status=j1_a_status,
                judge1_question_reasoning=j1_item.get('question_validation', {}).get('reasoning', ''),
                judge1_answer_reasoning=j1_item.get('answer_validation', {}).get('reasoning', ''),
                judge2_question_status=j2_q_status,
                judge2_answer_status=j2_a_status,
                judge2_question_reasoning=j2_item.get('question_validation', {}).get('reasoning', ''),
                judge2_answer_reasoning=j2_item.get('answer_validation', {}).get('reasoning', ''),
                context_snippet=j1_item.get('context_snippet', ''),
                metadata=j1_item.get('metadata', {})
            )
            
            self.refinement_items.append(refinement_item)
        
        return self.refinement_items
    
    def refine_all(self):
        """Refine all items that need refinement."""
        items_to_refine = [item for item in self.refinement_items if item.action != RefinementAction.NONE]
        total_to_refine = len(items_to_refine)
        
        if total_to_refine == 0:
            print("\nNo items need refinement!")
            return
        
        print(f"\nStarting refinement of {total_to_refine} items...")
        
        refined_count = 0
        for i, item in enumerate(self.refinement_items, 1):
            if item.action != RefinementAction.NONE:
                refined_count += 1
                self.refine_item(item, refined_count, total_to_refine)
        
        print(f"\n{'='*60}")
        print(f"Refinement complete! Refined {refined_count} items.")
        print(f"{'='*60}")
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        total = len(self.refinement_items)
        
        refine_question = sum(1 for item in self.refinement_items 
                             if item.action == RefinementAction.REFINE_QUESTION)
        refine_answer = sum(1 for item in self.refinement_items 
                           if item.action == RefinementAction.REFINE_ANSWER)
        refine_both = sum(1 for item in self.refinement_items 
                         if item.action == RefinementAction.REFINE_BOTH)
        no_refine = sum(1 for item in self.refinement_items 
                       if item.action == RefinementAction.NONE)
        
        total_needing_refinement = refine_question + refine_answer + refine_both
        
        return {
            "total_items": total,
            "no_refinement_needed": no_refine,
            "total_needing_refinement": total_needing_refinement,
            "refine_question_only": refine_question,
            "refine_answer_only": refine_answer,
            "refine_both": refine_both,
            "refinement_percentage": (total_needing_refinement / total * 100) if total > 0 else 0
        }
    
    def export_refined_dataset(self, output_path: str):
        """Export refined dataset in standard QA format."""
        refined_dataset = []
        
        for item in self.refinement_items:
            refined_item = {
                "question": item.refined_question if item.refined_question else item.question,
                "ground_truth": item.refined_answer if item.refined_answer else item.ground_truth,
                "metadata": item.metadata
            }
            
            # Add refinement info to metadata
            if item.action != RefinementAction.NONE:
                refined_item["metadata"]["refinement_info"] = {
                    "action": item.action.value,
                    "original_question": item.question,
                    "original_answer": item.ground_truth,
                    "refinement_reasoning": item.refinement_reasoning,
                    "judge1_feedback": {
                        "question_status": item.judge1_question_status,
                        "answer_status": item.judge1_answer_status
                    },
                    "judge2_feedback": {
                        "question_status": item.judge2_question_status,
                        "answer_status": item.judge2_answer_status
                    }
                }
            
            refined_dataset.append(refined_item)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(refined_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"\nRefined dataset exported to: {output_path}")
    
    def save_detailed_report(self, output_path: str):
        """Save detailed refinement report."""
        summary = self.generate_summary()
        
        # Group items by action
        items_by_action = {
            "refine_question": [],
            "refine_answer": [],
            "refine_both": [],
            "no_refinement": []
        }
        
        for item in self.refinement_items:
            item_dict = {
                "original_question": item.question,
                "original_answer": item.ground_truth,
                "refined_question": item.refined_question,
                "refined_answer": item.refined_answer,
                "refinement_reasoning": item.refinement_reasoning,
                "judge1_assessments": {
                    "question_status": item.judge1_question_status,
                    "question_reasoning": item.judge1_question_reasoning,
                    "answer_status": item.judge1_answer_status,
                    "answer_reasoning": item.judge1_answer_reasoning
                },
                "judge2_assessments": {
                    "question_status": item.judge2_question_status,
                    "question_reasoning": item.judge2_question_reasoning,
                    "answer_status": item.judge2_answer_status,
                    "answer_reasoning": item.judge2_answer_reasoning
                },
                "context_snippet": item.context_snippet,
                "metadata": item.metadata
            }
            
            if item.action == RefinementAction.REFINE_QUESTION:
                items_by_action["refine_question"].append(item_dict)
            elif item.action == RefinementAction.REFINE_ANSWER:
                items_by_action["refine_answer"].append(item_dict)
            elif item.action == RefinementAction.REFINE_BOTH:
                items_by_action["refine_both"].append(item_dict)
            else:
                items_by_action["no_refinement"].append(item_dict)
        
        output = {
            "summary": summary,
            "items_by_action": items_by_action
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze two judge evaluations, refine using LLM, and export refined dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic refinement with default model
  python refiner.py judge1.json judge2.json -o refined_dataset.json
  
  # Use different model with verbose output
  python refiner.py judge1.json judge2.json -m mistral -v
  
  # Interactive mode to approve each refinement
  python refiner.py judge1.json judge2.json -i
  
  # Generate detailed report
  python refiner.py judge1.json judge2.json -o refined.json -r report.json
  
  # Full workflow with all options
  python refiner.py judge1.json judge2.json -m llama3.1:latest -o refined.json -r report.json -v -i
        """
    )
    
    # Positional arguments for both judge files
    parser.add_argument('judge1_file', 
                       type=str, 
                       help='Path to first judge evaluation JSON file')
    parser.add_argument('judge2_file', 
                       type=str, 
                       help='Path to second judge evaluation JSON file')
    
    # Optional arguments
    parser.add_argument('-o', '--output', 
                       type=str, 
                       default='refined_dataset.json',
                       help='Path to output refined dataset JSON file (default: refined_dataset.json)')
    parser.add_argument('-r', '--report', 
                       type=str, 
                       default=None,
                       help='Path to save detailed refinement report (optional)')
    parser.add_argument('-m', '--model', 
                       type=str, 
                       default='llama3.2',
                       help='Ollama model to use for refinement (default: llama3.2)')
    parser.add_argument('-v', '--verbose', 
                       action='store_true',
                       help='Enable verbose output')
    parser.add_argument('-i', '--interactive', 
                       action='store_true',
                       help='Enable interactive mode to approve/edit each refinement')
    
    args = parser.parse_args()
    
    # Load judge datasets
    try:
        with open(args.judge1_file, 'r', encoding='utf-8') as f:
            judge1_data = json.load(f)
        # Handle both full report format and direct list format
        if isinstance(judge1_data, dict) and 'detailed_results' in judge1_data:
            judge1_data = judge1_data['detailed_results']
        print(f"Loaded {len(judge1_data)} items from Judge 1: {args.judge1_file}")
    except FileNotFoundError:
        print(f"Error: File '{args.judge1_file}' not found")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{args.judge1_file}': {e}")
        return
    
    try:
        with open(args.judge2_file, 'r', encoding='utf-8') as f:
            judge2_data = json.load(f)
        # Handle both full report format and direct list format
        if isinstance(judge2_data, dict) and 'detailed_results' in judge2_data:
            judge2_data = judge2_data['detailed_results']
        print(f"Loaded {len(judge2_data)} items from Judge 2: {args.judge2_file}")
    except FileNotFoundError:
        print(f"Error: File '{args.judge2_file}' not found")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{args.judge2_file}': {e}")
        return
    
    # Analyze datasets
    print(f"\nUsing model: {args.model}")
    refiner = DatasetRefiner(model=args.model, verbose=args.verbose, interactive=args.interactive)
    refiner.analyze_datasets(judge1_data, judge2_data)
    
    # Display initial summary
    summary = refiner.generate_summary()
    print("\n" + "="*60)
    print("INITIAL ANALYSIS")
    print("="*60)
    print(f"Total Items: {summary['total_items']}")
    print(f"No Refinement Needed: {summary['no_refinement_needed']}")
    print(f"\nRefinement Needed: {summary['total_needing_refinement']} ({summary['refinement_percentage']:.1f}%)")
    print(f"  - Refine Question Only: {summary['refine_question_only']}")
    print(f"  - Refine Answer Only: {summary['refine_answer_only']}")
    print(f"  - Refine Both: {summary['refine_both']}")
    
    # Refine items
    refiner.refine_all()
    
    # Export refined dataset
    refiner.export_refined_dataset(args.output)
    
    # Save detailed report if requested
    if args.report:
        refiner.save_detailed_report(args.report)
    
    print(f"\n{'='*60}")
    print("REFINEMENT COMPLETE!")
    print("="*60)
    print(f"Refined dataset: {args.output}")
    if args.report:
        print(f"Detailed report: {args.report}")


if __name__ == "__main__":
    main()