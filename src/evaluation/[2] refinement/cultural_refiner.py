import argparse
import json
from dataclasses import dataclass
from typing import Any

import ollama


@dataclass
class CulturalRefinementResult:
    question: str
    ground_truth: str
    original_cultural_answer: str
    refined_cultural_answer: str
    refinement_reasoning: str
    language: str
    context_snippet: str
    metadata: dict[str, Any]


class CulturalAnswerRefiner:
    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def refine_cultural_answer(self, question: str, ground_truth: str, original_cultural_answer: str, context: str, language: str) -> tuple[str, str]:
        """Aggressively refine cultural golden answer to be based on ground truth and context."""

        prompt = f"""You are refining a "cultural golden answer" for a Filipino QA dataset. 

CRITICAL RULE: The cultural golden answer MUST be based on the ground truth and context ONLY. Do NOT add information that isn't supported by these sources. A culturally
golden answer must be one sentence only.

Question:
{question}

Ground Truth (Factoid Answer):
{ground_truth}

Context (YOUR ONLY SOURCE):
{context}

Current Cultural Golden Answer (MAY BE WRONG - NEEDS AGGRESSIVE REFINEMENT):
{original_cultural_answer}

Language: {language}

YOUR TASK - AGGRESSIVE REFINEMENT:

The cultural golden answer should follow this formula:
1. START with the ground truth factoid (the base answer)
2. ONLY if the context contains cultural significance/context, ADD that specific information
3. If context has NO cultural significance mentioned, the cultural golden answer = ground truth (keep it the same)
4. Culturally golden answer must be one sentence maximum.

WHAT TO CHECK:
- Does the current cultural answer add information NOT in the context? → REMOVE IT
- Does it mention cultural significance not stated in context? → REMOVE IT  
- Does it add historical connections not mentioned in context? → REMOVE IT
- Does the current cultural answer exceed one sentence? → REMOVE IT

WHAT TO KEEP/ADD:
- Only cultural details EXPLICITLY stated in the context
- Only significance DIRECTLY mentioned in the source
- Only connections the context actually makes

Examples of GOOD refinement:

Example 1:
- Ground truth: "The document was written in 900 CE"
- Context: "The document was written in 900 CE and is the earliest known written record in the Philippines"
- Cultural golden: "The document was written in 900 CE and is the earliest known written record in the Philippines" (added detail from context)

Example 2:
- Ground truth: "Lady Angkatan was freed from debt"
- Context: "The document freed Lady Angkatan from debt slavery"
- Cultural golden: "Lady Angkatan was freed from debt slavery" (only added "slavery" which is in context)

Example 3:
- Ground truth: "The inscription was on copper"
- Context: "The inscription was carved on a copper plate"
- Cultural golden: "The inscription was on copper" (context adds nothing culturally significant, keep same as ground truth)

COMMON MISTAKES TO FIX:
❌ Adding "Sa kasalukuyan, ang kultura..." (modern connections not in context)
❌ Adding "nagsilbi itong patunay ng..." (interpretations not in context)
❌ Adding significance about "kalayaan at hustisya" not mentioned in context
❌ Making it longer without adding context-based information

WHAT TO DO:
✅ Start with ground truth
✅ Only add what's explicitly in the context
✅ If nothing to add from context, keep it the same as ground truth
✅ Culturally answer must be one sentence maximum

Respond in the following format:
REFINED_CULTURAL_ANSWER: [Your refined cultural golden answer here]
REASONING: [Explain what you removed/kept/added based on ground truth and context]"""

        try:
            response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])

            result = response["message"]["content"]

            # Parse response
            answer_lines = [line for line in result.split("\n") if line.startswith("REFINED_CULTURAL_ANSWER:")]
            reasoning_lines = [line for line in result.split("\n") if line.startswith("REASONING:")]

            if answer_lines:
                refined = answer_lines[0].replace("REFINED_CULTURAL_ANSWER:", "").strip()
                reasoning = reasoning_lines[0].replace("REASONING:", "").strip() if reasoning_lines else "No reasoning provided"
                return refined, reasoning

            return original_cultural_answer, "Error: Could not parse LLM response"

        except Exception as e:
            return original_cultural_answer, f"Error during refinement: {str(e)}"


class CulturalDatasetRefiner:
    def __init__(self, model: str = "llama3.2", verbose: bool = False, interactive: bool = False):
        self.refiner = CulturalAnswerRefiner(model)
        self.verbose = verbose
        self.interactive = interactive
        self.results: list[CulturalRefinementResult] = []

    def get_user_approval(self, original: str, refined: str) -> str:
        """Interactive CLI to approve or edit refinements."""
        print(f"\n{'='*60}")
        print("Original Cultural Answer:")
        print(f"  {original}")
        print("\nRefined Cultural Answer:")
        print(f"  {refined}")
        print(f"\n{'='*60}")

        while True:
            choice = input("\nOptions: [a]ccept / [e]dit / [k]eep original: ").lower().strip()

            if choice == "a":
                return refined
            elif choice == "k":
                return original
            elif choice == "e":
                print("\nEnter your edited cultural answer (press Enter twice to finish):")
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

    def refine_item(self, item: dict[str, Any], index: int, total: int) -> CulturalRefinementResult:
        """Refine a single item's cultural golden answer."""
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        metadata = item.get("metadata", {})

        original_cultural = metadata.get("cultural_golden_answer", "")
        language = metadata.get("language", "unknown")
        context = metadata.get("ground_truth_context", "")

        print(f"\n{'='*60}")
        print(f"[{index}/{total}] Refining cultural answer")
        print(f"{'='*60}")
        print(f"Question: {question[:80]}...")

        if self.verbose:
            print(f"Ground truth: {ground_truth}")
            print(f"Original cultural: {original_cultural[:100]}...")

        # Refine the cultural answer
        print("Refining cultural golden answer...")
        refined_cultural, reasoning = self.refiner.refine_cultural_answer(question, ground_truth, original_cultural, context, language)

        if self.interactive:
            refined_cultural = self.get_user_approval(original_cultural, refined_cultural)

        if self.verbose:
            print(f"Refined: {refined_cultural}")
            print(f"Reasoning: {reasoning}")

        return CulturalRefinementResult(
            question=question,
            ground_truth=ground_truth,
            original_cultural_answer=original_cultural,
            refined_cultural_answer=refined_cultural,
            refinement_reasoning=reasoning,
            language=language,
            context_snippet=context[:200] + "..." if len(context) > 200 else context,
            metadata=metadata,
        )

    def refine_all(self, dataset: list[dict[str, Any]]):
        """Refine all items in the dataset."""
        total = len(dataset)
        print(f"\nStarting refinement of {total} items...")

        for i, item in enumerate(dataset, 1):
            result = self.refine_item(item, i, total)
            self.results.append(result)

        print(f"\n{'='*60}")
        print(f"Refinement complete! Refined {total} items.")
        print(f"{'='*60}")

    def generate_summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        total = len(self.results)

        # Count how many were actually changed
        changed = sum(1 for r in self.results if r.original_cultural_answer != r.refined_cultural_answer)
        unchanged = total - changed

        return {"total_items": total, "changed": changed, "unchanged": unchanged, "change_percentage": (changed / total * 100) if total > 0 else 0}

    def export_refined_dataset(self, output_path: str):
        """Export refined dataset with updated cultural golden answers."""
        refined_dataset = []

        for result in self.results:
            # Update metadata with refined cultural answer
            metadata = result.metadata.copy()
            metadata["cultural_golden_answer"] = result.refined_cultural_answer

            # Add refinement tracking
            if result.original_cultural_answer != result.refined_cultural_answer:
                metadata["cultural_refinement_info"] = {"original_cultural_answer": result.original_cultural_answer, "refinement_reasoning": result.refinement_reasoning, "was_refined": True}
            else:
                metadata["cultural_refinement_info"] = {"was_refined": False}

            refined_item = {"question": result.question, "ground_truth": result.ground_truth, "metadata": metadata}

            refined_dataset.append(refined_item)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(refined_dataset, f, indent=2, ensure_ascii=False)

        print(f"\nRefined dataset exported to: {output_path}")

    def save_detailed_report(self, output_path: str):
        """Save detailed refinement report."""
        summary = self.generate_summary()

        detailed_results = []
        for result in self.results:
            detailed_results.append(
                {
                    "question": result.question,
                    "ground_truth": result.ground_truth,
                    "original_cultural_answer": result.original_cultural_answer,
                    "refined_cultural_answer": result.refined_cultural_answer,
                    "changed": result.original_cultural_answer != result.refined_cultural_answer,
                    "refinement_reasoning": result.refinement_reasoning,
                    "language": result.language,
                    "context_snippet": result.context_snippet,
                }
            )

        output = {"summary": summary, "detailed_results": detailed_results}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Detailed report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Refine cultural golden answers to be based on ground truth and context only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic refinement
  python cultural_refiner.py dataset.json -o refined_dataset.json
  
  # Use different model with verbose output
  python cultural_refiner.py dataset.json -m llama3.1:latest -v
  
  # Interactive mode to approve each refinement
  python cultural_refiner.py dataset.json -i
  
  # Generate detailed report
  python cultural_refiner.py dataset.json -o refined.json -r report.json
  
  # Full workflow with all options
  python cultural_refiner.py dataset.json -m llama3.2 -o refined.json -r report.json -v -i
        """,
    )

    parser.add_argument("input_file", type=str, help="Path to input JSON dataset file")
    parser.add_argument("-o", "--output", type=str, default="refined_cultural_dataset.json", help="Path to output refined dataset JSON file (default: refined_cultural_dataset.json)")
    parser.add_argument("-r", "--report", type=str, default=None, help="Path to save detailed refinement report (optional)")
    parser.add_argument("-m", "--model", type=str, default="llama3.2", help="Ollama model to use for refinement (default: llama3.2)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-i", "--interactive", action="store_true", help="Enable interactive mode to approve/edit each refinement")

    args = parser.parse_args()

    # Load dataset
    try:
        with open(args.input_file, encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} items from {args.input_file}")
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{args.input_file}': {e}")
        return

    # Refine dataset
    print(f"\nUsing model: {args.model}")
    refiner = CulturalDatasetRefiner(model=args.model, verbose=args.verbose, interactive=args.interactive)
    refiner.refine_all(dataset)

    # Display summary
    summary = refiner.generate_summary()
    print("\n" + "=" * 60)
    print("REFINEMENT SUMMARY")
    print("=" * 60)
    print(f"Total Items: {summary['total_items']}")
    print(f"Changed: {summary['changed']} ({summary['change_percentage']:.1f}%)")
    print(f"Unchanged: {summary['unchanged']}")

    # Export refined dataset
    refiner.export_refined_dataset(args.output)

    # Save detailed report if requested
    if args.report:
        refiner.save_detailed_report(args.report)

    print(f"\n{'='*60}")
    print("REFINEMENT COMPLETE!")
    print("=" * 60)
    print(f"Refined dataset: {args.output}")
    if args.report:
        print(f"Detailed report: {args.report}")


if __name__ == "__main__":
    main()
