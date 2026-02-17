"""
WiQAS Dataset Utility CLI

All-in-one tool for:
1. Reshaping datasets to RAGAS-ready format
2. Merging refined JSON files
3. Cleaning out refinement_info fields
4. Extract-questions (export all questions delimited by '?')


Usage Examples:
---------------
  # Single reshape
  python reshaper.py reshape input.json output.json

  # Batch reshape and combine
  python reshaper.py reshape-folder Final evaluation_dataset.json

  # Merge refined files (answer-refined + question-refined)
  python reshaper.py merge --a a_refined.json --q q_refined.json --out merged.json

  # Remove refinement info
  python reshaper.py clean dataset.json
  python reshaper.py clean dataset.json --overwrite

  # Extract qusetions
  python reshaper.py extract-questions dataset.json --out questions.txt
"""

import json
import logging
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path
import csv


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("WiQAS-Tool")


# --------------------------- #
#        JSON LOADING         #
# --------------------------- #
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --------------------------- #
#          CLEANER            #
# --------------------------- #
def remove_refinement_info(json_path: str, overwrite: bool = False):
    path = Path(json_path)
    if not path.exists():
        logger.error(f"File not found: {json_path}")
        return

    data = load_json(path)

    def clean_item(item: Dict[str, Any]):
        meta = item.get("metadata", {})
        if "refinement_info" in meta:
            meta.pop("refinement_info")

    if isinstance(data, dict) and "detailed_results" in data:
        for entry in data["detailed_results"]:
            clean_item(entry)
    elif isinstance(data, list):
        for entry in data:
            clean_item(entry)
    else:
        logger.warning("No recognizable structure (list or detailed_results). Skipping.")
        return

    output_path = path if overwrite else path.with_name(f"{path.stem}_no_refinement.json")
    save_json(data, output_path)

    logger.info(f"Saved cleaned file to {output_path}")


# --------------------------- #
#         REFINER MERGE       #
# --------------------------- #
class JSONRefiner:
    def __init__(self, a_path: Path, q_path: Path, output_path: Path):
        self.a_data = load_json(a_path)
        self.q_data = load_json(q_path)
        self.output_path = output_path

    def _build_question_index(self, data):
        return {entry["question"]: entry for entry in data}

    def merge(self):
        q_index = self._build_question_index(self.q_data)
        updated_count = 0

        for refined in self.a_data:
            refinement_info = refined.get("metadata", {}).get("refinement_info")
            if refinement_info:
                action = refinement_info.get("action", "").lower()
                if action in ("refine_answer", "refine_both"):
                    question = refined["question"]
                    if question in q_index:
                        q_index[question] = refined
                        updated_count += 1

        merged = list(q_index.values())
        save_json(merged, self.output_path)

        logger.info(f"Merged file saved to {self.output_path}")
        logger.info(f"{updated_count} entries updated.")


# --------------------------- #
#          RESHAPER           #
# --------------------------- #
@dataclass
class QAEntry:
    question: str
    ground_truth: str
    metadata: Dict[str, Any]


class DatasetReshaper:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.raw_data: Optional[List[Dict[str, Any]]] = None
        self.reshaped: List[QAEntry] = []

    def load(self) -> None:
        logger.info(f"Loading dataset from {self.input_path}")
        data = load_json(self.input_path)

        if isinstance(data, dict) and "qa_pairs" in data:
            self.raw_data = data["qa_pairs"]
        elif isinstance(data, list):
            self.raw_data = data
        else:
            raise ValueError("Input JSON must contain either a list or a 'qa_pairs' field.")

        logger.info(f"Loaded {len(self.raw_data)} QA pairs.")

    def reshape_entry(self, record: Dict[str, Any]) -> QAEntry:
        question = record.get("question", "").strip()
        ground_truth = (record.get("answer") or "").strip()
        context = record.get("context") or record.get("contexts", [])
        if isinstance(context, list):
            context = " ".join(context)
        elif not isinstance(context, str):
            context = ""

        ignored = {"question", "answer", "context", "contexts"}
        metadata = {k: v for k, v in record.items() if k not in ignored}
        metadata["ground_truth_context"] = context

        return QAEntry(question, ground_truth, metadata)

    def transform(self) -> None:
        for rec in self.raw_data:
            try:
                self.reshaped.append(self.reshape_entry(rec))
            except Exception as e:
                logger.warning(f"Skipping malformed record: {e}")

        logger.info(f"Reshaped {len(self.reshaped)} entries.")

    def save(self) -> None:
        save_json([asdict(entry) for entry in self.reshaped], self.output_path)
        logger.info(f"Saved reshaped dataset to {self.output_path}")

    def process(self) -> None:
        self.load()
        self.transform()
        self.save()

    @classmethod
    def process_folder(cls, folder_path: str, output_path: str = "evaluation_dataset.json"):
        folder = Path(folder_path)
        if not folder.exists():
            logger.error(f"Directory not found: {folder}")
            return

        all_entries = []
        for json_file in folder.glob("*.json"):
            try:
                logger.info(f"Processing {json_file.name}")
                reshaper = cls(str(json_file), "temp.json")
                reshaper.load()
                reshaper.transform()
                all_entries.extend([asdict(e) for e in reshaper.reshaped])
            except Exception as e:
                logger.warning(f"Skipping {json_file}: {e}")

        save_json(all_entries, Path(output_path))
        logger.info(f"Combined {len(all_entries)} entries into {output_path}")

# --------------------------- #
#       CONTEXT MERGER        #
# --------------------------- #
class ContextMerger:
    def __init__(self, base_path: Path, merge_path: Path, output_path: Path):
        self.base_data = load_json(base_path)
        self.merge_data = load_json(merge_path)
        self.output_path = output_path

    def _index_by_question(self, data):
        return {item["question"]: item for item in data}

    def merge(self):
        merge_index = self._index_by_question(self.merge_data)
        merged_count = 0

        for entry in self.base_data:
            q = entry.get("question")
            if q in merge_index:
                m = merge_index[q]

                # Add model answer
                entry["model_answer"] = m.get("answer", "")

                # Extract only text fields from contexts
                if "contexts" in m and isinstance(m["contexts"], list):
                    entry["contexts"] = [c.get("text", "") for c in m["contexts"] if "text" in c]

                merged_count += 1

        save_json(self.base_data, self.output_path)
        logger.info(f"Merged {merged_count} entries with contexts (texts only) and model answers.")
        logger.info(f"Saved to {self.output_path}")


# --------------------------- #
#     EXTRACT QUESTIONS        #
# --------------------------- #
def extract_questions(json_path: str, output_path: Optional[str] = None):
    """
    Extract all 'question' fields from a JSON dataset and save to a text file
    delimited by '?'.
    """
    path = Path(json_path)
    if not path.exists():
        logger.error(f"File not found: {json_path}")
        return

    data = None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Try to extract questions from either list or 'detailed_results'
    if isinstance(data, list):
        questions = [entry.get("question", "").strip() for entry in data if "question" in entry]
    elif isinstance(data, dict) and "detailed_results" in data:
        questions = [entry.get("question", "").strip() for entry in data["detailed_results"] if "question" in entry]
    else:
        logger.error("Unrecognized JSON structure. Must be a list or contain 'detailed_results'.")
        return

    # Join with '?', ensuring each ends properly
    cleaned = [q if q.endswith("?") else q + "?" for q in questions if q]
    text_output = "".join(cleaned)

    # Default output path
    if not output_path:
        output_path = path.with_name(f"{path.stem}_questions.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text_output)

    logger.info(f"Extracted {len(questions)} questions to {output_path}")



# --------------------------- #
#         CSV MERGER           #
# --------------------------- #

class CSVJSONMerger:
    def __init__(self, csv_path: Path, json_path: Path, output_path: Path):
        self.csv_path = csv_path
        self.json_path = json_path
        self.output_path = output_path
        self.csv_data = []
        self.json_data = []

    def load(self):
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.csv_data = list(reader)
        self.json_data = load_json(self.json_path)

    def _index_json_by_question(self):
        return {item["question"]: item for item in self.json_data}

    def merge(self):
        self.load()
        json_index = self._index_json_by_question()
        merged_rows = []

        for row in self.csv_data:
            q = row.get("question", "").strip()
            if q in json_index:
                j = json_index[q]
                row["ground_truth"] = j.get("ground_truth", "")
                row["model_answer"] = j.get("model_answer", "")
                # Flatten contexts if list
                contexts = j.get("contexts", [])
                if isinstance(contexts, list):
                    row["contexts"] = " ||| ".join(contexts)
                else:
                    row["contexts"] = str(contexts)
            else:
                row["ground_truth"] = ""
                row["model_answer"] = ""
                row["contexts"] = ""
            merged_rows.append(row)

        # Write out merged CSV
        fieldnames = list(merged_rows[0].keys()) if merged_rows else []
        with open(self.output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_rows)

        logger.info(f"Merged {len(merged_rows)} CSV rows with JSON data by question.")
        logger.info(f"Saved merged CSV to {self.output_path}")

# --------------------------- #
#   CULTURAL GOLDEN UPDATER   #
# --------------------------- #
class CulturalGoldenUpdater:
    """
    Updates the 'metadata.cultural_golden_answer' field of an evaluation dataset
    using entries from a refined JSON file that contains updated golden answers.

    Usage:
        python reshaper.py update-cga --base evaluation_dataset.json --refined refined.json --out updated.json
    """
    def __init__(self, base_path: Path, refined_path: Path, output_path: Path):
        self.base_path = base_path
        self.refined_path = refined_path
        self.output_path = output_path
        self.base_data = load_json(base_path)
        self.refined_data = load_json(refined_path)

    def _index_by_question(self, data):
        return {item["question"]: item for item in data}

    def update(self):
        refined_index = self._index_by_question(self.refined_data)
        updated_count = 0

        for entry in self.base_data:
            q = entry.get("question")
            if q in refined_index:
                refined_entry = refined_index[q]
                refined_cga = refined_entry.get("metadata", {}).get("cultural_gold_answer")
                if refined_cga:
                    entry.setdefault("metadata", {})["cultural_gold_answer"] = refined_cga
                    updated_count += 1


        save_json(self.base_data, self.output_path)
        logger.info(f"Updated {updated_count} entries with new cultural_golden_answer values.")
        logger.info(f"Saved updated dataset to {self.output_path}")


# --------------------------- #
#           CLI MAIN          #
# --------------------------- #
def main():
    parser = argparse.ArgumentParser(description="WiQAS Dataset Utility CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Reshape single file
    reshape_p = subparsers.add_parser("reshape", help="Reshape a single dataset file")
    reshape_p.add_argument("input", help="Input JSON path")
    reshape_p.add_argument("output", help="Output JSON path")

    # Batch reshape
    reshape_f = subparsers.add_parser("reshape-folder", help="Reshape all JSON files in a folder and combine them")
    reshape_f.add_argument("folder", help="Folder containing JSON files")
    reshape_f.add_argument("output", nargs="?", default="evaluation_dataset.json", help="Combined output file")

    # Merge
    merge_p = subparsers.add_parser("merge", help="Merge refined A/Q JSON files")
    merge_p.add_argument("--a", required=True, help="Path to a_refined.json")
    merge_p.add_argument("--q", required=True, help="Path to q_refined.json")
    merge_p.add_argument("--out", required=True, help="Output JSON path")

    # Merge contexts + model answers
    merge_c = subparsers.add_parser("merge-contexts", help="Merge contexts (texts only) and answers via question")
    merge_c.add_argument("--base", required=True, help="Base JSON (with ground_truth etc.)")
    merge_c.add_argument("--merge", required=True, help="JSON containing contexts and answers")
    merge_c.add_argument("--out", required=True, help="Output merged JSON path")

    # Clean
    clean_p = subparsers.add_parser("clean", help="Remove refinement_info from JSON")
    clean_p.add_argument("json_path", help="Path to JSON file")
    clean_p.add_argument("--overwrite", action="store_true", help="Overwrite the original file")

    # Extract questions
    extract_p = subparsers.add_parser("extract-questions", help="Extract all questions to a text file delimited by '?'")
    extract_p.add_argument("json_path", help="Path to dataset JSON")
    extract_p.add_argument("--out", help="Output text file path")
    
    # Merge CSV with JSON (add the ground truth and model answer)
    merge_csv = subparsers.add_parser("merge-csv", help="Merge CSV with JSON via question")
    merge_csv.add_argument("--csv", required=True, help="Path to CSV file")
    merge_csv.add_argument("--json", required=True, help="Path to JSON file")
    merge_csv.add_argument("--out", required=True, help="Output CSV path")

    # Update cultural golden answers
    update_cga = subparsers.add_parser("update-cga", help="Update cultural golden answers in base dataset")
    update_cga.add_argument("--base", required=True, help="Path to evaluation_dataset.json")
    update_cga.add_argument("--refined", required=True, help="Path to refined JSON with correct cultural_golden_answer")
    update_cga.add_argument("--out", required=True, help="Output JSON file path")

    args = parser.parse_args()

    if args.command == "reshape":
        DatasetReshaper(args.input, args.output).process()

    elif args.command == "reshape-folder":
        DatasetReshaper.process_folder(args.folder, args.output)

    elif args.command == "merge":
        JSONRefiner(Path(args.a), Path(args.q), Path(args.out)).merge()

    elif args.command == "clean":
        remove_refinement_info(args.json_path, args.overwrite)

    elif args.command == "extract-questions":
        extract_questions(args.json_path, args.out)
    
    elif args.command == "merge-contexts":
        ContextMerger(Path(args.base), Path(args.merge), Path(args.out)).merge()


    elif args.command == "merge-csv":
        CSVJSONMerger(Path(args.csv), Path(args.json), Path(args.out)).merge()

    elif args.command == "update-cga":
        CulturalGoldenUpdater(Path(args.base), Path(args.refined), Path(args.out)).update()


    else:
        parser.print_help()


if __name__ == "__main__":
    main()
