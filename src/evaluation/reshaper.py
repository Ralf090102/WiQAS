#!/usr/bin/env python3
"""
Reshaper for WiQAS datasets.

Converts generated bilingual/high-level QA datasets into a RAGAS-ready format.
Moves 'context' under 'metadata' as 'ground_truth_context' and renames 'answer' → 'ground_truth'.
Handles input JSONs with metadata wrappers or flat lists.
"""
import json
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("Reshaper")


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

    # --------------------------- #
    #           LOAD              #
    # --------------------------- #
    def load(self) -> None:
        logger.info(f"Loading dataset from {self.input_path}")
        with open(self.input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle wrapped format (metadata + qa_pairs)
        if isinstance(data, dict) and "qa_pairs" in data:
            self.raw_data = data["qa_pairs"]
            logger.info(f"Detected wrapped dataset with {len(self.raw_data)} QA pairs.")
        elif isinstance(data, list):
            self.raw_data = data
            logger.info(f"Detected flat dataset with {len(self.raw_data)} QA pairs.")
        else:
            raise ValueError("Input JSON must contain either a list or a 'qa_pairs' field.")

    # --------------------------- #
    #        RESHAPE ENTRY        #
    # --------------------------- #
    def reshape_entry(self, record: Dict[str, Any]) -> QAEntry:
        """Convert a single record to a RAGAS-ready QAEntry."""
        question = record.get("question", "").strip()

        # Rename answer → ground_truth
        ground_truth = (
            record.get("answer")
            or "").strip()

        # Extract context (single string or list)
        context = record.get("context")
        if context is None:
            context = record.get("contexts", [])
            if isinstance(context, list) and len(context) == 1:
                context = context[0]
            elif isinstance(context, list):
                context = " ".join(context)
            elif not isinstance(context, str):
                context = ""

        # Build metadata (move context inside)
        ignored_fields = {"question", "answer"} # should stay outside of metadata
        metadata = {k: v for k, v in record.items() if k not in ignored_fields}
        metadata["ground_truth_context"] = context

        return QAEntry(
            question=question,
            ground_truth=ground_truth,
            metadata=metadata,
        )

    # --------------------------- #
    #        TRANSFORM ALL        #
    # --------------------------- #
    def transform(self) -> None:
        if not self.raw_data:
            raise ValueError("Dataset not loaded. Call load() first.")

        logger.info("Reshaping entries...")
        for rec in self.raw_data:
            try:
                reshaped = self.reshape_entry(rec)
                self.reshaped.append(reshaped)
            except Exception as e:
                logger.warning(f"Skipping malformed record: {e}")

        logger.info(f"Successfully reshaped {len(self.reshaped)} entries.")

    # --------------------------- #
    #            SAVE             #
    # --------------------------- #
    def save(self) -> None:
        logger.info(f"Saving reshaped dataset to {self.output_path}")
        data = [asdict(entry) for entry in self.reshaped]
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Reshaped dataset saved successfully.")

    # --------------------------- #
    #           PROCESS           #
    # --------------------------- #
    def process(self) -> None:
        self.load()
        self.transform()
        self.save()
    
    # --------------------------- #
    #         BATCH MODE          #
    # --------------------------- #
    @classmethod
    def process_folder(cls, folder_path: str, combined_output: str = "evaluation_dataset.json") -> None:
        """
        Scans a folder for all .json files, reshapes each using DatasetReshaper,
        and merges everything into a single evaluation_dataset.json file.
        """
        folder = Path(folder_path)
        if not folder.exists():
            logger.error(f"Directory not found: {folder}")
            return

        json_files = list(folder.glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON files found in {folder}.")
            return

        logger.info(f"Found {len(json_files)} JSON files in {folder}.")

        all_entries = []

        for json_file in json_files:
            try:
                logger.info(f"Processing {json_file.name}")
                reshaper = cls(str(json_file), "temp_output.json")
                reshaper.load()
                reshaper.transform()
                all_entries.extend([asdict(entry) for entry in reshaper.reshaped])
            except Exception as e:
                logger.warning(f"Skipping {json_file.name}: {e}")

        combined_path = Path(combined_output)
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_entries, f, ensure_ascii=False, indent=2)

        logger.info(f"Combined {len(all_entries)} entries into {combined_output}")


if __name__ == "__main__":
    import sys

    # Usage:
    #   python reshaper.py <input_json> <output_json>
    #   python reshaper.py --folder Final
    #
    # The first runs single-file mode.
    # The second scans all JSONs in Final/ and merges them into evaluation_dataset.json.

    if len(sys.argv) == 3 and sys.argv[1] != "--folder":
        # Single file mode
        reshaper = DatasetReshaper(sys.argv[1], sys.argv[2])
        reshaper.process()

    elif len(sys.argv) >= 2 and sys.argv[1] == "--folder":
        # Folder batch mode
        folder = sys.argv[2] if len(sys.argv) >= 3 else "Final"
        DatasetReshaper.process_folder(folder)

    else:
        print("Usage:")
        print("  python reshaper.py <input_json> <output_json>")
        print("  python reshaper.py --folder [FOLDER_NAME]")
        sys.exit(1)