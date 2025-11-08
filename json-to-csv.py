import json
import csv

# Input and output file paths
input_file = "data/test.json"
output_file = "data/test.csv"

# Load JSON data
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Define CSV headers
headers = [
    "question",
    "answer",
    "query_type",
    "language",
    "contexts_combined",
    "classification_detected_type",
    "classification_detected_language",
    "classification_confidence",
    "classification_used_type",
    "classification_used_language",
    "timing_total_time"
]

# Write to CSV
with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()

    for item in data:
        # Combine all context info into one formatted string
        combined_contexts = []
        for c in item.get("contexts", []):
            combined_contexts.append(
                f"[Score: {c.get('final_score', '')}] "
                f"{c.get('text', '').strip()} "
                f"(Source: {c.get('source_file', '')}; Citation: {c.get('citation_text', '')})"
            )
        contexts_str = "\n\n".join(combined_contexts)

        row = {
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "query_type": item.get("query_type", ""),
            "language": item.get("language", ""),
            "contexts_combined": contexts_str,
            "classification_detected_type": item.get("classification", {}).get("detected_type", ""),
            "classification_detected_language": item.get("classification", {}).get("detected_language", ""),
            "classification_confidence": item.get("classification", {}).get("confidence", ""),
            "classification_used_type": item.get("classification", {}).get("used_type", ""),
            "classification_used_language": item.get("classification", {}).get("used_language", ""),
            "timing_total_time": item.get("timing", {}).get("total_time", ""),
        }

        writer.writerow(row)

print(f"Combined contexts and wrote {len(data)} entries to {output_file}")