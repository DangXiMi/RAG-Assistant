import json
from pathlib import Path
import sys

GOLDEN_FILE = Path("data/evaluation/golden.jsonl")

VALID_DOC_IDS = {
    "doc_1",
    "doc_2",
    "doc_3",
    "doc_4",
    "doc_5",
    "doc_6",
    "doc_7",
}

REQUIRED_FIELDS = {
    "question": str,
    "answer": str,
    "ground_truth_doc_ids": list,
    "relevant_chunks": list,
}


def validate_record(record: dict, line_no: int) -> list[str]:
    errors = []

    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in record:
            errors.append(f"Line {line_no}: Missing field '{field}'")
            continue

        if not isinstance(record[field], expected_type):
            errors.append(
                f"Line {line_no}: Field '{field}' should be "
                f"{expected_type.__name__}, got {type(record[field]).__name__}"
            )

    if errors:
        return errors

    invalid_ids = [
        doc_id
        for doc_id in record["ground_truth_doc_ids"]
        if doc_id not in VALID_DOC_IDS
    ]

    if invalid_ids:
        errors.append(
            f"Line {line_no}: Invalid ground_truth_doc_ids: {invalid_ids}"
        )

    return errors


def main():
    if not GOLDEN_FILE.exists():
        print(f"ERROR: {GOLDEN_FILE} not found.")
        sys.exit(1)

    all_errors = []
    total = 0

    with GOLDEN_FILE.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(
                    f"Line {line_no}: Invalid JSON ({e})"
                )
                continue

            all_errors.extend(validate_record(record, line_no))

    print(f"Validated {total} records.")

    if all_errors:
        print("\nValidation FAILED:\n")
        for err in all_errors:
            print(f"  - {err}")
        sys.exit(1)

    print("Validation PASSED")


if __name__ == "__main__":
    main()