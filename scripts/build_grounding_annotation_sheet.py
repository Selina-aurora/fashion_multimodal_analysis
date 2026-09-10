"""Build a manual annotation sheet for grounding evaluation."""

import csv
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "evaluation_sampling"
    / "eval_pool_300.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "evaluation_sampling"
    / "grounding_presence_annotations.csv"
)


def read_evaluation_pool(input_file: Path) -> List[Dict[str, str]]:
    """Read sampled image information from the evaluation pool.

    Args:
        input_file: CSV manifest containing sampled images.

    Returns:
        Rows read from the evaluation pool.

    Raises:
        FileNotFoundError: If the evaluation pool does not exist.
    """
    if not input_file.exists():
        raise FileNotFoundError(
            f"Evaluation pool does not exist: {input_file}"
        )

    with input_file.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)
        return list(reader)


def get_group_id(sample_id: int) -> int:
    """Determine the evaluation group of a sample.

    Args:
        sample_id: One-based sample identifier.

    Returns:
        Evaluation group identifier from 1 to 3.
    """
    return ((sample_id - 1) // 100) + 1


def build_annotation_rows(
    pool_rows: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Create blank annotation records for fine-grained targets.

    Args:
        pool_rows: Image records from the evaluation pool.

    Returns:
        Annotation rows with empty target-presence fields.
    """
    annotation_rows = []

    for row in pool_rows:
        sample_id = int(row["sample_id"])

        annotation_rows.append(
            {
                "sample_id": str(sample_id),
                "group_id": str(get_group_id(sample_id)),
                "image_name": row["image_name"],
                "relative_path": row["relative_path"],
                "sleeve_present": "",
                "collar_present": "",
                "button_present": "",
                "zipper_present": "",
                "notes": "",
            }
        )

    return annotation_rows


def save_annotation_sheet(
    annotation_rows: List[Dict[str, str]],
    output_file: Path,
) -> None:
    """Save the manual annotation sheet.

    Args:
        annotation_rows: Annotation records to save.
        output_file: Destination CSV file.
    """
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "sample_id",
        "group_id",
        "image_name",
        "relative_path",
        "sleeve_present",
        "collar_present",
        "button_present",
        "zipper_present",
        "notes",
    ]

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(annotation_rows)


def main() -> None:
    """Generate the grounding target-presence annotation sheet."""
    pool_rows = read_evaluation_pool(INPUT_FILE)

    annotation_rows = build_annotation_rows(
        pool_rows
    )

    save_annotation_sheet(
        annotation_rows=annotation_rows,
        output_file=OUTPUT_FILE,
    )

    print(
        f"Evaluation images: {len(annotation_rows)}"
    )
    print(
        f"Annotation file: {OUTPUT_FILE}"
    )
    print(
        "Presence labels are intentionally blank."
    )
    print(
        "Use 1 = present, 0 = absent."
    )


if __name__ == "__main__":
    main()