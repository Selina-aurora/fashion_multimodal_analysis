"""Analyze category distribution in DeepFashion2 annotations.

This script scans DeepFashion2 training annotation files, counts clothing
instances by category, and saves the category statistics as a CSV file.
"""

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

PROGRESS_INTERVAL = 10_000


def load_annotation(annotation_path: Path) -> dict[str, Any]:
    """Load one DeepFashion2 annotation file.

    Args:
        annotation_path: Path to the JSON annotation file.

    Returns:
        Parsed annotation data.

    Raises:
        FileNotFoundError: If the annotation file does not exist.
        json.JSONDecodeError: If the JSON file cannot be parsed.
        UnicodeDecodeError: If the file is not valid UTF-8 text.
    """
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_category_names(
    annotation: dict[str, Any],
) -> list[str]:
    """Extract clothing category names from one annotation.

    Args:
        annotation: Parsed DeepFashion2 annotation data.

    Returns:
        Category names for all valid clothing instances.
    """
    category_names: list[str] = []

    for key, value in annotation.items():
        if not key.startswith("item"):
            continue

        if not isinstance(value, dict):
            continue

        category_name = value.get("category_name")

        if isinstance(category_name, str) and category_name:
            category_names.append(category_name)

    return category_names


def analyze_annotations(
    annotation_dir: Path,
) -> tuple[Counter[str], int, int, int]:
    """Analyze all DeepFashion2 annotation files in a directory.

    Args:
        annotation_dir: Directory containing DeepFashion2 JSON annotations.

    Returns:
        Category counter, processed file count, clothing instance count,
        and failed file count.

    Raises:
        FileNotFoundError: If the annotation directory does not exist.
    """
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    category_counter: Counter[str] = Counter()
    processed_file_count = 0
    instance_count = 0
    failed_file_count = 0

    for annotation_path in sorted(annotation_dir.glob("*.json")):
        try:
            annotation = load_annotation(annotation_path)
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
            failed_file_count += 1
            LOGGER.warning(
                "Failed to read annotation %s: %s",
                annotation_path.name,
                error,
            )
            continue

        category_names = extract_category_names(annotation)

        category_counter.update(category_names)
        processed_file_count += 1
        instance_count += len(category_names)

        if processed_file_count % PROGRESS_INTERVAL == 0:
            LOGGER.info(
                "Processed %d annotation files",
                processed_file_count,
            )

    return (
        category_counter,
        processed_file_count,
        instance_count,
        failed_file_count,
    )


def save_category_summary(
    category_counter: Counter[str],
    output_path: Path,
) -> None:
    """Save category counts and percentages to a CSV file.

    Args:
        category_counter: Clothing category occurrence counts.
        output_path: Destination CSV path.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_instances = sum(category_counter.values())

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "category_name",
                "instance_count",
                "percentage",
            ]
        )

        for category_name, count in category_counter.most_common():
            percentage = count / total_instances * 100 if total_instances > 0 else 0.0

            writer.writerow(
                [
                    category_name,
                    count,
                    round(percentage, 2),
                ]
            )


def configure_logging() -> None:
    """Configure console logging for the analysis script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def main() -> None:
    """Run DeepFashion2 category distribution analysis."""
    configure_logging()

    project_root = Path(__file__).resolve().parents[1]

    annotation_dir = (
        project_root.parent / "fashion_data" / "raw" / "train" / "train" / "annos"
    )

    output_path = (
        project_root
        / "outputs"
        / "dataset_analysis"
        / "deepfashion2_category_summary.csv"
    )

    LOGGER.info(
        "Analyzing annotations in %s",
        annotation_dir,
    )

    (
        category_counter,
        processed_file_count,
        instance_count,
        failed_file_count,
    ) = analyze_annotations(annotation_dir)

    save_category_summary(
        category_counter,
        output_path,
    )

    LOGGER.info(
        "Processed annotation files: %d",
        processed_file_count,
    )
    LOGGER.info(
        "Clothing instances: %d",
        instance_count,
    )
    LOGGER.info(
        "Detected categories: %d",
        len(category_counter),
    )
    LOGGER.info(
        "Failed annotation files: %d",
        failed_file_count,
    )
    LOGGER.info(
        "Summary saved to: %s",
        output_path,
    )


if __name__ == "__main__":
    main()
