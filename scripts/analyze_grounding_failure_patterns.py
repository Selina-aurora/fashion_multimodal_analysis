"""Analyze Grounding DINO localization failure patterns.

The analysis uses the manually reviewed localization audit and summarizes:

1. Localization quality by target.
2. Error-type distribution.
3. Small-object versus larger-part failure patterns.
4. Representative failure cases.
5. A Markdown summary for experiment documentation.

This script uses only the Python standard library.
"""

import argparse
import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REVIEWED_AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "grounding_group_evaluation"
    / "manual_localization_audit_reviewed.csv"
)

FALLBACK_AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "grounding_group_evaluation"
    / "manual_localization_audit.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "grounding_failure_analysis"
)

QUALITY_LEVELS = (
    "correct",
    "coarse",
    "wrong",
)

SMALL_OBJECT_TARGETS = {
    "button",
    "zipper",
}

LARGER_PART_TARGETS = {
    "sleeve",
    "collar",
}

EXPECTED_TARGETS = (
    "sleeve",
    "collar",
    "button",
    "zipper",
)


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Analyze manually reviewed Grounding DINO "
            "localization failures."
        )
    )

    parser.add_argument(
        "--audit-file",
        type=Path,
        default=None,
        help=(
            "Reviewed manual localization audit CSV. "
            "If omitted, the script first looks for "
            "manual_localization_audit_reviewed.csv."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory used for generated reports.",
    )

    return parser.parse_args()


def resolve_audit_file(
    requested_path: Path | None,
) -> Path:
    """Resolve the input audit CSV.

    Args:
        requested_path: Optional user-supplied path.

    Returns:
        Existing audit file path.

    Raises:
        FileNotFoundError: If no suitable file exists.
    """
    if requested_path is not None:
        if not requested_path.is_file():
            raise FileNotFoundError(
                f"Audit file not found: {requested_path}"
            )

        return requested_path

    if DEFAULT_REVIEWED_AUDIT_FILE.is_file():
        return DEFAULT_REVIEWED_AUDIT_FILE

    if FALLBACK_AUDIT_FILE.is_file():
        LOGGER.warning(
            "Reviewed audit file was not found. "
            "Using fallback audit file: %s",
            FALLBACK_AUDIT_FILE,
        )

        return FALLBACK_AUDIT_FILE

    raise FileNotFoundError(
        "Could not find a manual localization audit CSV."
    )


def read_audit_rows(
    audit_file: Path,
) -> list[dict[str, str]]:
    """Read localization audit rows.

    Args:
        audit_file: Input CSV file.

    Returns:
        Parsed CSV rows.

    Raises:
        ValueError: If required columns are missing.
    """
    with audit_file.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        fieldnames = set(
            reader.fieldnames or []
        )

        required_fields = {
            "target",
            "prompt_type",
            "image_name",
            "localization_quality",
            "error_type",
            "notes",
        }

        missing_fields = (
            required_fields - fieldnames
        )

        if missing_fields:
            missing_text = ", ".join(
                sorted(missing_fields)
            )

            raise ValueError(
                "Audit CSV is missing required fields: "
                f"{missing_text}"
            )

        rows = list(reader)

    return rows


def normalize_text(
    value: str | None,
) -> str:
    """Normalize an optional text value.

    Args:
        value: Input text.

    Returns:
        Lowercase stripped text.
    """
    if value is None:
        return ""

    return value.strip().lower()


def validate_reviewed_rows(
    rows: list[dict[str, str]],
) -> None:
    """Validate that the audit contains manual review labels.

    Args:
        rows: Audit rows.

    Raises:
        ValueError: If no reviewed localization labels exist.
    """
    reviewed_count = sum(
        1
        for row in rows
        if normalize_text(
            row.get("localization_quality")
        )
        in QUALITY_LEVELS
    )

    if reviewed_count == 0:
        raise ValueError(
            "The audit CSV does not contain completed manual "
            "review labels. Use the reviewed CSV with "
            "localization_quality and error_type filled in."
        )


def calculate_rate(
    numerator: int,
    denominator: int,
) -> float:
    """Calculate percentage rate.

    Args:
        numerator: Event count.
        denominator: Total count.

    Returns:
        Percentage from 0 to 100.
    """
    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
        * 100.0
    )


def analyze_by_target(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Summarize localization quality for each target.

    Args:
        rows: Reviewed audit rows.

    Returns:
        Per-target summary records.
    """
    target_rows: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        target = normalize_text(
            row.get("target")
        )

        if target:
            target_rows[target].append(row)

    summary = []

    target_order = list(
        EXPECTED_TARGETS
    )

    extra_targets = sorted(
        set(target_rows.keys())
        - set(target_order)
    )

    target_order.extend(
        extra_targets
    )

    for target in target_order:
        current_rows = target_rows.get(
            target,
            [],
        )

        if not current_rows:
            continue

        quality_counter = Counter(
            normalize_text(
                row.get(
                    "localization_quality"
                )
            )
            for row in current_rows
        )

        total = len(current_rows)

        correct = quality_counter[
            "correct"
        ]

        coarse = quality_counter[
            "coarse"
        ]

        wrong = quality_counter[
            "wrong"
        ]

        summary.append(
            {
                "target": target,
                "total": total,
                "correct": correct,
                "coarse": coarse,
                "wrong": wrong,
                "correct_rate_percent": round(
                    calculate_rate(
                        correct,
                        total,
                    ),
                    2,
                ),
                "coarse_rate_percent": round(
                    calculate_rate(
                        coarse,
                        total,
                    ),
                    2,
                ),
                "wrong_rate_percent": round(
                    calculate_rate(
                        wrong,
                        total,
                    ),
                    2,
                ),
            }
        )

    return summary


def analyze_error_types(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Summarize manual error types.

    Args:
        rows: Reviewed audit rows.

    Returns:
        Error-type summary records.
    """
    error_counter: Counter[str] = Counter()

    target_counter: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    for row in rows:
        quality = normalize_text(
            row.get("localization_quality")
        )

        if quality == "correct":
            continue

        error_type = normalize_text(
            row.get("error_type")
        )

        if not error_type:
            error_type = "unspecified"

        target = normalize_text(
            row.get("target")
        )

        error_counter[
            error_type
        ] += 1

        target_counter[
            error_type
        ][target] += 1

    total_errors = sum(
        error_counter.values()
    )

    summary = []

    for (
        error_type,
        count,
    ) in error_counter.most_common():
        summary.append(
            {
                "error_type": error_type,
                "count": count,
                "error_rate_percent": round(
                    calculate_rate(
                        count,
                        total_errors,
                    ),
                    2,
                ),
                "sleeve": target_counter[
                    error_type
                ]["sleeve"],
                "collar": target_counter[
                    error_type
                ]["collar"],
                "button": target_counter[
                    error_type
                ]["button"],
                "zipper": target_counter[
                    error_type
                ]["zipper"],
            }
        )

    return summary


def classify_target_scale(
    target: str,
) -> str:
    """Assign target to a coarse object-scale group.

    Args:
        target: Fine-grained target.

    Returns:
        Scale-group name.
    """
    if target in SMALL_OBJECT_TARGETS:
        return "small_object"

    if target in LARGER_PART_TARGETS:
        return "larger_part"

    return "other"


def analyze_by_scale_group(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Compare small objects with larger garment parts.

    Args:
        rows: Reviewed audit rows.

    Returns:
        Per-scale-group summary records.
    """
    grouped_rows: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        target = normalize_text(
            row.get("target")
        )

        scale_group = (
            classify_target_scale(
                target
            )
        )

        grouped_rows[
            scale_group
        ].append(row)

    summary = []

    for scale_group in (
        "larger_part",
        "small_object",
        "other",
    ):
        current_rows = grouped_rows.get(
            scale_group,
            [],
        )

        if not current_rows:
            continue

        quality_counter = Counter(
            normalize_text(
                row.get(
                    "localization_quality"
                )
            )
            for row in current_rows
        )

        error_counter = Counter(
            normalize_text(
                row.get(
                    "error_type"
                )
            )
            or "unspecified"
            for row in current_rows
            if normalize_text(
                row.get(
                    "localization_quality"
                )
            )
            != "correct"
        )

        total = len(current_rows)

        correct = quality_counter[
            "correct"
        ]

        coarse = quality_counter[
            "coarse"
        ]

        wrong = quality_counter[
            "wrong"
        ]

        top_error = ""

        if error_counter:
            top_error = (
                error_counter
                .most_common(1)[0][0]
            )

        summary.append(
            {
                "scale_group": (
                    scale_group
                ),
                "total": total,
                "correct": correct,
                "coarse": coarse,
                "wrong": wrong,
                "correct_rate_percent": round(
                    calculate_rate(
                        correct,
                        total,
                    ),
                    2,
                ),
                "coarse_rate_percent": round(
                    calculate_rate(
                        coarse,
                        total,
                    ),
                    2,
                ),
                "wrong_rate_percent": round(
                    calculate_rate(
                        wrong,
                        total,
                    ),
                    2,
                ),
                "most_common_error": (
                    top_error
                ),
            }
        )

    return summary


def build_failure_cases(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Build detailed failure-case records.

    Args:
        rows: Reviewed audit rows.

    Returns:
        Non-correct audit cases.
    """
    failures = []

    for row in rows:
        quality = normalize_text(
            row.get(
                "localization_quality"
            )
        )

        if quality == "correct":
            continue

        target = normalize_text(
            row.get("target")
        )

        failures.append(
            {
                "audit_id": (
                    row.get(
                        "audit_id",
                        "",
                    )
                ),
                "target": target,
                "scale_group": (
                    classify_target_scale(
                        target
                    )
                ),
                "prompt_type": (
                    row.get(
                        "prompt_type",
                        "",
                    )
                ),
                "prompt": (
                    row.get(
                        "prompt",
                        "",
                    )
                ),
                "image_name": (
                    row.get(
                        "image_name",
                        "",
                    )
                ),
                "localization_quality": (
                    quality
                ),
                "error_type": (
                    normalize_text(
                        row.get(
                            "error_type"
                        )
                    )
                    or "unspecified"
                ),
                "score": (
                    row.get(
                        "score",
                        "",
                    )
                ),
                "bbox_area_ratio": (
                    row.get(
                        "bbox_area_ratio",
                        "",
                    )
                ),
                "notes": (
                    row.get(
                        "notes",
                        "",
                    )
                ),
                "visualization_path": (
                    row.get(
                        "visualization_path",
                        "",
                    )
                ),
            }
        )

    return failures


def save_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Save records to CSV.

    Args:
        output_path: Destination CSV path.
        rows: Output records.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        LOGGER.warning(
            "No rows to save: %s",
            output_path,
        )
        return

    fieldnames = list(
        rows[0].keys()
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info(
        "Saved: %s",
        output_path,
    )


def get_target_record(
    target_summary: list[dict[str, Any]],
    target: str,
) -> dict[str, Any] | None:
    """Find one target summary.

    Args:
        target_summary: Per-target records.
        target: Requested target.

    Returns:
        Matching target record if present.
    """
    for record in target_summary:
        if record["target"] == target:
            return record

    return None


def get_scale_record(
    scale_summary: list[dict[str, Any]],
    scale_group: str,
) -> dict[str, Any] | None:
    """Find one scale-group summary.

    Args:
        scale_summary: Per-scale-group records.
        scale_group: Requested group.

    Returns:
        Matching record if present.
    """
    for record in scale_summary:
        if (
            record["scale_group"]
            == scale_group
        ):
            return record

    return None


def format_target_table(
    target_summary: list[dict[str, Any]],
) -> str:
    """Build Markdown target summary table.

    Args:
        target_summary: Per-target summary.

    Returns:
        Markdown table.
    """
    lines = [
        "| Target | N | Correct | Coarse | Wrong | Wrong Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in target_summary:
        lines.append(
            "| "
            f'{row["target"]} | '
            f'{row["total"]} | '
            f'{row["correct"]} | '
            f'{row["coarse"]} | '
            f'{row["wrong"]} | '
            f'{row["wrong_rate_percent"]:.2f}% |'
        )

    return "\n".join(lines)


def format_error_table(
    error_summary: list[dict[str, Any]],
) -> str:
    """Build Markdown error-type table.

    Args:
        error_summary: Error summary rows.

    Returns:
        Markdown table.
    """
    lines = [
        (
            "| Error Type | Count | Share | "
            "Sleeve | Collar | Button | Zipper |"
        ),
        (
            "| --- | ---: | ---: | ---: | "
            "---: | ---: | ---: |"
        ),
    ]

    for row in error_summary:
        lines.append(
            "| "
            f'{row["error_type"]} | '
            f'{row["count"]} | '
            f'{row["error_rate_percent"]:.2f}% | '
            f'{row["sleeve"]} | '
            f'{row["collar"]} | '
            f'{row["button"]} | '
            f'{row["zipper"]} |'
        )

    return "\n".join(lines)


def build_summary_markdown(
    audit_file: Path,
    total_rows: int,
    target_summary: list[dict[str, Any]],
    error_summary: list[dict[str, Any]],
    scale_summary: list[dict[str, Any]],
) -> str:
    """Build the Markdown experiment summary.

    Args:
        audit_file: Source audit CSV.
        total_rows: Total reviewed cases.
        target_summary: Per-target results.
        error_summary: Error distribution.
        scale_summary: Scale-group comparison.

    Returns:
        Markdown report text.
    """
    small_record = get_scale_record(
        scale_summary,
        "small_object",
    )

    larger_record = get_scale_record(
        scale_summary,
        "larger_part",
    )

    button_record = get_target_record(
        target_summary,
        "button",
    )

    zipper_record = get_target_record(
        target_summary,
        "zipper",
    )

    sleeve_record = get_target_record(
        target_summary,
        "sleeve",
    )

    collar_record = get_target_record(
        target_summary,
        "collar",
    )

    lines = [
        "# Grounding DINO Failure Pattern Analysis",
        "",
        "## 1. Analysis Setup",
        "",
        f"- Source audit: `{audit_file}`",
        f"- Reviewed cases: {total_rows}",
        (
            "- Larger-part group: `sleeve`, `collar`"
        ),
        (
            "- Small-object group: `button`, `zipper`"
        ),
        "",
        (
            "> Important: this 40-case audit was sampled "
            "from emitted predictions. The statistics below "
            "describe localization quality among audited "
            "prediction cases and must not be interpreted "
            "as full-dataset localization accuracy."
        ),
        "",
        "## 2. Localization Quality by Target",
        "",
        format_target_table(
            target_summary
        ),
        "",
        "## 3. Failure Type Distribution",
        "",
        format_error_table(
            error_summary
        ),
        "",
        "## 4. Small-Object Analysis",
        "",
    ]

    if (
        small_record is not None
        and larger_record is not None
    ):
        lines.extend(
            [
                (
                    "- Small-object group wrong rate: "
                    f'{small_record["wrong_rate_percent"]:.2f}% '
                    f'({small_record["wrong"]}/'
                    f'{small_record["total"]}).'
                ),
                (
                    "- Larger-part group wrong rate: "
                    f'{larger_record["wrong_rate_percent"]:.2f}% '
                    f'({larger_record["wrong"]}/'
                    f'{larger_record["total"]}).'
                ),
                (
                    "- Most common small-object error: "
                    f'`{small_record["most_common_error"]}`.'
                ),
                (
                    "- Most common larger-part error: "
                    f'`{larger_record["most_common_error"]}`.'
                ),
                "",
            ]
        )

    lines.append(
        "### Target-specific observations"
    )
    lines.append("")

    if sleeve_record is not None:
        lines.append(
            "- `sleeve`: "
            f'{sleeve_record["correct"]} correct, '
            f'{sleeve_record["coarse"]} coarse, '
            f'{sleeve_record["wrong"]} wrong.'
        )

    if collar_record is not None:
        lines.append(
            "- `collar`: "
            f'{collar_record["correct"]} correct, '
            f'{collar_record["coarse"]} coarse, '
            f'{collar_record["wrong"]} wrong.'
        )

    if button_record is not None:
        lines.append(
            "- `button`: "
            f'{button_record["correct"]} correct, '
            f'{button_record["coarse"]} coarse, '
            f'{button_record["wrong"]} wrong.'
        )

    if zipper_record is not None:
        lines.append(
            "- `zipper`: "
            f'{zipper_record["correct"]} correct, '
            f'{zipper_record["coarse"]} coarse, '
            f'{zipper_record["wrong"]} wrong.'
        )

    lines.extend(
        [
            "",
            "## 5. Interpretation",
            "",
            (
                "The current failure distribution should be interpreted "
                "together with the ROI-conditioned diagnostic experiment."
            ),
            "",
            (
                "For larger garment parts, failures frequently include "
                "whole-garment or coarse localization, indicating that "
                "Grounding DINO may recognize semantic association with "
                "the garment without precisely isolating the requested "
                "local region."
            ),
            "",
            (
                "For small objects such as buttons and zippers, failures "
                "should be examined for target absence, non-clothing-object "
                "confusion, wrong-part activation, and low visual detail. "
                "These patterns are consistent with the hypothesis that "
                "small targets are more sensitive to limited pixel "
                "information and visual ambiguity."
            ),
            "",
            (
                "The ROI-conditioned experiment should therefore be treated "
                "as a search-space reduction diagnostic rather than proof "
                "that cropping alone solves fine-grained localization."
            ),
            "",
            "## 6. Recommended Next Step",
            "",
            (
                "1. Preserve the current full-image baseline at "
                "`threshold = 0.3`."
            ),
            (
                "2. Preserve ROI `threshold = 0.2` only as a diagnostic "
                "condition, because increased prediction coverage does not "
                "necessarily imply improved localization."
            ),
            (
                "3. Build a target-balanced evaluation set for `button` "
                "and `zipper` with verified positive examples."
            ),
            (
                "4. Separate `target absent` cases from true localization "
                "failures before reporting final localization accuracy."
            ),
            (
                "5. Evaluate whether a stronger fine-grained localization "
                "or segmentation refinement method is required after the "
                "Grounding DINO baseline."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def save_markdown(
    output_path: Path,
    content: str,
) -> None:
    """Save Markdown content.

    Args:
        output_path: Destination Markdown path.
        content: Markdown text.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        content,
        encoding="utf-8",
    )

    LOGGER.info(
        "Saved: %s",
        output_path,
    )


def print_console_summary(
    target_summary: list[dict[str, Any]],
    scale_summary: list[dict[str, Any]],
) -> None:
    """Print a concise analysis summary.

    Args:
        target_summary: Per-target summary.
        scale_summary: Scale-group summary.
    """
    print()
    print(
        "=== Localization Quality by Target ==="
    )

    for row in target_summary:
        print(
            f'{row["target"]:>8}: '
            f'N={row["total"]}, '
            f'correct={row["correct"]}, '
            f'coarse={row["coarse"]}, '
            f'wrong={row["wrong"]}, '
            f'wrong_rate='
            f'{row["wrong_rate_percent"]:.1f}%'
        )

    print()
    print(
        "=== Object Scale Comparison ==="
    )

    for row in scale_summary:
        print(
            f'{row["scale_group"]:>12}: '
            f'N={row["total"]}, '
            f'correct={row["correct"]}, '
            f'coarse={row["coarse"]}, '
            f'wrong={row["wrong"]}, '
            f'wrong_rate='
            f'{row["wrong_rate_percent"]:.1f}%, '
            f'top_error='
            f'{row["most_common_error"]}'
        )


def main() -> None:
    """Run Grounding DINO failure-pattern analysis."""
    configure_logging()

    args = parse_args()

    audit_file = resolve_audit_file(
        args.audit_file
    )

    LOGGER.info(
        "Reading audit file: %s",
        audit_file,
    )

    rows = read_audit_rows(
        audit_file
    )

    validate_reviewed_rows(
        rows
    )

    reviewed_rows = [
        row
        for row in rows
        if normalize_text(
            row.get(
                "localization_quality"
            )
        )
        in QUALITY_LEVELS
    ]

    LOGGER.info(
        "Reviewed cases: %d",
        len(reviewed_rows),
    )

    target_summary = (
        analyze_by_target(
            reviewed_rows
        )
    )

    error_summary = (
        analyze_error_types(
            reviewed_rows
        )
    )

    scale_summary = (
        analyze_by_scale_group(
            reviewed_rows
        )
    )

    failure_cases = (
        build_failure_cases(
            reviewed_rows
        )
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_csv(
        args.output_dir
        / "failure_by_target.csv",
        target_summary,
    )

    save_csv(
        args.output_dir
        / "failure_by_error_type.csv",
        error_summary,
    )

    save_csv(
        args.output_dir
        / "failure_by_scale_group.csv",
        scale_summary,
    )

    save_csv(
        args.output_dir
        / "failure_cases.csv",
        failure_cases,
    )

    markdown = (
        build_summary_markdown(
            audit_file=audit_file,
            total_rows=len(
                reviewed_rows
            ),
            target_summary=target_summary,
            error_summary=error_summary,
            scale_summary=scale_summary,
        )
    )

    save_markdown(
        args.output_dir
        / "failure_analysis_summary.md",
        markdown,
    )

    print_console_summary(
        target_summary,
        scale_summary,
    )

    print()
    print(
        "Failure analysis completed."
    )
    print(
        f"Reports: {args.output_dir}"
    )


if __name__ == "__main__":
    main()