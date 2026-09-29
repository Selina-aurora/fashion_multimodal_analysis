"""Summarize a completed PRD 3.1.3 manual audit CSV.

Fill these columns in manual_audit_template.csv:
    color_correct
    pattern_correct
    sleeve_length_correct
    neckline_correct

Accepted values:
    y / yes / 1 / true
    n / no  / 0 / false

Then run:
    python scripts/summarize_attribute_manual_audit.py \
        --audit reports/prd_attribute_extraction/batch_20/manual_audit_template.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ATTRS = [
    "color",
    "pattern",
    "sleeve_length",
    "neckline",
]


def resolve_file(raw: str) -> Path:
    p = Path(raw).expanduser()

    if p.is_absolute() and p.is_file():
        return p.resolve()

    cwd_candidate = (
        Path.cwd()
        / p
    ).resolve()

    if cwd_candidate.is_file():
        return cwd_candidate

    project_candidate = (
        PROJECT_ROOT
        / p
    ).resolve()

    if project_candidate.is_file():
        return project_candidate

    raise FileNotFoundError(
        raw
    )


def parse_bool(raw: str):
    value = (
        str(raw)
        .strip()
        .lower()
    )

    if value in {
        "y",
        "yes",
        "1",
        "true",
    }:
        return True

    if value in {
        "n",
        "no",
        "0",
        "false",
    }:
        return False

    return None


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audit",
        required=True,
    )

    args = parser.parse_args()

    audit_path = resolve_file(
        args.audit
    )

    with audit_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    lines = [
        "# PRD 3.1.3 Manual Audit Summary",
        "",
        f"- cases: {len(rows)}",
        "",
        "| Attribute | Reviewed | Correct | Accuracy |",
        "|---|---:|---:|---:|",
    ]

    total_reviewed = 0
    total_correct = 0

    for attr in ATTRS:
        reviewed = 0
        correct = 0

        field = (
            f"{attr}_correct"
        )

        for row in rows:
            value = parse_bool(
                row.get(
                    field,
                    "",
                )
            )

            if value is None:
                continue

            reviewed += 1

            if value:
                correct += 1

        accuracy = (
            correct / reviewed
            if reviewed
            else 0.0
        )

        total_reviewed += reviewed
        total_correct += correct

        lines.append(
            f"| {attr} | "
            f"{reviewed} | "
            f"{correct} | "
            f"{accuracy:.1%} |"
        )

    overall_accuracy = (
        total_correct
        / total_reviewed
        if total_reviewed
        else 0.0
    )

    lines.extend(
        [
            "",
            f"- reviewed attribute judgments: {total_reviewed}",
            f"- correct attribute judgments: {total_correct}",
            f"- overall micro accuracy: {overall_accuracy:.1%}",
            "",
            "Note: this is a small manual pilot, not a final PRD acceptance result.",
        ]
    )

    summary_path = (
        audit_path.parent
        / "manual_audit_summary.md"
    )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        "\n".join(lines)
    )

    print(
        f"\nsaved: {summary_path}"
    )


if __name__ == "__main__":
    main()
