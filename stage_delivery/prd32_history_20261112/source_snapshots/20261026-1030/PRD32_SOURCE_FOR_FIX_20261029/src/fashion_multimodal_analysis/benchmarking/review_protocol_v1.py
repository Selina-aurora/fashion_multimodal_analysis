"""Prepare review records without converting historical observations into new GT.

This CPU-only tool copies explicit identity columns, preserves each legacy row,
and leaves the revised human decisions pending. It never opens images, models,
sealed manifests or model outputs. Run the new review before computing quality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "prd31_review_v1_draft_20261026"
REGIONS = (
    "collar",
    "cuff",
    "hem",
    "pocket",
    "shoulder",
    "waist",
    "pattern",
    "decoration",
)
FIELDS = (
    "image_id",
    "instance_id",
    "region",
    "part_id",
    "source_image_path",
    "target_status",
    "presence",
    "visibility",
    "geometry_evaluable",
    "prediction_status",
    "localization",
    "reference_geometry_path",
    "reference_origin",
    "reviewer",
    "reviewed_at",
    "evidence_note",
    "protocol_version",
    "review_status",
    "priority_reason",
    "legacy_record_json",
)
ENUMS = {
    "target_status": {"pending", "correct", "wrong", "uncertain"},
    "presence": {"pending", "present", "absent", "not_applicable", "uncertain"},
    "visibility": {
        "pending",
        "visible",
        "partial",
        "occluded",
        "out_of_frame",
        "uncertain",
        "not_applicable",
    },
    "geometry_evaluable": {"pending", "yes", "no"},
    "prediction_status": {
        "pending",
        "detected",
        "missed",
        "empty_mask",
        "not_applicable",
    },
    "localization": {
        "pending",
        "reasonable",
        "wrong_extent",
        "wrong_object",
        "missed",
        "false_positive",
        "correct_rejection",
        "not_evaluable",
    },
    "reference_origin": {"pending", "human_independent", "model_adopted", "none"},
    "review_status": {"pending", "reviewed", "needs_adjudication"},
}


def new_record(image_id: str, instance_id: str, region: str) -> dict[str, str]:
    """Create a pending record with a stable, explicitly supplied identity."""
    if not image_id or not instance_id or region not in REGIONS:
        raise ValueError("Non-empty identities and a supported region are required")
    row = dict.fromkeys(FIELDS, "")
    row.update({key: "pending" for key in ENUMS})
    row.update(
        image_id=image_id,
        instance_id=instance_id,
        region=region,
        protocol_version=PROTOCOL_VERSION,
    )
    return row


def validate_record(row: dict[str, str]) -> list[str]:
    """Return decision conflicts; do not infer any missing human decision.

    A row may remain pending. Reviewed rows must explain their evidence and
    author. Geometry eligibility is separate from qualitative review status.
    """
    errors = []
    if not row.get("image_id") or not row.get("instance_id"):
        errors.append("missing_identity")
    if row.get("region") not in REGIONS:
        errors.append("unsupported_region")
    if row.get("protocol_version") != PROTOCOL_VERSION:
        errors.append("wrong_protocol_version")
    for key, choices in ENUMS.items():
        if row.get(key) not in choices:
            errors.append(f"invalid_{key}")
    if row.get("review_status") == "reviewed":
        for key in ENUMS:
            if row.get(key) == "pending":
                errors.append(f"reviewed_but_pending_{key}")
        for key in ("reviewer", "reviewed_at", "evidence_note"):
            if not row.get(key, "").strip():
                errors.append(f"missing_{key}")
        if (
            row.get("target_status") == "wrong"
            and row.get("localization") != "not_evaluable"
        ):
            errors.append("wrong_target_cannot_score_localization")
        if (
            row.get("target_status") == "uncertain"
            and row.get("localization") != "not_evaluable"
        ):
            errors.append("uncertain_target_cannot_score_localization")
        if (
            row.get("presence") == "uncertain"
            and row.get("localization") != "not_evaluable"
        ):
            errors.append("uncertain_presence_cannot_score_localization")
        if (
            row.get("visibility") in {"occluded", "out_of_frame", "uncertain"}
            and row.get("localization") != "not_evaluable"
        ):
            errors.append("unobservable_region_cannot_score_localization")
        if row.get("presence") in {"absent", "not_applicable"}:
            expected = {
                "detected": "false_positive",
                "empty_mask": "not_evaluable",
                "missed": "correct_rejection",
                "not_applicable": "correct_rejection",
            }.get(row.get("prediction_status", ""))
            if (
                row.get("target_status") == "correct"
                and row.get("localization") != expected
            ):
                errors.append("negative_case_requires_explicit_decision")
        if row.get("localization") in {"false_positive", "correct_rejection"} and not (
            row.get("target_status") == "correct"
            and row.get("presence") in {"absent", "not_applicable"}
        ):
            errors.append("negative_grade_requires_confirmed_negative")
        if row.get("localization") == "missed" and not (
            row.get("target_status") == "correct"
            and row.get("presence") == "present"
            and row.get("visibility") in {"visible", "partial"}
            and row.get("prediction_status") in {"missed", "empty_mask"}
        ):
            errors.append("missed_requires_observable_positive")
        if row.get("localization") in {
            "reasonable",
            "wrong_extent",
            "wrong_object",
        } and not (
            row.get("target_status") == "correct"
            and row.get("presence") == "present"
            and row.get("visibility") in {"visible", "partial"}
            and row.get("prediction_status") == "detected"
        ):
            errors.append("localization_grade_requires_detected_positive")
    if row.get("geometry_evaluable") == "yes":
        if not (
            row.get("target_status") == "correct"
            and row.get("presence") == "present"
            and row.get("visibility") in {"visible", "partial"}
            and row.get("reference_origin") == "human_independent"
            and row.get("reference_geometry_path")
        ):
            errors.append("geometry_requires_independent_observable_reference")
    return errors


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Summarize decisions and eligibility, never publish a geometry accuracy.

    Reference files are not opened by this preparation tool, so even a declared
    reference path is not evidence that geometry has been verified.
    """
    identities = [
        tuple(
            row.get(key, "") for key in ("image_id", "instance_id", "region", "part_id")
        )
        for row in rows
    ]
    counts = Counter(identities)
    row_errors = [validate_record(row) for row in rows]
    for index, identity in enumerate(identities):
        if counts[identity] > 1:
            row_errors[index].append("duplicate_region_identity")
    conflicts = [
        {"row": index + 2, "errors": errors}
        for index, errors in enumerate(row_errors)
        if errors
    ]
    valid = [row for index, row in enumerate(rows) if not row_errors[index]]
    reviewed = [row for row in valid if row["review_status"] == "reviewed"]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_status": "draft_for_calibration",
        "records": len(rows),
        "reviewed_records": len(reviewed),
        "pending_records": sum(row.get("review_status") == "pending" for row in rows),
        "decision_conflicts": conflicts,
        "reviewed_localization_counts": dict(
            Counter(row["localization"] for row in reviewed)
        ),
        "declared_geometry_candidates": sum(
            row["geometry_evaluable"] == "yes" for row in reviewed
        ),
        "reference_geometry_verified": False,
        "formal_accuracy": None,
        "model_inference_performed": False,
        "sealed_inputs_opened": False,
    }


def prepare_legacy(
    source: Path, image_column: str, instance_column: str, region_column: str
) -> list[dict[str, str]]:
    """Copy row identities and the raw legacy fields, requiring explicit mappings.

    Raises:
        ValueError: Headers, identities or regions are invalid or duplicated.
    """
    rows = []
    keys = set()
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {image_column, instance_column, region_column}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Required columns {sorted(required)}; found {reader.fieldnames}"
            )
        for index, legacy in enumerate(reader, start=2):
            if None in legacy or any(value is None for value in legacy.values()):
                raise ValueError(f"Malformed CSV row {index}")
            row = new_record(
                legacy[image_column].strip(),
                legacy[instance_column].strip(),
                legacy[region_column].strip(),
            )
            key = (row["image_id"], row["instance_id"], row["region"])
            if key in keys:
                raise ValueError(
                    f"Duplicate region identity at row {index}; split a multi-part table explicitly"
                )
            keys.add(key)
            row["legacy_record_json"] = json.dumps(
                legacy, ensure_ascii=False, sort_keys=True
            )
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a UTF-8 BOM table that opens correctly in spreadsheet editors."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def prepare_json_review(source: Path) -> list[dict[str, str]]:
    """Import selected legacy JSON reviews, excluding unselected pending records.

    Only the known 10.22 schema is accepted. Original target, review and model
    record fields stay inside legacy_record_json; no new judgment is inferred.

    Raises:
        ValueError: The schema, identities or selected record links are invalid.
    """
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    if data.get("schema") != "prd31_new20_visual_review_v1":
        raise ValueError("Unsupported legacy JSON schema")
    records = {}
    for record in data["records"]:
        key = "/".join(record[field] for field in ("image_id", "instance_id", "region"))
        if key in records:
            raise ValueError(f"Duplicate model record identity: {key}")
        records[key] = record
    rows = []
    for key, review in sorted(data["reviews"].items()):
        parts = key.split("/")
        if len(parts) != 3 or key not in records:
            raise ValueError(f"Review does not link to a unique model record: {key}")
        target_key = "/".join(parts[:2])
        if target_key not in data["targets"]:
            raise ValueError(f"Missing selected-target review: {target_key}")
        row = new_record(*parts)
        row["legacy_record_json"] = json.dumps(
            {
                "target": data["targets"][target_key],
                "review": review,
                "record": records[key],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        rows.append(row)
    return rows


def priority_reason(row: dict[str, str]) -> str:
    """Flag historical ambiguities for review, without declaring them errors."""
    legacy = json.loads(row.get("legacy_record_json") or "{}")
    target_data = legacy.get("target", {})
    target = target_data.get("verdict") if isinstance(target_data, dict) else None
    review = legacy.get("review", {})
    if not isinstance(review, dict):
        review = {}
    if target == "wrong_object":
        return "p0_upstream_target_review"
    if review.get("presence") == "occluded" and review.get("verdict") == "wrong":
        return "p1_occlusion_and_localization_mixed"
    if review.get("presence") == "absent" and review.get("verdict") == "wrong":
        return "p1_absence_and_wrong_extent_mixed"
    return "p2_other_definition_review"


def main(argv: list[str] | None = None) -> int:
    """Prepare an empty template or migrate one explicitly selected development CSV.

    Source review data must be a development batch. Never pass a sealed test
    manifest. Existing output directories are rejected to protect old results.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-csv", type=Path)
    parser.add_argument("--source-json", type=Path)
    parser.add_argument("--image-column")
    parser.add_argument("--instance-column")
    parser.add_argument("--region-column")
    parser.add_argument("--acknowledge-development-only", action="store_true")
    parser.add_argument("--validate-csv", type=Path)
    args = parser.parse_args(argv)
    if (
        sum(
            bool(value)
            for value in (args.source_csv, args.source_json, args.validate_csv)
        )
        > 1
    ):
        parser.error("Select preparation or validation, not both")
    if args.source_json and not args.acknowledge_development_only:
        parser.error("Legacy JSON requires development acknowledgement")
    if args.source_csv and not (
        args.acknowledge_development_only
        and args.image_column
        and args.instance_column
        and args.region_column
    ):
        parser.error(
            "Legacy input requires development acknowledgement and all three column names"
        )
    source = args.source_csv or args.source_json or args.validate_csv
    if source and ("holdout" in str(source).lower() or "sealed" in str(source).lower()):
        parser.error("Sealed or holdout inputs are outside this development tool")
    if args.source_csv:
        rows = prepare_legacy(
            args.source_csv, args.image_column, args.instance_column, args.region_column
        )
    elif args.source_json:
        rows = prepare_json_review(args.source_json)
    elif args.validate_csv:
        with args.validate_csv.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = []
    report = summarize(rows)
    if source:
        report["source_file"] = str(source)
        report["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.validate_csv:
        for row in rows:
            row["priority_reason"] = priority_reason(row)
        write_csv(args.output / "review_records.csv", rows)
        priority = [
            row for row in rows if priority_reason(row).startswith(("p0_", "p1_"))
        ]
        priority.sort(
            key=lambda row: (priority_reason(row), row["image_id"], row["region"])
        )
        write_csv(args.output / "review_priority.csv", priority)
        report["priority_records"] = len(priority)
        report["legacy_priority_reasons"] = dict(
            Counter(priority_reason(row) for row in rows)
        )
    (args.output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["decision_conflicts"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
