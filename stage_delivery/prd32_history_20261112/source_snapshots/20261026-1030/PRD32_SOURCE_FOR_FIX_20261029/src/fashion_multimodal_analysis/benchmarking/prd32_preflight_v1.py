"""Check draft PRD3.2 request identities and image assets without loading models.

Only the public request file is read. AI draft answer notes are kept separate
and never enter the model-input path. A passing preflight is not a QA score,
human-reference confirmation, or proof that any model weights are ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "case_id",
    "task",
    "split",
    "image_id",
    "image_path",
    "image_sha256",
    "source_size",
    "target_instance_id",
    "target_bbox_xyxy",
    "target_bbox_origin",
    "query",
    "output_format",
    "attributes_input",
}
OPTIONAL_FIELDS = {"style", "length_chars"}


def request_errors(request: dict[str, Any]) -> list[str]:
    """Check public inputs, target coordinates and draft evaluation boundaries."""
    errors = []
    missing = REQUIRED_FIELDS - request.keys()
    extra = request.keys() - REQUIRED_FIELDS - OPTIONAL_FIELDS
    if missing:
        errors.append(f"missing_fields:{','.join(sorted(missing))}")
    if extra:
        errors.append(f"unsupported_or_reference_fields:{','.join(sorted(extra))}")
    if request.get("split") != "development":
        errors.append("only_development_requests_allowed")
    if request.get("task") not in {"vqa", "tags", "copy"}:
        errors.append("unsupported_task")
    for key in ("case_id", "image_id", "image_path", "target_instance_id", "query"):
        if not isinstance(request.get(key), str) or not request[key].strip():
            errors.append(f"invalid_{key}")
    path = str(request.get("image_path", "")).lower()
    if "holdout" in path or "sealed" in path:
        errors.append("protected_path_not_allowed")
    digest = request.get("image_sha256", "")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        errors.append("invalid_image_sha256")
    size = request.get("source_size")
    box = request.get("target_bbox_xyxy")
    valid_size = (
        isinstance(size, list)
        and len(size) == 2
        and all(type(x) is int and x > 0 for x in size)
    )
    valid_box = (
        isinstance(box, list)
        and len(box) == 4
        and all(type(x) in {int, float} and math.isfinite(x) for x in box)
    )
    if (
        not valid_size
        or not valid_box
        or not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1])
    ):
        errors.append("invalid_target_box_or_source_size")
    if request.get("target_bbox_origin") != "prd31_frozen_prediction":
        errors.append("target_box_is_not_declared_frozen_prediction")
    if request.get("attributes_input") is not None:
        errors.append("this_batch_is_raw_image_baseline")
    expected_format = "json" if request.get("task") == "tags" else "text"
    if request.get("output_format") != expected_format:
        errors.append("wrong_output_format")
    if request.get("task") == "copy":
        length = request.get("length_chars")
        if not isinstance(request.get("style"), str) or not request["style"].strip():
            errors.append("missing_copy_style")
        if (
            not isinstance(length, list)
            or len(length) != 2
            or not all(type(x) is int for x in length)
            or not (0 < length[0] <= length[1])
        ):
            errors.append("invalid_copy_length")
    return errors


def inspect_requests(
    requests: list[dict[str, Any]], check_assets: bool
) -> dict[str, Any]:
    """Inspect unique declared image paths and content hashes, with no inference.

    Args:
        requests: Public development requests, excluding reference answers.
        check_assets: Whether to read and hash the explicitly listed image files.

    Returns:
        Structured preflight findings; no predicted answers or accuracy fields.
    """
    errors = []
    seen = set()
    assets: dict[str, dict[str, Any]] = {}
    for row_number, request in enumerate(requests, start=1):
        problems = request_errors(request)
        case_id = request.get("case_id", f"row_{row_number}")
        if not isinstance(case_id, str):
            case_id = f"row_{row_number}"
        if case_id in seen:
            problems.append("duplicate_case_id")
        seen.add(case_id)
        if problems:
            errors.append({"case_id": case_id, "errors": problems})
            continue
        path = request["image_path"]
        digest = request["image_sha256"]
        if path in assets:
            if assets[path]["expected_sha256"] != digest:
                errors.append(
                    {"case_id": case_id, "errors": ["conflicting_image_identity"]}
                )
            continue
        item = {
            "image_id": request["image_id"],
            "path": path,
            "expected_sha256": digest,
            "asset_check": "not_run",
        }
        if check_assets:
            file = Path(path)
            if not file.is_file():
                item["asset_check"] = "missing"
                errors.append({"case_id": case_id, "errors": ["image_file_missing"]})
            else:
                try:
                    actual = hashlib.sha256(file.read_bytes()).hexdigest()
                    item["actual_sha256"] = actual
                    item["asset_check"] = (
                        "passed" if actual == digest else "hash_mismatch"
                    )
                    if actual != digest:
                        errors.append(
                            {"case_id": case_id, "errors": ["image_hash_mismatch"]}
                        )
                except OSError as exc:
                    item["asset_check"] = "read_failed"
                    errors.append(
                        {
                            "case_id": case_id,
                            "errors": [f"image_read_failed:{type(exc).__name__}"],
                        }
                    )
        assets[path] = item
    return {
        "schema": "prd32_scope_preflight_v1",
        "status": (
            "PRD32_REQUESTS_AND_ASSETS_PASS"
            if check_assets and not errors
            else (
                "PRD32_REQUEST_SCHEMA_PASS"
                if not errors
                else "PRD32_PREFLIGHT_NEEDS_ATTENTION"
            )
        ),
        "request_count": len(requests),
        "task_counts": dict(Counter(row.get("task", "unknown") for row in requests)),
        "unique_declared_images": len(assets),
        "asset_checks_requested": check_assets,
        "image_assets": list(assets.values()),
        "errors": errors,
        "model_selected": False,
        "model_inference_performed": False,
        "reference_answers_confirmed": False,
        "accuracy_measured": False,
    }


def cache_metadata(project: Path) -> list[dict[str, Any]]:
    """List bounded local model-config metadata, without reading weights or tokens."""
    cache = project / "hf_cache/hub"
    result = []
    for file in sorted(cache.glob("models--*/snapshots/*/config.json"))[:1000]:
        if not file.resolve().is_relative_to(cache.resolve()):
            continue
        try:
            if file.stat().st_size > 2_000_000:
                continue
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        result.append(
            {
                "config_path": str(file),
                "model_type": data.get("model_type"),
                "architectures": data.get("architectures"),
                "weights_verified": False,
            }
        )
    return result


def main(argv: list[str] | None = None) -> int:
    """Write a fresh CPU preflight report for this development request batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema-only", action="store_true")
    args = parser.parse_args(argv)
    project = args.project.resolve(strict=True)
    source = args.requests or project / "configs/prd32_scope_20261026/requests.jsonl"
    if (
        "reference" in source.name.lower()
        or "holdout" in str(source).lower()
        or "sealed" in str(source).lower()
    ):
        parser.error(
            "Select public development requests, not references or protected data"
        )
    requests = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not requests or any(not isinstance(row, dict) for row in requests):
        parser.error("Expected non-empty JSONL request objects")
    report = inspect_requests(requests, check_assets=not args.schema_only)
    report["public_request_file"] = str(source)
    report["public_request_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    report["local_model_config_metadata"] = cache_metadata(project)
    report["cache_metadata_is_not_weight_readiness"] = True
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
