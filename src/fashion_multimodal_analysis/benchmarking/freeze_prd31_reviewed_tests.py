"""基准集管理：保留训练、开发、固定回归/审核划分，冻结后的标签不能随结果调整。

Validate human reviews and freeze v2 manifests without replacing frozen v1.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fashion_multimodal_analysis.analysis.audit_dataset_overlap import image_key
from fashion_multimodal_analysis.benchmarking.prepare_prd31_recovery import (
    ATTRIBUTES,
    REGIONS,
    applies,
    digest,
    labels_for,
)
from fashion_multimodal_analysis.common.io import read_csv, write_csv, write_json
from fashion_multimodal_analysis.common.paths import project_root, resolve_path
from fashion_multimodal_analysis.evaluation.protocol import flag, valid_box


def validate_config(config: dict[str, Any]) -> None:
    """校验 配置。

    Args:
        config: 当前模块的配置。

    Returns:
        当前条件的校验结果；失败条件及返回形式见函数体。

    Raises:
        ValueError: v2 thresholds, timing and coverage are fixed; changed rules need a
        new protocol versi
    """
    expected = {
        "score_threshold": 0.40,
        "mask_threshold": 0.50,
        "bbox_match_iou": 0.50,
        "grounding_box_threshold": 0.30,
        "grounding_text_threshold": 0.25,
        "warmup_runs": 10,
        "timed_passes": 5,
        "min_reviewed_positive_per_region": 5,
        "required_regions": REGIONS,
        "required_attributes": ATTRIBUTES,
        "accuracy_targets": {"mask_iou": 0.85, "grounding": 0.92, "attributes": 0.88},
        "latency_targets_ms": {"segmentation": 50, "grounding": 30, "attributes": 20},
    }
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError(
            "v2 thresholds, timing and coverage are fixed; changed rules need a new protocol version"
        )


EDITABLE_GROUND = {
    "target_present",
    "gt_region_bbox_x1",
    "gt_region_bbox_y1",
    "gt_region_bbox_x2",
    "gt_region_bbox_y2",
    "gt_region_mask_path",
    "ambiguous",
    "review_status",
    "reviewer",
    "annotation_note",
    "exclude_reason",
}
EDITABLE_ATTRIBUTE = {
    "gt_label",
    "ambiguous",
    "review_status",
    "reviewer",
    "annotation_note",
    "exclude_reason",
}


def validate_identity(
    drafts: Any, reviews: Any, keys: list[str], editable: bool
) -> list[Any]:
    """校验 identity。

    Args:
        drafts: drafts。
        reviews: reviews。
        keys: 标识。
        editable: editable。

    Returns:
        当前条件的校验结果；失败条件及返回形式见函数体。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    draft_map = {tuple(row[k] for k in keys): row for row in drafts}
    review_map = {tuple(row[k] for k in keys): row for row in reviews}
    if len(review_map) != len(reviews) or set(draft_map) != set(review_map):
        raise ValueError(
            "Review rows must contain every original candidate exactly once"
        )
    for key, original in draft_map.items():
        reviewed = review_map[key]
        for field, value in original.items():
            if field not in editable and str(reviewed.get(field, "")) != str(value):
                raise ValueError(f"Immutable review field changed: {key}: {field}")
    return [review_map[tuple(row[k] for k in keys)] for row in drafts]


def reviewed(row: dict[str, Any]) -> bool:
    """reviewed。

    Args:
        row: 一条实例、预测或审核记录。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return row.get("review_status") == "reviewed" and bool(
        row.get("reviewer", "").strip()
    )


def validate_reviews(
    ground_drafts: Any,
    attribute_drafts: Any,
    ground_reviews: Any,
    attribute_reviews: Any,
    minimum: int = 5,
) -> tuple[Any, ...]:
    """校验 reviews。

    Args:
        ground_drafts: ground drafts。
        attribute_drafts: 属性 drafts。
        ground_reviews: ground reviews。
        attribute_reviews: 属性 reviews。
        minimum: minimum。

    Returns:
        按顺序返回 selected, attrs, absent 等结果。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    ground = validate_identity(
        ground_drafts, ground_reviews, ("query_id",), EDITABLE_GROUND
    )
    attrs = validate_identity(
        attribute_drafts,
        attribute_reviews,
        ("sample_id", "attribute_name"),
        EDITABLE_ATTRIBUTE,
    )
    positive, absent, excluded = defaultdict(list), [], []
    for row in ground:
        if not reviewed(row):
            raise ValueError(
                "Grounding review pending / reviewer missing: " + row["query_id"]
            )
        if flag(row.get("ambiguous")):
            if not row.get("exclude_reason", "").strip():
                raise ValueError("Ambiguous query needs an exclusion reason")
            excluded.append(row)
            continue
        if row.get("target_present") == "no":
            absent.append(row)
            continue
        if row.get("target_present") != "yes":
            raise ValueError("Set target_present to yes/no: " + row["query_id"])
        try:
            box = [float(row["gt_region_bbox_" + k]) for k in ("x1", "y1", "x2", "y2")]
        except (ValueError, KeyError):
            raise ValueError(
                "Positive target needs an original-image GT box: " + row["query_id"]
            )
        if (
            not valid_box(box)
            or box[0] < 0
            or box[1] < 0
            or box[2] > float(row["image_width"])
            or box[3] > float(row["image_height"])
        ):
            raise ValueError(
                "Invalid / out-of-bounds original-image region box: " + row["query_id"]
            )
        positive[row["target_region"]].append(row)
    counts = {region: len(positive[region]) for region in REGIONS}
    if any(n < minimum for n in counts.values()):
        raise ValueError(
            "Insufficient positive region coverage; do not fabricate boxes: "
            + json.dumps(counts)
        )
    selected = [row for region in REGIONS for row in positive[region][:minimum]]
    eligible = []
    for row in attrs:
        applicable = applies(row["attribute_name"], row["garment_category"])
        if not applicable:
            if row["applicability"] != "not_applicable" or row.get("gt_label"):
                raise ValueError("Non-applicable attribute must have no GT label")
            continue
        if not reviewed(row):
            raise ValueError(
                "Attribute review pending / reviewer missing: "
                + row["sample_id"]
                + "/"
                + row["attribute_name"]
            )
        if flag(row.get("ambiguous")):
            if not row.get("exclude_reason", "").strip():
                raise ValueError("Ambiguous attribute needs an exclusion reason")
            continue
        if row["gt_label"] not in labels_for(
            row["attribute_name"], row["garment_category"]
        ):
            raise ValueError(
                "GT label outside category-specific taxonomy: "
                + row["sample_id"]
                + "/"
                + row["attribute_name"]
            )
        eligible.append(row)
    categories = {r["garment_category"] for r in attribute_drafts}
    required = {(a, c) for a in ATTRIBUTES for c in categories if applies(a, c)}
    actual = {(r["attribute_name"], r["garment_category"]) for r in eligible}
    if not required <= actual:
        raise ValueError(
            "Insufficient reviewed eligible attribute/category coverage: "
            + str(sorted(required - actual))
        )
    return (
        selected,
        attrs,
        {
            "reviewed_grounding_candidates": len(ground),
            "positive_by_region": counts,
            "formal_positive_grounding": len(selected),
            "absent_queries": len(absent),
            "ambiguous_queries": len(excluded),
            "eligible_attributes": len(eligible),
            "ambiguous_attributes": sum(flag(r.get("ambiguous")) for r in attrs),
            "attribute_rows": len(attrs),
        },
        absent,
    )


def verify_frozen(root: Path) -> Any:
    """确认实验没有改动保留的 V5 权重和固定评估清单。

    Args:
        root: 当前项目根目录。

    Returns:
        返回 meta，由函数体中同名变量的计算/收集过程得到。

    Raises:
        ValueError: Recovery configuration changed after benchmark freeze
    """
    frozen = root / "benchmark/prd_3_1_v2/frozen"
    meta = json.loads((frozen / "freeze_meta.json").read_text())
    for name, expected in meta["sha256"].items():
        if digest(frozen / name) != expected:
            raise ValueError("Frozen artifact changed: " + name)
    if digest(root / meta["training_manifest"]) != meta["training_manifest_sha256"]:
        raise ValueError("Training manifest changed after benchmark freeze")
    if digest(root / meta["validation_manifest"]) != meta["validation_manifest_sha256"]:
        raise ValueError("Validation manifest changed after benchmark freeze")
    if (
        digest(root / "configs/prd_3_1_recovery_v2.json")
        != meta["config_source_sha256"]
    ):
        raise ValueError("Recovery configuration changed after benchmark freeze")
    validate_config(json.loads((frozen / "config.json").read_text()))
    return meta


def freeze(
    root: Path | None = None, grounding_review: Any = None, attribute_review: Any = None
) -> Any:
    """freeze。

    Args:
        root: 当前项目根目录。
        grounding_review: grounding 审核。
        attribute_review: 属性 审核。

    Returns:
        结果字典包含 benchmark_version, status, sha256, review_summary, training_manifest,
        training_manifest_sha256, validation_manifest, validation_manifest_sha256,
        config_source_sha256, review_inputs_sha256。

    Raises:
        FileExistsError: v2 is already frozen; use a new benchmark version for revised
        annotations
        ValueError: Training still overlaps Core/validation source images
    """
    root = root or project_root()
    base = root / "benchmark/prd_3_1_v2"
    frozen = base / "frozen"
    if frozen.exists():
        raise FileExistsError(
            "v2 is already frozen; use a new benchmark version for revised annotations"
        )
    config_path = root / "configs/prd_3_1_recovery_v2.json"
    config = json.loads(config_path.read_text())
    validate_config(config)
    reference = root / "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv"
    if digest(reference) != config["reference_segmentation_sha256"]:
        raise ValueError("Frozen v1 reference changed")
    core = read_csv(reference)
    if len(core) != 400 or Counter(r["garment_category"] for r in core) != Counter(
        {c: 50 for c in config["required_prd_classes"]}
    ):
        raise ValueError("Core must retain the original 400 cases, 50 per PRD class")
    protected = {
        image_key(r["source_image"])
        for p in (reference, root / config["val_manifest"])
        for r in read_csv(p)
    }
    training = {
        image_key(r["source_image"]) for r in read_csv(root / config["train_manifest"])
    }
    if protected & training:
        raise ValueError("Training still overlaps Core/validation source images")
    ground_path = grounding_review or base / "review/grounding_reviewed.csv"
    attribute_path = attribute_review or base / "review/attribute_reviewed.csv"
    selected, attrs, summary, absent = validate_reviews(
        read_csv(base / "drafts/grounding_test.csv"),
        read_csv(base / "drafts/attribute_test.csv"),
        read_csv(ground_path),
        read_csv(attribute_path),
        config["min_reviewed_positive_per_region"],
    )
    # All validation finishes before creating an immutable frozen directory.
    frozen.mkdir(parents=True)
    write_csv(frozen / "grounding_test.csv", selected)
    write_csv(
        frozen / "grounding_absent_diagnostic.csv",
        absent,
        fields=list(read_csv(ground_path)[0]),
    )
    write_csv(frozen / "attribute_test.csv", attrs)
    core_path = root / "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv"
    (frozen / "segmentation_test.csv").write_bytes(core_path.read_bytes())
    by_sample = {r["sample_id"]: r for r in selected + attrs}
    cases = []
    for sid, row in sorted(by_sample.items()):
        item = {
            k: row[k]
            for k in read_csv(base / "drafts/end_to_end_test.csv")[0]
            if k in row
        }
        item.update(
            required_3_1_2_queries=json.dumps(
                sorted({r["target_region"] for r in selected if r["sample_id"] == sid})
            ),
            required_3_1_3_attributes=json.dumps(
                sorted(
                    {
                        r["attribute_name"]
                        for r in attrs
                        if r["sample_id"] == sid
                        and r["applicability"] == "eligible"
                        and not flag(r.get("ambiguous"))
                    }
                )
            ),
            review_status="reviewed",
            exclude_reason="",
        )
        cases.append(item)
    write_csv(frozen / "end_to_end_test.csv", cases)
    write_json(
        frozen / "config.json", {**config, "status": "FROZEN_REVIEWED_REGRESSION_TEST"}
    )
    meta = {
        "benchmark_version": config["benchmark_version"],
        "status": "FROZEN_REVIEWED_REGRESSION_TEST",
        "sha256": {p.name: digest(p) for p in sorted(frozen.iterdir())},
        "review_summary": summary,
        "training_manifest": config["train_manifest"],
        "training_manifest_sha256": digest(root / config["train_manifest"]),
        "validation_manifest": config["val_manifest"],
        "validation_manifest_sha256": digest(root / config["val_manifest"]),
        "config_source_sha256": digest(config_path),
        "review_inputs_sha256": {
            "grounding": digest(ground_path),
            "attributes": digest(attribute_path),
        },
        "boundary": config["test_isolation_note"],
    }
    write_json(frozen / "freeze_meta.json", meta)
    return meta


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grounding-review")
    p.add_argument("--attribute-review")
    args = p.parse_args()
    print(
        json.dumps(
            freeze(
                grounding_review=(
                    resolve_path(args.grounding_review)
                    if args.grounding_review
                    else None
                ),
                attribute_review=(
                    resolve_path(args.attribute_review)
                    if args.attribute_review
                    else None
                ),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
