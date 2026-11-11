"""基准集管理：保留训练、开发、固定回归/审核划分，冻结后的标签不能随结果调整。

Prepare clean training, reusable legacy reviews and new review candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fashion_multimodal_analysis.analysis.audit_dataset_overlap import image_key
from fashion_multimodal_analysis.common.io import read_csv, write_csv, write_json
from fashion_multimodal_analysis.common.paths import project_root, resolve_path
from fashion_multimodal_analysis.common.schema import CLASS_TO_ID

VERSION = "prd_3_1_v2"
REGIONS = [
    "collar",
    "cuff",
    "hem",
    "pocket",
    "shoulder",
    "waist",
    "pattern",
    "decoration",
]
ATTRIBUTES = [
    "primary_color",
    "pattern",
    "sleeve_length",
    "neckline",
    "silhouette_fit",
    "fashion_style",
]
UPPER = {"top", "outerwear", "dress"}
CLOTHING = UPPER | {"pants", "skirt"}
PROMPTS = {
    "collar": "the collar or neckline area of the garment",
    "cuff": "the sleeve cuff of the garment",
    "hem": "the lower hem edge of the garment",
    "pocket": "the pocket of the garment",
    "shoulder": "the shoulder area of the garment",
    "waist": "the waist area of the garment",
    "pattern": "the printed pattern on the garment",
    "decoration": "the decorative detail on the garment",
}
TAXONOMY = {
    "primary_color": [
        "black",
        "white",
        "gray",
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "purple",
        "pink",
        "brown",
        "beige",
    ],
    "pattern": [
        "solid",
        "striped",
        "checked_plaid",
        "polka_dot",
        "floral",
        "animal_print",
        "geometric_abstract",
        "graphic_logo",
        "camouflage",
        "other_pattern",
    ],
    "sleeve_length": ["sleeveless", "cap", "short", "elbow", "three_quarter", "long"],
    "neckline": [
        "crew",
        "v_neck",
        "round_scoop",
        "square",
        "boat",
        "halter",
        "off_shoulder",
        "one_shoulder",
        "turtleneck",
        "mock_neck",
        "polo",
        "shirt_collar",
        "hooded",
        "other",
    ],
    "silhouette_fit": [
        "slim_fitted",
        "regular_straight",
        "loose_oversized",
        "slim_tapered",
        "straight",
        "wide_leg",
        "flared",
        "a_line",
        "fitted",
    ],
    "fashion_style": [
        "casual",
        "formal",
        "sporty",
        "streetwear",
        "elegant",
        "minimalist",
        "vintage",
        "romantic",
    ],
}
FIT_BY_CATEGORY = {
    "top": ["slim_fitted", "regular_straight", "loose_oversized"],
    "outerwear": ["slim_fitted", "regular_straight", "loose_oversized"],
    "pants": ["slim_tapered", "straight", "wide_leg", "flared"],
    "skirt": ["straight", "a_line", "flared"],
    "dress": ["fitted", "straight", "a_line", "flared"],
}


def labels_for(attribute: str, category: str) -> Any:
    """标签 for。

    Args:
        attribute: 属性。
        category: 本次处理的服饰或区域类别。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return (
        FIT_BY_CATEGORY.get(category, [])
        if attribute == "silhouette_fit"
        else TAXONOMY[attribute]
    )


def applies(attribute: str, category: str) -> bool:
    """applies。

    Args:
        attribute: 属性。
        category: 本次处理的服饰或区域类别。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if attribute in ("primary_color", "pattern"):
        return True
    return category in (
        UPPER if attribute in ("sleeve_length", "neckline") else CLOTHING
    )


def digest(path: str | Path) -> str:
    """分块计算文件 SHA-256，核对文件身份而不加载整个权重到内存。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        64 位十六进制 SHA-256 摘要字符串。
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def core_rows(root: Path) -> Any:
    """Core 固定回归 逐行记录。

    Args:
        root: 当前项目根目录。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return read_csv(root / "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv")


def prepare_clean_training(root: Path) -> Any:
    """准备 clean 训练。

    Args:
        root: 当前项目根目录。

    Returns:
        结果字典包含 source_manifest, source_sha256, core_sha256, validation_sha256,
        clean_manifest, clean_sha256, original_instances, removed_instances,
        clean_instances, clean_images。

    Raises:
        ValueError: Clean training would lose an entire PRD class
    """
    source = root / "configs/prd_8class_train_v3.csv"
    validation = root / "configs/prd_8class_val_v1.csv"
    core = root / "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv"
    rows = read_csv(source)
    protected = {
        image_key(r["source_image"]) for p in (validation, core) for r in read_csv(p)
    }
    clean = [r for r in rows if image_key(r["source_image"]) not in protected]
    removed = [r for r in rows if image_key(r["source_image"]) in protected]
    if not clean or set(r["garment_category"] for r in clean) != set(CLASS_TO_ID):
        raise ValueError("Clean training would lose an entire PRD class")
    target = root / "configs/prd_8class_train_v4_clean.csv"
    write_csv(target, clean, fields=list(rows[0]))
    out = root / "reports/repository_audit/recovery_v2/data_isolation"
    write_csv(out / "removed_training_rows.csv", removed, fields=list(rows[0]))
    summary = {
        "source_manifest": str(source.relative_to(root)),
        "source_sha256": digest(source),
        "core_sha256": digest(core),
        "validation_sha256": digest(validation),
        "clean_manifest": str(target.relative_to(root)),
        "clean_sha256": digest(target),
        "original_instances": len(rows),
        "removed_instances": len(removed),
        "clean_instances": len(clean),
        "clean_images": len({image_key(r["source_image"]) for r in clean}),
        "class_counts": dict(Counter(r["garment_category"] for r in clean)),
        "source_counts": dict(Counter(r["source_dataset"] for r in clean)),
        "core_or_validation_overlap_images": 0,
        "scope": "Normalized source_image path isolation. Not image-content deduplication or a restored untouched blind test.",
    }
    write_json(out / "summary.json", summary)
    return summary


def import_legacy_reviews(root: Path, destination: Any) -> Any:
    """import legacy reviews。

    Args:
        root: 当前项目根目录。
        destination: 目标文件或目录。

    Returns:
        返回 summary，由函数体中同名变量的计算/收集过程得到。
    """
    rows, summary = [], {}
    for attribute, file, label, predicted in [
        (
            "primary_color",
            "reports/prd_attribute_extraction/color_v1/primary_color_holdout_audit_prefilled.csv",
            "manual_color",
            "predicted_color",
        ),
        (
            "pattern",
            "reports/prd_attribute_extraction/pattern_v3_hierarchical/pattern_v3_holdout_audit_final.csv",
            "manual_pattern",
            "final_pattern",
        ),
    ]:
        source = read_csv(root / file)
        eligible, correct = 0, 0
        for index, row in enumerate(source):
            gt = row.get(label, "").strip()
            ambiguous = gt == "ambiguous_multicolor" or row.get(
                "manual_correct"
            ) not in ("0", "1")
            if gt and not ambiguous:
                eligible += 1
                correct += int(gt == row[predicted])
            rows.append(
                {
                    "sample_id": f"LEGACY_{attribute}_{index+1:03d}",
                    "source_image": row["source_image"],
                    "garment_id": row["garment_id"],
                    "garment_category": row["garment_category"],
                    "attribute_name": attribute,
                    "gt_label": gt if not ambiguous else "",
                    "applicability": "ambiguous" if ambiguous else "eligible",
                    "ambiguous": int(ambiguous),
                    "review_status": "reviewed",
                    "reviewer": "uploaded_legacy_review",
                    "prediction_label": row[predicted],
                    "source_review_file": file,
                    "split": "development",
                    "notes": row.get("notes", ""),
                }
            )
        summary[attribute] = {
            "reviewed_rows": len(source),
            "eligible_rows": eligible,
            "correct": correct,
            "accuracy": correct / eligible if eligible else None,
            "split": "development",
            "formal_acceptance": "NOT_CLAIMED",
            "reason": "Previously inspected/tuned research holdout, retained as development evidence.",
        }
    write_csv(destination / "development/legacy_attribute_reviews.csv", rows)
    write_json(destination / "development/legacy_attribute_metrics.json", summary)
    return summary


def sample_case(row: dict[str, Any]) -> dict[str, Any]:
    """sample 案例。

    Args:
        row: 记录字段，使用 final_sample_id, source_dataset, source_image, garment_id,
        garment_category, image_width, image_height。

    Returns:
        结果字典，主要字段为 benchmark_version, split, sample_id, source_dataset, source_image,
        garment_id, garment_category, image_width, image_height。
    """
    return {
        "benchmark_version": VERSION,
        "split": "review_candidate",
        "sample_id": row["final_sample_id"],
        "source_dataset": row["source_dataset"],
        "source_image": row["source_image"],
        "garment_id": row["garment_id"],
        "garment_category": row["garment_category"],
        "image_width": row["image_width"],
        "image_height": row["image_height"],
        **{
            k: row[k]
            for k in (
                "gt_bbox_x1",
                "gt_bbox_y1",
                "gt_bbox_x2",
                "gt_bbox_y2",
                "gt_mask_path",
                "gt_overlay_path",
            )
        },
    }


def prepare_candidates(
    root: Path,
    destination: Any,
    per_class: int = 5,
    per_region: int = 10,
    seed: int = 20260930,
) -> dict[str, Any]:
    # Existing reviewer edits are never discarded by repeated prepare invocations.
    """准备 candidates。

    Args:
        root: 当前项目根目录。
        destination: 目标文件或目录。
        per_class: 逐 类别。
        per_region: 逐 区域。
        seed: 控制随机过程的种子。

    Returns:
        结果字典，主要字段为 attribute_rows, eligible_attribute_rows, attribute_garments,
        grounding_rows, end_to_end_images, reused_existing_drafts。
    """
    draft = destination / "drafts"
    if all(
        (draft / name).exists()
        for name in ("attribute_test.csv", "grounding_test.csv", "end_to_end_test.csv")
    ):
        return {
            "attribute_rows": len(read_csv(draft / "attribute_test.csv")),
            "grounding_rows": len(read_csv(draft / "grounding_test.csv")),
            "reused_existing_drafts": True,
        }
    core = core_rows(root)
    by_class = defaultdict(list)
    for row in core:
        by_class[row["garment_category"]].append(row)
    rng = random.Random(seed)
    attribute_rows = []
    for category in CLASS_TO_ID:
        for row in rng.sample(
            sorted(by_class[category], key=lambda r: r["final_sample_id"]), per_class
        ):
            case = sample_case(row)
            for attribute in ATTRIBUTES:
                applicable = applies(attribute, category)
                attribute_rows.append(
                    {
                        **case,
                        "attribute_name": attribute,
                        "applicability": "eligible" if applicable else "not_applicable",
                        "gt_label": "",
                        "ambiguous": 0,
                        "review_status": "pending" if applicable else "not_applicable",
                        "reviewer": "",
                        "annotation_note": "",
                        "exclude_reason": "",
                    }
                )
    grounding_rows = []
    for region in REGIONS:
        allowed = (
            UPPER
            if region in ("collar", "cuff", "shoulder")
            else CLOTHING if region in ("hem", "pocket", "waist") else set(CLASS_TO_ID)
        )
        pool = sorted(
            [r for r in core if r["garment_category"] in allowed],
            key=lambda r: r["final_sample_id"],
        )
        region_rng = random.Random(seed + REGIONS.index(region))
        for index, row in enumerate(region_rng.sample(pool, per_region)):
            grounding_rows.append(
                {
                    **sample_case(row),
                    "query_id": f"{region}_{index+1:03d}",
                    "query_text": PROMPTS[region],
                    "target_region": region,
                    "target_present": "",
                    "gt_region_bbox_x1": "",
                    "gt_region_bbox_y1": "",
                    "gt_region_bbox_x2": "",
                    "gt_region_bbox_y2": "",
                    "gt_region_mask_path": "",
                    "ambiguous": 0,
                    "review_status": "pending",
                    "reviewer": "",
                    "annotation_note": "",
                    "exclude_reason": "",
                }
            )
    write_csv(draft / "attribute_test.csv", attribute_rows)
    write_csv(draft / "grounding_test.csv", grounding_rows)
    cases = {
        r["final_sample_id"]: sample_case(r)
        for r in core
        if any(
            x["sample_id"] == r["final_sample_id"]
            for x in attribute_rows + grounding_rows
        )
    }
    chain = []
    for sid, row in sorted(cases.items()):
        attrs = sorted(
            {
                r["attribute_name"]
                for r in attribute_rows
                if r["sample_id"] == sid and r["applicability"] == "eligible"
            }
        )
        queries = sorted(
            {r["target_region"] for r in grounding_rows if r["sample_id"] == sid}
        )
        chain.append(
            {
                **row,
                "required_3_1_2_queries": json.dumps(queries),
                "required_3_1_3_attributes": json.dumps(attrs),
                "review_status": "pending",
                "exclude_reason": "",
            }
        )
    write_csv(draft / "end_to_end_test.csv", chain)
    return {
        "attribute_rows": len(attribute_rows),
        "eligible_attribute_rows": sum(
            applies(r["attribute_name"], r["garment_category"]) for r in attribute_rows
        ),
        "attribute_garments": len({r["sample_id"] for r in attribute_rows}),
        "grounding_rows": len(grounding_rows),
        "end_to_end_images": len(chain),
        "reused_existing_drafts": False,
    }


def expand_grounding(root: Path, extra: Any, region: str | None = None) -> Any:
    """Append GT review candidates based on presence coverage, preserving every prior row.

    Args:
        root: 当前项目根目录。
        extra: extra。
        region: 区域。

    Returns:
        结果字典包含 added_queries, total_grounding_candidates, selection_policy。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
        FileExistsError: Cannot expand a frozen benchmark
    """
    if extra < 1:
        raise ValueError("Extra candidates must be positive")
    base = root / "benchmark/prd_3_1_v2"
    if (base / "frozen").exists():
        raise FileExistsError("Cannot expand a frozen benchmark")
    draft = base / "drafts"
    rows = read_csv(draft / "grounding_test.csv")
    attrs = read_csv(draft / "attribute_test.csv")
    core = core_rows(root)
    added = []
    for name in ([region] if region else REGIONS):
        if name not in REGIONS:
            raise ValueError("Unknown region")
        existing = [r for r in rows if r["target_region"] == name]
        used = {r["sample_id"] for r in existing}
        allowed = (
            UPPER
            if name in ("collar", "cuff", "shoulder")
            else CLOTHING if name in ("hem", "pocket", "waist") else set(CLASS_TO_ID)
        )
        pool = sorted(
            [
                r
                for r in core
                if r["garment_category"] in allowed and r["final_sample_id"] not in used
            ],
            key=lambda r: r["final_sample_id"],
        )
        if len(pool) < extra:
            raise ValueError("Not enough unused Core candidates for " + name)
        rng = random.Random(20260930 + len(existing) + REGIONS.index(name))
        for index, row in enumerate(rng.sample(pool, extra), len(existing) + 1):
            added.append(
                {
                    **sample_case(row),
                    "query_id": f"{name}_{index:03d}",
                    "query_text": PROMPTS[name],
                    "target_region": name,
                    "target_present": "",
                    "gt_region_bbox_x1": "",
                    "gt_region_bbox_y1": "",
                    "gt_region_bbox_x2": "",
                    "gt_region_bbox_y2": "",
                    "gt_region_mask_path": "",
                    "ambiguous": 0,
                    "review_status": "pending",
                    "reviewer": "",
                    "annotation_note": "",
                    "exclude_reason": "",
                }
            )
    rows.extend(added)
    source_cases = {
        r["final_sample_id"]: sample_case(r)
        for r in core
        if r["final_sample_id"] in {x["sample_id"] for x in rows + attrs}
    }
    chain = [
        {
            **row,
            "required_3_1_2_queries": json.dumps(
                sorted({r["target_region"] for r in rows if r["sample_id"] == sid})
            ),
            "required_3_1_3_attributes": json.dumps(
                sorted(
                    {
                        r["attribute_name"]
                        for r in attrs
                        if r["sample_id"] == sid and r["applicability"] == "eligible"
                    }
                )
            ),
            "review_status": "pending",
            "exclude_reason": "",
        }
        for sid, row in sorted(source_cases.items())
    ]
    write_csv(draft / "grounding_test.csv", rows)
    write_csv(draft / "end_to_end_test.csv", chain)
    result = {
        "added_queries": [r["query_id"] for r in added],
        "total_grounding_candidates": len(rows),
        "selection_policy": "Unused Core sources; append based on human presence coverage, never based on model success",
    }
    write_json(base / "review/last_candidate_expansion.json", result)
    return result


def prepare(root: Path | None = None) -> dict[str, Any]:
    """准备当前实验所需数据，并执行来源/划分/有效性检查。

    Args:
        root: 当前项目根目录。

    Returns:
        结果字典，主要字段为 clean_training, reused_reviews, candidates, review_required。
    """
    root = root or project_root()
    destination = root / "benchmark/prd_3_1_v2"
    destination.mkdir(parents=True, exist_ok=True)
    clean = prepare_clean_training(root)
    reviews = import_legacy_reviews(root, destination)
    candidates = prepare_candidates(root, destination)
    config = {
        "benchmark_version": VERSION,
        "status": "DRAFT_REQUIRES_HUMAN_REVIEW",
        "seed": 20260930,
        "required_regions": REGIONS,
        "required_attributes": ATTRIBUTES,
        "required_prd_classes": list(CLASS_TO_ID),
        "min_reviewed_positive_per_region": 5,
        "attribute_taxonomy": TAXONOMY,
        "category_fit_taxonomy": FIT_BY_CATEGORY,
        "reference_segmentation_sha256": digest(
            root / "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv"
        ),
        "segmentation_checkpoint": "outputs/prd_instance_segmentation/maskrcnn_8class_v4_clean/checkpoint_last.pth",
        "train_manifest": "configs/prd_8class_train_v4_clean.csv",
        "val_manifest": "configs/prd_8class_val_v1.csv",
        "grounding_model": "IDEA-Research/grounding-dino-tiny",
        "sam_model": "facebook/sam-vit-base",
        "clip_model": "openai/clip-vit-base-patch32",
        "score_threshold": 0.40,
        "mask_threshold": 0.50,
        "bbox_match_iou": 0.50,
        "grounding_box_threshold": 0.30,
        "grounding_text_threshold": 0.25,
        "warmup_runs": 10,
        "timed_passes": 5,
        "accuracy_targets": {"mask_iou": 0.85, "grounding": 0.92, "attributes": 0.88},
        "latency_targets_ms": {"segmentation": 50, "grounding": 30, "attributes": 20},
        "test_isolation_note": "Core source images have prior segmentation regression history. New labels are not automatically a pristine blind holdout.",
    }
    config_file = root / "configs/prd_3_1_recovery_v2.json"
    if not config_file.exists():
        write_json(config_file, config)
    write_json(
        root / "reports/repository_audit/recovery_v2/preparation.json",
        {
            "data_isolation": clean,
            "legacy_reviews": reviews,
            "candidates": candidates,
            "status": "CODE_AND_DATA_PREPARED; REVIEW_AND_GPU_RUNS_REQUIRED",
        },
    )
    return {
        "clean_training": clean,
        "reused_reviews": reviews,
        "candidates": candidates,
        "review_required": True,
    }


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extra-grounding",
        type=int,
        default=0,
        help="Append unused candidate images when human positive coverage is insufficient",
    )
    parser.add_argument(
        "--region",
        choices=REGIONS,
        help="Expand only this region; otherwise expand all eight",
    )
    args = parser.parse_args()
    result = prepare()
    if args.extra_grounding:
        result["expansion"] = expand_grounding(
            project_root(), args.extra_grounding, args.region
        )
        from fashion_multimodal_analysis.benchmarking.build_prd31_review_packet import (
            build,
        )

        result["review_packet"] = build()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
