#!/usr/bin/env python3
"""PRD 3.2: metadata-only provenance audit for three *specific* PRD 3.1 CSV files.

SECURITY: does not open images, evaluate holdout, run models or approve samples.
Training/holdout independence cannot be established without separate authoritative splits.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import posixpath
import re
from collections import Counter, defaultdict
from pathlib import Path
from itertools import combinations

SOURCES = {
    "inventory": ("prd_8class_unified_manifest_v2.csv", "EIGHT_CLASS_INVENTORY_PROVENANCE_UNVERIFIED"),
    "fullval": ("prd_3_1_predicted_roi_fullval_v1.csv", "REFERENCE_ONLY_FULLVAL_DO_NOT_PROMOTE"),
    "matched": ("prd_3_1_predicted_roi_matched_v2.csv", "REFERENCE_ONLY_MATCHED_DO_NOT_PROMOTE"),
}
CLASSES = ("top", "pants", "skirt", "outerwear", "dress", "shoe", "bag", "accessory")
SPLIT_KEYS = ("split", "subset", "partition", "set_name", "phase", "fold", "data_split")


def clean(v):
    return str(v or "").strip()


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:18]


def norm_path(value):
    v = clean(value).replace("\\", "/")
    if not v:
        return ""
    return posixpath.normpath(re.sub(r"/+", "/", v))


def img_key(row):
    src = clean(row.get("source_dataset")).lower()
    path = norm_path(row.get("source_image"))
    return f"{src}|{path}" if path else ""


def loose_key(row):
    src = clean(row.get("source_dataset")).lower()
    path = norm_path(row.get("source_image"))
    return f"{src}|{posixpath.basename(path).lower()}" if path else ""


def float_value(value):
    try:
        n = float(clean(value))
        return n if math.isfinite(n) else None
    except (TypeError, ValueError, OverflowError):
        return None


def bbox_check(row):
    names = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
    v = [float_value(row.get(x)) for x in names]
    if any(x is None for x in v):
        return "MISSING_OR_NONFINITE_BBOX"
    x1, y1, x2, y2 = v
    if x1 < 0 or y1 < 0:
        return "NEGATIVE_BBOX"
    if x1 >= x2 or y1 >= y2:
        return "ZERO_OR_NEGATIVE_BBOX_AREA"
    w, h = float_value(row.get("image_width")), float_value(row.get("image_height"))
    if w is not None and h is not None:
        if w <= 0 or h <= 0 or x2 > w or y2 > h:
            return "OUTSIDE_DECLARED_IMAGE"
        return "GEOMETRY_PLAUSIBLE_SIZE_DECLARED_NOT_VERIFIED"
    return "GEOMETRY_PLAUSIBLE_SIZE_UNKNOWN"


def split_hint(row):
    """Hints are non-authoritative and NEVER approve a split."""
    parts = [clean(row.get(k)) for k in SPLIT_KEYS]
    parts.extend(clean(row.get(k)) for k in ("source_manifest", "source_image", "paired_manifest_role"))
    s = " ".join(parts).lower()
    if re.search(r"(^|[\\/_. -])(holdout|sealed|test)([\\/_. -]|$)", s):
        return "RESTRICTED_TOKEN_NEEDS_SOURCE_VERIFICATION"
    if re.search(r"(^|[\\/_. -])(train|training)([\\/_. -]|$)", s):
        return "TRAIN_TOKEN_NEEDS_SOURCE_VERIFICATION"
    if re.search(r"(^|[\\/_. -])(val|valid|validation|fullval)([\\/_. -]|$)", s):
        return "VAL_TOKEN_NOT_NEW_DEV_PROOF"
    return "UNKNOWN_NO_AUTHORITATIVE_SPLIT"


def read_rows(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        heads = reader.fieldnames or []
        for key in ("source_dataset", "source_image", "garment_id", "garment_category", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
            if key not in heads:
                raise ValueError(f"{path.name}: required actual-source field missing: {key}")
        return heads, list(reader)


def as_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit(project_root, output_dir):
    config_dir = Path(project_root) / "configs"
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output exists, choose new path: {output_dir}")
    # Fail fast on missing inputs; no partial output or modifications.
    files = {k: config_dir / value[0] for k, value in SOURCES.items()}
    loaded = {k: read_rows(path) for k, path in files.items()}
    output_dir.mkdir(parents=True, exist_ok=False)

    identities = {}; loose = {}; per_source = {}; processed = {}
    for source, (headers, rows) in loaded.items():
        counts = Counter(clean(r.get("garment_category")) or "<empty>" for r in rows)
        geometry = Counter(bbox_check(r) for r in rows)
        splits = Counter(split_hint(r) for r in rows)
        valid_ids = [img_key(r) for r in rows if img_key(r)]
        loose_ids = [loose_key(r) for r in rows if loose_key(r)]
        identities[source], loose[source] = set(valid_ids), set(loose_ids)
        processed[source] = rows
        per_source[source] = {
            "filename": files[source].name,
            "role": SOURCES[source][1],
            "row_count": len(rows),
            "unique_image_keys": len(identities[source]),
            "rows_missing_source_image": len(rows) - len(valid_ids),
            "rows_with_same_image_possible_multiple_instances": len(valid_ids) - len(set(valid_ids)),
            "columns_verified": headers,
            "category_distribution_UNVERIFIED_AS_GT": dict(sorted(counts.items())),
            "bbox_geometry_metadata_only": dict(sorted(geometry.items())),
            "split_string_hints_NOT_PROOF": dict(sorted(splits.items())),
            "is_independent_test_proven": False,
            "eligible_for_3_2_accuracy": False,
        }

    pairwise = []
    inventory_restricted = set()
    inventory_loose = set()
    for a, b in combinations(SOURCES, 2):
        exact = identities[a] & identities[b]
        weak = loose[a] & loose[b]
        pairwise.append({
            "source_a": a, "source_b": b,
            "exact_dataset_and_normalized_path_overlap_image_keys": len(exact),
            "weak_dataset_and_basename_overlap_image_keys": len(weak),
            "interpretation": "exact identity evidence only; basename overlaps are possible aliases, not proof of identical content",
        })
        if a == "inventory" and b in ("fullval", "matched"):
            inventory_restricted.update(exact)
            inventory_loose.update(weak)
        if b == "inventory" and a in ("fullval", "matched"):
            inventory_restricted.update(exact)
            inventory_loose.update(weak)

    candidates = []
    statuses = Counter()
    for i, row in enumerate(processed["inventory"], 1):
        image, weak = img_key(row), loose_key(row)
        geometry = bbox_check(row)
        hint = split_hint(row)
        overlap_exact = bool(image and image in inventory_restricted)
        overlap_weak = bool(weak and weak in inventory_loose)
        reasons = ["TRAIN_HOLDOUT_SPLIT_NOT_VERIFIED", "CATEGORY_SEMANTICS_NOT_HUMAN_GT"]
        if overlap_exact:
            reasons.append("EXACT_OVERLAP_WITH_REFERENCE_SOURCE")
        elif overlap_weak:
            reasons.append("BASENAME_OVERLAP_NEEDS_MANUAL_ALIAS_CHECK")
        if geometry not in ("GEOMETRY_PLAUSIBLE_SIZE_DECLARED_NOT_VERIFIED", "GEOMETRY_PLAUSIBLE_SIZE_UNKNOWN"):
            reasons.append("INVALID_DECLARED_BBOX")
        if hint != "UNKNOWN_NO_AUTHORITATIVE_SPLIT":
            reasons.append(hint)
        statuses.update(reasons)
        candidates.append({
            "inventory_row": i,
            "record_id_hash": digest(clean(row.get("record_id"))),
            "image_identity_hash": digest(image) if image else "",
            "source_dataset_hash": digest(clean(row.get("source_dataset"))),
            "garment_id_hash": digest(clean(row.get("garment_id"))),
            "category_UNVERIFIED": clean(row.get("garment_category")),
            "bbox_status": geometry,
            "provenance_hint": hint,
            "exact_reference_overlap": "YES" if overlap_exact else "NO_OR_UNKNOWN",
            "basename_reference_overlap": "YES" if overlap_weak else "NO_OR_UNKNOWN",
            "eligibility": "QUARANTINED_NOT_ALLOWED_TO_EVALUATE",
            "reason_codes": ";".join(reasons),
        })

    as_csv(output_dir / "pairwise_overlap.csv", list(pairwise[0].keys()), pairwise)
    as_csv(output_dir / "inventory_quarantined_review.csv", list(candidates[0].keys()) if candidates else ["inventory_row","eligibility"], candidates)
    class_rows = [{"class": c, "inventory_records": Counter(clean(r.get("garment_category")) for r in processed["inventory"])[c], "verified_gt": "NO", "eligible_for_evaluation": "NO"} for c in CLASSES]
    as_csv(output_dir / "eight_class_inventory.csv", list(class_rows[0].keys()), class_rows)
    summary = {
        "schema": "prd32_actual_schema_provenance_audit_v3",
        "status": "METADATA_RECONCILED_PROVENANCE_NOT_CLEARED",
        "project_root": str(project_root),
        "source_image_and_garment_id_recognized": True,
        "image_files_read": 0,
        "model_inference_calls": 0,
        "holdout_images_opened": 0,
        "source_csv_metadata_read_only": True,
        "fullval_promoted_to_dev": False,
        "sources": per_source,
        "pairwise_overlap": pairwise,
        "inventory_rows_quarantined": len(candidates),
        "inventory_rows_approved_for_accuracy": 0,
        "reason_totals_not_mutually_exclusive": dict(sorted(statuses.items())),
        "next_requirements": [
            "Find authoritative training/validation/holdout source splits and exclude overlap at image identity or actual content SHA256 level",
            "Confirm source_dataset/source_image resolves to original image and record_id/garment_id instance identity",
            "Confirm whether garment_category is curated GT, 3.1 prediction, or label mapping; do not assume",
            "Do not open sealed holdout images for development or tuning",
            "Only after provenance clearance: build manually reviewed GT and paired reference-box vs frozen-V5 predicted-box datasets"
        ]
    }
    (output_dir / "source_provenance_audit_v3.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# PRD 3.2 八类来源审计 V3 — 实际字段识别", "", "本次仅读取三个 CSV 的元数据，不读取图片或加载模型。`source_image`、`garment_id`、完整 bbox 列已识别。", "", "| 数据源 | 记录数 | 去重图片标识数 | 角色 |", "|---|---:|---:|---|",]
    for key in SOURCES:
        s = per_source[key]
        lines.append(f"| {s['filename']} | {s['row_count']} | {s['unique_image_keys']} | {s['role']} |")
    lines.extend(["", "## 不可跳过的约束", "", "- 216 条 inventory 记录全部暂存隔离，不代表 216 张独立图片或 GT。", "- `fullval` 和 `matched` 仅用于来源与历史参考，不自动变成新开发集。", "- 精确键是 `source_dataset + source_image` 归一化路径；同名文件只能作为疑似重叠，不能认定内容相同。", "- 需要权威训练/验证划分和独立内容校验，才能确定哪些数据可以进入 3.2 开发或验证。", "", "## 当前跨文件标识重叠", ""])
    for row in pairwise:
        lines.append(f"- {row['source_a']} × {row['source_b']}: 精确图片键重叠 {row['exact_dataset_and_normalized_path_overlap_image_keys']}，数据集+文件名疑似重叠 {row['weak_dataset_and_basename_overlap_image_keys']}。")
    lines.extend(["", "## 后续必须提供的来源证据", "", "1. 3.1 训练时实际使用的图片清单或原始划分 manifest（只需元数据，不需要复制训练图片）。", "2. `source_dataset/source_image` 到图片的映射规则、独立验证集/封存集身份清单。", "3. `garment_category` 的标注来源（人工 GT、映射标签还是预测）。", "4. 在来源核实前不得运行新八类准确率或将 216 条记录认定为新测试集。", ""])
    (output_dir / "README_审计结论.md").write_text("\n".join(lines), encoding="utf-8")
    print("PROVENANCE_METADATA_AUDIT_COMPLETE")
    print(json.dumps({"inventory_rows":len(candidates),"image_ids":per_source["inventory"]["unique_image_keys"],"pairwise_overlap":pairwise,"approved":0,"output_dir":str(output_dir)},ensure_ascii=False,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", default="/workspace/fashion_multimodal_analysis")
    p.add_argument("--output", required=True)
    a=p.parse_args()
    audit(a.project_root,a.output)

if __name__=="__main__":
    main()
