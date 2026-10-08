"""基准集管理：保留训练、开发、固定回归/审核划分，冻结后的标签不能随结果调整。

Audit geometry / difficulty bias before freezing PRD 3.1.1 benchmark.

Why
---
The current review pool is leakage-safe, but category is strongly confounded
with source dataset and object size:
- DeepFashion2 clothing items are generally large.
- Fashionpedia shoe/accessory items may be very small.

Before human-reviewing 480 cases, quantify whether the raw Fashionpedia pool
contains enough larger / product-like examples to build a more representative
core acceptance set, while keeping tiny objects as a separate stress stratum.

Inputs
------
benchmark/prd_3_1_v1/candidates/segmentation_review_pool_v1/
    segmentation_review_pool_v1.csv
benchmark/prd_3_1_v1/audit/used_sample_registry.csv
../fashion_data/raw/fashionpedia/annotations/instances_attributes_val2020.json

Outputs
-------
benchmark/prd_3_1_v1/audit/difficulty_audit_v1/
├── current_review_pool_geometry.csv
├── current_review_pool_geometry_summary.csv
├── fashionpedia_raw_size_distribution.csv
└── difficulty_audit.txt

Run
---
cd fashion_multimodal_analysis
python scripts/benchmarking/audit_prd31_benchmark_difficulty_v1.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

PROJECT_ROOT = get_project_root()
DATA_ROOT = data_root()

BENCH_ROOT = PROJECT_ROOT / "benchmark" / "prd_3_1_v1"

POOL_CSV = (
    BENCH_ROOT
    / "candidates"
    / "segmentation_review_pool_v1"
    / "segmentation_review_pool_v1.csv"
)

USED_REGISTRY = BENCH_ROOT / "audit" / "used_sample_registry.csv"

FP_ANN = (
    DATA_ROOT
    / "raw"
    / "fashionpedia"
    / "annotations"
    / "instances_attributes_val2020.json"
)

OUT_DIR = BENCH_ROOT / "audit" / "difficulty_audit_v1"

FP_ID_TO_PRD = {
    13: "accessory",
    14: "accessory",
    15: "accessory",
    16: "accessory",
    17: "accessory",
    18: "accessory",
    19: "accessory",
    20: "accessory",
    21: "accessory",
    22: "accessory",
    23: "shoe",
    24: "bag",
    25: "accessory",
}

CLASSES = [
    "top",
    "pants",
    "skirt",
    "outerwear",
    "dress",
    "shoe",
    "bag",
    "accessory",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """按确定的字段顺序保存实验 CSV。

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


def f(v: Any) -> float:
    """f。

    Args:
        v: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    try:
        return float(v)
    except Exception:
        return math.nan


def quantile(vals: Any, q: Any) -> Any:
    """quantile。

    Args:
        vals: vals。
        q: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    vals = sorted(v for v in vals if not math.isnan(v))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]

    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return vals[lo]

    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def area_bin(ratio: float) -> str:
    # Fine-grained diagnostic bins.
    """面积 bin。

    Args:
        ratio: 比例。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if ratio < 0.005:
        return "tiny_<0.5%"
    if ratio < 0.02:
        return "small_0.5-2%"
    if ratio < 0.10:
        return "medium_2-10%"
    return "large_>=10%"


def minside_bin(px: float) -> str:
    """minside bin。

    Args:
        px: px。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if px < 20:
        return "minside_<20px"
    if px < 50:
        return "minside_20-49px"
    if px < 100:
        return "minside_50-99px"
    return "minside_>=100px"


def used_fashionpedia_image_names() -> set[str]:
    """used Fashionpedia 图像 names。

    Returns:
        返回 out，由函数体中同名变量的计算/收集过程得到。
    """
    rows = read_csv(USED_REGISTRY)
    out = set()

    for r in rows:
        source = str(r.get("source_image", "")).replace("\\", "/")
        if "fashionpedia" in source.lower():
            out.add(Path(source).name)

    return out


def audit_current_pool() -> tuple[Any, ...]:
    """检查 current 候选池。

    Returns:
        按顺序返回 per_rows, summary 等结果。
    """
    rows = read_csv(POOL_CSV)
    per_rows = []

    for r in rows:
        width = f(r.get("image_width"))
        height = f(r.get("image_height"))

        x1 = f(r.get("gt_bbox_x1"))
        y1 = f(r.get("gt_bbox_y1"))
        x2 = f(r.get("gt_bbox_x2"))
        y2 = f(r.get("gt_bbox_y2"))

        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)

        ratio = f(r.get("bbox_area_ratio"))

        if math.isnan(ratio) and width > 0 and height > 0:
            ratio = (bw * bh) / (width * height)

        min_side = min(bw, bh)
        max_side = max(bw, bh)
        aspect = max_side / min_side if min_side > 0 else math.inf

        per_rows.append(
            {
                "review_pool_sample_id": r.get("review_pool_sample_id", ""),
                "source_dataset": r.get("source_dataset", ""),
                "source_image": r.get("source_image", ""),
                "garment_id": r.get("garment_id", ""),
                "garment_category": r.get("garment_category", ""),
                "fine_or_source_category": r.get("fine_or_source_category", ""),
                "bbox_area_ratio": f"{ratio:.8f}",
                "bbox_area_percent": f"{100.0 * ratio:.4f}",
                "bbox_width_px": f"{bw:.2f}",
                "bbox_height_px": f"{bh:.2f}",
                "bbox_min_side_px": f"{min_side:.2f}",
                "bbox_aspect_extreme_ratio": f"{aspect:.4f}",
                "area_bin": area_bin(ratio),
                "minside_bin": minside_bin(min_side),
                "geometry_review_flag": int(
                    ratio < 0.005 or min_side < 20 or aspect > 5.0
                ),
            }
        )

    summary = []

    for cls in CLASSES:
        rr = [r for r in per_rows if r["garment_category"] == cls]

        ratios = [f(r["bbox_area_ratio"]) for r in rr]
        minsides = [f(r["bbox_min_side_px"]) for r in rr]

        area_counts = Counter(r["area_bin"] for r in rr)
        min_counts = Counter(r["minside_bin"] for r in rr)

        summary.append(
            {
                "garment_category": cls,
                "n": len(rr),
                "bbox_area_ratio_median": (f"{quantile(ratios, 0.50):.6f}"),
                "bbox_area_ratio_p10": (f"{quantile(ratios, 0.10):.6f}"),
                "bbox_area_ratio_p90": (f"{quantile(ratios, 0.90):.6f}"),
                "bbox_min_side_px_median": (f"{quantile(minsides, 0.50):.2f}"),
                "tiny_lt_0_5pct": area_counts["tiny_<0.5%"],
                "small_0_5_to_2pct": area_counts["small_0.5-2%"],
                "medium_2_to_10pct": area_counts["medium_2-10%"],
                "large_ge_10pct": area_counts["large_>=10%"],
                "minside_lt_20px": min_counts["minside_<20px"],
                "geometry_review_flags": sum(
                    int(r["geometry_review_flag"]) for r in rr
                ),
            }
        )

    return per_rows, summary


def audit_fashionpedia_raw() -> Any:
    """检查 Fashionpedia raw。

    Returns:
        返回 out，由函数体中同名变量的计算/收集过程得到。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    if not FP_ANN.is_file():
        raise FileNotFoundError(FP_ANN)

    payload = json.loads(FP_ANN.read_text(encoding="utf-8"))

    images = {int(i["id"]): i for i in payload["images"]}

    categories = {
        int(c["id"]): str(c.get("name", "")).strip() for c in payload["categories"]
    }

    excluded_names = used_fashionpedia_image_names()

    counts = defaultdict(Counter)
    unique_images = defaultdict(set)
    fine_counts = defaultdict(Counter)

    for ann in payload["annotations"]:
        try:
            cid = int(ann["category_id"])
        except Exception:
            continue

        cls = FP_ID_TO_PRD.get(cid)
        if cls is None:
            continue

        image_info = images.get(int(ann["image_id"]))
        if not image_info:
            continue

        file_name = str(image_info.get("file_name", "")).strip()

        if not file_name or file_name in excluded_names:
            continue

        bbox = ann.get("bbox")

        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        try:
            x, y, bw, bh = map(float, bbox)
            width = float(image_info.get("width", 0))
            height = float(image_info.get("height", 0))
        except Exception:
            continue

        if bw <= 0 or bh <= 0 or width <= 0 or height <= 0:
            continue

        ratio = (bw * bh) / (width * height)
        min_side = min(bw, bh)

        counts[cls][area_bin(ratio)] += 1
        counts[cls][minside_bin(min_side)] += 1
        counts[cls]["instances"] += 1
        unique_images[cls].add(file_name)
        fine_counts[cls][categories.get(cid, str(cid))] += 1

    out = []

    for cls in ["shoe", "bag", "accessory"]:
        c = counts[cls]

        out.append(
            {
                "garment_category": cls,
                "eligible_instances_after_used_image_exclusion": c["instances"],
                "unique_source_images": len(unique_images[cls]),
                "tiny_lt_0_5pct": c["tiny_<0.5%"],
                "small_0_5_to_2pct": c["small_0.5-2%"],
                "medium_2_to_10pct": c["medium_2-10%"],
                "large_ge_10pct": c["large_>=10%"],
                "minside_lt_20px": c["minside_<20px"],
                "minside_20_49px": c["minside_20-49px"],
                "minside_50_99px": c["minside_50-99px"],
                "minside_ge_100px": c["minside_>=100px"],
                "fine_category_distribution": "; ".join(
                    f"{k}={v}" for k, v in sorted(fine_counts[cls].items())
                ),
            }
        )

    return out


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    current_rows, current_summary = audit_current_pool()
    fp_summary = audit_fashionpedia_raw()

    write_csv(
        OUT_DIR / "current_review_pool_geometry.csv",
        current_rows,
    )

    write_csv(
        OUT_DIR / "current_review_pool_geometry_summary.csv",
        current_summary,
    )

    write_csv(
        OUT_DIR / "fashionpedia_raw_size_distribution.csv",
        fp_summary,
    )

    by_class = {r["garment_category"]: r for r in current_summary}

    lines = [
        "PRD 3.1.1 Benchmark Difficulty / Geometry Audit v1",
        "===================================================",
        "",
        "Current review pool",
        "-------------------",
    ]

    for cls in CLASSES:
        r = by_class[cls]
        lines.append(
            f"{cls}: n={r['n']} "
            + f"median_area_ratio={r['bbox_area_ratio_median']} "
            + f"tiny={r['tiny_lt_0_5pct']} "
            + f"small={r['small_0_5_to_2pct']} "
            + f"medium={r['medium_2_to_10pct']} "
            + f"large={r['large_ge_10pct']} "
            + f"minside<20px={r['minside_lt_20px']} "
            + f"geometry_flags={r['geometry_review_flags']}"
        )

    lines += [
        "",
        "Fashionpedia raw leakage-safe pool",
        "----------------------------------",
    ]

    for r in fp_summary:
        lines.append(
            f"{r['garment_category']}: "
            + f"instances={r['eligible_instances_after_used_image_exclusion']} "
            + f"unique_images={r['unique_source_images']} "
            + f"tiny={r['tiny_lt_0_5pct']} "
            + f"small={r['small_0_5_to_2pct']} "
            + f"medium={r['medium_2_to_10pct']} "
            + f"large={r['large_ge_10pct']} "
            + f"minside<20px={r['minside_lt_20px']}"
        )

    lines += [
        "",
        "Decision principle",
        "------------------",
        "- Do NOT freeze the current 480 solely because class counts are balanced.",
        "- Check whether class is confounded with object size/source dataset.",
        "- Prefer a Core Acceptance set with reviewable/product-representative visibility.",
        "- Preserve tiny/small cases as a separate Stress subset rather than letting them dominate one category.",
        "- After difficulty design is fixed, generate mask-overlay review assets and perform human annotation review.",
    ]

    (OUT_DIR / "difficulty_audit.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("FINISHED")
    for p in [
        OUT_DIR / "difficulty_audit.txt",
        OUT_DIR / "current_review_pool_geometry_summary.csv",
        OUT_DIR / "fashionpedia_raw_size_distribution.csv",
        OUT_DIR / "current_review_pool_geometry.csv",
    ]:
        print(p.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
