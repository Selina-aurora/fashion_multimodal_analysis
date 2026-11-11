"""基准集管理：保留训练、开发、固定回归/审核划分，冻结后的标签不能随结果调整。

Discover untouched raw-data pools for PRD 3.1 benchmark_test v1.

This is a READ-ONLY inventory step. It does not freeze the benchmark yet.

It:
1) reads benchmark/prd_3_1_v1/audit/used_sample_registry.csv;
2) scans DeepFashion2 raw annotations and counts UNUSED candidates for the
   five clothing classes used by the current project;
3) scans Fashionpedia annotation JSON files and reports COCO-style category
   names/counts so shoe/bag/accessory can be mapped safely before sampling;
4) writes only audit reports.

Run
---
cd fashion_multimodal_analysis

python scripts/benchmarking/discover_prd31_raw_benchmark_pool_v1.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
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

USED_REGISTRY = (
    PROJECT_ROOT / "benchmark" / "prd_3_1_v1" / "audit" / "used_sample_registry.csv"
)

OUT_DIR = PROJECT_ROOT / "benchmark" / "prd_3_1_v1" / "audit" / "raw_pool_discovery_v1"

DF2_ANNO_DIR = DATA_ROOT / "raw" / "train" / "train" / "annos"
DF2_IMAGE_DIR = DATA_ROOT / "raw" / "train" / "train" / "image"
FASHIONPEDIA_ROOT = DATA_ROOT / "raw" / "fashionpedia"

# DeepFashion2 official fine-category IDs/names.
DF2_FINE = {
    1: "short_sleeve_top",
    2: "long_sleeve_top",
    3: "short_sleeve_outwear",
    4: "long_sleeve_outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short_sleeve_dress",
    11: "long_sleeve_dress",
    12: "vest_dress",
    13: "sling_dress",
}

DF2_TO_PRD = {
    1: "top",
    2: "top",
    3: "outerwear",
    4: "outerwear",
    5: "top",
    6: "top",
    7: "pants",
    8: "pants",
    9: "skirt",
    10: "dress",
    11: "dress",
    12: "dress",
    13: "dress",
}


def parse_args() -> Any:
    """解析并校验命令行参数。

    Returns:
        已通过参数合法性检查的命令行配置。
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--max-df2-files",
        type=int,
        default=0,
        help="0 = scan all DeepFashion2 annotation files.",
    )
    p.add_argument(
        "--max-fashionpedia-jsons",
        type=int,
        default=0,
        help="0 = inspect all JSON files under raw/fashionpedia.",
    )
    return p.parse_args()


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
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def norm_path(v: str) -> str:
    """规范化路径分隔符，兼容 Windows 与 Linux 清单。

    Args:
        v: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return str(v).replace("\\", "/").strip()


def project_source_path(abs_path: Path) -> str:
    """Match existing project convention:
    ../fashion_data/...

    Args:
        abs_path: 对应文件的相对路径或当前解析后的路径。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    try:
        rel = abs_path.resolve().relative_to(DATA_ROOT.resolve())
        return f"../fashion_data/{str(rel).replace(chr(92), '/')}"
    except Exception:
        return norm_path(str(abs_path.resolve()))


def load_used() -> set[tuple[str, str]]:
    """加载 used。

    Returns:
        当前读取操作得到的记录、图像或模型对象；保持原始格式约定。
    """
    rows = read_csv(USED_REGISTRY)
    return {
        (
            norm_path(r.get("source_image", "")),
            str(r.get("garment_id", "")).strip(),
        )
        for r in rows
        if str(r.get("source_image", "")).strip()
        and str(r.get("garment_id", "")).strip()
    }


def scan_deepfashion2(
    used: set[tuple[str, str]],
    max_files: int,
) -> tuple[Any, ...]:
    """扫描 DeepFashion2。

    Args:
        used: used。
        max_files: max 文件。

    Returns:
        按顺序返回 summary_rows, previews, meta 等结果。
    """
    if not DF2_ANNO_DIR.is_dir():
        return (
            [],
            [],
            {
                "status": "MISSING",
                "anno_dir": str(DF2_ANNO_DIR),
                "image_dir": str(DF2_IMAGE_DIR),
            },
        )

    files = sorted(DF2_ANNO_DIR.glob("*.json"))
    if max_files > 0:
        files = files[:max_files]

    counts_total = Counter()
    counts_used = Counter()
    counts_unused = Counter()
    fine_unused = Counter()
    previews = []
    errors = 0
    missing_images = 0

    for i, ann_path in enumerate(files, start=1):
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception:
            errors += 1
            continue

        image_path = DF2_IMAGE_DIR / f"{ann_path.stem}.jpg"
        if not image_path.is_file():
            # Some distributions may use another image suffix.
            candidates = list(DF2_IMAGE_DIR.glob(f"{ann_path.stem}.*"))
            if candidates:
                image_path = candidates[0]
            else:
                missing_images += 1
                continue

        source_image = project_source_path(image_path)

        for item_key, item in data.items():
            if not str(item_key).startswith("item"):
                continue
            if not isinstance(item, dict):
                continue

            try:
                category_id = int(item.get("category_id"))
            except Exception:
                continue

            prd_class = DF2_TO_PRD.get(category_id)
            fine = DF2_FINE.get(category_id)

            if not prd_class or not fine:
                continue

            garment_id = f"{item_key}_{fine}"
            key = (source_image, garment_id)

            counts_total[prd_class] += 1

            if key in used:
                counts_used[prd_class] += 1
                continue

            counts_unused[prd_class] += 1
            fine_unused[fine] += 1

            bbox = item.get("bounding_box", [])
            seg = item.get("segmentation", [])

            if len(previews) < 200:
                previews.append(
                    {
                        "source_dataset": "DeepFashion2",
                        "source_image": source_image,
                        "garment_id": garment_id,
                        "garment_category": prd_class,
                        "fine_category": fine,
                        "category_id": category_id,
                        "gt_bbox": json.dumps(bbox, ensure_ascii=False),
                        "has_segmentation": int(bool(seg)),
                        "annotation_file": project_source_path(ann_path),
                    }
                )

        if i % 10000 == 0:
            print(f"DeepFashion2 scanned {i}/{len(files)} annotation files...")

    summary_rows = []
    for cls in ["top", "pants", "skirt", "outerwear", "dress"]:
        summary_rows.append(
            {
                "source_dataset": "DeepFashion2",
                "garment_category": cls,
                "raw_total": counts_total.get(cls, 0),
                "already_used": counts_used.get(cls, 0),
                "unused_candidates": counts_unused.get(cls, 0),
            }
        )

    meta = {
        "status": "OK",
        "annotation_files_scanned": len(files),
        "parse_errors": errors,
        "missing_images": missing_images,
        "unused_total": sum(counts_unused.values()),
        "unused_fine_distribution": dict(sorted(fine_unused.items())),
    }
    return summary_rows, previews, meta


def scan_fashionpedia(max_jsons: int) -> tuple[Any, ...]:
    """扫描 Fashionpedia。

    Args:
        max_jsons: max jsons。

    Returns:
        按顺序返回 rows, meta 等结果。
    """
    rows = []

    if not FASHIONPEDIA_ROOT.exists():
        return rows, {
            "status": "MISSING",
            "root": str(FASHIONPEDIA_ROOT),
        }

    json_files = sorted(FASHIONPEDIA_ROOT.rglob("*.json"))
    if max_jsons > 0:
        json_files = json_files[:max_jsons]

    inspected = 0
    errors = 0

    for path in json_files:
        inspected += 1
        size_mb = path.stat().st_size / (1024 * 1024)

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            errors += 1
            rows.append(
                {
                    "annotation_json": project_source_path(path),
                    "size_mb": f"{size_mb:.2f}",
                    "format": "unreadable",
                    "images": "",
                    "annotations": "",
                    "categories": "",
                    "category_names": "",
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            continue

        if (
            isinstance(data, dict)
            and isinstance(data.get("images"), list)
            and isinstance(data.get("annotations"), list)
            and isinstance(data.get("categories"), list)
        ):
            cats = data.get("categories", [])
            cat_names = []
            cat_count_by_id = Counter()

            for a in data.get("annotations", []):
                try:
                    cat_count_by_id[int(a.get("category_id"))] += 1
                except Exception:
                    # 保留历史的跳过策略，但记录坏输入，避免扫描过程静默丢失标注。
                    logging.getLogger(__name__).warning(
                        "无法解析当前标注，已跳过并继续扫描", exc_info=True
                    )

            cat_parts = []
            for c in cats:
                cid = c.get("id", "")
                name = str(c.get("name", "")).strip()
                supercat = str(c.get("supercategory", "")).strip()

                cat_names.append(name)

                try:
                    cnt = cat_count_by_id.get(int(cid), 0)
                except Exception:
                    cnt = 0

                cat_parts.append(f"{cid}:{name}:{supercat}:{cnt}")

            rows.append(
                {
                    "annotation_json": project_source_path(path),
                    "size_mb": f"{size_mb:.2f}",
                    "format": "COCO-like",
                    "images": len(data.get("images", [])),
                    "annotations": len(data.get("annotations", [])),
                    "categories": len(cats),
                    "category_names": " | ".join(cat_names),
                    "category_id_name_supercat_count": " | ".join(cat_parts),
                    "error": "",
                }
            )
        else:
            keys = list(data.keys())[:20] if isinstance(data, dict) else []
            rows.append(
                {
                    "annotation_json": project_source_path(path),
                    "size_mb": f"{size_mb:.2f}",
                    "format": type(data).__name__,
                    "images": "",
                    "annotations": "",
                    "categories": "",
                    "category_names": "",
                    "top_level_keys": " | ".join(map(str, keys)),
                    "error": "",
                }
            )

    meta = {
        "status": "OK",
        "json_files_found": len(json_files),
        "json_files_inspected": inspected,
        "parse_errors": errors,
    }
    return rows, meta


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    used = load_used()
    print(f"used registry samples: {len(used)}")

    df2_summary, df2_preview, df2_meta = scan_deepfashion2(
        used,
        args.max_df2_files,
    )
    write_csv(
        OUT_DIR / "deepfashion2_unused_candidate_counts.csv",
        df2_summary,
    )
    write_csv(
        OUT_DIR / "deepfashion2_unused_candidates_preview.csv",
        df2_preview,
    )

    fp_rows, fp_meta = scan_fashionpedia(args.max_fashionpedia_jsons)
    write_csv(
        OUT_DIR / "fashionpedia_annotation_inventory.csv",
        fp_rows,
    )

    lines = [
        "PRD 3.1 Raw Benchmark Pool Discovery v1",
        "======================================",
        "",
        f"data_root={DATA_ROOT}",
        f"used_registry_samples={len(used)}",
        "",
        "DeepFashion2",
        "------------",
    ]

    for k, v in df2_meta.items():
        lines.append(f"{k}={v}")

    if df2_summary:
        lines.append("")
        lines.append("Unused DeepFashion2 candidates by PRD class")
        for r in df2_summary:
            lines.append(
                f"{r['garment_category']}: "
                + f"raw_total={r['raw_total']} "
                + f"already_used={r['already_used']} "
                + f"unused={r['unused_candidates']}"
            )

    lines += [
        "",
        "Fashionpedia",
        "------------",
    ]

    for k, v in fp_meta.items():
        lines.append(f"{k}={v}")

    lines += [
        "",
        "Next decision",
        "-------------",
        "- Use unused DeepFashion2 for top/pants/skirt/outerwear/dress.",
        "- Inspect fashionpedia_annotation_inventory.csv before defining the shoe/bag/accessory mapping.",
        "- Do not freeze or sample the blind benchmark until both source pools are confirmed.",
    ]

    (OUT_DIR / "raw_pool_discovery.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("FINISHED")
    for p in [
        OUT_DIR / "raw_pool_discovery.txt",
        OUT_DIR / "deepfashion2_unused_candidate_counts.csv",
        OUT_DIR / "deepfashion2_unused_candidates_preview.csv",
        OUT_DIR / "fashionpedia_annotation_inventory.csv",
    ]:
        print(p.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
