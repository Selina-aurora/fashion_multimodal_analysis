"""结果分析：依据已有逐例记录定位问题，不替代新的模型评估。

Audit train/validation/Core overlap by normalized source image path.
"""

from __future__ import annotations

import argparse
import csv
import json
import posixpath
from collections import Counter, defaultdict
from pathlib import Path

from fashion_multimodal_analysis.common.paths import project_root, resolve_path

CORE = "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv"


def image_key(value: str) -> str:
    """图像 标识。

    Args:
        value: 待解析或规范化的输入值。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    value = value.strip().replace("\\", "/")
    prefix = "/workspace/fashion_data/"
    if value.startswith(prefix):
        value = "../fashion_data/" + value[len(prefix) :]
    return posixpath.normpath(value)


def read_rows(path: Path) -> list[dict]:
    """读取 逐行记录。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        当前读取操作得到的记录、图像或模型对象；保持原始格式约定。
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit(root: Path) -> tuple[dict, list[dict]]:
    """检查。

    Args:
        root: 当前项目根目录。

    Returns:
        按顺序返回 result, overlaps 等结果。
    """
    core_rows = read_rows(root / CORE)
    val_rows = read_rows(root / "configs/prd_8class_val_v1.csv")
    core = defaultdict(list)
    for row in core_rows:
        core[image_key(row["source_image"])].append(row)
    val_images = {image_key(row["source_image"]) for row in val_rows}
    result = {
        "method": "Exact normalized source_image path; no content/perceptual duplicate-image detection.",
        "core_instances": len(core_rows),
        "core_images": len(core),
        "core_class_counts": dict(Counter(r["garment_category"] for r in core_rows)),
        "core_source_counts": dict(Counter(r["source_dataset"] for r in core_rows)),
        "validation_instances": len(val_rows),
        "validation_images": len(val_images),
        "validation_core_overlap_images": len(val_images & core.keys()),
        "training": {},
    }
    overlaps = []
    for version in (1, 2, 3):
        manifest = f"configs/prd_8class_train_v{version}.csv"
        rows = read_rows(root / manifest)
        images = {image_key(row["source_image"]) for row in rows}
        overlap_images = images & core.keys()
        result["training"][f"v{version}"] = {
            "manifest": manifest,
            "instances": len(rows),
            "images": len(images),
            "class_counts": dict(Counter(r["garment_category"] for r in rows)),
            "source_counts": dict(Counter(r["source_dataset"] for r in rows)),
            "core_overlap_images": len(overlap_images),
            "train_validation_overlap_images": len(images & val_images),
            "overlapping_core_class_counts": dict(
                Counter(
                    r["garment_category"]
                    for key in sorted(overlap_images)
                    for r in core[key]
                )
            ),
        }
        for row in rows:
            key = image_key(row["source_image"])
            for match in core.get(key, []):
                overlaps.append(
                    {
                        "train_manifest": manifest,
                        "train_record_id": row.get("record_id", ""),
                        "train_category": row["garment_category"],
                        "train_annotation": row.get("annotation_ref", ""),
                        "core_sample_id": match["final_sample_id"],
                        "core_category": match["garment_category"],
                        "core_annotation": match.get("annotation_id", "")
                        or match.get("garment_id", ""),
                        "source_image": key,
                    }
                )
    result["status"] = "OVERLAP_FOUND" if overlaps else "NO_PATH_OVERLAP_FOUND"
    return result, overlaps


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        SystemExit: 执行本函数的操作失败。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/repository_audit/data_overlap")
    parser.add_argument(
        "--fail-on-overlap",
        action="store_true",
        help="Return exit code 2 if any train/Core source image overlaps",
    )
    args = parser.parse_args()
    result, rows = audit(project_root())
    destination = resolve_path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "train_manifest",
        "train_record_id",
        "train_category",
        "train_annotation",
        "core_sample_id",
        "core_category",
        "core_annotation",
        "source_image",
    ]
    with (destination / "train_core_overlaps.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.fail_on_overlap and rows:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
