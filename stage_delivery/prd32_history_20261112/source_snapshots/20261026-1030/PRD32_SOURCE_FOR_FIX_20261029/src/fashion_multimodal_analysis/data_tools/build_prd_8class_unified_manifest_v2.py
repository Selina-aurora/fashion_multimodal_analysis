"""数据处理：保存原图来源和实例标识，避免同一图片的不同实例跨训练与评估划分。

Corrected PRD 3.1.1 8-class unified manifest validator.

Fix in v2:
- DeepFashion2 garment_id values such as "item1_sling_dress" are NOT globally unique.
- Uniqueness is therefore checked using:
      source_dataset + source_image + garment_id
  instead of:
      source_dataset + garment_id

Inputs:
    configs/garment_instances_gt_pilot_100.csv
    configs/fashionpedia_missing3_pilot_manifest_v1.csv

Outputs:
    configs/prd_8class_unified_manifest_v2.csv

    reports/prd_instance_segmentation/prd_8class_unified_v2/
        summary.txt
        class_counts.csv
        validation_issues.csv

Run:
    python scripts/data_tools/build_prd_8class_unified_manifest_v2.py
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

PROJECT_ROOT = get_project_root()

DEEPFASHION2_CSV = PROJECT_ROOT / "configs" / "garment_instances_gt_pilot_100.csv"

FASHIONPEDIA_CSV = (
    PROJECT_ROOT / "configs" / "fashionpedia_missing3_pilot_manifest_v1.csv"
)

OUTPUT_CSV = PROJECT_ROOT / "configs" / "prd_8class_unified_manifest_v2.csv"

REPORT_DIR = (
    PROJECT_ROOT / "reports" / "prd_instance_segmentation" / "prd_8class_unified_v2"
)

SUMMARY_PATH = REPORT_DIR / "summary.txt"
CLASS_COUNTS_PATH = REPORT_DIR / "class_counts.csv"
ISSUES_PATH = REPORT_DIR / "validation_issues.csv"

PRD_CLASSES = (
    "top",
    "pants",
    "skirt",
    "outerwear",
    "dress",
    "shoe",
    "bag",
    "accessory",
)

DEEPFASHION2_CLASSES = {
    "top",
    "pants",
    "skirt",
    "outerwear",
    "dress",
}

FASHIONPEDIA_CLASSES = {
    "shoe",
    "bag",
    "accessory",
}

OUTPUT_FIELDS = [
    "record_id",
    "source_dataset",
    "source_image",
    "garment_id",
    "garment_category",
    "fine_category",
    "source_category_id",
    "annotation_ref",
    "crop_path",
    "mask_path",
    "masked_preview_path",
    "visualization_path",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_width",
    "bbox_height",
    "image_width",
    "image_height",
    "source_manifest",
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
        raise FileNotFoundError(f"Missing input CSV:\n  {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    """按确定的字段顺序保存实验 CSV。

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。
        fields: 输出字段或分组字段的顺序。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def to_int_string(value: str | None) -> str:
    """转换 int string。

    Args:
        value: 待解析或规范化的输入值。

    Returns:
        返回 value，由函数体中同名变量的计算/收集过程得到。
    """
    if value is None:
        return ""

    value = str(value).strip()
    if value == "":
        return ""

    try:
        return str(int(float(value)))
    except ValueError:
        return value


def bbox_size(
    x1: str,
    y1: str,
    x2: str,
    y2: str,
) -> tuple[str, str]:
    """边界框 size。

    Args:
        x1: x1。
        y1: y1。
        x2: x2。
        y2: y2。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    try:
        w = int(float(x2)) - int(float(x1))
        h = int(float(y2)) - int(float(y1))
        return str(w), str(h)
    except (TypeError, ValueError):
        return "", ""


def normalize_deepfashion2(
    rows: list[dict[str, str]],
) -> tuple[list[dict], list[dict]]:
    """规范化 DeepFashion2。

    Args:
        rows: 待处理的逐行记录。

    Returns:
        按顺序返回 normalized, issues 等结果。
    """
    normalized = []
    issues = []

    for idx, row in enumerate(rows, start=1):
        category = str(row.get("garment_category", "")).strip().lower()

        if category not in DEEPFASHION2_CLASSES:
            issues.append(
                {
                    "source_dataset": "DeepFashion2",
                    "row_number": idx + 1,
                    "garment_id": row.get("garment_id", ""),
                    "issue": f"unexpected garment_category={category!r}",
                }
            )

        x1 = to_int_string(row.get("bbox_x1"))
        y1 = to_int_string(row.get("bbox_y1"))
        x2 = to_int_string(row.get("bbox_x2"))
        y2 = to_int_string(row.get("bbox_y2"))
        bbox_w, bbox_h = bbox_size(x1, y1, x2, y2)

        normalized.append(
            {
                "record_id": "",
                "source_dataset": "DeepFashion2",
                "source_image": row.get("source_image", ""),
                "garment_id": row.get("garment_id", ""),
                "garment_category": category,
                "fine_category": row.get("fine_category", ""),
                "source_category_id": row.get("category_id", ""),
                "annotation_ref": row.get("annotation_item", ""),
                "crop_path": row.get("crop_path", ""),
                "mask_path": row.get("mask_path", ""),
                "masked_preview_path": row.get("masked_preview_path", ""),
                "visualization_path": "",
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "bbox_width": bbox_w,
                "bbox_height": bbox_h,
                "image_width": "",
                "image_height": "",
                "source_manifest": "configs/garment_instances_gt_pilot_100.csv",
            }
        )

    return normalized, issues


def normalize_fashionpedia(
    rows: list[dict[str, str]],
) -> tuple[list[dict], list[dict]]:
    """规范化 Fashionpedia。

    Args:
        rows: 待处理的逐行记录。

    Returns:
        按顺序返回 normalized, issues 等结果。
    """
    normalized = []
    issues = []

    for idx, row in enumerate(rows, start=1):
        category = str(row.get("garment_category", "")).strip().lower()

        if category not in FASHIONPEDIA_CLASSES:
            issues.append(
                {
                    "source_dataset": "Fashionpedia",
                    "row_number": idx + 1,
                    "garment_id": row.get("garment_id", ""),
                    "issue": f"unexpected garment_category={category!r}",
                }
            )

        normalized.append(
            {
                "record_id": "",
                "source_dataset": "Fashionpedia",
                "source_image": row.get("source_image", ""),
                "garment_id": row.get("garment_id", ""),
                "garment_category": category,
                "fine_category": row.get("source_category", ""),
                "source_category_id": "",
                "annotation_ref": row.get("annotation_id", ""),
                "crop_path": row.get("crop_path", ""),
                "mask_path": row.get("mask_path", ""),
                "masked_preview_path": row.get("masked_crop_path", ""),
                "visualization_path": row.get("visualization_path", ""),
                "bbox_x1": to_int_string(row.get("bbox_x1")),
                "bbox_y1": to_int_string(row.get("bbox_y1")),
                "bbox_x2": to_int_string(row.get("bbox_x2")),
                "bbox_y2": to_int_string(row.get("bbox_y2")),
                "bbox_width": to_int_string(row.get("bbox_width")),
                "bbox_height": to_int_string(row.get("bbox_height")),
                "image_width": to_int_string(row.get("image_width")),
                "image_height": to_int_string(row.get("image_height")),
                "source_manifest": "configs/fashionpedia_missing3_pilot_manifest_v1.csv",
            }
        )

    return normalized, issues


def validate_rows(rows: list[dict]) -> list[dict]:
    """校验 逐行记录。

    Args:
        rows: 待处理的逐行记录。

    Returns:
        当前条件的校验结果；失败条件及返回形式见函数体。
    """
    issues = []

    # Correct composite identity:
    # same garment_id string may legitimately repeat on different images.
    seen_instance_keys = set()

    for idx, row in enumerate(rows, start=1):
        dataset = row["source_dataset"]
        garment_id = row["garment_id"]
        source_image = row["source_image"]

        instance_key = (
            dataset,
            source_image,
            garment_id,
        )

        if instance_key in seen_instance_keys:
            issues.append(
                {
                    "source_dataset": dataset,
                    "row_number": idx + 1,
                    "garment_id": garment_id,
                    "issue": ("duplicate source_dataset+source_image+garment_id"),
                }
            )

        seen_instance_keys.add(instance_key)

        if row["garment_category"] not in PRD_CLASSES:
            issues.append(
                {
                    "source_dataset": dataset,
                    "row_number": idx + 1,
                    "garment_id": garment_id,
                    "issue": (
                        "category outside PRD 8 classes: "
                        + f"{row['garment_category']!r}"
                    ),
                }
            )

        for field in (
            "source_image",
            "garment_id",
            "garment_category",
            "mask_path",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
        ):
            if str(row.get(field, "")).strip() == "":
                issues.append(
                    {
                        "source_dataset": dataset,
                        "row_number": idx + 1,
                        "garment_id": garment_id,
                        "issue": f"missing required field: {field}",
                    }
                )

        try:
            x1 = int(float(row["bbox_x1"]))
            y1 = int(float(row["bbox_y1"]))
            x2 = int(float(row["bbox_x2"]))
            y2 = int(float(row["bbox_y2"]))

            if x2 <= x1 or y2 <= y1:
                issues.append(
                    {
                        "source_dataset": dataset,
                        "row_number": idx + 1,
                        "garment_id": garment_id,
                        "issue": ("invalid bbox ordering: " + f"({x1},{y1},{x2},{y2})"),
                    }
                )
        except (TypeError, ValueError):
            issues.append(
                {
                    "source_dataset": dataset,
                    "row_number": idx + 1,
                    "garment_id": garment_id,
                    "issue": "bbox contains non-numeric value",
                }
            )

    return issues


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    deep_rows_raw = read_csv(DEEPFASHION2_CSV)
    fp_rows_raw = read_csv(FASHIONPEDIA_CSV)

    print("Input rows:")
    print(f"  DeepFashion2: {len(deep_rows_raw)}")
    print(f"  Fashionpedia: {len(fp_rows_raw)}")

    deep_rows, deep_issues = normalize_deepfashion2(deep_rows_raw)
    fp_rows, fp_issues = normalize_fashionpedia(fp_rows_raw)

    unified = deep_rows + fp_rows

    for idx, row in enumerate(unified, start=1):
        row["record_id"] = f"prd8_{idx:04d}"

    validation_issues = deep_issues + fp_issues + validate_rows(unified)

    counts = Counter(row["garment_category"] for row in unified)

    dataset_counts = Counter(row["source_dataset"] for row in unified)

    missing_classes = [cls for cls in PRD_CLASSES if counts.get(cls, 0) == 0]

    write_csv(
        OUTPUT_CSV,
        unified,
        OUTPUT_FIELDS,
    )

    class_count_rows = [
        {
            "garment_category": cls,
            "count": counts.get(cls, 0),
        }
        for cls in PRD_CLASSES
    ]

    write_csv(
        CLASS_COUNTS_PATH,
        class_count_rows,
        ["garment_category", "count"],
    )

    write_csv(
        ISSUES_PATH,
        validation_issues,
        [
            "source_dataset",
            "row_number",
            "garment_id",
            "issue",
        ],
    )

    summary = [
        "PRD 3.1.1 unified 8-class manifest v2",
        "===================================",
        "",
        "Inputs",
        "------",
        f"DeepFashion2 rows={len(deep_rows_raw)}",
        f"Fashionpedia rows={len(fp_rows_raw)}",
        f"Unified rows={len(unified)}",
        "",
        "Dataset counts",
        "--------------",
        f"DeepFashion2={dataset_counts.get('DeepFashion2', 0)}",
        f"Fashionpedia={dataset_counts.get('Fashionpedia', 0)}",
        "",
        "PRD class counts",
        "----------------",
    ]

    for cls in PRD_CLASSES:
        summary.append(f"{cls}={counts.get(cls, 0)}")

    summary += [
        "",
        f"PRD class coverage={len(PRD_CLASSES) - len(missing_classes)}/8",
        f"missing_classes={','.join(missing_classes) if missing_classes else 'none'}",
        f"validation_issues={len(validation_issues)}",
        "",
        "Validation note",
        "---------------",
        "- DeepFashion2 garment_id is only unique within an image.",
        "- Duplicate checking therefore uses source_dataset + source_image + garment_id.",
        "",
        "Important interpretation",
        "------------------------",
        "- This establishes unified DATA COVERAGE for the 8 PRD categories.",
        "- It does NOT prove the segmentation model meets final PRD accuracy/speed targets.",
        "- DeepFashion2 supplies top/pants/skirt/outerwear/dress.",
        "- Fashionpedia supplies shoe/bag/accessory.",
        "- Keep source_dataset visible during splitting/evaluation.",
    ]

    SUMMARY_PATH.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print()
    print("Class counts:")
    for cls in PRD_CLASSES:
        print(f"  {cls}: {counts.get(cls, 0)}")

    print()
    print(f"Unified rows: {len(unified)}")
    print(f"Coverage: {len(PRD_CLASSES) - len(missing_classes)}/8")
    print(f"Validation issues: {len(validation_issues)}")
    print()
    print("Wrote:")
    print(f"  {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  {CLASS_COUNTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  {ISSUES_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
