"""3.1.3 可视属性提取：关注颜色、图案和服装设计/几何属性；一致性不等于人工真值准确率。

PRD 3.1.3 unified garment attribute output v1.

Purpose
-------
Merge the three now-stable garment-level outputs into one integration artifact:

1) continuous style features
2) pattern v3 hierarchical baseline
3) primary color baseline v1

The merge key is:
    (source_image, garment_id)

No pandas dependency is required.

Expected inputs
---------------
A. style JSON:
   style_feature_vector_v1.json
   Root should contain a list named "garments", where each item includes:
       source_image
       garment_id
       garment_category
       style_features
       quality

B. pattern v3 CSV:
   reports/prd_attribute_extraction/pattern_v3_hierarchical/
       pattern_v3_predictions.csv

C. primary color CSV:
   reports/prd_attribute_extraction/color_v1/
       primary_color_predictions.csv

Outputs
-------
reports/prd_attribute_extraction/unified_v1/
    garment_attribute_vector_v1.json
    garment_attribute_vector_v1_flat.csv
    integration_summary.txt
    unmatched_records.csv
    run_info.txt

Example
-------
python scripts/attributes/design/build_unified_garment_attribute_vector_v1.py
--style-json reports/prd_attribute_extraction/style_feature_vector_v1.json
--pattern-csv
reports/prd_attribute_extraction/pattern_v3_hierarchical/pattern_v3_predictions.csv
--color-csv reports/prd_attribute_extraction/color_v1/primary_color_predictions.csv

If your style JSON is in a different folder, only change --style-json.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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

OUTPUT_DIR = PROJECT_ROOT / "reports" / "prd_attribute_extraction" / "unified_v1"


def resolve_path(raw: Any) -> Path:
    """将清单或配置中的相对路径解析到当前项目/数据目录。

    Args:
        raw: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        当前工作副本可使用的路径。
    """
    return resolve_artifact_path(raw)


def normalize_source_image(value: str) -> str:
    # Keep paths portable while making slash direction consistent.
    """规范化原图引用，便于清单连接与重复样本检查。

    Args:
        value: 待解析或规范化的输入值。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return str(value).replace("\\", "/").strip()


def make_key(
    source_image: str,
    garment_id: str,
) -> tuple[str, str]:
    """生成 标识。

    Args:
        source_image: 来源 图像。
        garment_id: garment ID。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return (
        normalize_source_image(source_image),
        garment_id.strip(),
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。
    """
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    """读取 JSON。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        当前读取操作得到的记录、图像或模型对象；保持原始格式约定。
    """
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def safe_float(value: Any) -> float | None:
    """可靠处理 float。

    Args:
        value: 待解析或规范化的输入值。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def extract_style_records(
    payload: Any,
) -> list[dict[str, Any]]:
    """提取 style records。

    Args:
        payload: 要保存的权重、优化器和进度状态。

    Returns:
        返回 payload，由函数体中同名变量的计算/收集过程得到。

    Raises:
        ValueError: Could not find garment records in style JSON. Expected a list root
        or a dict containi
    """
    if isinstance(payload, dict):
        for candidate_key in (
            "garments",
            "records",
            "items",
            "data",
        ):
            value = payload.get(candidate_key)

            if isinstance(value, list):
                return value

    if isinstance(payload, list):
        return payload

    raise ValueError(
        "Could not find garment records in style JSON. "
        + "Expected a list root or a dict containing garments/records/items/data."
    )


def build_index(
    rows: list[dict[str, Any]],
    source_field: str,
    garment_field: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    """构建 index。

    Args:
        rows: 待处理的逐行记录。
        source_field: 来源 field。
        garment_field: garment field。

    Returns:
        返回 index，由函数体中同名变量的计算/收集过程得到。
    """
    index = {}

    for row in rows:
        source_image = str(
            row.get(
                source_field,
                "",
            )
        ).strip()

        garment_id = str(
            row.get(
                garment_field,
                "",
            )
        ).strip()

        if not source_image or not garment_id:
            continue

        key = make_key(
            source_image,
            garment_id,
        )

        index[key] = row

    return index


def clean_style_features(
    value: Any,
) -> dict[str, Any]:
    """clean style 特征。

    Args:
        value: 待解析或规范化的输入值。

    Returns:
        返回 cleaned，由函数体中同名变量的计算/收集过程得到。
    """
    if not isinstance(value, dict):
        return {}

    cleaned = {}

    for key, raw in value.items():
        if raw is None:
            continue

        if isinstance(
            raw,
            (int, float, bool),
        ):
            cleaned[key] = raw
            continue

        text = str(raw).strip()

        if text == "":
            continue

        try:
            cleaned[key] = float(text)
        except ValueError:
            cleaned[key] = text

    return cleaned


def clean_quality(
    value: Any,
) -> dict[str, Any]:
    """clean 质量。

    Args:
        value: 待解析或规范化的输入值。

    Returns:
        返回 cleaned，由函数体中同名变量的计算/收集过程得到。
    """
    if not isinstance(value, dict):
        return {}

    cleaned = {}

    for key, raw in value.items():
        if raw is None:
            continue

        if isinstance(
            raw,
            (int, float, bool),
        ):
            cleaned[key] = raw
            continue

        text = str(raw).strip()

        if text == "":
            continue

        try:
            cleaned[key] = float(text)
        except ValueError:
            cleaned[key] = text

    return cleaned


def pattern_block(
    row: dict[str, str] | None,
) -> dict[str, Any] | None:
    """图案 block。

    Args:
        row: 一条实例、预测或审核记录。

    Returns:
        结果字典，主要字段为 label, presence_route, stage1_bucket, stage1_score, stage1_margin,
        stage1_entropy, stage1_quality, stage2_label, stage2_score, stage2_margin。
    """
    if row is None:
        return None

    return {
        "label": (
            row.get(
                "final_pattern",
                "",
            ).strip()
        ),
        "presence_route": (
            row.get(
                "pattern_presence",
                "",
            ).strip()
        ),
        "stage1_bucket": (
            row.get(
                "stage1_bucket",
                "",
            ).strip()
        ),
        "stage1_score": safe_float(row.get("stage1_top1_score")),
        "stage1_margin": safe_float(row.get("stage1_margin")),
        "stage1_entropy": safe_float(row.get("stage1_entropy")),
        "stage1_quality": (
            row.get(
                "stage1_quality",
                "",
            ).strip()
        ),
        "stage2_label": (
            row.get(
                "stage2_top1",
                "",
            ).strip()
        ),
        "stage2_score": safe_float(row.get("stage2_top1_score")),
        "stage2_margin": safe_float(row.get("stage2_margin")),
        "stage2_entropy": safe_float(row.get("stage2_entropy")),
        "stage2_quality": (
            row.get(
                "stage2_quality",
                "",
            ).strip()
        ),
        "method": ("pattern_v3_hierarchical"),
    }


def color_block(
    row: dict[str, str] | None,
) -> dict[str, Any] | None:
    """颜色 block。

    Args:
        row: 一条实例、预测或审核记录。

    Returns:
        结果字典，主要字段为 label, top1_score, top2_label, top2_score, margin, entropy, method,
        scope。
    """
    if row is None:
        return None

    return {
        "label": (
            row.get(
                "primary_color",
                "",
            ).strip()
        ),
        "top1_score": safe_float(row.get("top1_score")),
        "top2_label": (
            row.get(
                "top2_color",
                "",
            ).strip()
        ),
        "top2_score": safe_float(row.get("top2_score")),
        "margin": safe_float(row.get("top1_top2_margin")),
        "entropy": safe_float(row.get("normalized_entropy")),
        "method": ("CLIP_multi_background_embedding_average"),
        "scope": ("single_primary_color"),
    }


def flatten_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    """flatten 记录。

    Args:
        record: 记录字段，使用 source_image, garment_id, garment_category。

    Returns:
        结果字典包含 source_image, garment_id, garment_category。
    """
    flat = {
        "source_image": (record["source_image"]),
        "garment_id": (record["garment_id"]),
        "garment_category": (record["garment_category"]),
    }

    for key, value in record.get(
        "style_features",
        {},
    ).items():
        flat[f"style__{key}"] = value

    for key, value in record.get(
        "style_quality",
        {},
    ).items():
        flat[f"style_quality__{key}"] = value

    pattern = record.get("pattern")

    if isinstance(
        pattern,
        dict,
    ):
        for key, value in pattern.items():
            flat[f"pattern__{key}"] = value

    color = record.get("primary_color")

    if isinstance(
        color,
        dict,
    ):
        for key, value in color.items():
            flat[f"color__{key}"] = value

    flat["integration_complete"] = record.get(
        "integration_complete",
        False,
    )

    return flat


def write_flat_csv(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    """保存 flat CSV。

    Args:
        records: records。
        path: 要读取或写入的文件路径。
    """
    flat_rows = [flatten_record(record) for record in records]

    fieldnames = [
        "source_image",
        "garment_id",
        "garment_category",
    ]

    dynamic = []

    seen = set(fieldnames)

    for row in flat_rows:
        for key in row:
            if key not in seen:
                dynamic.append(key)
                seen.add(key)

    fieldnames += sorted(dynamic)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--style-json",
        required=True,
    )

    parser.add_argument(
        "--pattern-csv",
        required=True,
    )

    parser.add_argument(
        "--color-csv",
        required=True,
    )

    args = parser.parse_args()

    style_path = resolve_path(args.style_json)

    pattern_path = resolve_path(args.pattern_csv)

    color_path = resolve_path(args.color_csv)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    style_payload = read_json(style_path)

    style_rows = extract_style_records(style_payload)

    pattern_rows = read_csv(pattern_path)

    color_rows = read_csv(color_path)

    style_index = build_index(
        style_rows,
        "source_image",
        "garment_id",
    )

    pattern_index = build_index(
        pattern_rows,
        "source_image",
        "garment_id",
    )

    color_index = build_index(
        color_rows,
        "source_image",
        "garment_id",
    )

    all_keys = sorted(set(style_index) | set(pattern_index) | set(color_index))

    unified_records = []
    unmatched = []

    complete_count = 0

    category_counts = Counter()

    for key in all_keys:
        style = style_index.get(key)

        pattern = pattern_index.get(key)

        color = color_index.get(key)

        # Category preference:
        # style -> pattern -> color
        category = ""

        for row in (
            style,
            pattern,
            color,
        ):
            if row is None:
                continue

            candidate = str(
                row.get(
                    "garment_category",
                    "",
                )
            ).strip()

            if candidate:
                category = candidate.lower()
                break

        integration_complete = (
            style is not None and pattern is not None and color is not None
        )

        if integration_complete:
            complete_count += 1

        category_counts[category or "unknown"] += 1

        record = {
            "source_image": (key[0]),
            "garment_id": (key[1]),
            "garment_category": (category),
            "style_features": (
                clean_style_features(
                    (style or {}).get(
                        "style_features",
                        {},
                    )
                )
            ),
            "style_quality": (
                clean_quality(
                    (style or {}).get(
                        "quality",
                        {},
                    )
                )
            ),
            "pattern": (pattern_block(pattern)),
            "primary_color": (color_block(color)),
            "integration_complete": (integration_complete),
            "source_availability": {
                "style": (style is not None),
                "pattern": (pattern is not None),
                "color": (color is not None),
            },
        }

        unified_records.append(record)

        if not integration_complete:
            unmatched.append(
                {
                    "source_image": key[0],
                    "garment_id": key[1],
                    "has_style": int(style is not None),
                    "has_pattern": int(pattern is not None),
                    "has_color": int(color is not None),
                }
            )

    output_payload = {
        "schema_version": ("3.1.3-unified-v1"),
        "unit_of_analysis": ("one garment instance per record"),
        "scope": {
            "style": ("continuous garment style descriptors"),
            "pattern": ("pattern v3 hierarchical baseline"),
            "primary_color": ("single dominant color baseline"),
            "secondary_colors": ("out_of_scope"),
            "material": ("excluded"),
            "craftsmanship": ("excluded"),
        },
        "integration_key": [
            "source_image",
            "garment_id",
        ],
        "records": (unified_records),
    }

    json_path = OUTPUT_DIR / "garment_attribute_vector_v1.json"

    json_path.write_text(
        json.dumps(
            output_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    flat_csv_path = OUTPUT_DIR / "garment_attribute_vector_v1_flat.csv"

    write_flat_csv(
        unified_records,
        flat_csv_path,
    )

    unmatched_path = OUTPUT_DIR / "unmatched_records.csv"

    with unmatched_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_image",
                "garment_id",
                "has_style",
                "has_pattern",
                "has_color",
            ],
        )

        writer.writeheader()
        writer.writerows(unmatched)

    summary_lines = [
        "PRD 3.1.3 unified garment attribute output v1",
        "=============================================",
        f"style_records={len(style_index)}",
        f"pattern_records={len(pattern_index)}",
        f"color_records={len(color_index)}",
        f"union_records={len(all_keys)}",
        f"complete_records={complete_count}",
        f"incomplete_records={len(unmatched)}",
        "",
        "Category distribution",
        "---------------------",
    ]

    for category, count in category_counts.most_common():
        summary_lines.append(f"{category}={count}")

    summary_lines += [
        "",
        "Integration",
        "-----------",
        "- key=(source_image, garment_id)",
        "- continuous style descriptors are preserved as numeric features",
        "- pattern uses the final v3 hierarchical output",
        "- primary color uses the multi-background CLIP baseline",
        "- uncertainty / quality fields are preserved rather than collapsed",
        "- secondary colors remain out of scope",
        "- material and craftsmanship remain excluded",
        "",
        "Downstream use",
        "--------------",
        "- garment_attribute_vector_v1.json is the structured input for later 3.2 multimodal reasoning integration.",
        "- garment_attribute_vector_v1_flat.csv is for inspection, debugging and simple analysis.",
        "- unmatched_records.csv must be empty or explicitly explained before calling integration complete.",
    ]

    (OUTPUT_DIR / "integration_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    run_info = (
        "PRD 3.1.3 unified garment attribute output v1\n"
        + f"style_json={args.style_json}\n"
        + f"pattern_csv={args.pattern_csv}\n"
        + f"color_csv={args.color_csv}\n"
        + "merge_key=source_image+garment_id\n"
        + f"complete_records={complete_count}\n"
        + f"incomplete_records={len(unmatched)}\n"
        + "secondary_colors=out_of_scope\n"
        + "material_and_craftsmanship=excluded\n"
    )

    (OUTPUT_DIR / "run_info.txt").write_text(
        run_info,
        encoding="utf-8",
    )

    print("=== UNIFIED ATTRIBUTE OUTPUT V1 ===")

    print(f"style records    : {len(style_index)}")

    print(f"pattern records  : {len(pattern_index)}")

    print(f"color records    : {len(color_index)}")

    print(f"union records    : {len(all_keys)}")

    print(f"complete records : {complete_count}")

    print(f"incomplete       : {len(unmatched)}")

    print("")

    print(
        "JSON    : reports/prd_attribute_extraction/"
        + "unified_v1/garment_attribute_vector_v1.json"
    )

    print(
        "CSV     : reports/prd_attribute_extraction/"
        + "unified_v1/garment_attribute_vector_v1_flat.csv"
    )

    print(
        "SUMMARY : reports/prd_attribute_extraction/"
        + "unified_v1/integration_summary.txt"
    )

    print(
        "UNMATCH : reports/prd_attribute_extraction/"
        + "unified_v1/unmatched_records.csv"
    )

    print("===================================")


if __name__ == "__main__":
    main()
