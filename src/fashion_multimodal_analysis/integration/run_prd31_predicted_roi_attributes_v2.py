"""3.1 接口联调：区分预测 ROI 传递和使用 GT 的受控诊断，保留对象 ID 与坐标对应关系。

Run 3.1.3 attributes on the matched PREDICTED ROI manifest (v2) without overwriting
the frozen GT baselines, then compare predicted-ROI labels against the original
GT-ROI labels by (source_image, garment_id).

Modules:
- primary color v1
- pattern v3 hierarchical
- categorical design labels v1

Important:
- crop_path / mask_path come from 3.1.1 v2-balanced predictions.
- GT source_image + garment_id are used only as join keys in the matched manifest.
- An EMPTY style JSON is passed to design-label inference so no GT continuous
  geometry features leak into the predicted-ROI categorical outputs.
- shoe/bag/accessory are automatically skipped by design-label v1 if unsupported.

Run:
cd fashion_multimodal_analysis
python scripts/integration/run_prd31_predicted_roi_attributes_v2.py --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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

DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "prd_3_1_predicted_roi_matched_v2.csv"

REPORT_ROOT = (
    PROJECT_ROOT / "reports" / "prd_integration" / "predicted_roi_attributes_v2"
)

OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "prd_integration" / "predicted_roi_attributes_v2"
)

GT_COLOR = (
    PROJECT_ROOT
    / "reports"
    / "prd_attribute_extraction"
    / "color_v1"
    / "primary_color_predictions.csv"
)

GT_PATTERN = (
    PROJECT_ROOT
    / "reports"
    / "prd_attribute_extraction"
    / "pattern_v3_hierarchical"
    / "pattern_v3_predictions.csv"
)

GT_DESIGN = (
    PROJECT_ROOT
    / "reports"
    / "prd_attribute_extraction"
    / "design_labels_v1_fixed"
    / "design_attribute_predictions_v1_fixed.csv"
)


def parse_args() -> argparse.Namespace:
    """解析并校验命令行参数。

    Returns:
        已通过参数合法性检查的命令行配置。
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cuda",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=4,
    )
    return p.parse_args()


def resolve_path(raw: Any) -> Path:
    """将清单或配置中的相对路径解析到当前项目/数据目录。

    Args:
        raw: 清单中的路径字段；由公共路径解析器处理项目根目录与外部数据根目录。

    Returns:
        当前工作副本可使用的路径。
    """
    return resolve_artifact_path(raw)


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


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
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
        for key in row.keys():
            if key not in fields:
                fields.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def norm_source(v: str) -> str:
    """规范化来源标识，使不同表格中的同一原图可连接。

    Args:
        v: 原图路径标识；将反斜杠改为正斜杠并移除首尾空白。

    Returns:
        用于跨表匹配的字符串；不解析绝对路径、不检查文件是否存在。
    """
    return str(v).replace("\\", "/").strip()


def key_of(row: dict[str, str]) -> tuple[str, str]:
    """提取用于跨表连接的记录标识。

    Args:
        row: 一条实例、预测或审核记录。

    Returns:
        (规范化 source_image, garment_id) 二元组；字段缺失时对应位置为空字符串。
    """
    return (
        norm_source(row.get("source_image", "")),
        str(row.get("garment_id", "")).strip(),
    )


def run_module_main(module: str, argv: list[str]) -> None:
    """调用目标实验模块，保留命令行入口的运行语义。

    Args:
        module: 已导入且提供 main() 的属性模块对象。
        argv: 临时命令行参数列表，第一项为程序名；退出或异常时恢复原 sys.argv。
    """
    old = sys.argv[:]
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = old


def first_existing(row: dict[str, str], names: list[str]) -> str:
    """按字段优先级选择第一个非空标签值，不检查文件路径是否存在。

    Args:
        row: 一条实例、预测或审核记录。
        names: 按优先级排列的候选字段名，用于兼容历史结果表的不同列名。

    Returns:
        第一个去除首尾空白后的非空值；全部候选缺失或为空时返回空字符串。
    """
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def agreement(a: str, b: str) -> str:
    """计算两套预测在可比较样本上的一致性；不是人工真值准确率。

    Args:
        a: GT ROI 上的属性预测标签，不是人工审核的属性真值。
        b: 预测 ROI 上的属性预测标签。

    Returns:
        两者非空且相等时为 "1"，非空且不同为 "0"；任一为空则返回空字符串，
        后续统计将其排除，不当作预测错误。
    """
    if not a or not b:
        return ""
    return "1" if a == b else "0"


def summarize_binary(rows: list[dict[str, Any]], field: str) -> tuple[int, int, float]:
    """汇总二值判断的计数与比例。

    Args:
        rows: 待处理的逐行记录。
        field: 本次读取或统计的字段名。

    Returns:
        (可比较行数 n, 标签相同行数 c, c/n)。仅统计字符串 "0" 或 "1"；
        空值不进入分母，n=0 时比例为 NaN。此比例衡量传播一致性，不是准确率。
    """
    vals = []
    for row in rows:
        raw = str(row.get(field, "")).strip()
        if raw in {"0", "1"}:
            vals.append(int(raw))
    n = len(vals)
    c = sum(vals)
    return n, c, (c / n if n else math.nan)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    args = parse_args()
    manifest = resolve_path(args.manifest)

    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 属性模块在执行入口内导入；实现位于 src，命令入口位于 scripts/integration。
    import fashion_multimodal_analysis.attributes.color.run_primary_color_baseline_v1 as color_mod
    import fashion_multimodal_analysis.attributes.design.run_design_attribute_labels_v1 as design_mod
    import fashion_multimodal_analysis.attributes.pattern.run_pattern_hierarchical_v3 as pattern_mod

    # ------------------------------------------------------------------
    # 1) COLOR on predicted ROI
    # ------------------------------------------------------------------
    color_mod.REPORT_DIR = REPORT_ROOT / "color_v1"
    color_mod.OUTPUT_DIR = OUTPUT_ROOT / "color_v1"

    run_module_main(
        color_mod,
        [
            "run_primary_color_baseline_v1.py",
            "--manifest",
            str(manifest),
            "--holdout-size",
            "0",
            "--threads",
            str(args.threads),
        ],
    )

    # ------------------------------------------------------------------
    # 2) PATTERN on predicted ROI
    # ------------------------------------------------------------------
    pattern_mod.REPORT_DIR = REPORT_ROOT / "pattern_v3_hierarchical"
    pattern_mod.OUTPUT_DIR = OUTPUT_ROOT / "pattern_v3_hierarchical"

    run_module_main(
        pattern_mod,
        [
            "run_pattern_hierarchical_v3.py",
            "--manifest",
            str(manifest),
            "--holdout-size",
            "0",
            "--threads",
            str(args.threads),
        ],
    )

    # ------------------------------------------------------------------
    # 3) DESIGN categorical labels on predicted ROI
    #    Use deliberately empty style JSON to avoid GT geometry leakage.
    # ------------------------------------------------------------------
    empty_style = REPORT_ROOT / "empty_style_for_predicted_roi.json"
    empty_style.write_text(
        json.dumps(
            {
                "schema_version": "predicted-roi-empty-style-v1",
                "records": [],
                "note": (
                    "Intentionally empty: categorical design inference uses "
                    + "predicted ROI crop/mask only; no GT continuous style features."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    design_mod.REPORT_DIR = REPORT_ROOT / "design_labels_v1"

    run_module_main(
        design_mod,
        [
            "run_design_attribute_labels_v1.py",
            "--manifest",
            str(manifest),
            "--style-json",
            str(empty_style),
            "--audit-size",
            "0",
            "--device",
            args.device,
        ],
    )

    # ------------------------------------------------------------------
    # 4) Compare predicted-ROI labels against frozen GT-ROI labels.
    # ------------------------------------------------------------------
    pred_color_path = REPORT_ROOT / "color_v1" / "primary_color_predictions.csv"
    pred_pattern_path = (
        REPORT_ROOT / "pattern_v3_hierarchical" / "pattern_v3_predictions.csv"
    )
    pred_design_path = (
        REPORT_ROOT / "design_labels_v1" / "design_attribute_predictions_v1.csv"
    )

    for required in [
        pred_color_path,
        pred_pattern_path,
        pred_design_path,
        GT_COLOR,
        GT_PATTERN,
        GT_DESIGN,
    ]:
        if not required.is_file():
            raise FileNotFoundError(required)

    manifest_rows = read_csv(manifest)

    # 按原图与实例 ID 配对，避免多件服饰串行；同一键重复时字典保留最后一行。
    # 这里的 GT 表是 GT ROI 上的模型输出，一致性不能替代人工真值准确率。
    pred_color = {key_of(r): r for r in read_csv(pred_color_path)}
    pred_pattern = {key_of(r): r for r in read_csv(pred_pattern_path)}
    pred_design = {key_of(r): r for r in read_csv(pred_design_path)}

    gt_color = {key_of(r): r for r in read_csv(GT_COLOR)}
    gt_pattern = {key_of(r): r for r in read_csv(GT_PATTERN)}
    gt_design = {key_of(r): r for r in read_csv(GT_DESIGN)}

    comparisons = []

    for m in manifest_rows:
        key = key_of(m)
        cat = str(m.get("garment_category", "")).strip().lower()

        pc = pred_color.get(key, {})
        gc = gt_color.get(key, {})

        pp = pred_pattern.get(key, {})
        gp = gt_pattern.get(key, {})

        pd = pred_design.get(key, {})
        gd = gt_design.get(key, {})

        pred_color_label = first_existing(
            pc,
            ["primary_color", "predicted_color", "label"],
        )
        gt_color_label = first_existing(
            gc,
            ["primary_color", "predicted_color", "label"],
        )

        pred_pattern_label = first_existing(
            pp,
            ["final_pattern", "pattern", "label"],
        )
        gt_pattern_label = first_existing(
            gp,
            ["final_pattern", "pattern", "label"],
        )

        row = {
            "source_image": key[0],
            "garment_id": key[1],
            "garment_category": cat,
            "matched_gt_bbox_iou": m.get("matched_gt_bbox_iou", ""),
            "prediction_score": m.get("prediction_score", ""),
            "gt_primary_color": gt_color_label,
            "predroi_primary_color": pred_color_label,
            "color_agree": agreement(gt_color_label, pred_color_label),
            "gt_pattern": gt_pattern_label,
            "predroi_pattern": pred_pattern_label,
            "pattern_agree": agreement(gt_pattern_label, pred_pattern_label),
        }

        for attr in [
            "sleeve_length",
            "neckline",
            "silhouette_fit",
            "fashion_style",
        ]:
            gt_label = first_existing(
                gd,
                [
                    f"{attr}_label",
                    attr,
                ],
            )
            pred_label = first_existing(
                pd,
                [
                    f"{attr}_label",
                    attr,
                ],
            )

            row[f"gt_{attr}"] = gt_label
            row[f"predroi_{attr}"] = pred_label
            row[f"{attr}_agree"] = agreement(
                gt_label,
                pred_label,
            )

        comparisons.append(row)

    compare_csv = REPORT_ROOT / "per_instance_attribute_comparison.csv"
    write_csv(compare_csv, comparisons)

    # ------------------------------------------------------------------
    # 5) Summary
    # ------------------------------------------------------------------
    summary_rows = []

    metric_fields = [
        ("primary_color", "color_agree"),
        ("pattern", "pattern_agree"),
        ("sleeve_length", "sleeve_length_agree"),
        ("neckline", "neckline_agree"),
        ("silhouette_fit", "silhouette_fit_agree"),
        ("fashion_style", "fashion_style_agree"),
    ]

    for metric, field in metric_fields:
        n, correct, rate = summarize_binary(
            comparisons,
            field,
        )

        summary_rows.append(
            {
                "attribute": metric,
                "comparable_pairs": n,
                "same_label_pairs": correct,
                "gt_vs_predroi_label_agreement": (
                    "" if math.isnan(rate) else f"{rate:.4f}"
                ),
            }
        )

    write_csv(
        REPORT_ROOT / "attribute_agreement_summary.csv",
        summary_rows,
    )

    category_counts = Counter(
        str(r.get("garment_category", "")).strip().lower() for r in manifest_rows
    )

    lines = [
        "PRD 3.1 Predicted-ROI -> 3.1.3 Attribute Propagation Pilot v2",
        "=============================================================",
        "",
        f"manifest={manifest.relative_to(PROJECT_ROOT)}",
        f"matched_roi_pairs={len(manifest_rows)}",
        "",
        "Matched category distribution",
        "-----------------------------",
    ]

    for cls in [
        "top",
        "pants",
        "skirt",
        "outerwear",
        "dress",
        "shoe",
        "bag",
        "accessory",
    ]:
        lines.append(f"{cls}={category_counts.get(cls, 0)}")

    lines += [
        "",
        "GT-ROI vs predicted-ROI label agreement",
        "---------------------------------------",
    ]

    for row in summary_rows:
        lines.append(
            f"{row['attribute']}: "
            + f"{row['same_label_pairs']}/{row['comparable_pairs']} "
            + f"agreement={row['gt_vs_predroi_label_agreement'] or 'NA'}"
        )

    lines += [
        "",
        "Interpretation",
        "--------------",
        "- This is propagation consistency, NOT attribute accuracy against human ground truth.",
        "- A disagreement means the 3.1.3 label changed when GT ROI was replaced by the matched predicted ROI.",
        "- color and pattern are evaluated for every matched ROI with available frozen GT output.",
        "- categorical design labels are evaluated only for categories supported by design-label v1.",
        "- no GT continuous style geometry is injected into predicted-ROI categorical inference.",
        "- the matched pilot is small and has incomplete category coverage; do not present it as final 8-class acceptance.",
    ]

    (REPORT_ROOT / "summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== PREDICTED ROI ATTRIBUTE PILOT FINISHED ===")
    print(
        "Summary:",
        (REPORT_ROOT / "summary.txt").relative_to(PROJECT_ROOT),
    )
    print(
        "Agreement:",
        (REPORT_ROOT / "attribute_agreement_summary.csv").relative_to(PROJECT_ROOT),
    )
    print(
        "Per instance:",
        compare_csv.relative_to(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
