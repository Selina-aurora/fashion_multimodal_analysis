"""3.1 接口联调：区分预测 ROI 传递和使用 GT 的受控诊断，保留对象 ID 与坐标对应关系。

PRD 3.1 full-validation predicted-ROI -> 3.1.3 attribute propagation.

This is the full-validation counterpart of the earlier 13-pair pilot.
It uses:
    configs/prd_3_1_predicted_roi_fullval_matched_v1.csv

and writes to a NEW folder so the pilot results are preserved:
    reports/prd_integration/predicted_roi_attributes_fullval_v1/

Modules:
1) primary color v1
2) pattern v3 hierarchical
3) categorical design labels v1
4) GT-ROI vs predicted-ROI label agreement

Compatibility:
Patches CLIP text/image encoder helpers at runtime so current transformers
versions that reject `return_dict` in CLIPTextTransformer/CLIPVisionTransformer
do not crash.

Run
---
cd fashion_multimodal_analysis

python scripts/integration/run_prd31_predicted_roi_attributes_fullval_v1.py   --device
cuda
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

import torch

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

PROJECT_ROOT = get_project_root()

DEFAULT_MANIFEST = (
    PROJECT_ROOT / "configs" / "prd_3_1_predicted_roi_fullval_matched_v1.csv"
)

REPORT_ROOT = (
    PROJECT_ROOT / "reports" / "prd_integration" / "predicted_roi_attributes_fullval_v1"
)

OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "prd_integration" / "predicted_roi_attributes_fullval_v1"
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
        raw: 本步骤使用的原始值，转换/筛选规则见函数体。

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
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
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
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


def norm_source(v: str) -> str:
    """规范化来源标识，使不同表格中的同一原图可连接。

    Args:
        v: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return str(v).replace("\\", "/").strip()


def key_of(row: dict[str, str]) -> tuple[str, str]:
    """提取用于跨表连接的记录标识。

    Args:
        row: 一条实例、预测或审核记录。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return (
        norm_source(row.get("source_image", "")),
        str(row.get("garment_id", "")).strip(),
    )


def run_module_main(module: str, argv: list[str]) -> None:
    """调用目标实验模块，保留命令行入口的运行语义。

    Args:
        module: 要执行的 Python 模块名称。
        argv: argv。
    """
    old = sys.argv[:]
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = old


def first_existing(
    row: dict[str, str],
    names: list[str],
) -> str:
    """按优先顺序返回存在的候选路径。

    Args:
        row: 一条实例、预测或审核记录。
        names: names。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def agreement(a: str, b: str) -> str:
    """计算两套预测在可比较样本上的一致性；不是人工真值准确率。

    Args:
        a: 当前函数的第一个输入，含义随运算而定。
        b: 当前函数的第二个输入，含义随运算而定。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if not a or not b:
        return ""
    return "1" if a == b else "0"


def summarize_binary(
    rows: list[dict[str, Any]],
    field: str,
) -> tuple[int, int, float]:
    """汇总二值判断的计数与比例。

    Args:
        rows: 待处理的逐行记录。
        field: 本次读取或统计的字段名。

    Returns:
        按顺序返回 n, correct 等结果。
    """
    vals = []

    for row in rows:
        raw = str(row.get(field, "")).strip()
        if raw in {"0", "1"}:
            vals.append(int(raw))

    n = len(vals)
    correct = sum(vals)

    return (
        n,
        correct,
        correct / n if n else math.nan,
    )


def pooled_output(outputs: Any) -> Any:
    """pooled 输出。

    Args:
        outputs: 输出。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        TypeError: 执行本函数的操作失败。
    """
    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output

    if isinstance(outputs, (tuple, list)):
        if len(outputs) >= 2:
            return outputs[1]
        if len(outputs) == 1:
            x = outputs[0]
            return x[:, 0]

    raise TypeError(f"Cannot extract pooled output from {type(outputs).__name__}")


def install_clip_compat(module: str) -> None:
    """Runtime-only compatibility patch.
    Does not edit frozen baseline scripts.

    Args:
        module: 要执行的 Python 模块名称。
    """

    def encode_texts_compat(
        prompts: list[str],
        model: Any,
        processor: Any,
        device: Any,
    ) -> Any:
        """编码 texts compat。

        Args:
            prompts: 提示文本。
            model: 已构建的模型对象，由调用方负责选择权重。
            processor: 与模型配套的输入处理器。
            device: 当前计算设备，与输入张量和模型设备保持一致。

        Returns:
            本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
        """
        inputs = processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            try:
                outputs = model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                pooled = pooled_output(outputs)
                features = model.text_projection(pooled)
            except Exception:
                kwargs = {
                    "input_ids": inputs["input_ids"],
                }
                if inputs.get("attention_mask") is not None:
                    kwargs["attention_mask"] = inputs["attention_mask"]

                features = model.get_text_features(**kwargs)

        return module.l2_normalize(features)

    def encode_image_compat(
        image: Any,
        model: Any,
        processor: Any,
        device: Any,
    ) -> Any:
        """编码 图像 compat。

        Args:
            image: 本步骤处理的图像对象。
            model: 已构建的模型对象，由调用方负责选择权重。
            processor: 与模型配套的输入处理器。
            device: 当前计算设备，与输入张量和模型设备保持一致。

        Returns:
            本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
        """
        inputs = processor(
            images=image,
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(device)

        with torch.inference_mode():
            try:
                outputs = model.vision_model(
                    pixel_values=pixel_values,
                )
                pooled = pooled_output(outputs)
                features = model.visual_projection(pooled)
            except Exception:
                features = model.get_image_features(pixel_values=pixel_values)

        return module.l2_normalize(features)

    module.encode_texts = encode_texts_compat
    module.encode_image = encode_image_compat


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    args = parse_args()
    manifest = resolve_path(args.manifest)

    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    import fashion_multimodal_analysis.attributes.color.run_primary_color_baseline_v1 as color_mod
    import fashion_multimodal_analysis.attributes.design.run_design_attribute_labels_v1 as design_mod
    import fashion_multimodal_analysis.attributes.pattern.run_pattern_hierarchical_v3 as pattern_mod

    install_clip_compat(color_mod)
    install_clip_compat(pattern_mod)

    print("CLIP compatibility patch installed " + "(full-validation run).")

    # ---------------------------------------------------------
    # 1) COLOR
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 2) PATTERN
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 3) DESIGN LABELS
    # ---------------------------------------------------------
    empty_style = REPORT_ROOT / "empty_style_for_predicted_roi.json"

    empty_style.write_text(
        json.dumps(
            {
                "schema_version": ("predicted-roi-empty-style-fullval-v1"),
                "records": [],
                "note": (
                    "Intentionally empty to prevent GT "
                    + "continuous-style geometry leakage."
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

    # ---------------------------------------------------------
    # 4) COMPARE WITH FROZEN GT-ROI OUTPUTS
    # ---------------------------------------------------------
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

    pred_color = {key_of(r): r for r in read_csv(pred_color_path)}

    pred_pattern = {key_of(r): r for r in read_csv(pred_pattern_path)}

    pred_design = {key_of(r): r for r in read_csv(pred_design_path)}

    gt_color = {key_of(r): r for r in read_csv(GT_COLOR)}

    gt_pattern = {key_of(r): r for r in read_csv(GT_PATTERN)}

    gt_design = {key_of(r): r for r in read_csv(GT_DESIGN)}

    comparisons = []

    for m in manifest_rows:
        key = key_of(m)

        pc = pred_color.get(key, {})
        gc = gt_color.get(key, {})
        pp = pred_pattern.get(key, {})
        gp = gt_pattern.get(key, {})
        pd = pred_design.get(key, {})
        gd = gt_design.get(key, {})

        pred_color_label = first_existing(
            pc,
            [
                "primary_color",
                "predicted_color",
                "label",
            ],
        )

        gt_color_label = first_existing(
            gc,
            [
                "primary_color",
                "predicted_color",
                "label",
            ],
        )

        pred_pattern_label = first_existing(
            pp,
            [
                "final_pattern",
                "pattern",
                "label",
            ],
        )

        gt_pattern_label = first_existing(
            gp,
            [
                "final_pattern",
                "pattern",
                "label",
            ],
        )

        row = {
            "source_image": key[0],
            "garment_id": key[1],
            "garment_category": m.get(
                "garment_category",
                "",
            ),
            "matched_gt_bbox_iou": m.get(
                "matched_gt_bbox_iou",
                "",
            ),
            "prediction_score": m.get(
                "prediction_score",
                "",
            ),
            "gt_primary_color": gt_color_label,
            "predroi_primary_color": pred_color_label,
            "color_agree": agreement(
                gt_color_label,
                pred_color_label,
            ),
            "gt_pattern": gt_pattern_label,
            "predroi_pattern": pred_pattern_label,
            "pattern_agree": agreement(
                gt_pattern_label,
                pred_pattern_label,
            ),
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

    write_csv(
        compare_csv,
        comparisons,
    )

    metric_fields = [
        ("primary_color", "color_agree"),
        ("pattern", "pattern_agree"),
        ("sleeve_length", "sleeve_length_agree"),
        ("neckline", "neckline_agree"),
        ("silhouette_fit", "silhouette_fit_agree"),
        ("fashion_style", "fashion_style_agree"),
    ]

    summary_rows = []

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
        str(
            r.get(
                "garment_category",
                "",
            )
        )
        .strip()
        .lower()
        for r in manifest_rows
    )

    source_counts = Counter(
        str(
            r.get(
                "source_dataset",
                "",
            )
        ).strip()
        for r in manifest_rows
    )

    lines = [
        (
            "PRD 3.1 Full-Validation Predicted-ROI -> "
            + "3.1.3 Attribute Propagation v1"
        ),
        ("================================================" + "=================="),
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
        "Matched source distribution",
        "---------------------------",
    ]

    for src_name, count in sorted(source_counts.items()):
        lines.append(f"{src_name}={count}")

    lines += [
        "",
        "GT-ROI vs predicted-ROI label agreement",
        "---------------------------------------",
    ]

    for row in summary_rows:
        lines.append(
            f"{row['attribute']}: "
            + f"{row['same_label_pairs']}/"
            + f"{row['comparable_pairs']} "
            + f"agreement="
            + f"{row['gt_vs_predroi_label_agreement'] or 'NA'}"
        )

    lines += [
        "",
        "Interpretation",
        "--------------",
        (
            "- This measures propagation consistency, "
            + "NOT attribute accuracy against human GT."
        ),
        (
            "- GT source_image/garment_id are join keys only; "
            + "attribute inputs are predicted crop/mask."
        ),
        (
            "- No GT continuous style geometry is injected into "
            + "predicted-ROI categorical inference."
        ),
        (
            "- Missing GT attribute outputs for Fashionpedia "
            + "instances reduce the number of comparable pairs."
        ),
        (
            "- This is still a fixed-validation diagnostic, "
            + "not final production acceptance."
        ),
    ]

    (REPORT_ROOT / "summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== FULLVAL PREDICTED ROI ATTRIBUTE RUN FINISHED ===")
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
