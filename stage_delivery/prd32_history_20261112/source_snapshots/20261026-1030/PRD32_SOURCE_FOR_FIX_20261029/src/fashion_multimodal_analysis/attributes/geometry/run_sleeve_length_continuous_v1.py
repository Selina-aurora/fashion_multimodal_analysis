"""3.1.3 可视属性提取：关注颜色、图案和服装设计/几何属性；一致性不等于人工真值准确率。

PRD 3.1.3 continuous sleeve-length feature baseline v1.

Goal
----
Replace direct hard-label sleeve classification with a continuous 0~1 feature.

The model compares the garment image with ordered sleeve descriptions, then
combines the semantic probabilities into an expected continuous value:

    sleeve_length_ratio = sum(prob_i * anchor_i)

Anchors:
0.00 sleeveless
0.18 cap sleeve
0.35 short sleeve
0.55 elbow sleeve
0.75 three-quarter sleeve
1.00 long sleeve

Provisional thresholds for analysis:
[0.00, 0.30) -> short
[0.30, 0.60) -> medium
[0.60, 1.00] -> long

DeepFashion2 fine category names are used only for ordinal evaluation, not to
compute the continuous feature.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPModel, CLIPProcessor

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

PROJECT_ROOT = get_project_root()
MODEL_NAME = "openai/clip-vit-base-patch32"

REPORT_DIR = PROJECT_ROOT / "reports" / "prd_attribute_extraction" / "continuous_v1"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "prd_attribute_extraction" / "continuous_v1"

UPPER_BODY_CATEGORIES = {"top", "outerwear", "dress"}

SLEEVE_ANCHORS: Dict[str, tuple[float, str]] = {
    "sleeveless": (
        0.00,
        "a fashion garment with no sleeves and open armholes",
    ),
    "cap": (
        0.18,
        "a fashion garment with tiny cap sleeves covering only the shoulder edge",
    ),
    "short": (
        0.35,
        "a fashion garment with short sleeves ending on the upper arm",
    ),
    "elbow": (
        0.55,
        "a fashion garment with sleeves ending around the elbow",
    ),
    "three_quarter": (
        0.75,
        "a fashion garment with three-quarter sleeves ending below the elbow",
    ),
    "long": (
        1.00,
        "a fashion garment with long sleeves reaching the wrist",
    ),
}


def resolve_path(raw: Any) -> Path:
    """将清单或配置中的相对路径解析到当前项目/数据目录。

    Args:
        raw: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        当前工作副本可使用的路径。
    """
    return resolve_artifact_path(raw)


def read_manifest(path: Path) -> list[dict[str, str]]:
    """读取实例清单，不在读取阶段改变样本划分。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {
        "source_image",
        "garment_id",
        "garment_category",
        "crop_path",
        "mask_path",
    }

    if not rows:
        raise ValueError("Manifest has no rows.")

    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    return rows


def load_masked_garment(row: dict[str, str]) -> Image.Image:
    """读取服饰区域并应用掩码，减少背景对属性识别的干扰。

    Args:
        row: 记录字段，使用 crop_path。

    Returns:
        当前读取操作得到的记录、图像或模型对象；保持原始格式约定。
    """
    crop_path = resolve_path(row["crop_path"])
    crop = Image.open(crop_path).convert("RGB")

    mask_raw = row.get("mask_path", "").strip()
    if not mask_raw:
        return crop

    mask_path = resolve_path(mask_raw)
    mask = Image.open(mask_path).convert("L")

    if mask.size != crop.size:
        mask = mask.resize(crop.size, Image.Resampling.NEAREST)

    white = Image.new("RGB", crop.size, "white")
    return Image.composite(crop, white, mask)


def deepfashion2_gt_group(garment_id: str) -> str:
    """DeepFashion2 gt group。

    Args:
        garment_id: garment ID。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    gid = garment_id.lower()

    if "vest" in gid or "sling" in gid:
        return "sleeveless"
    if "short_sleeve" in gid:
        return "short"
    if "long_sleeve" in gid:
        return "long"
    return "unknown"


def ratio_to_bucket(
    ratio: float,
    short_threshold: float,
    long_threshold: float,
) -> str:
    """比例 转换 bucket。

    Args:
        ratio: 比例。
        short_threshold: 本步骤的 short 判定阈值。
        long_threshold: 本步骤的 long 判定阈值。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if ratio < short_threshold:
        return "short"
    if ratio < long_threshold:
        return "medium"
    return "long"


def infer_continuous_ratio(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> dict:
    """推理 continuous 比例。

    Args:
        image: 本步骤处理的图像对象。
        model: 已构建的模型对象，由调用方负责选择权重。
        processor: 与模型配套的输入处理器。
        device: 当前计算设备，与输入张量和模型设备保持一致。

    Returns:
        结果字典，主要字段为 sleeve_length_ratio, top_anchor, top_anchor_probability, top3,
        normalized_entropy。
    """
    labels: List[str] = list(SLEEVE_ANCHORS.keys())
    anchors = torch.tensor(
        [SLEEVE_ANCHORS[label][0] for label in labels],
        dtype=torch.float32,
    )
    prompts = [SLEEVE_ANCHORS[label][1] for label in labels]

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    probs = outputs.logits_per_image[0].softmax(dim=0).cpu()

    ratio = float(torch.sum(probs * anchors).item())

    order = torch.argsort(probs, descending=True)

    top3 = []
    for idx in order[:3].tolist():
        idx = int(idx)
        label = labels[idx]
        top3.append(
            {
                "anchor_label": label,
                "anchor_value": SLEEVE_ANCHORS[label][0],
                "probability": float(probs[idx].item()),
            }
        )

    entropy = float(-torch.sum(probs * torch.log(probs.clamp_min(1e-12))).item())
    max_entropy = math.log(len(labels))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "sleeve_length_ratio": ratio,
        "top_anchor": top3[0]["anchor_label"],
        "top_anchor_probability": top3[0]["probability"],
        "top3": top3,
        "normalized_entropy": normalized_entropy,
    }


def mean(values: list[float]) -> float:
    """计算本组数值的平均值，并按本模块策略处理空组。

    Args:
        values: 本步骤处理的数值或附加字段。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return sum(values) / len(values) if values else float("nan")


def median(values: list[float]) -> float:
    """计算本组数值的中位数，并处理空组。

    Args:
        values: 本步骤处理的数值或附加字段。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def pairwise_order_accuracy(
    lower_values: list[float],
    higher_values: list[float],
) -> tuple[int, int, float]:
    """pairwise order 准确率。

    Args:
        lower_values: lower values。
        higher_values: higher values。

    Returns:
        按顺序返回 correct, total, acc 等结果。
    """
    total = 0
    correct = 0

    for low in lower_values:
        for high in higher_values:
            total += 1
            if high > low:
                correct += 1

    acc = correct / total if total else float("nan")
    return correct, total, acc


def save_visual(
    image: Image.Image,
    record: dict,
    save_path: Path,
) -> None:
    """保存样本可视化，供错误案例复核。

    Args:
        image: 本步骤处理的图像对象。
        record: 记录字段，使用 garment_id, gt_group, sleeve_length_ratio, bucket_v1,
        top_anchor, top_anchor_probability, normalized_entropy。
        save_path: 对应文件的相对路径或当前解析后的路径。
    """
    panel_h = 170
    canvas = Image.new(
        "RGB",
        (image.width, image.height + panel_h),
        "white",
    )
    canvas.paste(image, (0, panel_h))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    lines = [
        f"{record['garment_id']} | GT={record['gt_group']}",
        f"sleeve_length_ratio: {record['sleeve_length_ratio']:.3f}",
        f"bucket_v1: {record['bucket_v1']}",
        (
            f"top_anchor: {record['top_anchor']} "
            + f"p={record['top_anchor_probability']:.3f}"
        ),
        f"uncertainty(entropy): {record['normalized_entropy']:.3f}",
    ]

    y = 10
    for line in lines:
        draw.text((10, y), line, fill="black", font=font)
        y += 29

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path, quality=92)


def make_contact_sheet(
    visual_records: list[tuple[float, Path]],
    output_path: Path,
) -> None:
    """把样本图与说明拼成审核联系表，便于逐例比较。

    Args:
        visual_records: 可视化 records。
        output_path: 对应文件的相对路径或当前解析后的路径。
    """
    if not visual_records:
        return

    visual_records = sorted(visual_records, key=lambda x: x[0])
    tiles = []

    for _, path in visual_records:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((390, 500))

            tile = Image.new("RGB", (400, 510), "white")
            x = (400 - img.width) // 2
            y = (510 - img.height) // 2
            tile.paste(img, (x, y))
            tiles.append(tile)

    cols = 4
    rows = (len(tiles) + cols - 1) // cols

    sheet = Image.new(
        "RGB",
        (cols * 400, rows * 510),
        "white",
    )

    for idx, tile in enumerate(tiles):
        x = (idx % cols) * 400
        y = (idx // cols) * 510
        sheet.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """按确定的字段顺序保存实验 CSV。

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。
        fieldnames: fieldnames。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        ValueError: No upper-body garment instances found in manifest.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--short-threshold", type=float, default=0.30)
    parser.add_argument("--long-threshold", type=float, default=0.60)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if not (0.0 <= args.short_threshold < args.long_threshold <= 1.0):
        raise ValueError("Require 0 <= short_threshold < long_threshold <= 1.")

    torch.set_num_threads(max(1, args.threads))

    manifest_path = resolve_path(args.manifest)
    all_rows = read_manifest(manifest_path)

    rows = [
        row
        for row in all_rows
        if row.get("garment_category", "").strip().lower() in UPPER_BODY_CATEGORIES
    ]

    if not rows:
        raise ValueError("No upper-body garment instances found in manifest.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    per_instance_dir = OUTPUT_DIR / "per_instance"
    per_instance_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    print(f"manifest       : {args.manifest}")
    print(f"all instances  : {len(all_rows)}")
    print(f"sleeve cases   : {len(rows)}")
    print(f"model          : {MODEL_NAME}")
    print("feature        : sleeve_length_ratio [0,1]")
    print(
        f"thresholds     : {args.short_threshold:.2f}, " + f"{args.long_threshold:.2f}"
    )
    print("loading CLIP...")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    prediction_rows = []
    visual_records = []
    grouped_ratios: dict[str, list[float]] = defaultdict(list)

    for index, row in enumerate(rows, start=1):
        garment_id = row["garment_id"].strip()
        image = load_masked_garment(row)

        result = infer_continuous_ratio(
            image=image,
            model=model,
            processor=processor,
            device=device,
        )

        gt_group = deepfashion2_gt_group(garment_id)
        ratio = result["sleeve_length_ratio"]

        bucket = ratio_to_bucket(
            ratio,
            args.short_threshold,
            args.long_threshold,
        )

        if gt_group != "unknown":
            grouped_ratios[gt_group].append(ratio)

        record = {
            "source_image": row["source_image"],
            "garment_id": garment_id,
            "garment_category": row["garment_category"],
            "gt_group": gt_group,
            "sleeve_length_ratio": ratio,
            "bucket_v1": bucket,
            "top_anchor": result["top_anchor"],
            "top_anchor_probability": result["top_anchor_probability"],
            "normalized_entropy": result["normalized_entropy"],
            "top3_json": json.dumps(
                result["top3"],
                ensure_ascii=False,
            ),
        }

        prediction_rows.append(record)

        print(
            f"[{index:02d}/{len(rows):02d}] "
            + f"{garment_id:<35} "
            + f"GT={gt_group:<10} "
            + f"ratio={ratio:.3f} "
            + f"bucket={bucket}"
        )

        visual_path = per_instance_dir / f"{index:02d}_{garment_id}.jpg"

        save_visual(
            image,
            record,
            visual_path,
        )

        visual_records.append((ratio, visual_path))

    write_csv(
        REPORT_DIR / "sleeve_continuous_predictions.csv",
        prediction_rows,
        [
            "source_image",
            "garment_id",
            "garment_category",
            "gt_group",
            "sleeve_length_ratio",
            "bucket_v1",
            "top_anchor",
            "top_anchor_probability",
            "normalized_entropy",
            "top3_json",
        ],
    )

    summary_rows = []

    for group_name in [
        "sleeveless",
        "short",
        "long",
    ]:
        values = grouped_ratios.get(group_name, [])

        if values:
            summary_rows.append(
                {
                    "gt_group": group_name,
                    "n": len(values),
                    "mean_ratio": mean(values),
                    "median_ratio": median(values),
                    "min_ratio": min(values),
                    "max_ratio": max(values),
                }
            )

    write_csv(
        REPORT_DIR / "sleeve_group_summary.csv",
        summary_rows,
        [
            "gt_group",
            "n",
            "mean_ratio",
            "median_ratio",
            "min_ratio",
            "max_ratio",
        ],
    )

    sleeveless = grouped_ratios.get("sleeveless", [])
    short = grouped_ratios.get("short", [])
    long = grouped_ratios.get("long", [])

    diagnostics = []

    for low_name, low_values, high_name, high_values in [
        ("sleeveless", sleeveless, "short", short),
        ("short", short, "long", long),
        ("sleeveless", sleeveless, "long", long),
    ]:
        correct, total, acc = pairwise_order_accuracy(
            low_values,
            high_values,
        )

        diagnostics.append(
            f"{low_name} < {high_name}: "
            + f"{correct}/{total} "
            + "pairwise-order accuracy="
            + (f"{acc:.1%}" if total else "N/A")
        )

    group_means = {
        group: mean(values) for group, values in grouped_ratios.items() if values
    }

    monotonic_mean_order = all(
        group in group_means
        for group in [
            "sleeveless",
            "short",
            "long",
        ]
    ) and (group_means["sleeveless"] < group_means["short"] < group_means["long"])

    diagnostic_text = (
        "PRD 3.1.3 sleeve continuous feature v1\n"
        + "=======================================\n"
        + f"model={MODEL_NAME}\n"
        + f"evaluated_upper_body_instances={len(rows)}\n"
        + f"short_threshold={args.short_threshold}\n"
        + f"long_threshold={args.long_threshold}\n"
        + "\n"
        + "Group means\n"
        + "-----------\n"
        + "\n".join(f"{k}: {v:.4f}" for k, v in group_means.items())
        + "\n\n"
        + "Ordinal diagnostics\n"
        + "-------------------\n"
        + "\n".join(diagnostics)
        + "\n\n"
        + f"mean_order_sleeveless<short<long={monotonic_mean_order}\n"
        + "\n"
        + "Interpretation\n"
        + "--------------\n"
        + "- Main question: does the scalar preserve sleeve-length order?\n"
        + "- Threshold accuracy is secondary because 0.30/0.60 are provisional.\n"
        + "- If ordering is weak, improve the feature representation before tuning thresholds.\n"
        + "- DeepFashion2 categories provide weak ordinal supervision, not continuous GT.\n"
    )

    (REPORT_DIR / "sleeve_order_diagnostics.txt").write_text(
        diagnostic_text,
        encoding="utf-8",
    )

    run_info = (
        "PRD 3.1.3 continuous sleeve feature baseline v1\n"
        + f"model={MODEL_NAME}\n"
        + f"manifest={args.manifest}\n"
        + f"upper_body_instances={len(rows)}\n"
        + "output_feature=sleeve_length_ratio_[0,1]\n"
        + "aggregation=expected_value_over_ordered_CLIP_semantic_anchors\n"
        + f"short_threshold={args.short_threshold}\n"
        + f"long_threshold={args.long_threshold}\n"
        + "gt_source=DeepFashion2_fine_category_name_for_ordinal_diagnostic_only\n"
        + "material_and_craftsmanship=excluded\n"
    )

    (REPORT_DIR / "run_info.txt").write_text(
        run_info,
        encoding="utf-8",
    )

    make_contact_sheet(
        visual_records,
        OUTPUT_DIR / "contact_sheet_sorted_by_ratio.jpg",
    )

    print("\n=== FINISHED ===")
    print(
        "PREDICTIONS : "
        + "reports/prd_attribute_extraction/continuous_v1/"
        + "sleeve_continuous_predictions.csv"
    )
    print(
        "GROUP SUMMARY: "
        + "reports/prd_attribute_extraction/continuous_v1/"
        + "sleeve_group_summary.csv"
    )
    print(
        "DIAGNOSTIC  : "
        + "reports/prd_attribute_extraction/continuous_v1/"
        + "sleeve_order_diagnostics.txt"
    )
    print(
        "CONTACT     : "
        + "outputs/prd_attribute_extraction/continuous_v1/"
        + "contact_sheet_sorted_by_ratio.jpg"
    )
    print("================")


if __name__ == "__main__":
    main()
