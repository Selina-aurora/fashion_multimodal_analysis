"""3.1.3 可视属性提取：关注颜色、图案和服装设计/几何属性；一致性不等于人工真值准确率。

PRD 3.1.3 garment pattern baseline v1.

Pattern is treated as a categorical attribute. The script keeps top-1/top-2
predictions plus uncertainty diagnostics for later manual audit.

Example:
python scripts/attributes/pattern/run_pattern_baseline_v1_fixed.py     --manifest
configs/garment_instances_gt_pilot_100.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
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

REPORT_DIR = PROJECT_ROOT / "reports" / "prd_attribute_extraction" / "pattern_v1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "prd_attribute_extraction" / "pattern_v1"

PATTERN_PROMPTS: Dict[str, str] = {
    "solid": "a garment with a plain solid color and no visible printed pattern",
    "striped": "a garment with clear horizontal or vertical stripes",
    "checked_plaid": "a garment with a checked, plaid, tartan, or grid pattern",
    "floral": "a garment with flowers, floral motifs, leaves, or botanical print",
    "graphic_logo": "a garment with a graphic print, large logo, letters, text, cartoon, illustration, or printed picture",
    "polka_dot": "a garment with repeated polka dots or round dot pattern",
    "animal_print": "a garment with leopard, zebra, snake, tiger, or other animal print",
    "camouflage": "a garment with camouflage or military camo pattern",
    "geometric_abstract": "a garment with repeated geometric shapes or abstract printed pattern",
    "other_pattern": "a garment with a visible pattern that is not solid, striped, checked, floral, graphic, polka dot, animal print, camouflage, or geometric",
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
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    return rows


def load_masked_garment(row: dict[str, str]) -> Image.Image:
    """读取服饰区域并应用掩码，减少背景对属性识别的干扰。

    Args:
        row: 记录字段，使用 crop_path。

    Returns:
        当前读取操作得到的记录、图像或模型对象；保持原始格式约定。
    """
    crop = Image.open(resolve_path(row["crop_path"])).convert("RGB")
    mask_raw = row.get("mask_path", "").strip()

    if not mask_raw:
        return crop

    mask = Image.open(resolve_path(mask_raw)).convert("L")
    if mask.size != crop.size:
        mask = mask.resize(crop.size, Image.Resampling.NEAREST)

    white = Image.new("RGB", crop.size, "white")
    return Image.composite(crop, white, mask)


def infer_pattern(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> dict:
    """推理 图案。

    Args:
        image: 本步骤处理的图像对象。
        model: 已构建的模型对象，由调用方负责选择权重。
        processor: 与模型配套的输入处理器。
        device: 当前计算设备，与输入张量和模型设备保持一致。

    Returns:
        结果字典，主要字段为 top1_pattern, top1_score, top2_pattern, top2_score, top1_top2_margin,
        normalized_entropy, top3。
    """
    labels: List[str] = list(PATTERN_PROMPTS.keys())
    prompts = [PATTERN_PROMPTS[label] for label in labels]

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    probs = outputs.logits_per_image[0].softmax(dim=0).cpu()
    order = torch.argsort(probs, descending=True)

    top3 = []
    for idx in order[:3].tolist():
        idx = int(idx)
        top3.append({"label": labels[idx], "score": float(probs[idx].item())})

    top1, top2 = top3[0], top3[1]
    margin = top1["score"] - top2["score"]

    entropy = float(-torch.sum(probs * torch.log(probs.clamp_min(1e-12))).item())
    normalized_entropy = entropy / math.log(len(labels))

    return {
        "top1_pattern": top1["label"],
        "top1_score": top1["score"],
        "top2_pattern": top2["label"],
        "top2_score": top2["score"],
        "top1_top2_margin": margin,
        "normalized_entropy": normalized_entropy,
        "top3": top3,
    }


def save_visual(image: Image.Image, record: dict, save_path: Path) -> None:
    """保存样本可视化，供错误案例复核。

    Args:
        image: 本步骤处理的图像对象。
        record: 记录字段，使用 garment_id, garment_category, top1_pattern, top1_score,
        top2_pattern, top2_score, top1_top2_margin。
        save_path: 对应文件的相对路径或当前解析后的路径。
    """
    panel_h = 200
    canvas = Image.new("RGB", (image.width, image.height + panel_h), "white")
    canvas.paste(image, (0, panel_h))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    lines = [
        f"{record['garment_id']} | {record['garment_category']}",
        f"top1: {record['top1_pattern']} ({record['top1_score']:.3f})",
        f"top2: {record['top2_pattern']} ({record['top2_score']:.3f})",
        f"margin: {record['top1_top2_margin']:.3f}",
        f"entropy: {record['normalized_entropy']:.3f}",
    ]

    y = 10
    for line in lines:
        draw.text((10, y), line, fill="black", font=font)
        y += 34

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path, quality=92)


def make_contact_sheet(
    records: list[tuple[float, Path]],
    output_path: Path,
    descending: bool,
    limit: int = 40,
) -> None:
    """把样本图与说明拼成审核联系表，便于逐例比较。

    Args:
        records: records。
        output_path: 对应文件的相对路径或当前解析后的路径。
        descending: descending。
        limit: limit。
    """
    if not records:
        return

    records = sorted(records, key=lambda x: x[0], reverse=descending)[:limit]
    tiles = []

    for _, path in records:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((390, 520))
            tile = Image.new("RGB", (400, 530), "white")
            x = (400 - img.width) // 2
            y = (530 - img.height) // 2
            tile.paste(img, (x, y))
            tiles.append(tile)

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 400, rows * 530), "white")

    for idx, tile in enumerate(tiles):
        x = (idx % cols) * 400
        y = (idx // cols) * 530
        sheet.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    manifest_path = resolve_path(args.manifest)
    rows = read_manifest(manifest_path)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    per_instance_dir = OUTPUT_DIR / "per_instance"
    per_instance_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    print(f"manifest       : {args.manifest}")
    print(f"instances      : {len(rows)}")
    print(f"model          : {MODEL_NAME}")
    print("unit           : one garment instance")
    print("task           : categorical pattern baseline")
    print("loading CLIP...")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    output_rows = []
    low_margin_visuals = []
    high_entropy_visuals = []
    label_counts = Counter()

    for index, row in enumerate(rows, start=1):
        image = load_masked_garment(row)
        result = infer_pattern(image, model, processor, device)

        record = {
            "source_image": row["source_image"],
            "garment_id": row["garment_id"],
            "garment_category": row["garment_category"].strip().lower(),
            "top1_pattern": result["top1_pattern"],
            "top1_score": result["top1_score"],
            "top2_pattern": result["top2_pattern"],
            "top2_score": result["top2_score"],
            "top1_top2_margin": result["top1_top2_margin"],
            "normalized_entropy": result["normalized_entropy"],
            "top3_json": json.dumps(result["top3"], ensure_ascii=False),
        }

        output_rows.append(record)
        label_counts[record["top1_pattern"]] += 1

        print(
            f"[{index:03d}/{len(rows):03d}] "
            + f"{record['garment_id']:<30} "
            + f"{record['top1_pattern']:<18} "
            + f"score={record['top1_score']:.3f} "
            + f"margin={record['top1_top2_margin']:.3f} "
            + f"H={record['normalized_entropy']:.3f}"
        )

        visual_path = per_instance_dir / f"{index:03d}_{record['garment_id']}.jpg"
        save_visual(image, record, visual_path)

        low_margin_visuals.append((record["top1_top2_margin"], visual_path))
        high_entropy_visuals.append((record["normalized_entropy"], visual_path))

    pred_fields = [
        "source_image",
        "garment_id",
        "garment_category",
        "top1_pattern",
        "top1_score",
        "top2_pattern",
        "top2_score",
        "top1_top2_margin",
        "normalized_entropy",
        "top3_json",
    ]

    with (REPORT_DIR / "pattern_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=pred_fields)
        writer.writeheader()
        writer.writerows(output_rows)

    audit_fields = [
        "source_image",
        "garment_id",
        "garment_category",
        "predicted_pattern",
        "top1_score",
        "top2_pattern",
        "top1_top2_margin",
        "normalized_entropy",
        "manual_pattern",
        "manual_correct",
        "error_type",
        "notes",
    ]

    with (REPORT_DIR / "pattern_manual_audit_template.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=audit_fields)
        writer.writeheader()

        for row in output_rows:
            writer.writerow(
                {
                    "source_image": row["source_image"],
                    "garment_id": row["garment_id"],
                    "garment_category": row["garment_category"],
                    "predicted_pattern": row["top1_pattern"],
                    "top1_score": row["top1_score"],
                    "top2_pattern": row["top2_pattern"],
                    "top1_top2_margin": row["top1_top2_margin"],
                    "normalized_entropy": row["normalized_entropy"],
                    "manual_pattern": "",
                    "manual_correct": "",
                    "error_type": "",
                    "notes": "",
                }
            )

    make_contact_sheet(
        low_margin_visuals,
        OUTPUT_DIR / "contact_sheet_low_margin.jpg",
        descending=False,
        limit=40,
    )

    make_contact_sheet(
        high_entropy_visuals,
        OUTPUT_DIR / "contact_sheet_high_entropy.jpg",
        descending=True,
        limit=40,
    )

    margins = np.asarray(
        [row["top1_top2_margin"] for row in output_rows],
        dtype=np.float64,
    )
    entropies = np.asarray(
        [row["normalized_entropy"] for row in output_rows],
        dtype=np.float64,
    )

    summary_lines = [
        "PRD 3.1.3 garment pattern baseline v1",
        "====================================",
        f"model={MODEL_NAME}",
        f"instances={len(output_rows)}",
        "unit_of_analysis=one_garment_instance",
        "gt_available=false",
        "",
        "Top-1 distribution",
        "------------------",
    ]

    for label in PATTERN_PROMPTS:
        summary_lines.append(f"{label}={label_counts[label]}")

    summary_lines += [
        "",
        "Uncertainty diagnostics",
        "-----------------------",
        (
            "top1_top2_margin: "
            + f"mean={margins.mean():.4f}, "
            + f"median={np.median(margins):.4f}, "
            + f"min={margins.min():.4f}, "
            + f"max={margins.max():.4f}"
        ),
        (
            "normalized_entropy: "
            + f"mean={entropies.mean():.4f}, "
            + f"median={np.median(entropies):.4f}, "
            + f"min={entropies.min():.4f}, "
            + f"max={entropies.max():.4f}"
        ),
        "",
        "Interpretation",
        "--------------",
        "- Pattern is modeled as a categorical attribute, not a continuous style scalar.",
        "- CLIP softmax scores are candidate-relative, not calibrated probabilities.",
        "- No pattern GT is available in this pilot, so accuracy is not reported.",
        "- Use the audit template plus low-margin/high-entropy sheets for manual error analysis.",
        "- Material/fabric and craftsmanship remain excluded.",
    ]

    (REPORT_DIR / "pattern_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    run_info = (
        "PRD 3.1.3 garment pattern baseline v1\n"
        + f"model={MODEL_NAME}\n"
        + f"manifest={args.manifest}\n"
        + f"instances={len(output_rows)}\n"
        + "unit_of_analysis=one_garment_instance\n"
        + "taxonomy=solid,striped,checked_plaid,floral,graphic_logo,polka_dot,animal_print,camouflage,geometric_abstract,other_pattern\n"
        + "uncertainty=top1_top2_margin+normalized_entropy\n"
        + "gt_available=false\n"
        + "material_and_craftsmanship=excluded\n"
    )

    (REPORT_DIR / "run_info.txt").write_text(run_info, encoding="utf-8")

    print("\n=== FINISHED ===")
    print(
        "PREDICTIONS : reports/prd_attribute_extraction/pattern_v1/pattern_predictions.csv"
    )
    print(
        "AUDIT       : reports/prd_attribute_extraction/pattern_v1/pattern_manual_audit_template.csv"
    )
    print(
        "SUMMARY     : reports/prd_attribute_extraction/pattern_v1/pattern_summary.txt"
    )
    print(
        "LOW MARGIN  : outputs/prd_attribute_extraction/pattern_v1/contact_sheet_low_margin.jpg"
    )
    print(
        "HIGH ENTROPY: outputs/prd_attribute_extraction/pattern_v1/contact_sheet_high_entropy.jpg"
    )
    print("================")


if __name__ == "__main__":
    main()
