"""3.1.3 可视属性提取：关注颜色、图案和服装设计/几何属性；一致性不等于人工真值准确率。

PRD 3.1.3 primary color baseline v1.

Purpose
-------
Run garment-level dominant / primary color recognition after closing the
continuous-style and pattern baselines.

Why CLIP instead of raw pixel quantization?
-------------------------------------------
Earlier pixel-only color quantization was sensitive to shadow, occlusion and
mask contamination. This baseline therefore keeps the semantic CLIP route.

To reduce background-color bias, each garment is rendered on THREE different
backgrounds (white / mid-gray / black). The three normalized CLIP image
embeddings are averaged before color classification.

This is still a baseline:
- color scores are relative CLIP similarities, not calibrated probabilities;
- multicolor garments can be intrinsically ambiguous if one dominant color is
  not obvious;
- secondary color extraction is intentionally out of scope for this v1.

Example
-------
python scripts/attributes/color/run_primary_color_baseline_v1.py     --manifest
configs/garment_instances_gt_pilot_100.csv

Outputs
-------
reports/prd_attribute_extraction/color_v1/
    primary_color_predictions.csv
    primary_color_summary.txt
    primary_color_holdout_audit_template.csv
    run_info.txt

outputs/prd_attribute_extraction/color_v1/
    per_instance/
    contact_sheet_low_margin.jpg
    color_holdout_contact_sheet.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
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

REPORT_DIR = PROJECT_ROOT / "reports" / "prd_attribute_extraction" / "color_v1"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "prd_attribute_extraction" / "color_v1"


COLOR_PROMPTS: Dict[str, List[str]] = {
    "black": [
        "a black garment",
        "clothing whose main dominant color is black",
    ],
    "white": [
        "a white garment",
        "clothing whose main dominant color is white",
    ],
    "gray": [
        "a gray garment",
        "a grey garment",
        "clothing whose main dominant color is gray",
    ],
    "beige": [
        "a beige garment",
        "a cream or ivory garment",
        "clothing whose main dominant color is beige or cream",
    ],
    "brown": [
        "a brown garment",
        "clothing whose main dominant color is brown",
    ],
    "red": [
        "a red garment",
        "clothing whose main dominant color is red",
    ],
    "orange": [
        "an orange garment",
        "clothing whose main dominant color is orange",
    ],
    "yellow": [
        "a yellow garment",
        "clothing whose main dominant color is yellow or golden yellow",
    ],
    "green": [
        "a green garment",
        "clothing whose main dominant color is green",
    ],
    "blue": [
        "a blue garment",
        "clothing whose main dominant color is blue",
        "a denim-blue garment",
    ],
    "purple": [
        "a purple garment",
        "a violet garment",
        "clothing whose main dominant color is purple",
    ],
    "pink": [
        "a pink garment",
        "clothing whose main dominant color is pink or rose",
    ],
}


BACKGROUND_RGB = [
    (255, 255, 255),
    (127, 127, 127),
    (0, 0, 0),
]


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
    """
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_manifest(path: Path) -> list[dict[str, str]]:
    """读取实例清单，不在读取阶段改变样本划分。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    rows = read_csv(path)

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


def l2_normalize(
    x: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """按 L2 范数归一化特征，供余弦相似度比较。

    Args:
        x: 本步骤处理的对象；使用 norm 接口。
        dim: dim。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return x / x.norm(
        dim=dim,
        keepdim=True,
    ).clamp_min(1e-12)


def encode_texts(
    prompts: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> torch.Tensor:
    """编码文本提示，供后续图文相似度比较。

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
        outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )

        pooled = outputs[1]
        projected = model.text_projection(pooled)

    return l2_normalize(projected)


def build_color_prototypes(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> tuple[list[str], torch.Tensor]:
    """构建 颜色 类别原型。

    Args:
        model: 已构建的模型对象，由调用方负责选择权重。
        processor: 与模型配套的输入处理器。
        device: 当前计算设备，与输入张量和模型设备保持一致。

    Returns:
        按顺序返回 labels 等结果。
    """
    labels = list(COLOR_PROMPTS.keys())
    prototypes = []

    for label in labels:
        features = encode_texts(
            COLOR_PROMPTS[label],
            model,
            processor,
            device,
        )

        proto = features.mean(
            dim=0,
            keepdim=True,
        )

        proto = l2_normalize(proto)
        prototypes.append(proto)

    return labels, torch.cat(
        prototypes,
        dim=0,
    )


def get_mask_bbox(mask: Image.Image) -> tuple[int, int, int, int] | None:
    """取得 掩码 边界框。

    Args:
        mask: 掩码。

    Returns:
        按顺序返回 x1, y1, x2, y2 等结果。
    """
    arr = np.asarray(mask, dtype=np.uint8)

    ys, xs = np.where(arr > 20)

    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    return x1, y1, x2, y2


def load_crop_and_mask(
    row: dict[str, str],
) -> tuple[Image.Image, Image.Image]:
    """读取服饰裁剪和对应掩码，保持二者尺寸一致。

    Args:
        row: 记录字段，使用 crop_path。

    Returns:
        按顺序返回 crop, mask 等结果。
    """
    crop = Image.open(resolve_path(row["crop_path"])).convert("RGB")

    mask_raw = row.get(
        "mask_path",
        "",
    ).strip()

    if not mask_raw:
        mask = Image.new(
            "L",
            crop.size,
            255,
        )

        return crop, mask

    mask = Image.open(resolve_path(mask_raw)).convert("L")

    if mask.size != crop.size:
        mask = mask.resize(
            crop.size,
            Image.Resampling.NEAREST,
        )

    bbox = get_mask_bbox(mask)

    if bbox is not None:
        crop = crop.crop(bbox)
        mask = mask.crop(bbox)

    return crop, mask


def composite_on_background(
    crop: Image.Image,
    mask: Image.Image,
    rgb: tuple[int, int, int],
) -> Image.Image:
    """composite on background。

    Args:
        crop: 裁剪。
        mask: 掩码。
        rgb: rgb。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    background = Image.new(
        "RGB",
        crop.size,
        rgb,
    )

    return Image.composite(
        crop,
        background,
        mask,
    )


def encode_image(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> torch.Tensor:
    """编码 图像。

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
        outputs = model.vision_model(
            pixel_values=pixel_values,
        )

        pooled = outputs[1]
        projected = model.visual_projection(pooled)

    return l2_normalize(projected)


def robust_garment_embedding(
    crop: Image.Image,
    mask: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> torch.Tensor:
    """robust garment 嵌入。

    Args:
        crop: 裁剪。
        mask: 掩码。
        model: 已构建的模型对象，由调用方负责选择权重。
        processor: 与模型配套的输入处理器。
        device: 当前计算设备，与输入张量和模型设备保持一致。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    embeddings = []

    for rgb in BACKGROUND_RGB:
        rendered = composite_on_background(
            crop,
            mask,
            rgb,
        )

        emb = encode_image(
            rendered,
            model,
            processor,
            device,
        )

        embeddings.append(emb)

    merged = torch.cat(
        embeddings,
        dim=0,
    ).mean(
        dim=0,
        keepdim=True,
    )

    return l2_normalize(merged)


def classify_color(
    image_feature: torch.Tensor,
    labels: list[str],
    prototypes: torch.Tensor,
    model: CLIPModel,
) -> dict:
    """classify 颜色。

    Args:
        image_feature: 图像 特征。
        labels: 标签。
        prototypes: 类别原型。
        model: 已构建的模型对象，由调用方负责选择权重。

    Returns:
        结果字典，主要字段为 primary_color, top1_score, top2_color, top2_score, top1_top2_margin,
        normalized_entropy, top3。
    """
    scale = model.logit_scale.exp().detach()

    logits = scale * image_feature @ prototypes.T

    probs = logits[0].softmax(dim=0).cpu()

    order = torch.argsort(
        probs,
        descending=True,
    )

    top3 = []

    for idx in order[:3].tolist():
        idx = int(idx)

        top3.append(
            {
                "label": labels[idx],
                "score": float(probs[idx].item()),
            }
        )

    top1 = top3[0]
    top2 = top3[1]

    entropy = float(-torch.sum(probs * torch.log(probs.clamp_min(1e-12))).item())

    normalized_entropy = entropy / math.log(len(labels))

    return {
        "primary_color": (top1["label"]),
        "top1_score": (top1["score"]),
        "top2_color": (top2["label"]),
        "top2_score": (top2["score"]),
        "top1_top2_margin": (top1["score"] - top2["score"]),
        "normalized_entropy": (normalized_entropy),
        "top3": top3,
    }


def visual_source(
    crop: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    # Mid-gray background only for the diagnostic image.
    """可视化 来源。

    Args:
        crop: 裁剪。
        mask: 掩码。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return composite_on_background(
        crop,
        mask,
        (127, 127, 127),
    )


def save_visual(
    image: Image.Image,
    record: dict,
    save_path: Path,
) -> None:
    """保存样本可视化，供错误案例复核。

    Args:
        image: 本步骤处理的图像对象。
        record: 记录字段，使用 garment_id, garment_category, primary_color, top1_score,
        top2_color, top2_score, top1_top2_margin。
        save_path: 对应文件的相对路径或当前解析后的路径。
    """
    panel_h = 205

    canvas = Image.new(
        "RGB",
        (
            image.width,
            image.height + panel_h,
        ),
        "white",
    )

    canvas.paste(
        image,
        (0, panel_h),
    )

    draw = ImageDraw.Draw(canvas)

    font = ImageFont.load_default()

    lines = [
        (f"{record['garment_id']} | " + f"{record['garment_category']}"),
        ("top1: " + f"{record['primary_color']} " + f"({record['top1_score']:.3f})"),
        ("top2: " + f"{record['top2_color']} " + f"({record['top2_score']:.3f})"),
        ("margin: " + f"{record['top1_top2_margin']:.3f}"),
        ("entropy: " + f"{record['normalized_entropy']:.3f}"),
    ]

    y = 10

    for line in lines:
        draw.text(
            (10, y),
            line,
            fill="black",
            font=font,
        )

        y += 36

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        save_path,
        quality=92,
    )


def make_contact_sheet(
    paths: list[Path],
    output_path: Path,
    row_numbers: list[int] | None = None,
    limit: int | None = None,
) -> None:
    """把样本图与说明拼成审核联系表，便于逐例比较。

    Args:
        paths: 路径。
        output_path: 对应文件的相对路径或当前解析后的路径。
        row_numbers: 记录 numbers。
        limit: limit。
    """
    if limit is not None:
        paths = paths[:limit]

    if not paths:
        return

    panels = []

    for idx, path in enumerate(paths):
        with Image.open(path) as img:
            img = img.convert("RGB")

            img.thumbnail((370, 500))

            panel = Image.new(
                "RGB",
                (400, 540),
                "white",
            )

            draw = ImageDraw.Draw(panel)

            font = ImageFont.load_default()

            if row_numbers is not None:
                draw.text(
                    (10, 8),
                    f"ROW {row_numbers[idx]:02d}",
                    fill="black",
                    font=font,
                )

            x = (400 - img.width) // 2

            y = 35

            panel.paste(
                img,
                (x, y),
            )

            panels.append(panel)

    cols = 4

    rows = math.ceil(len(panels) / cols)

    sheet = Image.new(
        "RGB",
        (
            cols * 400,
            rows * 540,
        ),
        "white",
    )

    for idx, panel in enumerate(panels):
        x = (idx % cols) * 400

        y = (idx // cols) * 540

        sheet.paste(
            panel,
            (x, y),
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        output_path,
        quality=92,
    )


def write_predictions(
    rows: list[dict],
) -> None:
    """保存 预测。

    Args:
        rows: 待处理的逐行记录。
    """
    fields = [
        "source_image",
        "garment_id",
        "garment_category",
        "primary_color",
        "top1_score",
        "top2_color",
        "top2_score",
        "top1_top2_margin",
        "normalized_entropy",
        "top3_json",
    ]

    path = REPORT_DIR / "primary_color_predictions.csv"

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def make_holdout(
    rows: list[dict],
    visual_paths: dict[tuple[str, str], Path],
    n: int,
    seed: int,
) -> int:
    """生成 holdout。

    Args:
        rows: 待处理的逐行记录。
        visual_paths: 可视化 路径。
        n: 本步骤使用的原始值，转换/筛选规则见函数体。
        seed: 控制随机过程的种子。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    rng = random.Random(seed)

    shuffled = rows.copy()
    rng.shuffle(shuffled)

    selected = shuffled[
        : min(
            n,
            len(shuffled),
        )
    ]

    fields = [
        "source_image",
        "garment_id",
        "garment_category",
        "predicted_color",
        "top1_score",
        "top2_color",
        "top2_score",
        "top1_top2_margin",
        "normalized_entropy",
        "manual_color",
        "manual_correct",
        "error_type",
        "notes",
    ]

    audit_path = REPORT_DIR / "primary_color_holdout_audit_template.csv"

    with audit_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in selected:
            writer.writerow(
                {
                    "source_image": (row["source_image"]),
                    "garment_id": (row["garment_id"]),
                    "garment_category": (row["garment_category"]),
                    "predicted_color": (row["primary_color"]),
                    "top1_score": (row["top1_score"]),
                    "top2_color": (row["top2_color"]),
                    "top2_score": (row["top2_score"]),
                    "top1_top2_margin": (row["top1_top2_margin"]),
                    "normalized_entropy": (row["normalized_entropy"]),
                    "manual_color": "",
                    "manual_correct": "",
                    "error_type": "",
                    "notes": "",
                }
            )

    holdout_paths = []

    for row in selected:
        key = (
            row["source_image"],
            row["garment_id"],
        )

        holdout_paths.append(visual_paths[key])

    make_contact_sheet(
        holdout_paths,
        OUTPUT_DIR / "color_holdout_contact_sheet.jpg",
        row_numbers=list(
            range(
                1,
                len(holdout_paths) + 1,
            )
        ),
    )

    return len(selected)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--holdout-size",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260918,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    torch.set_num_threads(
        max(
            1,
            args.threads,
        )
    )

    manifest_path = resolve_path(args.manifest)

    rows = read_manifest(manifest_path)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_instance_dir = OUTPUT_DIR / "per_instance"

    per_instance_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device("cpu")

    print(f"manifest       : {args.manifest}")

    print(f"instances      : {len(rows)}")

    print(f"model          : {MODEL_NAME}")

    print("attribute      : primary_color")

    print("backgrounds    : white + gray + black embedding average")

    print("loading CLIP...")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)

    model.eval()

    print("building color prototypes...")

    labels, prototypes = build_color_prototypes(
        model,
        processor,
        device,
    )

    output_rows = []
    visual_paths = {}
    color_counts = Counter()
    margin_records = []

    for index, row in enumerate(
        rows,
        start=1,
    ):
        crop, mask = load_crop_and_mask(row)

        feature = robust_garment_embedding(
            crop,
            mask,
            model,
            processor,
            device,
        )

        result = classify_color(
            feature,
            labels,
            prototypes,
            model,
        )

        record = {
            "source_image": (row["source_image"]),
            "garment_id": (row["garment_id"]),
            "garment_category": (row["garment_category"].strip().lower()),
            "primary_color": (result["primary_color"]),
            "top1_score": (result["top1_score"]),
            "top2_color": (result["top2_color"]),
            "top2_score": (result["top2_score"]),
            "top1_top2_margin": (result["top1_top2_margin"]),
            "normalized_entropy": (result["normalized_entropy"]),
            "top3_json": (
                json.dumps(
                    result["top3"],
                    ensure_ascii=False,
                )
            ),
        }

        output_rows.append(record)

        color_counts[record["primary_color"]] += 1

        print(
            f"[{index:03d}/{len(rows):03d}] "
            + f"{record['garment_id']:<30} "
            + f"{record['primary_color']:<8} "
            + f"score={record['top1_score']:.3f} "
            + f"margin={record['top1_top2_margin']:.3f} "
            + f"H={record['normalized_entropy']:.3f}"
        )

        display_image = visual_source(
            crop,
            mask,
        )

        visual_path = per_instance_dir / (
            f"{index:03d}_" + f"{record['garment_id']}.jpg"
        )

        save_visual(
            display_image,
            record,
            visual_path,
        )

        key = (
            record["source_image"],
            record["garment_id"],
        )

        visual_paths[key] = visual_path

        margin_records.append(
            (
                record["top1_top2_margin"],
                visual_path,
            )
        )

    write_predictions(output_rows)

    margin_records.sort(key=lambda x: x[0])

    make_contact_sheet(
        [path for _, path in margin_records],
        OUTPUT_DIR / "contact_sheet_low_margin.jpg",
        limit=40,
    )

    holdout_n = make_holdout(
        output_rows,
        visual_paths,
        n=args.holdout_size,
        seed=args.seed,
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
        "PRD 3.1.3 primary color baseline v1",
        "===================================",
        f"model={MODEL_NAME}",
        f"instances={len(output_rows)}",
        "method=CLIP_primary_color_with_multi_background_embedding_average",
        "secondary_colors=out_of_scope",
        "gt_available=false_until_holdout_manual_audit",
        "",
        "Primary-color distribution",
        "--------------------------",
    ]

    for color in COLOR_PROMPTS:
        summary_lines.append(f"{color}={color_counts[color]}")

    summary_lines += [
        "",
        "Uncertainty",
        "-----------",
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
        "Validation",
        "----------",
        f"holdout_rows={holdout_n}",
        f"holdout_seed={args.seed}",
        "- Manually label the holdout before reporting color accuracy.",
        "- Scores are candidate-relative CLIP similarities, not calibrated probabilities.",
        "- Multicolor garments can be intrinsically ambiguous under a single-primary-color schema.",
        "- Material/fabric and craftsmanship remain excluded.",
    ]

    (REPORT_DIR / "primary_color_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    run_info = (
        "PRD 3.1.3 primary color baseline v1\n"
        + f"model={MODEL_NAME}\n"
        + f"manifest={args.manifest}\n"
        + f"instances={len(output_rows)}\n"
        + "attribute=primary_color\n"
        + "method=CLIP_multi_background_embedding_average\n"
        + "backgrounds=white,midgray,black\n"
        + "color_taxonomy=black,white,gray,beige,brown,red,orange,yellow,green,blue,purple,pink\n"
        + "secondary_colors=out_of_scope\n"
        + f"holdout_size={holdout_n}\n"
        + f"seed={args.seed}\n"
        + "gt_available=false_until_holdout_manual_audit\n"
        + "material_and_craftsmanship=excluded\n"
    )

    (REPORT_DIR / "run_info.txt").write_text(
        run_info,
        encoding="utf-8",
    )

    print("\n=== FINISHED ===")

    print(
        "PREDICTIONS : "
        + "reports/prd_attribute_extraction/"
        + "color_v1/primary_color_predictions.csv"
    )

    print(
        "SUMMARY     : "
        + "reports/prd_attribute_extraction/"
        + "color_v1/primary_color_summary.txt"
    )

    print(
        "HOLDOUT     : "
        + "reports/prd_attribute_extraction/"
        + "color_v1/primary_color_holdout_audit_template.csv"
    )

    print(
        "HOLDOUT IMG : "
        + "outputs/prd_attribute_extraction/"
        + "color_v1/color_holdout_contact_sheet.jpg"
    )

    print(
        "LOW MARGIN  : "
        + "outputs/prd_attribute_extraction/"
        + "color_v1/contact_sheet_low_margin.jpg"
    )

    print("================")


if __name__ == "__main__":
    main()
