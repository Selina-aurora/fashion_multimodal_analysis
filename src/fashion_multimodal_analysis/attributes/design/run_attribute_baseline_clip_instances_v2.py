"""3.1.3 可视属性提取：关注颜色、图案和服装设计/几何属性；一致性不等于人工真值准确率。

PRD 3.1.3 per-garment attribute baseline v2.

Changes from v1
---------------
1. COLOR no longer uses CLIP.
   It is estimated from pixels INSIDE the garment mask:
       mask pixels -> adaptive RGB quantization -> semantic color bins
   This supports:
       primary_color
       secondary_colors
       color_distribution
       color_mode

2. Pattern / sleeve length / neckline still use CLIP.

3. One row = one garment instance.

Current scope
-------------
Included:
- primary_color
- secondary_colors
- color_mode
- pattern
- sleeve_length
- neckline

Excluded:
- material / fabric composition
- craftsmanship / manufacturing process

Example
-------
python scripts/attributes/design/run_attribute_baseline_clip_instances_v2.py
--manifest configs/garment_instances_gt_pilot.csv

Outputs
-------
reports/prd_attribute_extraction/per_garment_v2/
    garment_attribute_predictions_v2.json
    garment_attribute_predictions_v2.csv
    manual_audit_template_v2.csv
    run_info.txt

outputs/prd_attribute_extraction/per_garment_v2/
    per_instance/
    contact_sheet.jpg
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
from collections import defaultdict
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

REPORT_DIR = PROJECT_ROOT / "reports" / "prd_attribute_extraction" / "per_garment_v2"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "prd_attribute_extraction" / "per_garment_v2"

PATTERN_PROMPTS = {
    "solid": "a fashion garment with a plain solid appearance",
    "striped": "a fashion garment with stripes",
    "checked": "a fashion garment with a checkered plaid pattern",
    "floral": "a fashion garment with a floral flower pattern",
    "graphic": "a fashion garment with printed graphics letters or logos",
    "polka_dot": "a fashion garment with polka dots",
    "animal": "a fashion garment with an animal print pattern",
    "camouflage": "a fashion garment with camouflage pattern",
    "geometric": "a fashion garment with geometric shapes pattern",
    "abstract": "a fashion garment with an abstract pattern",
    "other": "a fashion garment with another visible decorative pattern",
    "unknown": "a fashion garment whose pattern cannot be determined",
}

SLEEVE_PROMPTS = {
    "sleeveless": "a sleeveless fashion garment with no sleeves",
    "cap": "a fashion garment with very short cap sleeves",
    "short": "a fashion garment with short sleeves above the elbow",
    "elbow": "a fashion garment with sleeves ending around the elbow",
    "three_quarter": "a fashion garment with three quarter sleeves below the elbow",
    "long": "a fashion garment with long sleeves reaching the wrist",
    "unknown": "a fashion garment whose sleeve length cannot be determined",
}

NECKLINE_PROMPTS = {
    "crew": "a fashion garment with a crew neck neckline",
    "v_neck": "a fashion garment with a V neck neckline",
    "round": "a fashion garment with a round scoop neckline",
    "square": "a fashion garment with a square neckline",
    "boat": "a fashion garment with a boat neckline",
    "halter": "a fashion garment with a halter neckline",
    "off_shoulder": "an off shoulder fashion garment",
    "one_shoulder": "a one shoulder fashion garment",
    "turtleneck": "a fashion garment with a turtleneck",
    "mock_neck": "a fashion garment with a mock neck",
    "polo": "a fashion garment with a polo collar",
    "shirt_collar": "a fashion garment with a shirt collar",
    "hooded": "a fashion garment with a hood",
    "other": "a fashion garment with another visible neckline type",
    "unknown": "a fashion garment whose neckline cannot be determined",
}

NO_UPPER_BODY_ATTRS = {
    "pants",
    "trousers",
    "jeans",
    "shorts",
    "skirt",
    "shoes",
    "shoe",
    "bag",
    "accessory",
    "accessories",
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


def load_crop_and_mask(row: dict[str, str]) -> tuple[Image.Image, Image.Image | None]:
    """读取服饰裁剪和对应掩码，保持二者尺寸一致。

    Args:
        row: 记录字段，使用 crop_path。

    Returns:
        按顺序返回 crop, mask 等结果。
    """
    crop_path = resolve_path(row["crop_path"])
    crop = Image.open(crop_path).convert("RGB")

    mask_raw = row.get("mask_path", "").strip()
    if not mask_raw:
        return crop, None

    mask_path = resolve_path(mask_raw)
    mask = Image.open(mask_path).convert("L")

    if mask.size != crop.size:
        mask = mask.resize(crop.size, Image.Resampling.NEAREST)

    return crop, mask


def masked_for_clip(crop: Image.Image, mask: Image.Image | None) -> Image.Image:
    """masked for CLIP。

    Args:
        crop: 裁剪。
        mask: 掩码。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if mask is None:
        return crop

    white = Image.new("RGB", crop.size, "white")
    return Image.composite(crop, white, mask)


def semantic_color_name(rgb: tuple[int, int, int]) -> str:
    """Map an RGB cluster center to a compact fashion color vocabulary.

    Args:
        rgb: rgb。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    r, g, b = [int(v) for v in rgb]

    rf = r / 255.0
    gf = g / 255.0
    bf = b / 255.0

    h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
    h *= 360.0

    # Achromatic colors first.
    if v <= 0.20:
        return "black"

    if s <= 0.12:
        if v >= 0.86:
            return "white"
        return "gray"

    # Very light warm low-saturation colors are usually beige/cream.
    if 20.0 <= h <= 60.0 and 0.08 <= s <= 0.38 and v >= 0.62:
        return "beige"

    # Dark warm hues.
    if 10.0 <= h <= 50.0 and v <= 0.58:
        return "brown"

    # Pink: high-value reds/magentas.
    if (h >= 325.0 or h <= 15.0) and v >= 0.62 and s <= 0.68:
        return "pink"

    if h < 15.0 or h >= 345.0:
        return "red"
    if h < 45.0:
        return "orange"
    if h < 70.0:
        return "yellow"
    if h < 170.0:
        return "green"
    if h < 255.0:
        return "blue"
    if h < 315.0:
        return "purple"
    return "pink"


def quantized_color_distribution(
    crop: Image.Image,
    mask: Image.Image | None,
    palette_colors: int = 8,
    max_pixels: int = 60000,
) -> list[dict]:
    """Estimate semantic color shares from garment pixels only.

    Args:
        crop: 裁剪。
        mask: 掩码。
        palette_colors: palette colors。
        max_pixels: max pixels。

    Returns:
        返回 distribution，由函数体中同名变量的计算/收集过程得到。
    """
    rgb = np.asarray(crop, dtype=np.uint8)

    if mask is not None:
        m = np.asarray(mask, dtype=np.uint8) > 127
        pixels = rgb[m]
    else:
        pixels = rgb.reshape(-1, 3)

    if pixels.size == 0:
        return [{"label": "unknown", "share": 1.0, "rgb": [0, 0, 0]}]

    # Deterministic subsampling for speed.
    if len(pixels) > max_pixels:
        indices = np.linspace(
            0,
            len(pixels) - 1,
            max_pixels,
            dtype=np.int64,
        )
        pixels = pixels[indices]

    # Pillow median-cut quantization on valid garment pixels only.
    strip = Image.fromarray(
        pixels.reshape(1, -1, 3),
        mode="RGB",
    )

    quantized = strip.quantize(
        colors=palette_colors,
        method=Image.Quantize.MEDIANCUT,
    )

    counts = quantized.getcolors(maxcolors=max_pixels + 1)
    palette = quantized.getpalette()

    if not counts or not palette:
        return [{"label": "unknown", "share": 1.0, "rgb": [0, 0, 0]}]

    semantic_counts: dict[str, int] = defaultdict(int)
    weighted_rgb: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(3, dtype=np.float64)
    )

    total = 0

    for count, palette_index in counts:
        base = palette_index * 3
        rgb_center = tuple(palette[base : base + 3])

        if len(rgb_center) != 3:
            continue

        label = semantic_color_name(rgb_center)

        semantic_counts[label] += int(count)
        weighted_rgb[label] += np.asarray(rgb_center, dtype=np.float64) * int(count)
        total += int(count)

    if total <= 0:
        return [{"label": "unknown", "share": 1.0, "rgb": [0, 0, 0]}]

    distribution = []

    for label, count in semantic_counts.items():
        avg_rgb = weighted_rgb[label] / max(count, 1)

        distribution.append(
            {
                "label": label,
                "share": count / total,
                "rgb": [int(round(x)) for x in avg_rgb.tolist()],
            }
        )

    distribution.sort(
        key=lambda item: item["share"],
        reverse=True,
    )

    return distribution


def infer_color_attributes(
    crop: Image.Image,
    mask: Image.Image | None,
    secondary_min_share: float = 0.08,
    secondary_relative_to_primary: float = 0.22,
) -> dict:
    """推理 颜色 属性。

    Args:
        crop: 裁剪。
        mask: 掩码。
        secondary_min_share: secondary min share。
        secondary_relative_to_primary: secondary relative 转换 primary。

    Returns:
        结果字典，主要字段为 primary_color, secondary_colors, color_mode, color_distribution,
        method。
    """
    distribution = quantized_color_distribution(
        crop,
        mask,
    )

    primary = distribution[0]

    secondary = []

    for item in distribution[1:]:
        if len(secondary) >= 2:
            break

        if (
            item["share"] >= secondary_min_share
            and item["share"] >= primary["share"] * secondary_relative_to_primary
        ):
            secondary.append(item["label"])

    return {
        "primary_color": {
            "label": primary["label"],
            "share": primary["share"],
        },
        "secondary_colors": secondary,
        "color_mode": ("multicolor" if secondary else "single_color"),
        "color_distribution": distribution,
        "method": "masked_pixel_adaptive_quantization",
    }


def clip_rank(
    image: Image.Image,
    label_prompts: Dict[str, str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> list[dict]:
    """使用 CLIP 图文相似度给当前候选标签排序。

    Args:
        image: 本步骤处理的图像对象。
        label_prompts: 标签 提示文本。
        model: 已构建的模型对象，由调用方负责选择权重。
        processor: 与模型配套的输入处理器。
        device: 当前计算设备，与输入张量和模型设备保持一致。

    Returns:
        返回 ranked，由函数体中同名变量的计算/收集过程得到。
    """
    labels: List[str] = list(label_prompts.keys())
    prompts: List[str] = [label_prompts[label] for label in labels]

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

    probs = outputs.logits_per_image[0].softmax(dim=0)
    order = torch.argsort(probs, descending=True)

    ranked = []

    for idx in order.tolist():
        idx = int(idx)
        ranked.append(
            {
                "label": labels[idx],
                "score": float(probs[idx].item()),
            }
        )

    return ranked


def infer_clip_attribute(
    image: Image.Image,
    prompts: Dict[str, str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> dict:
    """推理 CLIP 属性。

    Args:
        image: 本步骤处理的图像对象。
        prompts: 提示文本。
        model: 已构建的模型对象，由调用方负责选择权重。
        processor: 与模型配套的输入处理器。
        device: 当前计算设备，与输入张量和模型设备保持一致。

    Returns:
        结果字典，主要字段为 label, score, top3, score_note。
    """
    ranked = clip_rank(
        image,
        prompts,
        model,
        processor,
        device,
    )

    return {
        "label": ranked[0]["label"],
        "score": ranked[0]["score"],
        "top3": ranked[:3],
        "score_note": ("relative CLIP candidate-set score; not calibrated probability"),
    }


def save_visual(
    image: Image.Image,
    record: dict,
    save_path: Path,
) -> None:
    """保存样本可视化，供错误案例复核。

    Args:
        image: 本步骤处理的图像对象。
        record: 记录字段，使用 attributes, garment_id, garment_category。
        save_path: 对应文件的相对路径或当前解析后的路径。
    """
    panel_h = 230

    canvas = Image.new(
        "RGB",
        (image.width, image.height + panel_h),
        "white",
    )

    canvas.paste(
        image,
        (0, panel_h),
    )

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    attrs = record["attributes"]

    draw.text(
        (12, 10),
        (f"{record['garment_id']} | " + f"{record['garment_category']}"),
        fill="black",
        font=font,
    )

    y = 38

    lines = [
        (
            "primary_color: "
            + f"{attrs['primary_color']['label']} "
            + f"share={attrs['primary_color']['share']:.3f}"
        ),
        (
            "secondary_colors: "
            + (
                ", ".join(attrs["secondary_colors"])
                if attrs["secondary_colors"]
                else "[]"
            )
        ),
        f"color_mode: {attrs['color_mode']}",
        f"pattern: {attrs['pattern']['label']}",
        f"sleeve_length: {attrs['sleeve_length']['label']}",
        f"neckline: {attrs['neckline']['label']}",
    ]

    for line in lines:
        draw.text(
            (12, y),
            line,
            fill="black",
            font=font,
        )
        y += 28

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
) -> None:
    """把样本图与说明拼成审核联系表，便于逐例比较。

    Args:
        paths: 路径。
        output_path: 对应文件的相对路径或当前解析后的路径。
    """
    if not paths:
        return

    tiles = []

    for path in paths:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((420, 560))

            tile = Image.new(
                "RGB",
                (430, 570),
                "white",
            )

            x = (430 - img.width) // 2
            y = (570 - img.height) // 2
            tile.paste(img, (x, y))
            tiles.append(tile)

    cols = 3
    rows = (len(tiles) + cols - 1) // cols

    sheet = Image.new(
        "RGB",
        (430 * cols, 570 * rows),
        "white",
    )

    for idx, tile in enumerate(tiles):
        x = (idx % cols) * 430
        y = (idx // cols) * 570
        sheet.paste(tile, (x, y))

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    sheet.save(
        output_path,
        quality=92,
    )


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))

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

    print(f"manifest     : {args.manifest}")
    print(f"instances    : {len(rows)}")
    print(f"model        : {MODEL_NAME}")
    print("color method : masked pixel adaptive quantization")
    print("loading CLIP...")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    records = []
    csv_rows = []
    visual_paths = []

    for index, row in enumerate(rows, start=1):
        garment_id = row["garment_id"].strip() or f"garment_{index:02d}"

        category = row["garment_category"].strip().lower() or "unknown"

        crop, mask = load_crop_and_mask(row)

        clip_image = masked_for_clip(
            crop,
            mask,
        )

        print(f"[{index:02d}/{len(rows):02d}] " + f"{garment_id} | {category}")

        color_result = infer_color_attributes(
            crop,
            mask,
        )

        pattern_result = infer_clip_attribute(
            clip_image,
            PATTERN_PROMPTS,
            model,
            processor,
            device,
        )

        if category in NO_UPPER_BODY_ATTRS:
            sleeve_result = {
                "label": "not_applicable",
                "score": 1.0,
                "top3": [],
                "score_note": "forced by garment category",
            }
            neckline_result = {
                "label": "not_applicable",
                "score": 1.0,
                "top3": [],
                "score_note": "forced by garment category",
            }
        else:
            sleeve_result = infer_clip_attribute(
                clip_image,
                SLEEVE_PROMPTS,
                model,
                processor,
                device,
            )
            neckline_result = infer_clip_attribute(
                clip_image,
                NECKLINE_PROMPTS,
                model,
                processor,
                device,
            )

        record = {
            "source_image": row["source_image"],
            "garment_id": garment_id,
            "garment_category": category,
            "region_source": ("mask" if row.get("mask_path", "").strip() else "crop"),
            "crop_path": row["crop_path"],
            "mask_path": row.get("mask_path", ""),
            "model": MODEL_NAME,
            "attributes": {
                "primary_color": color_result["primary_color"],
                "secondary_colors": color_result["secondary_colors"],
                "color_mode": color_result["color_mode"],
                "color_distribution": color_result["color_distribution"],
                "color_method": color_result["method"],
                "pattern": pattern_result,
                "sleeve_length": sleeve_result,
                "neckline": neckline_result,
            },
        }

        records.append(record)

        csv_rows.append(
            {
                "source_image": row["source_image"],
                "garment_id": garment_id,
                "garment_category": category,
                "primary_color": (color_result["primary_color"]["label"]),
                "primary_color_share": (color_result["primary_color"]["share"]),
                "secondary_colors": ";".join(color_result["secondary_colors"]),
                "color_mode": color_result["color_mode"],
                "pattern": pattern_result["label"],
                "sleeve_length": sleeve_result["label"],
                "neckline": neckline_result["label"],
            }
        )

        visual_path = per_instance_dir / f"{index:02d}_{garment_id}.jpg"

        save_visual(
            clip_image,
            record,
            visual_path,
        )

        visual_paths.append(visual_path)

    json_path = REPORT_DIR / "garment_attribute_predictions_v2.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": "3.1.3-v2.1-per-garment",
                "model": MODEL_NAME,
                "scope": (
                    "per garment instance; "
                    + "color from garment-mask pixels; "
                    + "material and craftsmanship excluded"
                ),
                "garments": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = REPORT_DIR / "garment_attribute_predictions_v2.csv"

    csv_fields = [
        "source_image",
        "garment_id",
        "garment_category",
        "primary_color",
        "primary_color_share",
        "secondary_colors",
        "color_mode",
        "pattern",
        "sleeve_length",
        "neckline",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields,
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    audit_path = REPORT_DIR / "manual_audit_template_v2.csv"

    audit_fields = [
        "source_image",
        "garment_id",
        "garment_category",
        "primary_color_pred",
        "primary_color_gt",
        "primary_color_correct",
        "secondary_colors_pred",
        "secondary_colors_gt",
        "secondary_colors_correct",
        "pattern_pred",
        "pattern_gt",
        "pattern_correct",
        "sleeve_length_pred",
        "sleeve_length_gt",
        "sleeve_length_correct",
        "neckline_pred",
        "neckline_gt",
        "neckline_correct",
        "overall_note",
    ]

    audit_rows = []

    for row in csv_rows:
        audit_rows.append(
            {
                "source_image": row["source_image"],
                "garment_id": row["garment_id"],
                "garment_category": row["garment_category"],
                "primary_color_pred": row["primary_color"],
                "primary_color_gt": "",
                "primary_color_correct": "",
                "secondary_colors_pred": row["secondary_colors"],
                "secondary_colors_gt": "",
                "secondary_colors_correct": "",
                "pattern_pred": row["pattern"],
                "pattern_gt": "",
                "pattern_correct": "",
                "sleeve_length_pred": row["sleeve_length"],
                "sleeve_length_gt": "",
                "sleeve_length_correct": "",
                "neckline_pred": row["neckline"],
                "neckline_gt": "",
                "neckline_correct": "",
                "overall_note": "",
            }
        )

    with audit_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=audit_fields,
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    make_contact_sheet(
        visual_paths,
        OUTPUT_DIR / "contact_sheet.jpg",
    )

    run_info = (
        "PRD 3.1.3 per-garment baseline v2\n"
        + f"model_for_semantic_attributes={MODEL_NAME}\n"
        + "color_method=masked_pixel_adaptive_quantization\n"
        + f"instances={len(rows)}\n"
        + "unit_of_analysis=one garment instance per row\n"
        + "material_and_craftsmanship=excluded\n"
    )

    (REPORT_DIR / "run_info.txt").write_text(
        run_info,
        encoding="utf-8",
    )

    print("\n=== FINISHED ===")
    print(
        "JSON    : "
        + "reports/prd_attribute_extraction/per_garment_v2/"
        + "garment_attribute_predictions_v2.json"
    )
    print(
        "CSV     : "
        + "reports/prd_attribute_extraction/per_garment_v2/"
        + "garment_attribute_predictions_v2.csv"
    )
    print(
        "AUDIT   : "
        + "reports/prd_attribute_extraction/per_garment_v2/"
        + "manual_audit_template_v2.csv"
    )
    print(
        "CONTACT : "
        + "outputs/prd_attribute_extraction/per_garment_v2/"
        + "contact_sheet.jpg"
    )
    print("================")


if __name__ == "__main__":
    main()
