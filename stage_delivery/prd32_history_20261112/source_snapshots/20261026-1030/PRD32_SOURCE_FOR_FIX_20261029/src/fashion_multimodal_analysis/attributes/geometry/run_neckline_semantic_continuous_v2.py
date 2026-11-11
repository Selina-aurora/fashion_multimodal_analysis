"""3.1.3 可视属性提取：关注颜色、图案和服装设计/几何属性；一致性不等于人工真值准确率。

PRD 3.1.3 local-neckline continuous semantic pilot v2.

Goal
----
Represent neckline using several continuous semantic dimensions instead of a
single hard class such as crew / V-neck / shirt collar.

This version intentionally uses a LOCAL upper-center crop because the previous
whole-garment-mask geometry pilot showed that outer garment masks do not
reliably preserve the internal neckline boundary.

Continuous dimensions
---------------------
1. neckline_depth_score in [0,1]
   0.00 = very shallow / close to the neck
   1.00 = very deep / plunging opening

2. neckline_width_score in [0,1]
   0.00 = narrow opening
   1.00 = very wide opening across the shoulders/chest

3. collar_height_score in [0,1]
   0.00 = no raised collar / flat open neckline
   1.00 = tall raised collar covering much of the neck

Method
------
For each dimension, CLIP compares the local neckline crop against ordered
descriptions. The final scalar is the expected anchor value:

    continuous_score = sum(prob_i * anchor_i)

IMPORTANT
---------
- This is still a semantic continuous descriptor, not physical ground truth.
- DeepFashion2 provides no neckline GT for these dimensions.
- The current local crop is a top-center proxy. In the integration stage it
  should be replaced by the actual 3.1.2 collar/neckline ROI when available.
- No hard neckline labels or business thresholds are set here.
- Material/fabric and craftsmanship remain excluded.

Example
-------
python scripts/attributes/geometry/run_neckline_semantic_continuous_v2.py     --manifest
configs/garment_instances_gt_pilot_100.csv

Outputs
-------
reports/prd_attribute_extraction/neckline_semantic_v2/
    neckline_semantic_predictions.csv
    neckline_semantic_summary.txt
    run_info.txt

outputs/prd_attribute_extraction/neckline_semantic_v2/
    local_crops/
    per_instance/
    contact_sheet_sorted_by_depth.jpg
    contact_sheet_sorted_by_width.jpg
    contact_sheet_sorted_by_collar_height.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

REPORT_DIR = (
    PROJECT_ROOT / "reports" / "prd_attribute_extraction" / "neckline_semantic_v2"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "prd_attribute_extraction" / "neckline_semantic_v2"
)

UPPER_BODY_CATEGORIES = {
    "top",
    "outerwear",
    "dress",
}

DEPTH_ANCHORS: Dict[str, tuple[float, str]] = {
    "very_shallow": (
        0.00,
        "a garment neckline opening that is very shallow and sits very close to the neck",
    ),
    "shallow": (
        0.25,
        "a garment neckline opening that is shallow and close to the base of the neck",
    ),
    "medium": (
        0.50,
        "a garment neckline opening with medium depth on the upper chest",
    ),
    "deep": (
        0.75,
        "a garment neckline opening that extends clearly downward on the chest",
    ),
    "very_deep": (
        1.00,
        "a garment neckline opening that is very deep and plunging",
    ),
}

WIDTH_ANCHORS: Dict[str, tuple[float, str]] = {
    "very_narrow": (
        0.00,
        "a garment neckline opening that is very narrow around the neck",
    ),
    "narrow": (
        0.25,
        "a garment neckline opening that is relatively narrow",
    ),
    "medium": (
        0.50,
        "a garment neckline opening with medium horizontal width",
    ),
    "wide": (
        0.75,
        "a garment neckline opening that is wide across the upper chest",
    ),
    "very_wide": (
        1.00,
        "a garment neckline opening that is very wide toward the shoulders",
    ),
}

COLLAR_HEIGHT_ANCHORS: Dict[str, tuple[float, str]] = {
    "none_or_flat": (
        0.00,
        "a garment neckline with no raised collar, flat around the neck",
    ),
    "low": (
        0.25,
        "a garment with a low collar rising only slightly around the neck",
    ),
    "medium": (
        0.50,
        "a garment with a medium-height raised collar around the neck",
    ),
    "high": (
        0.75,
        "a garment with a high collar covering a substantial part of the neck",
    ),
    "very_high": (
        1.00,
        "a garment with a very tall high neck or turtleneck covering most of the neck",
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


def load_crop_and_mask(
    row: dict[str, str],
) -> tuple[Image.Image, Image.Image | None]:
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
        mask = mask.resize(
            crop.size,
            Image.Resampling.NEAREST,
        )

    return crop, mask


def tight_bbox_from_mask(
    mask: Image.Image | None,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """tight 边界框 from 掩码。

    Args:
        mask: 掩码。
        image_size: 图像的宽、高，用于处理坐标和掩码尺寸。

    Returns:
        按顺序返回 width, height 等结果。
    """
    width, height = image_size

    if mask is None:
        return (0, 0, width, height)

    arr = np.asarray(mask, dtype=np.uint8) > 127

    if not arr.any():
        return (0, 0, width, height)

    ys, xs = np.nonzero(arr)

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def make_local_neckline_crop(
    crop: Image.Image,
    mask: Image.Image | None,
) -> Image.Image:
    """Create a top-center local proxy crop around the neckline region.

    Args:
        crop: 裁剪。
        mask: 掩码。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    x1, y1, x2, y2 = tight_bbox_from_mask(
        mask,
        crop.size,
    )

    garment_w = max(1, x2 - x1)
    garment_h = max(1, y2 - y1)

    # Slightly wider than yesterday's proxy to preserve wide/boat necks.
    local_x1 = x1 + int(round(garment_w * 0.08))
    local_x2 = x1 + int(round(garment_w * 0.92))
    local_y1 = y1
    local_y2 = y1 + int(round(garment_h * 0.48))

    local_x1 = max(0, min(crop.width - 1, local_x1))
    local_x2 = max(local_x1 + 1, min(crop.width, local_x2))
    local_y1 = max(0, min(crop.height - 1, local_y1))
    local_y2 = max(local_y1 + 1, min(crop.height, local_y2))

    local_crop = crop.crop((local_x1, local_y1, local_x2, local_y2))

    if mask is None:
        return local_crop

    local_mask = mask.crop((local_x1, local_y1, local_x2, local_y2))

    white = Image.new(
        "RGB",
        local_crop.size,
        "white",
    )

    return Image.composite(
        local_crop,
        white,
        local_mask,
    )


def infer_dimension(
    image: Image.Image,
    anchors: Dict[str, tuple[float, str]],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> dict:
    """推理 dimension。

    Args:
        image: 本步骤处理的图像对象。
        anchors: anchors。
        model: 已构建的模型对象，由调用方负责选择权重。
        processor: 与模型配套的输入处理器。
        device: 当前计算设备，与输入张量和模型设备保持一致。

    Returns:
        结果字典，主要字段为 score, top_anchor, top_anchor_probability, normalized_entropy, top3。
    """
    labels: List[str] = list(anchors.keys())

    values = torch.tensor(
        [anchors[label][0] for label in labels],
        dtype=torch.float32,
    )

    prompts = [anchors[label][1] for label in labels]

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

    score = float(torch.sum(probs * values).item())

    order = torch.argsort(
        probs,
        descending=True,
    )

    top3 = []

    for idx in order[:3].tolist():
        idx = int(idx)
        label = labels[idx]

        top3.append(
            {
                "anchor_label": label,
                "anchor_value": anchors[label][0],
                "probability": float(probs[idx].item()),
            }
        )

    entropy = float(-torch.sum(probs * torch.log(probs.clamp_min(1e-12))).item())

    normalized_entropy = entropy / math.log(len(labels))

    return {
        "score": score,
        "top_anchor": top3[0]["anchor_label"],
        "top_anchor_probability": top3[0]["probability"],
        "normalized_entropy": normalized_entropy,
        "top3": top3,
    }


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    """按确定的字段顺序保存实验 CSV。

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。
        fieldnames: fieldnames。
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        writer.writerows(rows)


def save_visual(
    local_crop: Image.Image,
    record: dict,
    save_path: Path,
) -> None:
    """保存样本可视化，供错误案例复核。

    Args:
        local_crop: 局部 裁剪。
        record: 记录字段，使用 garment_id, garment_category, neckline_depth_score,
        depth_top_anchor, neckline_width_score, width_top_anchor, collar_height_score。
        save_path: 对应文件的相对路径或当前解析后的路径。
    """
    panel_h = 215

    canvas = Image.new(
        "RGB",
        (
            local_crop.width,
            local_crop.height + panel_h,
        ),
        "white",
    )

    canvas.paste(
        local_crop,
        (0, panel_h),
    )

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    lines = [
        (f"{record['garment_id']} | " + f"{record['garment_category']}"),
        (
            "depth_score: "
            + f"{record['neckline_depth_score']:.3f} "
            + f"({record['depth_top_anchor']})"
        ),
        (
            "width_score: "
            + f"{record['neckline_width_score']:.3f} "
            + f"({record['width_top_anchor']})"
        ),
        (
            "collar_height: "
            + f"{record['collar_height_score']:.3f} "
            + f"({record['collar_top_anchor']})"
        ),
        ("depth_uncertainty: " + f"{record['depth_entropy']:.3f}"),
        ("width_uncertainty: " + f"{record['width_entropy']:.3f}"),
    ]

    y = 10

    for line in lines:
        draw.text(
            (10, y),
            line,
            fill="black",
            font=font,
        )
        y += 31

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        save_path,
        quality=92,
    )


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

    visual_records = sorted(
        visual_records,
        key=lambda item: item[0],
    )

    tiles = []

    for _, path in visual_records:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((390, 500))

            tile = Image.new(
                "RGB",
                (400, 510),
                "white",
            )

            x = (400 - img.width) // 2
            y = (510 - img.height) // 2

            tile.paste(
                img,
                (x, y),
            )

            tiles.append(tile)

    cols = 4
    rows = (len(tiles) + cols - 1) // cols

    sheet = Image.new(
        "RGB",
        (
            cols * 400,
            rows * 510,
        ),
        "white",
    )

    for idx, tile in enumerate(tiles):
        x = (idx % cols) * 400
        y = (idx // cols) * 510

        sheet.paste(
            tile,
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


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        ValueError: No upper-body garment instances found.
    """
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

    all_rows = read_manifest(manifest_path)

    rows = [
        row
        for row in all_rows
        if (
            row.get(
                "garment_category",
                "",
            )
            .strip()
            .lower()
            in UPPER_BODY_CATEGORIES
        )
    ]

    if not rows:
        raise ValueError("No upper-body garment instances found.")

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    local_dir = OUTPUT_DIR / "local_crops"

    per_instance_dir = OUTPUT_DIR / "per_instance"

    local_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_instance_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device("cpu")

    print(f"manifest       : {args.manifest}")
    print(f"all instances  : {len(all_rows)}")
    print(f"neckline cases : {len(rows)}")
    print(f"model          : {MODEL_NAME}")
    print("features       : depth_score, width_score, collar_height_score")
    print("input          : local upper-center neckline proxy")
    print("loading CLIP...")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)

    model.eval()

    output_rows = []

    depth_visuals = []
    width_visuals = []
    collar_visuals = []

    for index, row in enumerate(
        rows,
        start=1,
    ):
        garment_id = row["garment_id"].strip()

        category = row["garment_category"].strip().lower()

        crop, mask = load_crop_and_mask(row)

        local_crop = make_local_neckline_crop(
            crop,
            mask,
        )

        depth = infer_dimension(
            local_crop,
            DEPTH_ANCHORS,
            model,
            processor,
            device,
        )

        width = infer_dimension(
            local_crop,
            WIDTH_ANCHORS,
            model,
            processor,
            device,
        )

        collar = infer_dimension(
            local_crop,
            COLLAR_HEIGHT_ANCHORS,
            model,
            processor,
            device,
        )

        record = {
            "source_image": (row["source_image"]),
            "garment_id": garment_id,
            "garment_category": category,
            "neckline_depth_score": (depth["score"]),
            "depth_top_anchor": (depth["top_anchor"]),
            "depth_top_anchor_probability": (depth["top_anchor_probability"]),
            "depth_entropy": (depth["normalized_entropy"]),
            "neckline_width_score": (width["score"]),
            "width_top_anchor": (width["top_anchor"]),
            "width_top_anchor_probability": (width["top_anchor_probability"]),
            "width_entropy": (width["normalized_entropy"]),
            "collar_height_score": (collar["score"]),
            "collar_top_anchor": (collar["top_anchor"]),
            "collar_top_anchor_probability": (collar["top_anchor_probability"]),
            "collar_entropy": (collar["normalized_entropy"]),
            "depth_top3_json": (
                json.dumps(
                    depth["top3"],
                    ensure_ascii=False,
                )
            ),
            "width_top3_json": (
                json.dumps(
                    width["top3"],
                    ensure_ascii=False,
                )
            ),
            "collar_top3_json": (
                json.dumps(
                    collar["top3"],
                    ensure_ascii=False,
                )
            ),
        }

        output_rows.append(record)

        local_path = local_dir / (f"{index:03d}_" + f"{garment_id}_local.jpg")

        local_crop.save(
            local_path,
            quality=95,
        )

        visual_path = per_instance_dir / (f"{index:03d}_" + f"{garment_id}.jpg")

        save_visual(
            local_crop,
            record,
            visual_path,
        )

        depth_visuals.append(
            (
                record["neckline_depth_score"],
                visual_path,
            )
        )

        width_visuals.append(
            (
                record["neckline_width_score"],
                visual_path,
            )
        )

        collar_visuals.append(
            (
                record["collar_height_score"],
                visual_path,
            )
        )

        print(
            f"[{index:03d}/{len(rows):03d}] "
            + f"{garment_id:<35} "
            + f"depth={record['neckline_depth_score']:.3f} "
            + f"width={record['neckline_width_score']:.3f} "
            + f"collar={record['collar_height_score']:.3f}"
        )

    write_csv(
        REPORT_DIR / "neckline_semantic_predictions.csv",
        output_rows,
        [
            "source_image",
            "garment_id",
            "garment_category",
            "neckline_depth_score",
            "depth_top_anchor",
            "depth_top_anchor_probability",
            "depth_entropy",
            "neckline_width_score",
            "width_top_anchor",
            "width_top_anchor_probability",
            "width_entropy",
            "collar_height_score",
            "collar_top_anchor",
            "collar_top_anchor_probability",
            "collar_entropy",
            "depth_top3_json",
            "width_top3_json",
            "collar_top3_json",
        ],
    )

    depth_values = [row["neckline_depth_score"] for row in output_rows]

    width_values = [row["neckline_width_score"] for row in output_rows]

    collar_values = [row["collar_height_score"] for row in output_rows]

    summary = (
        "PRD 3.1.3 local neckline continuous semantic pilot v2\n"
        + "=====================================================\n"
        + f"model={MODEL_NAME}\n"
        + f"upper_body_instances={len(output_rows)}\n"
        + "input=top-center local neckline proxy\n"
        + "gt_available=false\n"
        + "hard_classification=false\n"
        + "\n"
        + "neckline_depth_score\n"
        + "--------------------\n"
        + f"mean={float(np.mean(depth_values)):.4f}\n"
        + f"median={float(np.median(depth_values)):.4f}\n"
        + f"min={float(np.min(depth_values)):.4f}\n"
        + f"max={float(np.max(depth_values)):.4f}\n"
        + "\n"
        + "neckline_width_score\n"
        + "--------------------\n"
        + f"mean={float(np.mean(width_values)):.4f}\n"
        + f"median={float(np.median(width_values)):.4f}\n"
        + f"min={float(np.min(width_values)):.4f}\n"
        + f"max={float(np.max(width_values)):.4f}\n"
        + "\n"
        + "collar_height_score\n"
        + "-------------------\n"
        + f"mean={float(np.mean(collar_values)):.4f}\n"
        + f"median={float(np.median(collar_values)):.4f}\n"
        + f"min={float(np.min(collar_values)):.4f}\n"
        + f"max={float(np.max(collar_values)):.4f}\n"
        + "\n"
        + "Interpretation\n"
        + "--------------\n"
        + "- These are semantic continuous descriptors, not exact physical measurements.\n"
        + "- Main validation is qualitative ordering on the three sorted contact sheets.\n"
        + "- Current local crop is a proxy; replace with actual 3.1.2 collar/neckline ROI later.\n"
        + "- Do not define business thresholds from this pilot alone.\n"
    )

    (REPORT_DIR / "neckline_semantic_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    run_info = (
        "PRD 3.1.3 local neckline continuous semantic pilot v2\n"
        + f"model={MODEL_NAME}\n"
        + f"manifest={args.manifest}\n"
        + f"upper_body_instances={len(output_rows)}\n"
        + "input=local_top_center_proxy\n"
        + "output_features=neckline_depth_score,neckline_width_score,collar_height_score\n"
        + "aggregation=expected_value_over_ordered_CLIP_semantic_anchors\n"
        + "hard_classification=false\n"
        + "gt_available=false\n"
        + "future_integration=replace_proxy_with_3.1.2_collar_roi\n"
        + "material_and_craftsmanship=excluded\n"
    )

    (REPORT_DIR / "run_info.txt").write_text(
        run_info,
        encoding="utf-8",
    )

    make_contact_sheet(
        depth_visuals,
        OUTPUT_DIR / "contact_sheet_sorted_by_depth.jpg",
    )

    make_contact_sheet(
        width_visuals,
        OUTPUT_DIR / "contact_sheet_sorted_by_width.jpg",
    )

    make_contact_sheet(
        collar_visuals,
        OUTPUT_DIR / "contact_sheet_sorted_by_collar_height.jpg",
    )

    print("\n=== FINISHED ===")

    print(
        "PREDICTIONS : "
        + "reports/prd_attribute_extraction/neckline_semantic_v2/"
        + "neckline_semantic_predictions.csv"
    )

    print(
        "SUMMARY     : "
        + "reports/prd_attribute_extraction/neckline_semantic_v2/"
        + "neckline_semantic_summary.txt"
    )

    print(
        "DEPTH       : "
        + "outputs/prd_attribute_extraction/neckline_semantic_v2/"
        + "contact_sheet_sorted_by_depth.jpg"
    )

    print(
        "WIDTH       : "
        + "outputs/prd_attribute_extraction/neckline_semantic_v2/"
        + "contact_sheet_sorted_by_width.jpg"
    )

    print(
        "COLLAR      : "
        + "outputs/prd_attribute_extraction/neckline_semantic_v2/"
        + "contact_sheet_sorted_by_collar_height.jpg"
    )

    print("================")


if __name__ == "__main__":
    main()
