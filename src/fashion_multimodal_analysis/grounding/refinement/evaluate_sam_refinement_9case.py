"""3.1.2 文本引导区域定位：检测覆盖率、粗框可用率和人工定位准确率分开解释。

Run a 9-case SAM refinement pilot on Grounding DINO region proposals.

Goal:
    Test whether a segmentation refinement stage can turn coarse Grounding DINO
    region proposals into tighter local masks / bounding boxes.

Pilot cases:
    collar: collar_001, collar_010, collar_020
    cuff:   cuff_006, cuff_008, cuff_019
    hem:    hem_003, hem_008, hem_023

Proposal priority:
    1. Reuse SPATIAL_HR top box when available.
    2. Otherwise reuse the frozen BASELINE top box.
    3. Otherwise fall back to the fixed spatial prior for that region.

This is a diagnostic pilot. The resulting SAM mask/bbox still requires manual
review as correct / coarse / wrong / missed.

Expected files:
    reports/prd_region_coverage/formal_case_results.csv
    reports/prd_region_coverage/core_region_spatial_hr_15case/case_results.csv
    reports/prd_region_coverage/sam_refinement_9case_selection.csv

Outputs:
    reports/prd_region_coverage/sam_refinement_9case/
        case_results.csv
        summary.csv
        manual_audit_template.csv
        experiment_notes.md

    outputs/sam_refinement_9case/
        collar/
        cuff/
        hem/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

try:
    from transformers import SamModel, SamProcessor
except ImportError as exc:
    raise ImportError(
        "Your transformers installation does not expose SamModel/SamProcessor. "
        + "Please update transformers in the same environment before running."
    ) from exc


DEFAULT_MODEL_ID = "facebook/sam-vit-base"

PROJECT_ROOT = get_project_root()
REPORT_ROOT = PROJECT_ROOT / "reports" / "prd_region_coverage"

SELECTION_FILE = REPORT_ROOT / "sam_refinement_9case_selection.csv"
BASELINE_FILE = REPORT_ROOT / "formal_case_results.csv"
SPATIAL_FILE = REPORT_ROOT / "core_region_spatial_hr_15case" / "case_results.csv"

REPORT_DIR = REPORT_ROOT / "sam_refinement_9case"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sam_refinement_9case"

SPATIAL_WINDOWS = {
    "collar": {
        "upper_center": (0.15, 0.00, 0.85, 0.45),
    },
    "cuff": {
        "left_side": (0.00, 0.00, 0.46, 1.00),
        "right_side": (0.54, 0.00, 1.00, 1.00),
    },
    "hem": {
        "lower_band": (0.00, 0.55, 1.00, 1.00),
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        已通过参数合法性检查的命令行配置。

    Raises:
        ValueError: --limit must be greater than zero.
    """
    parser = argparse.ArgumentParser(
        description="Run the 9-case SAM refinement diagnostic."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"SAM checkpoint (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional case limit for a smoke test.",
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip visualization generation.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero.")
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file.

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV.

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_selection() -> list[dict[str, str]]:
    """Load and validate the fixed 9-case selection.

    Returns:
        当前读取操作得到的记录、图像或模型对象；保持原始格式约定。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    rows = read_csv(SELECTION_FILE)

    if len(rows) != 9:
        raise ValueError(f"Expected 9 pilot cases, got {len(rows)}.")

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["region"]] += 1

    expected = {"collar": 3, "cuff": 3, "hem": 3}
    if dict(counts) != expected:
        raise ValueError(f"Expected 3 cases per region; found {dict(counts)}.")

    return rows


def normalize_project_path(raw_path: str) -> Path:
    """Resolve a stored project-relative path.

    Args:
        raw_path: 对应文件的相对路径或当前解析后的路径。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return PROJECT_ROOT / Path(raw_path.replace("\\", "/"))


def load_baseline_map() -> dict[str, dict[str, str]]:
    """Load frozen 40-case baseline results by candidate id.

    Returns:
        当前读取操作得到的记录、图像或模型对象；保持原始格式约定。
    """
    return {row["candidate_id"]: row for row in read_csv(BASELINE_FILE)}


def load_spatial_map() -> dict[str, dict[str, str]]:
    """Load SPATIAL_HR result rows by candidate id.

    Returns:
        当前读取操作得到的记录、图像或模型对象；保持原始格式约定。
    """
    result = {}
    for row in read_csv(SPATIAL_FILE):
        if row["condition"] == "SPATIAL_HR":
            result[row["candidate_id"]] = row
    return result


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU.

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sam(
    model_id: str,
    device: torch.device,
) -> tuple[SamProcessor, SamModel]:
    """Load SAM processor and model.

    Args:
        model_id: 模型 ID。
        device: 当前计算设备，与输入张量和模型设备保持一致。

    Returns:
        按顺序返回 processor, model 等结果。
    """
    print(f"Loading SAM: {model_id}")
    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model


def normalized_window_box(
    image: Image.Image,
    coords: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    """Convert normalized window coordinates to pixels.

    Args:
        image: 本步骤处理的图像对象。
        coords: coords。

    Returns:
        按顺序返回 left, top, right, bottom 等结果。
    """
    width, height = image.size
    x1, y1, x2, y2 = coords

    left = max(0, min(width - 1, int(round(x1 * width))))
    top = max(0, min(height - 1, int(round(y1 * height))))
    right = max(left + 1, min(width, int(round(x2 * width))))
    bottom = max(top + 1, min(height, int(round(y2 * height))))
    return left, top, right, bottom


def clip_box(
    box: list[float] | tuple[float, float, float, float],
    width: int,
    height: int,
) -> list[float]:
    """Clip a box to image bounds.

    Args:
        box: xyxy 坐标的候选框。
        width: 图像或目标表示的宽度。
        height: 图像或目标表示的高度。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(x1 + 1.0, min(float(width), x2))
    y2 = max(y1 + 1.0, min(float(height), y2))
    return [x1, y1, x2, y2]


def map_local_box_to_parent(
    local_box: list[float],
    window_box: tuple[int, int, int, int],
) -> list[float]:
    """Map a local SPATIAL_HR box back into garment-crop coordinates.

    Args:
        local_box: 局部 边界框。
        window_box: 区域窗口 边界框。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    left, top, _, _ = window_box
    x1, y1, x2, y2 = local_box
    return [
        x1 + left,
        y1 + top,
        x2 + left,
        y2 + top,
    ]


def proposal_from_spatial(
    spatial_row: dict[str, str],
    region: str,
    image: Image.Image,
) -> list[float] | None:
    """Recover the top SPATIAL_HR box in parent coordinates.

    Args:
        spatial_row: 记录字段，使用 selected_window。
        region: 区域。
        image: 本步骤处理的图像对象。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        KeyError: 所需字段不存在。
    """
    if spatial_row.get("detected", "").lower() != "yes":
        return None

    detections = json.loads(spatial_row.get("detections_json", "[]") or "[]")
    if not detections:
        return None

    window_name = spatial_row["selected_window"]
    coords = SPATIAL_WINDOWS[region].get(window_name)
    if coords is None:
        raise KeyError(f"Unknown spatial window {window_name!r} for {region}.")

    window_box = normalized_window_box(image, coords)
    mapped = map_local_box_to_parent(detections[0]["box"], window_box)
    return clip_box(mapped, image.width, image.height)


def proposal_from_baseline(
    baseline_row: dict[str, str],
    image: Image.Image,
) -> list[float] | None:
    """Recover the frozen baseline top box.

    Args:
        baseline_row: baseline 记录。
        image: 本步骤处理的图像对象。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if baseline_row.get("detected", "").lower() != "yes":
        return None

    detections = json.loads(baseline_row.get("detections_json", "[]") or "[]")
    if not detections:
        return None

    return clip_box(
        detections[0]["box"],
        image.width,
        image.height,
    )


def proposal_from_prior(
    region: str,
    image: Image.Image,
    preferred_window: str | None,
) -> list[float]:
    """Use a fixed spatial prior when no detector proposal exists.

    Args:
        region: 区域。
        image: 本步骤处理的图像对象。
        preferred_window: preferred 区域窗口。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    windows = SPATIAL_WINDOWS[region]

    if preferred_window in windows:
        coords = windows[preferred_window]
    else:
        # Deterministic fallback.
        coords = next(iter(windows.values()))

    return [float(value) for value in normalized_window_box(image, coords)]


def choose_proposal(
    candidate_id: str,
    region: str,
    image: Image.Image,
    baseline_row: dict[str, str],
    spatial_row: dict[str, str],
) -> tuple[list[float], str]:
    """Choose SPATIAL_HR, BASELINE, then fixed-prior proposal.

    Args:
        candidate_id: candidate ID。
        region: 区域。
        image: 本步骤处理的图像对象。
        baseline_row: baseline 记录。
        spatial_row: 空间 记录。

    Returns:
        按顺序返回 proposal 等结果。
    """
    proposal = proposal_from_spatial(
        spatial_row=spatial_row,
        region=region,
        image=image,
    )
    if proposal is not None:
        return proposal, "spatial_hr"

    proposal = proposal_from_baseline(
        baseline_row=baseline_row,
        image=image,
    )
    if proposal is not None:
        return proposal, "baseline"

    preferred_window = spatial_row.get("selected_window", "")
    proposal = proposal_from_prior(
        region=region,
        image=image,
        preferred_window=preferred_window,
    )
    return proposal, "fixed_spatial_prior"


def run_sam(
    image: Image.Image,
    box: list[float],
    processor: SamProcessor,
    model: SamModel,
    device: torch.device,
) -> tuple[np.ndarray, float, float]:
    """Run SAM with one box prompt and return the best predicted mask.

    Args:
        image: 本步骤处理的图像对象。
        box: xyxy 坐标的候选框。
        processor: 与模型配套的输入处理器。
        model: 已构建的模型对象，由调用方负责选择权重。
        device: 当前计算设备，与输入张量和模型设备保持一致。

    Returns:
        按顺序返回 mask, best_score, elapsed 等结果。
    """
    inputs = processor(
        images=image,
        input_boxes=[[[float(v) for v in box]]],
        return_tensors="pt",
    )

    original_sizes = inputs["original_sizes"].clone()
    reshaped_sizes = inputs["reshaped_input_sizes"].clone()

    tensor_inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            **tensor_inputs,
            multimask_output=True,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        original_sizes,
        reshaped_sizes,
    )

    candidate_masks = masks[0][0]
    scores = outputs.iou_scores.detach().cpu()[0][0]

    best_index = int(torch.argmax(scores).item())
    best_score = float(scores[best_index].item())

    mask_tensor = candidate_masks[best_index]
    mask = mask_tensor.numpy().astype(bool)

    return mask, best_score, elapsed


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    """Return the tight bbox around a binary mask.

    Args:
        mask: 掩码。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]


def box_area_ratio(
    box: list[float] | list[int],
    width: int,
    height: int,
) -> float:
    """Return box area divided by image area.

    Args:
        box: xyxy 坐标的候选框。
        width: 图像或目标表示的宽度。
        height: 图像或目标表示的高度。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    x1, y1, x2, y2 = box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area / max(1.0, float(width * height))


def mask_area_ratio(mask: np.ndarray) -> float:
    """Return mask foreground proportion.

    Args:
        mask: 掩码。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return float(mask.mean())


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a readable font with fallback.

    Args:
        size: size。

    Returns:
        当前读取操作得到的记录、图像或模型对象；保持原始格式约定。
    """
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    """Overlay a semi-transparent mask for visual inspection.

    Args:
        image: 本步骤处理的图像对象。
        mask: 掩码。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

    mask_image = Image.fromarray(
        (mask.astype(np.uint8) * 110),
        mode="L",
    )
    tint = Image.new("RGBA", image.size, (255, 0, 0, 0))
    tint.putalpha(mask_image)
    overlay.alpha_composite(tint)

    return Image.alpha_composite(base, overlay).convert("RGB")


def draw_box(
    image: Image.Image,
    box: list[float] | list[int],
    title: str,
) -> Image.Image:
    """Draw a box and title.

    Args:
        image: 本步骤处理的图像对象。
        box: xyxy 坐标的候选框。
        title: title。

    Returns:
        返回 output，由函数体中同名变量的计算/收集过程得到。
    """
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = load_font(14)

    draw.rectangle(box, outline="red", width=4)
    draw.rectangle((0, 0, output.width, 28), fill="white")
    draw.text((5, 6), title, fill="black", font=font)
    return output


def fit_panel(
    image: Image.Image,
    size: tuple[int, int],
) -> Image.Image:
    """Letterbox an image into a fixed panel.

    Args:
        image: 本步骤处理的图像对象。
        size: size。

    Returns:
        返回 canvas，由函数体中同名变量的计算/收集过程得到。
    """
    canvas = Image.new("RGB", size, "white")
    copy = image.copy()
    copy.thumbnail(size)
    x = (size[0] - copy.width) // 2
    y = (size[1] - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def save_visual(
    image: Image.Image,
    proposal_box: list[float],
    mask: np.ndarray,
    refined_box: list[int],
    candidate_id: str,
    proposal_source: str,
    output_path: Path,
) -> None:
    """Save proposal / SAM mask / refined bbox comparison.

    Args:
        image: 本步骤处理的图像对象。
        proposal_box: proposal 边界框。
        mask: 掩码。
        refined_box: refined 边界框。
        candidate_id: candidate ID。
        proposal_source: proposal 来源。
        output_path: 对应文件的相对路径或当前解析后的路径。
    """
    size = (420, 520)

    proposal_panel = fit_panel(
        draw_box(
            image,
            proposal_box,
            f"{candidate_id} | proposal ({proposal_source})",
        ),
        size,
    )

    mask_panel_raw = overlay_mask(image, mask)
    mask_draw = ImageDraw.Draw(mask_panel_raw)
    mask_draw.rectangle((0, 0, mask_panel_raw.width, 28), fill="white")
    mask_draw.text(
        (5, 6),
        f"{candidate_id} | SAM mask",
        fill="black",
        font=load_font(14),
    )
    mask_panel = fit_panel(mask_panel_raw, size)

    bbox_panel = fit_panel(
        draw_box(
            image,
            refined_box,
            f"{candidate_id} | SAM refined bbox",
        ),
        size,
    )

    sheet = Image.new("RGB", (size[0] * 3, size[1]), "white")
    sheet.paste(proposal_panel, (0, 0))
    sheet.paste(mask_panel, (size[0], 0))
    sheet.paste(bbox_panel, (size[0] * 2, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def make_contact_sheet(
    region: str,
    paths: list[Path],
) -> None:
    """Combine case visualizations for one region.

    Args:
        region: 区域。
        paths: 路径。
    """
    if not paths:
        return

    panel_width = 1100
    panel_height = 430

    prepared = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
        prepared.append(fit_panel(image, (panel_width, panel_height)))

    sheet = Image.new(
        "RGB",
        (panel_width, panel_height * len(prepared)),
        "white",
    )

    for index, panel in enumerate(prepared):
        sheet.paste(panel, (0, index * panel_height))

    path = OUTPUT_DIR / region / f"{region}_sam_refinement_contact_sheet.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def _mean(values: list[float]) -> float | None:
    """Return arithmetic mean or None.

    Args:
        values: 本步骤处理的数值或附加字段。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize SAM refinement diagnostics.

    Args:
        rows: 待处理的逐行记录。

    Returns:
        返回 output，由函数体中同名变量的计算/收集过程得到。
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["region"]].append(row)

    output = []

    for region in ("collar", "cuff", "hem", "OVERALL"):
        group = rows if region == "OVERALL" else groups[region]

        proposal_areas = [float(row["proposal_box_area_ratio"]) for row in group]
        refined_areas = [
            float(row["sam_bbox_area_ratio"])
            for row in group
            if str(row["sam_bbox_area_ratio"]).strip()
        ]
        mask_areas = [
            float(row["sam_mask_area_ratio"])
            for row in group
            if str(row["sam_mask_area_ratio"]).strip()
        ]
        iou_scores = [
            float(row["sam_predicted_iou"])
            for row in group
            if str(row["sam_predicted_iou"]).strip()
        ]
        times = [float(row["sam_inference_seconds"]) for row in group]

        output.append(
            {
                "region": region,
                "n": len(group),
                "mean_proposal_box_area_ratio": round(
                    _mean(proposal_areas) or 0.0,
                    4,
                ),
                "mean_sam_bbox_area_ratio": (
                    round(_mean(refined_areas), 4) if refined_areas else ""
                ),
                "mean_sam_mask_area_ratio": (
                    round(_mean(mask_areas), 4) if mask_areas else ""
                ),
                "mean_sam_predicted_iou": (
                    round(_mean(iou_scores), 4) if iou_scores else ""
                ),
                "mean_sam_inference_seconds": round(
                    _mean(times) or 0.0,
                    4,
                ),
            }
        )

    return output


def write_notes(model_id: str) -> None:
    """Write experiment notes.

    Args:
        model_id: 模型 ID。
    """
    text = f"""# SAM refinement 9-case pilot

## Purpose

Test whether SAM can refine coarse Grounding DINO / spatial proposals into
tighter local masks and bounding boxes for collar, cuff, and hem.

## Cases

- collar: collar_001, collar_010, collar_020
- cuff: cuff_006, cuff_008, cuff_019
- hem: hem_003, hem_008, hem_023

## SAM model

`{model_id}`

## Proposal priority

1. SPATIAL_HR prediction when available.
2. Frozen BASELINE Grounding DINO prediction.
3. Fixed target-specific spatial prior when neither detector returns a box.

## Evaluation

Automatic:
- proposal bbox area ratio
- SAM refined bbox area ratio
- SAM mask area ratio
- SAM predicted IoU score
- inference time

Manual:
- correct
- coarse
- wrong
- missed

This pilot is intended to determine whether a segmentation-refinement stage is
worth integrating into PRD 3.1.2. It is not a final acceptance result.
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "experiment_notes.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the 9-case SAM refinement pilot.

    Raises:
        KeyError: 所需字段不存在。
        FileNotFoundError: 需要的文件不存在。
    """
    args = parse_args()

    selection = load_selection()
    if args.limit is not None:
        selection = selection[: args.limit]

    baseline_map = load_baseline_map()
    spatial_map = load_spatial_map()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Cases: {len(selection)}")
    print("Note: the first run may download a large SAM checkpoint.")

    processor, model = load_sam(args.model_id, device)

    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    visuals: dict[str, list[Path]] = defaultdict(list)

    for index, selected in enumerate(selection, start=1):
        candidate_id = selected["candidate_id"]
        region = selected["region"]

        if candidate_id not in baseline_map:
            raise KeyError(f"Missing baseline row: {candidate_id}")
        if candidate_id not in spatial_map:
            raise KeyError(f"Missing SPATIAL_HR row: {candidate_id}")

        baseline = baseline_map[candidate_id]
        spatial = spatial_map[candidate_id]

        image_path = normalize_project_path(baseline["candidate_crop_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")

        image = Image.open(image_path).convert("RGB")

        proposal_box, proposal_source = choose_proposal(
            candidate_id=candidate_id,
            region=region,
            image=image,
            baseline_row=baseline,
            spatial_row=spatial,
        )

        print(
            f"[{index}/{len(selection)}] "
            + f"{candidate_id} ({region}) "
            + f"proposal={proposal_source}"
        )

        mask, predicted_iou, elapsed = run_sam(
            image=image,
            box=proposal_box,
            processor=processor,
            model=model,
            device=device,
        )

        refined_box = mask_bbox(mask)

        proposal_ratio = box_area_ratio(
            proposal_box,
            image.width,
            image.height,
        )

        if refined_box is None:
            bbox_ratio: float | str = ""
            mask_ratio: float | str = ""
        else:
            bbox_ratio = box_area_ratio(
                refined_box,
                image.width,
                image.height,
            )
            mask_ratio = mask_area_ratio(mask)

        row = {
            "candidate_id": candidate_id,
            "region": region,
            "image_name": baseline["image_name"],
            "proposal_source": proposal_source,
            "proposal_box_json": json.dumps(proposal_box),
            "proposal_box_area_ratio": round(proposal_ratio, 6),
            "sam_predicted_iou": round(predicted_iou, 6),
            "sam_bbox_json": (
                json.dumps(refined_box) if refined_box is not None else ""
            ),
            "sam_bbox_area_ratio": (
                round(float(bbox_ratio), 6) if bbox_ratio != "" else ""
            ),
            "sam_mask_area_ratio": (
                round(float(mask_ratio), 6) if mask_ratio != "" else ""
            ),
            "sam_inference_seconds": elapsed,
        }
        rows.append(row)

        audit_rows.append(
            {
                "candidate_id": candidate_id,
                "region": region,
                "proposal_source": proposal_source,
                "proposal_box_area_ratio": round(proposal_ratio, 6),
                "sam_bbox_area_ratio": (
                    round(float(bbox_ratio), 6) if bbox_ratio != "" else ""
                ),
                "sam_mask_area_ratio": (
                    round(float(mask_ratio), 6) if mask_ratio != "" else ""
                ),
                "sam_predicted_iou": round(predicted_iou, 6),
                "sam_localization_quality": "",
                "review_note": "",
            }
        )

        if not args.skip_visuals and refined_box is not None:
            visual_path = OUTPUT_DIR / region / f"{candidate_id}_sam_refinement.jpg"
            save_visual(
                image=image,
                proposal_box=proposal_box,
                mask=mask,
                refined_box=refined_box,
                candidate_id=candidate_id,
                proposal_source=proposal_source,
                output_path=visual_path,
            )
            visuals[region].append(visual_path)

    write_csv(REPORT_DIR / "case_results.csv", rows)
    write_csv(REPORT_DIR / "summary.csv", summarize(rows))
    write_csv(
        REPORT_DIR / "manual_audit_template.csv",
        audit_rows,
    )
    write_notes(args.model_id)

    if not args.skip_visuals:
        for region, paths in visuals.items():
            make_contact_sheet(region, paths)

    print("\nFinished.")
    print(f"Case results: {REPORT_DIR / 'case_results.csv'}")
    print(f"Summary: {REPORT_DIR / 'summary.csv'}")
    print("Manual audit template: " + f"{REPORT_DIR / 'manual_audit_template.csv'}")
    print(f"Visuals: {OUTPUT_DIR}")
    print(
        "\nNext step: review the three region contact sheets "
        + "and label correct/coarse/wrong/missed."
    )


if __name__ == "__main__":
    main()
