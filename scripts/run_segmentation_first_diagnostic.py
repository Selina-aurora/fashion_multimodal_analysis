"""Run a segmentation-first tight-crop diagnostic for Grounding DINO.

This diagnostic answers whether restricting the visual input helps fine-grained
part localization after the prompt-only experiment showed that more explicit
language increases non-empty predictions but still produces garment-level boxes.

Design:
1. Reuse the 20 fixed cases from prompt_diagnostic_2026_09_14.
2. Reuse the existing part-specific ORIGINAL result (no extra baseline inference).
3. Build an oracle garment mask from the DeepFashion2 item annotation. This
   isolates crop/mask geometry from current instance-segmentation model errors.
4. Run the same part-specific prompt on:
   - SEGMENT_TIGHT: garment-only tight crop.
   - SEGMENT_SPATIAL: garment-only tight crop + fixed target-specific windows.
5. Report both:
   - box area relative to the current model input;
   - effective box area relative to the original image.

This is a diagnostic, not a final PRD accuracy result. Manual localization
quality still needs visual review (correct / coarse / wrong / missed).
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

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_THRESHOLD = 0.30

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "fashion_data" / "raw" / "train" / "train"
IMAGE_DIR = DATASET_ROOT / "image"
ANNO_DIR = DATASET_ROOT / "annos"

PROMPT_REPORT_DIR = PROJECT_ROOT / "reports" / "prompt_diagnostic_2026_09_14"
SELECTED_CASES_FILE = PROMPT_REPORT_DIR / "selected_cases.csv"
PROMPT_CASE_RESULTS_FILE = PROMPT_REPORT_DIR / "case_results.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "segmentation_first_diagnostic_2026_09_14"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "segmentation_first_diagnostic_2026_09_14"

PROMPTS = {
    "sleeve": "the sleeve area of the garment",
    "collar": "the collar around the neck opening of the garment",
    "button": "the small button detail on the garment",
    "zipper": "the zipper closure detail on the garment",
}

# These windows are fixed before this diagnostic. They follow the v3 local
# search geometry, but are applied AFTER garment segmentation/tight cropping.
SPATIAL_WINDOWS = {
    "sleeve": [
        (0.00, 0.02, 0.36, 0.75),
        (0.64, 0.02, 1.00, 0.75),
    ],
    "collar": [
        (0.20, 0.00, 0.80, 0.34),
    ],
    "button": [
        (0.28, 0.06, 0.50, 0.94),
        (0.39, 0.06, 0.61, 0.94),
        (0.50, 0.06, 0.72, 0.94),
    ],
    "zipper": [
        (0.28, 0.06, 0.50, 0.97),
        (0.39, 0.06, 0.61, 0.97),
        (0.50, 0.06, 0.72, 0.97),
    ],
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the segmentation-first tight-crop diagnostic."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Detection threshold (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional limit for a quick smoke test.",
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip comparison image generation.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.max_cases is not None and args.max_cases <= 0:
        raise ValueError("--max-cases must be greater than zero.")
    return args


def load_csv(path: Path) -> list[dict[str, str]]:
    """Load a CSV file as dictionaries."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_original_baselines() -> dict[str, dict[str, str]]:
    """Load the already-computed part-specific ORIGINAL results."""
    rows = load_csv(PROMPT_CASE_RESULTS_FILE)
    baselines = {}
    for row in rows:
        if row["prompt_variant"] == "part_specific":
            baselines[row["benchmark_id"]] = row
    return baselines


def load_model(device: torch.device) -> tuple[Any, Any]:
    """Load Grounding DINO and its processor."""
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    return processor, model


def _annotation_item(annotation: dict[str, Any], item_id: str) -> dict[str, Any]:
    """Return the requested DeepFashion2 item annotation."""
    if item_id in annotation:
        return annotation[item_id]

    # Defensive fallback for values such as item1 / item_1.
    digits = "".join(char for char in item_id if char.isdigit())
    candidates = [f"item{digits}", f"item_{digits}"] if digits else []
    for candidate in candidates:
        if candidate in annotation:
            return annotation[candidate]

    raise KeyError(
        f"Cannot find {item_id!r} in annotation. "
        f"Available item keys: {[key for key in annotation if key.startswith('item')]}"
    )


def _iter_polygons(segmentation: Any) -> list[list[float]]:
    """Normalize DeepFashion2 segmentation into polygon coordinate lists."""
    if not segmentation:
        return []

    if (
        isinstance(segmentation, list)
        and segmentation
        and all(isinstance(value, (int, float)) for value in segmentation)
    ):
        return [segmentation]

    polygons = []
    if isinstance(segmentation, list):
        for polygon in segmentation:
            if (
                isinstance(polygon, list)
                and polygon
                and all(isinstance(value, (int, float)) for value in polygon)
            ):
                polygons.append(polygon)
    return polygons


def build_oracle_garment_crop(
    image: Image.Image,
    annotation: dict[str, Any],
    item_id: str,
    padding_ratio: float = 0.02,
) -> tuple[Image.Image, tuple[int, int, int, int], float]:
    """Mask background using GT garment segmentation and return a tight crop.

    Returns:
        cropped_image: garment-only crop on white background.
        crop_box: (left, top, right, bottom) in original-image coordinates.
        crop_area_ratio: crop area divided by original-image area.
    """
    item = _annotation_item(annotation, item_id)
    width, height = image.size

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    polygons = _iter_polygons(item.get("segmentation"))
    for polygon in polygons:
        if len(polygon) < 6:
            continue
        points = [
            (float(polygon[index]), float(polygon[index + 1]))
            for index in range(0, len(polygon) - 1, 2)
        ]
        draw.polygon(points, fill=255)

    mask_bbox = mask.getbbox()

    # Fall back to the annotation bounding box if polygon data is unavailable.
    if mask_bbox is None:
        raw_bbox = item.get("bounding_box") or item.get("bbox")
        if not raw_bbox or len(raw_bbox) != 4:
            raise ValueError(
                f"No usable segmentation or bounding box for {item_id}."
            )

        x1, y1, x2, y2 = [float(value) for value in raw_bbox]

        # DeepFashion2 normally stores x1,y1,x2,y2. If a malformed/alternative
        # x,y,w,h representation is encountered, this fallback prevents a
        # zero/negative crop.
        if x2 <= x1 or y2 <= y1:
            x2 = x1 + max(1.0, float(raw_bbox[2]))
            y2 = y1 + max(1.0, float(raw_bbox[3]))

        mask_bbox = (
            int(math.floor(x1)),
            int(math.floor(y1)),
            int(math.ceil(x2)),
            int(math.ceil(y2)),
        )
        ImageDraw.Draw(mask).rectangle(mask_bbox, fill=255)

    left, top, right, bottom = mask_bbox
    box_width = max(1, right - left)
    box_height = max(1, bottom - top)

    pad_x = int(round(box_width * padding_ratio))
    pad_y = int(round(box_height * padding_ratio))

    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(width, right + pad_x)
    bottom = min(height, bottom + pad_y)

    crop_box = (left, top, right, bottom)

    white = Image.new("RGB", image.size, "white")
    masked = Image.composite(image, white, mask)
    cropped = masked.crop(crop_box)

    crop_area = max(1, (right - left) * (bottom - top))
    original_area = max(1, width * height)
    crop_area_ratio = crop_area / original_area

    return cropped, crop_box, crop_area_ratio


def normalized_window_to_box(
    image: Image.Image,
    window: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    """Convert a normalized crop window to integer pixel coordinates."""
    width, height = image.size
    x1, y1, x2, y2 = window

    left = max(0, min(width - 1, int(round(x1 * width))))
    top = max(0, min(height - 1, int(round(y1 * height))))
    right = max(left + 1, min(width, int(round(x2 * width))))
    bottom = max(top + 1, min(height, int(round(y2 * height))))
    return left, top, right, bottom


def infer(
    image: Image.Image,
    prompt: str,
    threshold: float,
    processor: Any,
    model: Any,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    """Run Grounding DINO on one image-prompt pair."""
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model(**inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    text_labels = result.get("text_labels")
    labels = text_labels if text_labels is not None else result.get("labels", [])

    detections = []
    for score, label, box in zip(
        result["scores"],
        labels,
        result["boxes"],
    ):
        detections.append(
            {
                "label": str(label),
                "score": float(score),
                "box": [float(value) for value in box.tolist()],
            }
        )
    return detections, elapsed


def box_area(box: list[float]) -> float:
    """Return non-negative box area."""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def detection_row(
    case: dict[str, str],
    condition: str,
    prompt: str,
    detections: list[dict[str, Any]],
    inference_seconds: float,
    input_size: tuple[int, int],
    input_region_original: tuple[int, int, int, int],
    original_size: tuple[int, int],
    source: str,
    selected_window: str = "",
) -> dict[str, Any]:
    """Build one structured result row."""
    input_width, input_height = input_size
    original_width, original_height = original_size

    top_detection = (
        max(detections, key=lambda item: item["score"]) if detections else None
    )

    input_area = max(1, input_width * input_height)
    original_area = max(1, original_width * original_height)

    region_left, region_top, region_right, region_bottom = input_region_original
    region_area = max(
        1,
        (region_right - region_left) * (region_bottom - region_top),
    )
    input_region_area_ratio = region_area / original_area

    if top_detection:
        predicted_area = box_area(top_detection["box"])
        top_score = top_detection["score"]
        top_box_area_ratio_input = predicted_area / input_area

        # The model prediction is expressed in current-input pixels. Scaling by
        # input-region/original area gives its effective area in original-image
        # coordinates without needing the exact mapped corner coordinates.
        top_box_area_ratio_original = (
            top_box_area_ratio_input * input_region_area_ratio
        )
    else:
        top_score = ""
        top_box_area_ratio_input = ""
        top_box_area_ratio_original = ""

    return {
        "benchmark_id": case["benchmark_id"],
        "target": case["target"],
        "image_name": case["image_name"],
        "item_id": case["item_id"],
        "category_name": case["category_name"],
        "condition": condition,
        "prompt": prompt,
        "threshold": "",
        "detected": "yes" if detections else "no",
        "num_detections": len(detections),
        "top_score": top_score,
        "top_box_area_ratio_input": top_box_area_ratio_input,
        "input_region_area_ratio_original": input_region_area_ratio,
        "top_box_area_ratio_original": top_box_area_ratio_original,
        "inference_seconds": inference_seconds,
        "selected_window": selected_window,
        "result_source": source,
        "detections_json": json.dumps(detections),
    }


def baseline_row(
    case: dict[str, str],
    baseline: dict[str, str],
    original_size: tuple[int, int],
) -> dict[str, Any]:
    """Convert the existing prompt diagnostic baseline to the new schema."""
    detected = baseline["detected"].strip().lower() == "yes"
    area_ratio = baseline["top_box_area_ratio"].strip()

    return {
        "benchmark_id": case["benchmark_id"],
        "target": case["target"],
        "image_name": case["image_name"],
        "item_id": case["item_id"],
        "category_name": case["category_name"],
        "condition": "ORIGINAL",
        "prompt": baseline["prompt"],
        "threshold": baseline["threshold"],
        "detected": "yes" if detected else "no",
        "num_detections": baseline["num_detections"],
        "top_score": baseline["top_score"],
        "top_box_area_ratio_input": area_ratio,
        "input_region_area_ratio_original": 1.0,
        "top_box_area_ratio_original": area_ratio,
        "inference_seconds": baseline["inference_seconds"],
        "selected_window": "",
        "result_source": "reused_prompt_diagnostic",
        "detections_json": baseline["detections_json"],
    }


def draw_prediction(
    image: Image.Image,
    detections: list[dict[str, Any]],
    title: str,
) -> Image.Image:
    """Draw detections and a title on an image."""
    panel = image.copy()
    draw = ImageDraw.Draw(panel)

    for detection in detections:
        draw.rectangle(detection["box"], outline="red", width=4)

    draw.rectangle((0, 0, panel.width, 30), fill="white")
    draw.text((5, 7), title, fill="black")
    return panel


def fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Letterbox an image to a fixed panel size."""
    canvas = Image.new("RGB", size, "white")
    copy = image.copy()
    copy.thumbnail(size)
    x = (size[0] - copy.width) // 2
    y = (size[1] - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def save_comparison(
    original: Image.Image,
    baseline_detections: list[dict[str, Any]],
    tight_image: Image.Image,
    tight_detections: list[dict[str, Any]],
    spatial_image: Image.Image | None,
    spatial_detections: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save ORIGINAL / SEGMENT_TIGHT / SEGMENT_SPATIAL side by side."""
    panel_size = (420, 520)

    panels = [
        fit_panel(
            draw_prediction(original, baseline_detections, "ORIGINAL"),
            panel_size,
        ),
        fit_panel(
            draw_prediction(tight_image, tight_detections, "SEGMENT_TIGHT"),
            panel_size,
        ),
    ]

    if spatial_image is None:
        spatial_image = Image.new("RGB", (420, 520), "white")

    panels.append(
        fit_panel(
            draw_prediction(
                spatial_image,
                spatial_detections,
                "SEGMENT_SPATIAL",
            ),
            panel_size,
        )
    )

    sheet = Image.new("RGB", (panel_size[0] * 3, panel_size[1]), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (index * panel_size[0], 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def _mean(values: list[float]) -> float | None:
    """Return arithmetic mean or None."""
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate results without pandas."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["target"]), str(row["condition"]))].append(row)

    output = []
    for (target, condition), group in sorted(groups.items()):
        detections = [
            row for row in group if str(row["detected"]).lower() == "yes"
        ]

        scores = [
            float(row["top_score"])
            for row in detections
            if str(row["top_score"]).strip()
        ]
        input_areas = [
            float(row["top_box_area_ratio_input"])
            for row in detections
            if str(row["top_box_area_ratio_input"]).strip()
        ]
        effective_areas = [
            float(row["top_box_area_ratio_original"])
            for row in detections
            if str(row["top_box_area_ratio_original"]).strip()
        ]
        input_regions = [
            float(row["input_region_area_ratio_original"]) for row in group
        ]

        output.append(
            {
                "target": target,
                "condition": condition,
                "n": len(group),
                "detection_rate": round(
                    100.0 * len(detections) / len(group),
                    1,
                ),
                "mean_top_score": (
                    round(_mean(scores), 4) if scores else ""
                ),
                "mean_top_box_area_ratio_input": (
                    round(_mean(input_areas), 4) if input_areas else ""
                ),
                "mean_input_region_area_ratio_original": round(
                    _mean(input_regions) or 0.0,
                    4,
                ),
                "mean_top_box_area_ratio_original": (
                    round(_mean(effective_areas), 4)
                    if effective_areas
                    else ""
                ),
            }
        )

    # Overall condition rows.
    for condition in ("ORIGINAL", "SEGMENT_TIGHT", "SEGMENT_SPATIAL"):
        group = [row for row in rows if row["condition"] == condition]
        if not group:
            continue
        detections = [
            row for row in group if str(row["detected"]).lower() == "yes"
        ]
        scores = [
            float(row["top_score"])
            for row in detections
            if str(row["top_score"]).strip()
        ]
        input_areas = [
            float(row["top_box_area_ratio_input"])
            for row in detections
            if str(row["top_box_area_ratio_input"]).strip()
        ]
        effective_areas = [
            float(row["top_box_area_ratio_original"])
            for row in detections
            if str(row["top_box_area_ratio_original"]).strip()
        ]
        input_regions = [
            float(row["input_region_area_ratio_original"]) for row in group
        ]
        output.append(
            {
                "target": "OVERALL",
                "condition": condition,
                "n": len(group),
                "detection_rate": round(
                    100.0 * len(detections) / len(group),
                    1,
                ),
                "mean_top_score": (
                    round(_mean(scores), 4) if scores else ""
                ),
                "mean_top_box_area_ratio_input": (
                    round(_mean(input_areas), 4) if input_areas else ""
                ),
                "mean_input_region_area_ratio_original": round(
                    _mean(input_regions) or 0.0,
                    4,
                ),
                "mean_top_box_area_ratio_original": (
                    round(_mean(effective_areas), 4)
                    if effective_areas
                    else ""
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV."""
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the diagnostic."""
    args = parse_args()

    selected_cases = load_csv(SELECTED_CASES_FILE)
    if args.max_cases is not None:
        selected_cases = selected_cases[: args.max_cases]

    baselines = load_original_baselines()

    missing_baselines = [
        case["benchmark_id"]
        for case in selected_cases
        if case["benchmark_id"] not in baselines
    ]
    if missing_baselines:
        raise ValueError(
            "Missing existing part-specific baseline rows for: "
            + ", ".join(missing_baselines)
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Cases: {len(selected_cases)}")
    print(f"Threshold: {args.threshold}")
    print("ORIGINAL baseline will be reused; only new crop conditions are inferred.")

    processor, model = load_model(device)
    all_rows: list[dict[str, Any]] = []

    for index, case in enumerate(selected_cases, start=1):
        benchmark_id = case["benchmark_id"]
        target = case["target"]
        prompt = PROMPTS[target]

        image_path = IMAGE_DIR / case["image_name"]
        anno_path = ANNO_DIR / f"{Path(case['image_name']).stem}.json"

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not anno_path.is_file():
            raise FileNotFoundError(f"Missing annotation: {anno_path}")

        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size

        with anno_path.open("r", encoding="utf-8") as file:
            annotation = json.load(file)

        print(f"[{index}/{len(selected_cases)}] {benchmark_id} ({target})")

        baseline = baselines[benchmark_id]
        baseline_result = baseline_row(
            case,
            baseline,
            image.size,
        )
        all_rows.append(baseline_result)

        # Parse baseline detections for visualization only.
        baseline_detections = json.loads(baseline["detections_json"] or "[]")

        tight_image, tight_box, tight_area_ratio = build_oracle_garment_crop(
            image=image,
            annotation=annotation,
            item_id=case["item_id"],
        )

        tight_detections, tight_seconds = infer(
            image=tight_image,
            prompt=prompt,
            threshold=args.threshold,
            processor=processor,
            model=model,
            device=device,
        )

        tight_result = detection_row(
            case=case,
            condition="SEGMENT_TIGHT",
            prompt=prompt,
            detections=tight_detections,
            inference_seconds=tight_seconds,
            input_size=tight_image.size,
            input_region_original=tight_box,
            original_size=image.size,
            source="new_inference",
        )
        tight_result["threshold"] = args.threshold
        all_rows.append(tight_result)

        # Evaluate all fixed spatial windows and retain the candidate with the
        # highest top detection score. If all miss, retain the first window for
        # visualization and record a missed result.
        spatial_candidates = []
        for window_index, window in enumerate(SPATIAL_WINDOWS[target], start=1):
            local_box = normalized_window_to_box(tight_image, window)
            local_image = tight_image.crop(local_box)
            detections, elapsed = infer(
                image=local_image,
                prompt=prompt,
                threshold=args.threshold,
                processor=processor,
                model=model,
                device=device,
            )

            if detections:
                top_score = max(item["score"] for item in detections)
            else:
                top_score = -1.0

            # Map this spatial region from tight-crop coordinates into the
            # original-image coordinates.
            tight_left, tight_top, _, _ = tight_box
            local_left, local_top, local_right, local_bottom = local_box
            region_original = (
                tight_left + local_left,
                tight_top + local_top,
                tight_left + local_right,
                tight_top + local_bottom,
            )

            spatial_candidates.append(
                {
                    "window_index": window_index,
                    "window": window,
                    "image": local_image,
                    "detections": detections,
                    "elapsed": elapsed,
                    "top_score": top_score,
                    "region_original": region_original,
                }
            )

        best_spatial = max(
            spatial_candidates,
            key=lambda item: item["top_score"],
        )

        spatial_result = detection_row(
            case=case,
            condition="SEGMENT_SPATIAL",
            prompt=prompt,
            detections=best_spatial["detections"],
            inference_seconds=best_spatial["elapsed"],
            input_size=best_spatial["image"].size,
            input_region_original=best_spatial["region_original"],
            original_size=image.size,
            source="new_inference",
            selected_window=(
                f"window_{best_spatial['window_index']}:"
                f"{best_spatial['window']}"
            ),
        )
        spatial_result["threshold"] = args.threshold
        all_rows.append(spatial_result)

        if not args.skip_visuals:
            save_comparison(
                original=image,
                baseline_detections=baseline_detections,
                tight_image=tight_image,
                tight_detections=tight_detections,
                spatial_image=best_spatial["image"],
                spatial_detections=best_spatial["detections"],
                output_path=(
                    OUTPUT_DIR
                    / target
                    / f"{benchmark_id}_segmentation_first.jpg"
                ),
            )

    case_results_path = REPORT_DIR / "case_results.csv"
    write_csv(case_results_path, all_rows)

    summary_rows = summarize(all_rows)
    summary_path = REPORT_DIR / "summary.csv"
    write_csv(summary_path, summary_rows)

    selected_copy = REPORT_DIR / "selected_cases.csv"
    write_csv(selected_copy, selected_cases)

    print("\nFinished.")
    print(f"Case results: {case_results_path}")
    print(f"Summary: {summary_path}")
    print(f"Visuals: {OUTPUT_DIR}")
    print(
        "\nImportant: detection rate is NOT localization accuracy. "
        "Review comparison images using correct/coarse/wrong/missed."
    )


if __name__ == "__main__":
    main()
