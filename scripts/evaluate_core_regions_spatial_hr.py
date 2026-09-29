"""Compare BASELINE vs target-specific SPATIAL_HR on 15 core-region cases.

This experiment reuses the frozen 40-case PRD coverage baseline and evaluates
only the 15 core cases:
    collar (5), cuff (5), hem (5)

BASELINE:
    Reused from reports/prd_region_coverage/formal_case_results.csv.
    No baseline inference is repeated.

SPATIAL_HR:
    1. Apply a fixed region-specific spatial prior to the existing garment crop.
    2. Run the SAME prompt and SAME threshold.
    3. Increase Grounding DINO image-processor resolution so the local region
       receives more effective visual detail.

The main question is whether coarse garment-level boxes can become tighter,
region-level boxes without simply increasing non-empty predictions.

Expected existing files:
    reports/prd_region_coverage/formal_case_results.csv
    reports/prd_region_coverage/prd_40case_formal_manual_audit_final.csv

Expected existing images:
    outputs/prd_region_coverage/<region>/crops/*.jpg

Outputs:
    reports/prd_region_coverage/core_region_spatial_hr_15case/
        case_results.csv
        summary.csv
        spatial_hr_manual_audit_template.csv
        experiment_notes.md

    outputs/core_region_spatial_hr_15case/
        collar/
        cuff/
        hem/

Important:
    Non-empty prediction rate is NOT localization accuracy.
    After inference, manually review SPATIAL_HR as:
        correct / coarse / wrong / missed
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
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_THRESHOLD = 0.30
DEFAULT_HR_SHORT_EDGE = 1000
DEFAULT_HR_LONG_EDGE = 1600

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_ROOT = PROJECT_ROOT / "reports" / "prd_region_coverage"
BASELINE_RESULTS_FILE = REPORT_ROOT / "formal_case_results.csv"
BASELINE_AUDIT_FILE = REPORT_ROOT / "prd_40case_formal_manual_audit_final.csv"

EXPERIMENT_REPORT_DIR = (
    REPORT_ROOT / "core_region_spatial_hr_15case"
)
EXPERIMENT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "core_region_spatial_hr_15case"
)

CORE_REGIONS = ("collar", "cuff", "hem")

# Fixed before this experiment. These are broad spatial priors, not tuned
# per image.
SPATIAL_WINDOWS = {
    "collar": [
        ("upper_center", 0.15, 0.00, 0.85, 0.45),
    ],
    "cuff": [
        ("left_side", 0.00, 0.00, 0.46, 1.00),
        ("right_side", 0.54, 0.00, 1.00, 1.00),
    ],
    "hem": [
        ("lower_band", 0.00, 0.55, 1.00, 1.00),
    ],
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen BASELINE against target-specific "
            "SPATIAL_HR on collar/cuff/hem."
        )
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Detection threshold (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--hr-short-edge",
        type=int,
        default=DEFAULT_HR_SHORT_EDGE,
        help=(
            "SPATIAL_HR processor shortest edge "
            f"(default: {DEFAULT_HR_SHORT_EDGE})."
        ),
    )
    parser.add_argument(
        "--hr-long-edge",
        type=int,
        default=DEFAULT_HR_LONG_EDGE,
        help=(
            "SPATIAL_HR processor longest edge "
            f"(default: {DEFAULT_HR_LONG_EDGE})."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of cases for a smoke test.",
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip per-case comparison images and contact sheets.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.hr_short_edge <= 0 or args.hr_long_edge <= 0:
        raise ValueError("HR processor sizes must be positive.")
    if args.hr_long_edge < args.hr_short_edge:
        raise ValueError("--hr-long-edge must be >= --hr-short-edge.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero.")

    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required CSV: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV."""
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def normalize_project_path(raw_path: str) -> Path:
    """Convert a stored Windows/portable relative path to a project path."""
    normalized = raw_path.replace("\\", "/")
    return PROJECT_ROOT / Path(normalized)


def load_baseline_cases() -> list[dict[str, str]]:
    """Load exactly the 15 collar/cuff/hem baseline cases."""
    rows = read_csv(BASELINE_RESULTS_FILE)

    selected = [
        row
        for row in rows
        if row["region"].strip().lower() in CORE_REGIONS
    ]

    counts = defaultdict(int)
    for row in selected:
        counts[row["region"].strip().lower()] += 1

    expected = {"collar": 5, "cuff": 5, "hem": 5}
    if dict(counts) != expected:
        raise ValueError(
            "Expected exactly 5 baseline cases for each core region. "
            f"Found: {dict(counts)}"
        )

    return selected


def load_baseline_audit() -> dict[str, dict[str, str]]:
    """Load frozen baseline manual labels keyed by candidate_id."""
    rows = read_csv(BASELINE_AUDIT_FILE)
    return {
        row["candidate_id"].strip(): row
        for row in rows
    }


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_model(
    device: torch.device,
) -> tuple[Any, Any]:
    """Load Grounding DINO and processor."""
    print(f"Loading model: {MODEL_ID}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID
    )
    model.to(device)
    model.eval()

    return processor, model


def normalized_window_to_box(
    image: Image.Image,
    window: tuple[str, float, float, float, float],
) -> tuple[str, tuple[int, int, int, int]]:
    """Convert one normalized spatial window to pixel coordinates."""
    name, x1, y1, x2, y2 = window
    width, height = image.size

    left = max(
        0,
        min(width - 1, int(round(x1 * width))),
    )
    top = max(
        0,
        min(height - 1, int(round(y1 * height))),
    )
    right = max(
        left + 1,
        min(width, int(round(x2 * width))),
    )
    bottom = max(
        top + 1,
        min(height, int(round(y2 * height))),
    )

    return name, (left, top, right, bottom)


def infer_spatial_hr(
    image: Image.Image,
    prompt: str,
    threshold: float,
    processor: Any,
    model: Any,
    device: torch.device,
    hr_short_edge: int,
    hr_long_edge: int,
) -> tuple[list[dict[str, Any]], float]:
    """Run Grounding DINO with a higher processor resolution."""
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
        size={
            "shortest_edge": hr_short_edge,
            "longest_edge": hr_long_edge,
        },
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

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
    labels = (
        text_labels
        if text_labels is not None
        else result.get("labels", [])
    )

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
                "box": [
                    float(value)
                    for value in box.tolist()
                ],
            }
        )

    detections.sort(
        key=lambda item: item["score"],
        reverse=True,
    )
    return detections, elapsed


def box_area(box: list[float]) -> float:
    """Return non-negative bbox area."""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def area_ratio(
    box: list[float],
    width: int,
    height: int,
) -> float:
    """Return bbox area divided by image area."""
    return box_area(box) / max(
        1.0,
        float(width * height),
    )


def map_box_to_parent(
    box: list[float],
    window_box: tuple[int, int, int, int],
) -> list[float]:
    """Map a local-window bbox into parent garment-crop coordinates."""
    left, top, _, _ = window_box
    x1, y1, x2, y2 = box
    return [
        x1 + left,
        y1 + top,
        x2 + left,
        y2 + top,
    ]


def run_best_window(
    parent_image: Image.Image,
    region: str,
    prompt: str,
    threshold: float,
    processor: Any,
    model: Any,
    device: torch.device,
    hr_short_edge: int,
    hr_long_edge: int,
) -> dict[str, Any]:
    """Run all fixed windows and return the highest-scoring candidate."""
    candidates = []

    for window in SPATIAL_WINDOWS[region]:
        window_name, window_box = normalized_window_to_box(
            parent_image,
            window,
        )
        local_image = parent_image.crop(window_box)

        detections, elapsed = infer_spatial_hr(
            image=local_image,
            prompt=prompt,
            threshold=threshold,
            processor=processor,
            model=model,
            device=device,
            hr_short_edge=hr_short_edge,
            hr_long_edge=hr_long_edge,
        )

        top_score = (
            detections[0]["score"]
            if detections
            else -1.0
        )

        candidates.append(
            {
                "window_name": window_name,
                "window_box": window_box,
                "local_image": local_image,
                "detections": detections,
                "elapsed": elapsed,
                "top_score": top_score,
            }
        )

    return max(
        candidates,
        key=lambda item: item["top_score"],
    )


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a readable font with a safe fallback."""
    try:
        return ImageFont.truetype(
            "arial.ttf",
            size=size,
        )
    except OSError:
        return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    detections: list[dict[str, Any]],
    title: str,
) -> Image.Image:
    """Draw detections and title."""
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = load_font(14)

    for detection in detections:
        draw.rectangle(
            detection["box"],
            outline="red",
            width=4,
        )

    draw.rectangle(
        (0, 0, output.width, 28),
        fill="white",
    )
    draw.text(
        (5, 6),
        title,
        fill="black",
        font=font,
    )
    return output


def fit_panel(
    image: Image.Image,
    size: tuple[int, int],
) -> Image.Image:
    """Letterbox an image into a fixed-size panel."""
    canvas = Image.new("RGB", size, "white")
    copy = image.copy()
    copy.thumbnail(size)

    x = (size[0] - copy.width) // 2
    y = (size[1] - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def save_case_comparison(
    parent_image: Image.Image,
    baseline_detections: list[dict[str, Any]],
    local_image: Image.Image,
    spatial_detections: list[dict[str, Any]],
    mapped_detections: list[dict[str, Any]],
    candidate_id: str,
    output_path: Path,
) -> None:
    """Save BASELINE / SPATIAL_HR / mapped-on-parent comparison."""
    panel_size = (420, 520)

    baseline_panel = fit_panel(
        draw_detections(
            parent_image,
            baseline_detections,
            f"{candidate_id} | BASELINE",
        ),
        panel_size,
    )

    local_panel = fit_panel(
        draw_detections(
            local_image,
            spatial_detections,
            f"{candidate_id} | SPATIAL_HR local",
        ),
        panel_size,
    )

    mapped_panel = fit_panel(
        draw_detections(
            parent_image,
            mapped_detections,
            f"{candidate_id} | SPATIAL_HR mapped",
        ),
        panel_size,
    )

    sheet = Image.new(
        "RGB",
        (panel_size[0] * 3, panel_size[1]),
        "white",
    )
    sheet.paste(baseline_panel, (0, 0))
    sheet.paste(local_panel, (panel_size[0], 0))
    sheet.paste(mapped_panel, (panel_size[0] * 2, 0))

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    sheet.save(output_path, quality=92)


def make_region_contact_sheet(
    region: str,
    paths: list[Path],
) -> None:
    """Combine five case-comparison images into one region sheet."""
    if not paths:
        return

    panel_width = 1000
    panel_height = 420

    prepared = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
        prepared.append(
            fit_panel(
                image,
                (panel_width, panel_height),
            )
        )

    sheet = Image.new(
        "RGB",
        (panel_width, panel_height * len(prepared)),
        "white",
    )

    for index, panel in enumerate(prepared):
        sheet.paste(
            panel,
            (0, index * panel_height),
        )

    output_path = (
        EXPERIMENT_OUTPUT_DIR
        / region
        / f"{region}_baseline_vs_spatial_hr_contact_sheet.jpg"
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    sheet.save(output_path, quality=92)


def _mean(values: list[float]) -> float | None:
    """Return arithmetic mean or None."""
    return (
        sum(values) / len(values)
        if values
        else None
    )


def summarize(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize BASELINE and SPATIAL_HR automatic results."""
    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        grouped[
            (str(row["region"]), str(row["condition"]))
        ].append(row)

    output = []

    def aggregate(
        region: str,
        condition: str,
        group: list[dict[str, Any]],
    ) -> dict[str, Any]:
        detected = [
            row
            for row in group
            if str(row["detected"]).lower() == "yes"
        ]

        scores = [
            float(row["top_score"])
            for row in detected
            if str(row["top_score"]).strip()
        ]
        parent_areas = [
            float(row["top_box_area_ratio_parent"])
            for row in detected
            if str(
                row["top_box_area_ratio_parent"]
            ).strip()
        ]
        times = [
            float(row["inference_seconds"])
            for row in group
            if str(row["inference_seconds"]).strip()
        ]

        return {
            "region": region,
            "condition": condition,
            "n": len(group),
            "non_empty_prediction_rate_pct": round(
                100.0 * len(detected) / len(group),
                1,
            ),
            "mean_top_score": (
                round(_mean(scores), 4)
                if scores
                else ""
            ),
            "mean_top_box_area_ratio_parent": (
                round(_mean(parent_areas), 4)
                if parent_areas
                else ""
            ),
            "mean_inference_seconds": (
                round(_mean(times), 4)
                if times
                else ""
            ),
        }

    for region in CORE_REGIONS:
        for condition in ("BASELINE", "SPATIAL_HR"):
            group = grouped[(region, condition)]
            if group:
                output.append(
                    aggregate(
                        region,
                        condition,
                        group,
                    )
                )

    for condition in ("BASELINE", "SPATIAL_HR"):
        group = [
            row
            for row in rows
            if row["condition"] == condition
        ]
        if group:
            output.append(
                aggregate(
                    "OVERALL",
                    condition,
                    group,
                )
            )

    return output


def write_notes(
    threshold: float,
    hr_short_edge: int,
    hr_long_edge: int,
) -> None:
    """Write a concise experiment protocol."""
    text = f"""# Core-region SPATIAL_HR diagnostic

## Purpose

Test whether target-specific spatial priors plus a higher Grounding DINO
processor resolution can convert coarse garment-level predictions into tighter
region-level predictions for the three priority PRD regions:

- collar
- cuff
- hem

## Frozen variables

- Baseline cases: the same 15 cases from the 40-case PRD coverage pilot.
- Prompt: reused exactly from the frozen baseline.
- Threshold: {threshold}.
- Model: `{MODEL_ID}`.
- Baseline inference: reused from `formal_case_results.csv`.

## SPATIAL_HR settings

- collar: upper-center window `(0.15, 0.00, 0.85, 0.45)`
- cuff: two side windows; keep the window with the highest top score
  - left `(0.00, 0.00, 0.46, 1.00)`
  - right `(0.54, 0.00, 1.00, 1.00)`
- hem: lower band `(0.00, 0.55, 1.00, 1.00)`
- processor shortest edge: {hr_short_edge}
- processor longest edge: {hr_long_edge}

## Evaluation

Automatic:
- non-empty prediction rate
- top confidence
- effective top-box area ratio relative to the original garment crop
- inference time

Manual:
- correct
- coarse
- wrong
- missed

The desired improvement is mainly `coarse -> correct`, not merely
`missed -> detected`.
"""

    path = EXPERIMENT_REPORT_DIR / "experiment_notes.md"
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    """Run the 15-case BASELINE vs SPATIAL_HR experiment."""
    args = parse_args()

    cases = load_baseline_cases()
    baseline_audit = load_baseline_audit()

    if args.limit is not None:
        cases = cases[: args.limit]

    EXPERIMENT_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    EXPERIMENT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print(f"Using device: {device}")
    print(f"Cases: {len(cases)}")
    print(f"Threshold: {args.threshold}")
    print(
        "SPATIAL_HR processor size: "
        f"{args.hr_short_edge}/{args.hr_long_edge}"
    )
    print(
        "BASELINE inference is reused; "
        "only SPATIAL_HR is newly inferred."
    )

    processor, model = load_model(device)

    all_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    visuals_by_region: dict[str, list[Path]] = defaultdict(list)

    for index, baseline in enumerate(cases, start=1):
        candidate_id = baseline["candidate_id"].strip()
        region = baseline["region"].strip().lower()
        prompt = baseline["prompt"]

        image_path = normalize_project_path(
            baseline["candidate_crop_path"]
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Candidate crop not found: {image_path}"
            )

        parent_image = Image.open(image_path).convert("RGB")

        print(
            f"[{index}/{len(cases)}] "
            f"{candidate_id} ({region})"
        )

        baseline_detections = json.loads(
            baseline["detections_json"] or "[]"
        )

        baseline_top_area = (
            baseline["top_box_area_ratio"]
            if baseline["top_box_area_ratio"].strip()
            else ""
        )

        all_rows.append(
            {
                "candidate_id": candidate_id,
                "region": region,
                "condition": "BASELINE",
                "prompt": prompt,
                "threshold": baseline["threshold"],
                "selected_window": "",
                "detected": baseline["detected"],
                "num_detections": baseline["num_detections"],
                "top_score": baseline["top_score"],
                "window_area_ratio_parent": 1.0,
                "top_box_area_ratio_input": baseline_top_area,
                "top_box_area_ratio_parent": baseline_top_area,
                "inference_seconds": baseline["inference_seconds"],
                "result_source": "reused_formal_baseline",
                "detections_json": baseline["detections_json"],
            }
        )

        best = run_best_window(
            parent_image=parent_image,
            region=region,
            prompt=prompt,
            threshold=args.threshold,
            processor=processor,
            model=model,
            device=device,
            hr_short_edge=args.hr_short_edge,
            hr_long_edge=args.hr_long_edge,
        )

        local_image = best["local_image"]
        detections = best["detections"]
        window_box = best["window_box"]

        parent_width, parent_height = parent_image.size
        left, top, right, bottom = window_box

        window_area_ratio = (
            (right - left) * (bottom - top)
            / max(1.0, float(parent_width * parent_height))
        )

        if detections:
            top_detection = detections[0]
            top_area_input = area_ratio(
                top_detection["box"],
                local_image.width,
                local_image.height,
            )
            top_area_parent = (
                top_area_input * window_area_ratio
            )
            top_score: float | str = top_detection["score"]
        else:
            top_area_input = ""
            top_area_parent = ""
            top_score = ""

        all_rows.append(
            {
                "candidate_id": candidate_id,
                "region": region,
                "condition": "SPATIAL_HR",
                "prompt": prompt,
                "threshold": args.threshold,
                "selected_window": best["window_name"],
                "detected": "yes" if detections else "no",
                "num_detections": len(detections),
                "top_score": top_score,
                "window_area_ratio_parent": round(
                    window_area_ratio,
                    6,
                ),
                "top_box_area_ratio_input": (
                    round(top_area_input, 6)
                    if top_area_input != ""
                    else ""
                ),
                "top_box_area_ratio_parent": (
                    round(top_area_parent, 6)
                    if top_area_parent != ""
                    else ""
                ),
                "inference_seconds": best["elapsed"],
                "result_source": "new_spatial_hr_inference",
                "detections_json": json.dumps(detections),
            }
        )

        baseline_quality = ""
        if candidate_id in baseline_audit:
            baseline_quality = baseline_audit[
                candidate_id
            ].get("localization_quality", "")

        area_reduction = ""
        if (
            baseline_top_area != ""
            and top_area_parent != ""
        ):
            area_reduction = round(
                float(baseline_top_area)
                - float(top_area_parent),
                6,
            )

        audit_rows.append(
            {
                "candidate_id": candidate_id,
                "region": region,
                "baseline_quality": baseline_quality,
                "baseline_top_box_area_ratio": baseline_top_area,
                "spatial_hr_detected": (
                    "yes" if detections else "no"
                ),
                "spatial_hr_top_score": top_score,
                "selected_window": best["window_name"],
                "spatial_hr_top_box_area_ratio_parent": (
                    round(top_area_parent, 6)
                    if top_area_parent != ""
                    else ""
                ),
                "area_ratio_reduction_vs_baseline": area_reduction,
                "spatial_hr_localization_quality": "",
                "review_note": "",
            }
        )

        if not args.skip_visuals:
            mapped_detections = []
            for detection in detections:
                mapped = dict(detection)
                mapped["box"] = map_box_to_parent(
                    detection["box"],
                    window_box,
                )
                mapped_detections.append(mapped)

            comparison_path = (
                EXPERIMENT_OUTPUT_DIR
                / region
                / f"{candidate_id}_baseline_vs_spatial_hr.jpg"
            )

            save_case_comparison(
                parent_image=parent_image,
                baseline_detections=baseline_detections,
                local_image=local_image,
                spatial_detections=detections,
                mapped_detections=mapped_detections,
                candidate_id=candidate_id,
                output_path=comparison_path,
            )

            visuals_by_region[region].append(
                comparison_path
            )

    write_csv(
        EXPERIMENT_REPORT_DIR / "case_results.csv",
        all_rows,
    )
    write_csv(
        EXPERIMENT_REPORT_DIR / "summary.csv",
        summarize(all_rows),
    )
    write_csv(
        EXPERIMENT_REPORT_DIR
        / "spatial_hr_manual_audit_template.csv",
        audit_rows,
    )

    write_notes(
        threshold=args.threshold,
        hr_short_edge=args.hr_short_edge,
        hr_long_edge=args.hr_long_edge,
    )

    if not args.skip_visuals:
        for region, paths in visuals_by_region.items():
            make_region_contact_sheet(
                region,
                paths,
            )

    print("\nFinished.")
    print(
        "Case results: "
        f"{EXPERIMENT_REPORT_DIR / 'case_results.csv'}"
    )
    print(
        "Summary: "
        f"{EXPERIMENT_REPORT_DIR / 'summary.csv'}"
    )
    print(
        "Manual audit template: "
        f"{EXPERIMENT_REPORT_DIR / 'spatial_hr_manual_audit_template.csv'}"
    )
    print(
        "Visuals: "
        f"{EXPERIMENT_OUTPUT_DIR}"
    )
    print(
        "\nReview goal: look for BASELINE coarse -> "
        "SPATIAL_HR correct improvements."
    )


if __name__ == "__main__":
    main()
