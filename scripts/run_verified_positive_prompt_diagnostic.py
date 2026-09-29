"""Run a controlled prompt diagnostic on verified-positive cases.

This experiment isolates the language side of Grounding DINO. It keeps the
image condition and threshold fixed and compares three prompt levels:

1. generic
2. garment_context
3. part_specific

The experiment is diagnostic only. It samples a fixed small subset from the
verified-positive benchmark and should not replace the frozen 88-case result.

Outputs:
- reports/prompt_diagnostic_2026_09_14/selected_cases.csv
- reports/prompt_diagnostic_2026_09_14/case_results.csv
- reports/prompt_diagnostic_2026_09_14/summary.csv
- outputs/prompt_diagnostic_2026_09_14/<target>/<benchmark_id>.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_THRESHOLD = 0.30
DEFAULT_PER_TARGET = 5
DEFAULT_SEED = 20260914

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "fashion_data" / "raw" / "train" / "train"
IMAGE_DIR = DATASET_ROOT / "image"

BENCHMARK_FILE = (
    PROJECT_ROOT
    / "reports"
    / "verified_positive_benchmark"
    / "verified_positive_benchmark.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "prompt_diagnostic_2026_09_14"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "prompt_diagnostic_2026_09_14"

TARGETS = ("sleeve", "collar", "button", "zipper")

PROMPT_VARIANTS = {
    "sleeve": {
        "generic": "sleeve",
        "garment_context": "clothing sleeve",
        "part_specific": "the sleeve area of the garment",
    },
    "collar": {
        "generic": "collar",
        "garment_context": "clothing collar",
        "part_specific": "the collar around the neck opening of the garment",
    },
    "button": {
        "generic": "button",
        "garment_context": "clothing button",
        "part_specific": "the small button detail on the garment",
    },
    "zipper": {
        "generic": "zipper",
        "garment_context": "clothing zipper",
        "part_specific": "the zipper closure detail on the garment",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a small verified-positive prompt diagnostic."
    )
    parser.add_argument(
        "--per-target",
        type=int,
        default=DEFAULT_PER_TARGET,
        help=f"Cases sampled per target (default: {DEFAULT_PER_TARGET}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Sampling seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Detection threshold (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip annotated comparison images.",
    )
    args = parser.parse_args()

    if args.per_target <= 0:
        raise ValueError("--per-target must be greater than zero.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    return args


def load_benchmark() -> list[dict[str, str]]:
    """Load usable verified-positive benchmark rows."""
    with BENCHMARK_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    selected = []
    for row in rows:
        if row.get("verified_positive", "").strip().lower() != "yes":
            continue
        if row.get("usable_for_evaluation", "").strip().lower() != "yes":
            continue
        if row.get("target", "").strip().lower() not in TARGETS:
            continue
        selected.append(row)
    return selected


def sample_cases(
    rows: list[dict[str, str]],
    per_target: int,
    seed: int,
) -> list[dict[str, str]]:
    """Sample a deterministic diagnostic subset, stratified by target."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["target"].strip().lower()].append(row)

    rng = random.Random(seed)
    selected = []
    for target in TARGETS:
        pool = sorted(grouped[target], key=lambda row: row["benchmark_id"])
        if len(pool) < per_target:
            raise ValueError(
                f"Target {target} has only {len(pool)} usable cases, "
                f"but --per-target={per_target}."
            )
        selected.extend(rng.sample(pool, per_target))

    return sorted(selected, key=lambda row: row["benchmark_id"])


def load_model(device: torch.device) -> tuple[Any, Any]:
    """Load Grounding DINO and its processor."""
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    return processor, model


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

    detections = []
    for score, label, box in zip(
        result["scores"],
        result["labels"],
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


def box_area_ratio(box: list[float], width: int, height: int) -> float:
    """Return predicted box area divided by image area."""
    x_min, y_min, x_max, y_max = box
    box_area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    image_area = max(1, width * height)
    return box_area / image_area


def draw_panel(
    image: Image.Image,
    detections: list[dict[str, Any]],
    prompt_name: str,
    prompt: str,
) -> Image.Image:
    """Draw detections for one prompt variant."""
    panel = image.copy()
    draw = ImageDraw.Draw(panel)

    for detection in detections:
        box = detection["box"]
        draw.rectangle(box, outline="red", width=4)

    caption = f"{prompt_name}: {prompt}"
    draw.rectangle((0, 0, panel.width, 30), fill="white")
    draw.text((5, 7), caption, fill="black")
    return panel


def save_comparison(
    image: Image.Image,
    per_variant: dict[str, tuple[str, list[dict[str, Any]]]],
    output_path: Path,
) -> None:
    """Save a horizontal comparison of all prompt variants."""
    panels = []
    for variant_name in ("generic", "garment_context", "part_specific"):
        prompt, detections = per_variant[variant_name]
        panels.append(draw_panel(image, detections, variant_name, prompt))

    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    resized = []
    for panel in panels:
        if panel.size != (width, height):
            panel = panel.resize((width, height))
        resized.append(panel)

    sheet = Image.new("RGB", (width * len(resized), height), "white")
    for index, panel in enumerate(resized):
        sheet.paste(panel, (index * width, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def main() -> None:
    """Run the prompt diagnostic and write structured results."""
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_benchmark()
    cases = sample_cases(rows, args.per_target, args.seed)

    selected_file = REPORT_DIR / "selected_cases.csv"
    with selected_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=cases[0].keys())
        writer.writeheader()
        writer.writerows(cases)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Selected cases: {len(cases)}")
    print(f"Threshold: {args.threshold}")

    processor, model = load_model(device)
    result_rows: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        target = case["target"].strip().lower()
        benchmark_id = case["benchmark_id"]
        image_path = IMAGE_DIR / case["image_name"]

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        comparison_data = {}

        print(f"[{index}/{len(cases)}] {benchmark_id} ({target})")

        for variant_name, prompt in PROMPT_VARIANTS[target].items():
            detections, elapsed = infer(
                image=image,
                prompt=prompt,
                threshold=args.threshold,
                processor=processor,
                model=model,
                device=device,
            )

            top_detection = (
                max(detections, key=lambda item: item["score"])
                if detections
                else None
            )

            top_score = top_detection["score"] if top_detection else None
            top_area = (
                box_area_ratio(top_detection["box"], width, height)
                if top_detection
                else None
            )

            result_rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "target": target,
                    "image_name": case["image_name"],
                    "category_name": case["category_name"],
                    "prompt_variant": variant_name,
                    "prompt": prompt,
                    "threshold": args.threshold,
                    "detected": "yes" if detections else "no",
                    "num_detections": len(detections),
                    "top_score": top_score,
                    "top_box_area_ratio": top_area,
                    "inference_seconds": elapsed,
                    "detections_json": json.dumps(detections),
                }
            )
            comparison_data[variant_name] = (prompt, detections)

        if not args.skip_visuals:
            save_comparison(
                image=image,
                per_variant=comparison_data,
                output_path=(
                    OUTPUT_DIR
                    / target
                    / f"{benchmark_id}_prompt_comparison.jpg"
                ),
            )

    result_file = REPORT_DIR / "case_results.csv"
    fieldnames = list(result_rows[0].keys())
    with result_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    frame = __import__("pandas").DataFrame(result_rows)
    summary = (
        frame.groupby(["target", "prompt_variant", "prompt"], as_index=False)
        .agg(
            n=("benchmark_id", "count"),
            detection_rate=("detected", lambda s: (s == "yes").mean()),
            mean_top_score=("top_score", "mean"),
            mean_top_box_area_ratio=("top_box_area_ratio", "mean"),
            mean_inference_seconds=("inference_seconds", "mean"),
        )
    )
    summary["detection_rate"] = (summary["detection_rate"] * 100).round(1)
    summary["mean_top_score"] = summary["mean_top_score"].round(4)
    summary["mean_top_box_area_ratio"] = (
        summary["mean_top_box_area_ratio"].round(4)
    )
    summary["mean_inference_seconds"] = (
        summary["mean_inference_seconds"].round(4)
    )
    summary.to_csv(
        REPORT_DIR / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFinished.")
    print(f"Results: {result_file}")
    print(f"Summary: {REPORT_DIR / 'summary.csv'}")
    print(f"Visuals: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
