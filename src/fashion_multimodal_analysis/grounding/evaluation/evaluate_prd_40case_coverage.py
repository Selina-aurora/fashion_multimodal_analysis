"""3.1.2 文本引导区域定位：检测覆盖率、粗框可用率和人工定位准确率分开解释。

Evaluate the 40-case PRD 3.1.2 region-coverage pilot with Grounding DINO.

Expected existing files:
- reports/prd_region_coverage/<region>_candidates.csv
- outputs/prd_region_coverage/<region>/crops/...
- reports/prd_region_coverage/prd_40case_verified_positive_selection.csv

The script joins the manual 40-case selection with the existing candidate
metadata, runs one explicit region-specific prompt per case, writes structured
results, per-region summaries, and visualizations.

Important:
- Non-empty prediction rate is NOT localization accuracy.
- After inference, visually audit every case as:
  correct / coarse / wrong / missed.
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

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_THRESHOLD = 0.30

PROJECT_ROOT = get_project_root()
REPORT_ROOT = PROJECT_ROOT / "reports" / "prd_region_coverage"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "prd_region_coverage_formal_eval"

SELECTION_FILE = REPORT_ROOT / "prd_40case_verified_positive_selection.csv"
CASE_RESULTS_FILE = REPORT_ROOT / "formal_case_results.csv"
SUMMARY_FILE = REPORT_ROOT / "formal_summary.csv"
MANUAL_AUDIT_FILE = REPORT_ROOT / "formal_manual_audit_template.csv"

REGIONS = (
    "collar",
    "cuff",
    "hem",
    "pocket",
    "shoulder",
    "waist",
    "pattern",
    "decoration",
)

PROMPTS = {
    "collar": "the collar or neckline area of the garment",
    "cuff": "the cuff at the end of the sleeve",
    "hem": "the bottom hem edge of the garment",
    "pocket": "the pocket on the garment",
    "shoulder": "the shoulder area of the garment",
    "waist": "the waist area of the garment",
    "pattern": "the visible pattern or print on the garment",
    "decoration": "the decorative detail or embellishment on the garment",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        已通过参数合法性检查的命令行配置。

    Raises:
        ValueError: --limit must be greater than zero.
    """
    parser = argparse.ArgumentParser(
        description="Run the 40-case PRD region-coverage evaluation."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Detection threshold (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional case limit for a quick smoke test.",
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip annotated-image generation.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
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
        raise FileNotFoundError(f"Missing CSV: {path}")
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


def load_candidate_metadata() -> dict[str, dict[str, str]]:
    """Load candidate rows for all PRD regions, keyed by candidate_id.

    Returns:
        当前读取操作得到的记录、图像或模型对象；保持原始格式约定。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    metadata: dict[str, dict[str, str]] = {}

    for region in REGIONS:
        path = REPORT_ROOT / f"{region}_candidates.csv"
        rows = read_csv(path)

        for row in rows:
            candidate_id = row["candidate_id"].strip()
            if candidate_id in metadata:
                raise ValueError(f"Duplicate candidate_id: {candidate_id}")
            metadata[candidate_id] = row

    return metadata


def build_cases() -> list[dict[str, str]]:
    """Join the 40-case selection with candidate metadata.

    Returns:
        返回 cases，由函数体中同名变量的计算/收集过程得到。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
        KeyError: 所需字段不存在。
    """
    selection = read_csv(SELECTION_FILE)
    metadata = load_candidate_metadata()

    cases = []
    for selected in selection:
        candidate_id = selected["candidate_id"].strip()
        region = selected["region"].strip().lower()

        if region not in REGIONS:
            raise ValueError(f"Unknown region in selection: {region}")
        if candidate_id not in metadata:
            raise KeyError(
                f"{candidate_id} is missing from candidate CSVs. "
                + "Re-run build_prd_region_candidate_sheets.py with "
                + "--max-annotations 500 --per-region 24 using the same seed."
            )

        candidate = metadata[candidate_id]
        if candidate["region"].strip().lower() != region:
            raise ValueError(
                f"Region mismatch for {candidate_id}: "
                + f"{candidate['region']} vs {region}"
            )

        case = dict(candidate)
        case.update(
            {
                "target_present": selected["target_present"],
                "visibility": selected["visibility"],
                "usable_for_evaluation": selected["usable_for_evaluation"],
                "selection_note": selected["review_note"],
            }
        )
        cases.append(case)

    if len(cases) != 40:
        raise ValueError(f"Expected 40 cases, got {len(cases)}.")

    return cases


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU.

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device: torch.device) -> tuple[Any, Any]:
    """Load Grounding DINO and processor.

    Args:
        device: 当前计算设备，与输入张量和模型设备保持一致。

    Returns:
        按顺序返回 processor, model 等结果。
    """
    print(f"Loading model: {MODEL_ID}")
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
    """Run Grounding DINO for one image and one region prompt.

    Args:
        image: 本步骤处理的图像对象。
        prompt: 提示文本。
        threshold: 当前判定规则使用的阈值。
        processor: 与模型配套的输入处理器。
        model: 已构建的模型对象，由调用方负责选择权重。
        device: 当前计算设备，与输入张量和模型设备保持一致。

    Returns:
        按顺序返回 detections, elapsed 等结果。
    """
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

    detections.sort(key=lambda item: item["score"], reverse=True)
    return detections, elapsed


def box_area_ratio(box: list[float], width: int, height: int) -> float:
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


def draw_result(
    image: Image.Image,
    candidate_id: str,
    region: str,
    prompt: str,
    detections: list[dict[str, Any]],
) -> Image.Image:
    """Draw all predictions on an image.

    Args:
        image: 本步骤处理的图像对象。
        candidate_id: candidate ID。
        region: 区域。
        prompt: 提示文本。
        detections: detections。

    Returns:
        返回 output，由函数体中同名变量的计算/收集过程得到。
    """
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = load_font(15)

    for detection in detections:
        box = detection["box"]
        score = detection["score"]
        draw.rectangle(box, outline="red", width=4)

        x1, y1, _, _ = box
        label = f"{region} {score:.3f}"
        label_y = max(34, int(y1))
        draw.rectangle(
            (
                int(x1),
                label_y - 20,
                int(x1) + 150,
                label_y,
            ),
            fill="white",
        )
        draw.text(
            (int(x1) + 3, label_y - 18),
            label,
            fill="red",
            font=font,
        )

    banner = f"{candidate_id} | {region} | {prompt}"
    draw.rectangle((0, 0, output.width, 30), fill="white")
    draw.text((5, 7), banner, fill="black", font=font)
    return output


def make_contact_sheets(
    region: str,
    visual_paths: list[Path],
) -> None:
    """Create one compact result contact sheet per region.

    Args:
        region: 区域。
        visual_paths: 可视化 路径。
    """
    if not visual_paths:
        return

    tile_width = 320
    tile_height = 360
    columns = 3

    rows_needed = math.ceil(len(visual_paths) / columns)
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows_needed * tile_height),
        "white",
    )

    for index, path in enumerate(visual_paths):
        with Image.open(path) as source:
            image = source.convert("RGB")
        image.thumbnail((tile_width - 10, tile_height - 10))

        tile = Image.new("RGB", (tile_width, tile_height), "white")
        x = (tile_width - image.width) // 2
        y = (tile_height - image.height) // 2
        tile.paste(image, (x, y))

        col = index % columns
        row = index // columns
        sheet.paste(tile, (col * tile_width, row * tile_height))

    sheet_path = OUTPUT_ROOT / region / f"{region}_formal_contact_sheet.jpg"
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path, quality=92)


def _mean(values: list[float]) -> float | None:
    """Return arithmetic mean or None.

    Args:
        values: 本步骤处理的数值或附加字段。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build summaries for regions actually present in this run.

    This allows --limit smoke tests without dividing by zero for
    regions that were not included in the limited run.

    Args:
        rows: 待处理的逐行记录。

    Returns:
        返回 output，由函数体中同名变量的计算/收集过程得到。
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["region"])].append(row)

    output = []

    def aggregate(label: str, group: list[dict[str, Any]]) -> dict[str, Any]:
        """汇总。

        Args:
            label: 标签。
            group: group。

        Returns:
            结果字典，主要字段为 region, n, non_empty_prediction_rate, mean_top_score,
            mean_top_box_area_ratio, mean_inference_seconds。
        """
        detected = [row for row in group if row["detected"] == "yes"]
        scores = [
            float(row["top_score"]) for row in detected if str(row["top_score"]).strip()
        ]
        area_ratios = [
            float(row["top_box_area_ratio"])
            for row in detected
            if str(row["top_box_area_ratio"]).strip()
        ]
        times = [float(row["inference_seconds"]) for row in group]

        return {
            "region": label,
            "n": len(group),
            "non_empty_prediction_rate": round(
                100.0 * len(detected) / len(group),
                1,
            ),
            "mean_top_score": (round(_mean(scores), 4) if scores else ""),
            "mean_top_box_area_ratio": (
                round(_mean(area_ratios), 4) if area_ratios else ""
            ),
            "mean_inference_seconds": round(_mean(times) or 0.0, 4),
        }

    for region in REGIONS:
        group = grouped[region]
        if group:
            output.append(aggregate(region, group))

    if rows:
        output.append(aggregate("OVERALL", rows))
    return output


def build_manual_audit_template(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create a human-review template.

    Args:
        rows: 待处理的逐行记录。

    Returns:
        返回 audit，由函数体中同名变量的计算/收集过程得到。
    """
    audit = []

    for row in rows:
        audit.append(
            {
                "candidate_id": row["candidate_id"],
                "region": row["region"],
                "image_name": row["image_name"],
                "prompt": row["prompt"],
                "detected": row["detected"],
                "top_score": row["top_score"],
                "top_box_area_ratio": row["top_box_area_ratio"],
                "localization_quality": "",
                "review_note": "",
            }
        )

    return audit


def main() -> None:
    """Run the formal 40-case coverage evaluation.

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    args = parse_args()
    cases = build_cases()

    if args.limit is not None:
        cases = cases[: args.limit]

    device = get_device()
    print(f"Using device: {device}")
    print(f"Cases: {len(cases)}")
    print(f"Threshold: {args.threshold}")

    processor, model = load_model(device)

    result_rows = []
    visuals_by_region: dict[str, list[Path]] = defaultdict(list)

    for index, case in enumerate(cases, start=1):
        candidate_id = case["candidate_id"]
        region = case["region"].strip().lower()
        prompt = PROMPTS[region]

        crop_path = PROJECT_ROOT / Path(case["candidate_crop_path"])
        if not crop_path.is_file():
            raise FileNotFoundError(
                f"Candidate crop not found for {candidate_id}: {crop_path}"
            )

        image = Image.open(crop_path).convert("RGB")

        print(f"[{index}/{len(cases)}] {candidate_id} ({region})")

        detections, elapsed = infer(
            image=image,
            prompt=prompt,
            threshold=args.threshold,
            processor=processor,
            model=model,
            device=device,
        )

        top = detections[0] if detections else None
        top_score = top["score"] if top else ""
        top_area = box_area_ratio(top["box"], image.width, image.height) if top else ""

        result_rows.append(
            {
                "candidate_id": candidate_id,
                "region": region,
                "image_name": case["image_name"],
                "item_id": case["item_id"],
                "category_name": case["category_name"],
                "candidate_crop_path": case["candidate_crop_path"],
                "prompt": prompt,
                "threshold": args.threshold,
                "detected": "yes" if detections else "no",
                "num_detections": len(detections),
                "top_score": top_score,
                "top_box_area_ratio": top_area,
                "inference_seconds": elapsed,
                "target_present": case["target_present"],
                "visibility": case["visibility"],
                "selection_note": case["selection_note"],
                "detections_json": json.dumps(detections),
            }
        )

        if not args.skip_visuals:
            visual = draw_result(
                image=image,
                candidate_id=candidate_id,
                region=region,
                prompt=prompt,
                detections=detections,
            )

            visual_path = OUTPUT_ROOT / region / f"{candidate_id}_prediction.jpg"
            visual_path.parent.mkdir(parents=True, exist_ok=True)
            visual.save(visual_path, quality=92)
            visuals_by_region[region].append(visual_path)

    write_csv(CASE_RESULTS_FILE, result_rows)
    write_csv(SUMMARY_FILE, summarize(result_rows))
    write_csv(
        MANUAL_AUDIT_FILE,
        build_manual_audit_template(result_rows),
    )

    if not args.skip_visuals:
        for region, paths in visuals_by_region.items():
            make_contact_sheets(region, paths)

    print("\nFinished.")
    print(f"Case results: {CASE_RESULTS_FILE}")
    print(f"Summary: {SUMMARY_FILE}")
    print(f"Manual audit template: {MANUAL_AUDIT_FILE}")
    print(f"Visual outputs: {OUTPUT_ROOT}")
    print(
        "\nReminder: non-empty prediction rate is not localization accuracy. "
        + "Fill localization_quality with correct/coarse/wrong/missed after "
        + "visual review."
    )


if __name__ == "__main__":
    main()
