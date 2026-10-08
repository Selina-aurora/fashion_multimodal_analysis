"""3.1.2 文本引导区域定位：检测覆盖率、粗框可用率和人工定位准确率分开解释。"""

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

PROJECT_ROOT = get_project_root()

INPUT_MANIFEST = (
    PROJECT_ROOT
    / "reports"
    / "prd_region_coverage"
    / "shoulder_waist_pattern_refinement_v1"
    / "refinement_manifest.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_region_coverage"
    / "shoulder_waist_pattern_refinement_v1"
)

WINDOW_RESULTS = REPORT_DIR / "window_results.csv"
CASE_RESULTS = REPORT_DIR / "case_results.csv"
AUDIT_TEMPLATE = REPORT_DIR / "manual_audit_template.csv"
CONTACT_SHEET = REPORT_DIR / "selected_refinement_contact_sheet.jpg"
SUMMARY_PATH = REPORT_DIR / "inference_summary.txt"


def parse_args() -> Any:
    """解析并校验命令行参数。

    Returns:
        已通过参数合法性检查的命令行配置。
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-id",
        default="IDEA-Research/grounding-dino-tiny",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    p.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
    )
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    """选择可用计算设备，并按本实验策略处理 CUDA 可用性。

    Args:
        name: 当前标签、字段或产物名称。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        RuntimeError: CUDA requested but unavailable.
    """
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_csv(path: Path) -> Any:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    """按确定的字段顺序保存实验 CSV。

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


def resolve_project_path(raw: Any) -> Path:
    """将项目相对文件引用解析到当前工作副本。

    Args:
        raw: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        当前工作副本可使用的路径。
    """
    return resolve_artifact_path(raw)


def parse_threshold(raw: str, default: float = 0.30) -> float:
    """解析 阈值。

    Args:
        raw: 本步骤使用的原始值，转换/筛选规则见函数体。
        default: default。

    Returns:
        返回 default，由函数体中同名变量的计算/收集过程得到。
    """
    try:
        return float(raw)
    except Exception:
        return default


def to_python(x: Any) -> Any:
    """转换 python。

    Args:
        x: 本步骤处理的对象；使用 detach 接口。

    Returns:
        返回 x，由函数体中同名变量的计算/收集过程得到。
    """
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x


def run_grounding_dino(
    image: Image.Image,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
    processor: Any,
    model: Any,
    device: Any,
) -> tuple[Any, ...]:
    # Grounding DINO text input works best with a trailing period.
    """用图像与文本查询生成局部区域候选。

    Args:
        image: 本步骤处理的图像对象。
        prompt: 提示文本。
        box_threshold: 本步骤的 box 判定阈值。
        text_threshold: 本步骤的 text 判定阈值。
        processor: 与模型配套的输入处理器。
        model: 已构建的模型对象，由调用方负责选择权重。
        device: 当前计算设备，与输入张量和模型设备保持一致。

    Returns:
        按顺序返回 detections, elapsed 等结果。
    """
    text = prompt.strip()
    if not text.endswith("."):
        text += "."

    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    with torch.inference_mode():
        outputs = model(**inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0

    target_sizes = torch.tensor(
        [[image.height, image.width]],
        device=device,
    )

    # Current transformers API.
    try:
        processed = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )
    except TypeError:
        # Compatibility with versions using threshold=...
        processed = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )

    result = processed[0]

    boxes = to_python(result.get("boxes", []))
    scores = to_python(result.get("scores", []))
    labels = result.get("text_labels", result.get("labels", []))
    labels = to_python(labels)

    detections = []

    for i, box in enumerate(boxes):
        score = float(scores[i]) if i < len(scores) else 0.0

        label = ""
        if i < len(labels):
            label = str(labels[i])

        detections.append(
            {
                "label": label,
                "score": score,
                "box": [float(v) for v in box],
            }
        )

    detections.sort(
        key=lambda d: d["score"],
        reverse=True,
    )

    return detections, elapsed


def map_box_to_parent(
    box: Any,
    saved_crop_size: Any,
    parent_window: Any,
) -> list[Any]:
    """map 边界框 转换 parent。

    Args:
        box: xyxy 坐标的候选框。
        saved_crop_size: saved 裁剪 size。
        parent_window: parent 区域窗口。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    sx1, sy1, sx2, sy2 = [float(v) for v in box]
    saved_w, saved_h = saved_crop_size

    wx1, wy1, wx2, wy2 = [float(v) for v in parent_window]
    win_w = wx2 - wx1
    win_h = wy2 - wy1

    px1 = wx1 + sx1 / saved_w * win_w
    py1 = wy1 + sy1 / saved_h * win_h
    px2 = wx1 + sx2 / saved_w * win_w
    py2 = wy1 + sy2 / saved_h * win_h

    return [px1, py1, px2, py2]


def box_area_ratio(box: Any, parent_size: Any) -> Any:
    """边界框 面积 比例。

    Args:
        box: xyxy 坐标的候选框。
        parent_size: parent size。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = parent_size

    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    denom = float(w * h)

    return area / denom if denom > 0 else 0.0


def make_contact_sheet(case_rows: list[dict]) -> None:
    """把样本图与说明拼成审核联系表，便于逐例比较。

    Args:
        case_rows: 待处理的 case 记录。
    """
    if not case_rows:
        return

    tile_w = 390
    tile_h = 430
    cols = 3
    rows_n = math.ceil(len(case_rows) / cols)

    sheet = Image.new(
        "RGB",
        (cols * tile_w, rows_n * tile_h),
        "white",
    )

    font = ImageFont.load_default()

    for i, row in enumerate(case_rows):
        parent_path = resolve_project_path(row["parent_crop_path"])
        image = Image.open(parent_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        detected = str(row["refinement_detected"]).lower() == "yes"

        if detected:
            box = json.loads(row["selected_box_parent_json"])
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=(0, 230, 0),
                width=4,
            )

            draw.text(
                (max(2, x1), max(2, y1 - 14)),
                f"{row['selected_window']} {float(row['refinement_top_score']):.3f}",
                fill=(255, 0, 0),
                font=font,
            )

        image.thumbnail(
            (tile_w - 20, 330),
            Image.Resampling.LANCZOS,
        )

        x0 = (i % cols) * tile_w
        y0 = (i // cols) * tile_h

        sheet.paste(
            image,
            (
                x0 + (tile_w - image.width) // 2,
                y0 + 5,
            ),
        )

        draw_sheet = ImageDraw.Draw(sheet)

        lines = [
            f"{row['candidate_id']} | {row['region']}",
            (
                f"baseline={row['baseline_quality']} | "
                + f"refined_detected={row['refinement_detected']}"
            ),
            (
                f"score={row['refinement_top_score']} | "
                + f"area_parent={row['refinement_top_box_area_ratio_parent']}"
            ),
            f"window={row['selected_window']}",
        ]

        ty = y0 + 340

        for line in lines:
            draw_sheet.text(
                (x0 + 7, ty),
                line,
                fill="black",
                font=font,
            )
            ty += 18

    CONTACT_SHEET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        CONTACT_SHEET,
        quality=92,
    )


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    args = parse_args()
    device = choose_device(args.device)

    rows = read_csv(INPUT_MANIFEST)

    print("Loading model...")
    print("model:", args.model_id)
    print("device:", device)

    processor = AutoProcessor.from_pretrained(args.model_id)

    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(
        device
    )

    model.eval()

    window_rows = []

    print()
    print(f"Running refinement inference on {len(rows)} crops...")

    for idx, row in enumerate(rows, start=1):
        crop_path = resolve_project_path(row["refinement_crop_path"])
        parent_path = resolve_project_path(row["parent_crop_path"])

        if not crop_path.is_file():
            raise FileNotFoundError(crop_path)
        if not parent_path.is_file():
            raise FileNotFoundError(parent_path)

        image = Image.open(crop_path).convert("RGB")
        parent = Image.open(parent_path).convert("RGB")

        threshold = parse_threshold(
            row.get("threshold", ""),
            default=0.30,
        )

        detections, elapsed = run_grounding_dino(
            image=image,
            prompt=row["prompt"],
            box_threshold=threshold,
            text_threshold=args.text_threshold,
            processor=processor,
            model=model,
            device=device,
        )

        wx1 = int(float(row["window_x1"]))
        wy1 = int(float(row["window_y1"]))
        wx2 = int(float(row["window_x2"]))
        wy2 = int(float(row["window_y2"]))

        mapped = []

        for det in detections:
            pbox = map_box_to_parent(
                det["box"],
                saved_crop_size=image.size,
                parent_window=(wx1, wy1, wx2, wy2),
            )

            mapped.append(
                {
                    **det,
                    "parent_box": pbox,
                    "parent_box_area_ratio": box_area_ratio(
                        pbox,
                        parent.size,
                    ),
                }
            )

        top = mapped[0] if mapped else None

        out = dict(row)
        out.update(
            {
                "refinement_detected": "yes" if top else "no",
                "refinement_num_detections": len(mapped),
                "refinement_top_score": (round(float(top["score"]), 6) if top else ""),
                "refinement_top_box_area_ratio_parent": (
                    round(
                        float(top["parent_box_area_ratio"]),
                        6,
                    )
                    if top
                    else ""
                ),
                "refinement_top_box_parent_json": (
                    json.dumps(top["parent_box"]) if top else ""
                ),
                "refinement_inference_seconds": round(
                    elapsed,
                    6,
                ),
                "refinement_detections_json": json.dumps(mapped),
            }
        )

        window_rows.append(out)

        print(
            f"[{idx:02d}/{len(rows)}] "
            + f"{row['candidate_id']} "
            + f"{row['window_name']} "
            + f"det={out['refinement_detected']} "
            + f"n={len(mapped)} "
            + f"score={out['refinement_top_score']}"
        )

    write_csv(
        WINDOW_RESULTS,
        window_rows,
    )

    grouped = defaultdict(list)

    for row in window_rows:
        grouped[row["candidate_id"]].append(row)

    case_rows = []

    for candidate_id, group in grouped.items():
        detected_rows = [
            r for r in group if str(r["refinement_detected"]).lower() == "yes"
        ]

        if detected_rows:
            selected = max(
                detected_rows,
                key=lambda r: float(r["refinement_top_score"]),
            )

            detected = "yes"
            top_score = selected["refinement_top_score"]
            area = selected["refinement_top_box_area_ratio_parent"]
            box_json = selected["refinement_top_box_parent_json"]
            selected_window = selected["window_name"]
        else:
            selected = group[0]
            detected = "no"
            top_score = ""
            area = ""
            box_json = ""
            selected_window = ""

        case_rows.append(
            {
                "candidate_id": candidate_id,
                "region": selected["region"],
                "image_name": selected["image_name"],
                "item_id": selected["item_id"],
                "parent_crop_path": selected["parent_crop_path"],
                "prompt": selected["prompt"],
                "threshold": selected["threshold"],
                "baseline_quality": selected["baseline_quality"],
                "baseline_detected": selected["baseline_detected"],
                "baseline_top_score": selected["baseline_top_score"],
                "baseline_top_box_area_ratio": selected["baseline_top_box_area_ratio"],
                "refinement_detected": detected,
                "refinement_top_score": top_score,
                "refinement_top_box_area_ratio_parent": area,
                "selected_window": selected_window,
                "selected_box_parent_json": box_json,
                "refinement_quality": "",
                "review_note": "",
            }
        )

    region_order = {
        "shoulder": 0,
        "waist": 1,
        "pattern": 2,
    }

    case_rows.sort(
        key=lambda r: (
            region_order.get(
                r["region"],
                99,
            ),
            r["candidate_id"],
        )
    )

    write_csv(
        CASE_RESULTS,
        case_rows,
    )

    audit_rows = []

    for row in case_rows:
        audit_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "region": row["region"],
                "baseline_quality": row["baseline_quality"],
                "baseline_top_box_area_ratio": row["baseline_top_box_area_ratio"],
                "refinement_detected": row["refinement_detected"],
                "refinement_top_score": row["refinement_top_score"],
                "selected_window": row["selected_window"],
                "refinement_top_box_area_ratio_parent": row[
                    "refinement_top_box_area_ratio_parent"
                ],
                "refinement_quality": "",
                "review_note": "",
            }
        )

    write_csv(
        AUDIT_TEMPLATE,
        audit_rows,
    )

    make_contact_sheet(case_rows)

    lines = [
        "PRD 3.1.2 Shoulder / Waist / Pattern Spatial Refinement v1",
        "==========================================================",
        "",
        f"model={args.model_id}",
        f"device={device}",
        f"text_threshold={args.text_threshold}",
        "box_threshold=per-case value inherited from baseline manifest",
        "",
    ]

    for region in ("shoulder", "waist", "pattern"):
        rr = [r for r in case_rows if r["region"] == region]

        n = len(rr)

        detected = sum(str(r["refinement_detected"]).lower() == "yes" for r in rr)

        lines.append(
            f"{region}: N={n}, "
            + f"non_empty={detected}, "
            + f"non_empty_rate={100.0 * detected / n:.1f}%"
            if n
            else f"{region}: N=0"
        )

    lines += [
        "",
        "Important:",
        "- This file reports automatic detection availability only.",
        "- Correct/coarse/wrong/missed must be assigned by manual visual review.",
        "- Shoulder uses two side windows per source case; the highest-score detected window is selected.",
        "- Waist and pattern use one fixed spatial window each.",
    ]

    SUMMARY_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== REFINEMENT INFERENCE FINISHED ===")
    print("Window results:", WINDOW_RESULTS.relative_to(PROJECT_ROOT))
    print("Case results  :", CASE_RESULTS.relative_to(PROJECT_ROOT))
    print("Audit template:", AUDIT_TEMPLATE.relative_to(PROJECT_ROOT))
    print("Contact sheet :", CONTACT_SHEET.relative_to(PROJECT_ROOT))
    print("Summary       :", SUMMARY_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
