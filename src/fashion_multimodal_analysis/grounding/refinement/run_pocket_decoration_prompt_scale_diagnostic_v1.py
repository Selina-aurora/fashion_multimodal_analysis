"""3.1.2 文本引导区域定位：检测覆盖率、粗框可用率和人工定位准确率分开解释。"""

from __future__ import annotations

import argparse
import csv
import json
import logging
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
DATA_ROOT = data_root()

INPUT_CASES = (
    PROJECT_ROOT
    / "reports"
    / "prd_region_coverage"
    / "remaining5_diagnostic_v1"
    / "cases.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "prd_region_coverage"
    / "pocket_decoration_prompt_scale_v1"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_region_coverage"
    / "pocket_decoration_prompt_scale_v1"
)

RESULTS_CSV = REPORT_DIR / "condition_results.csv"
CASE_RESULTS_CSV = REPORT_DIR / "case_results.csv"
AUDIT_CSV = REPORT_DIR / "manual_audit_template.csv"
CONTACT_SHEET = REPORT_DIR / "selected_predictions_contact_sheet.jpg"
SUMMARY_TXT = REPORT_DIR / "inference_summary.txt"
RECOVERY_CSV = REPORT_DIR / "parent_crop_resolution.csv"

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
TARGET_REGIONS = {"pocket", "decoration"}


def parse_args() -> Any:
    """解析并校验命令行参数。

    Returns:
        已通过参数合法性检查的命令行配置。
    """
    p = argparse.ArgumentParser()
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


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
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


def rel(path: Path) -> str:
    """rel。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    path = path.resolve()

    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # 路径不属于项目时，进入下方数据目录/外部路径分支；这是预期匹配失败。
        pass

    try:
        p = path.relative_to(PROJECT_ROOT.parent)
        return f"../{p.as_posix()}"
    except ValueError:
        return path.as_posix()


def valid_file(path: Path | None) -> bool:
    """有效 文件。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        当前条件的校验结果；失败条件及返回形式见函数体。
    """
    if path is None:
        return False
    try:
        return path.is_file() and path.stat().st_size > 0
    except Exception:
        return False


def find_deepfashion_image(image_name: str) -> Path | None:
    """查找 deepfashion 图像。

    Args:
        image_name: 图像 name。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    known = [
        DATA_ROOT / "raw" / "train" / "train" / "image" / image_name,
        DATA_ROOT / "raw" / "train" / "image" / image_name,
        DATA_ROOT / "raw" / "validation" / "image" / image_name,
        DATA_ROOT / "raw" / "validation" / "validation" / "image" / image_name,
    ]

    for p in known:
        if valid_file(p):
            return p.resolve()

    for root in (
        DATA_ROOT / "raw" / "train",
        DATA_ROOT / "raw" / "validation",
    ):
        if not root.exists():
            continue

        found = next(
            (p for p in root.rglob(image_name) if valid_file(p)),
            None,
        )

        if found is not None:
            return found.resolve()

    return None


def find_deepfashion_annotation(image_name: str) -> Path | None:
    """查找 deepfashion 标注。

    Args:
        image_name: 图像 name。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    filename = Path(image_name).stem + ".json"

    known = [
        DATA_ROOT / "raw" / "train" / "train" / "annos" / filename,
        DATA_ROOT / "raw" / "train" / "annos" / filename,
        DATA_ROOT / "raw" / "validation" / "annos" / filename,
        DATA_ROOT / "raw" / "validation" / "validation" / "annos" / filename,
    ]

    for p in known:
        if valid_file(p):
            return p.resolve()

    for root in (
        DATA_ROOT / "raw" / "train",
        DATA_ROOT / "raw" / "validation",
    ):
        if not root.exists():
            continue

        found = next(
            (
                p
                for p in root.rglob(filename)
                if p.parent.name.lower().startswith("anno") and valid_file(p)
            ),
            None,
        )

        if found is not None:
            return found.resolve()

    return None


def bbox_from_segmentation(segmentation: Any) -> None | tuple[Any, ...]:
    """边界框 from 分割。

    Args:
        segmentation: 原始分割标注，可能为多边形或 COCO RLE。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if not isinstance(segmentation, list):
        return None

    xs = []
    ys = []

    for poly in segmentation:
        if not isinstance(poly, list):
            continue

        for i in range(0, len(poly) - 1, 2):
            try:
                xs.append(float(poly[i]))
                ys.append(float(poly[i + 1]))
            except Exception:
                # 保留历史的跳过策略，但记录坏输入，避免扫描过程静默丢失标注。
                logging.getLogger(__name__).warning(
                    "无法解析当前标注，已跳过并继续扫描", exc_info=True
                )

    if not xs or not ys:
        return None

    return (
        int(min(xs)),
        int(min(ys)),
        int(max(xs)) + 1,
        int(max(ys)) + 1,
    )


def recover_parent_crop(case: dict[str, str]) -> tuple[Path, str, str]:
    """recover parent 裁剪。

    Args:
        case: 记录字段，使用 image_name, item_id, region, candidate_id。

    Returns:
        按顺序返回 out 等结果。

    Raises:
        FileNotFoundError: 需要的文件不存在。
        KeyError: 所需字段不存在。
        ValueError: 输入或实验状态不符合检查条件。
    """
    old = resolve_project_path(case.get("candidate_crop_path", ""))

    if valid_file(old):
        return old, "existing_candidate_crop", ""

    image_name = case["image_name"].strip()
    item_id = case["item_id"].strip()

    image_path = find_deepfashion_image(image_name)
    anno_path = find_deepfashion_annotation(image_name)

    if image_path is None:
        raise FileNotFoundError(f"DeepFashion2 source image not found: {image_name}")

    if anno_path is None:
        raise FileNotFoundError(f"DeepFashion2 annotation not found: {image_name}")

    with anno_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    item = data.get(item_id)

    if not isinstance(item, dict):
        raise KeyError(f"{anno_path} has no item key {item_id!r}")

    bbox = item.get("bounding_box")

    if not isinstance(bbox, list) or len(bbox) != 4:
        bbox = bbox_from_segmentation(item.get("segmentation"))

    if bbox is None:
        raise ValueError(f"No bbox/segmentation for {image_name} -> {item_id}")

    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    crop = image.crop((x1, y1, x2, y2))

    out = (
        OUTPUT_DIR
        / "parent_crops"
        / case["region"]
        / f"{case['candidate_id']}_{image_name}"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out, quality=95)

    return (
        out,
        "rebuilt_from_deepfashion_annotation",
        rel(anno_path),
    )


def prompt_variants(region: str, baseline_prompt: str) -> list[Any]:
    """提示文本 variants。

    Args:
        region: 区域。
        baseline_prompt: baseline 提示文本。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    baseline_prompt = baseline_prompt.strip()

    if region == "pocket":
        return [
            ("baseline_prompt", baseline_prompt),
            ("concise_prompt", "pocket"),
        ]

    if region == "decoration":
        return [
            ("baseline_prompt", baseline_prompt),
            (
                "semantic_prompt",
                "decorative detail . embellishment . bow . brooch . applique . ornament",
            ),
        ]

    raise ValueError(region)


def tiles_for_parent(width: int, height: int) -> Any:
    """Four overlapping 60%-size tiles.
    This increases the relative scale of small local objects without changing model.

    Args:
        width: 图像或目标表示的宽度。
        height: 图像或目标表示的高度。

    Returns:
        返回 out，由函数体中同名变量的计算/收集过程得到。
    """
    specs = [
        ("tile_upper_left", 0.00, 0.00, 0.60, 0.60),
        ("tile_upper_right", 0.40, 0.00, 1.00, 0.60),
        ("tile_lower_left", 0.00, 0.40, 0.60, 1.00),
        ("tile_lower_right", 0.40, 0.40, 1.00, 1.00),
    ]

    out = []

    for name, x1r, y1r, x2r, y2r in specs:
        x1 = max(0, min(width - 1, int(round(x1r * width))))
        y1 = max(0, min(height - 1, int(round(y1r * height))))
        x2 = max(x1 + 1, min(width, int(round(x2r * width))))
        y2 = max(y1 + 1, min(height, int(round(y2r * height))))

        out.append((name, x1, y1, x2, y2))

    return out


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

    try:
        processed = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )
    except TypeError:
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
    labels = result.get(
        "text_labels",
        result.get("labels", []),
    )
    labels = to_python(labels)

    detections = []

    for i, box in enumerate(boxes):
        score = float(scores[i]) if i < len(scores) else 0.0
        label = str(labels[i]) if i < len(labels) else ""

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
    crop_size: Any,
    parent_window: Any,
) -> list[Any]:
    """map 边界框 转换 parent。

    Args:
        box: xyxy 坐标的候选框。
        crop_size: 裁剪 size。
        parent_window: parent 区域窗口。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    cw, ch = crop_size

    wx1, wy1, wx2, wy2 = [float(v) for v in parent_window]

    ww = wx2 - wx1
    wh = wy2 - wy1

    return [
        wx1 + x1 / cw * ww,
        wy1 + y1 / ch * wh,
        wx1 + x2 / cw * ww,
        wy1 + y2 / ch * wh,
    ]


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


def save_crop(
    image: Image.Image,
    out_path: Path,
) -> None:
    """保存 裁剪。

    Args:
        image: 本步骤处理的图像对象。
        out_path: 结果文件路径。
    """
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    crop = image

    scale = max(
        1.0,
        900.0 / max(crop.size),
    )

    if scale > 1.0:
        crop = crop.resize(
            (
                int(round(crop.width * scale)),
                int(round(crop.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

    crop.save(
        out_path,
        quality=95,
    )


def build_conditions(
    case: dict[str, str],
    parent: Image.Image,
) -> Any:
    """构建 conditions。

    Args:
        case: 记录字段，使用 region。
        parent: parent。

    Returns:
        返回 conditions，由函数体中同名变量的计算/收集过程得到。
    """
    region = case["region"].strip().lower()
    threshold = float(case.get("threshold", "0.3") or 0.3)

    conditions = []

    # 1) Full garment with the inherited baseline prompt.
    # 2) Full garment with one concise/semantic prompt.
    for prompt_name, prompt in prompt_variants(
        region,
        case.get("prompt", ""),
    ):
        conditions.append(
            {
                "condition": f"full__{prompt_name}",
                "prompt_name": prompt_name,
                "prompt": prompt,
                "window_name": "full",
                "window": (0, 0, parent.width, parent.height),
                "threshold": threshold,
            }
        )

    # One scale diagnostic: 4 overlapping tiles using ONLY the alternative prompt.
    alt_prompt_name, alt_prompt = prompt_variants(
        region,
        case.get("prompt", ""),
    )[1]

    for (
        tile_name,
        x1,
        y1,
        x2,
        y2,
    ) in tiles_for_parent(
        parent.width,
        parent.height,
    ):
        conditions.append(
            {
                "condition": f"tile__{alt_prompt_name}__{tile_name}",
                "prompt_name": alt_prompt_name,
                "prompt": alt_prompt,
                "window_name": tile_name,
                "window": (x1, y1, x2, y2),
                "threshold": threshold,
            }
        )

    return conditions


def make_contact_sheet(
    case_rows: list[dict],
) -> None:
    """把样本图与说明拼成审核联系表，便于逐例比较。

    Args:
        case_rows: 待处理的 case 记录。
    """
    if not case_rows:
        return

    tile_w = 390
    tile_h = 430
    cols = 2
    rows_n = math.ceil(len(case_rows) / cols)

    sheet = Image.new(
        "RGB",
        (
            cols * tile_w,
            rows_n * tile_h,
        ),
        "white",
    )

    font = ImageFont.load_default()

    for i, row in enumerate(case_rows):
        parent_path = resolve_project_path(row["parent_crop_path"])

        image = Image.open(parent_path).convert("RGB")

        draw = ImageDraw.Draw(image)

        if row["selected_detected"] == "yes":
            box = json.loads(row["selected_box_parent_json"])

            x1, y1, x2, y2 = [int(round(float(v))) for v in box]

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=(0, 230, 0),
                width=4,
            )

            draw.text(
                (
                    max(2, x1),
                    max(2, y1 - 14),
                ),
                f"{row['selected_condition']} {float(row['selected_score']):.3f}",
                fill=(255, 0, 0),
                font=font,
            )

        image.thumbnail(
            (
                tile_w - 20,
                325,
            ),
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

        sd = ImageDraw.Draw(sheet)

        lines = [
            f"{row['candidate_id']} | {row['region']}",
            (
                f"baseline={row['baseline_quality']} "
                + f"| selected={row['selected_detected']}"
            ),
            (
                f"score={row['selected_score']} "
                + f"| area={row['selected_box_area_ratio_parent']}"
            ),
            f"condition={row['selected_condition']}",
        ]

        ty = y0 + 335

        for line in lines:
            sd.text(
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
    """解析运行参数并执行本模块的实验入口。"""
    args = parse_args()
    device = choose_device(args.device)

    cases = [
        row
        for row in read_csv(INPUT_CASES)
        if row.get(
            "region",
            "",
        )
        .strip()
        .lower()
        in TARGET_REGIONS
    ]

    if len(cases) != 10:
        print(f"WARNING: expected 10 pocket/decoration cases, found {len(cases)}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading model:", MODEL_ID)
    print("Device:", device)

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

    model.eval()

    condition_rows = []
    recovery_rows = []

    for case_idx, case in enumerate(
        cases,
        start=1,
    ):
        parent_path, recovery_method, source_ref = recover_parent_crop(case)

        recovery_rows.append(
            {
                "candidate_id": case["candidate_id"],
                "region": case["region"],
                "image_name": case["image_name"],
                "item_id": case["item_id"],
                "parent_crop_path": rel(parent_path),
                "recovery_method": recovery_method,
                "source_reference": source_ref,
            }
        )

        parent = Image.open(parent_path).convert("RGB")

        conditions = build_conditions(
            case,
            parent,
        )

        print()
        print(
            f"[{case_idx:02d}/{len(cases)}] "
            + f"{case['candidate_id']} {case['region']} "
            + f"conditions={len(conditions)}"
        )

        for cond_idx, cond in enumerate(
            conditions,
            start=1,
        ):
            x1, y1, x2, y2 = cond["window"]

            crop_original = parent.crop(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )

            crop_path = (
                OUTPUT_DIR
                / case["region"]
                / case["candidate_id"]
                / f"{cond_idx:02d}_{cond['condition']}.jpg"
            )

            save_crop(
                crop_original,
                crop_path,
            )

            run_image = Image.open(crop_path).convert("RGB")

            detections, elapsed = run_grounding_dino(
                image=run_image,
                prompt=cond["prompt"],
                box_threshold=cond["threshold"],
                text_threshold=args.text_threshold,
                processor=processor,
                model=model,
                device=device,
            )

            mapped = []

            for det in detections:
                parent_box = map_box_to_parent(
                    det["box"],
                    crop_size=run_image.size,
                    parent_window=(
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                )

                mapped.append(
                    {
                        **det,
                        "parent_box": parent_box,
                        "parent_box_area_ratio": box_area_ratio(
                            parent_box,
                            parent.size,
                        ),
                    }
                )

            top = mapped[0] if mapped else None

            row = {
                "candidate_id": case["candidate_id"],
                "region": case["region"],
                "image_name": case["image_name"],
                "item_id": case["item_id"],
                "baseline_quality": case["localization_quality"],
                "baseline_detected": case["detected"],
                "baseline_prompt": case["prompt"],
                "baseline_threshold": case["threshold"],
                "condition": cond["condition"],
                "prompt_name": cond["prompt_name"],
                "prompt": cond["prompt"],
                "window_name": cond["window_name"],
                "window_x1": x1,
                "window_y1": y1,
                "window_x2": x2,
                "window_y2": y2,
                "input_crop_path": rel(crop_path),
                "detected": "yes" if top else "no",
                "num_detections": len(mapped),
                "top_score": (round(float(top["score"]), 6) if top else ""),
                "top_box_area_ratio_parent": (
                    round(
                        float(top["parent_box_area_ratio"]),
                        6,
                    )
                    if top
                    else ""
                ),
                "top_box_parent_json": (json.dumps(top["parent_box"]) if top else ""),
                "inference_seconds": round(
                    elapsed,
                    6,
                ),
                "detections_json": json.dumps(mapped),
            }

            condition_rows.append(row)

            print(
                f"    {cond['condition']:<45s} "
                + f"det={row['detected']} "
                + f"n={row['num_detections']} "
                + f"score={row['top_score']}"
            )

    write_csv(
        RESULTS_CSV,
        condition_rows,
    )

    write_csv(
        RECOVERY_CSV,
        recovery_rows,
    )

    grouped = defaultdict(list)

    for row in condition_rows:
        grouped[row["candidate_id"]].append(row)

    case_rows = []

    for case in cases:
        group = grouped[case["candidate_id"]]

        detected_rows = [row for row in group if row["detected"] == "yes"]

        if detected_rows:
            selected = max(
                detected_rows,
                key=lambda r: float(r["top_score"]),
            )

            selected_detected = "yes"
            selected_score = selected["top_score"]
            selected_area = selected["top_box_area_ratio_parent"]
            selected_box = selected["top_box_parent_json"]
            selected_condition = selected["condition"]
        else:
            selected = group[0]
            selected_detected = "no"
            selected_score = ""
            selected_area = ""
            selected_box = ""
            selected_condition = ""

        parent_crop_path = next(
            r["parent_crop_path"]
            for r in recovery_rows
            if r["candidate_id"] == case["candidate_id"]
        )

        case_rows.append(
            {
                "candidate_id": case["candidate_id"],
                "region": case["region"],
                "image_name": case["image_name"],
                "item_id": case["item_id"],
                "parent_crop_path": parent_crop_path,
                "target_present": case.get("target_present", ""),
                "visibility": case.get("visibility", ""),
                "baseline_quality": case["localization_quality"],
                "baseline_detected": case["detected"],
                "selected_detected": selected_detected,
                "selected_score": selected_score,
                "selected_box_area_ratio_parent": selected_area,
                "selected_condition": selected_condition,
                "selected_box_parent_json": selected_box,
                "diagnostic_quality": "",
                "review_note": "",
            }
        )

    write_csv(
        CASE_RESULTS_CSV,
        case_rows,
    )

    audit_rows = [
        {
            "candidate_id": row["candidate_id"],
            "region": row["region"],
            "target_present": row["target_present"],
            "visibility": row["visibility"],
            "baseline_quality": row["baseline_quality"],
            "selected_detected": row["selected_detected"],
            "selected_score": row["selected_score"],
            "selected_condition": row["selected_condition"],
            "selected_box_area_ratio_parent": row["selected_box_area_ratio_parent"],
            "diagnostic_quality": "",
            "review_note": "",
        }
        for row in case_rows
    ]

    write_csv(
        AUDIT_CSV,
        audit_rows,
    )

    make_contact_sheet(case_rows)

    lines = [
        "PRD 3.1.2 Pocket / Decoration Prompt-Scale Diagnostic v1",
        "========================================================",
        "",
        f"model={MODEL_ID}",
        f"device={device}",
        f"text_threshold={args.text_threshold}",
        "",
        "Conditions per case:",
        "1) full garment + inherited baseline prompt",
        "2) full garment + one concise/semantic prompt",
        "3) four overlapping 60%-size tiles + the concise/semantic prompt",
        "",
    ]

    for region in ("pocket", "decoration"):
        rr = [r for r in case_rows if r["region"] == region]

        n = len(rr)

        non_empty = sum(r["selected_detected"] == "yes" for r in rr)

        lines.append(
            f"{region}: "
            + f"N={n}, "
            + f"any_condition_non_empty={non_empty}, "
            + f"non_empty_rate={100.0 * non_empty / n:.1f}%"
            if n
            else f"{region}: N=0"
        )

    lines += [
        "",
        "Important:",
        "- Non-empty prediction is not localization correctness.",
        "- Final correct/coarse/wrong/missed must be assigned by visual review.",
        "- This is a diagnostic only; do not tune more prompts/tiles after this run unless there is a clear failure mode worth testing.",
    ]

    SUMMARY_TXT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== POCKET / DECORATION DIAGNOSTIC FINISHED ===")
    print("Conditions :", RESULTS_CSV.relative_to(PROJECT_ROOT))
    print("Cases      :", CASE_RESULTS_CSV.relative_to(PROJECT_ROOT))
    print("Audit      :", AUDIT_CSV.relative_to(PROJECT_ROOT))
    print("Contact    :", CONTACT_SHEET.relative_to(PROJECT_ROOT))
    print("Summary    :", SUMMARY_TXT.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
