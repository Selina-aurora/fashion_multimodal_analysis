"""3.1.1 实例分割：数据与模型版本分别记录，指标按固定匹配和计时口径解释。"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

from fashion_multimodal_analysis.common.io import write_json
from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)
from fashion_multimodal_analysis.evaluation.metrics import box_iou, mask_iou
from fashion_multimodal_analysis.evaluation.protocol import (
    confidence_first_match as greedy_match,
)
from fashion_multimodal_analysis.evaluation.protocol import (
    detection_summary,
    macro_summary,
)
from fashion_multimodal_analysis.evaluation.runtime import predict_timed
from fashion_multimodal_analysis.segmentation.modeling import (
    build_evaluation_model as build_model,
)

PROJECT_ROOT = get_project_root()
VAL_CSV = PROJECT_ROOT / "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv"
DEFAULT_CKPT = (
    PROJECT_ROOT
    / "outputs"
    / "prd_instance_segmentation"
    / "maskrcnn_8class_baseline_v2"
    / "checkpoint_last.pth"
)
REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "reruns"
    / "protocol_v2"
    / "eval_prd31_core_regression_v2_balanced_v1"
)
VIS_DIR = REPORT_DIR / "visualizations"

CLASS_TO_ID = {
    "top": 1,
    "pants": 2,
    "skirt": 3,
    "outerwear": 4,
    "dress": 5,
    "shoe": 6,
    "bag": 7,
    "accessory": 8,
}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
NUM_CLASSES = 9
PRD_MASK_IOU_TARGET = 0.85
PRD_TIME_TARGET_MS = 50.0


def args_parse() -> Any:
    """解析本次实验的命令行参数。

    Returns:
        已通过参数合法性检查的命令行配置。
    """
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p.add_argument("--val-csv", default=str(VAL_CSV))
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--score-threshold", type=float, default=0.40)
    p.add_argument("--match-bbox-iou", type=float, default=0.50)
    p.add_argument("--mask-threshold", type=float, default=0.50)
    p.add_argument("--warmup-runs", type=int, default=10)
    p.add_argument("--max-visualizations", type=int, default=32)
    p.add_argument(
        "--report-dir",
        default=str(REPORT_DIR),
        help="Output directory for this evaluation run",
    )
    p.add_argument("--timed-passes", type=int, default=5)
    return p.parse_args()


def resolve_path(v: Any) -> Path:
    """将清单或配置中的相对路径解析到当前项目/数据目录。

    Args:
        v: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        当前工作副本可使用的路径。
    """
    return resolve_artifact_path(v)


def read_csv(path: str | Path) -> Any:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。
    """
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def group_rows(rows: list[dict[str, Any]]) -> Any:
    """按共同标识分组记录，避免同一对象被重复统计。

    Args:
        rows: 待处理的逐行记录。

    Returns:
        返回 out，由函数体中同名变量的计算/收集过程得到。
    """
    g = defaultdict(list)
    for r in rows:
        g[(r["source_dataset"].strip(), r["source_image"].strip())].append(r)
    out = []
    for (ds, img), items in g.items():
        out.append({"source_dataset": ds, "source_image": img, "instances": items})
    out.sort(key=lambda x: (x["source_dataset"], x["source_image"]))
    return out


def open_mask(path: str | Path) -> Any:
    """读取掩码并转换为当前实验使用的前景表示。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    arr = np.asarray(Image.open(path).convert("L"))
    return Image.fromarray(((arr > 0).astype(np.uint8) * 255), mode="L")


def reconstruct_mask(mask_path: str | Path, image_size: Any, bbox: Any) -> Any:
    """将 ROI 掩码放回原图坐标，保证后续 IoU 使用同一坐标系。

    Args:
        mask_path: 二值目标掩码的文件引用。
        image_size: 图像的宽、高，用于处理坐标和掩码尺寸。
        bbox: xyxy 坐标的目标框。

    Returns:
        返回 canvas，由函数体中同名变量的计算/收集过程得到。
    """
    iw, ih = image_size
    x1, y1, x2, y2 = bbox
    m = open_mask(mask_path)
    if m.size == (iw, ih):
        return m
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    if m.size != (bw, bh):
        m = m.resize((bw, bh), Image.Resampling.NEAREST)
    canvas = Image.new("L", (iw, ih), 0)
    cx1, cy1 = max(0, min(iw, x1)), max(0, min(ih, y1))
    cx2, cy2 = max(0, min(iw, x2)), max(0, min(ih, y2))
    if cx2 <= cx1 or cy2 <= cy1:
        return canvas
    mx1, my1 = cx1 - x1, cy1 - y1
    canvas.paste(m.crop((mx1, my1, mx1 + (cx2 - cx1), my1 + (cy2 - cy1))), (cx1, cy1))
    return canvas


class ValDataset(Dataset):
    """保存 ValDataset 的数据/运行职责。

    Attributes:
        samples: samples。
    """

    def __init__(self, csv_path: str | Path) -> None:
        """保存当前对象需要的配置、数据引用和状态。

        Args:
            csv_path: 对应文件的相对路径或当前解析后的路径。
        """
        self.samples = group_rows(read_csv(csv_path))

    def __len__(self) -> Any:
        """返回当前数据集或容器中的样本数量。

        Returns:
            本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        """取得指定样本，并生成本数据集约定的输入和目标。

        Args:
            idx: 待访问样本的整数下标。

        Returns:
            本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

        Raises:
            ValueError: 输入或实验状态不符合检查条件。
        """
        s = self.samples[idx]
        ip = resolve_path(s["source_image"])
        image = Image.open(ip).convert("RGB")
        iw, ih = image.size
        boxes = []
        labels = []
        masks = []
        meta = []
        for r in s["instances"]:
            cls = r["garment_category"].strip().lower()
            x1, y1, x2, y2 = [
                int(float(r[k]))
                for k in ("gt_bbox_x1", "gt_bbox_y1", "gt_bbox_x2", "gt_bbox_y2")
            ]
            x1 = max(0, min(iw - 1, x1))
            y1 = max(0, min(ih - 1, y1))
            x2 = max(x1 + 1, min(iw, x2))
            y2 = max(y1 + 1, min(ih, y2))
            mp = resolve_path(r["gt_mask_path"])
            m = np.asarray(reconstruct_mask(mp, (iw, ih), (x1, y1, x2, y2))) > 0
            if not m.any():
                raise ValueError(f"Empty GT mask: {mp}")
            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_TO_ID[cls])
            masks.append(m.astype(np.uint8))
            meta.append(
                {
                    "record_id": r.get("record_id", ""),
                    "garment_id": r.get("garment_id", ""),
                    "fine_category": r.get("fine_or_source_category", ""),
                    "gt_class": cls,
                }
            )
        return (
            F.to_tensor(image),
            {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "masks": torch.tensor(np.stack(masks), dtype=torch.uint8),
            },
            {
                "source_dataset": s["source_dataset"],
                "source_image": s["source_image"],
                "image_path": ip,
                "gt_meta": meta,
            },
        )


def choose_device(name: str) -> Any:
    """选择可用计算设备，并按本实验策略处理 CUDA 可用性。

    Args:
        name: 当前标签、字段或产物名称。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        RuntimeError: CUDA unavailable
    """
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
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
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def ratio(a: Any, b: Any) -> Any:
    """按当前调用方定义的分子与分母计算比例。

    Args:
        a: 当前函数的第一个输入，含义随运算而定。
        b: 当前函数的第二个输入，含义随运算而定。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return a / b if b else float("nan")


def mean(v: Any) -> Any:
    """计算本组数值的平均值，并按本模块策略处理空组。

    Args:
        v: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return sum(v) / len(v) if v else float("nan")


def fmt(v: Any) -> Any:
    """把统计值转换为报告显示文本。

    Args:
        v: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    try:
        v = float(v)
        return "NA" if math.isnan(v) else f"{v:.4f}"
    except Exception:
        return str(v)


def summarize(rows: list[dict[str, Any]], pred_count: int) -> dict[str, Any]:
    """汇总逐例结果，保持当前指标的分母定义。

    Args:
        rows: 待处理的逐行记录。
        pred_count: 当前样本的候选数量。

    Returns:
        结果字典，主要字段为 gt_count, prediction_count, bbox50_recall, bbox50_precision,
        category_accuracy_on_localized, end_to_end_bbox50_class_recall,
        mean_bbox_iou_localized, mean_mask_iou_localized, mean_mask_iou_correct_class,
        mask_iou_ge_0_50_rate。
    """
    localized = sum(bool(r["localized_bbox50"]) for r in rows)
    class_ok = sum(bool(r["class_correct"]) for r in rows)
    mask50 = sum(bool(r["mask_iou_ge_0_50"]) for r in rows)
    mask85 = sum(bool(r["mask_iou_ge_0_85"]) for r in rows)
    bi = [float(r["bbox_iou"]) for r in rows if r["localized_bbox50"]]
    mi = [float(r["mask_iou"]) for r in rows if r["localized_bbox50"]]
    mic = [
        float(r["mask_iou"])
        for r in rows
        if r["localized_bbox50"] and r["class_correct"]
    ]
    return {
        "gt_count": len(rows),
        "prediction_count": pred_count,
        "bbox50_recall": ratio(localized, len(rows)),
        "bbox50_precision": ratio(localized, pred_count),
        "category_accuracy_on_localized": ratio(class_ok, localized),
        "end_to_end_bbox50_class_recall": ratio(class_ok, len(rows)),
        "mean_bbox_iou_localized": mean(bi),
        "mean_mask_iou_localized": mean(mi),
        "mean_mask_iou_correct_class": mean(mic),
        "mask_iou_ge_0_50_rate": ratio(mask50, len(rows)),
        "mask_iou_ge_0_85_rate": ratio(mask85, len(rows)),
        "missed_gt_count": len(rows) - localized,
        "wrong_class_localized_count": localized - class_ok,
        "false_positive_count": max(0, pred_count - localized),
    }


def draw_vis(
    image_path: str | Path,
    gt_boxes: Any,
    gt_labels: Any,
    pred_boxes: Any,
    pred_labels: Any,
    pred_scores: Any,
    out_path: str | Path,
) -> None:
    """绘制预测与参考区域，便于检查位置和边界。

    Args:
        image_path: 目标原图的文件引用。
        gt_boxes: 参考目标的 xyxy 边界框。
        gt_labels: 参考目标的类别 ID。
        pred_boxes: 模型预测的 xyxy 边界框。
        pred_labels: 模型预测的类别 ID。
        pred_scores: 模型候选的置信度。
        out_path: 结果文件路径。
    """
    img = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for b, label_id in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = map(int, b)
        c = ID_TO_CLASS[int(label_id)]
        d.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=3)
        d.text((x1 + 2, max(2, y1 - 14)), f"GT {c}", fill=(0, 160, 0), font=font)
    for b, label_id, s in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = map(int, b)
        c = ID_TO_CLASS.get(int(label_id), str(int(label_id)))
        d.rectangle([x1, y1, x2, y2], outline=(220, 0, 0), width=2)
        d.text(
            (x1 + 2, min(img.height - 14, y2 + 2)),
            f"P {c} {float(s):.2f}",
            fill=(180, 0, 0),
            font=font,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=92)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        ValueError: timed-passes must be >=1 and warmup-runs >=0
    """
    global REPORT_DIR, VIS_DIR
    a = args_parse()
    REPORT_DIR = resolve_artifact_path(a.report_dir)
    VIS_DIR = REPORT_DIR / "visualizations"
    if a.timed_passes < 1 or a.warmup_runs < 0:
        raise ValueError("timed-passes must be >=1 and warmup-runs >=0")
    device = choose_device(a.device)
    ckpt = resolve_path(a.checkpoint)
    val_csv = resolve_path(a.val_csv)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    ds = ValDataset(val_csv)
    cp = torch.load(ckpt, map_location="cpu", weights_only=False)
    saved = cp.get("args", {})
    min_size = int(saved.get("min_size", 640))
    max_size = int(saved.get("max_size", 1024))
    model = build_model(min_size, max_size)
    model.load_state_dict(cp["model_state_dict"])
    model.to(device).eval()
    epoch = int(cp.get("epoch", -1))

    print(f"device={device} | val_images={len(ds)} | checkpoint_epoch={epoch}")

    warm, _, _ = ds[0]
    warm = warm.to(device)
    with torch.inference_mode():
        for _ in range(a.warmup_runs):
            _ = model([warm])
            if device.type == "cuda":
                torch.cuda.synchronize()

    per_gt = []
    timing = []
    image_stats = []
    pred_by_source = Counter()
    pred_by_class = Counter()

    with torch.inference_mode():
        for idx in range(len(ds)):
            image, target, meta = ds[idx]
            # Image decoding and GT loading are outside the timed module path.
            raw_image = Image.open(meta["image_path"])
            raw_image.load()
            first_prediction = None
            for timed_pass in range(a.timed_passes):
                pb, pl, ps, pm, ms = predict_timed(
                    raw_image, model, device, a.score_threshold, a.mask_threshold
                )
                timing.append(
                    {
                        "image_index": idx,
                        "timed_pass": timed_pass + 1,
                        "source_dataset": meta["source_dataset"],
                        "source_image": meta["source_image"],
                        "gt_instances": len(target["boxes"]),
                        "predictions_above_threshold": len(pb),
                        "inference_ms": ms,
                        "scope": "preprocess+forward+postprocess",
                    }
                )
                if first_prediction is None:
                    first_prediction = (pb, pl, ps, pm)
            pb, pl, ps, pm = first_prediction
            gb = target["boxes"].numpy()
            gl = target["labels"].numpy()
            gm = target["masks"].numpy().astype(np.uint8)

            matches = greedy_match(gb, pb, a.match_bbox_iou, scores=ps)
            by_gt = {gi: (pi, bi) for gi, pi, bi in matches}
            pred_by_source[meta["source_dataset"]] += len(pb)
            for x in pl:
                pred_by_class[ID_TO_CLASS.get(int(x), str(int(x)))] += 1

            image_stats.append(
                {
                    "source_dataset": meta["source_dataset"],
                    "pred_count": len(pb),
                    "match_count": len(matches),
                }
            )

            vis = VIS_DIR / f"img_{idx:03d}.jpg"
            if idx < a.max_visualizations:
                draw_vis(meta["image_path"], gb, gl, pb, pl, ps, vis)

            for gi in range(len(gb)):
                info = meta["gt_meta"][gi]
                gt_class = ID_TO_CLASS[int(gl[gi])]
                if gi in by_gt:
                    pi, bi = by_gt[gi]
                    pred_class = ID_TO_CLASS.get(int(pl[pi]), str(int(pl[pi])))
                    mi = mask_iou(gm[gi], pm[pi])
                    cc = int(pl[pi]) == int(gl[gi])
                    outcome = (
                        "wrong_class"
                        if not cc
                        else ("low_mask_iou" if mi < 0.50 else "correct")
                    )
                    score = float(ps[pi])
                    loc = True
                else:
                    bi = mi = score = 0.0
                    pred_class = ""
                    cc = False
                    outcome = "missed"
                    loc = False
                per_gt.append(
                    {
                        "image_index": idx,
                        "source_dataset": meta["source_dataset"],
                        "source_image": meta["source_image"],
                        "record_id": info["record_id"],
                        "garment_id": info["garment_id"],
                        "gt_class": gt_class,
                        "fine_category": info["fine_category"],
                        "pred_class": pred_class,
                        "pred_score": score,
                        "localized_bbox50": loc,
                        "bbox_iou": float(bi),
                        "class_correct": cc,
                        "mask_iou": float(mi),
                        "mask_iou_ge_0_50": bool(loc and cc and mi >= 0.50),
                        "mask_iou_ge_0_85": bool(
                            loc and cc and mi >= PRD_MASK_IOU_TARGET
                        ),
                        "outcome": outcome,
                        "visualization_path": (
                            str(vis.resolve()) if vis.is_file() else ""
                        ),
                    }
                )
            print(f"{idx+1:02d}/{len(ds)} done")

    total_pred = sum(x["pred_count"] for x in image_stats)
    overall = summarize(per_gt, total_pred)

    per_class = []
    for c in CLASS_TO_ID:
        rows = [r for r in per_gt if r["gt_class"] == c]
        per_class.append({"garment_category": c, **summarize(rows, pred_by_class[c])})

    per_source = []
    for src in sorted({r["source_dataset"] for r in per_gt}):
        rows = [r for r in per_gt if r["source_dataset"] == src]
        per_source.append(
            {"source_dataset": src, **summarize(rows, pred_by_source[src])}
        )

    errors = [r for r in per_gt if r["outcome"] != "correct"]
    times = [float(r["inference_ms"]) for r in timing]
    mean_ms = mean(times)
    med_ms = float(statistics.median(times))
    p95 = float(np.percentile(times, 95))

    write_csv(REPORT_DIR / "per_gt_results.csv", per_gt)
    write_csv(REPORT_DIR / "per_class_metrics.csv", per_class)
    write_csv(REPORT_DIR / "per_source_metrics.csv", per_source)
    write_csv(REPORT_DIR / "per_image_timing.csv", timing)
    write_csv(REPORT_DIR / "error_cases.csv", errors)

    micro = detection_summary(per_gt, total_pred)
    per_class_detection = {
        c: detection_summary(
            [r for r in per_gt if r["gt_class"] == c], pred_by_class[c]
        )
        for c in CLASS_TO_ID
    }
    macro = macro_summary(per_class_detection)
    write_json(
        REPORT_DIR / "protocol_metrics.json",
        {
            "matching": "confidence_first",
            "timing_scope": "preprocess+forward+postprocess",
            "warmup_runs": a.warmup_runs,
            "timed_passes": a.timed_passes,
            "micro": micro,
            "macro": macro,
            "per_class": per_class_detection,
        },
    )

    oc = Counter(r["outcome"] for r in per_gt)
    summary = [
        "PRD 3.1.1 Mask R-CNN 8-Class Baseline Evaluation v1",
        "====================================================",
        "",
        f"checkpoint={ckpt}",
        f"checkpoint_epoch={epoch}",
        f"device={device}",
        f"gpu={torch.cuda.get_device_name(0) if device.type=='cuda' else 'none'}",
        f"score_threshold={a.score_threshold}",
        f"match_bbox_iou={a.match_bbox_iou}",
        f"mask_threshold={a.mask_threshold}",
        f"warmup_runs={a.warmup_runs}",
        f"timed_passes={a.timed_passes}",
        f"validation_images={len(ds)}",
        f"gt_instances={overall['gt_count']}",
        f"predictions_above_threshold={overall['prediction_count']}",
        "",
        f"bbox50_recall={fmt(overall['bbox50_recall'])}",
        f"bbox50_precision={fmt(overall['bbox50_precision'])}",
        f"mean_bbox_iou_localized={fmt(overall['mean_bbox_iou_localized'])}",
        f"category_accuracy_on_localized={fmt(overall['category_accuracy_on_localized'])}",
        f"end_to_end_bbox50_class_recall={fmt(overall['end_to_end_bbox50_class_recall'])}",
        f"mean_mask_iou_localized={fmt(overall['mean_mask_iou_localized'])}",
        f"mean_mask_iou_correct_class={fmt(overall['mean_mask_iou_correct_class'])}",
        f"mask_iou_ge_0_50_rate={fmt(overall['mask_iou_ge_0_50_rate'])}",
        f"mask_iou_ge_0_85_rate={fmt(overall['mask_iou_ge_0_85_rate'])}",
        "",
        f"mean_inference_ms_per_image={fmt(mean_ms)}",
        f"median_inference_ms_per_image={fmt(med_ms)}",
        f"p95_inference_ms_per_image={fmt(p95)}",
        f"mean_inference_time_le_50ms={bool(mean_ms<=PRD_TIME_TARGET_MS)}",
        "",
        f"correct={oc.get('correct',0)}",
        f"low_mask_iou={oc.get('low_mask_iou',0)}",
        f"wrong_class={oc.get('wrong_class',0)}",
        f"missed={oc.get('missed',0)}",
        "",
        "Notes:",
        "- Frozen matching: predictions in descending confidence order, one-to-one bbox IoU >=0.50, independent of class.",
        "- Wrong-class localized predictions are counted as wrong_class rather than missed.",
        "- Mask IoU >= 0.85 is reported as a PRD-reference pass rate.",
        "- Timing includes RGB conversion, tensor preprocessing/device transfer, model and score/mask CPU postprocessing; disk decoding, GT, metrics and report I/O are excluded.",
        "- Do not run another GPU-heavy job during timing if you need trustworthy speed numbers.",
    ]
    (REPORT_DIR / "evaluation_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    print("\n=== EVALUATION FINISHED ===")
    print("bbox50 recall:", fmt(overall["bbox50_recall"]))
    print(
        "category accuracy on localized:",
        fmt(overall["category_accuracy_on_localized"]),
    )
    print("mean mask IoU correct-class:", fmt(overall["mean_mask_iou_correct_class"]))
    print("mask IoU >= 0.85 rate:", fmt(overall["mask_iou_ge_0_85_rate"]))
    print("mean inference ms/image:", fmt(mean_ms))
    print("summary:", REPORT_DIR / "evaluation_summary.txt")


if __name__ == "__main__":
    main()
