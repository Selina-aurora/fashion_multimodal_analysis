"""结果分析：依据已有逐例记录定位问题，不替代新的模型评估。

Read-only inference diagnosis of development cases that regressed from V5 to V6.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

DEV_SHA = "39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93"
CP_SHA = {
    "v5": "ea5a4ddf0f002256f1ce5164fc1b4b5b4d2263698f9525c9fbe7d8cf98bc1a65",
    "v6": "1ed99e67c141fc183a0502541ce0f151085a568f6434a0dc8f5573190c4e2209",
}


def digest(path: str | Path) -> str:
    """分块计算文件 SHA-256，核对文件身份而不加载整个权重到内存。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        64 位十六进制 SHA-256 摘要字符串。
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: str | Path) -> Any:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def diagnose(
    gi: Any, target: dict[str, Any], arrays: Any, id_to_class: Any
) -> tuple[Any, ...]:
    """诊断。

    Args:
        gi: gi。
        target: 记录字段，使用 boxes, masks, labels。
        arrays: arrays。
        id_to_class: ID 转换 类别。

    Returns:
        按顺序返回 details, matched, debug 等结果。
    """
    import numpy as np

    from fashion_multimodal_analysis.evaluation.metrics import box_iou, mask_iou
    from fashion_multimodal_analysis.evaluation.protocol import confidence_first_match

    boxes, labels, scores, masks = arrays
    kept = np.flatnonzero(scores >= 0.4)
    assignment = {
        g: int(kept[p])
        for g, p, _ in confidence_first_match(
            target["boxes"], boxes[kept], 0.5, scores=scores[kept]
        )
    }
    gt_box, gt_mask, gt_label = (
        target["boxes"][gi],
        target["masks"][gi],
        int(target["labels"][gi]),
    )
    ious = [box_iou(gt_box, box) for box in boxes]

    def describe(pi: Any) -> None | dict[str, Any]:
        """describe。

        Args:
            pi: pi。

        Returns:
            结果字典，主要字段为 prediction_index, pred_class, score, bbox, bbox_iou, mask_iou,
            pred_mask_pixels, gt_mask_pixels, mask_pixel_precision, mask_pixel_recall。
        """
        if pi is None:
            return None
        intersect = int(np.logical_and(gt_mask, masks[pi]).sum())
        pred_area, gt_area = int(masks[pi].sum()), int(gt_mask.sum())
        return {
            "prediction_index": int(pi),
            "pred_class": id_to_class[int(labels[pi])],
            "score": float(scores[pi]),
            "bbox": [float(x) for x in boxes[pi]],
            "bbox_iou": float(ious[pi]),
            "mask_iou": mask_iou(gt_mask, masks[pi]),
            "pred_mask_pixels": pred_area,
            "gt_mask_pixels": gt_area,
            "mask_pixel_precision": intersect / pred_area if pred_area else 0.0,
            "mask_pixel_recall": intersect / gt_area if gt_area else 0.0,
        }

    matched = assignment.get(gi)
    if matched is None:
        outcome = "missed"
    elif int(labels[matched]) != gt_label:
        outcome = "wrong_class"
    else:
        outcome = (
            "correct" if mask_iou(gt_mask, masks[matched]) >= 0.5 else "low_mask_iou"
        )
    best_any = (
        max(range(len(scores)), key=lambda p: (ious[p], float(scores[p])))
        if len(scores)
        else None
    )
    same = [p for p in range(len(scores)) if int(labels[p]) == gt_label]
    best_same = max(same, key=lambda p: (ious[p], float(scores[p]))) if same else None
    if outcome == "missed":
        if any(ious[p] >= 0.5 for p in kept):
            reason = "bbox50_candidate_exists_but_unmatched_by_frozen_assignment"
        elif any(iou >= 0.5 for iou in ious):
            reason = "bbox50_candidate_only_below_score_0_4"
        else:
            reason = "no_bbox50_candidate_in_returned_model_outputs"
    elif outcome == "wrong_class":
        reason = "matched_wrong_class"
    elif outcome == "low_mask_iou":
        reason = "correct_class_localized_but_mask_iou_below_0_5"
    else:
        reason = "correct"
    details = {
        "outcome": outcome,
        "diagnostic_reason": reason,
        "returned_candidates_at_0_05": len(scores),
        "frozen_candidates_at_0_4": len(kept),
        "frozen_match": describe(matched),
        "best_bbox_any_class_debug": describe(best_any),
        "best_bbox_same_class_debug": describe(best_same),
    }
    debug = best_same if best_same is not None else best_any
    return details, matched, debug


def check_reference(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """检查 reference。

    Args:
        actual: 记录字段，使用 frozen_match, outcome。
        expected: 记录字段，使用 outcome, pred_class, record_id。

    Returns:
        当前条件的校验结果；失败条件及返回形式见函数体。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    matched = actual["frozen_match"]
    if (
        actual["outcome"] != expected["outcome"]
        or (matched["pred_class"] if matched else "") != expected["pred_class"]
    ):
        raise ValueError(
            "Frozen result no longer reproduces recorded outcome: "
            + expected["record_id"]
        )
    for key, tolerance in [
        ("mask_iou", 1e-12),
        ("bbox_iou", 1e-6),
        ("pred_score", 1e-6),
    ]:
        value = matched["score" if key == "pred_score" else key] if matched else 0.0
        if abs(value - float(expected[key])) > tolerance:
            raise ValueError(
                "Frozen result differs from report: "
                + expected["record_id"]
                + ": "
                + key
            )


def montage(raw: Any, gt_mask: Any, gt_box: Any, panels: Any, destination: Any) -> None:
    """montage。

    Args:
        raw: 本步骤处理的对象；使用 convert, size 接口。
        gt_mask: gt 掩码。
        gt_box: gt 边界框。
        panels: panels。
        destination: 目标文件或目录。
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    width, height = raw.size
    x1, y1, x2, y2 = map(float, gt_box)
    padding = max(40, (x2 - x1) * 0.6, (y2 - y1) * 0.6)
    crop = (
        max(0, int(x1 - padding)),
        max(0, int(y1 - padding)),
        min(width, int(x2 + padding)),
        min(height, int(y2 + padding)),
    )
    output = Image.new("RGB", (1120, 1160), "white")
    for index, (title, mask, box, description) in enumerate(panels):
        arr = np.asarray(raw.convert("RGB")).copy()
        green = np.asarray(gt_mask).astype(bool)
        arr[green] = (arr[green] * 0.65 + np.array([0, 210, 60]) * 0.35).astype("uint8")
        if mask is not None:
            blue = np.asarray(mask).astype(bool)
            arr[blue] = (arr[blue] * 0.55 + np.array([40, 90, 255]) * 0.45).astype(
                "uint8"
            )
        pane = Image.fromarray(arr)
        draw = ImageDraw.Draw(pane)
        draw.rectangle([int(x) for x in gt_box], outline=(0, 190, 30), width=3)
        if box is not None:
            draw.rectangle([int(x) for x in box], outline=(30, 60, 255), width=3)
        pane = pane.crop(crop)
        pane.thumbnail((544, 480))
        col, row = index % 2, index // 2
        left, top = col * 560, row * 580
        d = ImageDraw.Draw(output)
        d.text((left + 8, top + 8), title, fill="black", font=font)
        d.text((left + 8, top + 32), description, fill="black", font=font)
        output.paste(
            pane,
            (left + 8 + (544 - pane.width) // 2, top + 80 + (480 - pane.height) // 2),
        )
    output.save(destination, quality=90)


def run(root: Path, data: Any) -> None:
    """组织本模块的准备、执行和结果保存步骤。

    Args:
        root: 当前项目根目录。
        data: 外部数据目录或本函数处理的数据结构；具体用途由操作对象决定。

    Raises:
        RuntimeError: CUDA is required for existing checkpoint inference
        ValueError: Incomplete regression case inference
    """
    os.environ["FASHION_PROJECT_ROOT"], os.environ["FASHION_DATA_ROOT"] = str(
        root
    ), str(data)
    sys.path.insert(0, str(root / "src"))
    import numpy as np
    import torch
    from PIL import Image

    from fashion_multimodal_analysis.common.schema import ID_TO_CLASS
    from fashion_multimodal_analysis.evaluation.runtime import postprocess_filtered
    from fashion_multimodal_analysis.segmentation.evaluation.eval_prd_8class_maskrcnn_v3_b1_dataexp import (
        ValDataset,
    )
    from fashion_multimodal_analysis.segmentation.modeling import build_evaluation_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for existing checkpoint inference")
    val = root / "configs/prd_8class_val_v2_full_targets.csv"
    if digest(val) != DEV_SHA:
        raise ValueError("Development manifest changed")
    reports = root / "reports/reruns/recovery_v2/v6_coverage"
    old = read_csv(reports / "v5_development_fast/per_gt_results.csv")
    new = read_csv(reports / "v6_development_fast/per_gt_results.csv")
    if (
        len(old) != 88
        or len(new) != 88
        or any(a["record_id"] != b["record_id"] for a, b in zip(old, new))
    ):
        raise ValueError("Development comparison identity differs")
    regression = [
        (a, b)
        for a, b in zip(old, new)
        if a["outcome"] == "correct" and b["outcome"] != "correct"
    ]
    if not regression:
        raise ValueError("No correct-to-error regression cases found")
    checkpoints = {
        "v5": root
        / "outputs/prd_instance_segmentation/maskrcnn_8class_v5_full_targets/checkpoint_last.pth",
        "v6": root
        / "outputs/prd_instance_segmentation/maskrcnn_8class_v6_coverage/checkpoint_last.pth",
    }
    for label, path in checkpoints.items():
        if digest(path) != CP_SHA[label]:
            raise ValueError("Checkpoint SHA differs: " + label)
        provenance = json.loads(
            (
                reports / (label + "_development_fast") / "comparison_provenance.json"
            ).read_text()
        )
        expected = provenance["report_sha256"]["per_gt_results.csv"]
        if (
            digest(reports / (label + "_development_fast") / "per_gt_results.csv")
            != expected
        ):
            raise ValueError("Reference result CSV changed: " + label)
    dataset = ValDataset(val)
    wanted = {a["source_image"] for a, b in regression}
    records = {
        a["record_id"]: {
            "record_id": a["record_id"],
            "gt_class": a["gt_class"],
            "source_image": a["source_image"],
            "reference_v5": a,
            "reference_v6": b,
        }
        for a, b in regression
    }
    visuals = {}
    device = torch.device("cuda")
    for label, path in checkpoints.items():
        cp = torch.load(path, map_location="cpu", weights_only=False)
        if int(cp.get("epoch", -1)) != 15:
            raise ValueError("Expected epoch 15 checkpoint: " + label)
        saved = cp.get("args", {})
        model = build_evaluation_model(
            int(saved.get("min_size", 640)), int(saved.get("max_size", 1024))
        )
        model.load_state_dict(cp["model_state_dict"])
        model.to(device).eval()
        del cp
        completed = 0
        with torch.inference_mode():
            for index, sample in enumerate(dataset.samples):
                if sample["source_image"] not in wanted:
                    continue
                tensor, target, meta = dataset[index]
                output = model([tensor.to(device)])[0]
                arrays = postprocess_filtered(output, 0.05, 0.5)
                gt = {key: value.numpy() for key, value in target.items()}
                for gi, info in enumerate(meta["gt_meta"]):
                    record_id = info["record_id"]
                    if record_id not in records:
                        continue
                    details, matched, debug = diagnose(gi, gt, arrays, ID_TO_CLASS)
                    check_reference(details, records[record_id]["reference_" + label])
                    records[record_id][label] = details
                    visual = visuals.setdefault(
                        record_id,
                        {
                            "image_path": meta["image_path"],
                            "gt_mask": gt["masks"][gi].copy(),
                            "gt_box": gt["boxes"][gi].copy(),
                        },
                    )
                    visual[label] = (
                        arrays[3][matched].copy() if matched is not None else None,
                        arrays[0][matched].copy() if matched is not None else None,
                    )
                    if label == "v6":
                        visual["debug"] = (
                            arrays[3][debug].copy() if debug is not None else None,
                            arrays[0][debug].copy() if debug is not None else None,
                        )
                    completed += 1
                    print(
                        label.upper(),
                        f"{completed}/{len(regression)}",
                        record_id,
                        details["diagnostic_reason"],
                        flush=True,
                    )
                del output, arrays, tensor, target
        if completed != len(regression):
            raise ValueError("Incomplete regression case inference")
        del model
        torch.cuda.empty_cache()
    folder = (
        root
        / "reports/reruns/recovery_v2/v6_regression_diagnosis"
        / time.strftime("%Y%m%d_%H%M%S")
    )
    folder.mkdir(parents=True, exist_ok=False)
    sections = []
    for record_id, item in records.items():
        visual = visuals[record_id]

        def caption(prediction: dict[str, Any]) -> Any:
            """caption。

            Args:
                prediction: 记录字段，使用 pred_class, score, mask_iou。

            Returns:
                本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
            """
            if prediction is None:
                return "No frozen match"
            return f"{prediction['pred_class']} score={prediction['score']:.3f} maskIoU={prediction['mask_iou']:.3f}"

        debug_info = (
            item["v6"]["best_bbox_same_class_debug"]
            or item["v6"]["best_bbox_any_class_debug"]
        )
        panels = [
            (
                "GT target",
                None,
                None,
                "Green=official GT; same GT-centered crop in all panels",
            ),
            (
                "V5 frozen score >= 0.40",
                *visual["v5"],
                caption(item["v5"]["frozen_match"]),
            ),
            (
                "V6 frozen score >= 0.40",
                *visual["v6"],
                caption(item["v6"]["frozen_match"]),
            ),
            (
                "V6 best same-class candidate (DEBUG)",
                *visual["debug"],
                caption(debug_info),
            ),
        ]
        with Image.open(visual["image_path"]) as raw:
            montage(
                raw.convert("RGB"),
                visual["gt_mask"],
                visual["gt_box"],
                panels,
                folder / (record_id + ".jpg"),
            )
        item["figure"] = record_id + ".jpg"
        sections.append(
            "<section><h2>"
            + html.escape(record_id + " / " + item["gt_class"])
            + "</h2><p>"
            + html.escape(item["v6"]["diagnostic_reason"])
            + '</p><img src="'
            + item["figure"]
            + '" style="width:100%;max-width:1120px"></section>'
        )
    payload = {
        "checkpoint_sha256": CP_SHA,
        "development_manifest_sha256": DEV_SHA,
        "frozen_score_threshold": 0.4,
        "debug_score_threshold": 0.05,
        "mask_threshold": 0.5,
        "bbox_match_iou": 0.5,
        "status": "REFERENCE_RESULTS_REPRODUCED_DIAGNOSTIC_COMPLETE",
        "cases": list(records.values()),
        "reason_counts": dict(
            Counter(item["v6"]["diagnostic_reason"] for item in records.values())
        ),
        "scope": "Development-only prediction diagnosis. GT and checkpoints unchanged. Debug candidates come from model-returned outputs after default internal filtering; scores below 0.05 and suppressed outputs are not inspected. No timing or acceptance claims.",
    }
    (folder / "diagnosis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    page = (
        '<!doctype html><meta charset="utf-8"><title>V5/V6 退步案例诊断</title><style>body{font-family:Arial,sans-serif;max-width:1150px;margin:24px auto}section{border-top:1px solid #ccc;padding:16px 0}</style><h1>V5/V6 退步案例诊断</h1><p>绿色是原始 GT，蓝色是预测掩码。四图使用同一目标周围的裁剪。DEBUG 图展示低至 0.05 的候选，只用于诊断，不改变 0.40 的正式评测阈值。若没有同类候选，DEBUG 图使用任意类别中框 IoU 最大的候选；以 diagnosis.json 的字段为准。</p>'
        + "".join(sections)
    )
    (folder / "review.html").write_text(page, encoding="utf-8")
    archive = root / "prd31_v6_regression_diagnosis.zip"
    if archive.exists():
        archive = root / ("prd31_v6_regression_diagnosis_" + folder.name + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.iterdir()):
            z.write(p, "diagnosis/" + p.name)
        z.write(Path(__file__), "patch/DIAGNOSE_PRD31_V6_REGRESSIONS.py")
    print("REASON_COUNTS=" + json.dumps(payload["reason_counts"]), flush=True)
    print("RESULTS_ZIP=" + str(archive), flush=True)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".")
    parser.add_argument(
        "--data-root", default=os.environ.get("FASHION_DATA_ROOT", "../fashion_data")
    )
    args = parser.parse_args()
    root, data = (
        Path(args.project).expanduser().resolve(),
        Path(args.data_root).expanduser().resolve(),
    )
    lock = root / "reports/reruns/recovery_v2/workflow.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl

        with lock.open("a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(root, data)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        ImportError,
    ) as error:
        parser.exit(2, f"诊断未完成：{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
