"""3.1 接口联调：区分预测 ROI 传递和使用 GT 的受控诊断，保留对象 ID 与坐标对应关系。

Run frozen v2 Track A and complete Track B; leave semantic grading to humans.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from fashion_multimodal_analysis.benchmarking.freeze_prd31_reviewed_tests import (
    verify_frozen,
)
from fashion_multimodal_analysis.benchmarking.prepare_prd31_recovery import (
    ATTRIBUTES,
    REGIONS,
    applies,
    digest,
)
from fashion_multimodal_analysis.common.io import read_csv, write_csv, write_json
from fashion_multimodal_analysis.common.paths import project_root, resolve_path
from fashion_multimodal_analysis.common.schema import CLASS_TO_ID
from fashion_multimodal_analysis.evaluation.metrics import box_iou, mask_iou
from fashion_multimodal_analysis.evaluation.protocol import (
    confidence_first_match,
    detection_summary,
    flag,
    macro_summary,
    metric_status,
    per_attribute_metrics,
    valid_box,
)
from fashion_multimodal_analysis.image_processing.masks import reconstruct_full_mask
from fashion_multimodal_analysis.integration.pipeline import FashionPipeline, GarmentROI


def gt_box(row: dict[str, Any], region: bool = False) -> list[Any]:
    """gt 边界框。

    Args:
        row: 一条实例、预测或审核记录。
        region: 区域。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    prefix = "gt_region_bbox_" if region else "gt_bbox_"
    return [float(row[prefix + k]) for k in ("x1", "y1", "x2", "y2")]


def gt_roi(image: Any, row: dict[str, Any]) -> Any:
    """gt roi。

    Args:
        image: 本步骤处理的图像对象。
        row: 记录字段，使用 garment_category, gt_mask_path, sample_id。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    box = gt_box(row)
    mask = reconstruct_full_mask(
        resolve_path(row["gt_mask_path"]), image.size, tuple(int(v) for v in box)
    )
    return GarmentROI(
        "gt_" + row["sample_id"],
        row["garment_category"],
        1.0,
        box,
        np.asarray(mask) > 0,
        "controlled_gt_garment_roi",
    )


def timing_summary(values: Any) -> dict[str, Any]:
    """汇总多次计时，记录均值、分位数和实际样本数量。

    Args:
        values: 本步骤处理的数值或附加字段。

    Returns:
        结果字典，主要字段为 n, mean_ms, median_ms, p95_ms。
    """
    return {
        "n": len(values),
        "mean_ms": float(np.mean(values)) if values else None,
        "median_ms": float(np.median(values)) if values else None,
        "p95_ms": float(np.percentile(values, 95)) if values else None,
    }


def serial_prediction(prediction: Any) -> dict[str, Any]:
    """serial 预测。

    Args:
        prediction: 预测。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return {k: v for k, v in prediction.items() if k != "mask"}


def draw_region(
    image: Any, truth: Any, prediction: dict[str, Any], path: str | Path
) -> None:
    """绘制 区域。

    Args:
        image: 本步骤处理的图像对象。
        truth: truth。
        prediction: 记录字段，使用 bbox。
        path: 要读取或写入的文件路径。
    """
    preview = image.copy().convert("RGB")
    draw = ImageDraw.Draw(preview)
    draw.rectangle(gt_box(truth), outline="yellow", width=3)
    draw.rectangle(gt_box(truth, True), outline="lime", width=3)
    if prediction.get("bbox") and valid_box(prediction["bbox"]):
        draw.rectangle(prediction["bbox"], outline="red", width=3)
    preview.thumbnail((1000, 1000))
    path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(path, quality=90)


def match_gt_cases(cases: Any, rois: Any, threshold: float) -> dict[str, Any]:
    """匹配 gt 案例。

    Args:
        cases: 案例。
        rois: rois。
        threshold: 当前判定规则使用的阈值。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    matches = confidence_first_match(
        [gt_box(row) for row in cases],
        [roi.bbox for roi in rois],
        threshold,
        [roi.score for roi in rois],
    )
    return {cases[gi]["sample_id"]: (pi, iou) for gi, pi, iou in matches}


def segmentation_records(cases: Any, rois: Any, matches: Any, image: Any) -> Any:
    """分割 records。

    Args:
        cases: 案例。
        rois: rois。
        matches: matches。
        image: 本步骤处理的图像对象。

    Returns:
        返回 records，由函数体中同名变量的计算/收集过程得到。
    """
    records = []
    for row in cases:
        sid = row["sample_id"]
        match = matches.get(sid)
        label = ""
        mi = 0.0
        score = 0.0
        codes = ["SEG_MISS"]
        iou = 0.0
        if match:
            pi, iou = match
            roi = rois[pi]
            label = roi.category
            score = roi.score
            truth_mask = np.asarray(gt_roi(image, row).mask) > 0
            mi = mask_iou(truth_mask, np.asarray(roi.mask) > 0)
            if label != row["garment_category"]:
                codes = ["SEG_WRONG_CLASS"]
            elif mi < 0.85:
                missed = np.count_nonzero(truth_mask & ~(np.asarray(roi.mask) > 0))
                extra = np.count_nonzero((np.asarray(roi.mask) > 0) & ~truth_mask)
                codes = ["SEG_MASK_UNDER" if missed >= extra else "SEG_MASK_OVER"]
            else:
                codes = []
        elif rois and max(box_iou(gt_box(row), r.bbox) for r in rois) >= 0.1:
            codes = ["SEG_BBOX_SHIFT"]
        records.append(
            {
                "sample_id": sid,
                "source_image": row["source_image"],
                "gt_class": row["garment_category"],
                "pred_class": label,
                "pred_score": score,
                "localized_bbox50": bool(match),
                "bbox_iou": iou,
                "class_correct": bool(match and label == row["garment_category"]),
                "mask_iou": mi,
                "failure_codes": json.dumps(codes),
            }
        )
    return records


def evaluate(
    root: Path | None = None, report_dir: Path | None = None, backend: Any = None
) -> Any:
    """按固定评估设置计算结果，保留逐例记录和运行凭据。

    Args:
        root: 当前项目根目录。
        report_dir: 保存本次报告的目录。
        backend: backend。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
        FileExistsError: Run output exists; choose a new --report-dir
        FileNotFoundError: 需要的文件不存在。
    """
    root = root or project_root()
    meta = verify_frozen(root)
    frozen = root / "benchmark/prd_3_1_v2/frozen"
    config = json.loads((frozen / "config.json").read_text())
    if config["warmup_runs"] < 10 or config["timed_passes"] < 5:
        raise ValueError("Formal run needs the frozen 10 warmups and 5 timed passes")
    output = report_dir or root / "reports/reruns/recovery_v2/acceptance"
    if (output / "run_metadata.json").exists():
        raise FileExistsError("Run output exists; choose a new --report-dir")
    raw_core = read_csv(frozen / "segmentation_test.csv")
    core = [{**r, "sample_id": r["final_sample_id"]} for r in raw_core]
    ground = read_csv(frozen / "grounding_test.csv")
    attrs = read_csv(frozen / "attribute_test.csv")
    cases = read_csv(frozen / "end_to_end_test.csv")
    # Data I/O and input validation occur before model initialization/timers.
    missing = sorted(
        {
            str(resolve_path(r[k]))
            for r in core
            for k in ("source_image", "gt_mask_path")
            if not resolve_path(r[k]).is_file()
        }
    )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} source images/masks unavailable; first: " + missing[0]
        )
    checkpoint = resolve_path(config["segmentation_checkpoint"])
    if backend is None:
        from fashion_multimodal_analysis.integration.model_backends import ModelBackends

        backend = ModelBackends(config)
    output.mkdir(parents=True, exist_ok=True)
    pipeline = FashionPipeline(backend)
    by_source = defaultdict(list)
    for row in core:
        by_source[row["source_image"]].append(row)
    ground_by_sample = defaultdict(list)
    attr_by_sample = defaultdict(list)
    for row in ground:
        ground_by_sample[row["sample_id"]].append(row)
    for row in attrs:
        if row["applicability"] == "eligible" and not flag(row.get("ambiguous")):
            attr_by_sample[row["sample_id"]].append(row)
    selected_ids = {r["sample_id"] for r in cases}
    warm_case = next(r for r in core if r["garment_category"] == "top")
    with Image.open(resolve_path(warm_case["source_image"])) as raw:
        warm_image = raw.convert("RGB")
    warm_roi = gt_roi(warm_image, warm_case)
    # Each module is warmed explicitly even when segmentation produces zero detections.
    for _ in range(config["warmup_runs"]):
        backend.segment(warm_image)
        backend.attributes(warm_image, warm_roi, {}, ATTRIBUTES, controlled=True)
        for region in REGIONS:
            backend.ground(warm_image, warm_roi, region)
    seg_rows = []
    pred_rows = []
    timing = []
    ground_rows = []
    attr_predictions = []
    chain_rows = []
    failed_calls = []
    total_pred = 0
    pred_by_class = Counter()
    for number, (source, gt_cases) in enumerate(by_source.items(), 1):
        with Image.open(resolve_path(source)) as raw:
            image = raw.convert("RGB")
        if any(
            image.size != (int(r["image_width"]), int(r["image_height"]))
            for r in gt_cases
        ):
            raise ValueError(
                "Source image dimensions differ from frozen metadata: " + source
            )
        first_rois = None
        for pass_id in range(config["timed_passes"]):
            rois, elapsed = backend.segment(image)
            timing.append(
                {
                    "module": "segmentation",
                    "track": "A",
                    "unit": "image",
                    "source_image": source,
                    "timed_pass": pass_id + 1,
                    "ms": elapsed,
                }
            )
            if first_rois is None:
                first_rois = rois
        rois = first_rois or []
        total_pred += len(rois)
        pred_by_class.update(r.category for r in rois)
        matches = match_gt_cases(gt_cases, rois, config["bbox_match_iou"])
        seg_rows.extend(segmentation_records(gt_cases, rois, matches, image))
        matched_pred = {value[0] for value in matches.values()}
        for pi, roi in enumerate(rois):
            duplicate = pi not in matched_pred and any(
                box_iou(roi.bbox, gt_box(r)) >= config["bbox_match_iou"]
                for r in gt_cases
            )
            matched_gt = next(
                (r for r in gt_cases if matches.get(r["sample_id"], (-1,))[0] == pi),
                None,
            )
            class_ok = bool(
                matched_gt and matched_gt["garment_category"] == roi.category
            )
            pred_rows.append(
                {
                    "source_image": source,
                    "instance_id": roi.instance_id,
                    "pred_class": roi.category,
                    "score": roi.score,
                    "bbox": json.dumps(roi.bbox),
                    "failure_code": (
                        ""
                        if class_ok
                        else (
                            "SEG_WRONG_CLASS"
                            if matched_gt
                            else "SEG_DUPLICATE" if duplicate else "SEG_FALSE_POSITIVE"
                        )
                    ),
                }
            )
        chosen = [r for r in gt_cases if r["sample_id"] in selected_ids]
        if not chosen:
            continue
        query_names = sorted(
            {
                q["target_region"]
                for r in chosen
                for q in ground_by_sample[r["sample_id"]]
            }
        )
        attr_names = (
            ATTRIBUTES if any(attr_by_sample[r["sample_id"]] for r in chosen) else []
        )
        first_chain = None
        for pass_id in range(config["timed_passes"]):
            chain = pipeline.run_downstream(image, rois, query_names, attr_names)
            failed_calls.extend(
                {**r, "source_image": source, "timed_pass": pass_id + 1}
                for r in chain["failures"]
            )
            timing.extend(
                {**t, "track": "B", "source_image": source, "timed_pass": pass_id + 1}
                for t in chain["timings"]
                if t["module"] != "attributes" or attr_names
            )
            if first_chain is None:
                first_chain = chain
        for truth in chosen:
            sid = truth["sample_id"]
            controlled = gt_roi(image, truth)
            match = matches.get(sid)
            item = first_chain["instances"][match[0]] if match else None
            class_ok = bool(match and item["roi"].category == truth["garment_category"])
            stage = (
                "SEG_MISS" if not match else "SEG_WRONG_CLASS" if not class_ok else ""
            )
            for query in ground_by_sample[sid]:
                first = None
                for pass_id in range(config["timed_passes"]):
                    prediction, elapsed = backend.ground(
                        image, controlled, query["target_region"]
                    )
                    timing.append(
                        {
                            "module": "grounding",
                            "track": "A",
                            "unit": "query_instance",
                            "source_image": source,
                            "sample_id": sid,
                            "query_id": query["query_id"],
                            "target_region": query["target_region"],
                            "timed_pass": pass_id + 1,
                            "ms": elapsed,
                        }
                    )
                    if first is None:
                        first = prediction
                b = (
                    item["regions"].get(query["target_region"], {})
                    if item
                    else {"status": "segmentation_missing", "bbox": None}
                )
                for track, prediction in [("A", first), ("B", b)]:
                    path = (
                        output
                        / "grounding_visuals"
                        / f'{track}_{query["query_id"]}.jpg'
                    )
                    draw_region(image, query, prediction, path)
                    iou = (
                        box_iou(prediction["bbox"], gt_box(query, True))
                        if prediction.get("bbox")
                        else 0.0
                    )
                    ground_rows.append(
                        {
                            "track": track,
                            "query_id": query["query_id"],
                            "sample_id": sid,
                            "target_region": query["target_region"],
                            "garment_category": truth["garment_category"],
                            "garment_class_correct": True if track == "A" else class_ok,
                            "prediction_status": prediction.get("status", "missing"),
                            "bbox_iou": iou,
                            "predicted_bbox": json.dumps(prediction.get("bbox")),
                            "gt_bbox": json.dumps(gt_box(query, True)),
                            "model_phrase": prediction.get("model_phrase", ""),
                            "score": prediction.get("score", ""),
                            "visualization_path": str(path.relative_to(root)),
                            "manual_grade": "",
                            "reviewer": "",
                            "review_status": "pending",
                            "annotation_note": "",
                            "coordinate_space": "original_image",
                        }
                    )
                    if prediction.get("mask") is not None:
                        mask_path = (
                            output / "region_masks" / f'{track}_{query["query_id"]}.png'
                        )
                        mask_path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(
                            (np.asarray(prediction["mask"]) > 0).astype("uint8") * 255
                        ).save(mask_path)
                if not stage and b.get("status") != "detected":
                    stage = "GROUNDING_MISS"
            if attr_by_sample[sid]:
                first_a = None
                for pass_id in range(config["timed_passes"]):
                    prediction, elapsed = backend.attributes(
                        image, controlled, {}, ATTRIBUTES, controlled=True
                    )
                    timing.append(
                        {
                            "module": "attributes",
                            "track": "A",
                            "unit": "instance",
                            "source_image": source,
                            "sample_id": sid,
                            "timed_pass": pass_id + 1,
                            "ms": elapsed,
                        }
                    )
                    if first_a is None:
                        first_a = prediction
                for truth_attr in attr_by_sample[sid]:
                    name = truth_attr["attribute_name"]
                    b = (
                        item["attributes"].get(name, {"status": "missing", "label": ""})
                        if item
                        else {"status": "segmentation_missing", "label": ""}
                    )
                    for track, prediction in [
                        ("A", first_a.get(name, {"status": "missing", "label": ""})),
                        ("B", b),
                    ]:
                        attr_predictions.append(
                            {
                                "track": track,
                                "sample_id": sid,
                                "attribute_name": name,
                                "garment_class_correct": (
                                    True if track == "A" else class_ok
                                ),
                                **prediction,
                            }
                        )
                    if not stage and b.get("status") != "predicted":
                        stage = (
                            "GROUNDING_MISS"
                            if b.get("status") == "localization_missing"
                            else "ATTRIBUTE_MISSING"
                        )
                    elif not stage and b.get("label") != truth_attr["gt_label"]:
                        stage = "ATTRIBUTE_WRONG_LABEL"
            chain_rows.append(
                {
                    "sample_id": sid,
                    "source_image": source,
                    "gt_class": truth["garment_category"],
                    "localized_bbox50": bool(match),
                    "class_correct": class_ok,
                    "initial_failure_stage": stage or "REQUIRES_SEMANTIC_REVIEW",
                    "required_queries": json.dumps(
                        [q["query_id"] for q in ground_by_sample[sid]]
                    ),
                    "required_attributes": json.dumps(
                        [a["attribute_name"] for a in attr_by_sample[sid]]
                    ),
                    "matched_predicted_instance": (
                        item["roi"].instance_id if item else ""
                    ),
                }
            )
        print(
            f"Core {number}/{len(by_source)}; evaluated downstream for {len(chosen)} GT cases",
            flush=True,
        )
    micro = detection_summary(seg_rows, total_pred)
    per_class = {
        c: detection_summary(
            [r for r in seg_rows if r["gt_class"] == c], pred_by_class[c]
        )
        for c in CLASS_TO_ID
    }
    segmentation = {
        "micro": micro,
        "per_class": per_class,
        "macro": macro_summary(per_class),
        "failure_counts": dict(
            Counter(
                code for row in seg_rows for code in json.loads(row["failure_codes"])
            )
        ),
        "prediction_failure_counts": dict(
            Counter(row["failure_code"] for row in pred_rows if row["failure_code"])
        ),
    }
    write_json(output / "segmentation_metrics.json", segmentation)
    write_csv(output / "segmentation_per_gt.csv", seg_rows)
    write_csv(output / "segmentation_predictions.csv", pred_rows)
    write_csv(output / "timing.csv", timing)
    write_csv(output / "attribute_predictions.csv", attr_predictions)
    write_csv(output / "grounding_predictions.csv", ground_rows)
    write_csv(output / "grounding_semantic_audit_template.csv", ground_rows)
    write_csv(output / "end_to_end_cases.csv", chain_rows)
    write_csv(
        output / "failed_calls.csv",
        failed_calls,
        fields=["source_image", "timed_pass", "instance_id", "stage", "target"],
    )
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "Pillow")}
    for name in ("torch", "torchvision", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    run_meta = {
        "benchmark_version": meta["benchmark_version"],
        "benchmark_freeze_sha256": digest(frozen / "freeze_meta.json"),
        "checkpoint_sha256": (
            digest(checkpoint) if checkpoint.is_file() else "injected_test_backend"
        ),
        "config": config,
        "precision_mode": "float32",
        "batch_size": 1,
        "timing_scope": "preprocess+forward+postprocess",
        "software": versions,
        "run_id": hashlib.sha256(
            (digest(frozen / "freeze_meta.json") + str(time.time_ns())).encode()
        ).hexdigest(),
        "device": str(getattr(backend, "device", "test_backend")),
        "review_status": "SEMANTIC_REVIEW_REQUIRED",
        "model_resolved_revisions": {
            name: (
                getattr(getattr(backend, name, None), "config", None)._commit_hash
                if getattr(getattr(backend, name, None), "config", None) is not None
                and hasattr(getattr(backend, name).config, "_commit_hash")
                else None
            )
            for name in ("ground_model", "sam_model", "clip_model")
        },
    }
    torch = getattr(backend, "torch", None)
    run_meta["gpu"] = (
        torch.cuda.get_device_name()
        if torch is not None and getattr(backend.device, "type", "") == "cuda"
        else "none"
    )
    run_meta["artifact_sha256"] = {
        name: digest(output / name)
        for name in (
            "segmentation_metrics.json",
            "timing.csv",
            "attribute_predictions.csv",
            "grounding_predictions.csv",
            "end_to_end_cases.csv",
        )
    }
    write_json(output / "run_metadata.json", run_meta)
    from fashion_multimodal_analysis.analysis.summarize_prd31_acceptance import (
        build_semantic_review_page,
        summarize,
    )

    build_semantic_review_page(root, output)
    return summarize(root, output)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir")
    args = p.parse_args()
    print(
        json.dumps(
            evaluate(
                report_dir=resolve_path(args.report_dir) if args.report_dir else None
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
