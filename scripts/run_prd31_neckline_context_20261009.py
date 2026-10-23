"""Rerun the same eight reviewed development images without GT overrides.

Verify image hashes before loading models. Preserve raw model classes and export
geometric guards/review flags; this run is not PRD acceptance evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def dump(path: Path, value: Any) -> None:
    """把运行记录写为UTF-8 JSON；调用方负责创建父目录。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path: Path) -> str:
    """分块读取文件并返回SHA-256，避免将大权重一次性载入内存。"""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def box_valid(box: Sequence[float] | None, width: int, height: int) -> bool:
    """检查原图xyxy框是否有限、有正面积且完全位于图像范围内。"""
    if box is None or len(box) != 4:
        return False
    a = np.asarray(box, dtype=float)
    return bool(
        np.isfinite(a).all()
        and 0 <= a[0] < a[2] <= width
        and 0 <= a[1] < a[3] <= height
    )


def save_prediction(
    image: Image.Image, result: dict[str, Any], folder: Path
) -> dict[str, Any]:
    """Persist arrays as PNGs and check original-image coordinate contracts."""
    folder.mkdir()
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    checks = []
    records = []
    seen = set()
    for index, item in enumerate(result["instances"]):
        roi = item["roi"]
        mask = np.asarray(roi.mask) > 0
        if roi.instance_id in seen:
            checks.append(f"duplicate_instance_id:{roi.instance_id}")
        seen.add(roi.instance_id)
        if not box_valid(roi.bbox, image.width, image.height):
            checks.append(f"invalid_garment_bbox:{roi.instance_id}")
        if mask.shape != (image.height, image.width):
            checks.append(f"invalid_garment_mask_shape:{roi.instance_id}")
        mask_name = f"instance_{index:03d}_mask.png"
        Image.fromarray(mask.astype("uint8") * 255).save(folder / mask_name)
        record = {
            "instance_id": roi.instance_id,
            "category": roi.category,
            "score": float(roi.score),
            "bbox": list(map(float, roi.bbox)),
            "provenance": roi.provenance,
            "mask_path": mask_name,
            "attributes": item["attributes"],
            "regions": {},
            "review_flags": item.get("review_flags", []),
        }
        if box_valid(roi.bbox, image.width, image.height):
            draw.rectangle(roi.bbox, outline="red", width=3)
            draw.text(
                (roi.bbox[0], roi.bbox[1]),
                f"{roi.instance_id} {roi.category}",
                fill="red",
            )
        for region, prediction in item["regions"].items():
            pred = dict(prediction)
            array = pred.pop("mask", None)
            if pred.get("status") == "detected":
                valid = box_valid(pred.get("bbox"), image.width, image.height)
                if not valid:
                    checks.append(f"invalid_region_bbox:{roi.instance_id}:{region}")
                if pred.get("coordinate_space") != "original_image":
                    checks.append(
                        f"invalid_coordinate_space:{roi.instance_id}:{region}"
                    )
                if array is None:
                    checks.append(f"missing_region_mask:{roi.instance_id}:{region}")
                else:
                    part = np.asarray(array) > 0
                    if part.shape != mask.shape or part.shape != (
                        image.height,
                        image.width,
                    ):
                        checks.append(
                            f"invalid_region_mask_shape:{roi.instance_id}:{region}"
                        )
                    elif np.any(part & ~mask):
                        checks.append(
                            f"region_mask_outside_instance:{roi.instance_id}:{region}"
                        )
                if valid:
                    draw.rectangle(pred["bbox"], outline="lime", width=2)
                    draw.text((pred["bbox"][0], pred["bbox"][1]), region, fill="lime")
            if array is not None:
                name = f"instance_{index:03d}_{region}_mask.png"
                Image.fromarray((np.asarray(array) > 0).astype("uint8") * 255).save(
                    folder / name
                )
                pred["mask_path"] = name
            record["regions"][region] = pred
        neckline = record["attributes"].get("neckline", {})
        if (
            neckline.get("status") == "predicted"
            and item["regions"].get("collar", {}).get("status") != "detected"
        ):
            checks.append(f"neckline_without_predicted_collar:{roi.instance_id}")
        records.append(record)
    # 导出 CLIP 实际输入供审核：原掩码裁剪和新的原图上下文裁剪。
    for index, record in enumerate(records):
        neck = record["attributes"].get("neckline", {})
        if "context_crop_bbox" not in neck:
            continue
        legacy_box, context_box = neck["legacy_crop_bbox"], neck["context_crop_bbox"]
        collar = record["regions"]["collar"]
        with Image.open(folder / collar["mask_path"]) as opened_mask:
            part = opened_mask.convert("L").crop(legacy_box)
        legacy_view = Image.composite(
            image.crop(legacy_box), Image.new("RGB", part.size, "white"), part
        )
        contextual_view = image.crop(context_box)
        legacy_name = f"instance_{index:03d}_neckline_legacy.jpg"
        context_name = f"instance_{index:03d}_neckline_context.jpg"
        legacy_view.save(folder / legacy_name, quality=95)
        contextual_view.save(folder / context_name, quality=95)
        neck["legacy_view_path"] = legacy_name
        neck["context_view_path"] = context_name
    canvas.thumbnail((1400, 1400))
    canvas.save(folder / "review.jpg", quality=90)
    payload = {k: v for k, v in result.items() if k != "instances"}
    payload.update(instances=records, interface_issues=checks)
    dump(folder / "predictions.json", payload)
    return payload


def main() -> None:
    """核对已审核图像的来源哈希，运行开发集推理并保存预测、计时及接口检查。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir")
    parser.add_argument(
        "--baseline-dir",
        help="Yesterday original guard-regression directory; auto-find if omitted",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--cases-manifest",
        default="configs/prd31_review_regression_cases_20261008.json",
    )
    args = parser.parse_args()
    from fashion_multimodal_analysis.common.paths import project_root, resolve_path
    from fashion_multimodal_analysis.integration.pipeline import FashionPipeline
    from fashion_multimodal_analysis.segmentation.training.checkpoint_state import (
        CONTROLLED_SETTINGS,
    )

    root = project_root()
    config = json.loads(
        (root / "configs/prd_3_1_recovery_v2.json").read_text(encoding="utf-8")
    )
    config.update(
        segmentation_checkpoint="outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth",
        train_manifest="configs/prd_8class_train_v5_balanced150_20261008.csv",
        val_manifest="configs/prd_8class_val_v2_full_targets.csv",
        checkpoint_settings={**CONTROLLED_SETTINGS, "workers": 2},
        neckline_context_pair=True,
        neckline_context_expansion=2.0,
        strict_region_grounding=True,
        max_local_region_crop_area_ratio=0.65,
        regression_patch_version="neckline_context_pair_20261009_v1",
    )
    # Require existing snapshots; the smoke run must not trigger more downloads.
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # 固定审核过的同一批输入，防止 manifest 排序变化导致换图比较。
    reviewed = json.loads(resolve_path(args.cases_manifest).read_text(encoding="utf-8"))
    if [row["case_id"] for row in reviewed] != [f"case_{n:02d}" for n in range(1, 9)]:
        raise ValueError("Regression requires the original eight cases in order")
    selected = []
    for row in reviewed:
        path = resolve_path(row["source_image"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha(path) != row["source_sha256"]:
            raise ValueError(
                f"Source image differs from reviewed input: {row['case_id']}"
            )
        selected.append(
            {
                "source_image": str(path),
                "selection_category": row["selection_category"],
                "source_sha256": row["source_sha256"],
            }
        )
    for key in ("segmentation_checkpoint", "train_manifest", "val_manifest"):
        if not resolve_path(config[key]).is_file():
            raise FileNotFoundError(resolve_path(config[key]))
    print(
        "Selected development images:",
        json.dumps(selected, ensure_ascii=False),
        flush=True,
    )
    # 按原图及 checkpoint/manifest 哈希查找昨日基线，不以候选训练结果代替。
    from compare_neckline_context_20261009 import compare_reports, find_baseline

    baseline = find_baseline(
        root, args.baseline_dir, selected, config, sha, resolve_path
    )
    print("BASELINE_DIR:", baseline, flush=True)
    if args.dry_run:
        print("DRY_RUN_PASSED: no models loaded, no inference started", flush=True)
        return
    output = (
        resolve_path(args.report_dir)
        if args.report_dir
        else root
        / "reports/reruns/recovery_20261009"
        / ("prd31_neckline_context_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "run_config.json", config)
    dump(output / "selected_images.json", selected)
    print("REPORT_DIR:", output, flush=True)
    from fashion_multimodal_analysis.integration.model_backends import ModelBackends

    pipeline = FashionPipeline(ModelBackends(config, device="cuda"))
    cases = []
    for number, selected_case in enumerate(selected, 1):
        print(f"Running image {number}/{len(selected)}", flush=True)
        with Image.open(selected_case["source_image"]) as raw:
            image = raw.convert("RGB")
        start = time.perf_counter()
        result = pipeline.run(
            image, config["required_regions"], config["required_attributes"]
        )
        elapsed = (time.perf_counter() - start) * 1000
        payload = save_prediction(image, result, output / f"case_{number:02d}")
        cases.append(
            dict(
                selected_case,
                case_dir=f"case_{number:02d}",
                pipeline_status=result["status"],
                instance_count=len(result["instances"]),
                failures=result["failures"],
                interface_issues=payload["interface_issues"],
                chain_wall_ms=elapsed,
                error=result.get("error"),
            )
        )
        dump(output / "cases.json", cases)
        print(
            f"Finished image {number}: {len(result['instances'])} predicted instances; "
            + f"{len(result['failures'])} failed calls; {len(payload['interface_issues'])} interface issues",
            flush=True,
        )
    summary = {
        "status": "smoke_completed",
        "images": len(cases),
        "predicted_instances": sum(c["instance_count"] for c in cases),
        "pipeline_status_counts": dict(Counter(c["pipeline_status"] for c in cases)),
        "failed_downstream_calls": sum(len(c["failures"]) for c in cases),
        "interface_issues": sum(len(c["interface_issues"]) for c in cases),
        "images_with_no_predictions": sum(c["instance_count"] == 0 for c in cases),
        "review_flag_counts": dict(
            Counter(
                flag["reason"]
                for case in cases
                for instance in json.loads(
                    (output / case["case_dir"] / "predictions.json").read_text()
                )["instances"]
                for flag in instance.get("review_flags", [])
            )
        ),
        "regression_patch_version": config["regression_patch_version"],
        "accuracy_evaluated": False,
        "prd_acceptance_evaluated": False,
        "timing_note": "Single-pass diagnostic includes first-call effects; not acceptance latency. Wall time excludes image reading and saving.",
        "review_note": "Use review.jpg and predictions.json for manual semantic review. Zero predictions do not prove absence or quantify missed GT.",
        "input_sha256": {
            k: sha(resolve_path(config[k]))
            for k in ("segmentation_checkpoint", "train_manifest", "val_manifest")
        },
    }
    dump(output / "summary.json", summary)
    print(
        "SMOKE_COMPLETED", json.dumps(summary, ensure_ascii=False, indent=2), flush=True
    )
    comparison = compare_reports(baseline, output)
    dump(output / "baseline_comparison.json", comparison)
    print("COMPARISON:", json.dumps(comparison, ensure_ascii=False), flush=True)
    # 导出本轮原图和全部结果，复核时不再需要另找图片。
    import shutil
    import zipfile

    source_folder = output / "source_images"
    source_folder.mkdir()
    for number, selected_case in enumerate(selected, 1):
        shutil.copyfile(
            selected_case["source_image"], source_folder / f"case_{number:02d}.jpg"
        )
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(output))
    print("REPORT_DIR:", output, flush=True)
    print("UPLOAD_THIS_ZIP:", archive, flush=True)


if __name__ == "__main__":
    main()
