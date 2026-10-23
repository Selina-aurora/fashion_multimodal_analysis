"""Small GT-free prediction-chain check; does not perform PRD acceptance.

Select one distinct development image per available garment class. Labels are
used only for image selection, never to filter or replace predicted instances.
Run from the project root; original configs, manifests and checkpoints are read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def box_valid(box, width, height):
    if box is None or len(box) != 4:
        return False
    a = np.asarray(box, dtype=float)
    return bool(np.isfinite(a).all() and 0 <= a[0] < a[2] <= width
                and 0 <= a[1] < a[3] <= height)


def save_prediction(image, result, folder):
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
        record = {"instance_id": roi.instance_id, "category": roi.category,
                  "score": float(roi.score), "bbox": list(map(float, roi.bbox)),
                  "provenance": roi.provenance, "mask_path": mask_name,
                  "attributes": item["attributes"], "regions": {}}
        if box_valid(roi.bbox, image.width, image.height):
            draw.rectangle(roi.bbox, outline="red", width=3)
            draw.text((roi.bbox[0], roi.bbox[1]),
                      f"{roi.instance_id} {roi.category}", fill="red")
        for region, prediction in item["regions"].items():
            pred = dict(prediction)
            array = pred.pop("mask", None)
            if pred.get("status") == "detected":
                valid = box_valid(pred.get("bbox"), image.width, image.height)
                if not valid:
                    checks.append(f"invalid_region_bbox:{roi.instance_id}:{region}")
                if pred.get("coordinate_space") != "original_image":
                    checks.append(f"invalid_coordinate_space:{roi.instance_id}:{region}")
                if array is None:
                    checks.append(f"missing_region_mask:{roi.instance_id}:{region}")
                else:
                    part = np.asarray(array) > 0
                    if part.shape != mask.shape or part.shape != (image.height, image.width):
                        checks.append(f"invalid_region_mask_shape:{roi.instance_id}:{region}")
                    elif np.any(part & ~mask):
                        checks.append(f"region_mask_outside_instance:{roi.instance_id}:{region}")
                if valid:
                    draw.rectangle(pred["bbox"], outline="lime", width=2)
                    draw.text((pred["bbox"][0], pred["bbox"][1]), region, fill="lime")
            if array is not None:
                name = f"instance_{index:03d}_{region}_mask.png"
                Image.fromarray((np.asarray(array) > 0).astype("uint8") * 255).save(folder / name)
                pred["mask_path"] = name
            record["regions"][region] = pred
        neckline = record["attributes"].get("neckline", {})
        if neckline.get("status") == "predicted" and item["regions"].get("collar", {}).get("status") != "detected":
            checks.append(f"neckline_without_predicted_collar:{roi.instance_id}")
        records.append(record)
    canvas.thumbnail((1400, 1400))
    canvas.save(folder / "review.jpg", quality=90)
    payload = {k: v for k, v in result.items() if k != "instances"}
    payload.update(instances=records, interface_issues=checks)
    dump(folder / "predictions.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from fashion_multimodal_analysis.common.paths import project_root, resolve_path
    from fashion_multimodal_analysis.integration.pipeline import FashionPipeline
    from fashion_multimodal_analysis.segmentation.training.checkpoint_state import CONTROLLED_SETTINGS

    root = project_root()
    config = json.loads((root / "configs/prd_3_1_recovery_v2.json").read_text(encoding="utf-8"))
    config.update(
        segmentation_checkpoint="outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth",
        train_manifest="configs/prd_8class_train_v5_balanced150_20261008.csv",
        val_manifest="configs/prd_8class_val_v2_full_targets.csv",
        checkpoint_settings={**CONTROLLED_SETTINGS, "workers": 2},
        strict_region_grounding=True,
    )
    # Require existing snapshots; the smoke run must not trigger more downloads.
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    with resolve_path(config["val_manifest"]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = []
    used = set()
    for category in config["required_prd_classes"]:
        row = next((r for r in rows if r["garment_category"] == category
                    and str(resolve_path(r["source_image"])) not in used), None)
        if row:
            path = resolve_path(row["source_image"])
            if not path.is_file():
                raise FileNotFoundError(path)
            used.add(str(path))
            selected.append({"source_image": str(path), "selection_category": category})
    if not selected:
        raise RuntimeError("No development images selected")
    for key in ("segmentation_checkpoint", "train_manifest", "val_manifest"):
        if not resolve_path(config[key]).is_file():
            raise FileNotFoundError(resolve_path(config[key]))
    print("Selected development images:", json.dumps(selected, ensure_ascii=False), flush=True)
    if args.dry_run:
        print("DRY_RUN_PASSED: no models loaded, no inference started", flush=True)
        return
    output = resolve_path(args.report_dir) if args.report_dir else root / "reports/reruns/recovery_20261008" / ("prd31_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "run_config.json", config)
    dump(output / "selected_images.json", selected)
    print("REPORT_DIR:", output, flush=True)
    from fashion_multimodal_analysis.integration.model_backends import ModelBackends
    pipeline = FashionPipeline(ModelBackends(config, device="cuda"))
    cases = []
    region_status_counts = Counter()
    attribute_status_counts = Counter()
    for number, selected_case in enumerate(selected, 1):
        print(f"Running image {number}/{len(selected)}", flush=True)
        with Image.open(selected_case["source_image"]) as raw:
            image = raw.convert("RGB")
        start = time.perf_counter()
        result = pipeline.run(image, config["required_regions"], config["required_attributes"])
        elapsed = (time.perf_counter() - start) * 1000
        payload = save_prediction(image, result, output / f"case_{number:02d}")
        for instance in payload["instances"]:
            for name, prediction in instance["regions"].items():
                region_status_counts[f"{name}:{prediction['status']}"] += 1
            for name, prediction in instance["attributes"].items():
                attribute_status_counts[f"{name}:{prediction['status']}"] += 1
        cases.append(dict(selected_case, case_dir=f"case_{number:02d}",
                          pipeline_status=result["status"], instance_count=len(result["instances"]),
                          failures=result["failures"], interface_issues=payload["interface_issues"],
                          chain_wall_ms=elapsed, error=result.get("error")))
        dump(output / "cases.json", cases)
        print(f"Finished image {number}: {len(result['instances'])} predicted instances; "
              f"{len(result['failures'])} failed calls; {len(payload['interface_issues'])} interface issues", flush=True)
    summary = {"status": "smoke_completed", "images": len(cases),
               "region_status_counts": dict(region_status_counts),
               "attribute_status_counts": dict(attribute_status_counts),
               "predicted_instances": sum(c["instance_count"] for c in cases),
               "pipeline_status_counts": dict(Counter(c["pipeline_status"] for c in cases)),
               "failed_downstream_calls": sum(len(c["failures"]) for c in cases),
               "interface_issues": sum(len(c["interface_issues"]) for c in cases),
               "images_with_no_predictions": sum(c["instance_count"] == 0 for c in cases),
               "accuracy_evaluated": False, "prd_acceptance_evaluated": False,
               "timing_note": "Single-pass diagnostic includes first-call effects; not acceptance latency. Wall time excludes image reading and saving.",
               "review_note": "Use review.jpg and predictions.json for manual semantic review. Zero predictions do not prove absence or quantify missed GT.",
               "input_sha256": {k: sha(resolve_path(config[k])) for k in ("segmentation_checkpoint", "train_manifest", "val_manifest")}}
    dump(output / "summary.json", summary)
    print("SMOKE_COMPLETED", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("REPORT_DIR:", output, flush=True)


if __name__ == "__main__":
    main()
