"""Compare full image, raw garment crop, and masked crop with fixed DINO queries.

Uses two already predicted development ROIs, never GT boxes. No thresholds,
weights, training data or historical reports are changed. Does not grade accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import zipfile

from PIL import Image, ImageDraw


def source_box(box, origin):
    x, y = origin
    a, b, c, d = map(float, box)
    return [a + x, b + y, c + x, d + y]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from fashion_multimodal_analysis.common.paths import project_root, resolve_path
    from fashion_multimodal_analysis.integration.pipeline import crop_bounds
    from fashion_multimodal_analysis.integration.region_semantics import REGION_PROMPTS
    from fashion_multimodal_analysis.grounding.refinement.run_shoulder_waist_pattern_groundingdino_refinement_v1 import run_grounding_dino

    root = project_root()
    base = root / "reports/reruns/recovery_20261008"
    if args.smoke_dir:
        run = resolve_path(args.smoke_dir)
    else:
        runs = [p for p in base.glob("prd31_smoke_*")
                if (p / "summary.json").is_file()
                and json.loads((p / "run_config.json").read_text()).get("strict_region_grounding")]
        if not runs:
            raise RuntimeError("No completed strict-region smoke run found")
        run = max(runs, key=lambda p: p.stat().st_mtime)
    config = json.loads((run / "run_config.json").read_text())
    cases = {c["case_dir"]: c for c in json.loads((run / "cases.json").read_text())}
    chosen = []
    for case_name, category in [("case_01", "top"), ("case_05", "dress")]:
        predictions = json.loads((run / case_name / "predictions.json").read_text())
        candidates = [i for i in predictions["instances"] if i["category"] == category]
        if not candidates:
            raise RuntimeError(f"Missing expected predicted ROI: {case_name}:{category}")
        chosen.append((case_name, max(candidates, key=lambda i: i["score"])))
    output = base / ("grounding_input_diagnostic_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    processor = AutoProcessor.from_pretrained(config["grounding_model"], local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(config["grounding_model"], local_files_only=True).to(device).eval()
    results = []
    for case_name, instance in chosen:
        source = resolve_path(cases[case_name]["source_image"])
        with Image.open(source) as raw:
            image = raw.convert("RGB")
        bounds = crop_bounds(instance["bbox"], image.size)
        mask_path = run / case_name / instance["mask_path"]
        with Image.open(mask_path) as raw:
            mask = raw.convert("L")
        if mask.size != image.size:
            raise ValueError("Prediction mask is not in original-image coordinates")
        crop = image.crop(bounds)
        isolated = Image.composite(crop, Image.new("RGB", crop.size, "white"), mask.crop(bounds))
        views = [("full_image", image, (0, 0)),
                 ("raw_crop", crop, tuple(bounds[:2])),
                 ("masked_crop", isolated, tuple(bounds[:2]))]
        for view_name, view, origin in views:
            for region in ["collar", "cuff", "hem"]:
                prompt = REGION_PROMPTS[region]
                detections, seconds = run_grounding_dino(
                    view, prompt, config["grounding_box_threshold"],
                    config["grounding_text_threshold"], processor, model, device)
                canvas = view.copy()
                draw = ImageDraw.Draw(canvas)
                for number, det in enumerate(detections[:8], 1):
                    box = det["box"]
                    draw.rectangle(box, outline="red", width=3)
                    draw.text((box[0], max(0, box[1])), f"{number}:{det['label']} {det['score']:.2f}", fill="red")
                canvas.thumbnail((1200, 1200))
                name = f"{case_name}_{view_name}_{region}.jpg"
                canvas.save(output / name, quality=90)
                results.append({"case_dir": case_name, "instance_id": instance["instance_id"],
                                "predicted_category": instance["category"], "source_image": str(source),
                                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                "view": view_name, "region": region, "prompt": prompt,
                                "view_size": list(view.size), "crop_origin": list(origin),
                                "garment_bbox": instance["bbox"], "visualization": name,
                                "detections": [dict(d, original_image_bbox=source_box(d["box"], origin)) for d in detections],
                                "dino_model_ms": seconds * 1000})
                write_json(output / "results.json", results)
                print(f"{case_name} {view_name} {region}: {len(detections)} proposals", flush=True)
    write_json(output / "run_metadata.json", {
        "source_smoke_run": str(run), "model": config["grounding_model"],
        "model_revision": getattr(model.config, "_commit_hash", None),
        "box_threshold": config["grounding_box_threshold"],
        "text_threshold": config["grounding_text_threshold"],
        "calls": len(results), "sam_used": False, "accuracy_evaluated": False,
        "note": "All DINO candidates are preserved, including wrong phrases. Full-image proposals are not automatically assigned to the selected garment. Timings exclude processing and include first-call effects."})
    target = root.parent / "grounding_input_diagnostic_20261008.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    print("DIAGNOSTIC_COMPLETED", flush=True)
    print("REPORT_DIR:", output, flush=True)
    print("UPLOAD_ZIP:", target, flush=True)


if __name__ == "__main__":
    main()
