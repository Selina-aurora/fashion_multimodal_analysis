#!/usr/bin/env python3
"""PRD 3.2 W7 Dev7: reference-candidate input A/B developer regression.

Two image conditions per case: full image with drawn candidate bbox vs candidate
ROI. Ask for unguided category/visible attributes. This is NOT GT evaluation,
not frozen 3.1 model-prediction benchmarking, and not PRD acceptance evidence.
No network/model download, no sealed holdout, no overwriting existing output.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_IDS = {"dev_02", "dev_04", "dev_05", "dev_06", "dev_08", "dev_09", "dev_12"}
FIELDS = {"category", "primary_color", "pattern", "visible_design"}
CATEGORIES = {"top", "pants", "skirt", "outerwear", "dress", "shoe", "bag", "accessory"}
BAD_DESIGNS = {"无", "无明显设计", "无设计", "无图案", "无装饰", "无明显装饰", "none", "n/a", "unknown"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_restricted(path) -> bool:
    return any(t in str(path).lower() for t in ("sealed", "holdout"))


def check_manifest(path: Path, max_images: int):
    if is_restricted(path):
        raise ValueError("Sealed/holdout manifest is forbidden")
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    ids = [r.get("image_id") for r in rows]
    if len(rows) != 7 or set(ids) != REQUIRED_IDS or len(set(ids)) != 7:
        raise ValueError("Expected seven unique Dev7 image IDs")
    selected = rows[:max_images]
    for row in selected:
        if (row.get("status") != "REFERENCE_CANDIDATE_VISUAL_REVIEW_REQUIRED"
            or row.get("geometry_valid", "").lower() != "true"
            or row.get("identity_errors", "").strip()):
            raise ValueError("Candidate status/identity invalid: " + row["image_id"])
        if row.get("bbox_visually_confirmed_for_prd32", "").lower() != "false":
            raise ValueError("Candidate provenance changed; review before running: " + row["image_id"])
        if row.get("eligible_for_accuracy_scoring", "").lower() != "false":
            raise ValueError("GT state changed, verify manifest separately")
        img = Path(row["image_path"])
        if is_restricted(img) or not img.is_file():
            raise ValueError("Missing or prohibited image: " + str(img))
        if digest(img) != row["image_sha256"]:
            raise ValueError("SHA256 mismatch: " + row["image_id"])
        box = json.loads(row["reference_bbox_xyxy"])
        if not (isinstance(box, list) and len(box) == 4 and
                all(type(v) in (float, int) and math.isfinite(v) for v in box)):
            raise ValueError("Invalid ROI coordinates: " + row["image_id"])
        from PIL import Image
        with Image.open(img) as original:
            w, h = original.size
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError("ROI outside image: " + row["image_id"])
    return selected


def parse_output(raw: str) -> dict:
    txt = raw.strip()
    native = txt.startswith("{") and txt.endswith("}")
    candidate = txt
    if re.match(r"^```(?:json)?\s*\n", txt, flags=re.I) and txt.endswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*\n", "", txt, count=1, flags=re.I)[:-3].strip()
    errors = []
    try:
        obj = json.loads(candidate)
    except (ValueError, TypeError) as exc:
        return {"native_json": native, "parseable": False, "schema_valid": False,
                "parsed": None, "issues": ["json_parse_error: " + str(exc)]}
    if not isinstance(obj, dict) or set(obj) != FIELDS:
        errors.append("keys_mismatch")
    else:
        if obj["category"] is not None and (not isinstance(obj["category"], str) or obj["category"] not in CATEGORIES):
            errors.append("category_not_in_8_classes")
        for key in ("primary_color", "pattern", "visible_design"):
            if obj[key] is not None and not isinstance(obj[key], str):
                errors.append("invalid_type:" + key)
        design = obj.get("visible_design")
        if isinstance(design, str) and design.strip().lower() in BAD_DESIGNS:
            errors.append("non_structural_visible_design")
        if isinstance(obj.get("pattern"), str) and any(
            x in obj["pattern"].lower() for x in ("蕾丝", "lace", "针织", "棉", "polyester")
        ):
            errors.append("pattern_ontology_review_required")
    return {"native_json": native, "parseable": True,
            "schema_valid": isinstance(obj, dict) and set(obj) == FIELDS and not any(
                e.startswith("invalid_type") or e in ("keys_mismatch", "category_not_in_8_classes") for e in errors),
            "parsed": obj, "issues": errors}


def prompt_for(mode: str):
    aim = "红框所指的一件目标服饰" if mode == "marked_full_image" else "裁剪图里居中的目标服饰"
    return (f"只识别{aim}。框或裁剪范围可能包含内搭、背包及其他服饰，不能把它们的属性算到目标上。"
            "请根据图像自行判断类别，不能依赖人工指定类别。"
            "只输出单个JSON对象，严格只含category、primary_color、pattern、visible_design四个键，不要Markdown。"
            "category仅为top,pants,skirt,outerwear,dress,shoe,bag,accessory之一；无法判断填null。"
            "颜色、图案、结构设计各用一个简短中文短语，无法确认填null。"
            "visible_design仅描述目标自身可见的肩带、袖型、领口、门襟或纽扣等结构，不得填‘无’或‘无明显设计’。"
            "pattern只描述目标本身的可见印花、格纹、条纹、纯色等；不要把面料成分或工艺当作图案。"
            "不要猜品牌、材料、功能或主观风格。")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="/workspace/PRD32_W7_DEV7_SCREEN_V4_20261026/dev7_candidate_manifest.csv")
    ap.add_argument("--model-id", default="/workspace/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--output", default="/workspace/PRD32_W7_DEV7_BATCH_V1_20261026")
    ap.add_argument("--limit", type=int, default=7)
    ap.add_argument("--mode", choices=["both", "roi", "marked"], default="both")
    ap.add_argument("--max-pixels", type=int, default=401408)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.limit <= 7:
        ap.error("--limit must be between 1 and 7")
    manifest, output, model_dir = Path(args.manifest), Path(args.output), Path(args.model_id)
    if any(is_restricted(p) for p in (manifest, output, model_dir)):
        ap.error("Sealed/holdout paths are forbidden")
    rows = check_manifest(manifest, args.limit)
    print(f"PREFLIGHT_OK: {len(rows)} input SHA256 verified; reference candidate bbox only, not GT", flush=True)
    if args.dry_run:
        print("DRY_RUN_OK: no model loaded and no files written", flush=True)
        return
    if not model_dir.is_dir():
        raise FileNotFoundError("Local model directory not found: " + str(model_dir))
    if output.exists():
        raise FileExistsError("Will not overwrite existing results: " + str(output))

    from PIL import Image, ImageDraw
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    modes = (["marked_full_image", "reference_roi"] if args.mode == "both"
             else ["reference_roi"] if args.mode == "roi" else ["marked_full_image"])
    output.mkdir(parents=True)
    shutil.copy2(manifest, output / "source_manifest.csv")
    imgs = []
    for row in rows:
        with Image.open(row["image_path"]) as raw:
            img = raw.convert("RGB")
        box = json.loads(row["reference_bbox_xyxy"])
        x1, y1, x2, y2 = box
        marked = img.copy()
        ImageDraw.Draw(marked).rectangle(tuple(map(round, box)), outline=(255, 45, 35),
                                         width=max(3, min(img.size) // 150))
        marked_path = output / f'{row["image_id"]}_marked.jpg'
        marked.save(marked_path, quality=92)
        pad_x, pad_y = .04 * (x2-x1), .04 * (y2-y1)
        crop_box = (max(0, math.floor(x1-pad_x)), max(0, math.floor(y1-pad_y)),
                    min(img.width, math.ceil(x2+pad_x)), min(img.height, math.ceil(y2+pad_y)))
        roi_path = output / f'{row["image_id"]}_reference_roi.jpg'
        img.crop(crop_box).save(roi_path, quality=92)
        imgs.append((row, {"marked_full_image": marked_path, "reference_roi": roi_path}))

    print("Loading local Qwen2.5-VL-7B once (network disabled by local_files_only)...", flush=True)
    load_start = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True, device_map="auto", dtype=torch.bfloat16,
        attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(
        str(model_dir), local_files_only=True, max_pixels=args.max_pixels)
    load_seconds = round(time.perf_counter() - load_start, 3)

    def infer(image_path, text):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path.resolve().as_uri()},
            {"type": "text", "text": text}]}]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        pics, vids = process_vision_info(messages)
        batch = processor(text=[formatted], images=pics, videos=vids,
                          padding=True, return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        raw_response = processor.batch_decode(ids[:, batch.input_ids.shape[1]:],
                                              skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0]
        return raw_response, {"end_to_end_ms": round((t2-t0)*1000, 2),
                              "preprocess_ms": round((t1-t0)*1000, 2),
                              "generation_ms": round((t2-t1)*1000, 2),
                              "generated_tokens": int(ids.shape[-1] - batch.input_ids.shape[-1])}

    # Warm-up is deliberately excluded from per-sample latency; no GT scoring.
    print("Warming up model...", flush=True)
    infer(imgs[0][1][modes[0]], "描述图中能看到的一件服饰，只输出一个词。")
    results = []
    print("Running W7 development A/B...", flush=True)
    for row, paths in imgs:
        for mode in modes:
            raw, timing = infer(paths[mode], prompt_for(mode))
            parsed = parse_output(raw)
            record = {
                "image_id": row["image_id"], "image_sha256": row["image_sha256"],
                "input_variant": mode, "image_sent": str(paths[mode]),
                "reference_category_prior_review_not_scoring_gt": row["reference_category_prior_review_not_scoring_gt"],
                "reference_bbox_source": row["reference_bbox_provenance"],
                "reference_bbox_visually_confirmed": False,
                "input_identity_verified": True,
                "raw_response": raw, **parsed, **timing,
                "accuracy": None, "human_attribute_gt_verified": False,
                "is_prd_acceptance": False,
            }
            results.append(record)
            print(f'{row["image_id"]} / {mode}: {timing["end_to_end_ms"]:.0f} ms; '
                  f'parsed={parsed["parseable"]}; issues={parsed["issues"]}', flush=True)

    with (output / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    timing = [r["end_to_end_ms"] for r in results]
    by_id = {}
    for r in results:
        by_id.setdefault(r["image_id"], {})[r["input_variant"]] = r.get("parsed")
    view_disagreements = []
    for case, variants in by_id.items():
        if len(variants) == 2 and variants.get("marked_full_image") != variants.get("reference_roi"):
            view_disagreements.append(case)
    summary = {
        "status": "W7_DEV7_DIAGNOSTIC_COMPLETED_NOT_PRD_ACCEPTED",
        "image_count": len(rows), "inference_calls": len(results),
        "input_variants": modes, "source": str(manifest), "model": str(model_dir),
        "reference_bbox_type": "historical_human_review_candidate_not_gt",
        "model_load_seconds": load_seconds,
        "median_latency_ms": round(statistics.median(timing), 2),
        "mean_latency_ms": round(statistics.mean(timing), 2),
        "json_parseable_count": sum(r["parseable"] for r in results),
        "native_json_count": sum(r["native_json"] for r in results),
        "schema_valid_count": sum(r["schema_valid"] for r in results),
        "issues": dict(Counter(i for r in results for i in r["issues"])),
        "cross_view_disagreement_ids": view_disagreements,
        "human_attribute_gt_verified": False,
        "classification_accuracy": None,
        "color_accuracy": None,
        "qa_accuracy": None,
        "copy_satisfaction": None,
        "sealed_holdout_accessed": False,
        "frozen_3_1_modified": False,
        "limitations": "Reference bbox is prior human-review candidate, not frozen 3.1 predicted ROI. 7 development images, not eight-class representative acceptance. No human attribute GT, no QA or copy acceptance."
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["image_id", "input_variant", "reference_category_not_gt", "pred_category",
              "pred_primary_color", "pred_pattern", "pred_visible_design",
              "parseable", "native_json", "schema_valid", "issues", "latency_ms",
              "human_review_status", "human_category_gt", "human_color_gt",
              "human_pattern_gt", "human_design_gt", "bbox_review_status", "review_notes"]
    with (output / "review_sheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            p = r["parsed"] if isinstance(r["parsed"], dict) else {}
            writer.writerow({
                "image_id": r["image_id"], "input_variant": r["input_variant"],
                "reference_category_not_gt": r["reference_category_prior_review_not_scoring_gt"],
                "pred_category": p.get("category"), "pred_primary_color": p.get("primary_color"),
                "pred_pattern": p.get("pattern"), "pred_visible_design": p.get("visible_design"),
                "parseable": r["parseable"], "native_json": r["native_json"],
                "schema_valid": r["schema_valid"], "issues": ";".join(r["issues"]),
                "latency_ms": r["end_to_end_ms"], "human_review_status": "PENDING",
                "human_category_gt": "", "human_color_gt": "", "human_pattern_gt": "",
                "human_design_gt": "", "bbox_review_status": "CANDIDATE_NOT_GT", "review_notes": ""
            })
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("RESULTS:", output, flush=True)

if __name__ == "__main__":
    main()
