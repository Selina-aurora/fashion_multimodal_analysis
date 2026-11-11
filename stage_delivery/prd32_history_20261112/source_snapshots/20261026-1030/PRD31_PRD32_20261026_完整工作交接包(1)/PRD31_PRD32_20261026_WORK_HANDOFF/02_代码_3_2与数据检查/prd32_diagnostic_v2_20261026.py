#!/usr/bin/env python3
"""PRD 3.2 development-only ROI/attribute diagnostic for frozen 3.1 handoff.

Creates new outputs only. No model training, no changes to PRD 3.1 artifacts,
no sealed holdout access. This is NOT PRD accuracy, latency, or QPS acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_FIELDS = {"category", "primary_color", "pattern", "visible_design", "uncertain_fields"}
CATEGORIES = {"top", "pants", "skirt", "outerwear", "dress", "shoe", "bag", "accessory", None}


def make_context(attrs: object) -> str:
    """Retain provenance but never turn upstream predictions into ground truth."""
    compact = []
    for x in attrs if isinstance(attrs, list) else []:
        if not isinstance(x, dict):
            continue
        name = x.get("name")
        val = x.get("predicted_value")
        if x.get("status") != "predicted" or not isinstance(name, str) or not isinstance(val, str):
            continue
        compact.append({"field": name, "unverified_prediction": val})
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def validate_json(raw: str) -> dict:
    candidate = raw.strip()
    strict = candidate.startswith("{") and candidate.endswith("}") and "```" not in candidate
    obj = None
    error = None
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        error = str(exc)
        if candidate.startswith("```json") or candidate.startswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                try:
                    obj = json.loads("\n".join(lines[1:-1]))
                except json.JSONDecodeError:
                    pass
    parseable = isinstance(obj, dict)
    schema_ok = (
        parseable
        and REQUIRED_FIELDS.issubset(obj)
        and obj.get("category") in CATEGORIES
        and (obj.get("primary_color") is None or isinstance(obj.get("primary_color"), str))
        and (obj.get("pattern") is None or isinstance(obj.get("pattern"), str))
        and (obj.get("visible_design") is None or isinstance(obj.get("visible_design"), str))
        and isinstance(obj.get("uncertain_fields"), list)
        and all(isinstance(z, str) for z in obj["uncertain_fields"])
    )
    return {
        "strict_json_syntax": bool(strict and parseable),
        "json_parseable_even_if_code_fence": parseable,
        "schema_valid_v2": bool(schema_ok),
        "parse_error": None if parseable else error,
        "parsed_json": obj,
        "note": "Format/schema validity is not attribute accuracy or human satisfaction.",
    }


def pctl(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    return s[low] + (s[high] - s[low]) * (pos - low)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bridge-input", default="/workspace/prd31_stage_freeze_20261026/bridge_inputs.jsonl")
    p.add_argument("--case-index", type=int, default=0)
    p.add_argument("--model-id", default="/workspace/models/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--output", default="/workspace/PRD32_DIAG_V2_20261026")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--max-pixels", type=int, default=401408)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--roi-padding", type=float, default=0.06)
    args = p.parse_args()
    if not (1 <= args.repeats <= 5):
        p.error("--repeats must be between 1 and 5")
    if not (0 <= args.roi_padding <= 0.3):
        p.error("--roi-padding must be between 0 and 0.3")
    bridge = Path(args.bridge_input).resolve(strict=True)
    lines = [json.loads(v) for v in bridge.read_text(encoding="utf-8").splitlines() if v.strip()]
    if not 0 <= args.case_index < len(lines):
        p.error(f"case-index must be 0..{len(lines)-1}")
    case = lines[args.case_index]
    if case.get("schema") != "prd32_engine_neutral_input_v1":
        raise ValueError("Unknown bridge input schema")
    src = Path(case["image_path"]).resolve(strict=True)
    if any(w in str(src).lower() for w in ("holdout", "sealed")):
        raise ValueError("Sealed/holdout images are not permitted")
    if src.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        raise ValueError("Image path must be a recognized raster image")
    if hashlib.sha256(src.read_bytes()).hexdigest() != case["image_sha256"]:
        raise ValueError("3.1 frozen bridge image hash mismatch")
    bbox = case.get("target_bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox):
        raise ValueError("Bridge has no valid predicted target bbox")

    from PIL import Image
    with Image.open(src) as original:
        rgb = original.convert("RGB")
    w, h = rgb.size
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("Predicted bbox outside source image")
    dx, dy = (x2 - x1) * args.roi_padding, (y2 - y1) * args.roi_padding
    box = (max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
           min(w, math.ceil(x2 + dx)), min(h, math.ceil(y2 + dy)))
    if box[2] - box[0] < 24 or box[3] - box[1] < 24:
        raise ValueError("Predicted ROI too small to diagnose fairly")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"diagnostic_case_{args.case_index:02d}.json"
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite previous diagnostic: {dest}")
    roi_path = out / f"predicted_roi_case_{args.case_index:02d}.jpg"
    if roi_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior crop: {roi_path}")
    rgb.crop(box).save(roi_path, quality=96)

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    free, _ = torch.cuda.mem_get_info()
    if free < 18 * 1024**3:
        raise RuntimeError(f"Need >=18 GiB free for isolated BF16 loading; got {free / 1024**3:.2f}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    local = Path(args.model_id).is_dir()
    print("[1/4] Load local Qwen2.5-VL-7B (3.1 unchanged)...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=dtype, device_map="auto", attn_implementation="sdpa",
        low_cpu_mem_usage=True, local_files_only=local,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model_id, min_pixels=128 * 28 * 28, max_pixels=args.max_pixels,
        local_files_only=local,
    )
    torch.cuda.reset_peak_memory_stats()

    context = make_context(case.get("predicted_attribute_context"))
    original_query = str(case.get("query") or "这件服饰的主要颜色是什么？").replace("框中目标服饰", "目标服饰")
    # Image-only condition isolates the VLM from low-confidence upstream predictions.
    variants = [
        {"id": "full_image_only", "image": src, "context": None,
         "caveat": "Entire image, no target marking: multi-instance identity may be ambiguous."},
        {"id": "predicted_roi_only", "image": roi_path, "context": None,
         "caveat": "ROI from frozen 3.1 prediction; wrong upstream box propagates."},
        {"id": "predicted_roi_plus_unverified_attrs", "image": roi_path, "context": context,
         "caveat": "Unverified 3.1 predictions are hints only, not ground truth."},
    ]

    def request_for(task: str, var: dict) -> str:
        prefix = "图片中的目标服饰" if var["id"] != "full_image_only" else "整张图中的目标服饰（若有多个候选，请说明无法唯一确定）"
        attr_part = ""
        if var["context"] is not None:
            attr_part = ("\n以下仅为3.1未核验的模型猜测，不是事实；与图像冲突时忽略，不确定就填null或明确说明："
                         + var["context"])
        if task == "visual_qa":
            return f"请直接回答：{prefix}，{original_query}。一句话作答；看不清就说无法确定。不要编造面料成分。" + attr_part
        if task == "tags_json":
            return (
                f"只根据图中可见的{prefix}输出一个严格的JSON对象，不要代码围栏、前后解释或Markdown。"
                "只允许字段category,primary_color,pattern,visible_design,uncertain_fields。"
                "category仅允许top,pants,skirt,outerwear,dress,shoe,bag,accessory或null。"
                "看不清的字段为null；uncertain_fields为存放不确定字段名的字符串数组。"
                "不得依据未经核验的标签猜测面料、工艺、风格、品牌。"
            ) + attr_part
        if task == "marketing_copy":
            return (
                f"为图中的{prefix}写一条不超过60个汉字的客观、简洁中文商品描述。"
                "只写肉眼可见的颜色、图案、形状和设计；不编造材质、工艺、品牌、复古风格或其他主观标签。"
                "不确定的内容不要写入文案。"
            ) + attr_part
        raise KeyError(task)

    def infer(task: str, var: dict, warmup: bool = False) -> dict:
        request = "请用简短中文回答：图中是否包含服饰？" if warmup else request_for(task, var)
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": var["image"].as_uri()},
            {"type": "text", "text": request},
        ]}]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        template = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(msgs)
        tokens = processor(text=[template], images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**tokens, do_sample=False,
                                 max_new_tokens=12 if warmup else args.max_new_tokens)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        new = ids[:, tokens["input_ids"].shape[1]:]
        raw = processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        if warmup:
            return {}
        result = {
            "variant": var["id"], "task": task, "request": request, "response": raw,
            "prompt_tokens_including_visual": int(tokens["input_ids"].shape[1]),
            "generated_tokens": int(new.shape[1]),
            "preprocess_ms": round((t1 - t0) * 1000, 2),
            "generation_ms": round((t2 - t1) * 1000, 2),
            "end_to_end_ms": round((t2 - t0) * 1000, 2),
        }
        if task == "tags_json":
            result.update(validate_json(raw))
        return result

    print("[2/4] GPU warmup (excluded from timings)...", flush=True)
    infer("visual_qa", variants[1], warmup=True)
    print(f"[3/4] {len(variants)} variants x 3 tasks x {args.repeats} repeats...", flush=True)
    measurements = []
    for rep in range(args.repeats):
        # Alternate variants between repetitions to reduce systematic early-run bias.
        ordered = variants if rep % 2 == 0 else list(reversed(variants))
        for task in ("visual_qa", "tags_json", "marketing_copy"):
            for var in ordered:
                row = infer(task, var)
                row["repeat_index"] = rep + 1
                measurements.append(row)
                print(f"  run={rep+1} {task} {var['id']}: {row['end_to_end_ms']:.1f}ms / {row['generated_tokens']}tok", flush=True)
    aggregates = []
    for task in ("visual_qa", "tags_json", "marketing_copy"):
        for var in variants:
            arr = [x for x in measurements if x["task"] == task and x["variant"] == var["id"]]
            vals = [x["end_to_end_ms"] for x in arr]
            aggregates.append({
                "task": task, "variant": var["id"], "n": len(vals),
                "latency_p50_ms": round(statistics.median(vals), 2),
                "latency_p95_nearest_rank_interpolated_ms": round(pctl(vals, 0.95), 2),
                "min_ms": round(min(vals), 2), "max_ms": round(max(vals), 2),
                "warning": "Very small repeated development sample, not stable latency acceptance.",
            })
    report = {
        "schema": "prd32_roi_attribute_diagnostic_v2",
        "status": "DIAGNOSTIC_COMPLETED_NOT_ACCEPTED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bridge_case_id": case.get("case_id"),
        "image_original": str(src), "image_sha256_verified": True,
        "frozen_predicted_target_bbox": bbox,
        "roi_crop_xyxy": list(box), "roi_image": str(roi_path),
        "target_bbox_is_ground_truth": False,
        "input_attributes_human_verified": False,
        "model_id": args.model_id, "gpu": torch.cuda.get_device_name(0),
        "dtype": str(dtype), "attn_implementation": "sdpa",
        "model_input_pixel_budget": args.max_pixels,
        "repeats": args.repeats,
        "max_new_tokens": args.max_new_tokens,
        "gpu_peak_allocated_gib_diagnostic": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "variants": [{"id": v["id"], "caveat": v["caveat"]} for v in variants],
        "aggregate": aggregates,
        "measurements": measurements,
        "limitations": [
            "No human ground truth has been checked: no accuracy result can be claimed.",
            "No human copy satisfaction review was performed.",
            "Single image and low repeated N are not PRD speed/QPS acceptance.",
            "Frozen 3.1 box might be wrong; verify ROI image visually.",
            "Reported times do not include initial model loading, nor service concurrency.",
        ],
    }
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[4/4] Written {dest}", flush=True)
    print(f"Status: {report['status']}", flush=True)


if __name__ == "__main__":
    main()
