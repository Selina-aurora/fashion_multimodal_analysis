#!/usr/bin/env python3
"""PRD 3.2 V3 target-binding diagnostic (development data only).

Three controlled input variants: bounding-box-marked full image, ROI crop,
ROI crop + *unverified* PRD 3.1 attributes. Prompt names the intended 8-class
category; never supplies a color / pattern / design ground truth.

Not an accuracy, latency, QPS, or end-to-end PRD acceptance test.
Does not modify frozen PRD 3.1 artifacts or access sealed holdout inputs.
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

CATEGORIES = ("top", "pants", "skirt", "outerwear", "dress", "shoe", "bag", "accessory")
CATEGORY_CN = {
    "top": "上衣", "pants": "裤子", "skirt": "半身裙", "outerwear": "外套",
    "dress": "连衣裙", "shoe": "鞋子", "bag": "包", "accessory": "配饰",
}
FIELDS = ("category", "primary_color", "pattern", "visible_design", "uncertain_fields")
UNCERTAIN_ALLOWED = {"category", "primary_color", "pattern", "visible_design"}


def load_json_output(raw: str) -> dict:
    s = raw.strip()
    strict = s.startswith("{") and s.endswith("}") and "```" not in s
    fallback = s
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            fallback = "\n".join(lines[1:-1]).strip()
    try:
        obj = json.loads(fallback)
    except (ValueError, TypeError) as e:
        return {"strict_json": False, "parseable_json": False, "schema_valid": False,
                "parsed_json": None, "json_error": str(e)}
    is_dict = isinstance(obj, dict)
    good = (is_dict and set(obj) == set(FIELDS)
            and obj.get("category") in (*CATEGORIES, None)
            and all(obj.get(k) is None or isinstance(obj.get(k), str)
                    for k in ("primary_color", "pattern", "visible_design"))
            and isinstance(obj.get("uncertain_fields"), list)
            and len(obj["uncertain_fields"]) == len(set(map(str, obj["uncertain_fields"])))
            and all(isinstance(z, str) and z in UNCERTAIN_ALLOWED for z in obj["uncertain_fields"]))
    return {"strict_json": bool(strict and is_dict), "parseable_json": bool(is_dict),
            "schema_valid": bool(good), "parsed_json": obj if is_dict else None,
            "json_error": None if is_dict else "JSON value is not an object"}


def percentile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    pos = (len(xs) - 1) * q
    a = math.floor(pos)
    b = math.ceil(pos)
    return xs[a] + (xs[b] - xs[a]) * (pos - a)


def safe_attrs(rows: object) -> str:
    compact = []
    for x in rows if isinstance(rows, list) else []:
        if not isinstance(x, dict) or x.get("status") != "predicted":
            continue
        name, value = x.get("name"), x.get("predicted_value")
        if isinstance(name, str) and isinstance(value, str):
            compact.append({"name": name, "value": value, "verified": False})
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def prompts(task: str, category: str, hint: str | None) -> str:
    cn = CATEGORY_CN[category]
    target = (f"【目标绑定】只分析图中类别为 {category}（{cn}）的指定服饰。"
              "不要把内搭、外搭、包、配饰或背景的颜色/条纹/袖长算到目标服饰上。"
              "当图中有叠穿时，仅描述指定目标本身。"
              "如无法准确区分目标边界，明确标记不确定，不要猜测。")
    note = ("\n【来源不可靠的可选提示】以下为3.1模型预测，未经人工核验。"
            "可能关联到另一件服饰，禁止直接照抄；仅在能从图片直接确认时使用：" + hint) if hint is not None else ""
    if task == "visual_qa":
        return target + "\n请回答：目标服饰本身的主要颜色是什么？只输出颜色或‘无法确定’，不加解释。" + note
    if task == "tags_json":
        return (target + "\n仅输出一个原生JSON对象，不要Markdown代码围栏，不要解释。"
                "JSON必须且只包含category、primary_color、pattern、visible_design、uncertain_fields五个键。"
                f"category必须为{category}；primary_color填目标自身主要可见颜色（中文）；"
                "pattern填目标本身图案（例如纯色、条纹）或null；"
                "visible_design写目标本身可见的结构设计特征（例如无袖、背带），不能写图案或其他衣物信息；"
                "不能判定时填null，并把对应的字段名放入uncertain_fields数组。"
                "uncertain_fields只准出现category、primary_color、pattern、visible_design字段名，不能放类别名。"
                "不预测面料成分、工艺、品牌、功能或主观风格。" + note)
    if task == "marketing_copy":
        return (target + "\n写一句不超过60个汉字的客观中文商品文案，只描述该目标服饰本身。"
                "颜色、图案或结构不确定就不要写；不要描述其他衣物、鞋、包或配饰。"
                "不编造面料、工艺、功能、品牌、主观风格。只输出文案。" + note)
    raise KeyError(task)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bridge-input", default="/workspace/prd31_stage_freeze_20261026/bridge_inputs.jsonl")
    p.add_argument("--case-index", type=int, default=0)
    p.add_argument("--target-category", choices=CATEGORIES, required=True,
                   help="Explicitly specify target category from reviewed development sample; never auto-assume GT")
    p.add_argument("--model-id", default="/workspace/models/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--output", default="/workspace/PRD32_TARGET_BINDING_V3_20261026")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--max-pixels", type=int, default=401408)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--roi-padding", type=float, default=0.06)
    args = p.parse_args()
    if not 1 <= args.repeats <= 5:
        p.error("--repeats must be between 1 and 5")
    if not 0 <= args.roi_padding <= 0.3:
        p.error("--roi-padding must be in [0, 0.3]")

    bridge = Path(args.bridge_input).resolve(strict=True)
    if any(token in str(bridge).lower() for token in ("holdout", "sealed")):
        raise ValueError("Sealed inputs are prohibited")
    cases = [json.loads(x) for x in bridge.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not 0 <= args.case_index < len(cases):
        p.error(f"case index out of range: 0..{len(cases)-1}")
    case = cases[args.case_index]
    if case.get("schema") != "prd32_engine_neutral_input_v1":
        raise ValueError("Unexpected frozen bridge schema")
    source = Path(case["image_path"]).resolve(strict=True)
    if any(token in str(source).lower() for token in ("holdout", "sealed")):
        raise ValueError("Sealed / holdout image prohibited")
    if source.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise ValueError("Unexpected image file type")
    if hashlib.sha256(source.read_bytes()).hexdigest() != case.get("image_sha256"):
        raise RuntimeError("Input image does not match frozen 3.1 SHA256")
    box = case.get("target_bbox_xyxy")
    if not isinstance(box, list) or len(box) != 4 or not all(
        type(v) in (int, float) and math.isfinite(v) for v in box
    ):
        raise ValueError("Invalid predicted target bbox")

    from PIL import Image, ImageDraw
    with Image.open(source) as im:
        rgb = im.convert("RGB")
    w, h = rgb.size
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("Bounding box outside image")
    dx, dy = (x2-x1)*args.roi_padding, (y2-y1)*args.roi_padding
    crop_box = (max(0, math.floor(x1-dx)), max(0, math.floor(y1-dy)),
                min(w, math.ceil(x2+dx)), min(h, math.ceil(y2+dy)))
    if crop_box[2]-crop_box[0] < 24 or crop_box[3]-crop_box[1] < 24:
        raise ValueError("ROI crop too small")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dest = output / f"target_binding_case_{args.case_index:02d}.json"
    crop_path = output / f"target_crop_case_{args.case_index:02d}.jpg"
    marked_path = output / f"target_box_case_{args.case_index:02d}.jpg"
    if any(z.exists() for z in (dest, crop_path, marked_path)):
        raise FileExistsError("Refusing to overwrite previous V3 results; choose a fresh --output")
    rgb.crop(crop_box).save(crop_path, quality=95)
    marked = rgb.copy()
    draw = ImageDraw.Draw(marked)
    line_width = max(3, round(min(w, h)/180))
    draw.rectangle((round(x1), round(y1), round(x2), round(y2)), outline=(255, 36, 35), width=line_width)
    marked.save(marked_path, quality=95)

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU")
    free, _ = torch.cuda.mem_get_info()
    if free < 18 * 1024**3:
        raise RuntimeError(f"Free GPU memory {free/1024**3:.2f} GiB < 18 GiB")
    local = Path(args.model_id).is_dir()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print("[1/4] Loading local VLM; 3.1 is read-only...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=dtype, device_map="auto", attn_implementation="sdpa",
        low_cpu_mem_usage=True, local_files_only=local,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model_id, min_pixels=128*28*28, max_pixels=args.max_pixels,
        local_files_only=local,
    )
    torch.cuda.reset_peak_memory_stats()

    hint = safe_attrs(case.get("predicted_attribute_context"))
    variants = [
        {"id": "marked_full_image", "image": marked_path, "unverified": None,
         "image_rule": "红框内有目标服饰，框内也可能包含其他叠穿服饰，仅分析指定类别的目标。"},
        {"id": "roi_target_only", "image": crop_path, "unverified": None,
         "image_rule": "此图为3.1预测框的裁剪，并非纯净分割掩码，裁剪内可能出现其他服饰。"},
        {"id": "roi_plus_unverified_attrs", "image": crop_path, "unverified": hint,
         "image_rule": "此图为3.1预测框的裁剪，并非纯净分割掩码，裁剪内可能出现其他服饰。"},
    ]

    def predict(task: str, var: dict, warmup: bool = False) -> dict:
        request = ("只答：图中有衣服吗？" if warmup else
                   var["image_rule"] + "\n" + prompts(task, args.target_category, var["unverified"]))
        msg = [{"role": "user", "content": [
            {"type": "image", "image": var["image"].as_uri()},
            {"type": "text", "text": request},
        ]}]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        raw_prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(msg)
        inputs = processor(text=[raw_prompt], images=images, videos=videos,
                           padding=True, return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**inputs, do_sample=False,
                                 max_new_tokens=12 if warmup else args.max_new_tokens)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        outputs = ids[:, inputs["input_ids"].shape[1]:]
        result = processor.batch_decode(outputs, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)[0]
        if warmup:
            return {}
        row = {
            "task": task, "variant": var["id"], "response": result, "request": request,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "generated_tokens": int(outputs.shape[1]),
            "preprocess_ms": round((t1-t0)*1000, 2),
            "generation_ms": round((t2-t1)*1000, 2),
            "end_to_end_ms": round((t2-t0)*1000, 2),
        }
        if task == "tags_json":
            row.update(load_json_output(result))
        return row

    print("[2/4] GPU warmup...", flush=True)
    predict("visual_qa", variants[1], warmup=True)
    measurements = []
    print(f"[3/4] 3 variants × 3 tasks × {args.repeats} repeats...", flush=True)
    for rep in range(args.repeats):
        order = variants if rep % 2 == 0 else list(reversed(variants))
        for task in ("visual_qa", "tags_json", "marketing_copy"):
            for var in order:
                row = predict(task, var)
                row["repeat_index"] = rep + 1
                measurements.append(row)
                print(f"  {task} / {var['id']} / rep{rep+1}: {row['end_to_end_ms']:.1f} ms: {row['response'][:90]!r}", flush=True)

    agg = []
    for task in ("visual_qa", "tags_json", "marketing_copy"):
        for var in variants:
            sub = [r for r in measurements if r["task"] == task and r["variant"] == var["id"]]
            times = [r["end_to_end_ms"] for r in sub]
            agg.append({"task": task, "variant": var["id"], "n": len(sub),
                        "p50_ms": round(statistics.median(times), 2),
                        "p95_interpolated_ms": round(percentile(times, 0.95), 2),
                        "min_ms": round(min(times), 2), "max_ms": round(max(times), 2),
                        "warning": "Small development-only N; not PRD latency acceptance."})
    report = {
        "schema": "prd32_target_binding_diagnostic_v3", "status": "DIAGNOSTIC_COMPLETED_NOT_ACCEPTED",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "bridge_case_id": case.get("case_id"), "case_index": args.case_index,
        "explicit_target_category": args.target_category,
        "target_category_is_human_provided_for_this_diagnostic": True,
        "target_bbox_source": "frozen_3_1_model_prediction_not_gt",
        "original_image": str(source), "original_image_sha256_verified": True,
        "source_bbox_xyxy": box, "roi_crop_xyxy": crop_box,
        "target_box_image": str(marked_path), "target_roi_image": str(crop_path),
        "unverified_attribute_context_included_only_in_variant": "roi_plus_unverified_attrs",
        "model_id": args.model_id, "gpu": torch.cuda.get_device_name(0),
        "dtype": str(dtype), "attention": "sdpa", "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens, "repeats": args.repeats,
        "gpu_peak_allocated_gib": round(torch.cuda.max_memory_allocated()/1024**3, 3),
        "aggregate": agg, "measurements": measurements,
        "limitations": [
            "Explicit category is manually set for debugging; do not score category accuracy on this guided condition.",
            "The cropped image can contain garments other than the target.",
            "No verified attribute GT or copy satisfaction labels; not an accuracy result.",
            "JSON shape validation is not semantic correctness.",
            "Single development image and low N are not PRD 3.2 speed acceptance.",
            "No end-to-end PRD 3.1 + 3.2 system time or concurrent throughput included.",
            "No sealed holdout is used.",
        ],
    }
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"[4/4] Wrote {dest}", flush=True)
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
