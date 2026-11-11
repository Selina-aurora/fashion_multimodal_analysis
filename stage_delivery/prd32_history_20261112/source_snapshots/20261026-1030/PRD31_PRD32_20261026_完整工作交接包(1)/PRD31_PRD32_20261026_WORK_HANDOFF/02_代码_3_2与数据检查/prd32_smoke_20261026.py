#!/usr/bin/env python3
"""PRD 3.2 isolated Qwen2.5-VL smoke run; never alters PRD 3.1 artifacts.

A smoke result is NOT PRD acceptance. It reports cold single-request latencies,
raw outputs, and format validity; accuracy/quality require labeled evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def version(package: str) -> str:
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return "NOT_INSTALLED"


def load_attrs(path: str | None):
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Attribute JSON not found: {p}")
    with p.open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError("--attributes-json must contain one attribute object (JSON dictionary).")
    return value


def clean_json(text: str):
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", help="Explicit NON-HOLDOUT image file; use only without --bridge-input")
    p.add_argument("--bridge-input", default="/workspace/prd31_stage_freeze_20261026/bridge_inputs.jsonl", help="Frozen 3.1 -> 3.2 handoff JSONL")
    p.add_argument("--case-index", type=int, default=0, help="Zero-based selected handoff request, default 0")
    p.add_argument("--attributes-json", help="Optional JSON dict of verified PRD 3.1 image attributes")
    p.add_argument("--question", default="请描述图中主要服饰的可见颜色、图案和设计细节；不可判断的内容请说明不确定。")
    p.add_argument("--style", default="专业、简洁、客观")
    p.add_argument("--max-chars", type=int, default=80)
    p.add_argument("--output", default="/workspace/PRD32_SMOKE_20261026")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    p.add_argument("--max-new-tokens", type=int, default=128)
    args = p.parse_args()

    bridge_case = None
    bridge_file = Path(args.bridge_input).expanduser().resolve() if args.bridge_input else None
    if bridge_file and bridge_file.is_file():
        bridge_rows = [json.loads(line) for line in bridge_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not 0 <= args.case_index < len(bridge_rows):
            p.error(f"--case-index out of range: 0..{len(bridge_rows)-1}")
        bridge_case = bridge_rows[args.case_index]
        if bridge_case.get("schema") != "prd32_engine_neutral_input_v1":
            p.error("Unexpected bridge schema")
        if args.image:
            p.error("Do not mix --image and --bridge-input. Pass --bridge-input '' for manual-image mode")
        args.image = bridge_case["image_path"]
    if not args.image:
        p.error("Missing frozen bridge_inputs.jsonl and no --image supplied")
    path = Path(args.image).expanduser().resolve()
    if not path.is_file():
        p.error(f"--image file does not exist: {path}")
    if "holdout" in str(path).lower() or "sealed" in str(path).lower():
        p.error("Refusing possible SEALED/holdout path; use a known development image instead")
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        p.error("--image must be an actual image file")
    from PIL import Image
    with Image.open(path) as im:
        im.verify()
    if bridge_case is not None:
        expected_hash = bridge_case.get("image_sha256")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_hash != actual_hash:
            raise RuntimeError("Bridge image SHA256 mismatch; refusing to run on changed input")
        attrs = bridge_case.get("predicted_attribute_context", [])
        attr_text = "3.1预测属性（未经人工验证，仅作不确定提示）：" + json.dumps(attrs, ensure_ascii=False, separators=(",", ":"))
        args.question = bridge_case.get("query", args.question)
        args.style = bridge_case.get("style", args.style)
        args.max_chars = int(bridge_case.get("length_chars") or args.max_chars)
    else:
        attrs = load_attrs(args.attributes_json)
        attr_text = "用户提供的属性上下文（验证状态未知）：" + (json.dumps(attrs, ensure_ascii=False, separators=(",", ":")) if attrs is not None else "无")
    if len(attr_text) > 1500:
        raise ValueError("Attributes too long (>1500 chars); provide only the selected image's verified attributes")

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not attempt this 7B CUDA smoke on CPU")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / 1024**3
    if free_gib < 18:
        raise RuntimeError(f"Only {free_gib:.2f} GiB CUDA memory is free; avoid OOM with 7B. Free GPU first.")

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    info = {
        "schema": "prd32_qwen25vl_smoke_v1",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STARTED",
        "model_id": args.model_id,
        "image": str(path),
        "attribute_source": str(bridge_file) if bridge_case else (str(Path(args.attributes_json).expanduser().resolve()) if args.attributes_json else None),
        "bridge_case_id": bridge_case.get("case_id") if bridge_case else None,
        "bridge_task": bridge_case.get("task") if bridge_case else None,
        "bridge_output_format": bridge_case.get("output_format") if bridge_case else None,
        "bridge_variant": bridge_case.get("variant") if bridge_case else None,
        "gpu": torch.cuda.get_device_name(0),
        "free_gpu_gib_before_load": round(free_gib, 3),
        "versions": {name: version(name) for name in ("torch", "transformers", "accelerate", "huggingface-hub", "qwen-vl-utils")},
        "attn_implementation": "sdpa",
        "dtype": str(dtype),
        "pixel_budget": args.max_pixels,
        "note": "Cold one-shot smoke. NOT PRD accuracy, speed, or QPS acceptance evidence.",
    }
    print("[1/4] Loading model; first run may download weights (disk/network required)...", flush=True)
    tload = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_id,
                                               min_pixels=128 * 28 * 28,
                                               max_pixels=args.max_pixels)
    info["model_load_seconds"] = round(time.perf_counter() - tload, 3)
    info["hf_device_map"] = getattr(model, "hf_device_map", None)
    print("[2/4] Model loaded, running three independent PRD 3.2 outputs...", flush=True)

    def infer(kind, request):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": path.as_uri()},
            {"type": "text", "text": request},
        ]}]
        torch.cuda.synchronize()
        start = time.perf_counter()
        prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(msgs)
        inputs = processor(text=[prompt], images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        preprocess_s = time.perf_counter() - start
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        inference_s = time.perf_counter() - start - preprocess_s
        new_ids = ids[:, inputs["input_ids"].shape[1]:]
        result = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        elapsed_s = time.perf_counter() - start
        result_record = {
            "task": kind, "question": request,
            "response": result,
            "preprocess_ms": round(preprocess_s * 1000, 2),
            "generation_ms": round(inference_s * 1000, 2),
            "end_to_end_ms": round(elapsed_s * 1000, 2),
            "generated_tokens": int(new_ids.shape[1]),
        }
        if kind == "tags_json":
            parsed, err = clean_json(result)
            result_record["json_valid"] = err is None and isinstance(parsed, dict)
            result_record["parsed_json"] = parsed if isinstance(parsed, dict) else None
            result_record["parse_error"] = err
        print(f"  {kind}: {result_record['end_to_end_ms']} ms | {result[:140]!r}", flush=True)
        return result_record

    try:
        info["results"] = [
            infer("visual_qa", args.question + "\n属性上下文：" + attr_text + "\n请优先依据图片判断，对未经核验的预测不得作为确定事实。"),
            infer("tags_json", "仅根据图像可见信息和下方带验证状态的属性上下文，生成严格合法的JSON对象，不要Markdown，"
                  "字段必须包含category、primary_color、pattern、visible_design、uncertain_fields。"
                  "category只允许top,pants,skirt,outerwear,dress,shoe,bag,accessory或null。"
                  "看不清的字段填null，不要臆测面料成分或工艺；未经人工核验的预测不能当作事实。属性上下文：" + attr_text),
            infer("marketing_copy", "只用图像可见信息及有验证状态的属性上下文，为商品生成卖点文案。"
                  f"风格：{args.style}；中文长度不超过{args.max_chars}字。"
                  "不编造材质成分、功能性能、工艺或品牌，不把未经核验预测当作事实。属性上下文：" + attr_text),
        ]
        info["status"] = "SMOKE_COMPLETED_NOT_ACCEPTED"
    except Exception as exc:
        info["status"] = "SMOKE_FAILED"
        info["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        info["gpu_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        destination = out / (f"smoke_result_case_{args.case_index:02d}.json" if bridge_case else "smoke_result.json")
        destination.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[4/4] Report: {destination} | status={info['status']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
