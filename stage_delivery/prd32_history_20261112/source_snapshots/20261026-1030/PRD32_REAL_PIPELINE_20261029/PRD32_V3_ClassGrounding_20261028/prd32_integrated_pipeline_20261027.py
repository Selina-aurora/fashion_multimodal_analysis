#!/usr/bin/env python3
"""PRD 3.2 integrated DEVELOPMENT pipeline on frozen PRD 3.1 bridge inputs.

Reads 3.1 frozen artifacts without changing them. Runs Qwen2.5-VL once loaded,
for VQA / visible-attribute JSON / product copy. Keeps raw outputs, schema gate,
case-level error isolation and latency evidence. NOT formal PRD acceptance.

Default selects three requests (one task each) on the SAME frozen target instance.
Use --all-cases after successful smoke to process every frozen development request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "prd32_engine_neutral_input_v1"
ALLOWED_TASKS = {"vqa", "tags", "copy"}
ALLOWED_CLASSES = {"top", "pants", "skirt", "outerwear", "dress", "shoe", "bag", "accessory"}
TAG_KEYS = {"category", "primary_color", "pattern", "visible_design"}
BAD_DESIGN = {"无", "无设计", "无明显设计", "无装饰", "无明显装饰", "没有", "未知", "无特殊设计", "none", "unknown", "n/a"}
DEFAULT_IDS = ("vqa_color_01", "tags_01", "copy_01")
DEFAULT_BRIDGE = "/workspace/prd31_stage_freeze_20261026/bridge_inputs.jsonl"
DEFAULT_FREEZE = "/workspace/prd31_stage_freeze_20261026"
DEFAULT_MODEL = "/workspace/models/Qwen2.5-VL-7B-Instruct"
DEFAULT_OUTPUT = "/workspace/PRD32_INTEGRATED_DEV_20261027"

# Adapted from 2026-10-26 PRD 3.2 Dev7 V2 attribute baseline.
# V2 experimental numbers do NOT transfer to this new frozen-input setting.
TAG_PROMPT = (
    "任务：只分析红框标注的目标服饰本身，不要把内搭、裤子、背包、背景当成目标属性。"
    "独立判断服饰类别，不根据来自其他模型的未核验预测推断。"
    "类别规则：top 包含T恤、衬衫、卫衣、针织套头衫、背心；"
    "outerwear 包含作为外层穿着的西装外套、夹克、大衣等；"
    "仅因长袖、宽松或有装饰不能归为 outerwear；dress 是连衣裙。"
    "仅输出一个JSON对象，必须且只包含category、primary_color、pattern、visible_design四个键。"
    "category只能是top,pants,skirt,outerwear,dress,shoe,bag,accessory或JSON null。"
    "primary_color是中文可见主色，不能判断时填JSON null。"
    "pattern只反映面料表面图案，如纯色、条纹、格纹、人字纹、波点、花卉、文字印花。"
    "蝴蝶结、纽扣、搭扣等独立装饰不能误当成图案。"
    "visible_design仅列可见领口、袖型、肩带、门襟、扣合、层叠或装饰，用短词分号分隔。"
    "不可判断时填JSON null，不得填'无设计'之类占位词。"
    "不得编造面料成分、功能、品牌、工艺。不要Markdown围栏、解释或多余字段。"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def is_restricted(path) -> bool:
    value = str(path).lower()
    return any(token in value for token in ("holdout", "sealed_do_not_open", "/sealed/", "holdout_sealed"))


def read_jsonl(path: Path):
    records = []
    for line_number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not text.strip():
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as ex:
            raise ValueError(f"Invalid JSONL line {line_number}: {ex}") from ex
        if not isinstance(obj, dict):
            raise ValueError(f"JSONL line {line_number} must be an object")
        records.append(obj)
    return records


def validate_freeze(freeze_dir: Path, bridge_path: Path) -> dict:
    if not freeze_dir.is_dir():
        raise FileNotFoundError(f"Freeze directory missing: {freeze_dir}")
    summary_file, record_file = freeze_dir / "summary.json", freeze_dir / "freeze_record.json"
    if not summary_file.is_file() or not record_file.is_file():
        raise FileNotFoundError("Freeze summary.json and freeze_record.json are BOTH required")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    record = json.loads(record_file.read_text(encoding="utf-8"))
    if summary.get("status") != "PRD31_STAGE_FREEZE_READY" or summary.get("phase_freeze_ready") is not True:
        raise ValueError("Frozen 3.1 phase status not READY")
    if record.get("frozen") is not True or record.get("candidate_id") != summary.get("candidate_id"):
        raise ValueError("Freeze record identity mismatch")
    if bridge_path.resolve() != (freeze_dir / "bridge_inputs.jsonl").resolve():
        raise ValueError("Use the frozen bridge_inputs.jsonl from the matching freeze directory")
    return {"candidate_id": record.get("candidate_id"), "checkpoint_sha256": record.get("checkpoint_sha256"),
            "freeze_record_sha256": sha256_file(record_file), "freeze_summary_sha256": sha256_file(summary_file)}


def validate_record(record: dict, *, check_assets: bool = True) -> dict:
    cid = record.get("case_id", "<unknown>")
    if record.get("schema") != SCHEMA:
        raise ValueError(f"{cid}: wrong bridge schema")
    if record.get("task") not in ALLOWED_TASKS:
        raise ValueError(f"{cid}: unsupported task")
    if record.get("target_bbox_origin") != "prd31_frozen_prediction":
        raise ValueError(f"{cid}: not a frozen 3.1 target bbox")
    for field in ("case_id", "image_id", "target_instance_id", "image_path", "image_sha256", "query"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"{cid}: missing {field}")
    p = Path(record["image_path"]).resolve()
    if is_restricted(p):
        raise ValueError(f"{cid}: restricted image path")
    if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError(f"{cid}: unsupported image suffix")
    b = record.get("target_bbox_xyxy")
    if not isinstance(b, list) or len(b) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in b):
        raise ValueError(f"{cid}: invalid bbox data")
    if not (b[0] < b[2] and b[1] < b[3] and b[0] >= 0 and b[1] >= 0):
        raise ValueError(f"{cid}: invalid bbox geometry")
    attrs = record.get("predicted_attribute_context", [])
    if not isinstance(attrs, list):
        raise ValueError(f"{cid}: predicted attrs must be list")
    for a in attrs:
        if not isinstance(a, dict) or not isinstance(a.get("name"), str):
            raise ValueError(f"{cid}: invalid attrs entry")
        # A predicted score is NOT a calibrated truth probability.
        if a.get("human_verified") is True or a.get("score_is_calibrated_truth_probability") is True:
            raise ValueError(f"{cid}: unexpected verified/calibrated attribute flags")
    dims = None
    if check_assets:
        if not p.is_file():
            raise FileNotFoundError(f"{cid}: image missing: {p}")
        if sha256_file(p) != record["image_sha256"]:
            raise ValueError(f"{cid}: image SHA256 mismatch")
        from PIL import Image
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            dims = im.size
        if not (b[2] <= dims[0] and b[3] <= dims[1]):
            raise ValueError(f"{cid}: bbox out of image bounds ({dims})")
    return {"case_id": cid, "task": record["task"], "image_path": str(p),
            "image_id": record["image_id"], "target_instance_id": record["target_instance_id"],
            "bbox": b, "dimensions": dims, "attribute_count": len(attrs)}


def select_records(all_records, *, selected_ids, all_cases):
    ids = [r.get("case_id") for r in all_records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case IDs in frozen bridge")
    if all_cases:
        chosen = all_records
    else:
        expected = list(selected_ids)
        if len(expected) != len(set(expected)):
            raise ValueError("Duplicate requested case IDs")
        misses = sorted(set(expected) - set(ids))
        if misses:
            raise ValueError("Missing requested bridge cases: " + ",".join(misses))
        chosen = [next(r for r in all_records if r["case_id"] == wanted) for wanted in expected]
    if not chosen:
        raise ValueError("No cases selected")
    return chosen


def parse_tag_output(raw: str):
    txt = raw.strip()
    native = txt.startswith("{") and txt.endswith("}")
    peeled = re.sub(r"^```(?:json)?\s*", "", txt, count=1, flags=re.I)
    if peeled.rstrip().endswith("```"):
        peeled = peeled.rstrip()[:-3].strip()
    try:
        obj = json.loads(peeled)
    except (ValueError, TypeError) as ex:
        return {"parseable": False, "raw_native_json": False, "schema_valid": False,
                "raw_parsed": None, "canonical_service_json": None, "issues": ["json_parse_error:" + str(ex)]}
    native = bool(native and isinstance(obj, dict))
    issues = []
    if not isinstance(obj, dict) or set(obj) != TAG_KEYS:
        issues.append("json_schema_keys_mismatch")
    if isinstance(obj, dict):
        if obj.get("category") is not None and obj.get("category") not in ALLOWED_CLASSES:
            issues.append("invalid_category")
        for k in ("primary_color", "pattern", "visible_design"):
            if obj.get(k) is not None and not isinstance(obj.get(k), str):
                issues.append("bad_value_type:" + k)
    strict = not issues
    safe = dict(obj) if strict else None
    if safe is not None:
        d = safe.get("visible_design")
        if isinstance(d, str) and (d.strip().lower() in BAD_DESIGN or not d.strip()):
            issues.append("nonstructural_design_abstained")
            safe["visible_design"] = None
        p = safe.get("pattern")
        if isinstance(p, str) and any(x in p.lower() for x in ("蝴蝶结", "纽扣", "搭扣", "蕾丝", "lace", "棉", "羊毛")):
            issues.append("pattern_semantics_needs_review_abstained")
            safe["pattern"] = None
        for k in ("primary_color", "pattern", "visible_design"):
            if safe[k] == "":
                safe[k] = None
    return {"parseable": True, "raw_native_json": native, "schema_valid": strict,
            "raw_parsed": obj, "canonical_service_json": safe, "issues": issues}



def bridge_tags_adapter(canonical):
    """Dev-only compatibility view for the frozen `tags` request fields.

    The V2 four-key attribute object remains unchanged. A decoration is proposed
    ONLY if an explicit visible-design term names one. Otherwise abstain; do not
    treat generic sleeves, neckline, layering or garment closure as decoration.
    """
    desc = canonical.get("visible_design")
    words = [x.strip() for x in re.split(r"[;；,，]", desc) if x.strip()] if isinstance(desc, str) else []
    # Only a small, explicit ornament vocabulary; not proof of correctness.
    visible = [x for x in words if any(t in x for t in ("蝴蝶结", "装饰", "饰扣", "刺绣", "印花贴饰"))]
    result = {
        "primary_color": canonical.get("primary_color"),
        "pattern": canonical.get("pattern"),
        "visible_decoration": ";".join(visible) if visible else None,
    }
    result["unconfirmed"] = [k for k, value in result.items() if value is None]
    return result


def make_marked_image(record, output_dir: Path, input_mode: str, pad: float):
    from PIL import Image, ImageDraw
    src = Path(record["image_path"])
    with Image.open(src) as original:
        im = original.convert("RGB")
    x1, y1, x2, y2 = record["target_bbox_xyxy"]
    if input_mode == "marked_full_image":
        shown = im.copy()
        ImageDraw.Draw(shown).rectangle((round(x1), round(y1), round(x2), round(y2)),
                                        outline=(250, 40, 40), width=max(3, min(im.size) // 150))
    else:
        dx, dy = (x2-x1) * pad, (y2-y1) * pad
        coords = (max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
                  min(im.width, math.ceil(x2 + dx)), min(im.height, math.ceil(y2 + dy)))
        shown = im.crop(coords)
    image_key = hashlib.sha256((record["image_sha256"] + "|" + record["target_instance_id"] + "|" + input_mode).encode()).hexdigest()[:14]
    dest = output_dir / f"{image_key}_{input_mode}.jpg"
    if not dest.exists():
        shown.save(dest, quality=93)
    return dest


def short_unverified_context(record):
    # Disabled by default. Preserve provenance; NEVER claim these are GT.
    data = [
        {"name": a["name"], "prediction_not_verified": a.get("predicted_value"), "status": a.get("status")}
        for a in record.get("predicted_attribute_context", [])
        if a.get("predicted_value") is not None
    ]
    payload = json.dumps(data[:8], ensure_ascii=False, separators=(",", ":"))
    return ("\n以下是3.1未人工核验的模型预测，不一定正确，不应覆盖图片可见事实：" + payload[:800]) if data else ""


def make_prompt(record, *, input_mode, pass_unverified):
    reference = "红框所指" if input_mode == "marked_full_image" else "当前裁剪画面的主要"
    context = short_unverified_context(record) if pass_unverified else ""
    kind = record["task"]
    if kind == "tags":
        return TAG_PROMPT + context
    if kind == "vqa":
        return (f"请针对{reference}目标服饰回答这个问题：{record['query']}。"
                "只依据图中可见证据作答，避免把内搭、包或其他服装的属性算到目标上。"
                "看不清就说无法确定；不能凭外观确定准确材质成分、品牌或生产工艺。"
                "回答简洁直接。" + context)
    if kind == "copy":
        length = record.get("length_chars", 80)
        maximum = length[1] if isinstance(length, list) and len(length) == 2 else length
        maximum = int(maximum) if isinstance(maximum, (int, float)) else 80
        maximum = max(20, min(maximum, 150))
        return (f"只为{reference}服饰撰写一段专业、简洁、客观的中文商品描述，最多{maximum}字。"
                "描述颜色、表面图案与明确可见的结构；不要把旁边的服装、包或背景当成目标特点。"
                "不要编造材质成分、品牌、产地、功能性能、营销承诺或工艺。"
                "只返回文案，不要前缀或说明。" + context)
    raise ValueError("unsupported task")


def run_model_case(model, processor, record, image_file: Path, prompt: str, *, task_limits):
    import torch
    from qwen_vl_utils import process_vision_info
    msg = [{"role": "user", "content": [
        {"type": "image", "image": image_file.resolve().as_uri()},
        {"type": "text", "text": prompt}
    ]}]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    txt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    pics, vids = process_vision_info(msg)
    inputs = processor(text=[txt], images=pics, videos=vids, padding=True, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.inference_mode():
        gen = model.generate(**inputs, do_sample=False, max_new_tokens=task_limits[record["task"]])
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    new_ids = gen[:, inputs.input_ids.shape[1]:]
    raw = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return raw, {
        "preprocess_ms": round(1000 * (t1-t0), 2),
        "generation_ms": round(1000 * (t2-t1), 2),
        "end_to_end_ms": round(1000 * (t2-t0), 2),
        "generated_tokens": int(new_ids.shape[1])
    }


def self_test():
    a = parse_tag_output('{"category":"top","primary_color":"白色","pattern":"纯色","visible_design":"圆领;短袖"}')
    assert a["parseable"] and a["raw_native_json"] and a["schema_valid"]
    b = parse_tag_output('```json\n{"category":"shirt","primary_color":"白色","pattern":null,"visible_design":"无"}\n```')
    assert b["parseable"] and not b["schema_valid"] and not b["raw_native_json"]
    c = parse_tag_output('{"category":"dress","primary_color":"黑色","pattern":"纯色","visible_design":"无"}')
    assert c["schema_valid"] and c["canonical_service_json"]["visible_design"] is None
    compat = bridge_tags_adapter({"primary_color":"黑色", "pattern":"纯色", "visible_design":"短袖;圆领"})
    assert compat["visible_decoration"] is None and "visible_decoration" in compat["unconfirmed"]
    try:
        validate_record({"schema":SCHEMA,"case_id":"x","task":"tags","target_bbox_origin":"prd31_frozen_prediction",
                         "image_id":"x","image_path":"/tmp/holdout_sealed/image.jpg","image_sha256":"x","query":"x",
                         "target_instance_id":"x","target_bbox_xyxy":[0,0,1,1],"predicted_attribute_context":[]}, check_assets=False)
        raise AssertionError("restricted path accepted")
    except ValueError:
        pass
    print("SELF_TEST_OK: strict JSON, fenced JSON, safe decoration abstention, unsafe design, sealed-path guard")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bridge-input", default=DEFAULT_BRIDGE)
    ap.add_argument("--freeze-dir", default=DEFAULT_FREEZE)
    ap.add_argument("--model-id", default=DEFAULT_MODEL)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--case-ids", default=",".join(DEFAULT_IDS), help="Comma-separated case IDs; default 3 tasks on one target")
    ap.add_argument("--all-cases", action="store_true", help="Run all frozen development bridge requests")
    ap.add_argument("--input-mode", choices=("marked_full_image", "predicted_roi"), default="marked_full_image")
    ap.add_argument("--roi-padding", type=float, default=0.05)
    ap.add_argument("--pass-unverified-context", action="store_true", help="Ablation only; 3.1 predictions are NOT human verified")
    ap.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--dry-run", action="store_true", help="Validate input files, hashes and freeze state; no GPU")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not 0 <= args.roi_padding <= 0.5:
        ap.error("--roi-padding must be in [0, 0.5]")
    if args.max_pixels <= 0:
        ap.error("--max-pixels must be positive")
    bridge = Path(args.bridge_input).resolve()
    freeze_dir = Path(args.freeze_dir).resolve()
    out = Path(args.output).resolve()
    model_dir = Path(args.model_id).resolve()
    if is_restricted(bridge) or is_restricted(out) or is_restricted(freeze_dir):
        ap.error("sealed/holdout path is not allowed")
    if not bridge.is_file():
        ap.error(f"missing bridge file: {bridge}")
    freeze_info = validate_freeze(freeze_dir, bridge)
    all_records = read_jsonl(bridge)
    if len(all_records) == 0:
        ap.error("frozen bridge is empty")
    chosen = select_records(all_records, selected_ids=[x.strip() for x in args.case_ids.split(",") if x.strip()],
                            all_cases=args.all_cases)
    if args.all_cases and any(r.get("task") not in ALLOWED_TASKS for r in all_records):
        ap.error("unsupported frozen task present")
    # Validate selected cases; no silent skip of broken frozen identities.
    checked = [validate_record(r, check_assets=True) for r in chosen]
    gate = {
        "schema": "prd32_integrated_preflight_v1", "status": "INPUT_PREFLIGHT_OK", "checked_at": utc_now(),
        "bridge_sha256": sha256_file(bridge), "frozen_3_1": freeze_info,
        "all_bridge_requests": len(all_records), "selected_count": len(chosen),
        "selected_case_ids": [r["case_id"] for r in chosen], "selected_tasks": dict(Counter(r["task"] for r in chosen)),
        "image_hashes_verified": True, "target_bbox_origin": "prd31_frozen_prediction",
        "selected_inputs": checked, "model_not_loaded": bool(args.dry_run),
        "holdout_used": False, "frozen_sources_modified": False,
        "caveat": "Only selected case identities checked; 3.1 frozen predictor was NOT rerun."
    }
    print(f"INPUT_PREFLIGHT_OK: {len(chosen)} case(s); tasks={gate['selected_tasks']}", flush=True)
    if args.dry_run:
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        print("DRY_RUN_OK: no GPU, no output artifacts written", flush=True)
        return 0
    if out.exists():
        ap.error(f"output already exists; refuse overwrite: {out}")
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        ap.error(f"local model is missing; no network download will be attempted: {model_dir}")
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available():
        ap.error("CUDA GPU not available")
    free_gib = torch.cuda.mem_get_info()[0] / 1024**3
    if free_gib < 18:
        ap.error(f"Only {free_gib:.2f} GiB CUDA available; release other GPU processes")
    out.mkdir(parents=True, exist_ok=False)
    imgs_dir = out / "marked_or_cropped_inputs"
    imgs_dir.mkdir()
    write_json(out / "input_preflight.json", gate)
    write_json(out / "run_config.json", {
        "time_utc": utc_now(), "model_path": str(model_dir), "bridge_input": str(bridge),
        "input_mode": args.input_mode, "roi_padding": args.roi_padding,
        "max_pixels": args.max_pixels, "pass_unverified_context": args.pass_unverified_context,
        "case_ids": [r["case_id"] for r in chosen], "tag_prompt_sha256": hashlib.sha256(TAG_PROMPT.encode()).hexdigest(),
        "ground_truth_passed_to_model": False, "frozen_prd31_not_modified": True
    })
    (out / "prompt_tags.txt").write_text(TAG_PROMPT + "\n", encoding="utf-8")
    with (out / "bridge_selected_snapshot.jsonl").open("w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"[1/3] Loading Qwen locally from {model_dir}...", flush=True)
    load_t0 = time.perf_counter()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True, device_map="auto", dtype=dtype,
        attn_implementation="sdpa", low_cpu_mem_usage=True
    ).eval()
    processor = AutoProcessor.from_pretrained(
        str(model_dir), local_files_only=True, max_pixels=args.max_pixels
    )
    load_s = round(time.perf_counter() - load_t0, 3)
    print(f"[2/3] Model loaded ({load_s}s), running {len(chosen)} requests...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    outputs = []
    # Incremental saving: a terminal disconnect or later failure cannot erase earlier cases.
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as stream, \
         (out / "service_outputs.jsonl").open("w", encoding="utf-8") as service_stream:
        for idx, record in enumerate(chosen, 1):
            result = {
                "case_id": record["case_id"], "task": record["task"], "image_id": record["image_id"],
                "target_instance_id": record["target_instance_id"], "bbox_origin": record["target_bbox_origin"],
                "input_mode": args.input_mode, "bridge_input_attrs_are_unverified": True,
                "status": "STARTED", "model_native_json": None, "content_quality_human_verified": False,
            }
            service = {"case_id": record["case_id"], "task": record["task"], "status": "UNAVAILABLE", "payload": None}
            try:
                marked = make_marked_image(record, imgs_dir, args.input_mode, args.roi_padding)
                prompt = make_prompt(record, input_mode=args.input_mode, pass_unverified=args.pass_unverified_context)
                raw, timing = run_model_case(model, processor, record, marked, prompt,
                                             task_limits={"vqa": 128, "tags": 170, "copy": 120})
                if not raw.strip():
                    raise ValueError("empty_model_response")
                result.update({"status": "GENERATED_NOT_ACCEPTED", "prompt": prompt, "raw_text": raw,
                               "input_sent": str(marked), **timing})
                if record["task"] == "tags":
                    parsed = parse_tag_output(raw)
                    result.update(parsed)
                    result["model_native_json"] = parsed["raw_native_json"]
                    if parsed["canonical_service_json"] is not None:
                        bridge_compat = bridge_tags_adapter(parsed["canonical_service_json"])
                        result["bridge_compat_tags"] = bridge_compat
                        service.update({"status": "CANONICALIZED_NOT_HUMAN_VERIFIED",
                                        "payload": bridge_compat,
                                        "canonical_attributes_v2": parsed["canonical_service_json"],
                                        "adapter_note": "bridge-compatible candidate; decoration may safely abstain; NOT ground truth",
                                        "issues": parsed["issues"]})
                    else:
                        service["issues"] = parsed["issues"]
                else:
                    if record["task"] == "copy":
                        requested = record.get("length_chars", [20, 80])
                        max_chars = int(requested[1]) if isinstance(requested, list) and len(requested) == 2 else 80
                        result["length_limit_chars"] = max_chars
                        result["length_warning"] = (len(raw.strip()) > max_chars)
                    if raw.strip():
                        service.update({"status": "DRAFT_UNVERIFIED", "payload": raw.strip()})
                print(f"  {idx}/{len(chosen)} {record['case_id']} {record['task']}: {timing['end_to_end_ms']}ms | {raw[:90]!r}", flush=True)
            except Exception as ex:
                result.update({"status": "CASE_FAILED", "error": type(ex).__name__ + ": " + str(ex)})
                service["issues"] = ["case_failed"]
                print(f"  {idx}/{len(chosen)} {record['case_id']} FAILED: {ex}", flush=True)
            outputs.append(result)
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            service_stream.write(json.dumps(service, ensure_ascii=False, allow_nan=False) + "\n")
            service_stream.flush()
    failed = [r for r in outputs if r["status"] == "CASE_FAILED"]
    tag_rows = [r for r in outputs if r["task"] == "tags"]
    times = [r["end_to_end_ms"] for r in outputs if "end_to_end_ms" in r]
    medians = {task: round(statistics.median([r["end_to_end_ms"] for r in outputs if r["task"] == task and "end_to_end_ms" in r]), 2)
               for task in ALLOWED_TASKS if any(r["task"] == task and "end_to_end_ms" in r for r in outputs)}
    status = "PIPELINE_DEV_SMOKE_COMPLETED_NOT_ACCEPTED" if not failed else "PIPELINE_DEV_PARTIAL_OR_FAILED"
    summary = {
        "schema": "prd32_integrated_pipeline_v1", "status": status, "completed_at_utc": utc_now(),
        "frozen_bridge_sha256": gate["bridge_sha256"], "frozen_candidate_id": freeze_info["candidate_id"],
        "selected_cases": len(chosen), "generated_cases": len(outputs)-len(failed), "failed_cases": len(failed),
        "task_counts": dict(Counter(r["task"] for r in chosen)),
        "input_mode": args.input_mode, "model": str(model_dir), "model_load_seconds": load_s,
        "gpu_name": torch.cuda.get_device_name(0), "gpu_peak_allocated_gib": round(torch.cuda.max_memory_allocated()/1024**3, 3),
        "median_single_request_ms": round(statistics.median(times), 2) if times else None,
        "median_latency_by_task_ms": medians,
        "tags_generated": len(tag_rows), "tags_schema_valid": sum(bool(r.get("schema_valid")) for r in tag_rows),
        "tags_native_json": sum(bool(r.get("raw_native_json")) for r in tag_rows),
        "tags_canonical_service_available": sum(r.get("canonical_service_json") is not None for r in tag_rows),
        "qa_and_copy_outputs_not_human_graded": True,
        "frozen_prd31_reexecuted": False, "frozen_prd31_modified": False,
        "holdout_used": False, "full_prd_acceptance": False,
        "limitations": ["Offline frozen interface case replay, NOT new 3.1 inference on uploaded images",
                        "Input uses actual 3.1 frozen predicted bboxes, NOT historical human reference ROI",
                        "Model predictions are not human GT; QA/copy quality and formal eight-class accuracy not measured",
                        "Single/small-batch local GPU timings NOT production service P95 or QPS"]
    }
    write_json(out / "summary.json", summary)
    print("[3/3] SUMMARY:", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("RESULTS:", out, flush=True)
    return 0 if not failed else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("INTERRUPTED: partial predictions.jsonl may exist; do not report success", file=sys.stderr)
        sys.exit(130)
