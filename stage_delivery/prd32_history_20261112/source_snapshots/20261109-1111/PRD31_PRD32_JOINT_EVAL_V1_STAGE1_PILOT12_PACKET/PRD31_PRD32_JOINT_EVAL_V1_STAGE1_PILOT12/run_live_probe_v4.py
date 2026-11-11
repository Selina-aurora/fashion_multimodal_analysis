#!/usr/bin/env python3
"""Opt-in Qwen DEV60 host-call probe. Does not alter original runner or output.

PRECHECK: validate pinned data/prompts without loading Qwen.
LIVE: use the identical original prompt/model-call functions to run 1-3 V5
predicted instances, in a NEW output directory, shadow-only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from live_observer_v4 import LiveShadowObserver, TASKS, sha_file
from verified_guard_loader import load_guard

EXPECTED_RUNNER_SHA256 = "98a833c429da4c39849942a81f1067239273258a8605eac9265d0aeb9defc6f3"
EXPECTED_CORE_SHA256 = "fdf1c55583b75f4692ddad07d459ceabff7a337d4f82d90e71a5c908f1323d2f"
DEFAULT_RUNNER = Path("/workspace/PRD32_DEV60_QWEN_AB_20261030/dev60_qwen_ab.py")
DEFAULT_REPO = Path("/workspace/fashion_multimodal_analysis")


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def verified_runner(path: Path):
    path = Path(path).resolve(strict=True)
    if sha_file(path) != EXPECTED_RUNNER_SHA256:
        raise ValueError("actual DEV60 runner source differs from reviewed GPU source")
    core = path.with_name("reference_12img_runner_READ_ONLY.py")
    if sha_file(core) != EXPECTED_CORE_SHA256:
        raise ValueError("pinned original prompt helper source differs from reviewed GPU source")
    spec = importlib.util.spec_from_file_location("prd32_pinned_dev60_qwen_runner_probe_v4", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load runner")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_triplets(*, rows, runner, core, helper, focus, variant,
                 model, processor, output: Path, guard, max_instances: int):
    """Calls REAL helper.make_marked_image and run_model_case, observed in-process.

    Args are injectable for tests; no synthetic ROI/mask provenance is created.
    """
    obs = LiveShadowObserver(variant)
    marked_dir = output / "marked_inputs"
    marked_dir.mkdir(exist_ok=False)
    result_path = output / "qwen_shadow_results.jsonl"
    event_path = output / "host_call_events.jsonl"
    selection = rows[:max_instances]
    if not selection:
        raise ValueError("no validated DEV instances")
    with result_path.open("x", encoding="utf-8") as rf, event_path.open("x", encoding="utf-8") as ef:
        for rec in selection:
            for task in TASKS:
                case, iid = rec["case_id"], rec["target_instance_id"]
                qrec = {"image_path": rec["image_path"], "image_sha256": rec["image_sha256"],
                        "target_bbox_xyxy": rec["target_bbox_xyxy"], "target_instance_id": iid,
                        "task": task, "query": runner.QUESTION, "length_chars": [20, 80],
                        "predicted_attribute_context": []}
                marked = helper.make_marked_image(qrec, marked_dir, "marked_full_image", .05)
                if not Path(marked).resolve(strict=True).is_relative_to(marked_dir.resolve(strict=True)):
                    raise ValueError("marked image outside newly created probe directory")
                prompt = core.build_prompt(helper, focus, qrec, task, variant)
                prompt_sha = runner.tsha(prompt)
                if prompt_sha != runner.PROMPT_SHA[variant][task]:
                    raise ValueError("locked Baseline/Focus prompt changed")
                job = "__".join((case, iid, task, variant, "attempt_01"))
                request = obs.before_model(rec=rec, qrec=qrec, task=task,
                                           marked=marked, prompt=prompt,
                                           prompt_sha=prompt_sha, job_id=job)
                raw, timing = helper.run_model_case(model, processor, qrec, marked,
                                                     prompt, task_limits={"tags": 170,
                                                                          "vqa": 128, "copy": 120})
                ev = obs.after_model(job_id=job, response_job_id=job,
                                     completed_host_request_id=request, raw_text=raw)
                row = {"case_id": case, "target_instance_id": iid, "model_variant": variant,
                       "task": task, "host_request_id": request,
                       "legacy_request_id": f"{case}__{iid}__{task}",
                       "status": "GENERATED_REVIEW_REQUIRED",
                       "image_sha256": rec["image_sha256"],
                       "marked_image_sha256": ev["marked_sha256"],
                       "v5_category_unverified": rec["predicted_category_v5_unverified"],
                       "v5_bbox_xyxy": rec["target_bbox_xyxy"],
                       "prompt_sha256": prompt_sha, "raw_text": raw,
                       "timing": timing, "publish_allowed": False,
                       "source_roi_identity_verified": False,
                       "source_mask_identity_verified": False}
                rf.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n");rf.flush()
                ef.write(json.dumps(ev, ensure_ascii=False, allow_nan=False) + "\n");ef.flush()
    report = obs.finalize(guard)
    write_json(output / "guard_shadow_report.json", report)
    return report


def select_dev_rows(rows: list[dict], targets: list[str] | None, max_instances: int) -> list[dict]:
    """Select only existing frozen DEV instance keys; never search other datasets."""
    if not targets:
        return rows[:max_instances]
    if len(targets) != len(set(targets)):
        raise ValueError("duplicate requested DEV target")
    index = {r["case_id"] + "/" + r["target_instance_id"]: r for r in rows}
    unknown = [x for x in targets if x not in index]
    if unknown:
        raise ValueError("target not found in pinned DEV60 instances: " + repr(unknown))
    requested = set(targets)
    return [r for r in rows if r["case_id"] + "/" + r["target_instance_id"] in requested]


def build_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["preflight", "live"], default="preflight")
    p.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    p.add_argument("--variant", choices=["baseline", "focus_v1"], default="baseline")
    p.add_argument("--max-instances", type=int, default=2)
    p.add_argument("--target", action="append", help="DEV instance case/instance, repeatable; e.g. fresh_dev_004/pred_0002")
    p.add_argument("--allow-full-dev60", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen2.5-VL-7B-Instruct"))
    return p.parse_args(argv)


def main(argv=None):
    args = build_args(argv)
    if args.output.exists():
        raise FileExistsError("refuse overwrite existing output path")
    if args.max_instances < 1 or args.max_instances > 171:
        raise ValueError("max instances must be 1..171")
    if args.max_instances > 3 and not args.allow_full_dev60 and not args.target:
        raise ValueError("more than 3 instances requires explicit --allow-full-dev60")
    runner = verified_runner(args.runner)
    # Preserve locked prompt/source/inference cohort validation unchanged.
    valid_rows, core, helper, focus = runner.prompt_check(argparse.Namespace(
        snapshot=runner.DEFAULT_SNAP,
        source=runner.DEFAULT_SOURCE,
        focus_script=runner.DEFAULT_FOCUS,
        raw=runner.DEFAULT_RAW,
        input=runner.DEFAULT_INPUT,
    ))
    if len(valid_rows) != 171 or len({r["case_id"] for r in valid_rows}) != 60:
        raise ValueError("DEV60 dataset size changed")
    selected_rows = select_dev_rows(valid_rows, args.target, args.max_instances)
    if len(selected_rows) > 3 and not args.allow_full_dev60:
        raise ValueError("more than 3 instances requires explicit --allow-full-dev60")
    guard = load_guard(args.repo / "src/fashion_multimodal_analysis/integration/prd32_output_guard.py")
    preflight = {"schema": "prd32_live_qwen_probe_preflight_v4",
                 "mode": args.mode, "target_variant": args.variant,
                 "images_validated": 60, "v5_instances_validated": 171,
                 "probe_instances": len(selected_rows),
                 "probe_requests": len(selected_rows) * 3,
                 "selected_dev_targets": [r["case_id"] + "/" + r["target_instance_id"] for r in selected_rows],
                 "runner_sha256": sha_file(args.runner),
                 "guard_sha256": sha_file(args.repo / "src/fashion_multimodal_analysis/integration/prd32_output_guard.py"),
                 "mask_roi_provenance_available": False,
                 "holdout_accessed": False,
                 "no_original_outputs_modified": True,
                 "production_approved": False}
    if args.mode == "preflight":
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "preflight.json", preflight)
        print("LIVE_SHADOW_PRECHECK_OK", json.dumps(preflight, ensure_ascii=False))
        return 0
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for live Qwen smoke; use preflight mode first")
    if torch.cuda.mem_get_info()[0]/1024**3 < 18:
        raise RuntimeError("needs >=18GiB free GPU, as original DEV60 runner")
    if not (args.model_dir / "config.json").is_file():
        raise FileNotFoundError("Qwen local config missing")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    start = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_dir), local_files_only=True, device_map="auto", torch_dtype=dtype,
        attn_implementation="sdpa", low_cpu_mem_usage=True).eval()
    processor = AutoProcessor.from_pretrained(str(args.model_dir), local_files_only=True,
                                               max_pixels=512*28*28)
    load_sec = round(time.perf_counter()-start, 2)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "preflight.json", {**preflight, "model_loaded": True,
                                                  "model_load_sec": load_sec})
    try:
        report = run_triplets(rows=selected_rows, runner=runner, core=core, helper=helper,
                              focus=focus, variant=args.variant, model=model,
                              processor=processor, output=args.output,
                              guard=guard, max_instances=len(selected_rows))
    except Exception as exc:
        write_json(args.output / "failure.json", {"status": "INCOMPLETE_NOT_APPROVED",
                                                    "error_type": type(exc).__name__,
                                                    "error": str(exc),
                                                    "publish_allowed": False})
        raise
    write_json(args.output / "run_summary.json", {
        "status": "DEV_ONLY_LIVE_SHADOW_COMPLETE_NOT_APPROVED",
        "model_inference_performed": True,
        "new_model_calls": len(selected_rows)*3,
        "model_variant": args.variant,
        "reused_original_prompt": True,
        "changed_original_qwen_outputs": False,
        "roi_mask_identity_verification": "NOT_AVAILABLE_AT_THIS_ENTRYPOINT",
        "model_load_seconds": load_sec,
        "guard_service_status": report["guard_service_status"],
        "publish_allowed": False,
        "holdout_used": False})
    print("LIVE_SHADOW_CALLS_DONE", report["tasks_correlated"],
          "GUARD", report["guard_service_status"], "NOT_PRODUCTION_APPROVED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("LIVE_SHADOW_BLOCKED", type(e).__name__, str(e), file=sys.stderr)
        sys.exit(2)
