#!/usr/bin/env python3
"""Run actual pinned V5 checkpoint for ten DEV60 images and capture producer-time masks.

Never edits original V5/Qwen data, never uses GT, never opens holdout.
If fresh predictions differ from frozen targets, STOP instead of relabeling/remapping.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from producer_contract_v7 import (TARGETS, EXPECTED_MODULE_HASHES, V5_CKPT_SHA256, V5_RAW_SHA256,
                                  canonical_mask_digest, canonical_sha,
                                  check_producer_match, digest_file,
                                  pinned_frozen_targets, refuse_restricted)

ROOT = Path("/workspace")
DEFAULT_REPO = ROOT / "fashion_multimodal_analysis"
DEFAULT_RAW = ROOT / "PRD32_DEV60_V5_REAL_20261030_run01"
DEFAULT_CKPT = DEFAULT_REPO / "outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth"


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def preflight(raw, image_root, repo, checkpoint):
    from PIL import Image
    for d in (raw, image_root, repo):
        if not d.is_dir():
            raise FileNotFoundError(d)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint unavailable: {checkpoint}; supply --checkpoint <actual V5 path>")
    if digest_file(checkpoint) != V5_CKPT_SHA256:
        raise ValueError("wrong checkpoint SHA-256; not running another V5 model")
    targets = pinned_frozen_targets(raw, image_root)
    for t in targets.values():
        with Image.open(t["image_path"]) as im:
            if list(im.size) != list(t["image_size"]):
                raise ValueError("image size mismatch")
        p = Path(t["frozen_mask_path"])
        if not p.is_file():
            raise FileNotFoundError("frozen DEV V5 mask not found: " + str(p))
    mandatory = [
        repo / "src/fashion_multimodal_analysis/evaluation/runtime.py",
        repo / "src/fashion_multimodal_analysis/segmentation/modeling.py",
        repo / "src/fashion_multimodal_analysis/common/schema.py",
    ]
    for p in mandatory:
        if not p.is_file():
            raise FileNotFoundError("V5 model module missing: " + str(p))
    current_modules = {p.relative_to(repo).as_posix(): digest_file(p) for p in mandatory}
    if current_modules != EXPECTED_MODULE_HASHES:
        raise ValueError("V5 source modules no longer match verified V6 runtime; refuse baseline drift")
    return targets, current_modules


def run_inference(raw, image_root, repo, checkpoint, out):
    import numpy as np
    from PIL import Image
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")
    if torch.cuda.mem_get_info()[0]/(1024**3) < 4:
        raise RuntimeError("insufficient CUDA memory for V5 inference")
    targets, module_hashes = preflight(raw, image_root, repo, checkpoint)
    # Select exact existing implementation, not a second segmentation algorithm.
    sys.path.insert(0, str(repo / "src"))
    from fashion_multimodal_analysis.common.schema import ID_TO_CLASS, CLASS_TO_ID
    from fashion_multimodal_analysis.evaluation.runtime import predict_timed
    from fashion_multimodal_analysis.segmentation.modeling import build_evaluation_model
    cp = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if cp.get("class_to_id") != CLASS_TO_ID or cp.get("epoch") != 15:
        raise ValueError("unexpected checkpoint class map/epoch")
    saved = cp.get("args", {})
    if saved.get("epochs") != 15:
        raise ValueError("unexpected checkpoint training epochs")
    model = build_evaluation_model(int(saved.get("min_size", 640)), int(saved.get("max_size", 1024)))
    model.load_state_dict(cp["model_state_dict"], strict=True)
    model.to(torch.device("cuda")).eval()
    del cp
    run_id = uuid.uuid4().hex
    out.mkdir(parents=True, exist_ok=False)
    mask_dir = out / "producer_masks"
    mask_dir.mkdir()
    events = []
    model_calls = 0
    try:
        for case in dict.fromkeys(k[0] for k in TARGETS):
            case_keys = [key for key in TARGETS if key[0] == case]
            frozen0 = targets[case_keys[0]]
            with Image.open(frozen0["image_path"]) as im:
                image = im.convert("RGB")
                boxes, labels, scores, masks, duration = predict_timed(
                    image, model, torch.device("cuda"), score_threshold=.4, mask_threshold=.5
                )
            model_calls += 1
            for key in case_keys:
                frozen = targets[key]
                _, iid = key
                idx = int(iid.split("_")[-1])
                if idx >= len(labels):
                    raise ValueError(f"live V5 lost target {case}/{iid}")
                category = ID_TO_CLASS.get(int(labels[idx]))
                if category not in CLASS_TO_ID:
                    raise ValueError("unknown live V5 category")
                new_mask = (np.asarray(masks[idx]) > 0).astype(np.uint8)
                if new_mask.shape != (frozen["image_size"][1], frozen["image_size"][0]):
                    raise ValueError("live V5 full-image mask shape changed")
                with Image.open(frozen["frozen_mask_path"]) as fm:
                    frozen_mask = np.asarray(fm.convert("L")) > 127
                producer = {"instance_id": iid, "category": category,
                            "score": float(scores[idx]), "bbox": [float(v) for v in boxes[idx]]}
                comparison = check_producer_match(
                    target=frozen, ins=producer, reference_mask=frozen_mask, produced_mask=new_mask
                )
                saved_mask_path = mask_dir / f"{case}_{iid}_live_mask.png"
                Image.fromarray(new_mask * 255, mode="L").save(saved_mask_path)
                event = {
                    "schema": "prd32_live_v5_mask_producer_event_v6",
                    "status": "V5_RUNTIME_PRODUCER_CAPTURED_SHADOW_NOT_RELEASED",
                    "run_id": run_id,
                    "case_id": case,
                    "instance_id": iid,
                    "source_image_path": frozen["image_path"],
                    "source_image_sha256": frozen["image_sha256"],
                    "source_mask_relative_path": saved_mask_path.relative_to(out).as_posix(),
                    "producer_mask_sha256": canonical_mask_digest(new_mask),
                    "mask_png_sha256": digest_file(saved_mask_path),
                    "producer_bbox_xyxy": producer["bbox"],
                    "producer_category": category,
                    "producer_score": producer["score"],
                    "v5_checkpoint_sha256": V5_CKPT_SHA256,
                    "v5_frozen_raw_sha256": V5_RAW_SHA256,
                    "producer_model_impl_sha256": module_hashes,
                    "producer_num_instances_in_image": int(len(labels)),
                    "producer_inference_ms": round(float(duration), 4),
                    "frozen_comparison": comparison,
                    "gt_used": False,
                    "holdout_accessed": False,
                    "publish_allowed": False,
                }
                event["producer_event_sha256"] = canonical_sha(event)
                events.append(event)
        if len(events) != len(TARGETS) or model_calls != len({case for case,_ in TARGETS}):
            raise ValueError("DEV12 producer capture incomplete")
        dump(out / "producer_events.json", events)
        dump(out / "producer_summary.json", {
            "status": "LIVE_V5_PRODUCER_CAPTURE_COMPLETE_SHADOW_ONLY",
            "run_id": run_id, "cases": len(set(k[0] for k in TARGETS)), "targets":len(events), "model_calls": model_calls,
            "all_frozen_comparisons_passed": True,
            "qwen_model_called": False, "producer_identity_verified_at_v5_forward": True,
            "qwen_request_bound_to_producer": False, "visual_target_binding_verified": False,
            "original_outputs_modified": False, "holdout_used": False,
            "production_approved": False,
        })
        print("LIVE_V5_PRODUCER_CAPTURE_COMPLETE", len(events), "targets;", model_calls, "image forwards; QWEN_NOT_CALLED; NO_RELEASE")
    except Exception as e:
        dump(out / "failure.json", {"status": "INCOMPLETE_NOT_RELEASED", "error_type": type(e).__name__,
                                      "message": str(e), "publish_allowed": False})
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("preflight", "live"), default="preflight")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    p.add_argument("--image-root", type=Path, default=ROOT / "fashion_data")
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    for path in [a.raw_dir, a.image_root, a.repo, a.checkpoint, a.out]:
        refuse_restricted(path)
    if a.out.exists():
        raise FileExistsError("refuse overwriting a previous experiment directory")
    targets, impl = preflight(a.raw_dir, a.image_root, a.repo, a.checkpoint)
    if a.mode == "preflight":
        a.out.mkdir(parents=True, exist_ok=False)
        dump(a.out / "preflight.json", {
            "status": "V5_PRODUCER_PREFLIGHT_PASS_NO_GPU_FORWARD",
            "targets": ["/".join(k) for k in targets], "checkpoint_sha256": V5_CKPT_SHA256,
            "module_source_hashes": impl, "holdout_used": False,
            "publish_allowed": False,
        })
        print("V5_PRODUCER_PREFLIGHT_PASS", len(targets), "DEV targets; NO_MODEL_FORWARD")
        return 0
    run_inference(a.raw_dir, a.image_root, a.repo, a.checkpoint, a.out)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as ex:
        print("V5_PRODUCER_V7_BLOCKED", type(ex).__name__, str(ex), file=sys.stderr)
        sys.exit(2)
