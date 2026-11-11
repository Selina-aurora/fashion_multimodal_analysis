#!/usr/bin/env python3
"""Deterministic, synthetic, host-API smoke. NEVER invokes a model or reads DEV files."""
from __future__ import annotations
import argparse
import json
from hashlib import sha256
from pathlib import Path
from runtime_bridge_v3 import RuntimeGuardBridge
from verified_guard_loader import load_guard, GUARD_RELATIVE


def simulate(guard_path: Path) -> dict:
    b = RuntimeGuardBridge(load_guard(guard_path))
    original = b"synthetic-DEV-image-used-only-in-API-test"
    imagehash = sha256(original).hexdigest()
    for variant in ("baseline", "focus_v1"):
        for i, inst in enumerate(("pred_0000", "pred_0001")):
            mask = f"synthetic-mask-{i}".encode()
            marked = f"synthetic-marked-{i}".encode()
            for task in ("tags", "vqa", "copy"):
                job = f"host-{variant}-{inst}-{task}"
                rid = b.dispatch(case_id="fresh_dev_001", instance_id=inst,
                                 source_image_bytes=original, expected_image_sha256=imagehash,
                                 bbox_xyxy=(1.0+i,2.0,101.0+i,102.0),
                                 v5_category="top" if i == 0 else "pants",
                                 roi_source_instance_id=inst, mask_source_instance_id=inst,
                                 roi_crop_bytes=b"synthetic-crop", expected_roi_sha256=sha256(b"synthetic-crop").hexdigest(),
                                 mask_bytes=mask, expected_mask_sha256=sha256(mask).hexdigest(),
                                 marked_image_bytes=marked, task=task, variant=variant,
                                 job_id=job, attempt_id="smoke01",
                                 vqa_question="What is the target color?" if task=="vqa" else None,
                                 duplicate_candidate=False)
                response={"tags":json.dumps({"category":"top"}),
                          "vqa":"blue", "copy":"A blue top."}[task]
                b.complete(host_job_id=job, response_host_job_id=job,
                           completed_request_id=rid, raw_text=response)
    return {"baseline": b.finalize_variant("baseline"),
            "focus_v1": b.finalize_variant("focus_v1")}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,help="GPU project root; SHA-verify the existing guard")
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    guard = args.repo / GUARD_RELATIVE if args.repo else Path(__file__).resolve().parent / 'reference_guard' / 'prd32_output_guard.py'
    reports=simulate(guard)
    summary={"schema":"prd32_runtime_bridge_synthetic_smoke_v3",
             "data":"SYNTHETIC_NOT_DEV60", "tests_with_mock_host":True,
             "model_called":False,"holdout_accessed":False,"production_approved":False,
             "variants":{k:{"instances":v["instance_count"],"jobs":v["host_job_correlated"],
                            "service_status":v["guard_service_status"],
                            "release_allowed":v["release_allowed"],
                            "flags":v["guard_flags"],
                            "overlap_cues":len(v["guard_geometric_overlap_pairs"])}
                        for k,v in reports.items()}}
    if any(v["service_status"] != "REVIEW_REQUIRED_NOT_RELEASED" for v in summary["variants"].values()):
        raise RuntimeError("guard status not fail-closed")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():
        raise SystemExit("output exists; choose fresh output path")
    args.out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == "__main__":main()
