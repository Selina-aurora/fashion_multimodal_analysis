#!/usr/bin/env python3
"""Run real Qwen from freshly produced V5 bbox identities, via an auditable staged handoff.

The V5 mask is VERIFIED and bound to the target event but not fed to Qwen;
Qwen still sees the original image with the target rectangle, not the mask.
Never overwrites original model predictions or releases a result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from producer_contract_v7 import (TARGETS, QWEN_SMOKE_TARGETS, canonical_mask_digest, digest_file,
                                  pinned_frozen_targets, read_json, refuse_restricted,
                                  verified_event)
from run_live_probe_v4 import run_triplets, verified_runner, write_json
from verified_guard_loader import load_guard

ROOT = Path('/workspace')


def load_and_verify_producer(producer_dir: Path, raw_dir: Path, image_root: Path):
    from PIL import Image
    import numpy as np
    summary = read_json(producer_dir / 'producer_summary.json')
    if summary.get('status') != 'LIVE_V5_PRODUCER_CAPTURE_COMPLETE_SHADOW_ONLY' or summary.get('model_calls') != len({case for case,_ in TARGETS}) or summary.get('targets') != len(TARGETS):
        raise ValueError('V5 producer step incomplete')
    if summary.get('qwen_model_called') is not False or summary.get('holdout_used') is not False or summary.get('production_approved') is not False:
        raise ValueError('producer summary unsafe')
    rows = read_json(producer_dir / 'producer_events.json')
    if len(rows) != len(TARGETS):
        raise ValueError('expected all 12 producer events')
    frozen = pinned_frozen_targets(raw_dir, image_root)
    found = {}
    for e in rows:
        if 'source_mask_relative_path' not in e:
            raise ValueError('producer mask path missing')
        m = (producer_dir / e['source_mask_relative_path']).resolve(strict=True)
        if not m.is_relative_to(producer_dir.resolve(strict=True)):
            raise ValueError('mask escapes producer output directory')
        f = verified_event(e, m, frozen)
        key = (e['case_id'], e['instance_id'])
        if key in found:
            raise ValueError('duplicate producer event')
        if not isinstance(e.get('frozen_comparison'), dict) or e['frozen_comparison'].get('bbox_iou', 0) < .995 or e['frozen_comparison'].get('mask_iou', 0) < .98:
            raise ValueError('producer/frozen comparison did not pass')
        # Re-evaluate mask alignment against frozen pixels, not just signed claims.
        from producer_contract_v7 import mask_iou, bbox_iou
        with Image.open(m) as fresh, Image.open(f['frozen_mask_path']) as old:
            similarity = mask_iou(np.asarray(fresh.convert('L')), np.asarray(old.convert('L')))
        if similarity < .98 or bbox_iou(e['producer_bbox_xyxy'], f['expected_bbox']) < .995:
            raise ValueError('producer event not aligned with frozen target')
        if digest_file(Path(f['image_path'])) != e['source_image_sha256']:
            raise ValueError('original image changed since producer')
        found[key] = e
    if set(found) != set(TARGETS):
        raise ValueError('wrong DEV targets in producer output')
    return found, frozen


def prepare_qwen_rows(produced: dict, frozen: dict):
    rows = []
    for key in QWEN_SMOKE_TARGETS:
        e = produced[key]
        f = frozen[key]
        rows.append({
            'case_id': key[0],
            'target_instance_id': key[1],
            'image_path': f['image_path'],
            'image_sha256': f['image_sha256'],
            'target_bbox_xyxy': e['producer_bbox_xyxy'],
            'predicted_category_v5_unverified': e['producer_category'],
            'predicted_score_v5_uncalibrated': e['producer_score'],
            'producer_event_sha256': e['producer_event_sha256'],
        })
    return rows


def evidence_join(out: Path, produced: dict, frozen: dict, guard_report: dict):
    from producer_contract_v7 import digest_bytes
    from live_observer_v4 import sha_file
    rows = []
    qwen = [json.loads(s) for s in (out/'qwen_shadow_results.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
    evs = [json.loads(s) for s in (out/'host_call_events.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
    by_request = {e['host_request_id']: e for e in evs}
    if len(qwen) != len(QWEN_SMOKE_TARGETS)*3 or len(evs) != len(QWEN_SMOKE_TARGETS)*3 or len(by_request) != len(QWEN_SMOKE_TARGETS)*3:
        raise ValueError('incomplete/duplicate host Qwen events')
    expected_tasks = {(case,iid,task) for case,iid in QWEN_SMOKE_TARGETS for task in ("tags","vqa","copy")}
    actual_tasks = {(q["case_id"],q["target_instance_id"],q["task"]) for q in qwen}
    if actual_tasks != expected_tasks:
        raise ValueError("Qwen tasks differ from selected smoke cohort")
    for q in qwen:
        key = (q['case_id'], q['target_instance_id'])
        e = produced.get(key)
        h = by_request.get(q['host_request_id'])
        if e is None or h is None or q['task'] not in {'tags','vqa','copy'}:
            raise ValueError('orphan Qwen task')
        if q['image_sha256'] != e['source_image_sha256'] or h['bbox_xyxy'] != e['producer_bbox_xyxy'] or q['v5_bbox_xyxy'] != e['producer_bbox_xyxy']:
            raise ValueError('producer -> Qwen target bbox/image not preserved')
        if q['v5_category_unverified'] != e['producer_category']:
            raise ValueError('producer category not preserved')
        if q['host_request_id'] != f'{key[0]}__{key[1]}__{q["task"]}__baseline__attempt_01':
            raise ValueError('unexpected host request correlation')
        if (h['marked_sha256'] != q['marked_image_sha256'] or h['image_sha256'] != e['source_image_sha256']
                or h['case_id'] != key[0] or h['instance_id'] != key[1] or h['task'] != q['task']):
            raise ValueError('Qwen host marked-image or identity mismatch')
        if digest_bytes(q['raw_text'].encode()) != h['response_sha256']:
            raise ValueError('response data changed')
        marked_matches = [m for m in (out/'marked_inputs').glob('*_marked_full_image.jpg') if digest_file(m) == q['marked_image_sha256']]
        if len(marked_matches) != 1:
            raise ValueError('marked image missing or ambiguous')
        rows.append({'case_id':key[0], 'instance_id':key[1], 'task':q['task'],
                     'host_request_id':q['host_request_id'], 'producer_event_sha256':e['producer_event_sha256'],
                     'live_mask_sha256':e['producer_mask_sha256'], 'image_sha256':e['source_image_sha256'],
                     'producer_bbox_xyxy':e['producer_bbox_xyxy'],
                     'qwen_marked_image_sha256':q['marked_image_sha256'],
                     'qwen_response_sha256':h['response_sha256'],
                     'identity_link_status':'V5_PRODUCER_EVENT_TO_QWEN_HOST_BOUND',
                     'mask_actually_fed_to_qwen':False, 'publish_allowed':False})
    write_json(out/'producer_to_qwen_identity_join.json', rows)
    report = {
        'schema': 'prd32_v5_runtime_to_qwen_host_v7',
        'status': 'DEV12_PRODUCER_QWEN_SMOKE4_STAGED_SHADOW_COMPLETE_NOT_RELEASED',
        'producer_instances':len(produced), 'qwen_smoke_instances':len(QWEN_SMOKE_TARGETS), 'qwen_tasks':len(qwen),
        'staged_producer_to_qwen_id_verified':True,
        'v5_masks_verified_as_producer_outputs':True,
        'mask_actually_fed_to_qwen':False,
        'qwen_visual_attention_verified':False,
        'synchronous_host_correlation_only':True,
        'independent_async_scheduler_verified':False,
        'guard_service_status': guard_report.get('guard_service_status'),
        'all_outputs_review_only': True,
        'model_training_performed':False, 'original_model_outputs_modified':False,
        'independent_holdout_used':False, 'production_approved':False,
    }
    if report['guard_service_status'] != 'REVIEW_REQUIRED_NOT_RELEASED':
        raise ValueError('unexpected Guard decision')
    write_json(out/'v7_end_to_end_shadow_report.json', report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['preflight','live'], default='preflight')
    p.add_argument('--producer-dir', type=Path, required=True)
    p.add_argument('--raw-dir', type=Path, default=ROOT/'PRD32_DEV60_V5_REAL_20261030_run01')
    p.add_argument('--image-root', type=Path, default=ROOT/'fashion_data')
    p.add_argument('--repo', type=Path, default=ROOT/'fashion_multimodal_analysis')
    p.add_argument('--runner', type=Path, default=ROOT/'PRD32_DEV60_QWEN_AB_20261030/dev60_qwen_ab.py')
    p.add_argument('--model-dir', type=Path, default=ROOT/'models/Qwen2.5-VL-7B-Instruct')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    for path in (a.producer_dir,a.raw_dir,a.image_root,a.repo,a.runner,a.model_dir,a.out):
        refuse_restricted(path)
    if a.out.exists():
        raise FileExistsError('refuse overwriting existing experiment')
    produced, frozen = load_and_verify_producer(a.producer_dir, a.raw_dir, a.image_root)
    runner = verified_runner(a.runner)
    # Pin original DEV60 prompt templates, model helper, snapshot, and input dataset.
    snapshot_rows, core, helper, focus = runner.prompt_check(argparse.Namespace(
        snapshot=runner.DEFAULT_SNAP, source=runner.DEFAULT_SOURCE,
        focus_script=runner.DEFAULT_FOCUS, raw=runner.DEFAULT_RAW, input=runner.DEFAULT_INPUT))
    if len(snapshot_rows) != 171:
        raise ValueError('original DEV60 snapshot unexpectedly changed')
    guard = load_guard(a.repo/'src/fashion_multimodal_analysis/integration/prd32_output_guard.py')
    target_rows = prepare_qwen_rows(produced, frozen)
    if a.mode == 'preflight':
        a.out.mkdir(parents=True,exist_ok=False)
        write_json(a.out/'preflight.json', {'status':'QWEN_PRODUCER_HANDOFF_PREFLIGHT_PASS',
            'producer_events_verified':len(produced), 'qwen_smoke_instances':len(QWEN_SMOKE_TARGETS), 'qwen_requests_expected':len(QWEN_SMOKE_TARGETS)*3,
            'mask_not_fed_to_qwen':True,'holdout_used':False,'publish_allowed':False})
        print('QWEN_PRODUCER_HANDOFF_PREFLIGHT_PASS',len(produced),'producer events',len(QWEN_SMOKE_TARGETS)*3,'planned requests')
        return 0
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]/(1024**3) < 18:
        raise RuntimeError('CUDA GPU with >=18GiB free is required for Qwen')
    if not (a.model_dir/'config.json').is_file():
        raise FileNotFoundError('local Qwen config missing')
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(a.model_dir), local_files_only=True, device_map='auto', torch_dtype=dtype,
        attn_implementation='sdpa', low_cpu_mem_usage=True).eval()
    processor = AutoProcessor.from_pretrained(str(a.model_dir),local_files_only=True,max_pixels=512*28*28)
    a.out.mkdir(parents=True,exist_ok=False)
    write_json(a.out/'preflight.json',{'status':'QWEN_PRODUCER_HANDOFF_LIVE_RUNNING',
        'producer_events_verified':len(produced), 'qwen_smoke_instances':len(QWEN_SMOKE_TARGETS), 'qwen_requests_expected':len(QWEN_SMOKE_TARGETS)*3, 'holdout_used':False, 'publish_allowed':False})
    try:
        report = run_triplets(rows=target_rows, runner=runner, core=core, helper=helper,
            focus=focus, variant='baseline', model=model,processor=processor,
            output=a.out,guard=guard,max_instances=len(QWEN_SMOKE_TARGETS))
        result = evidence_join(a.out, produced, frozen, report)
    except Exception as exc:
        write_json(a.out/'failure.json',{'status':'INCOMPLETE_NOT_RELEASED',
            'error_type':type(exc).__name__,'message':str(exc),'publish_allowed':False})
        raise
    print('V5_TO_QWEN_V7_STAGED_SHADOW_COMPLETE',result['qwen_tasks'],'calls; GUARD',result['guard_service_status'],'NO_RELEASE')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as ex:
        print('QWEN_V7_BLOCKED',type(ex).__name__,str(ex),file=sys.stderr)
        sys.exit(2)
