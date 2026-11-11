#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2026-10-30: PRD3.2 fixed-dev V5 instance cohort preflight and immutable snapshot.

Read-only with respect to all prior pipeline artifacts. Does NOT invoke GPU, select
'best' model, load human GT, or use sealed/test images. Do NOT treat V5 predicted
class counts as true category distribution.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_ROOT = Path('/workspace')
DEFAULT_SOURCE = DEFAULT_ROOT/'PRD32_20261029_NextBatch_GTFree'
DEFAULT_PREP = DEFAULT_ROOT/'PRD32_V5_QWEN_PREP_20261029_run01'
DEFAULT_RAW = DEFAULT_ROOT/'PRD32_V5_REAL_20261029_run01'
DEFAULT_GATE = DEFAULT_ROOT/'PRD32_V5_LIVE_PREFLIGHT_20261029_run01'
DEFAULT_DATA = DEFAULT_ROOT/'fashion_data'
DEFAULT_BASELINE = DEFAULT_ROOT/'PRD32_NEXTBATCH_GPU_20261029_run01/qwen_results.jsonl'
DEFAULT_FOCUS = DEFAULT_ROOT/'PRD32_FOCUS_GPU_20261029_run04/qwen_results.jsonl'
DEFAULT_FOCUS_SCRIPT = DEFAULT_ROOT/'PRD32_FOCUS_AB_PATCH_20261029/prd32_focus_ab.py'
DEFAULT_OUTPUT = DEFAULT_ROOT/'PRD32_FIXED_DEV_FREEZE_20261030'
TASKS = ('tags', 'vqa', 'copy')
QUESTION = '红框内目标服饰的主要可见颜色是什么？'
CLASS_ORDER = ('top','pants','skirt','outerwear','dress','shoe','bag','accessory')
EXPECTED_FROZEN_SOURCE_HASHES = {
 'v5_to_qwen_bridge.py':'affee7342646d808b4f0214f135d60d2d2611db920655c5e064268092e2dd02d',
 'next_batch.py':'08b35135223b49e2a2a7bbb448cdad3a4ce90a6a1373e57af29d1119102489f6',
 'prd32_guard.py':'deeaaea28d9f5e80805da013ef23689fc0df95ee8a6514c583beb468b766fc0f',
 'frozen_helpers/prd32_integrated_pipeline_20261027.py':'90dda6974dc533aaae30fe1343095864ee74d536713ad80ce2006617286fa6b4',
}

def sha_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()

def sha_text(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()
def readj(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def readjl(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return [json.loads(x) for x in f if x.strip()]
def writej(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def writejl(path,rows):
    with Path(path).open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def bbox_iou(a,b):
    if any(not isinstance(z,(int,float)) or not math.isfinite(z) for z in a+b): return 0.0
    ix=max(0.,min(a[2],b[2])-max(a[0],b[0]))
    iy=max(0.,min(a[3],b[3])-max(a[1],b[1]))
    ai=(a[2]-a[0])*(a[3]-a[1]);bi=(b[2]-b[0])*(b[3]-b[1])
    return ix*iy/(ai+bi-ix*iy) if ai+bi>ix*iy else 0.0

def verify_original_source(source):
    verified={}
    for rel, expected in EXPECTED_FROZEN_SOURCE_HASHES.items():
        p=source/rel
        if not p.is_file():raise ValueError('FROZEN_SOURCE_MISSING:'+str(p))
        got=sha_file(p)
        if got!=expected:raise ValueError('FROZEN_SOURCE_CHANGED:'+rel+'; do not overwrite original')
        verified[rel]=got
    return verified

def load_original_bridge(source):
    p=source/'v5_to_qwen_bridge.py'
    spec=importlib.util.spec_from_file_location('prd32_original_bridge_1030',p)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def validate_existing_pair(path,manifest_by_id):
    """Read existing 10/29 Qwen records purely to quantify fixed-cohort coverage."""
    path=Path(path)
    if not path.is_file():return {'status':'NOT_FOUND','path':str(path)}
    rows=readjl(path)
    keys=set();bad=[]
    for row in rows:
        cid=row.get('case_id');iid=row.get('target_instance_id');task=row.get('task')
        key=(cid,iid,task)
        if key in keys:bad.append('DUPLICATE_REQUEST:'+str(key))
        keys.add(key)
        if task not in TASKS or (cid,iid) not in manifest_by_id:bad.append('UNKNOWN_FROZEN_INSTANCE:'+str(key))
        else:
            frozen=manifest_by_id[(cid,iid)]
            for rf,mf in [('v5_predicted_bbox_xyxy','target_bbox_xyxy'),('v5_category_unverified','predicted_category_v5_unverified'),('v5_predicted_score_uncalibrated','predicted_score_v5_uncalibrated')]:
                if row.get(rf)!=frozen[mf]:bad.append('PREDICTION_MISMATCH:'+str(key)+':'+rf)
        if row.get('request_id')!=f'{cid}__{iid}__{task}':bad.append('REQUEST_ID_MISMATCH:'+str(key))
        if row.get('status')!='GENERATED_REVIEW_REQUIRED':bad.append('NOT_SUCCESSFUL:'+str(key))
        if row.get('human_label_or_gt_bbox_used') is not False:bad.append('HUMAN_GT_IN_INPUT:'+str(key))
        if row.get('publish_allowed') is not False:bad.append('PUBLISH_NOT_BLOCKED:'+str(key))
    full={(cid,iid,t) for cid,iid in manifest_by_id for t in TASKS}
    return {'status':'INCOMPATIBLE_WITH_FIXED_COHORT' if bad else ('FULL_COVERAGE' if keys==full else 'PARTIAL_COVERAGE'),
      'path':str(path),'sha256':sha_file(path),'observed_requests':len(rows),
      'observed_images':len(set(k[0] for k in keys)),
      'observed_instances':len(set((k[0],k[1]) for k in keys)),
      'required_requests':len(full),'missing_requests':len(full-keys),
      'errors':bad[:30]}

def snapshot(args):
    if args.output.exists():raise FileExistsError('REFUSE_OVERWRITE:'+str(args.output))
    original=verify_original_source(args.source)
    if not (args.prep/'bridge_preparation_summary.json').is_file():raise FileNotFoundError('PREP_NOT_FOUND; locate approved original GT-free bridge preparation')
    bridge=load_original_bridge(args.source)
    # This is the original 10/29 V5 checkpoint/threshold/SHA/path validator, not a rewritten approximation.
    instances,v5_summary=bridge.validated(args.raw,args.gate,args.data_root)
    old_prep=readj(args.prep/'bridge_preparation_summary.json')
    prepared_path=args.prep/'gt_free_all_instances.jsonl'
    if old_prep.get('v5_raw_predictions_sha256')!=v5_summary['v5_predictions_sha256']:
        raise ValueError('PREP_V5_SHA_MISMATCH')
    if sha_file(prepared_path)!=old_prep.get('all_instances_manifest_sha256'):
        raise ValueError('PREP_MANIFEST_HASH_MISMATCH')
    if instances!=readjl(prepared_path):raise ValueError('BRIDGE_PREPARED_INSTANCES_CHANGED')
    # Deterministic ordering, all V5 predicted instances; no GT / best-ROI selection.
    instances.sort(key=lambda x:(x['case_id'],x['target_instance_id']))
    identities=[(x['case_id'],x['target_instance_id']) for x in instances]
    if len(identities)!=len(set(identities)):raise ValueError('DUPLICATE_INSTANCE_ID')
    cases=sorted(set(cid for cid,_ in identities))
    if len(cases)!=12:raise ValueError('EXPECTED_12_APPROVED_TRAIN_DEV_IMAGES; investigate source')
    if not instances:raise ValueError('EMPTY_V5_INSTANCE_SET')
    by_case=defaultdict(list)
    for row in instances:by_case[row['case_id']].append(row)
    duplicate_pairs=[]
    for cid,members in by_case.items():
        for i,a in enumerate(members):
            for b in members[i+1:]:
                iou=bbox_iou(a['target_bbox_xyxy'],b['target_bbox_xyxy'])
                if iou>=args.iou_threshold:
                    duplicate_pairs.append({'case_id':cid,'instance_a':a['target_instance_id'],
                                            'instance_b':b['target_instance_id'],'bbox_iou':round(iou,6),
                                            'status':'POSSIBLE_DUPLICATE_REVIEW_ONLY'})
    req=[{'request_id':f"{r['case_id']}__{r['target_instance_id']}__{task}",
          'case_id':r['case_id'],'target_instance_id':r['target_instance_id'],
          'task':task,'vqa_question':QUESTION if task=='vqa' else None}
          for r in instances for task in TASKS]
    compare_id={(x['case_id'],x['target_instance_id']):x for x in instances}
    historic={'baseline':validate_existing_pair(args.baseline,compare_id),
              'focus_run04':validate_existing_pair(args.focus,compare_id)}
    # Freeze operation does NOT generate model outputs or render/upload images.
    args.output.mkdir(parents=True,exist_ok=False)
    writejl(args.output/'frozen_v5_instances.jsonl',instances)
    writejl(args.output/'frozen_request_ids.jsonl',req)
    with (args.output/'instance_coverage.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['case_id','instance_id','v5_predicted_category_NOT_GT','v5_score_uncalibrated','bbox_xyxy','image_sha256'])
        writer.writeheader()
        for r in instances:writer.writerow({'case_id':r['case_id'],'instance_id':r['target_instance_id'],
            'v5_predicted_category_NOT_GT':r['predicted_category_v5_unverified'],
            'v5_score_uncalibrated':r['predicted_score_v5_uncalibrated'],
            'bbox_xyxy':json.dumps(r['target_bbox_xyxy']), 'image_sha256':r['image_sha256']})
    writej(args.output/'possible_duplicate_pairs.json',duplicate_pairs)
    c=Counter(r['predicted_category_v5_unverified'] for r in instances)
    per_case={cid:len(rows) for cid,rows in by_case.items()}
    hist={'schema':'prd32_20261030_fixed_dev_coverage_v1','scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',
      'images':len(cases),'instances':len(instances),'requests_for_three_tasks':len(req),
      'case_ids':cases,'v5_predicted_class_counts_NOT_GT':{x:c[x] for x in CLASS_ORDER},
      'instances_per_image':per_case,'bbox_iou_threshold':args.iou_threshold,
      'possible_duplicate_pairs':len(duplicate_pairs),'historical_comparison_coverage':historic,
      'labels_or_gt_used_for_freeze':False,'heldout_images_opened':False,
      'key_limitations':['Existing source includes previously inspected images: development diagnostic only',
       'Predicted V5 labels are NOT ground truth; cannot use these numbers as class accuracy',
       '12 images may be too small or have missing class/scenario coverage',
       'Old 3-image baseline and focus are not full fixed-cohort regressions']}
    writej(args.output/'coverage_report.json',hist)
    focus_hash=sha_file(args.focus_script) if args.focus_script.is_file() else None
    f={'schema':'prd32_20261030_fixed_dev_freeze_v1','status':'FROZEN_DEV_PILOT_ONLY_NOT_OFFICIAL_ACCEPTANCE',
       'scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY','case_ids':cases,'images':len(cases),'instances':len(instances),
       'tasks':list(TASKS),'qwen_vqa_question':QUESTION,'vqa_question_sha256':sha_text(QUESTION),
       'v5_sha256':v5_summary['v5_predictions_sha256'],
       'prepared_instance_manifest_sha256':sha_file(prepared_path),
       'frozen_instance_manifest_sha256':sha_file(args.output/'frozen_v5_instances.jsonl'),
       'request_manifest_sha256':sha_file(args.output/'frozen_request_ids.jsonl'),
       'frozen_original_source_sha256':original,'focus_trial_script_sha256':focus_hash,
       'gt_or_human_labels_in_inference':False,'holdout_used':False,'ready_to_select_main_strategy':False,
       'official_prd_acceptance':False,'publish_allowed':False}
    writej(args.output/'freeze.json',f)
    print('PRD32_1030_FIXED_DEV_SNAPSHOT_OK')
    print('DEV_IMAGES:',len(cases),'V5_INSTANCES:',len(instances),'PLANNED_TASKS:',len(req))
    print('POSSIBLE_DUPLICATE_PAIRS:',len(duplicate_pairs),'(review only)')
    print('V5_PREDICTED_CLASSES_NOT_GT:',json.dumps(hist['v5_predicted_class_counts_NOT_GT'],ensure_ascii=False))
    for k,v in historic.items():print('HISTORICAL_'+k.upper()+':',v['status'],'covered_images:',v.get('observed_images',0),'missing_requests:',v.get('missing_requests','N/A'))
    print('OUTPUT:',args.output)
    print('PILOT_ONLY_NOT_STABLE_NOT_PRD_ACCEPTANCE')
    return hist

def verify(args):
    f=readj(args.output/'freeze.json')
    if f['schema']!='prd32_20261030_fixed_dev_freeze_v1':raise ValueError('BAD_FREEZE_SCHEMA')
    if sha_file(args.output/'frozen_v5_instances.jsonl')!=f['frozen_instance_manifest_sha256']:raise ValueError('FROZEN_INSTANCE_TAMPERED')
    if sha_file(args.output/'frozen_request_ids.jsonl')!=f['request_manifest_sha256']:raise ValueError('FROZEN_REQUESTS_TAMPERED')
    if sha_file(args.prep/'gt_free_all_instances.jsonl')!=f['prepared_instance_manifest_sha256']:raise ValueError('PREP_CHANGED')
    if sha_file(args.raw/'v5_raw_predictions.jsonl')!=f['v5_sha256']:raise ValueError('V5_CHANGED')
    verify_original_source(args.source)
    if f['focus_trial_script_sha256'] and args.focus_script.is_file() and sha_file(args.focus_script)!=f['focus_trial_script_sha256']:
        raise ValueError('FOCUS_PROMPT_CODE_CHANGED')
    print('FIXED_DEV_VERIFY_OK',f['images'],'images',f['instances'],'instances',f['instances']*3,'planned requests')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','verify'))
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--raw',type=Path,default=DEFAULT_RAW)
    parser.add_argument('--gate',type=Path,default=DEFAULT_GATE)
    parser.add_argument('--prep',type=Path,default=DEFAULT_PREP)
    parser.add_argument('--data-root',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--baseline',type=Path,default=DEFAULT_BASELINE)
    parser.add_argument('--focus',type=Path,default=DEFAULT_FOCUS)
    parser.add_argument('--focus-script',type=Path,default=DEFAULT_FOCUS_SCRIPT)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--iou-threshold',type=float,default=.8)
    args=parser.parse_args()
    try:
        if not 0<args.iou_threshold<=1:raise ValueError('BAD_IOU_THRESHOLD')
        (snapshot if args.command=='freeze' else verify)(args)
    except Exception as exc:
        print('PRD32_FIXED_DEV_BLOCKED',type(exc).__name__+': '+str(exc),file=sys.stderr)
        raise SystemExit(2)
if __name__=='__main__':main()
