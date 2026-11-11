#!/usr/bin/env python3
"""Read-only manifest generator for PRD 3.2 development evaluation.

Deduplicates the 3.1 -> 3.2 bridge by image SHA256 + target instance id.
Produces a CSV with intentionally empty HUMAN GT columns. The case/task
entries are development requests, NOT automatically human-verified categories.
Does not open sealed holdout files, run inference, or mutate bridge inputs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

COLS = [
    'target_key', 'image_id', 'image_path', 'target_instance_id',
    'predicted_target_bbox_xyxy', 'bridge_case_ids', 'bridge_tasks',
    'bridge_request_count', 'image_sha256', 'input_attribute_names_unverified',
    'human_review_status', 'human_target_category', 'human_primary_color',
    'human_pattern', 'human_visible_design', 'target_binding_correct', 'notes'
]


def blocked(path):
    return any(s in str(path).lower() for s in ('sealed', 'holdout'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bridge-input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if blocked(args.bridge_input) or blocked(args.output):
        ap.error('Cannot read or write sealed/holdout paths')
    if args.output.exists():
        ap.error('Output file exists: use another path to preserve history')
    if not args.bridge_input.is_file():
        ap.error('Bridge file missing')
    records = [json.loads(line) for line in args.bridge_input.read_text(encoding='utf-8').splitlines() if line.strip()]
    unique = {}
    tasks = Counter()
    for r in records:
        if r.get('schema') != 'prd32_engine_neutral_input_v1':
            raise ValueError('Unexpected bridge schema')
        path = r.get('image_path', '')
        if blocked(path):
            raise ValueError('Sealed/holdout image path found; refusing entire manifest')
        h = r.get('image_sha256', '')
        iid = r.get('target_instance_id', '')
        if not isinstance(h,str) or not h or not isinstance(iid,str) or not iid:
            raise ValueError('Missing image SHA or instance id')
        key=(h,iid)
        task=r.get('task','')
        tasks[task]+=1
        bbox = r.get('target_bbox_xyxy')
        if key not in unique:
            fingerprint = hashlib.sha256((h+'|'+iid).encode()).hexdigest()[:12]
            attr = r.get('predicted_attribute_context') or []
            names = [str(x['name']) for x in attr if isinstance(x,dict) and isinstance(x.get('name'),str)]
            unique[key]={
                'target_key':fingerprint, 'image_id':r.get('image_id',''),
                'image_path':path, 'target_instance_id':iid,
                'predicted_target_bbox_xyxy':json.dumps(bbox,ensure_ascii=False),
                'bridge_case_ids':[], 'bridge_tasks':[], 'bridge_request_count':0,
                'image_sha256':h,
                'input_attribute_names_unverified':';'.join(sorted(set(names))),
                'human_review_status':'NOT_REVIEWED',
                'human_target_category':'', 'human_primary_color':'',
                'human_pattern':'', 'human_visible_design':'',
                'target_binding_correct':'', 'notes':'',
            }
        rec=unique[key]
        if rec['image_path'] != path or rec['predicted_target_bbox_xyxy'] != json.dumps(bbox,ensure_ascii=False):
            raise ValueError('Conflicting target path or bbox for same image+instance key')
        rec['bridge_case_ids'].append(r.get('case_id',''))
        rec['bridge_tasks'].append(task)
        rec['bridge_request_count']+=1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=COLS)
        writer.writeheader()
        for row in unique.values():
            row['bridge_case_ids']=';'.join(map(str,row['bridge_case_ids']))
            row['bridge_tasks']=';'.join(map(str,row['bridge_tasks']))
            writer.writerow(row)
    print(json.dumps({'status':'DEV_MANIFEST_PREPARED_NO_GT',
        'bridge_requests':len(records), 'unique_image_instance_targets':len(unique),
        'unique_images':len({k[0] for k in unique}),
        'tasks':dict(tasks),'output':str(args.output),
        'human_review_status':'NOT_REVIEWED',
        'sealed_holdout_accessed':False},ensure_ascii=False,indent=2))
if __name__=='__main__':
    main()
