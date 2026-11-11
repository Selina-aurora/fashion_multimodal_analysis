#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create ISOLATED fault-injection input from verified, never modified PRD 3.1 freeze.

This is a reliability test fixture, NOT another PRD 3.1 freeze, not a new GT,
not a new set of model accuracy measurements, and never the sealed holdout.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, sys
from pathlib import Path
from PIL import Image

SELECT_IDS=('vqa_color_01','tags_01','copy_01')

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024), b''):h.update(x)
    return h.hexdigest()

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--freeze-dir',default='/workspace/prd31_stage_freeze_20261026')
    ap.add_argument('--output',default='/workspace/PRD32_P0_MIXED_FIXTURE_20261027')
    args=ap.parse_args(argv)
    freeze=Path(args.freeze_dir).resolve()
    out=Path(args.output).resolve()
    if any(x in str(p).lower() for p in (freeze,out) for x in ('holdout','sealed_do_not_open')):raise ValueError('Sealed holdout path is prohibited')
    if out==freeze or freeze in out.parents:raise ValueError('Fault fixture must be OUTSIDE original freeze directory')
    if out.exists():raise FileExistsError('Will not overwrite fixture: '+str(out))
    source=freeze/'bridge_inputs.jsonl'
    if not source.is_file():raise FileNotFoundError(source)
    # Actual freeze status/version is validated independently by the V1 helper.
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import prd32_integrated_pipeline_20261027 as v1
    info=v1.validate_freeze(freeze,source)
    recs=v1.read_jsonl(source)
    matches={r['case_id']:r for r in recs if isinstance(r,dict) and isinstance(r.get('case_id'),str)}
    if any(sum(r.get('case_id')==cid for r in recs)!=1 for cid in SELECT_IDS):
        raise ValueError('Expected EXACTLY one of each selected source case')
    selected=[]
    for cid in SELECT_IDS:
        item=copy.deepcopy(matches[cid]);v1.validate_record(item,check_assets=True)
        selected.append(item)
    base=selected[0]
    with Image.open(base['image_path']) as im: w,h=im.size
    # All mutated request IDs and instance IDs are unique, so a bad item cannot
    # poison a good record through the cross-request identity consistency gate.
    invalid=[]
    def corrupted(label,patch):
        b=copy.deepcopy(base)
        b['case_id']='injected_'+label
        b['image_id']='synthetic_'+label
        b['target_instance_id']='synthetic_pred_'+label
        b.update(patch)
        invalid.append((label,b))
    corrupted('bbox_negative',{'target_bbox_xyxy':[-1,2,25,35]})
    corrupted('bbox_outside',{'target_bbox_xyxy':[1,1,w+5,h+5]})
    corrupted('bbox_zero_area',{'target_bbox_xyxy':[5,5,5,15]})
    corrupted('bbox_null',{'target_bbox_xyxy':None})
    corrupted('invalid_class',{'target_category':'shirt'})
    corrupted('category_conflict',{'target_category':'top','predicted_category':'outerwear'})
    corrupted('task_type_list',{'task':['tags']})
    corrupted('missing_instance',{'target_instance_id':None})
    corrupted('wrong_sha',{'image_sha256':'0'*64})
    corrupted('missing_image',{'image_path':str(out/'intentionally_missing.jpg')})
    corrupt_json='{not valid JSON'
    nan_row=copy.deepcopy(base)
    nan_row.update({'case_id':'injected_nonfinite_bbox', 'image_id':'synthetic_nan',
                    'target_instance_id':'synthetic_pred_nan',
                    'target_bbox_xyxy':[0, float('nan'),20,20]})
    lines=[json.dumps(item,ensure_ascii=False,allow_nan=False) for item in selected]
    lines += [json.dumps(row,ensure_ascii=False,allow_nan=False) for _,row in invalid]
    lines += [corrupt_json,json.dumps(nan_row,ensure_ascii=False,allow_nan=True)]
    out.mkdir(parents=True,exist_ok=False)
    bridge=out/'fault_injection_only.jsonl'
    bridge.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fixture={
        'schema':'prd32_isolated_fault_injection_fixture_v1',
        'purpose':'P0_fault_injection_only',
        'not_prd_acceptance':True,'do_not_use_for_accuracy':True,
        'source_frozen_bridge_sha256':digest(source),
        'injected_bridge_sha256':digest(bridge),
        'source_candidate_id':info.get('candidate_id'),
        'valid_source_case_ids':list(SELECT_IDS),
        'source_request_count':len(recs),
        'injected_request_count':len(lines),
        'expected_valid':3,'expected_blocked':12,
        'injected_failure_types':[name for name,_ in invalid]+['malformed_json','nonfinite_bbox'],
        'original_files_modified':False,'holdout_used':False,
        'note':'Original frozen bridge only read and validated. All mutations are in this separate fixture directory.'
    }
    (out/'manifest.json').write_text(json.dumps(fixture,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('ISOLATED_FAULT_FIXTURE_CREATED',json.dumps({'directory':str(out),'requests':len(lines),'expected_valid':3,'expected_blocked':12,'source_bridge_sha256':fixture['source_frozen_bridge_sha256']},ensure_ascii=False))
    return 0

if __name__=='__main__':sys.exit(main())
