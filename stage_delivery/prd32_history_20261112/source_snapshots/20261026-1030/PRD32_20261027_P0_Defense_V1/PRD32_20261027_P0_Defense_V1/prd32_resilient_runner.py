#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resilient PRD 3.2 DEVELOPMENT replay: per-case boundary validation and fallbacks.

Uses the existing prd32_integrated_pipeline_20261027.py model functions but DOES
NOT call its batch-wide all-or-nothing main(). Keeps original frozen 3.1 files
read-only. NOT for untrusted public uploads and NOT a production API.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from prd32_defensive_gate import (TASKS,guard_input_record, sanitize_service_output,
                                  read_jsonl_lenient, restricted, sha256_file)

DEFAULT_ROOT=Path('/workspace')


def load_v1(path):
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(p)
    spec=importlib.util.spec_from_file_location('prd32_readonly_v1_helpers',p)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select_with_guard(records):
    ids=Counter(r.get('case_id') for r in records if isinstance(r,dict) and isinstance(r.get('case_id'),str))
    audits=[]
    for n,r in enumerate(records,1):
        g=guard_input_record(r,image_check=True)
        cid=g['identity'].get('case_id')
        if cid and ids[cid]>1:
            g['decision']='BLOCKED';g['guarded_record']=None
            g['issues'].append({'code':'DUPLICATE_CASE_ID','detail':'Ambiguous case identity'})
        g['source_line']=n
        audits.append(g)
    return audits


def fallback_response(g):
    identity=g.get('identity',{})
    cid=identity.get('case_id') or f'__source_line_{g["source_line"]}'
    return {'case_id':cid,'task':identity.get('task'),
            'status':'BLOCKED_INVALID_INPUT','payload':None,
            'issues':g['issues'],'provenance':identity,
            'unconfirmed':['category','primary_color','pattern','visible_design','visible_decoration'],
            'source_line':g['source_line'], 'inference_performed':False}


def infer_case(v1,model,processor,rec,images_dir,input_mode,roi_padding):
    cid=rec['case_id'];task=rec['task']
    source={'image_id':rec['image_id'],'target_instance_id':rec['target_instance_id'],
            'image_sha256':rec.get('image_sha256'),
            'bbox_origin':rec.get('target_bbox_origin')}
    pred={'case_id':cid,'task':task,**source,'status':'GENERATED_NOT_ACCEPTED',
          'human_verified':False,'upstream_predictions_are_gt':False}
    marked=v1.make_marked_image(rec,images_dir,input_mode,roi_padding)
    # No unverified 3.1 attributes given to the model by default.
    prompt=v1.make_prompt(rec,input_mode=input_mode,pass_unverified=False)
    raw,timing=v1.run_model_case(model,processor,rec,marked,prompt,
                                  task_limits={'vqa':128,'tags':170,'copy':120})
    pred.update({'raw_text':raw,'input_sent':str(marked),'prompt':prompt,**timing})
    basic={'case_id':cid,'task':task,'payload':raw.strip()}
    if task=='tags':
        parsed=v1.parse_tag_output(raw)
        pred.update(parsed)
        pred['model_native_json']=parsed['raw_native_json']
        if parsed['canonical_service_json'] is not None:
            basic['canonical_attributes_v2']=parsed['canonical_service_json']
            basic['payload']=v1.bridge_tags_adapter(parsed['canonical_service_json'])
        else:
            basic['payload']=None
    service=sanitize_service_output(basic,prediction=pred)
    service['provenance'].update(source)
    return pred,service


def run(args):
    root=Path(args.output)
    bridge=Path(args.bridge)
    freeze=Path(args.freeze_dir)
    v1_path=Path(args.v1_script)
    if any(restricted(p) for p in (root,bridge,freeze,v1_path)):
        raise ValueError('Restricted sealed/holdout path not allowed')
    v1=load_v1(v1_path)
    # Fail closed on wrong freeze identity; this is a batch-level integrity gate.
    freeze_info=v1.validate_freeze(freeze,bridge)
    input_rows=read_jsonl_lenient(bridge)
    if not input_rows:raise ValueError('Empty bridge')
    guarded=select_with_guard(input_rows)
    allowed=[g for g in guarded if g['guarded_record'] is not None]
    blocked=[g for g in guarded if g['guarded_record'] is None]
    report={'status':'DEV_DEFENSIVE_PREFLIGHT_NOT_ACCEPTED',
            'source_requests':len(guarded),'eligible_for_inference':len(allowed),
            'blocked_input_cases':len(blocked),
            'error_counts':dict(Counter(x['code'] for g in guarded for x in g['issues'])),
            'model_not_loaded':args.dry_run,'frozen_candidate_id':freeze_info.get('candidate_id'),
            'frozen_bridge_sha256':sha256_file(bridge),'holdout_used':False,
            'note':'A blocked case returns an explicit null fallback; it does not get an inferred answer.'}
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if args.dry_run:
        print('DEFENSIVE_PREFLIGHT_OK',flush=True)
        return 0
    if root.exists():raise FileExistsError(root)
    model_dir=Path(args.model_id)
    if allowed and not (model_dir/'config.json').is_file():
        raise FileNotFoundError('Local model missing; no download attempted')
    # Create outputs after checking model availability. Never overwrite prior results.
    root.mkdir(parents=True,exist_ok=False)
    image_dir=root/'inputs_marked';image_dir.mkdir()
    with (root/'input_decisions.jsonl').open('w',encoding='utf-8') as f:
        for g in guarded:
            meta={'source_line':g['source_line'],'decision':g['decision'],
                  'identity':g['identity'],'issues':g['issues']}
            f.write(json.dumps(meta,ensure_ascii=False)+'\n')
    model=processor=None
    load_seconds=None
    if allowed:
        import torch
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        if torch.cuda.mem_get_info()[0]/1024**3 < 18:raise RuntimeError('Less than 18 GiB free VRAM')
        t=time.perf_counter()
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir),local_files_only=True,device_map='auto',dtype=dtype,
            attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
        processor=AutoProcessor.from_pretrained(str(model_dir),local_files_only=True,
                                                  max_pixels=512*28*28)
        load_seconds=round(time.perf_counter()-t,3)
    record_list=[];predictions=[];service_rows=[]
    with (root/'predictions.jsonl').open('w',encoding='utf-8') as pf,\
         (root/'service_outputs_guarded.jsonl').open('w',encoding='utf-8') as sf:
        for idx,g in enumerate(guarded,1):
            case=g['guarded_record']
            if case is None:
                out=fallback_response(g)
                pred={'case_id':out['case_id'],'task':out['task'],'status':'SKIPPED_INVALID_INPUT',
                      'error_codes':[e['code'] for e in g['issues']],'source_line':idx}
            else:
                try:
                    pred,out=infer_case(v1,model,processor,case,image_dir,args.input_mode,args.roi_padding)
                except Exception as ex:
                    pred={'case_id':case['case_id'],'task':case['task'],'status':'CASE_FAILED',
                          'error':type(ex).__name__+': '+str(ex)[:200]}
                    out={'case_id':case['case_id'],'task':case['task'],
                         'status':'DEGRADED_INFERENCE_ERROR','payload':None,'unverified':True,
                         'issues':[{'code':'INFERENCE_FAILURE','detail':type(ex).__name__}],
                         'provenance':{'image_id':case['image_id'],
                                       'target_instance_id':case['target_instance_id']}}
            pred['source_line']=idx;out['source_line']=idx
            predictions.append(pred);service_rows.append(out)
            pf.write(json.dumps(pred,ensure_ascii=False,allow_nan=False)+'\n');pf.flush()
            sf.write(json.dumps(out,ensure_ascii=False,allow_nan=False)+'\n');sf.flush()
            print(f'{idx}/{len(guarded)} {out["case_id"]}: {out["status"]}',flush=True)
    lats=[p['end_to_end_ms'] for p in predictions if isinstance(p.get('end_to_end_ms'),(int,float))]
    summary={**report,'status':'DEV_DEFENSIVE_BATCH_COMPLETED_NOT_PRD_ACCEPTED',
             'model_load_seconds':load_seconds,
             'inference_completed':sum(p.get('status')=='GENERATED_NOT_ACCEPTED' for p in predictions),
             'inference_failed':sum(p.get('status')=='CASE_FAILED' for p in predictions),
             'output_status_counts':dict(Counter(r.get('status') for r in service_rows)),
             'median_inference_ms':round(statistics.median(lats),2) if lats else None,
             'full_prd_acceptance':False,'frozen_prd31_modified':False,
             'limitations':['Offline freeze replay, not live 3.1 model execution',
                            'Schema/semantic caution is not verified visual accuracy',
                            'No formal QA/copy human scoring, eight-class evaluation or P95/QPS']}
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return 0 if summary['inference_failed']==0 else 2


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bridge',default='/workspace/prd31_stage_freeze_20261026/bridge_inputs.jsonl')
    ap.add_argument('--freeze-dir',default='/workspace/prd31_stage_freeze_20261026')
    ap.add_argument('--v1-script',default='/workspace/prd32_integrated_pipeline_20261027.py')
    ap.add_argument('--model-id',default='/workspace/models/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--output',default='/workspace/PRD32_DEFENSIVE_BATCH_20261027')
    ap.add_argument('--input-mode',choices=('marked_full_image','predicted_roi'),default='marked_full_image')
    ap.add_argument('--roi-padding',type=float,default=0.05)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if not 0<=args.roi_padding<=0.5:ap.error('--roi-padding out of range')
    return run(args)

if __name__=='__main__':
    sys.exit(main())
