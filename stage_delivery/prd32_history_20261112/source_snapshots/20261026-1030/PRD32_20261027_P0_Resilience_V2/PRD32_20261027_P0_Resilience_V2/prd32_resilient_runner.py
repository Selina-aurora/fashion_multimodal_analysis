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
                                  read_jsonl_lenient, restricted, sha256_file, err)

DEFAULT_ROOT=Path('/workspace')


def load_v1(path):
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(p)
    spec=importlib.util.spec_from_file_location('prd32_readonly_v1_helpers',p)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select_with_guard(records):
    """Fail closed per source line; detect duplicate IDs and conflicting identities.

    It is unsafe to choose the first bbox for a repeated target with contradictory
    image SHA/bbox, so every conflicting member is isolated. Genuine repeated tasks
    for the same image/instance with identical geometry remain allowed.
    """
    ids=Counter(r.get('case_id') for r in records if isinstance(r,dict)
                and isinstance(r.get('case_id'),str))
    audits=[]
    for n,r in enumerate(records,1):
        try:
            g=guard_input_record(r,image_check=True)
        except Exception as ex:
            ident={k:r.get(k) for k in ('case_id','image_id','target_instance_id','task')
                   if isinstance(r,dict) and isinstance(r.get(k),str)}
            g={'decision':'BLOCKED','guarded_record':None,'identity':ident,
               'issues':[err('GUARD_EXCEPTION',type(ex).__name__)]}
        cid=g['identity'].get('case_id')
        if cid and ids[cid]>1:
            g['decision']='BLOCKED';g['guarded_record']=None
            g['issues'].append(err('DUPLICATE_CASE_ID'))
        g['source_line']=n
        # These are not relied on to approve an input, only to detect mismatches.
        if isinstance(r,dict):
            g['_raw_record']=r
        audits.append(g)
    consistency={}
    image_hashes={}
    category_hints={}
    for g in audits:
        r=g.get('_raw_record',{})
        ident=g.get('identity',{})
        image_id=ident.get('image_id')
        target_id=ident.get('target_instance_id')
        if image_id and target_id:
            key=(image_id,target_id)
            identity=(r.get('image_sha256'), repr(r.get('target_bbox_xyxy')),
                      r.get('image_path'))
            consistency.setdefault(key,{}).setdefault(repr(identity),[]).append(g)
            for field in ('target_category','predicted_category','predicted_target_category'):
                category=r.get(field)
                if isinstance(category,str) and category:
                    category_hints.setdefault(key,{}).setdefault(category,[]).append(g)
        if image_id:
            hashval=r.get('image_sha256')
            if isinstance(hashval,str):
                image_hashes.setdefault(image_id,{}).setdefault(hashval,[]).append(g)
    for variants in consistency.values():
        if len(variants)>1:
            for cases in variants.values():
                for g in cases:
                    g['decision']='BLOCKED';g['guarded_record']=None
                    g['issues'].append(err('TARGET_IDENTITY_GEOMETRY_CONFLICT'))
    for variants in category_hints.values():
        if len(variants)>1:
            for cases in variants.values():
                for g in cases:
                    g['decision']='BLOCKED';g['guarded_record']=None
                    g['issues'].append(err('CROSS_REQUEST_CATEGORY_CONFLICT'))
    for variants in image_hashes.values():
        if len(variants)>1:
            for cases in variants.values():
                for g in cases:
                    g['decision']='BLOCKED';g['guarded_record']=None
                    g['issues'].append(err('IMAGE_ID_SHA_CONFLICT'))
    for g in audits:g.pop('_raw_record',None)
    return audits


def fallback_response(g):
    identity=g.get('identity',{})
    cid=identity.get('case_id') or f'__source_line_{g["source_line"]}'
    return {'case_id':cid,'task':identity.get('task'),
            'status':'BLOCKED_INVALID_INPUT','payload':None,
            'issues':g['issues'],'provenance':identity,
            'unconfirmed':['category','primary_color','pattern','visible_design','visible_decoration'],
            'source_line':g['source_line'], 'inference_performed':False}


def standardize_service(row, guard, *, inference_performed, simulated_backend):
    """One consistent downstream interface for success, quarantine and exceptions.

    A nonempty candidate payload means only that the guarded model generated it;
    it is NOT GT and is NOT automatically publishable.
    """
    ident=guard.get('identity',{})
    prov=dict(row.get('provenance') or {})
    for key in ('case_id','task','image_id','target_instance_id'):
        if key in ident:prov[key]=ident[key]
    result={
        'schema':'prd32_guarded_service_contract_dev_v2',
        'case_id':row.get('case_id') or ident.get('case_id') or f"__source_line_{guard.get('source_line')}",
        'task':row.get('task') or ident.get('task'),
        'status':row.get('status','BLOCKED_UNSPECIFIED'),
        'payload':row.get('payload'),
        'issues':row.get('issues') if isinstance(row.get('issues'),list) else [],
        'provenance':prov,
        'unconfirmed':row.get('unconfirmed') if isinstance(row.get('unconfirmed'),list) else [],
        'inference_performed':bool(inference_performed),
        'unverified':True,
        'simulated_backend':bool(simulated_backend),
        'source_line':guard.get('source_line')
    }
    if result['payload'] is None and not result['unconfirmed']:
        result['unconfirmed']=['response']
    return result


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
            'guard_decision_counts':dict(Counter(g['decision'] for g in guarded)),
            'model_not_loaded':args.dry_run,'frozen_candidate_id':freeze_info.get('candidate_id'),
            'frozen_bridge_sha256':sha256_file(bridge),'holdout_used':False,
            'note':'A blocked case returns an explicit null fallback; it does not get an inferred answer.'}
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if args.dry_run:
        if args.preflight_output:
            preflight_out=Path(args.preflight_output)
            if preflight_out.exists():raise FileExistsError(preflight_out)
            preflight_out.mkdir(parents=True,exist_ok=False)
            (preflight_out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            with (preflight_out/'input_decisions.jsonl').open('w',encoding='utf-8') as f:
                for g in guarded:
                    f.write(json.dumps({'source_line':g['source_line'],
                        'decision':g['decision'],'identity':g['identity'],
                        'issues':g['issues']},ensure_ascii=False,allow_nan=False)+'\n')
        print('DEFENSIVE_PREFLIGHT_OK',flush=True)
        return 0
    if root.exists():raise FileExistsError(root)
    model_dir=Path(args.model_id)
    if allowed and not args.simulate and not (model_dir/'config.json').is_file():
        raise FileNotFoundError('Local model missing; no download attempted')
    # Check GPU before creating output files. A failed CUDA allocation should
    # not leave a misleading empty successful-looking run directory.
    if allowed and not args.simulate:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        if torch.cuda.mem_get_info()[0]/1024**3 < 18:raise RuntimeError('Less than 18 GiB free VRAM')
    root.mkdir(parents=True,exist_ok=False)
    image_dir=root/'inputs_marked';image_dir.mkdir()
    with (root/'input_decisions.jsonl').open('w',encoding='utf-8') as f:
        for g in guarded:
            meta={'source_line':g['source_line'],'decision':g['decision'],
                  'identity':g['identity'],'issues':g['issues']}
            f.write(json.dumps(meta,ensure_ascii=False,allow_nan=False)+'\n')
    model=processor=None
    load_seconds=None
    if allowed and args.simulate:
        def fake_run(_model,_processor,record,_marked,_prompt,*,task_limits):
            task=record['task']
            raw={
              'vqa':'无法确定',
              'tags':'{"category":"top","primary_color":"白色","pattern":"无","visible_design":"无设计"}',
              'copy':'这是一件白色服饰，具体材质未确认。'
            }[task]
            return raw,{'preprocess_ms':0.1,'generation_ms':0.1,'end_to_end_ms':0.2,
                        'generated_tokens':0,'simulated_backend':True}
        v1.run_model_case=fake_run
        load_seconds=0.0
    elif allowed:
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
            pred['source_line']=idx
            out=standardize_service(out,g,inference_performed=(case is not None and pred.get('status')!='CASE_FAILED'),simulated_backend=args.simulate)
            predictions.append(pred);service_rows.append(out)
            pf.write(json.dumps(pred,ensure_ascii=False,allow_nan=False)+'\n');pf.flush()
            sf.write(json.dumps(out,ensure_ascii=False,allow_nan=False)+'\n');sf.flush()
            print(f'{idx}/{len(guarded)} {out["case_id"]}: {out["status"]}',flush=True)
    lats=[p['end_to_end_ms'] for p in predictions if isinstance(p.get('end_to_end_ms'),(int,float))]
    summary={**report,'status':'SIMULATED_FAULT_INJECTION_NOT_REAL_MODEL' if args.simulate else 'DEV_DEFENSIVE_BATCH_COMPLETED_NOT_PRD_ACCEPTED',
             'model_load_seconds':load_seconds,'simulated_backend':args.simulate,
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
    ap.add_argument('--preflight-output',help='Optional non-overwriting JSON audit directory when --dry-run')
    ap.add_argument('--simulate',action='store_true',help='NO GPU. Fake model outputs ONLY for fault injection; never quality benchmark')
    args=ap.parse_args()
    if not 0<=args.roi_padding<=0.5:ap.error('--roi-padding out of range')
    if args.preflight_output and not args.dry_run:ap.error('--preflight-output requires --dry-run')
    if args.simulate and args.dry_run:ap.error('--simulate and --dry-run are mutually exclusive')
    return run(args)

if __name__=='__main__':
    sys.exit(main())
