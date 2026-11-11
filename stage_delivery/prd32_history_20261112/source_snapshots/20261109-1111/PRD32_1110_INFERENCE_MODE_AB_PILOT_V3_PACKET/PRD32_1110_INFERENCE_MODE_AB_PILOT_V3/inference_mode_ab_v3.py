#!/usr/bin/env python3
"""DEV60-only paired warm Qwen inference_mode A/B pilot (not production).

One controlled change: candidate wraps *only* helper.run_model_case(...) with
``torch.inference_mode()``. Baseline uses the original helper, no changes.
Original V2 fixed V5/Qwen/Guard code, prompt, image construction and sample
manifest are vendored unchanged. 8 unique targets (one per frozen V5 category),
2 repeat pairs per target, balanced AB/BA order, 32 measured requests total.

A speed improvement is NOT assumed; any byte mismatch fails the candidate gate.
No Holdout, no releases, no weight / prompt / generation limit modifications.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import zipfile
from datetime import datetime

import warm_e2e_stratified_v2 as base

ROOT=Path(__file__).resolve().parent
WORKSPACE=Path('/workspace')
LOCKFILE=ROOT/'PINNED_SOURCE_SHA256.json'
ORIGINAL_FILES=('warm_e2e_stratified_v2.py','sample_manifest_v2.json',
    'producer_contract_v7.py','run_v5_runtime_producer_v7.py',
    'run_live_probe_v4.py','live_observer_v4.py','verified_guard_loader.py')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):
            h.update(part)
    return h.hexdigest()


def check_lock():
    expected=json.loads(LOCKFILE.read_text(encoding='utf-8'))
    if set(expected)!=set(ORIGINAL_FILES):raise ValueError('pin file names changed')
    actual={n:sha(ROOT/n) for n in ORIGINAL_FILES}
    if actual!=expected:
        diffs=[n for n in ORIGINAL_FILES if actual[n]!=expected[n]]
        raise ValueError('vendored original source or sampling manifest changed: '+repr(diffs))
    return actual


def choose_eight(manifest):
    """Exactly one fixed target per V5 predicted category from pinned 24."""
    first={}
    for r in manifest['targets']:
        first.setdefault(r['v5_frozen_category'],(r['case_id'],r['instance_id']))
    if set(first)!=base.CATEGORIES or len(set(first.values()))!=8:
        raise ValueError('8 V5 category target selection failed')
    return [first[k] for k in sorted(first)]


def sequence(keys):
    """2 repeats, balanced AB/BA order (one of each order per target)."""
    seq=[('warmup_baseline',keys[0],'baseline',None),
         ('warmup_candidate',keys[0],'candidate',None)]
    for rep in (1,2):
        for index,key in enumerate(keys,1):
            # Alternate order; the second repeat reverses first repeat order.
            order=('baseline','candidate') if (index+rep)%2==0 else ('candidate','baseline')
            for arm in order:
                seq.append((f'r{rep:02d}_t{index:02d}_{arm}',key,arm,(rep,index)))
    return seq


class InferenceModeHelper:
    """Delegates all original helpers; only model execution context differs."""
    def __init__(self,inner,torch):
        self._inner=inner
        self._torch=torch
    def __getattr__(self,name):return getattr(self._inner,name)
    def run_model_case(self,*args,**kwargs):
        with self._torch.inference_mode():
            return self._inner.run_model_case(*args,**kwargs)


def distribution(xs):
    return base.stats(xs)


def compare(rows,calls,seq,keys):
    actual={r['cohort']:r for r in rows}
    expected_labels=[label for label,_,_,pair in seq if pair is not None]
    if len(actual)!=32 or set(actual)!=set(expected_labels):
        raise ValueError('expected exactly 32 unique, complete measured A/B requests')
    grouped={}
    for x in calls:
        if x['cohort'].startswith('warmup_'):continue
        k=(x['cohort'],x['task'])
        if k in grouped:raise ValueError('duplicate Qwen task record: '+str(k))
        grouped[k]=x
    if len(grouped)!=96 or set(x[1] for x in grouped)!=set(base.TASKS):
        raise ValueError('incomplete 96 measured Qwen task calls')
    pairs=[];task_pairs=[]
    for rep in (1,2):
        for ix,key in enumerate(keys,1):
            root=f'r{rep:02d}_t{ix:02d}'
            b=actual[root+'_baseline']; c=actual[root+'_candidate']
            for field in ('case_id','instance_id','producer_category','source_image_sha256','fresh_mask_sha256'):
                if b[field]!=c[field]:raise ValueError('baseline/candidate producer identity mismatch '+field)
            if (b['case_id'],b['instance_id'])!=key:raise ValueError('manifest paired key mismatch')
            if b['guard_status']!='REVIEW_REQUIRED_NOT_RELEASED' or c['guard_status']!='REVIEW_REQUIRED_NOT_RELEASED':
                raise ValueError('guard publication protection not retained')
            pd={'case_id':key[0],'instance_id':key[1],'repeat':rep,'category':b['producer_category'],
                'baseline_total_ms':float(b['total_host_wall_ms']),
                'candidate_total_ms':float(c['total_host_wall_ms']),
                'delta_ms_candidate_minus_baseline':round(float(c['total_host_wall_ms'])-float(b['total_host_wall_ms']),3)}
            for task in base.TASKS:
                q_b=grouped[(root+'_baseline',task)];q_c=grouped[(root+'_candidate',task)]
                for field in ('case_id','instance_id','image_sha256','marked_sha256','prompt_sha256'):
                    if q_b[field]!=q_c[field]:raise ValueError('Qwen input prompt/marked image drift: '+field)
                if q_b['response'] is None or q_c['response'] is None:raise ValueError('empty output')
                same=q_b['response']==q_c['response']
                model_b=q_b['model_runner_timing'];model_c=q_c['model_runner_timing']
                task_pairs.append({'case_id':key[0],'instance_id':key[1],'repeat':rep,'task':task,
                    'output_bytes_identical':same,
                    'baseline_generated_tokens':model_b['generated_tokens'],
                    'candidate_generated_tokens':model_c['generated_tokens'],
                    'baseline_generation_ms':model_b['generation_ms'],
                    'candidate_generation_ms':model_c['generation_ms'],
                    'baseline_preprocess_ms':model_b['preprocess_ms'],
                    'candidate_preprocess_ms':model_c['preprocess_ms'],
                    'baseline_task_host_ms':q_b['qwen_task_host_wall_ms'],
                    'candidate_task_host_ms':q_c['qwen_task_host_wall_ms']})
            pd['all_three_outputs_identical']=all(x['output_bytes_identical'] for x in task_pairs[-3:])
            pairs.append(pd)
    b=[x['baseline_total_ms'] for x in pairs];c=[x['candidate_total_ms'] for x in pairs]
    by_key={}
    for p in pairs:
        by_key.setdefault((p['case_id'],p['instance_id']),[]).append(p)
    target_deltas={f'{k[0]}/{k[1]}':statistics.mean(t['delta_ms_candidate_minus_baseline'] for t in v)
                   for k,v in by_key.items()}
    same=sum(int(x['output_bytes_identical']) for x in task_pairs)
    total_gain_pct=(statistics.mean(b)-statistics.mean(c))/statistics.mean(b)*100
    target_wins=sum(1 for x in target_deltas.values() if x < 0)
    p95_b=base.percentile(b,95);p95_c=base.percentile(c,95)
    promising=(same==48 and total_gain_pct>=5 and target_wins>=6 and p95_c<=p95_b*1.05)
    return pairs,task_pairs,{
        'status':'AB_PILOT_COMPLETE_SHADOW_ONLY',
        'distinct_dev60_targets':len(keys),'repeats_per_target':2,
        'measured_requests_baseline':16,'measured_requests_candidate':16,
        'measured_qwen_calls_per_arm':48,'warmup_requests_per_arm':1,
        'baseline_total_ms':distribution(b),'candidate_total_ms':distribution(c),
        'mean_speedup_pct_positive_means_candidate_faster':round(total_gain_pct,3),
        'paired_request_delta_ms_candidate_minus_baseline':distribution([x['delta_ms_candidate_minus_baseline'] for x in pairs]),
        'target_mean_delta_ms_candidate_minus_baseline':target_deltas,
        'unique_targets_faster':target_wins,
        'exact_output_matches':same,'exact_output_pairs_total':len(task_pairs),
        'pair_guard_review_required':len(pairs),'model_or_prompt_changed':False,
        'candidate_single_variable':'torch.inference_mode() wrapping original helper.run_model_case',
        'candidate_decision':'CONSIDER_FURTHER_DEV60_TIMING_ONLY' if promising else 'KEEP_BASELINE_NOT_PROVEN_FASTER_OR_OUTPUT_DRIFT',
        'decision_rule':'Require 48/48 byte-identical outputs, >=5% mean improvement, >=6/8 unique targets faster, and P95 no worse by >5%; pilot is not deployment proof',
        'independent_holdout_accessed':False,'publication_allowed':False,
        'scope_warning':'One target per V5 class; selected subset of same fixed 24; all measures correlated; not an acceptance SLA or full DEV60 regression'}


def write_csv(path,rows):
    if not rows:raise ValueError('no rows to write')
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)


def default_output():
    return WORKSPACE/f'PRD32_1110_INFERENCE_AB_V3_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'


def zip_output(out):
    path=out.with_suffix('.zip')
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():z.write(f,arcname=f.relative_to(out.parent).as_posix())
    print('UPLOAD_THIS:',path,flush=True)


def do_run(args,out):
    pins=check_lock()
    for p in (args.repo,args.raw_dir,args.image_root,args.checkpoint,args.runner,args.model_dir,args.manifest,out):
        base.refuse_restricted(p)
    original_targets,modules=base.v5_preflight(args.raw_dir,args.image_root,args.repo,args.checkpoint)
    manifest,all_keys=base.load_manifest(args.manifest)
    if len(all_keys)!=24 or len(original_targets)!=12:raise ValueError('frozen V2/V7 source drift')
    keys=choose_eight(manifest)
    found=base.load_extended_targets(args.raw_dir,args.image_root,manifest,all_keys)
    runner=base.verified_runner(args.runner)
    valid_rows,core,helper,focus=runner.prompt_check(argparse.Namespace(
       snapshot=runner.DEFAULT_SNAP,source=runner.DEFAULT_SOURCE,
       focus_script=runner.DEFAULT_FOCUS,raw=runner.DEFAULT_RAW,input=runner.DEFAULT_INPUT))
    if len(valid_rows)!=171:raise ValueError('DEV60 Qwen snapshot changed')
    guard=base.load_guard(args.repo/'src/fashion_multimodal_analysis/integration/prd32_output_guard.py')
    if not (args.model_dir/'config.json').is_file():raise FileNotFoundError('missing Qwen local model')
    env={'run_type':'one-variable A/B pilot','status':'PINNED_DEV60_PREFLIGHT_OK',
        'target_count':8,'targets':[f'{k[0]}/{k[1]}' for k in keys],
        'v5_checkpoint_sha256':base.V5_CKPT_SHA256,
        'v5_source_hashes':modules,'qwen_runner_sha256':base.digest_file(args.runner),
        'guard_sha256':base.digest_file(args.repo/'src/fashion_multimodal_analysis/integration/prd32_output_guard.py'),
        'original_v2_manifest_sha256':base.MANIFEST_SHA256,'vendored_original_source_hashes':pins,
        'candidate':'inference_mode only','holdout_accessed':False,
        'production_approved':False,'publish_allowed':False,
        'actual_model_calls_performed':False}
    base.write_json(out/'preflight.json',env)
    if args.mode=='preflight':
        print('DEV60_INFERENCE_MODE_AB_PREFLIGHT_PASS; no model calls',flush=True)
        return
    if not args.execute_models:raise ValueError('explicit --execute-models required for live')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch,numpy as np
    from PIL import Image
    if not torch.cuda.is_available():raise RuntimeError('GPU CUDA not available')
    gpu_free=torch.cuda.mem_get_info()[0]/1024**3
    if gpu_free < 18:raise RuntimeError(f'Qwen 18 GiB GPU headroom gate blocked: {gpu_free:.3f}')
    import transformers
    from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    t0=time.perf_counter()
    qwen=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_dir),local_files_only=True,device_map='auto',
        torch_dtype=dtype,attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    processor=AutoProcessor.from_pretrained(str(args.model_dir),local_files_only=True,max_pixels=512*28*28)
    base.sync(torch)
    device_map=getattr(qwen,'hf_device_map',None)
    if isinstance(device_map,dict) and any(str(v).lower() in {'cpu','disk'} for v in device_map.values()):
        raise RuntimeError('Qwen offloaded to CPU/disk: '+str(device_map))
    qwen_load_ms=(time.perf_counter()-t0)*1000
    sys.path.insert(0,str(args.repo/'src'))
    from fashion_multimodal_analysis.common.schema import ID_TO_CLASS,CLASS_TO_ID
    from fashion_multimodal_analysis.segmentation.modeling import build_evaluation_model
    t1=time.perf_counter()
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if cp.get('class_to_id')!=CLASS_TO_ID or cp.get('epoch')!=15 or cp.get('args',{}).get('epochs')!=15:
        raise ValueError('V5 checkpoint changed')
    saved=cp.get('args',{})
    v5=build_evaluation_model(int(saved.get('min_size',640)),int(saved.get('max_size',1024)))
    v5.load_state_dict(cp['model_state_dict'],strict=True)
    device=torch.device('cuda');v5.to(device).eval();del cp
    base.sync(torch)
    free_after=torch.cuda.mem_get_info()[0]/1024**3
    base.write_json(out/'co_residency.json',{'status':'TWO_MODELS_LOADED_SAME_PROCESS',
        'torch_version':torch.__version__,'transformers_version':transformers.__version__,
        'initial_free_gib':round(gpu_free,3),'free_after_load_gib':round(free_after,3),
        'qwen_load_ms':round(qwen_load_ms,3),'v5_load_ms':round((time.perf_counter()-t1)*1000,3),
        'load_ms_not_in_measured_latency':True})
    if free_after<1:raise RuntimeError('insufficient GPU headroom after two model loads')
    torch.cuda.reset_peak_memory_stats()
    seq=sequence(keys)
    alt_helper=InferenceModeHelper(helper,torch)
    rows=[]
    for label,key,arm,pair in seq:
        h=helper if arm=='baseline' else alt_helper
        row=base.run_one(key=key,label=label,targets=found,
           v5_model=v5,qwen=qwen,processor=processor,runner=runner,
           core=core,helper=h,focus=focus,guard=guard,device=device,
           torch=torch,np=np,Image=Image,ID_TO_CLASS=ID_TO_CLASS,out=out)
        if pair is not None:rows.append(row)
        print(f'AB_EVENT {label} {key[0]}/{key[1]} {row["total_host_wall_ms"]:.3f}ms {row["guard_status"]}',flush=True)
    calls=base.read_jsonl(out/'qwen_calls.jsonl')
    if len(calls)!=102 or len({x['invocation_uid'] for x in calls})!=len(calls):
        raise ValueError('incorrect 102 Qwen calls or non-unique call IDs')
    pairs,task_pairs,report=compare(rows,calls,seq,keys)
    write_csv(out/'per_request_timing.csv',rows)
    write_csv(out/'paired_request_deltas.csv',pairs)
    write_csv(out/'paired_task_output_and_timing.csv',task_pairs)
    base.write_json(out/'summary.json',report)
    print('DEV60_AB_V3_COMPLETE_SHADOW_ONLY:',report['candidate_decision'],
        'speedup%=',report['mean_speedup_pct_positive_means_candidate_faster'],
        'byte_equal=',f'{report["exact_output_matches"]}/{report["exact_output_pairs_total"]}',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['preflight','live'],default='preflight')
    p.add_argument('--execute-models',action='store_true')
    p.add_argument('--repo',type=Path,default=base.REPO)
    p.add_argument('--raw-dir',type=Path,default=base.V5_RAW)
    p.add_argument('--image-root',type=Path,default=base.IMAGE_ROOT)
    p.add_argument('--checkpoint',type=Path,default=base.CKPT)
    p.add_argument('--runner',type=Path,default=base.RUNNER)
    p.add_argument('--model-dir',type=Path,default=base.QWEN_MODEL)
    p.add_argument('--manifest',type=Path,default=base.DEFAULT_MANIFEST)
    p.add_argument('--out',type=Path,default=None)
    args=p.parse_args(argv)
    out=args.out or default_output()
    if out.exists():raise FileExistsError('refuse overwrite old output '+str(out))
    base.refuse_restricted(out)
    out.mkdir(parents=True,exist_ok=False)
    code=0
    try:
        do_run(args,out)
    except BaseException as ex:
        code=2
        base.write_json(out/'failure.json',{'status':'FAIL_CLOSED_NOT_BENCHMARK_COMPLETE',
           'error_type':type(ex).__name__,'message':str(ex),'traceback':traceback.format_exc()[-20000:],
           'publish_allowed':False,'holdout_accessed':False,
           'partial_results_do_not_count_as_full_benchmark':True})
        print('DEV60_AB_V3_BLOCKED:',type(ex).__name__,str(ex),file=sys.stderr,flush=True)
    finally:
        zip_output(out)
    return code

if __name__=='__main__':sys.exit(main())
