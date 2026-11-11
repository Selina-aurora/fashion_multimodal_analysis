#!/usr/bin/env python3
"""DEV60-only single-process, co-resident V5->Qwen->Guard pilot (shadow only).

Inputs: exact pinned V7 code and DEV60 image/checkpoint/prompt/guard.  No holdout,
no new model weights/prompt edits, no production release.  The warm-up target
is excluded from the four measured request-level observations.

CAUTION: On 24 GiB RTX4090 this may OOM; it fails closed and exports an audit ZIP.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import uuid
import zipfile
from datetime import datetime

from producer_contract_v7 import (
    QWEN_SMOKE_TARGETS, EXPECTED_MODULE_HASHES, V5_CKPT_SHA256,
    check_producer_match, canonical_mask_digest, pinned_frozen_targets,
    digest_file, refuse_restricted,
)
from run_v5_runtime_producer_v7 import preflight as v5_preflight
from run_live_probe_v4 import verified_runner, write_json
from live_observer_v4 import LiveShadowObserver, TASKS
from verified_guard_loader import load_guard

WORKSPACE=Path('/workspace')
REPO=WORKSPACE/'fashion_multimodal_analysis'
V5_RAW=WORKSPACE/'PRD32_DEV60_V5_REAL_20261030_run01'
IMAGE_ROOT=WORKSPACE/'fashion_data'
CKPT=REPO/'outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth'
RUNNER=WORKSPACE/'PRD32_DEV60_QWEN_AB_20261030/dev60_qwen_ab.py'
QWEN_MODEL=WORKSPACE/'models/Qwen2.5-VL-7B-Instruct'


def percentile(values, p):
    if not values:
        return None
    if not 0<=p<=100:
        raise ValueError('percentile range')
    v=sorted(float(x) for x in values)
    k=(len(v)-1)*p/100
    l=int(math.floor(k)); u=int(math.ceil(k))
    return round(v[l]*(u-k)+v[u]*(k-l),3) if l!=u else round(v[l],3)


def stats(values):
    if not values:return None
    vals=[float(x) for x in values]
    return {'n':len(vals),'mean_ms':round(statistics.mean(vals),3),
            'median_ms':round(statistics.median(vals),3),
            'p95_ms':percentile(vals,95),'max_ms':round(max(vals),3)}


def dump_jsonl(path, obj):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n')


def default_output():
    return WORKSPACE/f'PRD32_1110_WARM_E2E_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'


def zip_results(out):
    z=out.with_suffix('.zip')
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as fp:
        for x in sorted(out.rglob('*')):
            if x.is_file(): fp.write(x,x.relative_to(out.parent).as_posix())
    print('UPLOAD_THIS:',str(z),flush=True)
    return z


def sync(torch):
    torch.cuda.synchronize()


def run_one(*,key,label,targets,v5_model,qwen,processor,runner,core,helper,focus,guard,device,torch,np,Image,ID_TO_CLASS,out):
    case,iid=key
    frozen=targets[key]
    t_start=time.perf_counter()
    sync(torch)
    t0=time.perf_counter()
    with Image.open(frozen['image_path']) as im:
        pil=im.convert('RGB')
    from fashion_multimodal_analysis.evaluation.runtime import predict_timed
    boxes,labels,scores,masks,v5_inner_ms=predict_timed(
        pil,v5_model,device,score_threshold=.4,mask_threshold=.5)
    sync(torch)
    t_v5=time.perf_counter()
    idx=int(iid.split('_')[-1])
    if idx>=len(labels):raise RuntimeError(f'V5 lost instance {case}/{iid}')
    category=ID_TO_CLASS.get(int(labels[idx]))
    new_mask=(np.asarray(masks[idx])>0).astype(np.uint8)
    with Image.open(frozen['frozen_mask_path']) as old:
        old_mask=np.asarray(old.convert('L'))>127
    ins={'instance_id':iid,'category':category,'score':float(scores[idx]),
         'bbox':[float(x) for x in boxes[idx]]}
    match=check_producer_match(target=frozen,ins=ins,
                               reference_mask=old_mask,produced_mask=new_mask)
    mask_sha=canonical_mask_digest(new_mask)
    sync(torch)
    t_verified=time.perf_counter()

    obs=LiveShadowObserver('baseline')
    marked_dir=out/label/'marked_inputs'
    marked_dir.mkdir(parents=True,exist_ok=False)
    rec={'case_id':case,'target_instance_id':iid,'image_path':frozen['image_path'],
         'image_sha256':frozen['image_sha256'],'target_bbox_xyxy':ins['bbox'],
         'predicted_category_v5_unverified':category}
    stage_task_times={}
    for task in TASKS:
        qrec={'image_path':rec['image_path'],'image_sha256':rec['image_sha256'],
              'target_bbox_xyxy':rec['target_bbox_xyxy'], 'target_instance_id':iid,
              'task':task,'query':runner.QUESTION,'length_chars':[20,80],
              'predicted_attribute_context':[]}
        sync(torch)
        qstart=time.perf_counter()
        marked=helper.make_marked_image(qrec,marked_dir,'marked_full_image',.05)
        if not Path(marked).resolve(strict=True).is_relative_to(marked_dir.resolve(strict=True)):
            raise ValueError('marked image escaped output')
        prompt=core.build_prompt(helper,focus,qrec,task,'baseline')
        prompt_sha=runner.tsha(prompt)
        if prompt_sha!=runner.PROMPT_SHA['baseline'][task]:
            raise ValueError('baseline prompt SHA mismatch')
        job='__'.join((case,iid,task,'baseline','attempt_01'))
        host_id=obs.before_model(rec=rec,qrec=qrec,task=task,
                                 marked=marked,prompt=prompt,
                                 prompt_sha=prompt_sha,job_id=job)
        sync(torch)
        qwen_start=time.perf_counter()
        raw,timing=helper.run_model_case(qwen,processor,qrec,marked,prompt,
                                        task_limits={'tags':170,'vqa':128,'copy':120})
        sync(torch)
        qwen_end=time.perf_counter()
        event=obs.after_model(job_id=job,response_job_id=job,
                              completed_host_request_id=host_id,raw_text=raw)
        sync(torch)
        qend=time.perf_counter()
        dump_jsonl(out/'qwen_calls.jsonl',{
            'cohort':label,'case_id':case,'instance_id':iid,'task':task,
            'host_request_id':host_id,'prompt_sha256':prompt_sha,
            'image_sha256':rec['image_sha256'],
            'marked_sha256':event['marked_sha256'],
            'response_sha256':event['response_sha256'],
            'response':raw,'model_runner_timing':timing,
            'qwen_model_call_wall_ms':round((qwen_end-qwen_start)*1000,3),
            'qwen_task_host_wall_ms':round((qend-qstart)*1000,3),
            'publish_allowed':False})
        stage_task_times[task]={'host_ms':round((qend-qstart)*1000,3),
                                'model_call_ms':round((qwen_end-qwen_start)*1000,3)}

    sync(torch)
    guard_start=time.perf_counter()
    guard_result=obs.finalize(guard)
    sync(torch)
    guard_end=time.perf_counter()
    if guard_result.get('guard_service_status')!='REVIEW_REQUIRED_NOT_RELEASED':
        raise RuntimeError('Guard failed closed-release gate: '+str(guard_result.get('guard_service_status')))
    # Timing ends before audit-mask disk write and evidence packaging.
    total_ms=round((guard_end-t_start)*1000,3)
    audit=out/label
    write_json(audit/'guard_result.json',guard_result)
    Image.fromarray(new_mask*255,mode='L').save(audit/'live_mask.png')
    row={'cohort':label,'case_id':case,'instance_id':iid,
         'total_host_wall_ms':total_ms,
         'v5_host_wall_ms':round((t_v5-t0)*1000,3),
         'v5_inner_ms':round(float(v5_inner_ms),3),
         'frozen_identity_verification_ms':round((t_verified-t_v5)*1000,3),
         'tags_task_host_wall_ms':stage_task_times['tags']['host_ms'],
         'vqa_task_host_wall_ms':stage_task_times['vqa']['host_ms'],
         'copy_task_host_wall_ms':stage_task_times['copy']['host_ms'],
         'guard_host_wall_ms':round((guard_end-guard_start)*1000,3),
         'guard_status':guard_result['guard_service_status'],
         'fresh_mask_sha256':mask_sha,
         'fresh_mask_png_sha256':digest_file(audit/'live_mask.png'),
         'source_image_sha256':frozen['image_sha256'],
         'producer_category':category,
         'frozen_bbox_iou':match['bbox_iou'],
         'frozen_mask_iou':match['mask_iou'],
         'gpu_peak_allocated_mib':round(torch.cuda.max_memory_allocated()/1048576,1),
         'publish_allowed':False,
         'holdout_accessed':False}
    dump_jsonl(out/'per_target_events.jsonl',row)
    return row


def run(args,out):
    # preflight: pinned weights, source modules, DEV60 training images, Qwen prompt bytes.
    for p in (args.repo,args.raw_dir,args.image_root,args.checkpoint,args.runner,args.model_dir,out):
        refuse_restricted(p)
    targets,modules=v5_preflight(args.raw_dir,args.image_root,args.repo,args.checkpoint)
    runner=verified_runner(args.runner)
    valid_rows,core,helper,focus=runner.prompt_check(argparse.Namespace(
        snapshot=runner.DEFAULT_SNAP, source=runner.DEFAULT_SOURCE,
        focus_script=runner.DEFAULT_FOCUS,raw=runner.DEFAULT_RAW,input=runner.DEFAULT_INPUT))
    if len(valid_rows)!=171:raise ValueError('Qwen original DEV60 runner snapshot changed')
    guard=load_guard(args.repo/'src/fashion_multimodal_analysis/integration/prd32_output_guard.py')
    if not (args.model_dir/'config.json').is_file():raise FileNotFoundError('Qwen local model config missing')
    write_json(out/'preflight.json',{'status':'PINNED_DEV60_PREFLIGHT_OK',
       'qwen_smoke_targets':[f'{c}/{i}' for c,i in QWEN_SMOKE_TARGETS],
       'v5_source_hashes':modules,'v5_checkpoint_sha256':V5_CKPT_SHA256,
       'qwen_runner_sha256':digest_file(args.runner),
       'guard_sha256':digest_file(args.repo/'src/fashion_multimodal_analysis/integration/prd32_output_guard.py'),
       'mode':args.mode,'model_calls_performed':False,
       'holdout_used':False,'publish_allowed':False})
    if args.mode=='preflight':
        print('DEV60_WARM_E2E_PREFLIGHT_PASS; NO MODEL EXECUTION; NO HOLDOUT')
        return
    if not args.execute_models:
        raise ValueError('live GPU test requires explicit --execute-models')

    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch, numpy as np
    from PIL import Image
    if not torch.cuda.is_available():raise RuntimeError('CUDA not available')
    before_free=torch.cuda.mem_get_info()[0]/1024**3
    if before_free<18:raise RuntimeError(f'Qwen frozen 18 GiB minimum not met: {before_free:.2f} GiB')
    t_loading=time.perf_counter()
    from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    qwen=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_dir),local_files_only=True,device_map='auto',
        torch_dtype=dtype,attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    processor=AutoProcessor.from_pretrained(str(args.model_dir),local_files_only=True,
                                              max_pixels=512*28*28)
    sync(torch)
    device_map=getattr(qwen,'hf_device_map',None)
    if isinstance(device_map,dict) and any(str(v).lower() in {'cpu','disk'} for v in device_map.values()):
        raise RuntimeError(f'Qwen model offloaded to CPU/disk: {device_map}; not comparable with GPU-resident test')
    qwen_load_ms=(time.perf_counter()-t_loading)*1000
    write_json(out/'qwen_loaded.json',{'qwen_load_ms':round(qwen_load_ms,3),
        'gpu_free_gib_before':round(before_free,3),
        'gpu_free_gib_after':round(torch.cuda.mem_get_info()[0]/1024**3,3),
        'device_map':device_map,'included_in_warm_request_latency':False})

    # Load V5 into the very SAME Python process without clearing Qwen.
    sys.path.insert(0,str(args.repo/'src'))
    from fashion_multimodal_analysis.common.schema import ID_TO_CLASS,CLASS_TO_ID
    from fashion_multimodal_analysis.segmentation.modeling import build_evaluation_model
    vt0=time.perf_counter()
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if cp.get('class_to_id')!=CLASS_TO_ID or cp.get('epoch')!=15 or cp.get('args',{}).get('epochs')!=15:
        raise ValueError('V5 checkpoint category/epoch drift')
    saved=cp.get('args',{})
    v5=build_evaluation_model(int(saved.get('min_size',640)),int(saved.get('max_size',1024)))
    v5.load_state_dict(cp['model_state_dict'],strict=True)
    device=torch.device('cuda')
    v5.to(device).eval()
    del cp
    sync(torch)
    v5_load_ms=(time.perf_counter()-vt0)*1000
    both_free=torch.cuda.mem_get_info()[0]/1024**3
    write_json(out/'co_residency.json',{'status':'BOTH_MODELS_LOADED_SAME_PROCESS',
        'v5_load_ms':round(v5_load_ms,3),
        'free_gpu_gib_after_both_loaded':round(both_free,3),
        'included_in_warm_request_latency':False})
    if both_free<1.0:
        raise RuntimeError('under 1 GiB GPU headroom with both models; refuse inference to avoid OOM')
    torch.cuda.reset_peak_memory_stats()

    # One real extra call for warm-up, deliberately EXCLUDED from measured cohort.
    keys=[QWEN_SMOKE_TARGETS[0],*QWEN_SMOKE_TARGETS]
    measures=[]
    for idx,key in enumerate(keys):
        label='warmup_excluded' if idx==0 else f'measured_{idx:02d}'
        row=run_one(key=key,label=label,targets=targets,v5_model=v5,
            qwen=qwen,processor=processor,runner=runner,core=core,helper=helper,
            focus=focus,guard=guard,device=device,torch=torch,np=np,Image=Image,
            ID_TO_CLASS=ID_TO_CLASS,out=out)
        if idx:
            measures.append(row)
        print(f'WARM_E2E_EVENT {label} {key[0]}/{key[1]} total_ms={row["total_host_wall_ms"]} guard={row["guard_status"]}',flush=True)
    names=['total_host_wall_ms','v5_host_wall_ms','tags_task_host_wall_ms',
           'vqa_task_host_wall_ms','copy_task_host_wall_ms','guard_host_wall_ms']
    summary={'status':'DEV60_4_TARGET_WARM_E2E_PILOT_COMPLETE_SHADOW_ONLY',
             'co_resident_models_same_python_process':True,
             'warmup_excluded':1,'measured_instances':len(measures),
             'measured_qwen_calls':len(measures)*3,
             'total_qwen_calls_including_warmup':len(keys)*3,
             'warm_request_latency_stats':{n:stats([r[n] for r in measures]) for n in names},
             'qwen_load_ms':round(qwen_load_ms,3),
             'v5_load_ms':round(v5_load_ms,3),
             'source_mask_supplied_to_qwen':False,
             'source_mask_verified_against_frozen':True,
             'guard_review_required_all':True,
             'model_weights_and_prompts_unchanged':True,
             'independent_holdout_used':False,'production_approved':False,
             'release_allowed':False,
             'scope_warning':'4 hand-selected DEV targets; model and audit I/O; not full DEV60 PRD acceptance; 4-point p95 unstable'}
    write_json(out/'summary.json',summary)
    with (out/'per_request_timing.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(measures[0]),extrasaction='ignore');w.writeheader();w.writerows(measures)
    print('DEV60_WARM_E2E_PILOT_COMPLETE_SHADOW_ONLY',len(measures),'measured instances')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['preflight','live'],default='preflight')
    p.add_argument('--execute-models',action='store_true')
    p.add_argument('--repo',type=Path,default=REPO)
    p.add_argument('--raw-dir',type=Path,default=V5_RAW)
    p.add_argument('--image-root',type=Path,default=IMAGE_ROOT)
    p.add_argument('--checkpoint',type=Path,default=CKPT)
    p.add_argument('--runner',type=Path,default=RUNNER)
    p.add_argument('--model-dir',type=Path,default=QWEN_MODEL)
    p.add_argument('--out',type=Path,default=None)
    a=p.parse_args(argv)
    out=a.out or default_output()
    if out.exists():raise FileExistsError('refuse overwriting previous run: '+str(out))
    refuse_restricted(out)
    out.mkdir(parents=True,exist_ok=False)
    code=0
    try:
        run(a,out)
    except Exception as ex:
        code=2
        write_json(out/'failure.json',{'status':'INCOMPLETE_FAIL_CLOSED_SHADOW_ONLY',
           'error_type':type(ex).__name__,'message':str(ex),
           'traceback':traceback.format_exc()[-18000:],
           'model_results_complete':False,'production_approved':False,
           'release_allowed':False,'holdout_used':False})
        print('WARM_E2E_BLOCKED',type(ex).__name__,str(ex),file=sys.stderr)
    finally:
        zip_results(out)
    return code

if __name__=='__main__':sys.exit(main())
