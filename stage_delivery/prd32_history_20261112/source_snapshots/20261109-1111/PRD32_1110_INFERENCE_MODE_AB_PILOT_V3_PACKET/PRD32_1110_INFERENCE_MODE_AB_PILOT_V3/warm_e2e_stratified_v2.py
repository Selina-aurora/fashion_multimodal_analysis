#!/usr/bin/env python3
"""DEV60-only 24-target, 2-repeat warm V5->Qwen->Guard stratified shadow timing.

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
    QWEN_SMOKE_TARGETS, EXPECTED_MODULE_HASHES, V5_CKPT_SHA256, V5_RAW_SHA256, CATEGORIES, verified_train_image, read_jsonl, read_json,
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
MANIFEST_SHA256='6b8a3d1ddaa8f65a712e43294f26d2f07436f8213edaa70aba76bc3dc90877b1'
DEFAULT_MANIFEST=Path(__file__).resolve().parent/'sample_manifest_v2.json'


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


def load_manifest(path):
    path=Path(path)
    if digest_file(path)!=MANIFEST_SHA256:
        raise ValueError('target sampling manifest changed; refuse uncontrolled selection drift')
    m=json.loads(path.read_text(encoding='utf-8'))
    if m.get('schema')!='prd32_dev60_stratified_timing_manifest_v2' or m.get('target_count')!=24:
        raise ValueError('unexpected manifest schema/size')
    keys=[]; counts={}
    for r in m.get('targets',[]):
        key=(r['case_id'],r['instance_id'])
        if not (len(key[0])==13 and key[0].startswith('fresh_dev_') and key[0][10:].isdigit() and
                len(key[1])==9 and key[1].startswith('pred_') and key[1][5:].isdigit()):
            raise ValueError('unexpected DEV key')
        cat=r['v5_frozen_category']
        if cat not in CATEGORIES or r['stratum']!=cat:
            raise ValueError('unexpected stratum')
        if key in keys:raise ValueError('duplicate target key')
        keys.append(key)
        counts[cat]=counts.get(cat,0)+1
    if set(counts)!=CATEGORIES or set(counts.values())!={3}:
        raise ValueError('manifest not 3 samples per V5 predicted category')
    return m,keys


def load_extended_targets(raw_dir, image_root, manifest, selected_keys):
    # Existing v5_preflight validates original hashes, checkpoint, mask and runtime implementation.
    # This additional read extends the keys WITHOUT editing the pinned V7 contract.
    raw_file=raw_dir/'v5_raw_predictions.jsonl'
    if digest_file(raw_file)!=V5_RAW_SHA256:raise ValueError('frozen DEV60 V5 raw hash drift')
    selected=set(selected_keys)
    found={}
    records=read_jsonl(raw_file)
    if len(records)!=60:raise ValueError('DEV60 source image count drift')
    map_cat={(r['case_id'],r['instance_id']):r['v5_frozen_category'] for r in manifest['targets']}
    for idx,rec in enumerate(records,1):
        cid=f'fresh_dev_{idx:03d}'
        if rec.get('case_id')!=cid:raise ValueError('frozen case order changed')
        wanted=[k for k in selected if k[0]==cid]
        if not wanted:continue
        img=verified_train_image(Path(rec['source_image_path']),image_root,rec['image_sha256'])
        from PIL import Image
        with Image.open(img) as im:
            if list(im.size)!=list(rec['image_size']):raise ValueError('image dimensions changed')
        for j,ins in enumerate(rec['instances']):
            key=(cid,f'pred_{j:04d}')
            if key not in selected:continue
            if ins.get('instance_id')!=key[1] or ins.get('predicted_category')!=map_cat[key]:
                raise ValueError('frozen instance category drift: '+str(key))
            mask=raw_dir/f'{cid}_{key[1]}_mask.png'
            if not mask.is_file():raise FileNotFoundError(mask)
            found[key]={'case_id':cid,'instance_id':key[1], 'image_path':str(img),
                'image_sha256':rec['image_sha256'],'image_size':rec['image_size'],
                'expected_category':ins['predicted_category'],
                'expected_bbox':ins['bbox_xyxy'],'expected_score':ins['score'],
                'frozen_mask_path':str(mask),'expected_instance_count':len(rec['instances'])}
    if set(found)!=selected:raise ValueError('target missing from frozen 171-instance DEV60')
    return found


def token_stats(values):
    vals=[float(v) for v in values]
    if not vals:return None
    return {'n':len(vals),'mean_tokens':round(statistics.mean(vals),2),
            'p50_tokens':percentile(vals,50),'p95_tokens':percentile(vals,95),
            'max_tokens':int(max(vals))}


def calculate_report(measures, calls, keys, repeats):
    fields=['total_host_wall_ms','v5_host_wall_ms','tags_task_host_wall_ms',
            'vqa_task_host_wall_ms','copy_task_host_wall_ms','guard_host_wall_ms']
    by_target={}
    for r in measures:
        key=(r['case_id'],r['instance_id'])
        by_target.setdefault(key,[]).append(r)
    if len(by_target)!=len(keys) or any(len(v)!=repeats for v in by_target.values()):
        raise ValueError('incomplete target/repeat timing rows')
    target_avg=[{'case_id':k[0],'instance_id':k[1],
        'category':v[0]['producer_category'],
        **{name:round(statistics.mean(float(r[name]) for r in v),3) for name in fields}}
        for k,v in sorted(by_target.items())]
    by_cat={}
    for r in target_avg:by_cat.setdefault(r['category'],[]).append(r)
    per_task={}
    for task in TASKS:
        q=[c for c in calls if c['task']==task and c['cohort']!='warmup_excluded']
        if len(q)!=len(measures):raise ValueError('incomplete task call timing records')
        per_task[task]={
            'generated_tokens':token_stats([c['model_runner_timing']['generated_tokens'] for c in q]),
            'model_generation_ms':stats([c['model_runner_timing']['generation_ms'] for c in q]),
            'model_preprocess_ms':stats([c['model_runner_timing']['preprocess_ms'] for c in q]),
            'tokens_per_generation_second_mean':round(statistics.mean(
                1000*float(c['model_runner_timing']['generated_tokens'])/float(c['model_runner_timing']['generation_ms'])
                for c in q if float(c['model_runner_timing']['generation_ms'])>0),3)}
    return {'request_level':{f:stats([r[f] for r in measures]) for f in fields},
      'unique_target_level':{f:stats([r[f] for r in target_avg]) for f in fields},
      'strata_target_means':{cat:{f:stats([r[f] for r in rr]) for f in fields}
                  for cat,rr in sorted(by_cat.items())},
      'qwen_token_and_generation_cost':per_task,'per_target_avg':target_avg}


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
            'cohort':label,'invocation_uid':label+'__'+host_id,'case_id':case,'instance_id':iid,'task':task,
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
         'producer_category':category, 'image_width':int(frozen['image_size'][0]),
         'image_height':int(frozen['image_size'][1]), 'num_v5_predicted_instances':int(len(labels)),
         'frozen_bbox_iou':match['bbox_iou'],
         'frozen_mask_iou':match['mask_iou'],
         'gpu_peak_allocated_mib':round(torch.cuda.max_memory_allocated()/1048576,1),
         'publish_allowed':False,
         'holdout_accessed':False}
    dump_jsonl(out/'per_target_events.jsonl',row)
    return row


def run(args,out):
    # preflight: pinned weights, source modules, DEV60 training images, Qwen prompt bytes.
    for p in (args.repo,args.raw_dir,args.image_root,args.checkpoint,args.runner,args.model_dir,args.manifest,out):
        refuse_restricted(p)
    original_pinned_targets,modules=v5_preflight(args.raw_dir,args.image_root,args.repo,args.checkpoint)
    manifest,keys=load_manifest(args.manifest)
    targets=load_extended_targets(args.raw_dir,args.image_root,manifest,keys)
    if len(original_pinned_targets)!=12:raise ValueError('original V7 pinned preflight drift')
    runner=verified_runner(args.runner)
    valid_rows,core,helper,focus=runner.prompt_check(argparse.Namespace(
        snapshot=runner.DEFAULT_SNAP, source=runner.DEFAULT_SOURCE,
        focus_script=runner.DEFAULT_FOCUS,raw=runner.DEFAULT_RAW,input=runner.DEFAULT_INPUT))
    if len(valid_rows)!=171:raise ValueError('Qwen original DEV60 runner snapshot changed')
    guard=load_guard(args.repo/'src/fashion_multimodal_analysis/integration/prd32_output_guard.py')
    if not (args.model_dir/'config.json').is_file():raise FileNotFoundError('Qwen local model config missing')
    write_json(out/'preflight.json',{'status':'PINNED_DEV60_PREFLIGHT_OK',
       'selected_targets':[f'{c}/{i}' for c,i in keys], 'selection_manifest_sha256':MANIFEST_SHA256,
       'v5_source_hashes':modules,'v5_checkpoint_sha256':V5_CKPT_SHA256,
       'qwen_runner_sha256':digest_file(args.runner),
       'guard_sha256':digest_file(args.repo/'src/fashion_multimodal_analysis/integration/prd32_output_guard.py'),
       'mode':args.mode,'model_calls_performed':False,
       'holdout_used':False,'publish_allowed':False})
    if args.mode=='preflight':
        print('DEV60_STRATIFIED_24_PREFLIGHT_PASS; NO MODEL EXECUTION; NO HOLDOUT')
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

    # One real warm-up, excluded. Repeated measured requests are correlated, not independent samples.
    plan=[('warmup_excluded',keys[0])]
    for repeat in range(1,args.repeats+1):
        for index,key in enumerate(keys,1):
            plan.append((f'measured_r{repeat:02d}_t{index:02d}',key))
    measures=[]
    for label,key in plan:
        row=run_one(key=key,label=label,targets=targets,v5_model=v5,
            qwen=qwen,processor=processor,runner=runner,core=core,helper=helper,
            focus=focus,guard=guard,device=device,torch=torch,np=np,Image=Image,
            ID_TO_CLASS=ID_TO_CLASS,out=out)
        if label!='warmup_excluded':measures.append(row)
        print(f'STRATIFIED_WARM_EVENT {label} {key[0]}/{key[1]} total_ms={row["total_host_wall_ms"]} guard={row["guard_status"]}',flush=True)
    calls=[json.loads(s) for s in (out/'qwen_calls.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(calls)!=3*len(plan) or len({x['invocation_uid'] for x in calls})!=len(calls):
        raise ValueError('Qwen invocation count or unique identity mismatch')
    report=calculate_report(measures,calls,keys,args.repeats)
    summary={'status':'DEV60_24_TARGET_2_REPEAT_WARM_E2E_COMPLETE_SHADOW_ONLY',
             'co_resident_models_same_python_process':True,
             'sample_manifest_sha256':MANIFEST_SHA256,'n_unique_targets':len(keys),
             'n_unique_source_images':len({targets[k]['image_sha256'] for k in keys}),
             'n_repeats_per_target':args.repeats,
             'warmup_excluded':1,'measured_request_count':len(measures),
             'measured_qwen_calls':len(measures)*3,
             'total_qwen_calls_including_warmup':len(plan)*3,
             'timing_analysis':{k:v for k,v in report.items() if k!='per_target_avg'},
             'qwen_load_ms':round(qwen_load_ms,3),'v5_load_ms':round(v5_load_ms,3),
             'source_mask_supplied_to_qwen':False,
             'source_mask_verified_against_frozen':True,
             'guard_review_required_all':True,
             'model_weights_and_prompts_unchanged':True,
             'independent_holdout_used':False,'production_approved':False,
             'release_allowed':False,
             'scope_warning':'Balanced by V5 predicted class (3 each), NOT prevalence weighted or random DEV60; 2 repeats correlated; p95 exploratory; no online service or Holdout'}
    write_json(out/'summary.json',summary)
    with (out/'per_request_timing.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(measures[0]),extrasaction='ignore');w.writeheader();w.writerows(measures)
    with (out/'per_target_mean_timing.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(report['per_target_avg'][0]),extrasaction='ignore');w.writeheader();w.writerows(report['per_target_avg'])
    print('DEV60_STRATIFIED_24x2_COMPLETE_SHADOW_ONLY',len(measures),'measured requests; mean/p95 ms',
          report['request_level']['total_host_wall_ms']['mean_ms'],
          report['request_level']['total_host_wall_ms']['p95_ms'],flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['preflight','live'],default='preflight')
    p.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    p.add_argument('--repeats',type=int,choices=[2],default=2)
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
    except BaseException as ex:
        code=2
        write_json(out/'failure.json',{'status':'INCOMPLETE_FAIL_CLOSED_SHADOW_ONLY',
           'error_type':type(ex).__name__,'message':str(ex),
           'traceback':traceback.format_exc()[-18000:],
           'model_results_complete':False,'partial_results_are_not_a_completed_benchmark':True,'production_approved':False,
           'release_allowed':False,'holdout_used':False})
        print('WARM_E2E_BLOCKED',type(ex).__name__,str(ex),file=sys.stderr)
    finally:
        zip_results(out)
    return code

if __name__=='__main__':sys.exit(main())
