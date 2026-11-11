#!/usr/bin/env python3
"""Fixed 171-predicted-instance DEV60 warm E2E evaluation, 8 separately rerunnable chunks.

Uses UNMODIFIED Stage1 run_one for V5 + Qwen Tags/VQA/Copy + Guard.
Reads NO holdout. Does NO training/production release. All metrics development-only.
"""
from __future__ import annotations
import argparse,csv,json,math,os,sys,time,traceback,statistics
from datetime import datetime
from pathlib import Path
from stage1_verified_runtime import (
    WORKSPACE,REPO,V5_RAW,IMAGE_ROOT,CKPT,RUNNER,QWEN_MODEL,
    v5_preflight, load_extended_targets, verified_runner,load_guard,
    refuse_restricted,run_one,calculate_report,write_json,dump_jsonl,zip_results,
    digest_file,sync,stats,TASKS, V5_CKPT_SHA256,
)

MANIFEST_SHA256 = "eff0be8fb8e53d0ed7f8f228e668571b2461fa896c376545de72f65e252482f9"
MANIFEST = Path(__file__).resolve().parent/'full_dev60_171_manifest_v1.json'
EXPECTED_CHUNKS = 8
CHUNK_SIZE = 24

def manifest_verified(path=MANIFEST):
    if digest_file(path)!=MANIFEST_SHA256:
        raise ValueError('DEV60 171 manifest SHA changed; refuse execution')
    d=json.loads(Path(path).read_text(encoding='utf-8'))
    if d.get('schema')!='prd31_prd32_joint_eval_stage2_fixed_dev60_171_v1' or d.get('target_count')!=171 or d.get('source_image_count')!=60 or d.get('chunk_count')!=8 or d.get('chunk_size')!=24:
        raise ValueError('DEV60 171 manifest contract drift')
    if d['release_allowed'] is not False or d['independent_holdout_used'] is not False:
        raise ValueError('holdout/release not permitted')
    rr=d['targets'];keys=[(r['case_id'],r['instance_id']) for r in rr]
    if len(keys)!=171 or len(set(keys))!=171 or len({r['case_id'] for r in rr})!=60 or {r['fixed_plan_index'] for r in rr}!=set(range(1,172)):
        raise ValueError('not exactly 171 distinct fixed-ordered DEV60 prediction targets across 60 images')
    if any(r['fixed_plan_index']!=i or r['chunk_id']!=(i-1)//24+1 for i,r in enumerate(rr,1)):
        raise ValueError('chunk ordering differs')
    if sum(r['coverage_group']=='NEW_NEVER_LIVE_MEASURED' for r in rr)!=135:
        raise ValueError('original coverage groups drift')
    if sum(r['coverage_group']=='STAGE1_PILOT12_PREVIOUSLY_MEASURED' for r in rr)!=12:
        raise ValueError('stage1 coverage group drift')
    if sum(r['coverage_group']=='PRIOR_WARM24_PREVIOUSLY_MEASURED' for r in rr)!=24:
        raise ValueError('warm24 coverage group drift')
    if any(not r['case_id'].startswith('fresh_dev_') or not r['instance_id'].startswith('pred_') for r in rr):
        raise ValueError('non-DEV60 target identifier')
    return d,keys

def load_all_targets(raw_dir,image_root,manifest,keys):
    adapter={'targets':[{'case_id':r['case_id'],'instance_id':r['instance_id'],
       'v5_frozen_category':r['v5_frozen_category']} for r in manifest['targets']]}
    found=load_extended_targets(raw_dir,image_root,adapter,keys)
    sha_map={(r['case_id'],r['instance_id']):r['source_image_sha256'] for r in manifest['targets']}
    for k,v in found.items():
        if v['image_sha256']!=sha_map[k]:
            raise ValueError('source image hash mismatch '+str(k))
    return found

def run(args,out):
    for p in (args.repo,args.raw_dir,args.image_root,args.checkpoint,args.runner,args.model_dir,MANIFEST,out):
        refuse_restricted(p)
    if not 1<=args.chunk<=EXPECTED_CHUNKS:raise ValueError('chunk outside 1..8')
    d,all_keys=manifest_verified()
    # Tested V7 pinned set, hashes of source files/checkpoint, original snapshots.
    pinned,modules=v5_preflight(args.raw_dir,args.image_root,args.repo,args.checkpoint)
    if len(pinned)!=12:raise ValueError('original pinned source contract drift')
    targets=load_all_targets(args.raw_dir,args.image_root,d,all_keys)
    chosen=[(r['case_id'],r['instance_id']) for r in d['targets'] if r['chunk_id']==args.chunk]
    if len(chosen)!=(24 if args.chunk<=7 else 3):
        raise ValueError('unexpected chunk target count')
    runner=verified_runner(args.runner)
    valid_rows,core,helper,focus=runner.prompt_check(argparse.Namespace(
        snapshot=runner.DEFAULT_SNAP,source=runner.DEFAULT_SOURCE,
        focus_script=runner.DEFAULT_FOCUS,raw=runner.DEFAULT_RAW,input=runner.DEFAULT_INPUT))
    if len(valid_rows)!=171:raise ValueError('Qwen original snapshot count drift')
    guard=load_guard(args.repo/'src/fashion_multimodal_analysis/integration/prd32_output_guard.py')
    if not (args.model_dir/'config.json').is_file():raise FileNotFoundError('Qwen model missing')
    write_json(out/'preflight.json',{'status':'FIXED_DEV60_171_CHUNK_PREFLIGHT_OK',
        'manifest_sha256':MANIFEST_SHA256,'chunk':args.chunk,
        'chunk_targets':[f'{a}/{b}' for a,b in chosen],
        'all_171_targets_and_60_images_verified':True,
        'v5_source_hashes':modules,'v5_checkpoint_sha256':V5_CKPT_SHA256,
        'qwen_runner_sha256':digest_file(args.runner),
        'guard_sha256':digest_file(args.repo/'src/fashion_multimodal_analysis/integration/prd32_output_guard.py'),
        'holdout_accessed':False,'publish_allowed':False,'live_calls_performed':False})
    if args.mode=='preflight':
        print('JOINT_STAGE2_PREFLIGHT_PASS: CHUNK',args.chunk,'targets',len(chosen),'NO_GPU_FORWARD')
        return
    if not args.execute_models:raise ValueError('live requires --execute-models')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch,numpy as np
    from PIL import Image
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    gpu_free=torch.cuda.mem_get_info()[0]/1024**3
    if gpu_free<18:raise RuntimeError('less than 18 GiB free GPU before loading models')
    from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    t_load=time.perf_counter()
    qwen=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_dir),local_files_only=True,device_map='auto',
        torch_dtype=dtype,attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    processor=AutoProcessor.from_pretrained(str(args.model_dir),local_files_only=True,max_pixels=512*28*28)
    sync(torch)
    dm=getattr(qwen,'hf_device_map',None)
    if isinstance(dm,dict) and any(str(v).lower() in {'cpu','disk'} for v in dm.values()):
        raise RuntimeError('Qwen offloaded to CPU/disk')
    qload=(time.perf_counter()-t_load)*1000
    write_json(out/'qwen_loaded.json',{'qwen_load_ms':round(qload,3),'device_map':dm,'warm_timing_excludes_load':True})
    sys.path.insert(0,str(args.repo/'src'))
    from fashion_multimodal_analysis.common.schema import ID_TO_CLASS,CLASS_TO_ID
    from fashion_multimodal_analysis.segmentation.modeling import build_evaluation_model
    t_vload=time.perf_counter()
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if cp.get('class_to_id')!=CLASS_TO_ID or cp.get('epoch')!=15 or cp.get('args',{}).get('epochs')!=15:
        raise ValueError('wrong V5 epoch/class map')
    saved=cp.get('args',{})
    v5=build_evaluation_model(int(saved.get('min_size',640)),int(saved.get('max_size',1024)))
    v5.load_state_dict(cp['model_state_dict'],strict=True)
    device=torch.device('cuda');v5.to(device).eval();del cp
    sync(torch)
    vload=(time.perf_counter()-t_vload)*1000
    free_after=torch.cuda.mem_get_info()[0]/1024**3
    write_json(out/'co_residency.json',{'status':'V5_QWEN_LOADED_IN_SAME_PROCESS',
        'gpu_free_gib_after_both_loaded':round(free_after,3), 'v5_load_ms':round(vload,3)})
    if free_after<1.0:raise RuntimeError('insufficient GPU headroom: <1 GiB')
    torch.cuda.reset_peak_memory_stats()
    plan=[('warmup_excluded',chosen[0])]+[(f'measured_t{j:03d}',key) for j,key in enumerate(chosen,1)]
    measures=[]
    for ix,(label,key) in enumerate(plan):
        row=run_one(key=key,label=label,targets=targets,v5_model=v5,
            qwen=qwen,processor=processor,runner=runner,core=core,helper=helper,
            focus=focus,guard=guard,device=device,torch=torch,np=np,Image=Image,
            ID_TO_CLASS=ID_TO_CLASS,out=out)
        if label!='warmup_excluded':measures.append(row)
        print('JOINT_STAGE2_EVENT',args.chunk,ix,'/',len(plan)-1,
              key[0]+'/'+key[1],round(row['total_host_wall_ms'],1),'ms',flush=True)
    calls=[json.loads(s) for s in (out/'qwen_calls.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(calls)!=3*len(plan) or len({c['invocation_uid'] for c in calls})!=len(calls):
        raise ValueError('missing or duplicate Qwen calls')
    if len(measures)!=len(chosen) or len(set((r['case_id'],r['instance_id']) for r in measures))!=len(chosen):
        raise ValueError('incomplete target coverage')
    if any(r['guard_status']!='REVIEW_REQUIRED_NOT_RELEASED' or r['publish_allowed'] is not False or
           float(r['frozen_mask_iou'])<0.9999 or r['holdout_accessed'] is not False for r in measures):
        raise ValueError('risk guard or frozen identity contract drift')
    rep=calculate_report(measures,calls,chosen,1)
    summary={'status':'PRD31_PRD32_STAGE2_CHUNK_COMPLETE_SHADOW_ONLY','chunk':args.chunk,
        'manifest_sha256':MANIFEST_SHA256,'target_count':len(chosen),
        'new_previously_unmeasured_target_count':sum(x['coverage_group']=='NEW_NEVER_LIVE_MEASURED' for x in d['targets'] if x['chunk_id']==args.chunk),
        'source_image_count':len({targets[k]['image_sha256'] for k in chosen}),
        'warmup_excluded':1,'measured_qwen_calls':3*len(chosen),
        'total_qwen_calls_including_warmup':3*(len(chosen)+1),
        'timing_analysis':{k:v for k,v in rep.items() if k!='per_target_avg'},
        'qwen_model_load_ms':round(qload,3),'v5_model_load_ms':round(vload,3),
        'source_mask_supplied_as_qwen_pixel_input':False,'all_frozen_mask_comparisons_passed':True,
        'all_guard_blocked':True,'publish_allowed':False,
        'holdout_accessed':False,'production_approved':False,
        'quality_gt':False,'scope_warning':'DEV60 V5-predicted instances only; not full true garment census, not formal PRD3.2 VQA/Tags/Copy acceptance. Baseline prompt only.'}
    write_json(out/'summary.json',summary)
    with (out/'per_request_timing.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(measures[0]));w.writeheader();w.writerows(measures)
    with (out/'per_target_mean_timing.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rep['per_target_avg'][0]));w.writeheader();w.writerows(rep['per_target_avg'])
    print('JOINT_STAGE2_CHUNK_COMPLETE',args.chunk,'measured',len(measures),
          'mean_ms',rep['request_level']['total_host_wall_ms']['mean_ms'])

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['preflight','live'],default='preflight')
    p.add_argument('--chunk',type=int,choices=range(1,9),required=True)
    p.add_argument('--execute-models',action='store_true')
    p.add_argument('--repo',type=Path,default=REPO)
    p.add_argument('--raw-dir',type=Path,default=V5_RAW)
    p.add_argument('--image-root',type=Path,default=IMAGE_ROOT)
    p.add_argument('--checkpoint',type=Path,default=CKPT)
    p.add_argument('--runner',type=Path,default=RUNNER)
    p.add_argument('--model-dir',type=Path,default=QWEN_MODEL)
    p.add_argument('--out',type=Path,default=None)
    args=p.parse_args(argv)
    out=args.out or WORKSPACE/f'PRD31_PRD32_STAGE2_CHUNK{args.chunk:02d}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    if out.exists():raise FileExistsError('refuse overwrite: '+str(out))
    refuse_restricted(out)
    out.mkdir(parents=True,exist_ok=False)
    code=0
    try:
        run(args,out)
    except BaseException as exc:
        code=2
        write_json(out/'failure.json',{'status':'INCOMPLETE_FAIL_CLOSED_SHADOW_ONLY',
            'error_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()[-18000:],
            'partial_not_valid_benchmark':True,'publish_allowed':False,'holdout_accessed':False})
        print('JOINT_STAGE2_BLOCKED',type(exc).__name__,str(exc),file=sys.stderr)
    finally:
        zip_results(out)
    return code

if __name__=='__main__':sys.exit(main())
