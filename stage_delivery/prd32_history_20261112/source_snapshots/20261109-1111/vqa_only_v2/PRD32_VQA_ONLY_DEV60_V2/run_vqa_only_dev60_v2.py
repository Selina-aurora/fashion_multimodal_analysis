#!/usr/bin/env python3
"""Fixed DEV60, real co-resident V5+Qwen, VQA-only SHADOW regression.

Reuses frozen V7 V5/Qwen and original baseline VQA prompt, not a synthetic
benchmark. Never invokes original 3-task Guard on partial output. Nothing can
be released. This evaluates original color VQA prompt, not all four PRD VQA types.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,statistics,sys,time,traceback,zipfile
from datetime import datetime
from pathlib import Path
from stage1_verified_runtime import (WORKSPACE,REPO,V5_RAW,IMAGE_ROOT,CKPT,RUNNER,QWEN_MODEL,
 v5_preflight,verified_runner,load_guard,refuse_restricted,digest_file,sync,write_json)
from joint_eval_stage2_dev60 import (MANIFEST_SHA256,manifest_verified,load_all_targets)
from run_architecture_ab_v1 import run_routed
from vqa_route_contract import validate_shadow_vqa,STATUS

HERE=Path(__file__).resolve().parent
SNAP=HERE/'VQA_FROZEN_BASELINE_171.json'
CHUNK_SIZE=24

def read_snap(manifest):
    p=SNAP.read_bytes()
    expected=(HERE/'SOURCE_SNAPSHOT_SHA256.txt').read_text('ascii').strip()
    if hashlib.sha256(p).hexdigest()!=expected:raise ValueError('frozen reference modified')
    s=json.loads(p)
    targets=s.get('targets',{})
    keys={r['case_id']+'/'+r['instance_id'] for r in manifest['targets']}
    if s.get('schema')!='PRD32_VQA_ONLY_DEV60_V2_FROZEN_BASELINE_REFERENCE' or len(targets)!=171 or set(targets)!=keys:
        raise ValueError('snapshot schema/coverage mismatch')
    return s

def choose(manifest,ch):
    rows=[r for r in manifest['targets'] if r['chunk_id']==ch]
    if len(rows)!=(24 if ch<8 else 3):raise ValueError('incomplete chunk')
    return [(r['case_id'],r['instance_id']) for r in rows]

def preflight(args,out):
    for p in (args.repo,args.raw_dir,args.image_root,args.checkpoint,args.runner,args.model_dir,SNAP,out):refuse_restricted(p)
    manifest,_=manifest_verified()
    ref=read_snap(manifest)
    keys=choose(manifest,args.chunk)
    _,source_hashes=v5_preflight(args.raw_dir,args.image_root,args.repo,args.checkpoint)
    targets=load_all_targets(args.raw_dir,args.image_root,manifest,keys)
    runner=verified_runner(args.runner)
    valid,core,helper,focus=runner.prompt_check(argparse.Namespace(
        snapshot=runner.DEFAULT_SNAP,source=runner.DEFAULT_SOURCE,
        focus_script=runner.DEFAULT_FOCUS,raw=runner.DEFAULT_RAW,input=runner.DEFAULT_INPUT))
    if len(valid)!=171:raise ValueError('baseline prompt snapshot drift')
    guard_source=args.repo/'src/fashion_multimodal_analysis/integration/prd32_output_guard.py'
    load_guard(guard_source)  # only checks source identity; never call 3-task guard on 1-task response
    if not (args.model_dir/'config.json').is_file():raise FileNotFoundError('Qwen missing config.json')
    write_json(out/'preflight.json',{
      'status':'VQA_ONLY_V2_PREFLIGHT_OK','chunk':args.chunk,'target_count':len(keys),
      'original_manifest_sha256':MANIFEST_SHA256,'baseline_vqa_reference_sha256':digest_file(SNAP),
      'v5_source_hashes':source_hashes,'v5_checkpoint_sha256':digest_file(args.checkpoint),
      'qwen_runner_sha256':digest_file(args.runner),'guard_source_sha256':digest_file(guard_source),
      'original_guard_checked_not_called':True,'source_images_verified':True,
      'no_gt_from_holdout':True,'production_release':False,'three_task_guard_contract_approved_for_vqa_only':False,
      'preflight_gpu_forward':False})
    return keys,targets,runner,core,helper,focus,ref

def approx_p95(a):
    b=sorted(float(x) for x in a); t=.95*(len(b)-1); lo=int(t);hi=min(lo+1,len(b)-1)
    return round(b[lo]+(b[hi]-b[lo])*(t-lo),3)

def run(args,out):
    keys,targets,runner,core,helper,focus,ref=preflight(args,out)
    if args.mode=='preflight':
        print('VQA_ONLY_V2_PREFLIGHT_PASS chunk='+str(args.chunk)+' targets='+str(len(keys))+' NO_GPU',flush=True);return
    if not args.execute_models:raise ValueError('real model run requires --execute-models')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch,numpy as np
    from PIL import Image
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
    free=torch.cuda.mem_get_info()[0]/1024**3
    if free<18:raise RuntimeError(f'need >=18GiB free GPU before model load, got {free:.2f}')
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    t_load=time.perf_counter()
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    qwen=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_dir),local_files_only=True,device_map='auto',torch_dtype=dtype,
        attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    processor=AutoProcessor.from_pretrained(str(args.model_dir),local_files_only=True,max_pixels=512*28*28)
    sync(torch)
    dm=getattr(qwen,'hf_device_map',None)
    if isinstance(dm,dict) and any(str(v).lower() in {'cpu','disk'} for v in dm.values()):
        raise ValueError('Qwen CPU/DISK offload detected')
    sys.path.insert(0,str(args.repo/'src'))
    from fashion_multimodal_analysis.common.schema import ID_TO_CLASS,CLASS_TO_ID
    from fashion_multimodal_analysis.segmentation.modeling import build_evaluation_model
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if cp.get('class_to_id')!=CLASS_TO_ID or cp.get('epoch')!=15 or cp.get('args',{}).get('epochs')!=15:
        raise ValueError('V5 checkpoint provenance mismatch')
    kw=cp.get('args',{})
    v5=build_evaluation_model(int(kw.get('min_size',640)),int(kw.get('max_size',1024)))
    v5.load_state_dict(cp['model_state_dict'],strict=True)
    device=torch.device('cuda');v5.to(device).eval();del cp
    sync(torch)
    write_json(out/'models_loaded.json',{'qwen_v5_same_gpu_process':True,
        'load_ms_excluded':round((time.perf_counter()-t_load)*1000,2),'model_dtype':str(dtype),
        'gpu_free_after_load_gib':round(torch.cuda.mem_get_info()[0]/1024**3,3),
        'cuda_model_name':torch.cuda.get_device_name(0),'qwen_device_map':dm})
    # Warm-up uses the same DEV60 key, but is never part of measured rows.
    run_routed(arm='vqa_only',label='warmup_excluded_vqa_only',key=keys[0],targets=targets,
        v5_model=v5,qwen=qwen,processor=processor,runner=runner,core=core,helper=helper,
        focus=focus,device=device,torch=torch,np=np,Image=Image,ID_TO_CLASS=ID_TO_CLASS,out=out)
    records=[]
    for i,key in enumerate(keys,1):
        case,inst=key
        label=f'measured_t{i:03d}_vqa_only'
        result=run_routed(arm='vqa_only',label=label,key=key,targets=targets,
           v5_model=v5,qwen=qwen,processor=processor,runner=runner,core=core,helper=helper,
           focus=focus,device=device,torch=torch,np=np,Image=Image,ID_TO_CLASS=ID_TO_CLASS,out=out)
        old=ref['targets'][case+'/'+inst]
        if result['source_mask_supplied_as_qwen_pixels'] is not False or result['qwen_calls']!=1 or result['task_calls_made']!=['vqa']:
            raise ValueError('route unexpectedly executed non-VQA task')
        if result['guard_contract_validated_for_this_arm'] is not False or result['publish_allowed'] is not False:
            raise ValueError('partial Guard contract incorrectly marked approved')
        audit=validate_shadow_vqa({'case_id':case,'instance_id':inst,'vqa_answer':result['outputs']['vqa'],
            'v5_mask_iou':result['frozen_mask_iou'],'source_image_sha256':targets[key]['image_sha256'],
            'publish_allowed':False})
        if old['image_sha256']!=targets[key]['image_sha256']:raise ValueError('baseline image drift')
        rec={'case_id':case,'instance_id':inst,'chunk':args.chunk,'route':'VQA_ONLY_SHADOW',
          'vqa_answer':result['outputs']['vqa'],'vqa_response_sha256':audit['response_sha256'],
          'vqa_matches_frozen_baseline':audit['response_sha256']==old['vqa_response_sha256'],
          'baseline_vqa_response_sha256':old['vqa_response_sha256'],
          'vqa_host_task_ms':result['vqa_task_host_wall_ms'],'v5_host_ms':result['v5_host_wall_ms'],
          'vqa_only_total_ms':result['total_host_wall_ms'],
          'historical_full_three_task_ms':old['baseline_full_ms'],
          'historical_vqa_task_ms':old['vqa_host_task_ms'],
          'frozen_mask_iou':result['frozen_mask_iou'],
          'status':STATUS,'holdout_used':False,'production_release':False,
          'original_guard_invoked':False,'human_quality_gt':False}
        records.append(rec)
        print('VQA_ONLY_V2_RESULT',i,'/',len(keys),case,inst,
          f'total_ms={rec["vqa_only_total_ms"]:.2f}',
          f'output_same={rec["vqa_matches_frozen_baseline"]}',
          'release=false',flush=True)
        with (out/'VQA_ONLY_PER_TARGET.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
    # Verify actual task call count, including excluded warmup.
    calls=[json.loads(line) for line in (out/'alternative_qwen_calls.jsonl').read_text(encoding='utf8').splitlines()]
    if len(calls)!=len(keys)+1 or any(x['task']!='vqa' or x.get('publish_allowed') is not False for x in calls):
        raise ValueError('unexpected Qwen calls (not VQA-only)')
    if len(records)!=len(keys) or len({(r['case_id'],r['instance_id']) for r in records})!=len(keys):
        raise ValueError('missing/duplicate measured cases')
    with (out/'VQA_ONLY_PER_TARGET.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    a=[r['vqa_only_total_ms'] for r in records]
    summary={'status':'VQA_ONLY_DEV60_V2_CHUNK_COMPLETE_SHADOW_BLOCKED','schema':'vqa_only_dev60_v2',
      'chunk':args.chunk,'target_count':len(keys),'measured_vqa_calls':len(keys),
      'excluded_warmups':1,'model_configuration':'frozen BF16 Qwen2.5-VL-7B SDPA + V5',
      'mean_total_ms':round(statistics.mean(a),3),'p95_total_ms':approx_p95(a),
      'under_200ms_count':sum(x<=200 for x in a),
      'under_400ms_count':sum(x<=400 for x in a),
      'vqa_response_equal_baseline_count':sum(r['vqa_matches_frozen_baseline'] for r in records),
      'per_instance_host_vqa_ms_mean':round(statistics.mean(r['vqa_host_task_ms'] for r in records),3),
      'original_guard_invoked':False,'new_guard_contract_approved':False,
      'release_allowed':False,'independent_holdout_used':False,'official_acceptance':False,
      'notes':'Original VQA is one fixed color question; no all-type VQA or production Guard approval.'}
    write_json(out/'VQA_ONLY_CHUNK_SUMMARY.json',summary)
    print('VQA_ONLY_V2_CHUNK_COMPLETE',args.chunk,len(keys),'shadow_release=false',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['preflight','live'],default='preflight')
    p.add_argument('--execute-models',action='store_true')
    p.add_argument('--chunk',type=int,choices=range(1,9),default=1)
    p.add_argument('--out',type=Path)
    p.add_argument('--repo',type=Path,default=REPO);p.add_argument('--raw-dir',type=Path,default=V5_RAW)
    p.add_argument('--image-root',type=Path,default=IMAGE_ROOT);p.add_argument('--checkpoint',type=Path,default=CKPT)
    p.add_argument('--runner',type=Path,default=RUNNER);p.add_argument('--model-dir',type=Path,default=QWEN_MODEL)
    a=p.parse_args()
    out=a.out or WORKSPACE/f'PRD32_VQA_ONLY_V2_CHUNK{a.chunk:02d}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}'
    if out.exists():raise SystemExit('output exists; use a new output directory')
    out.mkdir(parents=True)
    status=0
    try:run(a,out)
    except BaseException as ex:
        status=2
        write_json(out/'FAILED_CLOSED.json',{'error_type':type(ex).__name__,'reason':str(ex),
            'traceback':traceback.format_exc()[-16000:],'publish_allowed':False,
            'completed':False,'holdout_used':False})
        print('VQA_ONLY_V2_BLOCKED',type(ex).__name__,str(ex),file=sys.stderr,flush=True)
    finally:
        z=out.with_suffix('.zip')
        with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
            for f in sorted(out.rglob('*')):
                if f.is_file():archive.write(f,f.relative_to(out).as_posix())
        print('UPLOAD_THIS:',z,flush=True)
    return status
if __name__=='__main__':sys.exit(main())
