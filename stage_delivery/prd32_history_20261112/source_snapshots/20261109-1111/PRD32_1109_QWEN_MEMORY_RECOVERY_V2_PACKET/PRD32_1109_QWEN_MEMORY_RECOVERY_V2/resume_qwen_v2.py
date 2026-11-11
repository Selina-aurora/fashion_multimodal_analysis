#!/usr/bin/env python3
"""Read-only diagnosis or bounded Qwen-only DEV12 retry of a blocked timing run.

Never modifies frozen V5 artifacts, runner code or Holdout. This is NOT a warm
end-to-end benchmark. Qwen live output is always written to a new folder.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

REQUIRED_FREE_GIB = 18.0
TARGET_COUNT = 12
QWEN_TARGET_COUNT = 4
TASK_SET = {'tags', 'vqa', 'copy'}
BAD_PARTS = ('holdout', 'sealed', 'blind', 'independent_test')


def sha256_path(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_dev_path(path):
    if any(any(b in part.lower() for b in BAD_PARTS) for part in path.parts):
        raise ValueError('restricted path: ' + str(path))
    return path


def audit_producer(run_dir):
    check_dev_path(run_dir)
    failure = json.loads((run_dir/'failure.json').read_text(encoding='utf8'))
    summary = json.loads((run_dir/'V5_PRODUCER'/'producer_summary.json').read_text(encoding='utf8'))
    events = json.loads((run_dir/'V5_PRODUCER'/'producer_events.json').read_text(encoding='utf8'))
    if failure.get('status')!='INCOMPLETE_NO_RELEASE' or not re.search(r'Qwen.*18GiB', failure.get('message',''),re.I | re.S):
        raise ValueError('this utility accepts only the recorded Qwen >=18GiB block')
    if summary.get('model_calls')!=10 or summary.get('targets')!=TARGET_COUNT or summary.get('all_frozen_comparisons_passed') is not True:
        raise ValueError('V5 producer summary is not complete')
    if summary.get('holdout_used') is not False or summary.get('production_approved') is not False:
        raise ValueError('producer not in DEV-only, nonrelease mode')
    if len(events)!=TARGET_COUNT or len({(x['case_id'], x['instance_id']) for x in events})!=TARGET_COUNT:
        raise ValueError('producer event count or keys invalid')
    per_image={}
    checked=0
    producer_root=(run_dir/'V5_PRODUCER').resolve(strict=True)
    for e in events:
        rel=e['source_mask_relative_path']
        f=(producer_root/rel).resolve(strict=True)
        if not f.is_relative_to(producer_root) or sha256_path(f)!=e['mask_png_sha256']:
            raise ValueError('producer mask path/hash mismatch: '+str(rel))
        checked+=1
        if e.get('holdout_accessed') is not False or e.get('publish_allowed') is not False:
            raise ValueError('unsafe producer event')
        if e['frozen_comparison']['mask_iou'] < .98 or e['frozen_comparison']['bbox_iou'] < .995:
            raise ValueError('producer frozen comparison failed')
        c=e['case_id']; t=float(e['producer_inference_ms'])
        if not math.isfinite(t) or t<0 or (c in per_image and abs(t-per_image[c])>.001):
            raise ValueError('invalid per-image V5 timing')
        per_image[c]=t
    if len(per_image)!=10: raise ValueError('not 10 unique image forwards')
    return {'status':'V5_ARTIFACTS_VERIFIED_QWEN_BLOCKED','producer_events':checked,
            'unique_image_forwards':len(per_image), 'mask_sha256_verified':checked,
            'failure_cause':'Qwen minimum free GPU VRAM check >=18GiB not met',
            'v5_inference_times_ms':per_image,'qwen_calls_measured':0,
            'true_full_chain_latency_ms':None,'production_approved':False,'holdout_accessed':False}


def gpu_snapshot():
    result={'nvidia_smi':None,'cuda_free_gib':None,'cuda_total_gib':None,
            'allowed_for_qwen':False, 'threshold_free_gib':REQUIRED_FREE_GIB}
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
                         '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=12)
        result['nvidia_smi']=p.stdout.strip() if p.returncode==0 else 'nvidia-smi query failed'
    except (OSError,subprocess.TimeoutExpired):
        result['nvidia_smi']='nvidia-smi unavailable'
    try:
        import torch
        if torch.cuda.is_available():
            free,total=torch.cuda.mem_get_info()
            result['cuda_free_gib']=round(free/1024**3,3)
            result['cuda_total_gib']=round(total/1024**3,3)
            result['allowed_for_qwen']=(free/1024**3)>=REQUIRED_FREE_GIB
    except (ImportError,RuntimeError) as exc:
        result['cuda_error']=type(exc).__name__+': '+str(exc)[:200]
    return result


def summarize_qwen_output(qwen_dir):
    qpath=qwen_dir/'qwen_shadow_results.jsonl'
    gpath=qwen_dir/'guard_shadow_report.json'
    rows=[json.loads(s) for s in qpath.read_text(encoding='utf8').splitlines() if s.strip()]
    guard=json.loads(gpath.read_text(encoding='utf8'))
    if len(rows)!=QWEN_TARGET_COUNT*3:
        raise ValueError('unexpected number of Qwen task rows')
    by_target=defaultdict(set)
    by_task=defaultdict(list)
    for r in rows:
        if r.get('status')!='GENERATED_REVIEW_REQUIRED' or r.get('publish_allowed') is not False:
            raise ValueError('unverified release policy on model output')
        task=r.get('task')
        if task not in TASK_SET:raise ValueError('unknown task')
        key=(r.get('case_id'),r.get('target_instance_id'))
        if task in by_target[key]:raise ValueError('duplicate Qwen task')
        by_target[key].add(task)
        value=float(r['timing']['end_to_end_ms'])
        if not math.isfinite(value) or value<0:raise ValueError('invalid Qwen timing')
        by_task[task].append(value)
    if len(by_target)!=QWEN_TARGET_COUNT or any(s!=TASK_SET for s in by_target.values()):
        raise ValueError('Qwen task triplets incomplete')
    if guard.get('guard_service_status')!='REVIEW_REQUIRED_NOT_RELEASED' or guard.get('production_approved') is not False:
        raise ValueError('Guard report did not block release')
    def metric(t):
        x=sorted(t)
        return {'n':len(x),'mean_ms':round(sum(x)/len(x),3), 'p95_ms':round(x[math.ceil(.95*len(x))-1],3)}
    return {'status':'QWEN_DEV12_RETRY_COMPLETE_REVIEW_ONLY', 'qwen_target_count':len(by_target),
            'qwen_task_count':len(rows),'qwen_task_e2e_ms':{k:metric(v) for k,v in sorted(by_task.items())},
            'guard_status':guard['guard_service_status'], 'full_chain_warm_e2e_ms':None,
            'prd_latency_pass':'NOT_ASSESSED','holdout_accessed':False, 'production_approved':False}


def write_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def package(root,output):
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.rglob('*')):
            if f.is_file():
                check_dev_path(f)
                z.write(f, f.relative_to(root.parent))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('diagnose','resume'),default='diagnose')
    p.add_argument('--timed-dir',type=Path,default=Path('/workspace/PRD32_1109_TIMED_V7_20261109_093701_294993'))
    p.add_argument('--v7-dir',type=Path,default=Path('/workspace/PRD32_1106_DEV12_RUNTIME_V7'))
    p.add_argument('--workspace',type=Path,default=Path('/workspace'))
    a=p.parse_args()
    for path in (a.timed_dir,a.v7_dir,a.workspace):check_dev_path(path)
    state=audit_producer(a.timed_dir)
    gpu=gpu_snapshot()
    print('PRODUCER_VERIFIED',state['producer_events'],'masks /',state['unique_image_forwards'],'DEV images')
    print('GPU_SNAPSHOT',json.dumps(gpu,ensure_ascii=False))
    if a.mode=='diagnose':
        print('NO_MODEL_CALLED; QWEN_WOULD_BE_ALLOWED:',gpu['allowed_for_qwen'])
        return 0
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output=a.workspace/('PRD32_1109_QWEN_RESUME_V2_'+stamp)
    output.mkdir(parents=True,exist_ok=False)
    archive=a.workspace/(output.name+'.zip')
    try:
        write_json(output/'verified_previous_stage.json',state)
        write_json(output/'gpu_snapshot_before.json',gpu)
        if not gpu['allowed_for_qwen']:
            raise RuntimeError('CUDA free VRAM still below 18GiB; no Qwen call performed')
        script=a.v7_dir/'run_qwen_from_v5_producer_v7.py'
        if not script.is_file(): raise FileNotFoundError('pinned original V7 Qwen entrypoint missing')
        for mode,name in [('preflight','QWEN_PREFLIGHT'),('live','QWEN_LIVE')]:
            target=output/name
            cmd=[sys.executable,str(script),'--mode',mode,'--producer-dir',str(a.timed_dir/'V5_PRODUCER'),'--out',str(target)]
            start=time.perf_counter_ns()
            completed=subprocess.run(cmd,cwd=a.v7_dir,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            wall_ms=(time.perf_counter_ns()-start)/1e6
            (output/(mode+'_stdout.txt')).write_text(completed.stdout,encoding='utf8')
            write_json(output/(mode+'_process_wall_ms.json'),{'process_wall_ms':round(wall_ms,3), 'exit_code':completed.returncode})
            if completed.returncode!=0:
                raise RuntimeError(mode+' exited '+str(completed.returncode)+': '+completed.stdout[-600:])
        result=summarize_qwen_output(output/'QWEN_LIVE')
        result.update({'qwen_live_process_wall_ms':round(float(json.loads((output/'live_process_wall_ms.json').read_text())['process_wall_ms']),3),
                       'same_batch_as_v5_stage':False,'staged_gpu_recovery_run':True})
        write_json(output/'qwen_resume_metrics.json',result)
        print('QWEN_RESUME_COMPLETE_REVIEW_ONLY',result['qwen_task_count'],'tasks')
    except Exception as exc:
        write_json(output/'failure.json',{'status':'RETRY_INCOMPLETE_NO_RELEASE', 'error_type':type(exc).__name__,
                   'message':str(exc), 'holdout_accessed':False,'production_approved':False})
        print('QWEN_RETRY_BLOCKED',type(exc).__name__,str(exc),file=sys.stderr)
        return_code=2
    else:
        return_code=0
    finally:
        package(output,archive)
        print('UPLOAD_THIS:',archive)
    return return_code

if __name__=='__main__':
    sys.exit(main())
