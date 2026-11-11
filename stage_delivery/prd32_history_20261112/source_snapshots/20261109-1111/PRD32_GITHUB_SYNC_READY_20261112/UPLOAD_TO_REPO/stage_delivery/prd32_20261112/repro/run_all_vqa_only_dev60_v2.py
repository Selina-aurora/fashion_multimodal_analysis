#!/usr/bin/env python3
"""One command for all 171 frozen DEV60 VQA-only SHADOW requests.

Eight isolated GPU processes, skip only fully validated ZIPs, and aggregate one
result ZIP. VQA-only has NO production Guard approval; all output is blocked.
"""
from __future__ import annotations
import argparse,collections,hashlib,io,json,os,re,statistics,subprocess,sys,zipfile
from datetime import datetime
from pathlib import Path
from joint_eval_stage2_dev60 import manifest_verified
from run_vqa_only_dev60_v2 import read_snap,choose,approx_p95
from vqa_route_contract import STATUS

ROOT=Path(__file__).resolve().parent
PREFIX='PRD32_VQA_ONLY_V2_CHUNK'
GOOD='VQA_ONLY_DEV60_V2_CHUNK_COMPLETE_SHADOW_BLOCKED'

def read_required(z,name):
    hits=[n for n in z.namelist() if n==name]
    if len(hits)!=1:raise ValueError('ZIP missing or duplicate '+name)
    return z.read(name)

def audit_zip(path,chunk,manifest,baseline):
    wanted={r['case_id']+'/'+r['instance_id'] for r in manifest['targets'] if r['chunk_id']==chunk}
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:raise ValueError('corrupt ZIP member')
        summary=json.loads(read_required(z,'VQA_ONLY_CHUNK_SUMMARY.json'))
        if (summary.get('status')!=GOOD or summary.get('chunk')!=chunk or
            summary.get('target_count')!=len(wanted) or summary.get('measured_vqa_calls')!=len(wanted) or
            summary.get('excluded_warmups')!=1 or summary.get('original_guard_invoked') is not False or
            summary.get('new_guard_contract_approved') is not False or
            summary.get('release_allowed') is not False or summary.get('independent_holdout_used') is not False):
            raise ValueError('wrong chunk, count, completion or Guard safety state')
        records=[json.loads(s) for s in read_required(z,'VQA_ONLY_PER_TARGET.jsonl').decode('utf8').splitlines() if s]
        if len(records)!=len(wanted):raise ValueError('wrong result record count')
        keys=[r['case_id']+'/'+r['instance_id'] for r in records]
        if set(keys)!=wanted or len(set(keys))!=len(keys):raise ValueError('incomplete or repeated DEV60 target')
        for r,k in zip(records,keys):
            old=baseline['targets'][k]
            if r['status']!=STATUS or r['original_guard_invoked'] is not False or r['production_release'] is not False or r['holdout_used'] is not False:
                raise ValueError('unsafe release/Guard claim')
            if r['frozen_mask_iou']<0.9999:raise ValueError('frozen mask mismatch')
            if r['vqa_matches_frozen_baseline'] != (r['vqa_response_sha256']==old['vqa_response_sha256']):
                raise ValueError('wrong reference comparison')
            if hashlib.sha256(r['vqa_answer'].encode('utf8')).hexdigest()!=r['vqa_response_sha256']:
                raise ValueError('response SHA does not match actual text')
        calls=[json.loads(s) for s in read_required(z,'alternative_qwen_calls.jsonl').decode('utf8').splitlines() if s]
        if len(calls)!=len(wanted)+1 or any(x.get('task')!='vqa' or x.get('publish_allowed') is not False for x in calls):
            raise ValueError('Qwen request scope not VQA-only')
        if sum(c.get('cohort','').startswith('measured_') for c in calls)!=len(wanted):
            raise ValueError('Qwen measured/warmup allocation mismatch')
        read_required(z,'preflight.json')
    return {'archive':str(path),'chunk':chunk,'target_count':len(records),'records':records,
      'archive_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}

def scan(workspace,manifest,baseline):
    selected={};rejected=[]
    for p in sorted(workspace.glob(PREFIX+'??_*.zip')):
        match=re.match(r'^'+PREFIX+r'(\d\d)_',p.name)
        if match is None:continue
        ch=int(match.group(1))
        if not 1<=ch<=8:continue
        try:rec=audit_zip(p,ch,manifest,baseline)
        except Exception as ex:rejected.append((p,str(ex)));continue
        if ch in selected:
            # prefer most recently generated independently verified archive
            if p.stat().st_mtime_ns>selected[ch]['path'].stat().st_mtime_ns:
                selected[ch]={'path':p,'audit':rec}
        else:selected[ch]={'path':p,'audit':rec}
    return selected,rejected

def child(cmd):
    print('RUN:', ' '.join(map(str,cmd)),flush=True)
    proc=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:print(line,end='',flush=True)
    except KeyboardInterrupt:
        proc.terminate();proc.wait(timeout=20);raise
    code=proc.wait()
    if code:raise RuntimeError(f'GPU child returned {code}; already completed chunks remain reusable')

def aggregate(selected,manifest,baseline,workspace):
    entries=[];records=[]
    for i in range(1,9):
        a=selected[i]['audit'];entries.append({k:a[k] for k in ['archive','chunk','target_count','archive_sha256']});records.extend(a['records'])
    if len(records)!=171 or len({r['case_id']+'/'+r['instance_id'] for r in records})!=171:
        raise ValueError('missing or duplicated full 171 DEV60 coverage')
    all_times=[r['vqa_only_total_ms'] for r in records]
    original=[r['historical_full_three_task_ms'] for r in records]
    host=[r['vqa_host_task_ms'] for r in records]
    summary={'schema':'PRD32_VQA_ONLY_DEV60_V2_ALL171','status':'SHADOW_ONLY_COMPLETE_171',
        'target_count':171,'source_images':60,'chunk_count':8,'measured_qwen_calls':171,
        'excluded_warmup_calls':8,'mean_vqa_only_e2e_ms':round(statistics.mean(all_times),3),
        'p95_vqa_only_e2e_ms':approx_p95(all_times),
        'mean_vqa_host_task_ms':round(statistics.mean(host),3),
        'mean_historical_three_task_e2e_ms':round(statistics.mean(original),3),
        'matched_baseline_vqa_responses':sum(r['vqa_matches_frozen_baseline'] for r in records),
        'vqa_only_e2e_under_200ms':sum(x<=200 for x in all_times),
        'vqa_only_e2e_under_400ms':sum(x<=400 for x in all_times),
        'note':'Times compare separate warm runs, not same-sitting paired A/B. VQA is fixed color question. No production Guard.',
        'independent_gt':False,'original_guard_invoked':False,'new_guard_contract_approved':False,
        'publish_allowed':False,'holdout_used':False,'prd_acceptance':False,'chunks':entries}
    ts=datetime.now().strftime('%Y%m%d_%H%M%S')
    out=workspace/f'PRD32_VQA_ONLY_DEV60_V2_ALL171_UPLOAD_THIS_{ts}_{os.getpid()}.zip'
    with zipfile.ZipFile(out,'w',allowZip64=True) as z:
        z.writestr('README_RESULT.txt','DEV60 VQA-only shadow, all 171 frozen predicted instances. No production release. Original three-task Guard NOT adapted. Not independent GT or PRD official acceptance.\n')
        z.writestr('ALL171_SUMMARY.json',json.dumps(summary,ensure_ascii=False,indent=2),compress_type=zipfile.ZIP_DEFLATED)
        z.writestr('ALL171_VQA_ONLY_PER_TARGET.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),compress_type=zipfile.ZIP_DEFLATED)
        for i in range(1,9):
            p=selected[i]['path'];z.write(p,'CHUNK_ZIPS/'+p.name,compress_type=zipfile.ZIP_STORED)
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:raise ValueError('final ZIP corrupted')
    return out

def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--dry-run',action='store_true')
    pa.add_argument('--execute-models',action='store_true')
    pa.add_argument('--workspace',type=Path,default=Path('/workspace'))
    a=pa.parse_args()
    workspace=a.workspace
    manifest,_=manifest_verified()
    baseline=read_snap(manifest)
    if not workspace.is_dir():raise SystemExit('workspace missing '+str(workspace))
    prior,invalid=scan(workspace,manifest,baseline)
    print('COMPLETED_VERIFIED_CHUNKS',sorted(prior),flush=True)
    print('PENDING_CHUNKS',[i for i in range(1,9) if i not in prior],flush=True)
    for p,reason in invalid:print('IGNORED_INVALID',p,reason,flush=True)
    if a.dry_run:
        print('DRY_RUN_ONLY_NO_GPU');return 0
    if not a.execute_models:pa.error('use --execute-models to perform live GPU tests')
    for ch in range(1,9):
        if ch in prior:
            print('SKIP_VERIFIED_CHUNK',ch,prior[ch]['path'],flush=True);continue
        if not any(i not in prior for i in range(1,9)):break
        out=workspace/f'{PREFIX}{ch:02d}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}'
        child([sys.executable,'-u',str(ROOT/'run_vqa_only_dev60_v2.py'),
            '--mode','preflight','--chunk',str(ch),'--out',str(out)+'_preflight'])
        child([sys.executable,'-u',str(ROOT/'run_vqa_only_dev60_v2.py'),
            '--mode','live','--chunk',str(ch),'--execute-models','--out',str(out)])
        zip_path=out.with_suffix('.zip')
        prior[ch]={'path':zip_path,'audit':audit_zip(zip_path,ch,manifest,baseline)}
        print('CHUNK_VERIFIED',ch,flush=True)
    result=aggregate(prior,manifest,baseline,workspace)
    print('VQA_ONLY_V2_ALL171_COMPLETE_VERIFIED_SHADOW_ONLY',flush=True)
    print('UPLOAD_THIS:',result,flush=True)
    return 0
if __name__=='__main__':sys.exit(main())
