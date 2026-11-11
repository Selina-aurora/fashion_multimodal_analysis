#!/usr/bin/env python3
"""Read-only merge of eight completed Stage2 result ZIPs. Refuses partial/duplicate/missing targets."""
import argparse,csv,io,json,zipfile,statistics,sys
from pathlib import Path
from joint_eval_stage2_dev60 import manifest_verified,MANIFEST_SHA256,stats

def inside(z,name):
    matching=[n for n in z.namelist() if n.endswith('/'+name)]
    if len(matching)!=1:raise ValueError('missing/duplicate file: '+name)
    return z.read(matching[0])

def audit_zips(paths):
    manifest,_=manifest_verified()
    expected={(r['case_id'],r['instance_id']):r for r in manifest['targets']}
    seen={};grouped={};total_calls=0
    for p in paths:
        with zipfile.ZipFile(p) as z:
            if z.testzip() is not None:raise ValueError('corrupt zip '+str(p))
            summ=json.loads(inside(z,'summary.json'))
            if summ['status']!='PRD31_PRD32_STAGE2_CHUNK_COMPLETE_SHADOW_ONLY' or summ['manifest_sha256']!=MANIFEST_SHA256:
                raise ValueError('wrong chunk status or manifest '+str(p))
            ch=summ['chunk']
            if ch in grouped:raise ValueError('duplicate chunk '+str(ch))
            records=list(csv.DictReader(io.StringIO(inside(z,'per_request_timing.csv').decode('utf-8-sig'))))
            calls=[json.loads(x) for x in inside(z,'qwen_calls.jsonl').decode().splitlines() if x.strip()]
            if len(calls)!=3*(len(records)+1) or summ['measured_qwen_calls']!=3*len(records):
                raise ValueError('Qwen call count mismatch chunk '+str(ch))
            if len(set(c['invocation_uid'] for c in calls))!=len(calls):raise ValueError('duplicate Qwen invocation')
            for r in records:
                key=(r['case_id'],r['instance_id'])
                if key not in expected or expected[key]['chunk_id']!=ch or key in seen:
                    raise ValueError('unexpected/duplicate target '+str(key))
                if r['guard_status']!='REVIEW_REQUIRED_NOT_RELEASED' or r['publish_allowed'] not in {'False','false'} or r['holdout_accessed'] not in {'False','false'}:
                    raise ValueError('Guard unsafe or holdout flag true')
                if float(r['frozen_mask_iou'])<0.9999:raise ValueError('frozen mask changed')
                seen[key]=r
            if len(records)!=summ['target_count']:
                raise ValueError('measured target count mismatch')
            grouped[ch]=summ;total_calls+=len(calls)
    if len(grouped)!=8 or set(grouped)!=set(range(1,9)):
        raise ValueError('missing chunks: '+str(sorted(set(range(1,9))-set(grouped))))
    if set(seen)!=set(expected):raise ValueError('not all 171 target keys present')
    if len(seen)!=171 or total_calls!=3*(171+8):raise ValueError('total count or warmup count mismatch')
    return manifest,seen,grouped,total_calls

def main():
    p=argparse.ArgumentParser()
    p.add_argument('zips',type=Path,nargs=8)
    p.add_argument('--out',type=Path,default=Path('DEV60_STAGE2_171_COMBINED_SUMMARY.json'))
    a=p.parse_args();m,rows,groups,calls=audit_zips(a.zips)
    names=['total_host_wall_ms','v5_host_wall_ms','tags_task_host_wall_ms','vqa_task_host_wall_ms','copy_task_host_wall_ms','guard_host_wall_ms']
    out={'status':'ALL_8_DEV60_CHUNKS_COMPLETED_DEVELOPMENT_BASELINE_NOT_FORMAL_ACCEPTANCE',
         'manifest_sha256':MANIFEST_SHA256,'unique_predicted_instances':171,'source_images':60,
         'unique_newly_tested_before_stage2':135,'measured_qwen_calls':513,'total_qwen_calls_including_chunk_warmups':calls,
         'timings':{n:stats([float(r[n]) for r in rows.values()]) for n in names},
         'guard_blocked':171,'released':0,'holdout_used':False,
         'caveats':['171 are V5 predicted instances, not true-object GT','not PRD quality acceptance or full 3.1.2/3.1.3 integration','same baseline prompt, marked full image; V5 mask is checked, not supplied as Qwen pixels']}
    if a.out.exists():raise FileExistsError('refuse overwrite')
    a.out.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('STAGE2_ALL_171_AUDIT_PASS',str(a.out));print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
