#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only V4 development experiment audit; no GPU or 3.1 changes.

Checks output identities, field/subtype coherence, copy text, timing, and review
coverage. Copies selected artifacts into a compact handoff ZIP; it NEVER claims
visual verification or official PRD acceptance. Standard-library only.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import zipfile
from pathlib import Path

CLASSES={'top','pants','skirt','outerwear','dress','shoe','bag','accessory'}
PATTERNS={'纯色','条纹','格纹','人字纹','波点','花卉印花','文字印花','字母印花','几何印花','动物纹','拼色'}
SUBTYPE_CLASS={
 '背带裙':'dress','吊带裙':'dress','连衣裙':'dress',
 '半身裙':'skirt','T恤':'top','衬衫':'top','卫衣':'top','针织衫':'top',
 '夹克':'outerwear','西装外套':'outerwear','大衣':'outerwear',
 '长裤':'pants','短裤':'pants',
 '运动鞋':'shoe','皮鞋':'shoe','背包':'bag','手提包':'bag','配饰':'accessory'
}
DEBUG_TERMS=('红框','红色框','框内','标注框','模特')
UNSUPPORTED=('100%','纯棉','真丝','产地','正品','保证','显瘦','品质保证')
EXPECTED={'copy_01','copy_02','copy_03'}
ARTIFACT_FILES=('summary.json','candidate_drafts.jsonl','COPY_V4_compare_review.html',
                'human_review_blank.csv','prompt_visible_facts.txt','source_run_config.json')


def read_jsonl(path):
    result=[]
    with open(path,encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            if not line.strip():continue
            try:val=json.loads(line,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            except (ValueError,TypeError) as e: raise ValueError(f'Invalid JSONL line {n}: {e}') from e
            if not isinstance(val,dict):raise ValueError(f'JSONL line {n} is not an object')
            result.append(val)
    return result


def p95(values):
    v=sorted(values)
    if not v:return None
    if len(v)==1:return v[0]
    p=.95*(len(v)-1);lo=math.floor(p);hi=math.ceil(p)
    return v[lo]*(hi-p)+v[hi]*(p-lo)


def check_case(r):
    errors=[];flags=[]
    cid=r.get('case_id');draft=r.get('draft');facts=r.get('facts');status=r.get('status','')
    if r.get('task')!='copy':errors.append('TASK_NOT_COPY')
    if not isinstance(cid,str) or not cid:errors.append('INVALID_CASE_ID')
    if not isinstance(status,str):errors.append('STATUS_INVALID')
    if not isinstance(facts,dict):
        if draft is not None:errors.append('DRAFT_WITHOUT_FACTS')
        flags.append('NO_PARSED_VISUAL_FACTS')
    else:
        cat=facts.get('category');sub=facts.get('subtype');color=facts.get('primary_color');pat=facts.get('pattern')
        if cat is not None and (not isinstance(cat,str) or cat not in CLASSES):errors.append('INVALID_CATEGORY')
        if sub is not None and (not isinstance(sub,str) or sub not in SUBTYPE_CLASS):errors.append('INVALID_SUBTYPE')
        if pat is not None and (not isinstance(pat,str) or pat not in PATTERNS):errors.append('INVALID_PATTERN')
        if isinstance(sub,str) and sub in SUBTYPE_CLASS and cat is not None and SUBTYPE_CLASS[sub]!=cat:
            errors.append('CATEGORY_SUBTYPE_CONFLICT')
        if not isinstance(facts.get('visible_structures'),list):errors.append('STRUCTURES_NOT_LIST')
        if not isinstance(facts.get('distinct_decorations'),list):errors.append('DECORATIONS_NOT_LIST')
        if not isinstance(facts.get('uncertainty'),list):errors.append('UNCERTAINTY_NOT_LIST')
        if isinstance(facts.get('distinct_decorations'),list) and any(v in ('口袋','插袋','侧袋') for v in facts['distinct_decorations'] if isinstance(v,str)):
            flags.append('NORMAL_POCKET_NOT_AUTOMATIC_DECORATION')
        if pat=='纯色':flags.append('PURE_COLOR_VISUAL_CONFIRMATION_REQUIRED')
        if draft is not None and isinstance(draft,str):
            if isinstance(color,str) and color and color not in draft:errors.append('DRAFT_COLOR_MISMATCH_FACTS')
            if isinstance(sub,str) and sub and sub not in draft:errors.append('DRAFT_SUBTYPE_MISMATCH_FACTS')
            if isinstance(pat,str) and pat not in ('纯色',) and pat not in draft:flags.append('PATTERN_NOT_EXPLICIT_IN_DRAFT')
    if draft is None:
        flags.append('NO_COPY_DRAFT_HUMAN_REVIEW_REQUIRED')
    elif not isinstance(draft,str) or not draft.strip():errors.append('INVALID_DRAFT')
    else:
        if len(draft)>80:errors.append('EXCEEDS_80_CHINESE_CHARACTER_LIMIT')
        if any(x in draft for x in DEBUG_TERMS):errors.append('DEBUG_TARGET_LANGUAGE')
        if any(x in draft for x in UNSUPPORTED):flags.append('POSSIBLE_UNSUPPORTED_MARKETING_OR_MATERIAL_CLAIM')
        flags.append('VISUAL_FACTS_NOT_HUMAN_GROUND_TRUTH')
    native=False
    raw=r.get('raw_model_output')
    if isinstance(raw,str):
        try:native=isinstance(json.loads(raw),dict)
        except (ValueError,TypeError):pass
    t=r.get('latency_ms');timing=r.get('timing') or {}
    if not (type(t) in (int,float) and math.isfinite(t) and t>=0):flags.append('MISSING_VALID_LATENCY')
    return {'case_id':cid,'image_id':r.get('image_id'),'target_instance_id':r.get('target_instance_id'),
            'status':status,'category':facts.get('category') if isinstance(facts,dict) else None,
            'color':facts.get('primary_color') if isinstance(facts,dict) else None,
            'pattern':facts.get('pattern') if isinstance(facts,dict) else None,
            'subtype':facts.get('subtype') if isinstance(facts,dict) else None,
            'draft':draft,'char_count':len(draft) if isinstance(draft,str) else None,
            'latency_ms':t,'generated_tokens':timing.get('generated_tokens'),
            'preprocess_ms':timing.get('preprocess_ms'),'generation_ms':timing.get('generation_ms'),
            'model_native_json_object':native,'hard_errors':errors,'review_flags':flags,
            'original_issues':r.get('issues') or []}


def run_audit(run_dir,review=None,old_review=None):
    source=run_dir/'candidate_drafts.jsonl'
    if not source.is_file():raise FileNotFoundError(f'Missing {source}; V4 summary alone cannot evaluate drafts')
    rows=read_jsonl(source)
    if not rows:raise ValueError('Empty candidate_drafts.jsonl')
    ids=[r.get('case_id') for r in rows]
    if any(not isinstance(x,str) for x in ids):raise ValueError('V4 case IDs must be strings')
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate V4 case IDs')
    checks=[check_case(r) for r in rows]
    summary=json.loads((run_dir/'summary.json').read_text(encoding='utf-8'))
    declared=summary.get('source_cases')
    if declared!=len(rows):raise ValueError(f'V4 source cases mismatch: summary={declared} rows={len(rows)}')
    if summary.get('candidate_count')!=sum(x['draft'] is not None for x in rows):
        raise ValueError('V4 candidate count mismatch')
    if summary.get('inference_performed')!=sum(x.get('inference_performed') is True for x in rows):
        raise ValueError('V4 inference count mismatch')
    latency=[x['latency_ms'] for x in checks if type(x['latency_ms']) in (int,float) and math.isfinite(x['latency_ms'])]
    baseline={}
    if old_review:
        o=json.loads(old_review.read_text(encoding='utf-8'))
        baseline={x['case_id']:x['verdict'] for x in o.get('results',[]) if x.get('task')=='copy'}
    scored={}
    if review:
        o=json.loads(review.read_text(encoding='utf-8'))
        if o.get('schema')!='prd32_copy_v4_review_dev_v1':raise ValueError('V4 review JSON schema mismatch')
        labels=[x['case_id'] for x in o.get('results',[])]
        if len(labels)!=len(set(labels)):raise ValueError('Duplicate V4 reviewer IDs')
        scored={x['case_id']:x.get('verdict','PENDING') for x in o.get('results',[])}
        for key,val in scored.items():
            if val not in ('PENDING','SATISFIED','NOT_SATISFIED','BLOCKED_REVIEW'):raise ValueError(f'Unknown verdict {val}')
        if any(v=='SATISFIED' and not any(x['case_id']==key and x['draft'] is not None for x in checks) for key,v in scored.items()):
            raise ValueError('No-draft case marked SATISFIED')
    for x in checks:
        x['v2_human_verdict']=baseline.get(x['case_id'])
        x['v4_human_verdict']=scored.get(x['case_id'],'PENDING')
    matched=[x for x in checks if x['case_id'] in baseline]
    valid_reviews=[x for x in checks if x['v4_human_verdict']!='PENDING']
    report={
      'schema':'prd32_copy_v4_postrun_audit_v1',
      'scope':'same_three_development_images_not_independent_validation',
      'source_run_directory':str(run_dir),'case_count':len(rows),
      'case_ids':ids,'expected_development_ids_match':set(ids)==EXPECTED,
      'generated_drafts':sum(x['draft'] is not None for x in checks),
      'raw_model_native_json_object_count':sum(x['model_native_json_object'] for x in checks),
      'hard_error_case_ids':[x['case_id'] for x in checks if x['hard_errors']],
      'error_counts':dict(collections.Counter(e for x in checks for e in x['hard_errors'])),
      'review_flag_counts':dict(collections.Counter(f for x in checks for f in x['review_flags'])),
      'latency':{'n':len(latency),'median_ms':round(statistics.median(latency),2) if latency else None,
                 'p95_ms_descriptive_n_too_small':round(p95(latency),2) if latency else None,
                 'prd_service_p95_measured':False},
      'v2_manual_review':{'matched_cases':len(matched),'satisfied_count':sum(x['v2_human_verdict']=='SATISFIED' for x in matched)},
      'v4_manual_review':{'reviewed':len(valid_reviews),'total':len(rows),'satisfied':sum(x['v4_human_verdict']=='SATISFIED' for x in checks),
                          'complete':len(valid_reviews)==len(rows)},
      'cannot_claim_improvement_without_v4_human_review':len(valid_reviews)!=len(rows),
      'original_summary_status':summary.get('status'),
      'no_holdout_used':True,'official_prd_acceptance':False,
      'caution':'V4 prompt was tuned using these development cases; same-case improvement is not independent generalization. '
                'Facts are unverified model outputs, not authoritative GT.'}
    return checks,report


def emit(out,rows,report):
    out.mkdir(parents=True,exist_ok=False)
    (out/'v4_postrun_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    cols=('case_id','image_id','target_instance_id','status','category','color','pattern','subtype','draft',
          'char_count','latency_ms','generated_tokens','preprocess_ms','generation_ms','model_native_json_object',
          'hard_errors','review_flags','v2_human_verdict','v4_human_verdict')
    with (out/'v4_per_case_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in rows:
            flat={k:r.get(k) for k in cols}
            for k in ('hard_errors','review_flags'):flat[k]='; '.join(flat[k])
            w.writerow(flat)
    (out/'READ_ME.txt').write_text(
        'V4 DEVELOPMENT AUDIT. No GPU inference. Original experiment files unchanged.\n'
        '1) Open v4_per_case_audit.csv to inspect risky fields and timing.\n'
        '2) Open original COPY_V4_compare_review.html and review each visual claim.\n'
        '3) Export PRD32_COPY_V4_human_review.json from that page and rerun with --review.\n'
        '4) Do not treat model facts as GT or these three cases as PRD acceptance.\n',encoding='utf-8')


def bundle(zip_path,run_dir,out,review=None):
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=8) as z:
        for name in ARTIFACT_FILES:
            p=run_dir/name
            if p.is_file(): z.write(p,'V4_SOURCE/'+name)
        for p in sorted(out.iterdir()):
            if p.is_file():z.write(p,'V4_AUDIT/'+p.name)
        if review and review.is_file():z.write(review,'V4_REVIEW/'+review.name)
    with zipfile.ZipFile(zip_path) as z:
        bad=z.testzip()
        if bad:raise IOError(f'ZIP integrity failure {bad}')
    return zip_path


def self_test():
    good={'case_id':'copy_01','task':'copy','status':'DRAFT_REVIEW_REQUIRED','image_id':'i','target_instance_id':'p',
          'facts':{'category':'dress','primary_color':'黑色','pattern':None,'subtype':'背带裙',
                   'visible_structures':['背带'],'distinct_decorations':[],'uncertainty':['pattern']},
          'draft':'黑色背带裙，带有背带设计。','latency_ms':1987.46,
          'raw_model_output':'{"category":"dress"}','timing':{'generated_tokens':80}}
    result=check_case(good)
    assert not result['hard_errors'],result
    assert result['model_native_json_object'] is True
    bad=json.loads(json.dumps(good));bad['facts']['category']='top'
    assert 'CATEGORY_SUBTYPE_CONFLICT' in check_case(bad)['hard_errors']
    bad['draft']=None
    assert 'NO_COPY_DRAFT_HUMAN_REVIEW_REQUIRED' in check_case(bad)['review_flags']
    assert round(p95([1,2,3]),2)==2.9
    print('POSTRUN_AUDIT_SELF_TEST_PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--old-review',type=Path,default=None)
    p.add_argument('--review',type=Path,default=None)
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--zip',type=Path,default=None)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:return self_test()
    if not args.run_dir:p.error('--run-dir is required')
    if not args.run_dir.is_dir():p.error('V4 run directory not found')
    if any('holdout' in str(x).lower() or 'sealed' in str(x).lower() for x in (args.run_dir,args.review,args.output,args.zip) if x):
        p.error('Do not audit sealed holdout artifacts')
    out=args.output or args.run_dir.parent/(args.run_dir.name+'_POSTRUN_AUDIT')
    if out.exists():p.error(f'Output exists; choose a new --output: {out}')
    checks,report=run_audit(args.run_dir,args.review,args.old_review)
    emit(out,checks,report)
    dest=args.zip or out.parent/(out.name+'.zip')
    if dest.exists():p.error(f'ZIP exists; choose a new --zip: {dest}')
    bundle(dest,args.run_dir,out,args.review)
    print(json.dumps({'audit':'POSTRUN_AUDIT_COMPLETE_NOT_PRD_ACCEPTED',
                       'cases':report['case_count'],'human_v4_review_complete':report['v4_manual_review']['complete'],
                       'median_latency_ms':report['latency']['median_ms'],
                       'hard_error_case_ids':report['hard_error_case_ids'],
                       'report_dir':str(out),'handoff_zip':str(dest)},ensure_ascii=False,indent=2))

if __name__=='__main__':
    try:main()
    except (OSError,ValueError) as e:
        print('AUDIT_INPUT_ERROR:',e,file=sys.stderr);sys.exit(2)
