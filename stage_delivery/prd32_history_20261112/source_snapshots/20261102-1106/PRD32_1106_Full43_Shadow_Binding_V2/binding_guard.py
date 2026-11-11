#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Non-intrusive instance-identity and class-arbitration audit for frozen DEV60.
Never overrides V5 categories and never reassigns Qwen text to another instance.
Can be imported into a future experiment after local code review (NOT integrated yet).
"""
import csv, json, math, argparse
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parent
TASKS=('tags','vqa','copy')

class BindingIntegrityError(ValueError):pass

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def check_binding(model_instance, image_record, task_record, tol=1e-3):
    """Validate logged metadata only. It CANNOT establish which garment model actually described."""
    problems=[]
    case=image_record['case_id'];iid=model_instance['instance_id']
    if task_record.get('case_id')!=case:problems.append('CASE_ID_MISMATCH')
    if task_record.get('target_instance_id')!=iid:problems.append('INSTANCE_ID_MISMATCH')
    if task_record.get('image_sha256')!=image_record['image_sha256']:problems.append('IMAGE_HASH_MISMATCH')
    if task_record.get('v5_category_unverified')!=model_instance['predicted_category']:
        problems.append('V5_CATEGORY_METADATA_MISMATCH')
    src=model_instance['bbox_xyxy'];got=task_record.get('v5_predicted_bbox_xyxy')
    try:
        if (not isinstance(got,list) or len(src)!=4 or len(got)!=4 or
            any(not math.isfinite(float(x)) for x in src+got) or
            any(abs(float(a)-float(b))>tol for a,b in zip(src,got))):
            problems.append('BBOX_METADATA_MISMATCH')
    except (TypeError,ValueError):problems.append('BBOX_METADATA_MISMATCH')
    if task_record.get('task') not in TASKS:problems.append('UNEXPECTED_TASK')
    if task_record.get('human_label_or_gt_bbox_used') is not False:
        problems.append('NOT_FROZEN_V5_PREDICTION')
    return problems

def choose_category_state(v5_category, tags_category):
    """Conflict states, not ground-truth correctness or automatic corrections."""
    if not tags_category:return 'TAG_CATEGORY_MISSING'
    if v5_category==tags_category:return 'AGREE_NOT_PROOF_OF_TRUTH'
    return 'CONFLICT_REVIEW_REQUIRED_KEEP_V5_UNVERIFIED'

def make_report(root=ROOT, output=None):
    inp=root/'input';out=(output or root/'output').resolve();out.mkdir(parents=True,exist_ok=True)
    if out==root.resolve() or inp.resolve() in out.parents or out==inp.resolve():raise ValueError('unsafe output path')
    imgs=read_jsonl(inp/'v5_raw_predictions.jsonl'); tasks=read_jsonl(inp/'focus_v1_qwen_results_513.jsonl')
    assert len(imgs)==60 and len(tasks)==513,'frozen DEV60 cardinality mismatch'
    with open(inp/'nov2_v5_attribution_171.csv',encoding='utf-8-sig',newline='') as f:
        hist={(r['case_id'],r['instance_id']):r for r in csv.DictReader(f)}
    tk=defaultdict(list)
    for rec in tasks:tk[(rec['case_id'],rec['target_instance_id'])].append(rec)
    rows=[];allproblems=[]
    for im in imgs:
        for pi in im['instances']:
            cid=im['case_id'];iid=pi['instance_id']; q=sorted(tk[(cid,iid)],key=lambda x:x['task'])
            if len(q)!=3 or Counter(x['task'] for x in q)!=Counter(TASKS):
                raise BindingIntegrityError(f'incomplete 3 tasks: {cid}/{iid}')
            problems={r['task']:check_binding(pi,im,r) for r in q}
            for task, pr in problems.items():
                if pr:allproblems.append({'case_id':cid,'instance_id':iid,'task':task,'errors':'|'.join(pr)})
            tag=next(r for r in q if r['task']=='tags')
            parsed=tag.get('tag_parse') or {}
            if isinstance(parsed,dict):
                tc=(parsed.get('canonical_service_json') or {}).get('category') or (parsed.get('raw_parsed') or {}).get('category') or parsed.get('category')
            else:tc=None
            # Fallback is explicitly marked, never treats raw text as verified class.
            v5=pi['predicted_category']; quality=hist.get((cid,iid),{})
            state=choose_category_state(v5,tc)
            rows.append({'case_id':cid,'instance_id':iid,'v5_category_unverified':v5,
                'qwen_tags_parsed_category':tc or '', 'class_relation':state,
                'historical_v5_category_review':quality.get('v5_category_quality',''),
                'historical_v5_bbox_review':quality.get('bbox_quality',''),
                'metadata_integrity':'PASS' if all(not x for x in problems.values()) else 'FAIL',
                'metadata_failures':'|'.join(f'{k}:{"/".join(v)}' for k,v in problems.items() if v),
                'identity_perception_verified':'NO','proposed_class_overwrite':'NEVER'})
    assert len(rows)==171
    def csvwrite(path, r, hdr):
        with open(path,'w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=hdr);w.writeheader();w.writerows(r)
    csvwrite(out/'binding_class_diagnostic_171.csv',rows,list(rows[0]))
    flagged=[r for r in rows if r['historical_v5_category_review']=='CORRECT' and r['class_relation'].startswith('CONFLICT')]
    csvwrite(out/'priority_v5_correct_tags_disagreement.csv',flagged,list(rows[0]))
    csvwrite(out/'metadata_integrity_failures.csv',allproblems,['case_id','instance_id','task','errors'])
    stats={'expected_records':513,'validated_instances':171,'integrity_error_tasks':len(allproblems),
           'v5_tags_class_disagreement':sum(x['class_relation'].startswith('CONFLICT') for x in rows),
           'v5_historically_correct_but_tags_disagree':len(flagged),
           'scope':'DEV60_ONLY_METADATA_AND_CLASS_CONFLICT_DIAGNOSTIC',
           'warning':'Class mismatch is NOT an automatically validated classification error; no class replacement performed.'}
    (out/'binding_guard_summary.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding='utf-8')
    return stats

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output-dir',type=Path,default=ROOT/'output')
    print(json.dumps(make_report(output=a.parse_args().output_dir),ensure_ascii=False,indent=2))
