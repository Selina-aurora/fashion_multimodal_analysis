#!/usr/bin/env python3
"""Combine 23 preserved human reviews with the 17 new ones; score reviewed development cohort only."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
VALID={'top','pants','skirt','outerwear','dress','shoe','bag','accessory'}

def jsonl(path):
    if not path: return None
    rows=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows)!=40 or len({x['case_id'] for x in rows})!=40:raise ValueError('Expected 40 unique class-only rows')
    return {x['case_id']:x for x in rows}

def score(newpath,out,classonly=None):
    registry=json.loads((ROOT/'all40_case_registry.json').read_text(encoding='utf-8'))
    orig19=json.loads((ROOT/'previous_human_review_19.json').read_text(encoding='utf-8'))['rows']
    added4=json.loads((ROOT/'previous_human_review_4.json').read_text(encoding='utf-8'))['rows']
    new=json.loads(Path(newpath).read_text(encoding='utf-8'))
    if new.get('schema')!='prd32_20261028_new17_blind_visual_review_v1':raise ValueError('Incorrect exported review schema')
    new17=new['rows']; expected={x['case_id'] for x in registry if x['priority']=='P2_NOT_HUMAN_REVIEWED'}
    if len(new17)!=17 or {x['case_id'] for x in new17}!=expected:raise ValueError('Missing/duplicate/out-of-scope new17 case')
    if not all(x.get('human_category') and x.get('bbox_coverage') and x.get('visually_identifiable') and x.get('other_garments_visible') for x in new17):raise ValueError('Incomplete review fields')
    prior={x['case_id']:x for x in orig19}; prior.update({x['case_id']:x for x in added4})
    if len(prior)!=23:raise ValueError('Historical review collision')
    for x in new17: prior[x['case_id']]=x
    if len(prior)!=40:raise ValueError('Not all 40 cases reviewed')
    c_only=jsonl(classonly)
    v3full={x['case_id']:x for x in [json.loads(s) for s in (ROOT/'v3_full_tags_40_predictions.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]}
    rows=[]
    for c in registry:
        cid=c['case_id'];h=prior[cid];human=h.get('human_category','uncertain')
        reasons=[]
        if human not in VALID:reasons.append('UNCERTAIN_CATEGORY')
        if cid=='train_dev_020':reasons.append('REVIEW_CONFLICT_VISUAL_AMBIGUITY')
        if cid=='train_dev_019':reasons.append('INCOMPLETE_HISTORICAL_BBOX')
        if h.get('bbox_coverage')=='NO':reasons.append('BBOX_COVERAGE_NO')
        if h.get('visually_identifiable')=='NO':reasons.append('NOT_VISUALLY_IDENTIFIABLE')
        if h.get('visually_identifiable')=='UNCERTAIN':reasons.append('VISIBILITY_UNCERTAIN')
        if cid in v3full and c_only:
            before=v3full[cid].get('input_image_sha256');after=c_only[cid].get('input_image_sha256')
            if before and after and before!=after:raise ValueError('Image SHA mismatch '+cid)
        predclass=(c_only[cid].get('category') if c_only else None)
        good=not reasons
        rows.append(dict(case_id=cid,human_category=human,review_quality='SCOREABLE' if good else 'EXCLUDE',review_exclusion=';'.join(reasons),source_hint_NOT_GT=c['source_hint_NOT_GT'],v2=c['v2_category'],v3_full_tags=c['v3_category'],v3_class_only=predclass,original_review_notes=h.get('notes','')))
    validrows=[x for x in rows if x['review_quality']=='SCOREABLE']
    metrics={'schema':'prd32_20261028_full40_human_review_dev_eval_v1','scope':'TRAIN_SOURCE_DEV_ONLY_NO_INDEPENDENT_GROUND_TRUTH','total':40,'human_reviewed':40,
             'scoreable':len(validrows),'excluded':len(rows)-len(validrows),'excluded_cases':[{'case_id':x['case_id'],'reason':x['review_exclusion']} for x in rows if x['review_quality']!='SCOREABLE']}
    for col in ['v2','v3_full_tags']+(['v3_class_only'] if c_only else []):
        metrics[col]={'matched':sum(x[col]==x['human_category'] for x in validrows),'denominator':len(validrows)}
    metrics['limitations']=['40 train-origin development images; not independent test','historical GT-pilot bbox, not frozen V5 predicted bbox','repeatedly analyzed subset; no formal PRD accuracy','confidence / class agreement are not calibrated']
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'comparison40.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'status':'DEV_40_HUMAN_REVIEW_EVALUATION_DONE','output_dir':str(out),'scoreable':len(validrows),'scores':{k:v for k,v in metrics.items() if k in ['v2','v3_full_tags','v3_class_only']}},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review-json',required=True);p.add_argument('--output',required=True);p.add_argument('--classonly-predictions',default=None);a=p.parse_args()
    score(a.review_json,a.output,a.classonly_predictions)
