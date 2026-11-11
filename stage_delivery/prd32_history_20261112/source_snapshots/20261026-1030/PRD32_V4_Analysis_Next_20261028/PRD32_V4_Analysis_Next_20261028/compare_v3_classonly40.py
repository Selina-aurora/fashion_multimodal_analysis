#!/usr/bin/env python3
"""Pair V3 category-only 40-case run to V3 full tags and V4 posthoc human-reviewed cases.

Read-only; never uses historical source hints as independent ground truth. No GPU needed.
"""
import argparse, collections, csv, json, statistics
from pathlib import Path

VALID = {'top','pants','skirt','outerwear','dress','shoe','bag','accessory'}

def records(path):
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    ids=[x['case_id'] for x in rows]
    if len(set(ids))!=len(ids): raise ValueError(f'duplicate case_id in {path}')
    return {x['case_id']:x for x in rows}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--class40',default='/workspace/PRD32_V3_CLASS40_GPU_20261028_run01/predictions.jsonl')
    p.add_argument('--full40',default='/workspace/PRD32_V3_TAG40_GPU_20261028_run01/predictions.jsonl')
    p.add_argument('--v4-preds',default='/workspace/PRD32_V4_FULLTAGS19_GPU_20261028_run01/predictions.jsonl')
    p.add_argument('--v4-reviewed',default='/workspace/PRD32_V4_FULLTAGS19_GPU_20261028_run01/reviewed_pairs.jsonl')
    p.add_argument('--output',default='/workspace/PRD32_V3_CLASS40_ANALYSIS_20261028_run01')
    a=p.parse_args()
    cls,full,v4,review=[records(s) for s in (a.class40,a.full40,a.v4_preds,a.v4_reviewed)]
    if len(cls)!=40 or len(full)!=40 or set(cls)!=set(full): raise ValueError('Expected same 40 unique case IDs in class40 and full40')
    if len(v4)!=19 or len(review)!=18 or not set(review).issubset(v4):raise ValueError('Expected 19 V4 predictions and 18 reviewed cases')
    if not set(v4).issubset(cls):raise ValueError('V4 case IDs not in class40')
    for cid in cls:
        if not cls[cid].get('input_image_sha256') or cls[cid]['input_image_sha256']!=full[cid].get('input_image_sha256'):
            raise ValueError(f'image identity mismatch: {cid} class40 vs full40')
    for cid in v4:
        if cls[cid]['input_image_sha256'] != v4[cid].get('input_image_sha256'):
            raise ValueError(f'image identity mismatch: {cid} class40 vs v4')
    if any(x.get('scope')!='TRAIN_DEV_DIAGNOSTIC_ONLY' for x in cls.values()):raise ValueError('Unexpected classification source scope')
    if any(x.get('status')!='GENERATED_UNVERIFIED' for x in cls.values()):raise ValueError('Some category-only predictions did not complete; inspect failures')
    rows=[]
    for cid in sorted(review):
        r=review[cid]
        gt=r.get('reviewed_human_category_POSTHOC_ONLY')
        if gt not in VALID:raise ValueError(f'non-scorable human label for {cid}: {gt}')
        row={'case_id':cid,'review_set':r['review_set'],'human_category':gt,
             'v2':r['v2_category'],'v3_full_tags':full[cid]['category'],
             'v3_class_only':cls[cid]['category'],'v4_full_tags':v4[cid]['category'],
             'v2_correct':r['v2_category']==gt,'v3_full_correct':full[cid]['category']==gt,
             'v3_class_correct':cls[cid]['category']==gt,'v4_correct':v4[cid]['category']==gt,
             'classonly_time_ms':(cls[cid].get('timing') or {}).get('end_to_end_ms')}
        if row['v3_full_tags']!=r['v3_full_tags_category'] or row['v4_full_tags']!=r['v4_category']:
            raise ValueError(f'Previous run inconsistent with V4 reviewed pair for {cid}')
        rows.append(row)
    scored={m:sum(x[m] for x in rows) for m in ('v2_correct','v3_full_correct','v3_class_correct','v4_correct')}
    cm=collections.Counter(x['category'] for x in cls.values()); fm=collections.Counter(x['category'] for x in full.values())
    v3cls_latency=[x['timing']['end_to_end_ms'] for x in cls.values() if x.get('timing') and 'end_to_end_ms' in x['timing']]
    summary={'schema':'prd32_v3_classonly40_posthoc_comparison_v1','scope':'TRAIN_DEV_DIAGNOSTIC_ONLY_NOT_INDEPENDENT',
             'all40_count':40,'reviewed_scored':18,'ambiguous_excluded':['train_dev_020'],
             'scores_18':scored, 'v3_classonly_distribution_40':dict(cm),'v3_full_tags_distribution_40':dict(fm),
             'class_only_correct_new_top4':sum(x['v3_class_correct'] for x in rows if x['review_set']=='new_regression4'),
             'class_only_correct_historical14':sum(x['v3_class_correct'] for x in rows if x['review_set']=='historical14'),
             'class_only_median_latency_ms':round(statistics.median(v3cls_latency),2) if v3cls_latency else None,
             'class_only_normalized_schema_valid_40':sum(bool(x['schema_valid']) for x in cls.values()),
             'class_only_native_json_40':sum(bool(x['native_json']) for x in cls.values()),
             'posthoc_human_labels_sent_to_model':False,
             'warning':'Same train-source development cases, repeatedly inspected; no independent accuracy and no frozen V5 end-to-end metrics.'}
    out=Path(a.output)
    if out.exists():raise FileExistsError(f'Output already exists; choose a new run directory: {out}')
    out.mkdir(parents=True)
    with (out/'reviewed18_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('COMPARISON_OK',json.dumps(summary,ensure_ascii=False,indent=2))
    print('Please upload:',str(out/'summary.json'),'and',str(out/'reviewed18_comparison.csv'))

if __name__=='__main__': main()
