#!/usr/bin/env python3
"""Validate self-reported image-level human review with frozen SHA keys. NO model, GT boxes, or holdout read."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

CATS = ('top','pants','skirt','outerwear','dress','shoe','bag','accessory')
SCENES = ('多实例','遮挡','模糊低清','目标重叠','裙裤易混淆','复杂背景')
DECISIONS=('ACCEPT','REJECT','UNCERTAIN','UNREVIEWED')

def expected_from_audit(audit_path):
    with open(audit_path,encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    expected={}
    for r in rows:
        case_id=r['case_id']
        if case_id in expected:raise ValueError('DUPLICATE_CASE_IN_SELECTION_AUDIT:'+case_id)
        expected[case_id]=r['image_sha256']
    if len(expected)!=48 or set(expected)!={'fresh_dev_'+str(i).zfill(3) for i in range(13,61)}:
        raise ValueError('BAD_SELECTION_AUDIT_48')
    return expected

def summarize(review, expected):
    if review.get('schema') != 'prd32_dev60_new48_human_image_review_v1':
        raise ValueError('REVIEW_SCHEMA_MISMATCH')
    rows=review.get('rows')
    if not isinstance(rows,list) or len(rows)!=48:raise ValueError('REVIEW_NOT_48_ROWS')
    seen={}
    for r in rows:
        k=r.get('case_id')
        if k in seen:raise ValueError('DUPLICATE_REVIEW_ID:'+str(k))
        if k not in expected:raise ValueError('UNEXPECTED_REVIEW_ID:'+str(k))
        if r.get('image_sha256')!=expected[k]:raise ValueError('REVIEW_IMAGE_HASH_MISMATCH:'+k)
        if r.get('decision') not in DECISIONS:raise ValueError('BAD_REVIEW_DECISION:'+k)
        if type(r.get('confirmed')) is not bool:raise ValueError('BAD_CONFIRM_FLAG:'+k)
        for fld, allowed in [('visible_categories_human',CATS),('scene_flags_human',SCENES)]:
            vals=r.get(fld)
            if not isinstance(vals,list) or len(vals)!=len(set(vals)) or any(v not in allowed for v in vals):
                raise ValueError('BAD_REVIEW_FIELD:'+k+':'+fld)
        if r['confirmed'] and r['decision']=='UNREVIEWED':raise ValueError('CONFIRMED_BUT_UNREVIEWED:'+k)
        seen[k]=r
    confirmed = [r for r in rows if r['confirmed'] and r['decision']!='UNREVIEWED']
    reviewed_all=len(confirmed)==48
    claimed_completed=review.get('review_completed')
    if claimed_completed is not reviewed_all:raise ValueError('COMPLETION_FLAG_DISAGREES')
    accepted=[r for r in confirmed if r['decision']=='ACCEPT']
    reject=[r for r in confirmed if r['decision']=='REJECT']
    uncertain=[r for r in confirmed if r['decision']=='UNCERTAIN']
    cc=collections.Counter(v for r in accepted for v in r['visible_categories_human'])
    sc=collections.Counter(v for r in accepted for v in r['scene_flags_human'])
    missing=[c for c in CATS if cc.get(c,0)==0]
    return {
        'schema':'prd32_dev60_stage1d_human_image_level_qc_v1',
        'review_source':'HUMAN_SELF_CONFIRMED_IMAGE_LEVEL_REVIEW_WITH_AI_SUGGESTIONS',
        'review_completion_verified_structurally':reviewed_all,
        'reviewed_confirmed_images':len(confirmed),
        'images_expected':48,
        'accepted_images':len(accepted),
        'rejected_images':len(reject),
        'uncertain_images':len(uncertain),
        'unreviewed_case_ids':[k for k,r in sorted(seen.items()) if not r['confirmed'] or r['decision']=='UNREVIEWED'],
        'rejected_case_ids':[r['case_id'] for r in reject],
        'uncertain_case_ids':[r['case_id'] for r in uncertain],
        'visible_categories_per_accepted_photo_NOT_INSTANCE_GT':{c:cc.get(c,0) for c in CATS},
        'human_visible_category_uncovered':missing,
        'hard_scene_counts_per_accepted_photo':{c:sc.get(c,0) for c in SCENES},
        'missing_accepted_categories':missing,
        'remaining_to_target_if_only_accepted_retained':48-len(accepted),
        'holdout_sha_exclusion_verified':False,
        'cohort_frozen':False,
        'v5_inference_ran':False,
        'ready_for_main_ab_regression':False,
        'why_not_ready':'Need confirmed holdout exclusion / duplicate audit, retain exactly 48 eligible new images, then freeze. Image categories are not target-instance GT or detector recall.',
        'original_12_preserved_unmodified':True,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--review',type=Path,required=True)
    ap.add_argument('--selection-audit',type=Path,default=Path(__file__).resolve().parent/'new48_selection_audit.csv')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise SystemExit('REFUSE_TO_OVERWRITE_OUTPUT')
    summary=summarize(json.loads(a.review.read_text(encoding='utf-8-sig')),expected_from_audit(a.selection_audit))
    a.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('IMAGE_REVIEW_QC',json.dumps({'confirmed':summary['reviewed_confirmed_images'],'accepted':summary['accepted_images'],'remaining':summary['remaining_to_target_if_only_accepted_retained'],'missing_classes':summary['human_visible_category_uncovered']},ensure_ascii=False))
    print('NOT_FROZEN_NOT_READY_FOR_MAIN_AB')

if __name__=='__main__':main()
