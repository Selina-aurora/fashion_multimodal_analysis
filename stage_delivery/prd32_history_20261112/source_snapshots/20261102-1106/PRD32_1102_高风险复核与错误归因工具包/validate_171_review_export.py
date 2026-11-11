#!/usr/bin/env python3
"""Validate a user-exported PRD32 171-instance paired review without fabricating confirmation.

Usage: python validate_171_review_export.py --review /path/to/export.json
Inputs beside this script: 原始171实例_AI预填草稿_仅供复核.json
"""
import argparse
import json
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent
TS = {'CORRECT','PARTIALLY_CORRECT','INCORRECT','UNCERTAIN'}
VS = {'CORRECT','INCORRECT','UNCERTAIN','REASONABLE_ABSTENTION'}
CS = {'CORRECT','DEBUG_TEXT','OTHER_ERROR','UNSUPPORTED_CLAIM','UNCERTAIN'}
BIND = {'CORRECT','INCORRECT','UNCERTAIN'}
DUP = {'DISTINCT','DUPLICATE','UNCERTAIN'}

def check(path):
    reference = json.loads((BASE/'原始171实例_AI预填草稿_仅供复核.json').read_text(encoding='utf-8'))
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    errs = []
    original = {(r['case_id'],r['instance_id']):r for r in reference['rows']}
    rows = obj.get('rows',[])
    keys = [(r.get('case_id'),r.get('instance_id')) for r in rows]
    if len(keys) != 171 or len(set(keys)) != 171 or set(keys) != set(original):
        errs.append('ROW_IDS_COUNT_OR_SET_MISMATCH')
    cnt = 0; confirmed_new = 0
    historical_bad = 0
    for r in rows:
        k = (r.get('case_id'),r.get('instance_id'))
        if k not in original: continue
        old = original[k]
        for immutable in ['image_sha256','marked_image_sha256','target_bbox_xyxy']:
            if immutable in old and r.get(immutable) != old.get(immutable):
                errs.append('IMMUTABLE_MISMATCH_'+str(k)+'_'+immutable)
        if r.get('instance_binding') not in BIND or r.get('duplicate_assessment') not in DUP:
            errs.append('BINDING_OR_DUP_ENUM_'+str(k))
        for variant in ('baseline','focus'):
            row = r.get(variant+'_quality',{})
            if row.get('tags') not in TS or row.get('vqa') not in VS or row.get('copy') not in CS:
                errs.append('QUALITY_ENUM_'+str(k)+'_'+variant)
        if old.get('review_origin') == 'HISTORICAL_USER_CONFIRMED_REUSED_UNMODIFIED':
            if not r.get('instance_review_confirmed'):
                errs.append('HISTORICAL_CONFIRMATION_LOST_'+str(k))
            # Old 31 cannot silently be re-scored
            for field in ['instance_binding','duplicate_assessment','baseline_quality','focus_quality']:
                if r.get(field) != old.get(field):
                    historical_bad += 1
                    errs.append('HISTORICAL_RESULT_CHANGED_'+str(k)+'_'+field)
        else:
            confirmed_new += bool(r.get('instance_review_confirmed'))
        cnt += bool(r.get('instance_review_confirmed'))
    if obj.get('reviewed_by_human') != (cnt==171):
        errs.append('REVIEW_COMPLETION_FLAG_INCONSISTENT')
    if obj.get('publish_allowed',False) or obj.get('official_prd_acceptance',False):
        errs.append('UNAUTHORIZED_OFFICIAL_OR_PUBLISH_FLAG')
    result = {
      'status':'PASS' if not errs else 'FAIL',
      'schema':'prd32_dev60_171_review_validation_v1',
      'total_rows':len(rows),
      'historical_confirmed':31,
      'new_confirmed':confirmed_new,
      'new_pending':140-confirmed_new,
      'full_user_review_finished':confirmed_new==140 and not errs,
      'historical_rescored_entries':historical_bad,
      'errors':errs[:30],
      'warning':'AI prefill for unconfirmed entries is not human quality. Development source only, not PRD acceptance.'
    }
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--review',type=Path,required=True)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    result=check(args.review)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.output:
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    raise SystemExit(0 if result['status']=='PASS' else 2)
