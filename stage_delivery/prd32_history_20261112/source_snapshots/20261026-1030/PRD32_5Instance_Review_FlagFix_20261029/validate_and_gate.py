#!/usr/bin/env python3
"""Check frozen five-case review identity, explicit human attestation and safe disposition.

This never edits the submitted review and never marks content publishable.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

EXPECTED = {
    ('fresh_dev_001','pred_0000'),('fresh_dev_001','pred_0001'),
    ('fresh_dev_004','pred_0000'),('fresh_dev_004','pred_0001'),
    ('fresh_dev_004','pred_0002'),
}
ENUMS = {
    'instance_binding':{'CORRECT','WRONG_INSTANCE','UNCERTAIN','UNREVIEWED'},
    'duplicate_assessment':{'DUPLICATE','DISTINCT','UNCERTAIN','UNREVIEWED'},
    'tags_quality':{'CORRECT','INCORRECT','UNCERTAIN','UNREVIEWED'},
    'vqa_quality':{'CORRECT','INCORRECT','ABSTENTION_REASONABLE','UNCERTAIN','UNREVIEWED'},
    'copy_quality':{'CORRECT','WRONG_INSTANCE','UNSUPPORTED_CLAIM','DEBUG_TEXT','OTHER_ERROR','UNCERTAIN','UNREVIEWED'},
}

def evaluate(review:dict)->dict:
    errors=[]
    if review.get('schema')!='prd32_v5_qwen_human_review_v1':errors.append('SCHEMA_MISMATCH')
    if review.get('scope')!='TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY':errors.append('SCOPE_MISMATCH')
    human=review.get('reviewed_by_human')
    if type(human) is not bool:errors.append('INVALID_HUMAN_CONFIRMATION_TYPE')
    rows=review.get('rows',[])
    if not isinstance(rows,list):
        errors.append('ROWS_NOT_LIST');rows=[]
    pairs=[]
    for r in rows:
        if not isinstance(r,dict):errors.append('NON_OBJECT_ROW');continue
        pairs.append((r.get('case_id'),r.get('instance_id')))
        for field,vals in ENUMS.items():
            if r.get(field) not in vals:errors.append(f'INVALID_{field}:{pairs[-1]}')
        if not isinstance(r.get('reviewer_note'),str):errors.append(f'MISSING_NOTE:{pairs[-1]}')
    if set(pairs)!=EXPECTED or len(pairs)!=len(EXPECTED):errors.append('FIVE_FROZEN_INSTANCES_IDENTITY_MISMATCH')
    unreviewed=sum(any(r.get(k)=='UNREVIEWED' for k in ENUMS) for r in rows if isinstance(r,dict))
    # Duplicate pair assessment must agree on both members; do not infer unique target IDs.
    byid={(r.get('case_id'),r.get('instance_id')):r for r in rows if isinstance(r,dict)}
    for a,b in [ (('fresh_dev_001','pred_0000'),('fresh_dev_001','pred_0001')), (('fresh_dev_004','pred_0000'),('fresh_dev_004','pred_0002')) ]:
        if a in byid and b in byid:
            if byid[a]['duplicate_assessment']!=byid[b]['duplicate_assessment']:
                errors.append(f'ASYMMETRIC_DUPLICATE_ASSESSMENT:{a[0]}')
    if errors: status='INVALID_REVIEW_FILE'
    elif human is not True or unreviewed:status='DRAFT_NOT_HUMAN_ATTESTED'
    else:status='HUMAN_REVIEW_RECORDED_NOT_PRD_ACCEPTANCE'
    counts={k:dict(collections.Counter(r[k] for r in rows if isinstance(r,dict))) for k in ENUMS} if not errors else {}
    return {
        'schema':'prd32_v5_qwen_review_confirmation_gate_v1',
        'status':status,'human_confirmation_recorded':human is True,
        'reviewed_rows':len(rows),'unreviewed_rows':unreviewed,
        'identity_check_passed':not errors,
        'errors':errors,'count_by_field':counts,
        'duplicate_handling':'REVIEW_ONLY_NO_AUTOMATIC_SUPPRESSION',
        'all_content_publish_allowed':False,
        'official_prd_acceptance':False,
        'note':'An explicit checkbox confirmation records that a person reviewed the prefilled choices; semantic correctness still needs ongoing QA and downstream dedup testing.'
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--review',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    src=Path(args.review)
    result=evaluate(json.loads(src.read_text(encoding='utf8')))
    result['review_sha256']=hashlib.sha256(src.read_bytes()).hexdigest()
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(result['status'])
    if result['errors']:raise SystemExit(2)
if __name__=='__main__':main()
