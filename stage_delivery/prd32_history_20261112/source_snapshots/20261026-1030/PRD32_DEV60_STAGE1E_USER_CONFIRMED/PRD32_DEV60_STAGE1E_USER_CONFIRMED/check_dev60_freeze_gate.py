#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD32 V2 60-img freeze gate: immutable identity, local train image hashes, review, holdout digest exclusion.

Only reads candidate train photos and an explicitly supplied APPROVED DIGEST-ONLY registry.
Never searches or opens sealed/holdout/test/validation images or manifests.
No model invocation, no inference output changes, no acceptance decision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

HEX=re.compile(r'[0-9a-f]{64}\Z')
CATS=('top','pants','skirt','outerwear','dress','shoe','bag','accessory')
RESTRICTED={'holdout','sealed','test','testing','testset','eval','evaluation','val','valid','validation','fullval','devtest'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def jsonread(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    with Path(path).open('r',encoding='utf-8-sig') as f:
        return [json.loads(x) for x in f if x.strip()]


def parse_registry(path):
    """User-provided approved SHA-256-only record. Strictly no holdout paths/names/labels."""
    if path is None:
        return None
    p=Path(path)
    if any(x.lower() in RESTRICTED for x in p.parts):
        raise ValueError('RESTRICTED_REGISTRY_LOCATION_USE_A_SAFE_COPY_OF_DIGESTS_ONLY')
    if p.stat().st_size>200_000:
        raise ValueError('REGISTRY_TOO_LARGE')
    data=p.read_text(encoding='utf-8-sig')
    if p.suffix.lower()=='.json':
        obj=json.loads(data)
        if isinstance(obj,dict) and set(obj)=={'sha256'}:
            obj=obj['sha256']
        if not isinstance(obj,list):
            raise ValueError('REGISTRY_JSON_EXPECTS_LIST_OR_SHA256_KEY_ONLY')
        raw=obj
    elif p.suffix.lower()=='.txt':
        raw=[x.strip() for x in data.splitlines() if x.strip()]
    elif p.suffix.lower()=='.csv':
        import csv, io
        rows=list(csv.DictReader(io.StringIO(data)))
        if not rows or list(rows[0])!=['sha256']:
            raise ValueError('REGISTRY_CSV_ONE_COLUMN_SHA256')
        raw=[r['sha256'] for r in rows]
    else:
        raise ValueError('REGISTRY_DIGEST_ONLY_TXT_CSV_JSON')
    if not raw or any(not isinstance(s,str) or not HEX.fullmatch(s.lower()) for s in raw):
        raise ValueError('REGISTRY_SHA256_ONLY')
    if len(set(s.lower() for s in raw))<8:
        raise ValueError('REGISTRY_MUST_INCLUDE_AT_LEAST_8_DISTINCT_HOLDOUT_HASHES')
    return set(s.lower() for s in raw)


def cohort_checks(rows,reference):
    expected=reference['images']
    if len(rows)!=60 or len(expected)!=60:
        raise ValueError('COHORT_NOT_60')
    exp=[(x['case_id'],x['image_sha256']) for x in expected]
    got=[(x['case_id'],x['image_sha256']) for x in rows]
    if got!=exp:
        raise ValueError('COHORT_CHANGED_FROM_REVIEWED_V2_60_REFERENCE')
    if [r['case_id'] for r in rows]!=[f'fresh_dev_{i:03}' for i in range(1,61)]:
        raise ValueError('BAD_60_CASE_IDS')
    if len(set(x['image_sha256'] for x in rows))!=60 or any(not HEX.fullmatch(x['image_sha256']) for x in rows):
        raise ValueError('DUPLICATE_OR_INVALID_IMAGE_DIGEST')
    for x in rows:
        # Inputs contain only image locators. No GT labels or bounding boxes.
        if {'bbox','gt_bbox','category','visible_categories_human'} & set(x):
            raise ValueError('MODEL_INPUT_GT_LEAKAGE')
    return True


def review_checks(review,rows):
    if review.get('schema')!='prd32_dev60_new48_human_image_review_v1' or not review.get('review_completed'):
        raise ValueError('REVIEW_INCOMPLETE_OR_SCHEMA_MISMATCH')
    if not review.get('ai_prefill_disclosure') or review.get('official_prd_acceptance') is not False:
        raise ValueError('REVIEW_PROVENANCE_INVALID')
    mm={r['case_id']:r['image_sha256'] for r in rows}
    rr=review.get('rows',[])
    if len(rr)!=48 or len({r['case_id'] for r in rr})!=48:
        raise ValueError('REVIEW_NOT_48_DISTINCT')
    cc={k:0 for k in CATS}
    for r in rr:
        if r['case_id'] not in set(f'fresh_dev_{i:03}' for i in range(13,61)) or mm.get(r['case_id'])!=r['image_sha256']:
            raise ValueError('REVIEW_HASH_OR_CASE_MISMATCH')
        if r.get('decision')!='ACCEPT' or r.get('confirmed') is not True:
            raise ValueError('REVIEW_NOT_ALL_ACCEPTED_AND_USER_CONFIRMED')
        vals=r['visible_categories_human']
        if not isinstance(vals,list) or len(vals)!=len(set(vals)) or any(c not in CATS for c in vals):
            raise ValueError('INVALID_VISIBLE_CLASSES')
        for c in vals:cc[c]+=1
    return cc


def run(manifest,reference,review,approved=None,verify_bytes=True,train_root='/workspace/fashion_data'):
    rows=read_jsonl(manifest)
    ref=jsonread(reference)
    rv=jsonread(review)
    cohort_checks(rows,ref)
    categories=review_checks(rv,rows)
    if digest(manifest)!=ref['original_stage1_model_manifest_sha256']:
        raise ValueError('ORIGINAL_MANIFEST_SHA_CHANGED')
    if verify_bytes:
        base=Path(train_root).resolve()
        for r in rows:
            p=Path(r['image_path']).resolve()
            if not p.is_relative_to(base) or 'train' not in (x.lower() for x in p.parts) or any(x.lower() in RESTRICTED for x in p.parts):
                raise ValueError('UNSAFE_IMAGE_PATH:'+r['case_id'])
            if digest(p)!=r['image_sha256']:
                raise ValueError('TRAIN_IMAGE_MISSING_OR_CHANGED:'+r['case_id'])
    hashes={r['image_sha256'] for r in rows}
    reg=parse_registry(approved)
    overlaps=sorted(hashes & reg) if reg is not None else []
    all_eight=all(categories[c]>0 for c in CATS)
    ready=reg is not None and not overlaps and all_eight and verify_bytes
    return {
       'schema':'prd32_dev60_stage1e_holdout_exclusion_gate_v1',
       'status': 'BLOCKED_MISSING_APPROVED_HOLDOUT_DIGESTS' if reg is None else ('BLOCKED_HOLDOUT_SHA_OVERLAP' if overlaps else ('ELIGIBLE_FOR_FREEZE_PENDING_PROVENANCE_APPROVAL' if ready else 'BLOCKED_INCOMPLETE_EVIDENCE')),
       'original_12_preserved':True,
       'new48_user_batch_confirmed':True,
       'ai_assisted_image_review_not_instance_gt':True,
       'total_image_ids':60,
       'distinct_image_sha256':60,
       'all_eight_categories_visible_in_accepted_new48':all_eight,
       'new48_visible_categories_image_level_not_gt':categories,
       'train_images_rehashed':verify_bytes,
       'holdout_digest_registry_provided':reg is not None,
       'provided_registry_digest_count':len(reg) if reg is not None else 0,
       'registry_provenance_independently_verified':False,
       'holdout_overlap_sha_count':len(overlaps),
       'holdout_overlap_case_ids':[r['case_id'] for r in rows if r['image_sha256'] in overlaps],
       'holdout_sha_exclusion_checked_against_supplied_registry':reg is not None,
       'holdout_sha_exclusion_pass_against_supplied_registry':reg is not None and not overlaps,
       'near_duplicate_cross_split_exclusion_verified':False,
       'v2_freeze_clearance_conditional_on_registry_provenance':ready,
       'v2_frozen':False,
       'main_ab_inference_ran':False,
       'official_prd_acceptance':False,
       'publish_allowed':False,
       'notes':[
           'A digest check can only be trusted if the supplied registry is genuine, complete, approved and comes from the eight held-out images.',
           'Image-level visual classes are not instance ground truth and must never be passed to the model.',
           'Exact SHA matching does not rule out perceptual near duplicates between train and held-out splits.',
           'Does not modify the original candidate cohort, freeze state, or model pipeline.'
       ]
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,default=Path('/workspace/PRD32_FIXED_DEV_V2_60IMG_CANDIDATES_20261030/v5_image_only_inputs_60.jsonl'))
    ap.add_argument('--reference',type=Path,default=Path(__file__).with_name('dev60_image_sha_reference.json'))
    ap.add_argument('--review',type=Path,default=Path(__file__).with_name('PRD32_DEV60_NEW48_UserConfirmed_ChatAttestation.json'))
    ap.add_argument('--approved-holdout-sha',type=Path,default=None,help='Preapproved 8+ SHA-256 values only; never use sealed files/paths')
    ap.add_argument('--train-root',type=Path,default=Path('/workspace/fashion_data'))
    ap.add_argument('--output',type=Path,default=Path('/workspace/PRD32_DEV60_STAGE1E_GATE_RESULT.json'))
    ap.add_argument('--offline-snapshot-only',action='store_true',help='Debug only: skip train image byte verification, never passes freeze gate')
    args=ap.parse_args()
    if args.output.exists():raise SystemExit('REFUSE_OVERWRITE_OUTPUT:'+str(args.output))
    try:
        result=run(args.manifest,args.reference,args.review,args.approved_holdout_sha,not args.offline_snapshot_only,args.train_root)
    except Exception as e:
        result={'schema':'prd32_dev60_stage1e_holdout_exclusion_gate_v1',
                'status':'BLOCKED_VALIDATION_ERROR','error':str(e),'v2_frozen':False,'publish_allowed':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PRD32_DEV60_STAGE1E:',result['status'])
    print('REPORT:',args.output)
    if result.get('v2_freeze_clearance_conditional_on_registry_provenance'):
        print('HOLDOUT_DIGEST_CHECK_PASS_CONDITIONAL; DO_NOT_CALL_THIS_OFFICIAL_FREEZE')
    else:
        print('NO_FREEZE_NO_MAIN_AB')

if __name__=='__main__':main()
