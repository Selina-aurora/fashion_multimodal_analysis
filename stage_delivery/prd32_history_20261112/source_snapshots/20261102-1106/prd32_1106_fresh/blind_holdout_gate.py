#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit metadata-only gate for an *approved*, previously unseen PRD3.2 holdout.
Does not read any image, human GT, or model predictions. Does not trigger inference.
By default this gate BLOCKS. W7's eight sealed images are NOT presumed PRD3.2 holdout.
"""
import argparse,json,csv,hashlib,re,sys
from pathlib import Path
HEX64=re.compile(r'^[0-9a-f]{64}$')

def fail(msg):
    print('BLIND_GATE_BLOCKED:',msg)
    return 2

def main():
    pa=argparse.ArgumentParser(description='PRD3.2 independent blind holdout metadata-only gate')
    pa.add_argument('--candidate-lock',required=True)
    pa.add_argument('--approved-holdout',required=True)
    pa.add_argument('--dev-manifest',default=str(Path(__file__).parent/'output/DEV60_IMAGE_FROZEN_MANIFEST.csv'))
    a=pa.parse_args()
    try:
        lock=json.loads(Path(a.candidate_lock).read_text(encoding='utf-8'))
    except Exception as e: return fail('candidate lock missing/unreadable')
    if lock.get('status')!='FROZEN':return fail('candidate must be FROZEN before access')
    for field in ['model_checkpoint_sha256','pipeline_code_sha256','evaluation_protocol_sha256']:
        if not HEX64.fullmatch(str(lock.get(field,''))):return fail('invalid or absent candidate '+field)
    if lock.get('variant_name') in (None,'','TBD'):return fail('candidate variant unspecified')
    if lock.get('dev_set_images')!=60 or lock.get('dev_set_instances')!=171:
        return fail('frozen dev cohort metadata not 60/171')
    try: hold=json.loads(Path(a.approved_holdout).read_text(encoding='utf-8'))
    except Exception: return fail('independent holdout approval missing/unreadable')
    if hold.get('official_prd32_holdout_confirmed') is not True:
        return fail('formal PRD3.2 holdout cohort NOT confirmed')
    if hold.get('approved_by') in ('',None,'TBD') or not hold.get('approval_record_id'):
        return fail('approval identity / record absent')
    if hold.get('sealed_before_candidate_freeze') is not True:
        return fail('holdout not documented as sealed before candidate freeze')
    if hold.get('source_cohort') in ('W7_SEALED_8_UNCONFIRMED','TRAIN_SOURCE_DEV60'):
        return fail('unapproved or development cohort')
    if hold.get('cross_split_duplicate_review_approved') is not True:
        return fail('near-duplicate screening not approved')
    hashes=hold.get('holdout_image_sha256',[])
    if not isinstance(hashes,list) or not hashes or len(set(hashes))!=len(hashes):
        return fail('holdout digest manifest absent or duplicate')
    if not all(HEX64.fullmatch(str(x)) for x in hashes):return fail('bad holdout digest')
    with open(a.dev_manifest,encoding='utf-8-sig',newline='') as f:
        dev=list(csv.DictReader(f))
    if len(dev)!=60 or len(set(x['image_sha256'] for x in dev))!=60:return fail('bad dev manifest')
    if set(hashes)&{x['image_sha256'] for x in dev}:return fail('EXACT SHA overlap with dev60')
    print('BLIND_METADATA_GATE_PASS:',len(dev),'development images,',len(hashes),'approved holdout images')
    print('Important: metadata-only gate; no model inference, no labels opened; evaluation remains blinded.')
    return 0
if __name__=='__main__':sys.exit(main())
