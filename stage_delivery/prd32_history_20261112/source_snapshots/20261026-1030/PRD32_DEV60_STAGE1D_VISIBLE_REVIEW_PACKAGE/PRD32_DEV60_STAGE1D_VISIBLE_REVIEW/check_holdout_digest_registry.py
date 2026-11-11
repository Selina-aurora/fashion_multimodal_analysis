#!/usr/bin/env python3
"""Only compare approved SHA256 exclusion registry to selected image digests.
Does NOT open holdout images, holdout manifests, or GT.
"""
import argparse
import json
import re
import hashlib
from pathlib import Path

SHA256_PATTERN=re.compile(r'^[a-fA-F0-9]{64}$')

def compare_cohort(lines, registry):
    cases=[json.loads(x) for x in lines if x.strip()]
    digests=[r['image_sha256'].lower() for r in cases]
    hashes=[s.strip().lower() for s in registry if s.strip() and not s.lstrip().startswith('#')]
    if len(cases)!=60 or len(set(digests))!=60:raise ValueError('COHORT_NOT_60_UNIQUE_IMAGES')
    if any(not SHA256_PATTERN.fullmatch(x) for x in digests):raise ValueError('BAD_COHORT_HASH')
    if len(hashes)<8 or any(not SHA256_PATTERN.fullmatch(x) for x in hashes) or len(set(hashes))!=len(hashes):
        raise ValueError('UNAPPROVED_OR_INVALID_DIGEST_REGISTRY')
    overlap=set(digests)&set(hashes)
    return {'cohort_images':len(cases),'approved_exclusion_hash_count':len(hashes),
            'overlap_images':len(overlap),'overlap_case_ids':[r['case_id'] for r in cases if r['image_sha256'].lower() in overlap],
            'no_exact_sha_overlap':len(overlap)==0,
            'metadata_only':True,'separate_holdout_unopened':True,
            'note':'Exact SHA absence does NOT rule out near-duplicates or leakage from a related train source.'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cohort',type=Path,required=True,help='v5_image_only_inputs_60.jsonl')
    ap.add_argument('--approved-sha256',type=Path,required=True,help='Pre-approved exclusion digest list; NOT a sealed manifest')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise SystemExit('REFUSE_OVERWRITE')
    result=compare_cohort(a.cohort.read_text().splitlines(),a.approved_sha256.read_text().splitlines())
    result['approved_registry_sha256']=hashlib.sha256(a.approved_sha256.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('EXACT_HOLDOUT_DIGEST_CHECK',json.dumps({'overlap_images':result['overlap_images'],'no_exact_sha_overlap':result['no_exact_sha_overlap']}))
    print('NO_MODEL_INFERENCE_NO_HOLDOUT_READ')
    if not result['no_exact_sha_overlap']:raise SystemExit(2)
if __name__=='__main__':main()
