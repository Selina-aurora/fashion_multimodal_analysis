#!/usr/bin/env python3
"""Check a returned human review JSON. Does not overwrite reviewer data."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter

CHOICES={'pending','yes','no','uncertain'}
FIELDS=('association','bbox','type')

def validate(review, cases, source_digest):
    flags=[]
    if review.get('schema_version')!='prd312_pocket_human_review_v1':
        flags.append('Unexpected human review schema')
    if review.get('source_cases_sha256')!=source_digest:
        flags.append('Candidate manifest SHA256 mismatch')
    expected={c['record_id']:c for c in cases}
    records=review.get('records',[])
    actual={r.get('record_id'):r for r in records}
    if set(actual)!=set(expected):flags.append('Missing, extra or renamed record ID')
    if len(actual)!=len(records):flags.append('Duplicate record ID')
    counts=Counter()
    incomplete=[]
    for rid,c in expected.items():
        r=actual.get(rid)
        if not r:continue
        if r.get('split')!=c['split'] or r.get('category')!=c['category'] or r.get('source_image_sha256')!=c['source_image_sha256']:
            flags.append(f'{rid}: split/category/source SHA altered')
        if r.get('garment_annotation_id')!=c['native_garment_annotation_id']:
            flags.append(f'{rid}: garment annotation changed')
        by_id={p.get('part_annotation_id'):p for p in r.get('parts',[])}
        expected_ids={p['part_annotation_id'] for p in c['parts']}
        if set(by_id)!=expected_ids or len(by_id)!=len(r.get('parts',[])):
            flags.append(f'{rid}: part annotation set changed')
        for pid in expected_ids:
            if pid not in by_id:continue
            human=by_id[pid].get('review',{})
            for field in FIELDS:
                val=human.get(field)
                if val not in CHOICES: flags.append(f'{rid} P{pid}: invalid {field}={val}')
                counts[f'{field}:{val}']+=1
            if any(human.get(field)=='pending' for field in FIELDS):
                incomplete.append({'record_id':rid,'part_annotation_id':pid})
    return {'structural_valid':not flags,'flags':flags,'completed_parts':42-len(incomplete),'total_parts':42,
            'pending_parts':incomplete,'answer_counts':dict(counts),
            'note':'Pass of structural review does not mean proposals were confirmed.'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--review',required=True)
    ap.add_argument('--cases',default=str(Path(__file__).parent/'inputs/review_cases.json'))
    ap.add_argument('--report',default=None)
    args=ap.parse_args()
    p=Path(args.cases);cases=json.loads(p.read_text(encoding='utf8'))
    rv=json.loads(Path(args.review).read_text(encoding='utf8'))
    result=validate(rv,cases,hashlib.sha256(p.read_bytes()).hexdigest())
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.report:Path(args.report).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    raise SystemExit(0 if result['structural_valid'] else 2)
if __name__=='__main__':main()
