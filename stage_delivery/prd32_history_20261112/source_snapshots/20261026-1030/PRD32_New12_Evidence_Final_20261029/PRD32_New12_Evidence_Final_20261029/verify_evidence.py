#!/usr/bin/env python3
from pathlib import Path
import json,csv,collections
root=Path(__file__).resolve().parent
with (root/'paired_comparison.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
s=json.loads((root/'comparison_summary.json').read_text(encoding='utf-8'))
blind={x['case_id']:x for x in json.loads((root/'new12_blind_review.json').read_text(encoding='utf-8'))['rows']}
review=json.loads((root/'new12_rootcause_human_review.json').read_text(encoding='utf-8'))['rows']
assert len(rows)==12 and len(blind)==12 and len({x['case_id'] for x in rows})==12
scored=[x for x in rows if x['scoreable']=='True']
assert len(scored)==s['scoreable']==9 and len(rows)-len(scored)==s['excluded']==3
counts=collections.Counter(x['pair_group'] for x in rows)
assert dict(counts)==s['paired_outcomes']
for x in rows:
 assert x['case_id'] in blind
 assert x['image_sha256']==blind[x['case_id']]['image_sha256']
 assert x['human_category']==blind[x['case_id']]['human_category']
 if x in scored:
  assert (x['v2_correct']=='True') == (x['v2_category']==x['human_category'])
  assert (x['v3_correct']=='True') == (x['v3_category']==x['human_category'])
assert sum(x['v2_correct']=='True' for x in scored)==s['v2_match']==8
assert sum(x['v3_correct']=='True' for x in scored)==s['v3_match']==8
for x in review:
 p=next(y for y in rows if y['case_id']==x['case_id'])
 assert x['image_sha256']==p['image_sha256']
 assert x['human_category']==p['human_category']
 assert (x['v2_root_cause']=='NOT_ERROR')==(p['v2_correct']=='True')
 assert (x['v3_root_cause']=='NOT_ERROR')==(p['v3_correct']=='True')
print('VERIFY_OK: 12 aligned; 9 scoreable; V2=8/9; V3=8/9; rootcause cases=2')
