#!/usr/bin/env python3
from __future__ import annotations
import csv, json
from pathlib import Path
from collections import Counter
P=Path(__file__).resolve().parent
S=P/'sources'
R=json.loads((S/'registry40.json').read_text(encoding='utf-8'))
old=json.loads((S/'historical19.json').read_text(encoding='utf-8'))['rows']+json.loads((S/'historical4.json').read_text(encoding='utf-8'))['rows']
new=json.loads((S/'new17_blind_review.json').read_text(encoding='utf-8'))['rows']
reviews={r['case_id']:r for r in old+new}
def readj(path):
 rows=[json.loads(z) for z in (S/path).read_text(encoding='utf-8').splitlines() if z.strip()]
 assert len(rows)==40 and len({r['case_id'] for r in rows})==40
 return {r['case_id']:r for r in rows}
v2=readj('v2_raw40.jsonl');v3=readj('v3_full_raw40.jsonl')
assert len(R)==len(reviews)==40
out=[]
for case in sorted(R,key=lambda z:z['case_id']):
 cid=case['case_id'];h=reviews[cid];cat=h['human_category'];reason=[]
 assert v2[cid]['input_image_sha256']==v3[cid]['input_image_sha256']
 assert case['v2_category']==v2[cid]['canonical_tags']['category']
 assert case['v3_category']==v3[cid]['category']
 if cat not in {'top','pants','skirt','outerwear','dress','shoe','bag','accessory'}:reason.append('CATEGORY_UNCERTAIN')
 if cid=='train_dev_020':reason.append('VISUAL_REVIEW_CONFLICT_SKIRT_VS_PANTS')
 if cid=='train_dev_019' or h.get('bbox_status')=='INCOMPLETE':reason.append('INCOMPLETE_HISTORICAL_BBOX')
 if h.get('bbox_coverage')=='NO':reason.append('BBOX_COVERAGE_NO')
 if h.get('visually_identifiable')=='NO':reason.append('NOT_VISUALLY_IDENTIFIABLE')
 if h.get('visually_identifiable')=='UNCERTAIN':reason.append('VISIBILITY_UNCERTAIN')
 a=case['v2_category'];b=case['v3_category'];ok=not reason
 ac=a==cat if ok else None;bc=b==cat if ok else None
 outcome='EXCLUDED' if not ok else ('BOTH_CORRECT' if ac and bc else 'V2_WRONG_V3_CORRECT' if bc else 'V2_CORRECT_V3_WRONG' if ac else 'BOTH_WRONG')
 out.append(dict(case_id=cid,human_category=cat,scoreable='YES' if ok else 'NO',exclusion_reason=';'.join(dict.fromkeys(reason)),v2_category=a,v3_full_tags_category=b,v2_correct='YES' if ac else 'NO' if ok else 'EXCLUDED',v3_correct='YES' if bc else 'NO' if ok else 'EXCLUDED',change_type=outcome,notes=h.get('notes','').replace('\n',' ')))
valid=[r for r in out if r['scoreable']=='YES'];assert len(valid)==35
stat={'scoreable':35,'v2_correct':sum(r['v2_correct']=='YES' for r in valid),'v3_correct':sum(r['v3_correct']=='YES' for r in valid),'paired':dict(Counter(r['change_type'] for r in valid))}
with (P/'replayed_comparison40.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
(P/'replayed_summary.json').write_text(json.dumps(stat,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(stat,ensure_ascii=False,indent=2))
