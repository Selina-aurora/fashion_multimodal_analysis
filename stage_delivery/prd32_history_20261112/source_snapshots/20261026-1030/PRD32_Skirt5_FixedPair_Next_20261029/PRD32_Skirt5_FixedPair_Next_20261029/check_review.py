#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check five-case skirt review against the original blind HTML metadata or prepared manifest."""
import argparse,json,re,sys,hashlib
from collections import Counter
from pathlib import Path

ID={f'skirt_diag_{i:03d}' for i in range(1,6)}
CAT={'top','pants','skirt','outerwear','dress','shoe','bag','accessory','uncertain'}
EVID={'HIGH','MEDIUM','LOW'}
SPLIT={'YES','NO','UNCERTAIN'}
BOUND={'CLEAR','TOP_OUTERWEAR','TOP_DRESS','SKIRT_PANTS','OTHER_BOUNDARY'}

def die(msg):raise ValueError(msg)
def load_review(path):
    obj=json.loads(Path(path).read_text(encoding='utf8'))
    if obj.get('schema')!='prd32_new_train_dev_blind_review_v1' or obj.get('scope')!='TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY':die('Review schema/scope invalid')
    rows=obj.get('rows')
    if not isinstance(rows,list) or len(rows)!=5 or {r.get('case_id') for r in rows}!=ID:die('Expected exactly 5 distinct skirt_diag_001..005')
    checks=[]
    for r in rows:
        cid=r['case_id'];s=r.get('image_sha256','')
        if not re.fullmatch('[0-9a-f]{64}',s):die('Missing SHA: '+cid)
        if r.get('human_category') not in CAT:die('Category invalid: '+cid)
        if r.get('bbox_status') not in SPLIT:die('bbox_status invalid: '+cid)
        if r.get('visually_identifiable') not in {'YES','NO'}:die('visually_identifiable invalid: '+cid)
        if r.get('overlap_other_garments') not in SPLIT:die('overlap invalid: '+cid)
        if r.get('boundary_case') not in BOUND:die('boundary_case invalid: '+cid)
        if r.get('evidence_level') not in EVID:die('evidence level invalid: '+cid)
        if not (r.get('reviewer_note') or '').strip():die('No reviewer note: '+cid)
        scoreable=r['human_category']!='uncertain' and r['bbox_status']=='YES' and r['visually_identifiable']=='YES' and r['evidence_level']!='LOW'
        checks.append({'case_id':cid,'human_category':r['human_category'],'scoreable':scoreable,'sha':s})
    if len({r['sha'] for r in checks})!=5:die('Duplicate image SHA')
    return checks

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--review',default=str(Path(__file__).with_name('Skirt5_BlindReview_Reviewed.json')));ap.add_argument('--prepared');ap.add_argument('--html');a=ap.parse_args()
    checks=load_review(a.review)
    known={r['case_id']:r['sha'] for r in checks}
    if a.html:
        path=Path(a.html)
        if not path.is_file():die('Blind HTML missing: '+str(path))
        s=path.read_text(encoding='utf8')
        m=re.search(r'const META=(\[.*?\]);',s,re.S)
        if not m:die('Original blind HTML has no META array')
        html=json.loads(m.group(1))
        if {x['case_id']:x['image_sha256'] for x in html}!=known:die('Review vs blind HTML image identities mismatch')
    if a.prepared:
        d=Path(a.prepared)
        if not d.is_dir():die('Prepared directory missing: '+str(d))
        p=d/'candidate_manifest.jsonl';meta=d/'summary.json'
        items=[json.loads(l) for l in p.read_text(encoding='utf8').splitlines() if l.strip()]
        if {x['case_id']:x['sha256'] for x in items}!=known:die('Review vs candidate manifest image identities mismatch')
        summary=json.loads(meta.read_text(encoding='utf8'))
        if summary.get('selected_count')!=5:die('Prepared 5-count mismatch')
        if summary.get('candidate_manifest_sha256')!=hashlib.sha256(p.read_bytes()).hexdigest():die('Manifest hash mismatch')
    score=sum(x['scoreable'] for x in checks)
    if score!=4:die(f'Expected 4 scoreable, instead {score}; review changed, stop and inspect')
    if next(x for x in checks if x['case_id']=='skirt_diag_002')['human_category']!='skirt':die('Unexpected change to case 002')
    if next(x for x in checks if x['case_id']=='skirt_diag_005')['human_category']!='uncertain':die('Unexpected change to case 005')
    print('SKIRT5_REVIEW_CHECK_OK',json.dumps({'total':5,'scoreable':score,'excluded':5-score,'categories':dict(Counter(x['human_category'] for x in checks)),'identities_checked':bool(a.html or a.prepared),'case_002_user_label_retained':'skirt','case_005_excluded':True},ensure_ascii=False))
if __name__=='__main__':main()
