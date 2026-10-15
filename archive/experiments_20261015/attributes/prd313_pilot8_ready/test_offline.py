from pathlib import Path
import json, hashlib, ast, csv
R=Path(__file__).resolve().parent
script=(R/'run_pilot8_fixed.py').read_text(encoding='utf-8')
ast.parse(script)
sha=lambda name:hashlib.sha256((R/name).read_bytes()).hexdigest()
pilot=json.loads((R/'pilot8_candidates_FINAL.json').read_text())['annotations']
old=json.loads((R/'prior_8case_frozen_classifiers.json').read_text())
pre=json.loads((R/'pilot8_preflight_core400_report.json').read_text())
assert len(pilot)==8 and len(old)==8
assert {r['record_id'] for r in pilot}=={r['record_id'] for r in old}
assert all(r['current_class'] in ('cap','short','elbow') and r['review_status']=='confirmed' for r in pilot)
assert pre['status']=='PREFLIGHT_PASSED_NO_OVERLAP' and pre['checks']['core400_overlap_free']
assert len(sha('pilot8_candidates_FINAL.json'))==64 and sha('pilot8_candidates_FINAL.json') in script
assert 'DRY_RUN_PASS' in script and 'fixed_anchor_class_correct' in script
print('OFFLINE_CHECK_PASS: 8 paired records, source SHA, preflight, and Python syntax')
