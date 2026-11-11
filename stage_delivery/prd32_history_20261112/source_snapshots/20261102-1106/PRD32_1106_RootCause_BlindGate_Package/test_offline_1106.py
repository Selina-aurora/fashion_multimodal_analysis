from pathlib import Path
import json,subprocess,tempfile,csv,sys
ROOT=Path(__file__).parent
summary=json.loads((ROOT/'output/ROOTCAUSE_DEV60_SUMMARY.json').read_text(encoding='utf-8'))
assert summary['unique_images']==60 and summary['v5_predictions']==171
assert summary['overlap_pairs_ge_05']==43 and summary['overlap_pairs_ge_07']==22 and summary['overlap_pairs_ge_08']==20
assert summary['overlap_pairs_ge_05_cross_predicted_category']==43
assert summary['bbox_and_class_both_reviewed_correct']==89
assert summary['nov4_focus_subtypes']['SPATIAL_DRIFT']==5
assert summary['nov4_focus_subtypes']['SEMANTIC_CATEGORY_INTERFERENCE']==1
assert summary['official_holdout_used'] is False
r=subprocess.run([sys.executable,str(ROOT/'blind_holdout_gate.py'),'--candidate-lock',str(ROOT/'CANDIDATE_LOCK_TEMPLATE.json'),'--approved-holdout',str(ROOT/'APPROVED_HOLDOUT_METADATA_TEMPLATE.json')],text=True,capture_output=True)
assert r.returncode==2 and 'BLIND_GATE_BLOCKED' in r.stdout
with tempfile.TemporaryDirectory() as d:
 t=Path(d);lock=json.loads((ROOT/'CANDIDATE_LOCK_TEMPLATE.json').read_text());lock.update(status='FROZEN',variant_name='A_TEST_SYNTHETIC',model_checkpoint_sha256='a'*64,pipeline_code_sha256='b'*64,evaluation_protocol_sha256='c'*64)
 hold=json.loads((ROOT/'APPROVED_HOLDOUT_METADATA_TEMPLATE.json').read_text());hold.update(official_prd32_holdout_confirmed=True,source_cohort='SYNTHETIC_FOR_TEST',approved_by='SYNTHETIC_TEST_ONLY',approval_record_id='SYNTHETIC',sealed_before_candidate_freeze=True,cross_split_duplicate_review_approved=True,holdout_image_sha256=['f'*64])
 (t/'lock.json').write_text(json.dumps(lock));(t/'hold.json').write_text(json.dumps(hold))
 cmd=[sys.executable,str(ROOT/'blind_holdout_gate.py'),'--candidate-lock',str(t/'lock.json'),'--approved-holdout',str(t/'hold.json')]
 ok=subprocess.run(cmd,text=True,capture_output=True)
 assert ok.returncode==0,ok.stdout+ok.stderr
 with open(ROOT/'output/DEV60_IMAGE_FROZEN_MANIFEST.csv',encoding='utf-8-sig',newline='') as f:
  dev=list(csv.DictReader(f))
 hold['holdout_image_sha256']=[dev[0]['image_sha256']];(t/'hold.json').write_text(json.dumps(hold))
 bad=subprocess.run(cmd,text=True,capture_output=True)
 assert bad.returncode==2 and 'overlap' in bad.stdout.lower()
print('PRD32_1106_OFFLINE_TESTS_PASS: schema+geometry+rootcause+blind-gate-block+synthetic-pass+SHA-leak-block')
