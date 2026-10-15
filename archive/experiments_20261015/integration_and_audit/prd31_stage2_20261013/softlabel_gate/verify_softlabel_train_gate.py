#!/usr/bin/env python3
"""Fail-closed snapshot, candidate-source, and frozen-set gate for 4-class sleeve soft-label pilot.
NO training is performed and no data is modified. Only exact known CSV snapshots pass.
"""
import argparse,csv,hashlib,json
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def verify(repo, data_dir, candidates_file, annotations_file, holdout_file, core_file):
 notes=[]; failures=[]
 ann=json.loads(Path(annotations_file).read_text('utf-8-sig'))
 train=Path(repo)/'configs/prd_8class_train_v5_balanced150_20261008.csv'
 val=Path(repo)/'configs/prd_8class_val_v2_full_targets.csv'
 expected_tr=ann.get('source_train_csv_sha256','')
 expected_va=ann.get('source_val_csv_sha256','')
 dig={}
 for label,path,expected in [('train',train,expected_tr),('val',val,expected_va)]:
  if not path.is_file():failures.append(f'{label}_snapshot_missing: {path}');continue
  actual=sha(path);dig[label]=actual
  if actual!=expected:failures.append(f'{label}_snapshot_sha_changed: expected {expected}, actual {actual}')
 # Never scan or decide training eligibility on modified/absent snapshots.
 if failures:return {'status':'BLOCKED','failures':failures,'snapshot_sha256':dig,'checked_candidates':0}
 tr=train.read_text('utf-8-sig').lower();va=val.read_text('utf-8-sig').lower()
 core={ln.strip().split()[0].lower() for ln in Path(core_file).read_text('utf8').splitlines() if ln.strip()}
 hold=json.loads(Path(holdout_file).read_text('utf-8-sig'))
 hs={r['sha256'].lower() for r in hold.get('records',[])}
 if len(core)!=400 or len(hs)!=20:
  failures.append(f'frozen_hash_list_count_changed: core={len(core)} holdout={len(hs)}')
 look={a['record_id']:a for a in ann.get('annotations',[])}
 with Path(candidates_file).open(encoding='utf-8-sig',newline='') as f: candidates=list(csv.DictReader(f))
 if len(candidates)!=10: failures.append(f'expected_10_dev_candidates_got_{len(candidates)}')
 results=[]
 for c in candidates:
  rid=c['record_id'];src=look.get(rid)
  if not src:failures.append(f'not_found_in_annotated_source: {rid}');continue
  source_sha=src['source_image_sha256'].lower()
  filename=Path(src['source_image']).name.lower()
  p=(Path(repo)/src['source_image']).resolve()
  expected_root=(Path(data_dir)/'raw/fashionpedia/images').resolve()
  issues=[]
  if not p.is_relative_to(expected_root):issues.append('unexpected_image_path')
  elif not p.is_file():issues.append('source_image_missing')
  elif sha(p)!=source_sha:issues.append('source_image_hash_mismatch')
  if filename not in tr:issues.append('not_found_in_train_manifest')
  if filename in va:issues.append('present_in_validation_manifest')
  if source_sha in hs:issues.append('overlaps_holdout20')
  if source_sha in core:issues.append('overlaps_core400')
  if src.get('prescreen_decision')!='priority' or src.get('state')!='measurable':issues.append('not_priority_measurable')
  if c['source_image_sha256'].lower()!=source_sha:issues.append('candidate_sha_not_equal_annotation')
  results.append({'record_id':rid,'source_file':filename,'image_sha256':source_sha,'check':'PASS' if not issues else 'BLOCKED','issues':issues})
  if issues:failures.append(f'{rid}: '+','.join(issues))
 return {'status':'CANDIDATE_SOURCE_GATE_PASSED_NO_TRAINING' if not failures else 'BLOCKED',
         'training_authorized':False,'training_done':False,'classes':['sleeveless','cap','short','elbow'],
         'snapshot_sha256':dig,'candidate_count':len(candidates),'checked_candidates':len(results),
         'candidate_results':results,'failures':failures,
         'limitations':['This verifies only the 10 pre-screened candidates against fixed CSV snapshots; not new GT approval.',
           'Four-class pilot does not constitute six-class supervised soft-label validation.',
           'Image filename membership is used for CSV linkage; request independent cross-split image-hash audit before training.']}

def main():
 d=Path(__file__).parent
 ap=argparse.ArgumentParser();ap.add_argument('--repo',default='/workspace/fashion_multimodal_analysis')
 ap.add_argument('--data-dir',default='/workspace/fashion_data')
 ap.add_argument('--output',default='/workspace/softlabel_gate_report_20261013.json')
 a=ap.parse_args()
 x=verify(a.repo,a.data_dir,d/'inputs/future_train_candidates_REQUIRES_ISOLATION_CHECK.csv',d/'inputs/sleeve_continuous_candidates_review_20261012.json',d/'inputs/holdout20_frozen_manual_for_sha_audit.json',d/'inputs/core400_source_sha256_20261012.txt')
 op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True);op.write_text(json.dumps(x,ensure_ascii=False,indent=2),'utf8')
 print(x['status']);print('REPORT:',op)
 print('TRAINING_AUTHORIZED:',x.get('training_authorized',False))
 print('ISSUES:',x.get('failures',[]))
 if x['status']=='BLOCKED':raise SystemExit(2)
if __name__=='__main__':main()
