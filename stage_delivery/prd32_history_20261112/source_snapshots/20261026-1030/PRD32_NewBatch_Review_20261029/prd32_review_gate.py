#!/usr/bin/env python3
"""PRD3.2 GT-free review coverage/safety gate: never authorizes publishing or PRD acceptance."""
import argparse, json, pathlib, sys
from collections import Counter
FIELDS={
 'instance_binding':{'CORRECT','WRONG_INSTANCE','UNCERTAIN'},
 'duplicate_assessment':{'DUPLICATE','DISTINCT','UNCERTAIN'},
 'tags_quality':{'CORRECT','INCORRECT','UNCERTAIN'},
 'vqa_quality':{'CORRECT','INCORRECT','ABSTENTION_REASONABLE','UNCERTAIN'},
 'copy_quality':{'CORRECT','WRONG_INSTANCE','DEBUG_TEXT','UNSUPPORTED_CLAIM','OTHER_ERROR','UNCERTAIN'},
}
EXPECTED={'fresh_dev_002':2,'fresh_dev_003':3,'fresh_dev_005':3}

def gate(review, summary):
 errors=[];warns=[]
 rows=review.get('rows')
 if not isinstance(rows,list):rows=[];errors.append('rows must be an array')
 ids=[]
 for i,r in enumerate(rows):
  if not isinstance(r,dict): errors.append(f'row #{i} is not an object');continue
  case=r.get('case_id');inst=r.get('instance_id')
  ids.append((case,inst))
  if case not in EXPECTED: errors.append(f'unknown case: {case}')
  for field,values in FIELDS.items():
   if r.get(field) not in values:errors.append(f'{case}/{inst}: bad or missing {field}: {r.get(field)}')
  if not isinstance(r.get('reviewer_note'),str) or not r['reviewer_note'].strip():
   errors.append(f'{case}/{inst}: reviewer_note missing')
 if len(rows)!=8:errors.append(f'expected 8 rows, got {len(rows)}')
 if len(ids)!=len(set(ids)):errors.append('duplicate case_id+instance_id keys')
 counts=Counter(c for c,i in ids)
 if dict(counts)!=EXPECTED:errors.append(f'case coverage mismatch: {dict(counts)} != {EXPECTED}')
 for case,count in EXPECTED.items():
  exp={f'pred_{i:04d}' for i in range(count)}
  found={i for c,i in ids if c==case}
  if found!=exp:errors.append(f'{case}: instance ids {sorted(found)} != {sorted(exp)}')
 if review.get('schema')!='prd32_v5_qwen_human_review_v1':errors.append('review schema mismatch')
 if summary.get('schema')!='prd32_gtfree_newbatch_service_gate_summary_v1':errors.append('service summary schema mismatch')
 if summary.get('instances')!=8 or set(summary.get('case_ids',[]))!=set(EXPECTED):errors.append('summary case/instance mismatch')
 if summary.get('publish_allowed') is not False or summary.get('official_PRD_acceptance') is not False:errors.append('unexpected input gate: publishing or acceptance enabled')
 if summary.get('scope')!='TRAIN_SOURCE_DEV_ONLY_NOT_INDEPENDENT':errors.append('source scope mismatch')
 is_human=review.get('reviewed_by_human') is True
 if not is_human:errors.append('reviewed_by_human must be explicitly true: AI draft is NOT human review')
 if any(isinstance(r,dict) and r.get('vqa_quality')=='UNCERTAIN' for r in rows):warns.append('VQA UNCERTAIN present; cannot infer VQA correctness')
 if any(isinstance(r,dict) and r.get('duplicate_assessment')=='DUPLICATE' for r in rows):warns.append('duplicate candidates remain; do not auto-deduplicate')
 if any(isinstance(r,dict) and r.get('copy_quality')!='CORRECT' for r in rows):warns.append('non-CORRECT Copy judgments remain; keep content blocked')
 result={
  'schema':'prd32_newbatch_human_review_validation_v1',
  'input_scope':'TRAIN_SOURCE_DEV_ONLY_NOT_INDEPENDENT',
  'validation_status': 'REVIEW_COMPLETED_DIAGNOSTIC_ONLY' if not errors else 'REVIEW_NOT_COMPLETED',
  'human_review_completed':bool(not errors and is_human),
  'validation_errors':errors,
  'risk_warnings':warns,
  'received_rows':len(rows),
  'per_field_review_counts':{field:dict(Counter(r.get(field,'MISSING') for r in rows if isinstance(r,dict))) for field in FIELDS},
  'service_runtime_counts_unchanged':summary.get('runtime_counts'),
  'source_sha256_unchanged':{'gt_free_raw_predictions_sha256':summary.get('gt_free_raw_predictions_sha256'), 'qwen_results_sha256':summary.get('qwen_results_sha256')},
  'publish_allowed':False,
  'official_PRD_acceptance':False,
  'independent_accuracy_available':False,
  'automatically_deduplicated_instances':0,
  'note':'A complete human review is a diagnostic audit, not independent GT, pass/fail acceptance or release authorization.'
 }
 return result

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--review',required=True)
 ap.add_argument('--summary',required=True)
 ap.add_argument('--out',default='prd32_newbatch_human_review_gate_result.json')
 args=ap.parse_args()
 try:review=json.loads(pathlib.Path(args.review).read_text(encoding='utf8'));summary=json.loads(pathlib.Path(args.summary).read_text(encoding='utf8'))
 except (OSError,ValueError) as e: print('INPUT_ERROR',e,file=sys.stderr);return 2
 result=gate(review,summary)
 pathlib.Path(args.out).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
 print(json.dumps(result,indent=2,ensure_ascii=False))
 return 0 if result['human_review_completed'] else 1
if __name__=='__main__':sys.exit(main())
