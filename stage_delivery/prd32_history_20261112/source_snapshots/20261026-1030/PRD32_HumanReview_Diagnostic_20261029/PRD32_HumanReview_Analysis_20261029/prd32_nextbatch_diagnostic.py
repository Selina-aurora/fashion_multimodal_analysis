#!/usr/bin/env python3
"""Strict read-only PRD3.2 developer batch audit. No PRD acceptance, no auto dedup or publish.
Inputs: human review JSON, original service summary, extracted 8 candidate outputs.
Outputs: JSON + CSV, preserving all original service counts and identifiers.
"""
import argparse, csv, hashlib, importlib.util, json, pathlib, re
from collections import Counter, defaultdict

EXPECTED={'fresh_dev_002':2,'fresh_dev_003':3,'fresh_dev_005':3}
ACTIONS={
 ('fresh_dev_002','pred_0000'):('P1','Correct visually unsupported design tags; enforce garment-only copy; exclude unrelated bottom item.'),
 ('fresh_dev_002','pred_0001'):('P0','Block Qwen top label and mixed-instance tags; request image-to-target binding check for bottom garment.'),
 ('fresh_dev_003','pred_0000'):('P0','Same-shirt duplicate group; investigate V5 dress/top conflict; remove red-box phrases in copy.'),
 ('fresh_dev_003','pred_0001'):('P0','Same-shirt duplicate group; investigate V5 outerwear/top conflict; remove red-box phrases in copy.'),
 ('fresh_dev_003','pred_0002'):('P1','Same-shirt duplicate group; compare against other candidates before choosing representative; remove red-box phrases.'),
 ('fresh_dev_005','pred_0000'):('P0','Partial top ROI: resolve localization/binding before describing collar or full garment; review VQA prompt.'),
 ('fresh_dev_005','pred_0001'):('P0','Duplicate denim shorts group; resolve pants/skirt disagreement; remove debug text and weak details.'),
 ('fresh_dev_005','pred_0002'):('P0','Duplicate denim shorts group; double wrong model agreement requires visual class check; remove unsupported details.'),
}
DUP_GROUPS={
 'fresh_dev_003':{'pred_0000','pred_0001','pred_0002'},
 'fresh_dev_005':{'pred_0001','pred_0002'},
}
DEBUG_COPY=re.compile(r'红色(?:边框|框|标注框)|红框|框内展示|如图[示中]',re.I)
ABSTAIN=re.compile(r'无法确定|无法判断|不知道|不确定|不能确定')

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(pathlib.Path(p).read_text(encoding='utf8'))
def output_parts(record):
 out=record.get('outputs',[])
 if len(out)!=3 or any(not isinstance(s,str) for s in out): raise ValueError('Input extracted outputs must have exactly TAGS, VQA, COPY sections')
 if not all(x.startswith(y+'\n') for x,y in zip(out,('TAGS','VQA','COPY'))): raise ValueError('Unexpected output order or labels')
 m=re.search(r'```json\s*(\{.*?\})\s*```',out[0],re.DOTALL)
 if not m: raise ValueError('Missing structured TAGS JSON')
 return json.loads(m.group(1)),out[1][4:].strip(),out[2][5:].strip()

def analyze(review,summary,extracted,review_hash):
 spec=importlib.util.spec_from_file_location('gate',str(pathlib.Path(__file__).with_name('prd32_review_gate_prefilled.py')))
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 gate=module.gate(review,summary)
 if not gate['human_review_completed']:
  raise ValueError('Review gate NOT complete: '+ '; '.join(gate['validation_errors']))
 if review.get('scope')!='TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY' or review.get('review_status')!='HUMAN_REVIEW_COMPLETED_DIAGNOSTIC_ONLY':
  raise ValueError('Review scope/status mismatch')
 if review.get('original_vqa_questions_provided') is not False:
  raise ValueError('VQA question availability unexpected; re-evaluate quality policy')
 indexed={item['id']:item for item in extracted}
 expected={r['case_id']+'__'+r['instance_id'] for r in review['rows']}
 if len(indexed)!=8 or set(indexed)!=expected: raise ValueError('Extracted output coverage mismatch')
 rows=[]
 for r in review['rows']:
  case,inst=r['case_id'],r['instance_id'];id_=case+'__'+inst
  rec=indexed[id_]; meta=rec['metadata']; tags,vqa,copy=output_parts(rec)
  cl=re.search(r'V5:\s*([a-z]+)\s*/\s*Qwen:\s*([a-z]+)',meta)
  if not cl:raise ValueError('Missing model categories for '+id_)
  v5,qwen=cl.group(1),cl.group(2)
  svc_dup='Duplicate candidate: True' in meta
  blocked='Copy: BLOCKED' in meta
  flags=re.search(r'Flags:\s*(.+)$',meta)
  flags=(flags.group(1) if flags else '').strip()
  if (r['duplicate_assessment']=='DUPLICATE')!=svc_dup: raise ValueError('Human and service duplicate flags conflict for '+id_)
  if r['vqa_quality']!='UNCERTAIN': raise ValueError('VQA marked decided without original question for '+id_)
  if qwen!=tags.get('category'): raise ValueError('Meta/TAGS category inconsistency for '+id_)
  issues=[]
  if r['instance_binding']=='UNCERTAIN':issues.append('TARGET_BINDING_UNCERTAIN')
  if r['instance_binding']=='WRONG_INSTANCE':issues.append('WRONG_TARGET')
  if svc_dup: issues.append('DUPLICATE_CANDIDATE')
  if v5!=qwen:issues.append('MODEL_CLASS_CONFLICT')
  if r['tags_quality']=='INCORRECT':issues.append('TAGS_HUMAN_MARKED_INCORRECT')
  if DEBUG_COPY.search(copy):issues.append('COPY_DEBUG_FRAME_REFERENCE')
  if r['copy_quality']!='CORRECT':issues.append('COPY_HUMAN_MARKED_PROBLEM')
  if ABSTAIN.search(vqa):issues.append('VQA_ABSTENTION_UNVERIFIABLE')
  priority,action=ACTIONS[(case,inst)]
  row={
    'case_id':case,'instance_id':inst,'v5_category':v5,'qwen_category':qwen,
    'target_binding_review':r['instance_binding'],'duplicate_review':r['duplicate_assessment'],
    'tags_review':r['tags_quality'],'vqa_review':r['vqa_quality'],
    'copy_review':r['copy_quality'],'service_copy_blocked':blocked,
    'service_flags':flags,'detected_issues':issues,'priority':priority,
    'recommended_next_step':action,'reviewer_note':r['reviewer_note'],
    'copy_debug_reference_detected':bool(DEBUG_COPY.search(copy)),
    'vqa_answer':vqa,'copy_original':copy,
    'publish_allowed':False,'official_PRD_acceptance':False,
  }
  rows.append(row)
 # Service counts are original source only; compare derived checks for visible fields, do not rewrite
 checks={
  'duplicate_candidate_instances':sum('DUPLICATE_CANDIDATE' in a['detected_issues'] for a in rows),
  'v5_qwen_class_disagreements':sum('MODEL_CLASS_CONFLICT' in a['detected_issues'] for a in rows),
  'copy_blocked':sum(a['service_copy_blocked'] for a in rows),
  'vqa_abstentions':sum('VQA_ABSTENTION_UNVERIFIABLE' in a['detected_issues'] for a in rows),
 }
 for k,v in checks.items():
  if summary['runtime_counts'].get(k)!=v: raise ValueError(f'Service count reconciliation failure {k}: {summary["runtime_counts"].get(k)} != {v}')
 copy_counts=dict(Counter(a['copy_review'] for a in rows))
 counts={
  'instance_binding':dict(Counter(a['target_binding_review'] for a in rows)),
  'duplicate_assessment':dict(Counter(a['duplicate_review'] for a in rows)),
  'tags_quality':dict(Counter(a['tags_review'] for a in rows)),
  'vqa_quality':dict(Counter(a['vqa_review'] for a in rows)),
  'copy_quality':copy_counts,
 }
 # Distinct candidates after human-mediated duplicate resolution are only upper bound/hypothesis
 groups=[{'case_id':case,'candidate_ids':sorted(ids),'human_confirmed_duplicate_group':True,
         'canonical_instance_selected':False,'group_resolution_status':'REVIEW_REQUIRED'} for case,ids in DUP_GROUPS.items()]
 result={
  'schema':'prd32_human_review_diagnostic_analysis_v1',
  'scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',
  'review_status':gate['validation_status'],
  'human_review_attested_in_uploaded_file':True,
  'human_review_authenticity_not_independently_verified':True,
  'review_sha256':review_hash,
  'validation_errors':[],
  'images':3,'reviewed_prediction_rows':len(rows),'vqa_question_provided':False,
  'human_review_label_counts':counts,
  'service_runtime_counts_original':summary['runtime_counts'],
  'service_count_crosscheck':checks,
  'duplicate_groups':groups,
  'hypothetical_distinct_candidate_groups_if_duplicates_resolved':5,
  'actual_deduplicated_instances':0,
  'automatic_canonical_selection_performed':False,
  'raw_model_output_sha256_as_reported_by_service':{
    'gt_free_raw_predictions_sha256':summary['gt_free_raw_predictions_sha256'],
    'qwen_results_sha256':summary['qwen_results_sha256'],
  },
  'original_summary_says_human_review_incomplete': True,
  'interpretation':'Original service summary is an immutable pre-review snapshot; uploaded review attests completion. This analysis does not overwrite it.',
  'independent_accuracy_available':False,
  'publish_allowed':False,'official_PRD_acceptance':False,
  'per_instance_analysis':rows,
 }
 return result, rows

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--review',required=True);p.add_argument('--summary',required=True)
 p.add_argument('--extracted',required=True);p.add_argument('--output-dir',required=True)
 args=p.parse_args()
 report, rows=analyze(load(args.review),load(args.summary),load(args.extracted),sha(args.review))
 d=pathlib.Path(args.output_dir);d.mkdir(parents=True,exist_ok=True)
 (d/'PRD32_Diagnostic_Analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 cols=['case_id','instance_id','v5_category','qwen_category','target_binding_review','duplicate_review',
       'tags_review','vqa_review','copy_review','priority','service_copy_blocked','service_flags',
       'copy_debug_reference_detected','detected_issues','recommended_next_step','reviewer_note',
       'publish_allowed','official_PRD_acceptance']
 with (d/'PRD32_Instance_Issues.csv').open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');writer.writeheader()
  for r in rows:
   rec=r.copy();rec['detected_issues']='; '.join(rec['detected_issues']);writer.writerow(rec)
 print('AUDIT_OK 8/8 reviewed, original service counts reconciled, release/acceptance blocked')
 print('written:',d/'PRD32_Diagnostic_Analysis.json',d/'PRD32_Instance_Issues.csv')
if __name__=='__main__':main()
