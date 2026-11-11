#!/usr/bin/env python3
# coding: utf-8
"""Local paired objective diagnostics for candidate Qwen output; NOT a correctness evaluator."""
from pathlib import Path
import argparse,csv,hashlib,json,re,collections
BASE=Path(__file__).resolve().parent
INPUTS=BASE.parent/'02_GPU候选输入方案_不含人工评分'
DEBUG=re.compile(r'红框|框内|标注框|bounding\s+box|\bbbox\b|红色边框')
TASKS=('tags','vqa','copy')
def readjl(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate-results',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
 a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 old=json.loads((BASE/'11.03_历史用户确认评分_只读.json').read_text(encoding='utf8'))
 oldmap={f'{r["case_id"]}__{r["instance_id"]}__{t}':(r,t) for r in old['records'] for t in TASKS}
 pinned=readjl(INPUTS/'requests513_input_only.jsonl');instances=readjl(INPUTS/'instances171_input_only.jsonl')
 insmap={(x['case_id'],x['target_instance_id']):x for x in instances}
 cand=readjl(a.candidate_results)
 if len(cand)!=513 or len({x.get('request_id') for x in cand})!=513:raise ValueError('CANDIDATE_RESULT_NOT_513_UNIQUE')
 if [x['request_id'] for x in cand]!=[x['request_id'] for x in pinned]:raise ValueError('CANDIDATE_REQUEST_KEYS_OR_ORDER_CHANGED')
 rows=[];changes=collections.Counter()
 for x,req in zip(cand,pinned):
  key=req['request_id']; oldr,t=oldmap[key]; rec=insmap[(req['case_id'],req['target_instance_id'])]
  if x.get('case_id')!=req['case_id'] or x.get('target_instance_id')!=req['target_instance_id'] or x.get('task')!=t:raise ValueError('INSTANCE_BINDING_KEY_ERROR:'+key)
  if x.get('image_sha256')!=rec['image_sha256']:raise ValueError('SOURCE_SHA_MISMATCH:'+key)
  if x.get('v5_predicted_bbox_xyxy')!=rec['target_bbox_xyxy']:raise ValueError('V5_BBOX_CHANGED:'+key)
  if x.get('human_label_or_gt_bbox_used') is not False:raise ValueError('HUMAN_LABEL_LEAK:'+key)
  if x.get('status')!='GENERATED_REVIEW_REQUIRED':raise ValueError('INCOMPLETE_GENERATION:'+key)
  raw=x.get('raw_text')
  if not isinstance(raw,str) or not raw.strip():raise ValueError('EMPTY_MODEL_OUTPUT:'+key)
  ref=oldr['tasks'][t]['original_text']
  changed=(raw!=ref);changes[t]+=int(changed)
  rows.append({'request_id':key,'instance_id':oldr['case_id']+'/'+oldr['instance_id'],'task':t,
   'text_changed':changed,'original_focus_score':oldr['tasks'][t]['original_human_attested_score'],
   'candidate_score':'REVIEW_REQUIRED','candidate_copy_debug_heuristic':bool(DEBUG.search(raw)) if t=='copy' else False,
   'original_focus_text':ref,'candidate_text':raw,'review_notes':''})
 with (a.output/'candidate_513逐条核查模板.csv').open('w',encoding='utf-8-sig',newline='') as fp:
  w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (a.output/'objective_summary.json').write_text(json.dumps({'status':'MATCHED_513_COMPLETE_NEEDS_HUMAN_SCORING','source_cohort':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',
   'instances':171,'requests':513,'changed_text_by_task':dict(changes),'accuracy_not_calculated':True,'note':'Text differences and debug heuristics are NOT quality judgements.'},ensure_ascii=False,indent=2),encoding='utf8')
 print('CANDIDATE_513_OBJECTIVE_GATE_PASS; MANUAL_REVIEW_REQUIRED',dict(changes))
if __name__=='__main__':main()
