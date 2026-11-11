#!/usr/bin/env python3
# coding: utf-8
"""Offline gate for fixed 60-image, 171-instance, 513-request candidate prompts.
No GPU, no human scores, no GT read. This is not a Qwen runner.
"""
from pathlib import Path
import argparse, hashlib, json, sys
BASE=Path(__file__).resolve().parent
TASKS=('tags','vqa','copy')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readjl(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def check():
 f=json.loads((BASE/'fixed_manifest.json').read_text())
 for n,h in f['pinned_files'].items():
  if sha(BASE/n)!=h:raise RuntimeError('PIN_MISMATCH:'+n)
 ins=readjl(BASE/'instances171_input_only.jsonl');req=readjl(BASE/'requests513_input_only.jsonl')
 if len(ins)!=171 or len(req)!=513:raise RuntimeError('COHORT_SIZE_CHANGED')
 if len({x['image_sha256'] for x in ins})!=60 or len({(x['case_id'],x['target_instance_id']) for x in ins})!=171:raise RuntimeError('IMAGE_OR_INSTANCE_ID_CHANGED')
 if len({x['request_id'] for x in req})!=513:raise RuntimeError('DUPLICATE_REQUEST')
 exp=[(r['case_id'],r['target_instance_id'],t) for r in ins for t in TASKS]
 got=[(r['case_id'],r['target_instance_id'],r['task']) for r in req]
 if exp!=got:raise RuntimeError('REQUEST_ORDER_OR_TASK_CHANGED')
 for r in ins:
  if r.get('human_reference_used') is not False or r.get('scope')!='TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY':raise RuntimeError('LABEL_OR_SCOPE_UNSAFE')
  if not r['image_path'].startswith('/workspace/fashion_data/raw/train/') or any(z in r['image_path'].lower() for z in ['holdout','sealed','validation']):raise RuntimeError('RESTRICTED_IMAGE_PATH')
  if 'gt' in r or 'human_gt' in r:raise RuntimeError('GT_IN_INPUT')
 print('FIXED_DEV60_171_513_INPUT_GATE_PASS')
 return ins,req
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['ID-V1','S-V1'],required=True);ap.add_argument('--out',type=Path,default=BASE/'candidate_prompt_delta.jsonl')
 a=ap.parse_args();ins,req=check()
 addenda=json.loads((BASE/'candidate_addenda.json').read_text());suffix=addenda[a.variant]
 # Each request is appended to ORIGINAL FocusV1 prompt. Do not include V5 label or any human label.
 lines=[]
 for r in req:
  label=r['target_instance_id']; text=suffix.format(instance_id=label)
  key=r['request_id']
  lines.append({'request_id':key,'case_id':r['case_id'],'target_instance_id':label,'task':r['task'],'variant':a.variant,
   'append_to_original_focus_v1_prompt':'\n\n'+text,'suffix_sha256':hashlib.sha256(text.encode()).hexdigest(),
   'human_reference_used':False})
 if a.out.exists():raise FileExistsError('NO_OVERWRITE_EXISTING:'+str(a.out))
 a.out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in lines),encoding='utf8')
 print('PROMPT_DELTA_PREPARED',a.variant,len(lines),'SAVED',a.out)
if __name__=='__main__':main()
