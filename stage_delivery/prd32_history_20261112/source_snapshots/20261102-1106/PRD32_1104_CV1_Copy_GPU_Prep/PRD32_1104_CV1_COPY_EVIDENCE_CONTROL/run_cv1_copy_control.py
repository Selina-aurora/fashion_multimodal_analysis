#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Copy-only C-V1 controlled evidence instruction test for fixed DEV60/171.

Never reads review scores, GT, holdout, or attribute labels for inference.
Preflight checks legacy source, frozen input/marked-image hashes and Focus V1
prompt prior to model loading. Original Focus V1 outputs are strictly read-only.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,sys,time
from collections import Counter
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace')
VARIANT='focus_v1_copy_evidence_c_v1'
PINNED_LEGACY_SHA='98a833c429da4c39849942a81f1067239273258a8605eac9265d0aeb9defc6f3'
PINNED_SMOKE_SHA='f4037719bf42c64a98eca6cfda44591c34159bf7bd9186cca97282379000dd6e'
CV1_DELTA_EXPECTED_SHA='f8e64dce59b287610325d40bd883f28bfb1d4b33807a7e647bc3758aadce57c7'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
def th(s): return hashlib.sha256(s.encode('utf8')).hexdigest()
def readjl(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def readj(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def writej(p,obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def appendjl(p,obj):
 with Path(p).open('a',encoding='utf8') as f:
  f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n')
  f.flush();os.fsync(f.fileno())

def oldmod():
 p=HERE/'reference_dev60_qwen_ab.py'
 if sha(p)!=PINNED_LEGACY_SHA:raise ValueError('LEGACY_SOURCE_SHA_CHANGED')
 spec=importlib.util.spec_from_file_location('prd32_cv1_legacy_readonly',p)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m

def oldargs(args):
 return argparse.Namespace(raw=args.raw,input=args.input,source=args.source,
   focus_script=args.focus_script,model_dir=args.model_dir,
   snapshot=args.snapshot,focus=args.original_focus)

def check_static():
 manifest=readj(HERE/'fixed_manifest.json')
 rows=readjl(HERE/'instances171_input_only.jsonl')
 requests=readjl(HERE/'requests513_input_only.jsonl')
 smoke=readjl(HERE/'smoke24_input_only.jsonl')
 delta=readj(HERE/'cv1_copy_delta.json')
 for fn,expect in manifest['pinned_files'].items():
  if sha(HERE/fn)!=expect:raise ValueError('FROZEN_INPUT_HASH_CHANGED:'+fn)
 if sha(HERE/'smoke24_input_only.jsonl')!=PINNED_SMOKE_SHA:
  raise ValueError('SMOKE24_HASH_CHANGED')
 if sha(HERE/'cv1_copy_delta.json')!=CV1_DELTA_EXPECTED_SHA:
  raise ValueError('C_V1_DELTA_CHANGED_AFTER_FREEZE')
 assert len(rows)==171 and len(requests)==513 and len(smoke)==24
 keys=[(r['case_id'],r['target_instance_id']) for r in rows]
 if len(set(keys))!=171 or len({x['case_id'] for x in rows})!=60:
  raise ValueError('COHORT_NOT_60_IMAGES_171_INSTANCES')
 expected=[(r['case_id'],r['target_instance_id'],task) for r in rows for task in ('tags','vqa','copy')]
 got=[(x['case_id'],x['target_instance_id'],x['task']) for x in requests]
 if expected!=got or any(x['request_id']!='__'.join(k) for k,x in zip(expected,requests)):
  raise ValueError('FROZEN_REQUESTS_MISMATCH')
 smoke_keys=[(r['case_id'],r['target_instance_id']) for r in smoke]
 if len(set(smoke_keys))!=24 or not set(smoke_keys).issubset(keys):
  raise ValueError('SMOKE_NOT_24_FROZEN_TARGETS')
 forbidden=('gt','ground_truth','human_score','human_review','reviewer_score','confirmed_category','holdout')
 for entry in rows+requests+smoke:
  if any(f in entry for f in forbidden):raise ValueError('LABEL_FIELD_NOT_ALLOWED')
  if entry.get('human_reference_used',False) is not False:raise ValueError('HUMAN_REFERENCE_NOT_ALLOWED')
 if delta.get('variant')!='C-V1' or delta.get('task')!='copy' or delta.get('human_gt_used') is not False:
  raise ValueError('INVALID_CV1_DELTA_METADATA')
 suffix=delta.get('append_to_locked_focus_v1_copy_prompt')
 if not isinstance(suffix,str) or len(suffix)<100:raise ValueError('EMPTY_OR_TRUNCATED_CV1_DELTA')
 return manifest,rows,requests,smoke,delta

def check_runtime(args):
 manifest,rows,requests,smoke,delta=check_static()
 old=oldmod(); ori=oldargs(args)
 remote_rows,core,helper,focus=old.prompt_check(ori)
 if remote_rows!=rows:raise ValueError('REMOTE_COHORT_CHANGED')
 original=old.completed_run(args.original_focus,'focus_v1',old.expected_keys(rows),rows,ori)
 if old.CKPT!=manifest['V5_checkpoint_sha256'] or old.V5_HASH!=manifest['canonical_V5_predictions_sha256']:
  raise ValueError('V5_SOURCE_HASH_CHANGED')
 if old.PROMPT_SHA['focus_v1']!=manifest['focus_v1_original_prompt_sha256']:
  raise ValueError('ORIGINAL_FOCUS_PROMPT_HASH_CHANGED')
 oldmap={(r['case_id'],r['target_instance_id'],r['task']):r for r in original}
 if len(oldmap)!=513:raise ValueError('ORIGINAL_FOCUS_NOT_COMPLETE_513')
 for row in rows:
  key=(row['case_id'],row['target_instance_id'],'copy')
  b=oldmap[key]
  if b['image_sha256']!=row['image_sha256']:raise ValueError('ORIGINAL_IMAGE_SHA_CHANGED')
  marked=Path(b['marked_image'])
  if not marked.is_file() or sha(marked)!=b['marked_image_sha256']:
   raise ValueError('ORIGINAL_MARKED_IMAGE_SHA_CHANGED:'+str(marked))
 return old,rows,core,helper,focus,oldmap,smoke,delta

def ordered_keys(scope,rows,smoke):
 wanted={(s['case_id'],s['target_instance_id']) for s in smoke}
 return [(r['case_id'],r['target_instance_id'],'copy') for r in rows
         if scope=='full' or (r['case_id'],r['target_instance_id']) in wanted]

def outdir(args):return args.smoke_output if args.scope=='smoke' else args.full_output

def provenance(args,old,delta):
 return {'schema':'prd32_copy_only_cv1_run_v1','scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',
   'model_variant':VARIANT,'run_scope':args.scope,'target_instances_total':171,
   'new_copy_requests':24 if args.scope=='smoke' else 171,
   'same_original_focus_marked_image':True,'same_v5_predictions':True,
   'original_focus_prompt_sha256':old.PROMPT_SHA['focus_v1']['copy'],
   'original_focus_results_sha256':sha(args.original_focus/'qwen_results.jsonl'),
   'canonical_171_sha256':sha(HERE/'instances171_input_only.jsonl'),
   'canonical_513_sha256':sha(HERE/'requests513_input_only.jsonl'),
   'smoke24_sha256':PINNED_SMOKE_SHA,'cv1_delta_sha256':sha(HERE/'cv1_copy_delta.json'),
   'source_code_sha256':sha(__file__),'task_limits':{'tags':170,'vqa':128,'copy':120},
   'human_labels_used_for_model':False,'holdout_used':False,
   'new_results_human_verified':False,'official_prd_acceptance':False}

def check_prior(prior,keys,oldmap,old,delta):
 if len(prior)>len(keys):raise ValueError('RESUME_EXTRA_ROWS')
 suffix=delta['append_to_locked_focus_v1_copy_prompt']
 for r,k in zip(prior,keys):
  b=oldmap[k]
  if (r.get('request_id')!='__'.join(k) or r.get('model_variant')!=VARIANT
     or r.get('status')!='GENERATED_REVIEW_REQUIRED' or r.get('task')!='copy'):
   raise ValueError('RESUME_ID_OR_TASK_MISMATCH')
  if r.get('image_sha256')!=b['image_sha256'] or r.get('marked_image_sha256')!=b['marked_image_sha256']:
   raise ValueError('RESUME_IMAGE_CHANGED')
  if r.get('base_prompt_sha256')!=old.PROMPT_SHA['focus_v1']['copy']:
   raise ValueError('RESUME_BASE_PROMPT_CHANGED')
  if th(r.get('original_prompt',''))!=old.PROMPT_SHA['focus_v1']['copy']:
   raise ValueError('RESUME_ORIGINAL_PROMPT_TEXT_CHANGED')
  if r.get('final_prompt_sha256')!=th(r.get('original_prompt','')+suffix):
   raise ValueError('RESUME_FINAL_PROMPT_CHANGED')
  if not isinstance(r.get('raw_text'),str) or not r['raw_text'].strip():
   raise ValueError('RESUME_BAD_RAW_TEXT')
  if r.get('human_label_or_gt_bbox_used') is not False:raise ValueError('RESUME_LABEL_LEAK')

def progress(out,scope,n,total,done,load_s=None):
 writej(out/'run_summary.json',{'schema':'prd32_cv1_run_summary_v1','scope':scope,
   'status':('COMPLETE_FOR_REVIEW' if done else 'INCOMPLETE_RESUMABLE'),
   'successful':n,'expected':total,'remaining':total-n,
   'model_load_seconds':load_s,'human_verified':False,'official_prd_acceptance':False})

def run(args):
 old,rows,core,helper,focus,oldmap,smoke,delta=check_runtime(args)
 keys=ordered_keys(args.scope,rows,smoke)
 out=outdir(args);prov=provenance(args,old,delta)
 if args.scope=='full':
  s=args.smoke_output/'run_summary.json'
  if not s.exists() or readj(s).get('status')!='COMPLETE_FOR_REVIEW' or readj(s).get('successful')!=24:
   raise ValueError('SMOKE24_REQUIRED_BEFORE_FULL')
 if out.exists():
  if not args.resume:raise FileExistsError('RUN_OUTPUT_EXISTS_USE_RESUME')
  if readj(out/'provenance.json')!=prov:raise ValueError('RESUME_PROVENANCE_CHANGED')
  prior=readjl(out/'qwen_results.jsonl');check_prior(prior,keys,oldmap,old,delta)
 else:
  if args.resume:raise ValueError('RESUME_DIRECTORY_NOT_FOUND')
  out.mkdir(parents=True);writej(out/'provenance.json',prov)
  (out/'qwen_results.jsonl').touch();prior=[]
 if len(prior)==len(keys):
  progress(out,args.scope,len(prior),len(keys),True)
  print('CV1_ALREADY_COMPLETE',args.scope,len(prior));return
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 import torch
 if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
 if torch.cuda.mem_get_info()[0]/1024**3<18:raise RuntimeError('GPU_FREE_BELOW_18_GiB')
 from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
 dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
 start=time.perf_counter()
 model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.model_dir),local_files_only=True,
   device_map='auto',torch_dtype=dtype,attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
 processor=AutoProcessor.from_pretrained(str(args.model_dir),local_files_only=True,max_pixels=512*28*28)
 load_s=round(time.perf_counter()-start,2)
 rmap={(r['case_id'],r['target_instance_id']):r for r in rows}
 suffix=delta['append_to_locked_focus_v1_copy_prompt']
 for n in range(len(prior),len(keys)):
  k=keys[n];cid,iid,task=k;row=rmap[cid,iid];base=oldmap[k]
  qrec={'image_path':row['image_path'],'image_sha256':row['image_sha256'],
    'target_bbox_xyxy':row['target_bbox_xyxy'],'target_instance_id':iid,
    'task':'copy','query':old.QUESTION,'length_chars':[20,80],
    'predicted_attribute_context':[]}
  prompt=core.build_prompt(helper,focus,qrec,'copy','focus_v1')
  if th(prompt)!=old.PROMPT_SHA['focus_v1']['copy']:
   raise ValueError('FOCUS_PROMPT_CHANGED_DURING_RUN')
  marked=Path(base['marked_image'])
  if sha(marked)!=base['marked_image_sha256']:
   raise ValueError('MARKED_IMAGE_CHANGED_DURING_RUN')
  rec={'case_id':cid,'target_instance_id':iid,'task':'copy','request_id':'__'.join(k),
    'status':'GENERATED_REVIEW_REQUIRED','model_variant':VARIANT,
    'image_sha256':row['image_sha256'],'v5_predicted_bbox_xyxy':row['target_bbox_xyxy'],
    'v5_category_unverified':row['predicted_category_v5_unverified'],
    'marked_image':str(marked),'marked_image_sha256':base['marked_image_sha256'],
    'base_prompt_sha256':th(prompt),'final_prompt_sha256':th(prompt+suffix),
    'delta_sha256':th(suffix),'original_prompt':prompt,
    'visual_change_only':False,'prompt_change_only':True,
    'human_label_or_gt_bbox_used':False,'quality_human_verified':False,
    'official_prd_acceptance':False}
  try:
   raw,timing=helper.run_model_case(model,processor,qrec,marked,prompt+suffix,
      task_limits={'tags':170,'vqa':128,'copy':120})
   if not isinstance(raw,str) or not raw.strip():raise ValueError('EMPTY_MODEL_RESPONSE')
   rec.update(raw_text=raw,timing=timing,
     copy_review_flags=['DEBUG_MARKER_IN_COPY'] if old.DEBUG.search(raw) else [])
   appendjl(out/'qwen_results.jsonl',rec);prior.append(rec)
   print('CV1_COPY',args.scope,n+1,'/',len(keys),rec['request_id'],'OK',flush=True)
   if (n+1)%12==0:progress(out,args.scope,len(prior),len(keys),False,load_s)
  except Exception as exc:
   writej(out/'last_error.json',{'request_index':n+1,'request_id':rec['request_id'],
     'error_type':type(exc).__name__,'message':str(exc)[:400]})
   progress(out,args.scope,len(prior),len(keys),False,load_s)
   raise
 progress(out,args.scope,len(prior),len(keys),True,load_s)
 print('CV1_RUN_COMPLETE',args.scope,len(prior),'/',len(keys))

def export(args):
 old,rows,core,helper,focus,oldmap,smoke,delta=check_runtime(args)
 keys=ordered_keys(args.scope,rows,smoke);out=outdir(args)
 if readj(out/'run_summary.json').get('status')!='COMPLETE_FOR_REVIEW':
  raise ValueError('RUN_NOT_COMPLETE')
 current=readjl(out/'qwen_results.jsonl')
 if len(current)!=len(keys):raise ValueError('RUN_RESULT_COUNT_WRONG')
 check_prior(current,keys,oldmap,old,delta)
 if args.export.exists():raise FileExistsError('REFUSE_OVERWRITE_EXPORT')
 paired=[];counts=Counter()
 for key,new in zip(keys,current):
  oldcopy=oldmap[key]
  same=new['raw_text']==oldcopy['raw_text'];counts['unchanged' if same else 'changed']+=1
  paired.append({'request_id':'__'.join(key),'case_id':key[0],
   'target_instance_id':key[1],'task':'copy',
   'focus_v1_raw':oldcopy['raw_text'],'c_v1_raw':new['raw_text'],
   'text_unchanged':same,'marked_image_sha256':new['marked_image_sha256'],
   'base_prompt_sha256':new['base_prompt_sha256'],
   'c_v1_final_prompt_sha256':new['final_prompt_sha256'],
   'new_score':'NOT_HUMAN_REVIEWED'})
 with ZipFile(args.export,'w',ZIP_DEFLATED,compresslevel=6) as z:
  for fn in ('provenance.json','run_summary.json','qwen_results.jsonl'):
   z.write(out/fn,arcname=f'{out.name}/{fn}')
  for fn in ('fixed_manifest.json','cv1_copy_delta.json'):
   z.write(HERE/fn,arcname=fn)
  z.writestr('paired_copy_for_review.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in paired))
  z.writestr('paired_text_diagnostic_NOT_ACCURACY.json',json.dumps({
   'scope':args.scope,'copy_requests':len(keys),'counts_TEXT_ONLY_NOT_ACCURACY':dict(counts),
   'new_results_human_review_required':True,'official_prd_acceptance':False,
   'do_not_infer_improvement_without_human_scoring':True},ensure_ascii=False,indent=2))
 print('CV1_UPLOAD_THIS',args.export,round(args.export.stat().st_size/1024**2,3),'MiB')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('command',choices=['check-static','preflight','run','export'])
 p.add_argument('--scope',choices=['smoke','full'],default='smoke')
 p.add_argument('--resume',action='store_true')
 p.add_argument('--raw',type=Path,default=ROOT/'PRD32_DEV60_V5_REAL_20261030_run01')
 p.add_argument('--input',type=Path,default=ROOT/'PRD32_FIXED_DEV_V2_60IMG_CANDIDATES_20261030/v5_image_only_inputs_60.jsonl')
 p.add_argument('--source',type=Path,default=ROOT/'PRD32_20261029_NextBatch_GTFree')
 p.add_argument('--focus-script',type=Path,default=ROOT/'PRD32_FOCUS_AB_PATCH_20261029/prd32_focus_ab.py')
 p.add_argument('--model-dir',type=Path,default=ROOT/'models/Qwen2.5-VL-7B-Instruct')
 p.add_argument('--snapshot',type=Path,default=ROOT/'PRD32_DEV60_QWEN_DIAGNOSTIC_SNAPSHOT_20261030')
 p.add_argument('--original-focus',type=Path,default=ROOT/'PRD32_DEV60_QWEN_FOCUS_V1_20261030_run01')
 p.add_argument('--smoke-output',type=Path,default=ROOT/'PRD32_1104_CV1_SMOKE_RUN01')
 p.add_argument('--full-output',type=Path,default=ROOT/'PRD32_1104_CV1_FULL_RUN01')
 p.add_argument('--export',type=Path)
 args=p.parse_args()
 if args.export is None:args.export=ROOT/f'PRD32_1104_CV1_{args.scope.upper()}_UPLOAD_THIS.zip'
 try:
  if args.command=='check-static':
   manifest,rows,req,smoke,delta=check_static()
   print('CV1_STATIC_60_IMAGES_171_INSTANCES_513_ORIGINAL_24_SMOKE_PASS')
   print('CV1_COPY_ONLY_171_REQUESTS_NO_GT_OR_HUMAN_LABELS')
  elif args.command=='preflight':
   check_runtime(args)
   print('CV1_RUNTIME_PREFLIGHT_PASS 60 images 171 instances; 24 smoke Copy requests')
   print('ORIGINAL_MARKED_IMAGES_AND_PROMPT_LOCKED; ONLY_COPY_SUFFIX_CHANGED; GPU_NOT_LOADED')
  elif args.command=='run':run(args)
  else:export(args)
 except Exception as exc:
  print('CV1_BLOCKED',type(exc).__name__,str(exc),file=sys.stderr)
  raise SystemExit(2)
if __name__=='__main__':main()
