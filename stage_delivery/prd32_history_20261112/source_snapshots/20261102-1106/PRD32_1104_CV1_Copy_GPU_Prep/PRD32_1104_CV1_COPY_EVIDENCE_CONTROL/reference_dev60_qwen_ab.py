#!/usr/bin/env python3
# coding: utf-8
"""PRD 3.2 DEV60 diagnostic snapshot, paired Qwen Baseline/Focus V1.

Never reads holdout/GT; never promotes to PRD acceptance. The original 12-image
paired code is imported ONLY for its locked prompt-generation methods.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, os, re, statistics, sys, time
from pathlib import Path
from collections import Counter, defaultdict

TASKS=('tags','vqa','copy')
QUESTION='红框内目标服饰的主要可见颜色是什么？'
SCOPED='TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY'
CKPT='c5ba795225349b1c521aac0517ac7c493d9818dcc17d558ac4a3f6f8669452f0'
DEV60_INPUT='c629ff0ca557a0c57c7b97c93cf389502f9660de63a934463cb1f5e8e722752f'
V5_HASH='688cb801daebd0484e8740c877c021eca86c83f362857d4fc982d02fb50f2710'
V5_LEGACY_MANIFEST='22c6bd468df6795920eac592209129228fb1a91cc219486acdcf2685cc1f27ee'
SOURCE_EXPECTED={
'v5_to_qwen_bridge.py':'affee7342646d808b4f0214f135d60d2d2611db920655c5e064268092e2dd02d',
'next_batch.py':'08b35135223b49e2a2a7bbb448cdad3a4ce90a6a1373e57af29d1119102489f6',
'prd32_guard.py':'deeaaea28d9f5e80805da013ef23689fc0df95ee8a6514c583beb468b766fc0f',
'frozen_helpers/prd32_integrated_pipeline_20261027.py':'90dda6974dc533aaae30fe1343095864ee74d536713ad80ce2006617286fa6b4'}
FOCUS_SHA='2004c73ae5bf5441bacc5ebc10c630f4d5368dbf990bcf3443596e7f1d3a91a8'
CORE_SHA='fdf1c55583b75f4692ddad07d459ceabff7a337d4f82d90e71a5c908f1323d2f'
PROMPT_SHA={
 'baseline':{'tags':'4821bc893d412cffa3a7f8c13709d09b9f3ded05ccfb2a12b72be9298bad6e2b','vqa':'5e288115687cf5a18e4fa049e637716f04e1070c9f02615cdbe9cf7cc8289470','copy':'3d685428b8840d0a0353da711074ddcb0bf3b4da2ae19b6d3c89c41bea10b4aa'},
 'focus_v1':{'tags':'a36a818b9b30792d83a0970f7e7c8d5f16a6e409d1f46dd917878c3057c3b92a','vqa':'9557d90c5baa484f96be718ff1575aaea93145752de1310a7bdc04bd6e106ccd','copy':'c0373dfa0bb16f634d7944fe756ef692a64e404ceaaab1f7b0ee2f9fd8a63193'}}
CLASS=('top','pants','skirt','outerwear','dress','shoe','bag','accessory')
RESTRICTED={'holdout','sealed','fullval','val','valid','validation','test','testing','testset','eval','evaluation','devtest'}
DEBUG=re.compile(r'红框|红色.{0,2}(?:框|矩形|边框)|框内|标注框|框选|画框|方框|\bbbox\b|bounding\s+box',re.I)
ABSTAIN=re.compile(r'无法确定|无法判断|看不清|无法辨认|不能确定|不确定')
ROOT=Path('/workspace')
DEFAULT_RAW=ROOT/'PRD32_DEV60_V5_REAL_20261030_run01'
DEFAULT_INPUT=ROOT/'PRD32_FIXED_DEV_V2_60IMG_CANDIDATES_20261030/v5_image_only_inputs_60.jsonl'
DEFAULT_SOURCE=ROOT/'PRD32_20261029_NextBatch_GTFree'
DEFAULT_FOCUS=ROOT/'PRD32_FOCUS_AB_PATCH_20261029/prd32_focus_ab.py'
DEFAULT_MODEL=ROOT/'models/Qwen2.5-VL-7B-Instruct'
DEFAULT_SNAP=ROOT/'PRD32_DEV60_QWEN_DIAGNOSTIC_SNAPSHOT_20261030'
DEFAULT_BASE=ROOT/'PRD32_DEV60_QWEN_BASELINE_20261030_run01'
DEFAULT_FOCUS_RUN=ROOT/'PRD32_DEV60_QWEN_FOCUS_V1_20261030_run01'
DEFAULT_COMPARE=ROOT/'PRD32_DEV60_QWEN_AB_COMPARE_20261030_run01'


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()

def tsha(t):return hashlib.sha256(t.encode('utf8')).hexdigest()
def loadj(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def loadjl(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def savej(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def savejl(p,ls):Path(p).write_text(''.join(json.dumps(d,ensure_ascii=False,allow_nan=False)+'\n' for d in ls),encoding='utf8')
def safe_path(p,root=Path('/workspace/fashion_data'),require=True):
 p=Path(p)
 if not p.is_absolute() or any(x.lower() in RESTRICTED for x in p.parts):raise ValueError('RESTRICTED_OR_RELATIVE_IMAGE')
 p=p.resolve(strict=False);r=root.resolve(strict=False)
 if not p.is_relative_to(r) or 'train' not in [v.lower() for v in p.parts]:raise ValueError('NONTRAIN_IMAGE')
 if p.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'}:raise ValueError('BAD_IMAGE_TYPE')
 if require and not p.is_file():raise FileNotFoundError(p)
 return p

def oldcore(p):
 p=Path(p)
 if sha(p)!=CORE_SHA:raise ValueError('ORIGINAL_12_PROMPT_CORE_CHANGED')
 spec=importlib.util.spec_from_file_location('prd32_1030_original12_prompt_core',p)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 return mod

def source_pins(source,focus):
 for rel,d in SOURCE_EXPECTED.items():
  f=source/rel
  if not f.is_file() or sha(f)!=d:raise ValueError('QWEN_SOURCE_CHANGED:'+rel)
 if not focus.is_file() or sha(focus)!=FOCUS_SHA:raise ValueError('FOCUS_SCRIPT_CHANGED')

def read_and_validate_raw(raw,input_file,check_images=True):
 if sha(input_file)!=DEV60_INPUT:raise ValueError('DEV60_INPUT_SHA_CHANGED')
 inputs=loadjl(input_file)
 if len(inputs)!=60:raise ValueError('EXPECTED_60_INPUTS')
 summary=loadj(raw/'v5_real_summary.json');run=loadj(raw/'run_manifest.json')
 if summary.get('status')!='V5_RAW_INFERENCE_COMPLETED' or (summary.get('n_images'),summary.get('n_predicted_instances'))!=(60,171):raise ValueError('V5_SUMMARY_CHANGED')
 if summary.get('checkpoint_sha256')!=CKPT or run.get('checkpoint_sha256')!=CKPT:raise ValueError('V5_CHECKPOINT_CHANGED')
 if run.get('score_threshold')!=.4 or run.get('mask_threshold')!=.5 or run.get('human_gt_used_for_inference') is not False:raise ValueError('V5_PROVENANCE_UNSAFE')
 if run.get('input_manifest_sha256')!=V5_LEGACY_MANIFEST:raise ValueError('V5_INPUT_MANIFEST_CHANGED')
 if summary.get('v5_predictions_sha256')!=V5_HASH or sha(raw/'v5_raw_predictions.jsonl')!=V5_HASH:raise ValueError('V5_OUTPUT_SHA_CHANGED')
 preds=loadjl(raw/'v5_raw_predictions.jsonl')
 if len(preds)!=60:raise ValueError('V5_IMAGE_COUNT_WRONG')
 result=[];visited=set();ct=Counter();marked=set(); suspicious=[]
 for ix,(inp,pred) in enumerate(zip(inputs,preds),1):
  case=f'fresh_dev_{ix:03d}'
  if inp.get('case_id')!=case or pred.get('case_id')!=case or pred.get('image_sha256')!=inp.get('image_sha256'):raise ValueError('IMAGE_ORDER_OR_SHA_CHANGED:'+case)
  if set(inp)!={'case_id','image_path','image_sha256','do_not_use_historical_gt_bbox','human_reference_withheld'} or inp['human_reference_withheld'] is not True or inp['do_not_use_historical_gt_bbox'] is not True:raise ValueError('GT_FIELD_LEAK')
  img=safe_path(inp['image_path'],require=check_images)
  if pred.get('source_image_path')!=str(img):raise ValueError('IMAGE_PATH_MISMATCH')
  if check_images and sha(img)!=inp['image_sha256']:raise ValueError('IMAGE_BYTES_CHANGED:'+case)
  if inp['image_sha256'] in visited:raise ValueError('DUPLICATE_IMAGE_SHA')
  visited.add(inp['image_sha256']);w,h=pred['image_size']
  ins=pred['instances']
  if len(ins)!=pred['instance_count']:raise ValueError('V5_INSTANCE_COUNT_MISMATCH')
  for j,it in enumerate(ins):
   iid=f'pred_{j:04d}'
   if it.get('instance_id')!=iid or it.get('predicted_category') not in CLASS:raise ValueError('BAD_INSTANCE_ID_OR_CLASS')
   b=it.get('bbox_xyxy')
   if not isinstance(b,list) or len(b)!=4 or any(type(x) not in (int,float) or not math.isfinite(x) for x in b) or not (0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h):raise ValueError('BAD_V5_BBOX')
   if it.get('bbox_coordinate_space')!='original_image' or it.get('prediction_origin')!='V5_CHECKPOINT_LIVE_MODEL_OUTPUT_NOT_GT' or not(.4<=it.get('score',-1)<=1):raise ValueError('BAD_V5_INSTANCE_ORIGIN_OR_SCORE')
   row={'case_id':case,'image_path':str(img),'image_sha256':inp['image_sha256'],'target_instance_id':iid,
    'target_bbox_xyxy':b,'target_bbox_origin':'prd31_v5_live_prediction',
    'predicted_category_v5_unverified':it['predicted_category'],'predicted_score_v5_uncalibrated':it['score'],
    'human_reference_used':False,'scope':SCOPED}
   result.append(row);ct[it['predicted_category']]+=1
  # bbox-only duplicate candidates; never automatically merge
  for a in range(len(ins)):
   for b in range(a+1,len(ins)):
    x=ins[a]['bbox_xyxy'];y=ins[b]['bbox_xyxy'];ww=max(0,min(x[2],y[2])-max(x[0],y[0]));hh=max(0,min(x[3],y[3])-max(x[1],y[1]));
    iou=ww*hh/((x[2]-x[0])*(x[3]-x[1])+(y[2]-y[0])*(y[3]-y[1])-ww*hh)
    if iou>=.8:suspicious.append({'case_id':case,'a':ins[a]['instance_id'],'b':ins[b]['instance_id'],'bbox_iou':round(iou,5),'flag':'REVIEW_ONLY_NOT_AUTO_DEDUP'})
 if len(result)!=171 or sum(ct.values())!=171:raise ValueError('EXPECTED_171_V5_INSTANCES')
 if any(ct[c]<1 for c in CLASS):raise ValueError('EIGHT_PREDICTED_CLASSES_NOT_COVERED')
 if len(set((r['case_id'],r['target_instance_id']) for r in result))!=171:raise ValueError('DUPLICATE_INSTANCE_KEY')
 return result,dict(ct),suspicious

def make_snapshot(args):
 if args.snapshot.exists():raise FileExistsError('SNAPSHOT_EXISTS_NO_OVERWRITE')
 source_pins(args.source,args.focus_script)
 rows,cats,dups=read_and_validate_raw(args.raw,args.input)
 req=[{'request_id':f'{r["case_id"]}__{r["target_instance_id"]}__{t}',
       'case_id':r['case_id'],'target_instance_id':r['target_instance_id'],'task':t,
       'vqa_question':QUESTION if t=='vqa' else None} for r in rows for t in TASKS]
 args.snapshot.mkdir(parents=True)
 savejl(args.snapshot/'instances.jsonl',rows)
 savejl(args.snapshot/'requests.jsonl',req)
 report={'schema':'prd32_dev60_qwen_diagnostic_snapshot_v1','scope':SCOPED,'status':'DEV60_QWEN_INPUT_SNAPSHOT_READY_NOT_OFFICIAL_FREEZE',
  'images':60,'instances':171,'requests_per_variant':513,'tasks':list(TASKS),
  'image_input_sha256':DEV60_INPUT,'v5_sha256':V5_HASH,'checkpoint_sha256':CKPT,
  'instance_snapshot_sha256':sha(args.snapshot/'instances.jsonl'),'request_snapshot_sha256':sha(args.snapshot/'requests.jsonl'),
  'source_script_hashes':SOURCE_EXPECTED,'focus_script_sha256':FOCUS_SHA,
  'prompt_hashes':PROMPT_SHA,'question_text':QUESTION,'question_sha256':tsha(QUESTION),
  'categories_V5_NOT_GT':cats,'high_bbox_iou_pairs':dups,
  'holdout_used':False,'human_label_gt_used':False,'official_v2_freeze':False,'official_prd_acceptance':False,'publish_allowed':False}
 savej(args.snapshot/'snapshot.json',report)
 print('DEV60_SNAPSHOT_READY',json.dumps({'images':60,'instances':171,'requests_per_variant':513,'high_bbox_iou_pairs':len(dups)},ensure_ascii=False))

def validate_snapshot(args,verify_images=True):
 f=loadj(args.snapshot/'snapshot.json')
 if f.get('status')!='DEV60_QWEN_INPUT_SNAPSHOT_READY_NOT_OFFICIAL_FREEZE' or f.get('scope')!=SCOPED or f.get('official_v2_freeze') is not False:raise ValueError('UNSAFE_SNAPSHOT')
 source_pins(args.source,args.focus_script)
 rows,cats,dups=read_and_validate_raw(args.raw,args.input,check_images=verify_images)
 if sha(args.snapshot/'instances.jsonl')!=f.get('instance_snapshot_sha256') or loadjl(args.snapshot/'instances.jsonl')!=rows:raise ValueError('SNAPSHOT_INSTANCES_CHANGED')
 if sha(args.snapshot/'requests.jsonl')!=f.get('request_snapshot_sha256'):raise ValueError('SNAPSHOT_REQUESTS_CHANGED')
 req=loadjl(args.snapshot/'requests.jsonl')
 expected=[f'{r["case_id"]}__{r["target_instance_id"]}__{t}' for r in rows for t in TASKS]
 if len(req)!=513 or [r['request_id'] for r in req]!=expected:raise ValueError('REQUEST_IDENTITIES_CHANGED')
 return f,rows

def prompt_check(args):
 _,rows=validate_snapshot(args)
 core=oldcore(Path(__file__).with_name('reference_12img_runner_READ_ONLY.py'))
 bridge,helper,focus=core.load_runtime(args.source,args.focus_script)
 ps={v:{t:tsha(core.build_prompt(helper,focus,{},t,v)) for t in TASKS} for v in ('baseline','focus_v1')}
 if ps!=PROMPT_SHA:raise ValueError('PROMPT_SHA_DIFFERENT_FROM_12IMG_V1')
 print('DEV60_AB_PREFLIGHT_PASS',len(set(r['case_id'] for r in rows)),'images',len(rows),'instances',len(rows)*3,'requests/variant')
 print('PROMPT_LOCKED_SAME_AS_12IMG_V1; NOT_PRD_ACCEPTANCE')
 return rows,core,helper,focus

def expected_keys(rows):return [(r['case_id'],r['target_instance_id'],t) for r in rows for t in TASKS]

def check_prefix(lines,keys,rows,variant):
 if len(lines)>len(keys):raise ValueError('RESULT_OVERLONG')
 inst={(r['case_id'],r['target_instance_id']):r for r in rows}
 for n,x in enumerate(lines):
  k=(x.get('case_id'),x.get('target_instance_id'),x.get('task'))
  if k!=keys[n] or x.get('request_id')!='__'.join(k) or x.get('status')!='GENERATED_REVIEW_REQUIRED' or x.get('model_variant')!=variant:raise ValueError('RESUME_PREFIX_BAD')
  f=inst[k[:2]]
  if x.get('image_sha256')!=f['image_sha256'] or x.get('v5_predicted_bbox_xyxy')!=f['target_bbox_xyxy'] or x.get('v5_category_unverified')!=f['predicted_category_v5_unverified'] or x.get('human_label_or_gt_bbox_used') is not False:raise ValueError('RESULT_INPUT_CHANGED')
  if x.get('prompt_sha256')!=PROMPT_SHA[variant][k[2]] or not isinstance(x.get('raw_text'),str) or not x['raw_text'].strip():raise ValueError('INVALID_RESULT_TEXT_OR_PROMPT_SHA')
  if x.get('publish_allowed') is not False:raise ValueError('PUBLISH_FLAG_UNSAFE')
  if not re.fullmatch('[0-9a-f]{64}',str(x.get('marked_image_sha256',''))):raise ValueError('MISSING_MARKED_IMAGE_SHA')
  marked=Path(x.get('marked_image',''))
  if not marked.is_file() or sha(marked)!=x['marked_image_sha256']:raise ValueError('MARKED_INPUT_MISSING_OR_CHANGED')
  if k[2]=='vqa' and x.get('question_sha256')!=tsha(QUESTION):raise ValueError('WRONG_QUESTION')
 return True

def get_provenance(args,variant):
 return {'schema':'prd32_dev60_qwen_ab_run_v1','variant':variant,'scope':SCOPED,'images':60,'instances':171,'requests':513,
 'snapshot_sha256':sha(args.snapshot/'snapshot.json'),'instance_sha256':sha(args.snapshot/'instances.jsonl'),
 'request_sha256':sha(args.snapshot/'requests.jsonl'),'v5_sha256':V5_HASH,'checkpoint_sha256':CKPT,
 'focus_script_sha256':FOCUS_SHA,'prompt_hashes':PROMPT_SHA[variant],
 'question_sha256':tsha(QUESTION),'holdout_used':False,'human_gt_used':False,
 'official_v2_freeze':False,'official_prd_acceptance':False,'publish_allowed':False}

def progress(out,variant,lines,finished,model_load_s=None):
 sums={'schema':'prd32_dev60_qwen_ab_progress_v1','variant':variant,'total_requests':513,'successful':len(lines),
 'remaining':513-len(lines),'failed_this_session':not finished,'status':'COMPLETE_FOR_HUMAN_REVIEW' if finished else 'INCOMPLETE_RESUMABLE',
 'model_load_seconds_this_session':model_load_s,'publish_allowed':False,'official_prd_acceptance':False}
 savej(out/'run_summary.json',sums)

def run(args):
 rows,core,helper,focus=prompt_check(args)
 keys=expected_keys(rows);variant=args.variant;out=args.output
 expected=get_provenance(args,variant)
 if out.exists():
  if not args.resume:raise FileExistsError('OUTPUT_EXISTS_USE_EXPLICIT_RESUME')
  if loadj(out/'provenance.json')!=expected:raise ValueError('RUN_PROVENANCE_CHANGED')
  lines=loadjl(out/'qwen_results.jsonl');check_prefix(lines,keys,rows,variant)
  print('DEV60_RESUME_FROM',len(lines),'/513',flush=True)
 else:
  if args.resume:raise ValueError('RESUME_REQUIRES_EXISTING_OUTPUT')
  out.mkdir(parents=True)
  (out/'marked_inputs').mkdir()
  savej(out/'provenance.json',expected)
  (out/'qwen_results.jsonl').touch()
  lines=[]
 if len(lines)==513:
  progress(out,variant,lines,True);print('DEV60_ALL_REQUESTS_ALREADY_DONE',variant);return
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 import torch
 if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
 if torch.cuda.mem_get_info()[0]/(1024**3)<18:raise RuntimeError('GPU_FREE_MEMORY_BELOW_18GiB')
 from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
 dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
 t0=time.perf_counter()
 model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.model_dir),local_files_only=True,device_map='auto',torch_dtype=dtype,attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
 processor=AutoProcessor.from_pretrained(str(args.model_dir),local_files_only=True,max_pixels=512*28*28)
 load_s=round(time.perf_counter()-t0,2)
 for idx in range(len(lines),len(keys)):
  key=keys[idx];cid,iid,task=key;rec=rows[idx//3]
  qrec={'image_path':rec['image_path'],'image_sha256':rec['image_sha256'],'target_bbox_xyxy':rec['target_bbox_xyxy'],
     'target_instance_id':iid,'task':task,'query':QUESTION,'length_chars':[20,80],'predicted_attribute_context':[]}
  record={'case_id':cid,'target_instance_id':iid,'request_id':'__'.join(key),'task':task,'model_variant':variant,
   'status':'GENERATED_REVIEW_REQUIRED','v5_category_unverified':rec['predicted_category_v5_unverified'],
   'v5_predicted_bbox_xyxy':rec['target_bbox_xyxy'],'v5_predicted_score_uncalibrated':rec['predicted_score_v5_uncalibrated'],
   'target_bbox_origin':'prd31_v5_live_prediction','image_sha256':rec['image_sha256'],
   'selection_policy':'ALL_171_V5_PREDICTED_INSTANCES','human_label_or_gt_bbox_used':False,
   'quality_human_verified':False,'publish_allowed':False,
   'question_text':QUESTION if task=='vqa' else None,'question_sha256':tsha(QUESTION) if task=='vqa' else None}
  try:
   marked=helper.make_marked_image(qrec,out/'marked_inputs','marked_full_image',.05)
   prompt=core.build_prompt(helper,focus,qrec,task,variant)
   if tsha(prompt)!=PROMPT_SHA[variant][task]:raise ValueError('PROMPT_SHA_CHANGED')
   raw,timing=helper.run_model_case(model,processor,qrec,marked,prompt,task_limits={'tags':170,'vqa':128,'copy':120})
   if not isinstance(raw,str) or not raw.strip():raise ValueError('EMPTY_MODEL_OUTPUT')
   record.update(raw_text=raw,timing=timing,marked_image=str(marked),marked_image_sha256=sha(marked),prompt_sha256=tsha(prompt))
   if task=='tags':
    parsed=helper.parse_tag_output(raw)
    record['tag_parse']=parsed
    if isinstance(parsed,dict) and parsed.get('canonical_service_json') is not None:
     record['tag_bridge_candidate']=helper.bridge_tags_adapter(parsed['canonical_service_json'])
   if task=='vqa':record['vqa_visible_abstention_marker']=bool(ABSTAIN.search(raw))
   if task=='copy':record['copy_review_flags']=['DEBUG_MARKER_IN_COPY'] if DEBUG.search(raw) else []
   with (out/'qwen_results.jsonl').open('a',encoding='utf8') as fp:
    fp.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');fp.flush();os.fsync(fp.fileno())
   lines.append(record)
   print('DEV60_REQUEST',idx+1,'/513',record['request_id'],'OK',flush=True)
   if (idx+1)%20==0:progress(out,variant,lines,False,load_s)
  except Exception as e:
   savej(out/'last_error.json',{'request_index':idx+1,'request_id':'__'.join(key),'error_type':type(e).__name__,'error_message':str(e),'safe_to_resume_after_fix':True})
   progress(out,variant,lines,False,load_s)
   print('DEV60_STOP_PRESERVE_COMPLETED_ROWS',len(lines),'ERR',type(e).__name__,str(e),file=sys.stderr)
   raise
 progress(out,variant,lines,True,load_s)
 print('DEV60_QWEN_VARIANT_COMPLETE',variant,'513/513',flush=True)

def completed_run(path,variant,keys,rows,args):
 if loadj(path/'provenance.json')!=get_provenance(args,variant):raise ValueError('AB_RUN_PROVENANCE_MISMATCH')
 ls=loadjl(path/'qwen_results.jsonl')
 if len(ls)!=513:raise ValueError('AB_RUN_NOT_513')
 check_prefix(ls,keys,rows,variant)
 if loadj(path/'run_summary.json').get('status')!='COMPLETE_FOR_HUMAN_REVIEW':raise ValueError('AB_RUN_INCOMPLETE')
 return ls

def compare(args):
 f,rows=validate_snapshot(args,verify_images=False)
 if args.output.exists():raise FileExistsError('COMPARISON_ALREADY_EXISTS_NO_OVERWRITE')
 keys=expected_keys(rows)
 base=completed_run(args.baseline,'baseline',keys,rows,args)
 focus=completed_run(args.focus,'focus_v1',keys,rows,args)
 for a,b in zip(base,focus):
  if a.get('marked_image_sha256')!=b.get('marked_image_sha256'):raise ValueError('AB_IMAGE_RENDERING_CHANGED')
 paired=[];counts={'baseline':Counter(),'focus_v1':Counter()}
 for i in range(171):
  old=base[i*3:(i+1)*3];new=focus[i*3:(i+1)*3]
  obj={'case_id':rows[i]['case_id'],'instance_id':rows[i]['target_instance_id'],'V5_category_NOT_GT':rows[i]['predicted_category_v5_unverified']}
  for name,group in [('baseline',old),('focus_v1',new)]:
   for entry in group:
    t=entry['task'];obj[name+'_'+t]=entry['raw_text']
    if t=='tags':
     parsed=entry.get('tag_parse',{}).get('canonical_service_json') if isinstance(entry.get('tag_parse'),dict) else None
     if not isinstance(parsed,dict):counts[name]['tags_unparseable']+=1
     elif parsed.get('category')!=rows[i]['predicted_category_v5_unverified']:counts[name]['v5_qwen_disagreements_NOT_ACCURACY']+=1
    if t=='copy' and DEBUG.search(entry['raw_text']):counts[name]['copy_debug_word_heuristic']+=1
    if t=='vqa' and ABSTAIN.search(entry['raw_text']):counts[name]['vqa_abstention_heuristic']+=1
  paired.append(obj)
 args.output.mkdir(parents=True)
 with (args.output/'paired_instances.csv').open('w',newline='',encoding='utf-8-sig') as fp:
  writer=csv.DictWriter(fp,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
 report={'schema':'prd32_dev60_qwen_ab_compare_v1','scope':SCOPED,'instances_paired':171,'requests_per_variant':513,
  'total_requests':1026,'baseline_sha256':sha(args.baseline/'qwen_results.jsonl'),'focus_sha256':sha(args.focus/'qwen_results.jsonl'),
  'same_input_images_and_bboxes':True,'same_marked_image_sha_per_task':True,
  'baseline_diagnostics_NOT_ACCURACY':dict(counts['baseline']),'focus_diagnostics_NOT_ACCURACY':dict(counts['focus_v1']),
  'human_quality_review_required':True,'duplicate_candidates_NOT_GT':f['high_bbox_iou_pairs'],
  'official_v2_freeze':False,'official_prd_acceptance':False,'publish_allowed':False,'promote_focus_to_main':False}
 savej(args.output/'comparison.json',report)
 print('DEV60_PAIRED_COMPARISON_OK',json.dumps({'baseline':dict(counts['baseline']),'focus':dict(counts['focus_v1'])},ensure_ascii=False))
 print('NOT_ACCURACY_REQUIRES_HUMAN_REVIEW')

def export(args):
 from zipfile import ZipFile,ZIP_DEFLATED
 if args.export.exists():raise FileExistsError('EXPORT_ALREADY_EXISTS')
 f,rows=validate_snapshot(args,verify_images=False)
 keys=expected_keys(rows)
 for path,v in [(args.baseline,'baseline'),(args.focus,'focus_v1')]:completed_run(path,v,keys,rows,args)
 if not (args.compare/'comparison.json').exists():raise ValueError('COMPARISON_REQUIRED')
 comp=loadj(args.compare/'comparison.json')
 if comp.get('instances_paired')!=171 or comp.get('baseline_sha256')!=sha(args.baseline/'qwen_results.jsonl') or comp.get('focus_sha256')!=sha(args.focus/'qwen_results.jsonl'):raise ValueError('COMPARISON_OUTDATED_OR_CHANGED')
 with ZipFile(args.export,'w',ZIP_DEFLATED,compresslevel=6) as z:
  for folder,include in [(args.snapshot,['snapshot.json','instances.jsonl','requests.jsonl']),
   (args.baseline,['provenance.json','run_summary.json','qwen_results.jsonl']),
   (args.focus,['provenance.json','run_summary.json','qwen_results.jsonl']),
   (args.compare,['comparison.json','paired_instances.csv'])]:
   for name in include:z.write(folder/name,arcname=f'{folder.name}/{name}')
 print('DEV60_AB_UPLOAD_THIS',args.export,round(args.export.stat().st_size/1024**2,2),'MiB')

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('command',choices=['snapshot','preflight','run','compare','export'])
 parser.add_argument('--raw',type=Path,default=DEFAULT_RAW)
 parser.add_argument('--input',type=Path,default=DEFAULT_INPUT)
 parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
 parser.add_argument('--focus-script',type=Path,default=DEFAULT_FOCUS)
 parser.add_argument('--model-dir',type=Path,default=DEFAULT_MODEL)
 parser.add_argument('--snapshot',type=Path,default=DEFAULT_SNAP)
 parser.add_argument('--baseline',type=Path,default=DEFAULT_BASE)
 parser.add_argument('--focus',type=Path,default=DEFAULT_FOCUS_RUN)
 parser.add_argument('--compare',type=Path,default=DEFAULT_COMPARE)
 parser.add_argument('--variant',choices=['baseline','focus_v1'])
 parser.add_argument('--output',type=Path)
 parser.add_argument('--resume',action='store_true')
 parser.add_argument('--export',type=Path,default=ROOT/'PRD32_DEV60_QWEN_AB_RESULTS_20261030_UPLOAD_THIS.zip')
 args=parser.parse_args(argv)
 if args.command=='snapshot':make_snapshot(args)
 elif args.command=='preflight':prompt_check(args)
 elif args.command=='run':
  if not args.variant:parser.error('RUN_REQUIRES_VARIANT')
  if args.output is None:args.output=args.baseline if args.variant=='baseline' else args.focus
  run(args)
 elif args.command=='compare':
  args.output=args.compare;compare(args)
 elif args.command=='export':export(args)

if __name__=='__main__':
 try:main()
 except Exception as ex:
  print('DEV60_QWEN_AB_BLOCKED',type(ex).__name__+': '+str(ex),file=sys.stderr)
  raise SystemExit(2)
