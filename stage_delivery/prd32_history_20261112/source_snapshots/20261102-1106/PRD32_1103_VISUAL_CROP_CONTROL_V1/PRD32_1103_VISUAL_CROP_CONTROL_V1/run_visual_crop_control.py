#!/usr/bin/env python3
# coding: utf-8
"""PRD32 2026-11-03: independent Focus V1 visual-crop experiment.

One independent factor: marked full image -> marked full image + same V5 bbox crop panel.
No label/GT/holdout read by this runner; no existing outputs modified.
Run only on user's own GPU. No accuracy is claimed by nonhuman diagnostics.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace')
LEGACY_SHA='98a833c429da4c39849942a81f1067239273258a8605eac9265d0aeb9defc6f3'
SMOKE_SHA='f4037719bf42c64a98eca6cfda44591c34159bf7bd9186cca97282379000dd6e'
FIXED171_SHA='6aae3f64d05b342ea87f1afa8d197c803092ba345906653a1170dcde48ca678a'
FIXED513_SHA='bfd32ce63d294ea5af20510f51ac4b4466cfeab1aa99d1846a91b225e956e53f'
TASKS=('tags','vqa','copy')
VARIANT='focus_v1_full_marked_plus_v5_crop_v1'
SCOPE='TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY'
CROP_SPEC={'schema':'prd32_1103_visual_crop_spec_v1',
 'source':'original_image_pixels_only','crop_bbox_origin':'same_V5_live_bbox',
 'roi_padding_fraction_of_bbox_max_side':0.12,
 'panel_width_fraction_of_marked_image':0.60,
 'panel_crop_max_height_fraction':0.84,
 'red_border_rgb':[218,48,57],
 'background_rgb':[245,245,245],
 'separation_rgb':[170,170,170],
 'no_text_or_class_labels_in_overlay':True,
 'baseline_full_marked_image_unchanged_left':True,
 'output_format':'PNG_lossless'}


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as fp:
  for c in iter(lambda:fp.read(1024*1024),b''):h.update(c)
 return h.hexdigest()

def jload(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def jlread(path):return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
def jwrite(path, data):
 Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def jline(fp, record):
 with Path(fp).open('a',encoding='utf8') as out:
  out.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno())
def tsha(s):return hashlib.sha256(s.encode('utf8')).hexdigest()

def import_legacy():
 p=HERE/'reference_dev60_qwen_ab.py'
 if sha(p)!=LEGACY_SHA:raise ValueError('PINNED_REFERENCE_RUNNER_CHANGED')
 spec=importlib.util.spec_from_file_location('prd32_1103_legacy_dev60',p)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module

def validate_fixed_lists(rows):
 pinned171=HERE/'instances171_input_only.jsonl';pinned513=HERE/'requests513_input_only.jsonl';pinned24=HERE/'smoke24_input_only.jsonl'
 for p,expect in [(pinned171,FIXED171_SHA),(pinned513,FIXED513_SHA),(pinned24,SMOKE_SHA)]:
  if sha(p)!=expect:raise ValueError('FIXED_LIST_FILE_CHANGED:'+p.name)
 data=jlread(pinned171)
 if rows!=data or len(rows)!=171:raise ValueError('INSTANCE_LIST_CHANGED')
 req=jlread(pinned513)
 expect=[{'request_id':f'{r["case_id"]}__{r["target_instance_id"]}__{t}',
   'case_id':r['case_id'],'target_instance_id':r['target_instance_id'],'task':t,
   'vqa_question':'红框内目标服饰的主要可见颜色是什么？' if t=='vqa' else None}
  for r in rows for t in TASKS]
 if req!=expect:raise ValueError('CANONICAL_REQUESTS_CHANGED')
 smoke=jlread(pinned24)
 mapping={(r['case_id'],r['target_instance_id']):r for r in data}
 keys=[(s['case_id'],s['target_instance_id']) for s in smoke]
 if len(keys)!=24 or len(set(keys))!=24:raise ValueError('SMOKE_KEYS_NOT_24_UNIQUE')
 for s,k in zip(smoke,keys):
  if k not in mapping or mapping[k]!=s:raise ValueError('SMOKE_LABEL_OR_INPUT_MISMATCH')
 if any(('review' in x.lower() or 'ground_truth' in x.lower() or x.lower()=='gt') for row in smoke for x in row):
  raise ValueError('REVIEW_OR_GT_IN_SMOKE_LIST')
 return keys

def legacy_args(args):
 import argparse as _argparse
 return _argparse.Namespace(raw=args.raw,input=args.input,source=args.source,focus_script=args.focus_script,
   model_dir=args.model_dir,snapshot=args.snapshot,focus=args.original_focus)

def baseline_and_instances(args):
 old=import_legacy();la=legacy_args(args)
 rows,core,helper,focus=old.prompt_check(la)
 validate_fixed_lists(rows)
 focus_results=old.completed_run(args.original_focus,'focus_v1',old.expected_keys(rows),rows,la)
 for i in range(171):
  gr=focus_results[i*3:(i+1)*3]
  if len({r['marked_image_sha256'] for r in gr})!=1:raise ValueError('ORIGINAL_MARKED_IMAGE_NOT_IDENTICAL_ACROSS_TASKS')
  if len({r['prompt_sha256'] for r in gr})!=3:raise ValueError('EXPECTED_DISTINCT_TASK_PROMPTS')
 return old,rows,core,helper,focus,focus_results

def target_crop_box(bbox, origsize, frac=0.12):
 w,h=origsize;x1,y1,x2,y2=(float(x) for x in bbox)
 if w<=0 or h<=0 or not (0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('INVALID_CROP_BBOX')
 p=max(x2-x1,y2-y1)*frac
 l=max(0,int((x1-p)//1));t=max(0,int((y1-p)//1))
 r=min(w,int((x2+p)+0.999999));b=min(h,int((y2+p)+0.999999))
 if r<=l or b<=t:raise ValueError('EMPTY_CROP')
 return l,t,r,b

def make_composite(marked_path, source_path, bbox, output, expect_source_sha, expect_marked_sha):
 """Uses *only* original image and unchanged V5 bbox. No model scores/classes/human info rendered."""
 if sha(source_path)!=expect_source_sha:raise ValueError('SOURCE_IMAGE_SHA_MISMATCH')
 if sha(marked_path)!=expect_marked_sha:raise ValueError('ORIGINAL_MARKED_IMAGE_SHA_MISMATCH')
 with Image.open(marked_path) as a, Image.open(source_path) as b:
  marked=a.convert('RGB'); source=b.convert('RGB')
 W,H=marked.size
 if W<100 or H<100:raise ValueError('MARKED_IMAGE_TOO_SMALL')
 crop_box=target_crop_box(bbox,source.size,CROP_SPEC['roi_padding_fraction_of_bbox_max_side'])
 x1,y1,x2,y2=(float(x) for x in bbox)
 l,t,r,b=crop_box; crop=source.crop(crop_box)
 panel_w=max(100,round(W*CROP_SPEC['panel_width_fraction_of_marked_image']))
 content_w=max(32,panel_w-28);content_h=max(32,round(H*CROP_SPEC['panel_crop_max_height_fraction']))
 scale=min(content_w/crop.width,content_h/crop.height)
 cw=max(1,round(crop.width*scale));ch=max(1,round(crop.height*scale))
 crop=crop.resize((cw,ch),Image.Resampling.LANCZOS)
 # Both red boxes refer to exactly the same V5 target: enlarged view is not a second object.
 dr=ImageDraw.Draw(crop)
 rr=[max(0,round((x1-l)*cw/(r-l))),max(0,round((y1-t)*ch/(b-t))),
     min(cw-1,round((x2-l)*cw/(r-l))),min(ch-1,round((y2-t)*ch/(b-t)))]
 bw=max(2,round(min(cw,ch)*0.008))
 if rr[2]>rr[0] and rr[3]>rr[1]:dr.rectangle(rr,outline=tuple(CROP_SPEC['red_border_rgb']),width=bw)
 canvas=Image.new('RGB',(W+panel_w,H),tuple(CROP_SPEC['background_rgb']))
 canvas.paste(marked,(0,0))
 px=W+(panel_w-cw)//2;py=(H-ch)//2
 canvas.paste(crop,(px,py))
 ImageDraw.Draw(canvas).line([(W,0),(W,H)],fill=tuple(CROP_SPEC['separation_rgb']),width=2)
 output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
 temp=output.with_name(output.stem+'.writing.png')
 try:
  canvas.save(temp,format='PNG',optimize=False)
  os.replace(temp,output)
 finally:
  if temp.exists():temp.unlink()
 return {'composite_sha256':sha(output),'original_marked_sha256':expect_marked_sha,
  'image_sha256':expect_source_sha,'crop_rect_xyxy':list(crop_box),
  'original_marked_size':[W,H],'composite_size':[W+panel_w,H],
  'crop_zoomed_size':[cw,ch]}

def expected_provenance(args,old,rows):
 return {'schema':'prd32_1103_focus_v1_visual_crop_v1_provenance', 'scope':SCOPE,
  'visual_change_only':'unchanged_full_marked_image_plus_same_V5_bbox_crop_panel',
  'render_spec':CROP_SPEC,'render_spec_sha256':tsha(json.dumps(CROP_SPEC,sort_keys=True,ensure_ascii=False)),
  'snapshot_sha256':sha(args.snapshot/'snapshot.json'),
  'canonical_171_instances_sha256':FIXED171_SHA,'canonical_513_requests_sha256':FIXED513_SHA,
  'smoke24_sha256':SMOKE_SHA,'original_focus_results_sha256':sha(args.original_focus/'qwen_results.jsonl'),
  'original_focus_prompt_sha256':old.PROMPT_SHA['focus_v1'],
  'v5_predictions_sha256':old.V5_HASH, 'checkpoint_sha256':old.CKPT,
  'images':60,'instances':171,'model_variant':VARIANT,'no_human_labels_gt_in_model_inputs':True,
  'holdout_used':False,'main_pipeline_modified':False,'official_freeze_approved':False,
  'official_prd_acceptance':False,'publish_allowed':False,
  'smoke_not_for_effect_size_or_parameter_selection':True}

def render(args):
 old,rows,core,helper,focus,oldres=baseline_and_instances(args)
 prov=expected_provenance(args,old,rows)
 if args.render_dir.exists():
  if not args.resume:raise FileExistsError('RENDER_EXISTS_USE_RESUME')
  if jload(args.render_dir/'provenance.json')!=prov:raise ValueError('RENDER_PROVENANCE_CHANGED')
  items=jlread(args.render_dir/'visual_inputs.jsonl')
 else:
  if args.resume:raise ValueError('RESUME_REQUIRES_EXISTING_RENDER')
  args.render_dir.mkdir(parents=True)
  (args.render_dir/'composite_images').mkdir()
  jwrite(args.render_dir/'provenance.json',prov)
  (args.render_dir/'visual_inputs.jsonl').touch()
  items=[]
 for i,entry in enumerate(items):
  row=rows[i];r=oldres[i*3]
  if (entry['case_id'],entry['target_instance_id'])!=(row['case_id'],row['target_instance_id']):raise ValueError('RENDER_PREFIX_KEYS_CHANGED')
  if entry['original_marked_sha256']!=r['marked_image_sha256'] or entry['image_sha256']!=row['image_sha256']:raise ValueError('RENDER_PREFIX_SOURCE_CHANGED')
  if sha(entry['composite_path'])!=entry['composite_sha256']:raise ValueError('RENDER_PREFIX_COMPOSITE_CHANGED')
 for i in range(len(items),171):
  row=rows[i];r=oldres[i*3]
  out=args.render_dir/'composite_images'/f'{row["case_id"]}__{row["target_instance_id"]}_crop_visual_v1.png'
  values=make_composite(r['marked_image'],row['image_path'],row['target_bbox_xyxy'],out,row['image_sha256'],r['marked_image_sha256'])
  vals={'case_id':row['case_id'],'target_instance_id':row['target_instance_id'],
    'composite_path':str(out),'original_marked_path':r['marked_image'],**values}
  jline(args.render_dir/'visual_inputs.jsonl',vals);items.append(vals)
  if (i+1)%20==0:print('VISUAL_RENDERED',i+1,'/171',flush=True)
 jwrite(args.render_dir/'run_summary.json',{'schema':'prd32_1103_visual_crop_render_summary','status':'RENDER_COMPLETE',
   'images':60,'instances':171,'rendered':171,'visual_spec_sha256':prov['render_spec_sha256'],
   'zero_human_labels':True,'official_freeze_approved':False,'publish_allowed':False})
 print('VISUAL_CROP_RENDER_COMPLETE_171',flush=True)

def validated_render(args,old,rows):
 prov=expected_provenance(args,old,rows)
 if jload(args.render_dir/'provenance.json')!=prov:raise ValueError('VISUAL_PROVENANCE_WRONG')
 if jload(args.render_dir/'run_summary.json').get('rendered')!=171:raise ValueError('VISUAL_RENDER_NOT_COMPLETE')
 render_rows=jlread(args.render_dir/'visual_inputs.jsonl')
 if len(render_rows)!=171:raise ValueError('EXPECTED_171_RENDERS')
 for r,v in zip(rows,render_rows):
  if (v['case_id'],v['target_instance_id'])!=(r['case_id'],r['target_instance_id']):raise ValueError('VISUAL_ORDER_CHANGED')
  if v['image_sha256']!=r['image_sha256']:raise ValueError('RENDER_IMAGE_ID_CHANGED')
  if not Path(v['composite_path']).is_file() or sha(v['composite_path'])!=v['composite_sha256']:raise ValueError('RENDER_IMAGE_SHA_CHANGED')
 return render_rows

def smoke_keys():return [(r['case_id'],r['target_instance_id']) for r in jlread(HERE/'smoke24_input_only.jsonl')]
def planned_scope_keys(scope,rows):
 rows_keys=[(r['case_id'],r['target_instance_id']) for r in rows]
 if scope=='smoke':
  sel=set(smoke_keys());return [(c,i,t) for c,i in rows_keys if (c,i) in sel for t in TASKS]
 if scope=='full':return [(c,i,t) for c,i in rows_keys for t in TASKS]
 raise ValueError('BAD_SCOPE')

def scope_output(args):return args.smoke_output if args.scope=='smoke' else args.full_output

def check_prefix(rows,keys,render_rows):
 pos={(r['case_id'],r['target_instance_id']):r for r in render_rows}
 for idx,r in enumerate(rows):
  k=keys[idx]
  if (r.get('case_id'),r.get('target_instance_id'),r.get('task'))!=k or r.get('request_id')!='__'.join(k):raise ValueError('RESUME_KEY_CHANGED')
  ref=pos[k[:2]]
  if r.get('model_variant')!=VARIANT or r.get('status')!='GENERATED_REVIEW_REQUIRED':raise ValueError('RESUME_BAD_VARIANT')
  if r.get('composite_sha256')!=ref['composite_sha256'] or r.get('original_marked_sha256')!=ref['original_marked_sha256']:\
   raise ValueError('RESUME_VISUAL_INPUT_CHANGED')
  if not isinstance(r.get('raw_text'),str) or not r['raw_text'].strip():raise ValueError('RESUME_EMPTY_TEXT')
  if r.get('human_label_or_gt_bbox_used') is not False or r.get('publish_allowed') is not False:raise ValueError('RESUME_BAD_PROVENANCE')

def run(args):
 old,rows,core,helper,focus,oldres=baseline_and_instances(args)
 render_rows=validated_render(args,old,rows)
 keys=planned_scope_keys(args.scope,rows);out=scope_output(args)
 prov=expected_provenance(args,old,rows)
 prov.update({'run_scope':args.scope,'request_count':len(keys),'task_limits':{'tags':170,'vqa':128,'copy':120}})
 if args.scope=='full':
  smoke=args.smoke_output
  if not smoke.is_dir() or jload(smoke/'run_summary.json').get('status')!='COMPLETE_72_FOR_REVIEW':
   raise ValueError('FIRST_COMPLETE_SMOKE72_FLOW_CHECK')
 if out.exists():
  if not args.resume:raise FileExistsError('RESULT_EXISTS_USE_RESUME')
  if jload(out/'provenance.json')!=prov:raise ValueError('QWEN_RUN_PROVENANCE_CHANGED')
  results=jlread(out/'qwen_results.jsonl');check_prefix(results,keys,render_rows)
 else:
  if args.resume:raise ValueError('RESUME_REQUIRES_EXISTING_QWEN_RUN')
  out.mkdir(parents=True)
  jwrite(out/'provenance.json',prov);(out/'qwen_results.jsonl').touch();results=[]
 if len(results)==len(keys):
  progress(out,args.scope,results,len(keys),True)
  print('CROP_VISUAL_RUN_ALREADY_COMPLETE',args.scope,len(results));return
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 import torch
 if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
 if torch.cuda.mem_get_info()[0]/1024**3<18:raise RuntimeError('GPU_FREE_MEMORY_BELOW_18GiB')
 from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
 dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
 t0=time.perf_counter()
 model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.model_dir),local_files_only=True,
  device_map='auto',torch_dtype=dtype,attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
 processor=AutoProcessor.from_pretrained(str(args.model_dir),local_files_only=True,max_pixels=512*28*28)
 model_load_s=round(time.perf_counter()-t0,2)
 rowmap={(r['case_id'],r['target_instance_id']):r for r in rows}
 imgmap={(r['case_id'],r['target_instance_id']):r for r in render_rows}
 for idx in range(len(results),len(keys)):
  cid,iid,task=keys[idx];r=rowmap[cid,iid];im=imgmap[cid,iid]
  qrec={'image_path':r['image_path'],'image_sha256':r['image_sha256'],'target_bbox_xyxy':r['target_bbox_xyxy'],
    'target_instance_id':iid,'task':task,'query':old.QUESTION,'length_chars':[20,80],'predicted_attribute_context':[]}
  prompt=core.build_prompt(helper,focus,qrec,task,'focus_v1')
  if tsha(prompt)!=old.PROMPT_SHA['focus_v1'][task]:raise ValueError('FOCUS_V1_PROMPT_CHANGED')
  record={'case_id':cid,'target_instance_id':iid,'request_id':'__'.join((cid,iid,task)),
    'task':task,'model_variant':VARIANT,'status':'GENERATED_REVIEW_REQUIRED',
    'image_sha256':r['image_sha256'],'v5_predicted_bbox_xyxy':r['target_bbox_xyxy'],
    'v5_category_unverified':r['predicted_category_v5_unverified'],
    'prompt_sha256':tsha(prompt),'question_sha256':old.tsha(old.QUESTION) if task=='vqa' else None,
    'original_marked_sha256':im['original_marked_sha256'],
    'composite_sha256':im['composite_sha256'],'composite_path':im['composite_path'],
    'visual_change_only':True,'human_label_or_gt_bbox_used':False,
    'quality_human_verified':False,'publish_allowed':False,'official_prd_acceptance':False}
  try:
   raw,timing=helper.run_model_case(model,processor,qrec,Path(im['composite_path']),prompt,
       task_limits={'tags':170,'vqa':128,'copy':120})
   if not isinstance(raw,str) or not raw.strip():raise ValueError('EMPTY_GENERATION')
   record.update(raw_text=raw,timing=timing)
   if task=='tags':
    parsed=helper.parse_tag_output(raw);record['tag_parse']=parsed
    if isinstance(parsed,dict) and parsed.get('canonical_service_json') is not None:
     record['tag_bridge_candidate']=helper.bridge_tags_adapter(parsed['canonical_service_json'])
   if task=='vqa':record['vqa_visible_abstention_marker']=bool(old.ABSTAIN.search(raw))
   if task=='copy':record['copy_review_flags']=['DEBUG_MARKER_IN_COPY'] if old.DEBUG.search(raw) else []
   jline(out/'qwen_results.jsonl',record);results.append(record)
   print('CROP_VISUAL_REQUEST',args.scope,idx+1,'/',len(keys),record['request_id'],'OK',flush=True)
   if (idx+1)%12==0:progress(out,args.scope,results,len(keys),False,model_load_s)
  except Exception as e:
   jwrite(out/'last_error.json',{'request_index':idx+1,'request_id':record['request_id'],
       'error_type':type(e).__name__,'message':str(e)[:400],'safe_to_resume':True})
   progress(out,args.scope,results,len(keys),False,model_load_s)
   print('CROP_VISUAL_STOP_RETAIN_PREFIX',len(results),type(e).__name__,str(e),file=sys.stderr)
   raise
 progress(out,args.scope,results,len(keys),True,model_load_s)
 print('CROP_VISUAL_RUN_COMPLETE',args.scope,len(results),'/',len(keys))

def progress(out,scope,results,total,finished,load_seconds=None):
 jwrite(out/'run_summary.json',{'schema':'prd32_1103_visual_crop_run_summary_v1',
  'status': ('COMPLETE_72_FOR_REVIEW' if scope=='smoke' else 'COMPLETE_513_FOR_REVIEW') if finished else 'INCOMPLETE_RESUMABLE',
  'scope':scope,'successful':len(results),'expected':total,'remaining':total-len(results),
  'model_load_seconds_session':load_seconds,'human_review_required':True,
  'official_freeze_approved':False,'publish_allowed':False})

def compare_scope(args):
 old,rows,core,helper,focus,oldres=baseline_and_instances(args)
 vrows=validated_render(args,old,rows)
 keys=planned_scope_keys(args.scope,rows);out=scope_output(args)
 s=jload(out/'run_summary.json')
 if s.get('successful')!=len(keys) or s.get('status') not in ('COMPLETE_72_FOR_REVIEW','COMPLETE_513_FOR_REVIEW'):raise ValueError('QWEN_RUN_NOT_COMPLETE')
 got=jlread(out/'qwen_results.jsonl')
 if len(got)!=len(keys):raise ValueError('QWEN_RESULT_COUNT_WRONG')
 check_prefix(got,keys,vrows)
 prev={(x['case_id'],x['target_instance_id'],x['task']):x for x in oldres}
 pairs=[];c=Counter()
 for k,row in zip(keys,got):
  before=prev[k];a=before['raw_text'];b=row['raw_text']
  if row.get('prompt_sha256')!=before['prompt_sha256']:raise ValueError('PROMPT_DIFFERS')
  if row['image_sha256']!=before['image_sha256'] or row['v5_predicted_bbox_xyxy']!=before['v5_predicted_bbox_xyxy']:raise ValueError('INPUT_CHANGED')
  same=a==b
  c['text_changed' if not same else 'text_unchanged']+=1
  c[f'{k[2]}_changed' if not same else f'{k[2]}_unchanged']+=1
  pairs.append({'request_id':'__'.join(k),'case_id':k[0],'instance_id':k[1],'task':k[2],
    'focus_v1_original_raw':a,'focus_v1_with_crop_raw':b,
    'same_text':same,'original_prompt_sha256':before['prompt_sha256'],
    'original_marked_sha256':before['marked_image_sha256'],
    'composite_sha256':row['composite_sha256']})
 return pairs,dict(c)

def export_scope(args):
 pairs,c=compare_scope(args)
 if args.export.exists():raise FileExistsError('EXPORT_EXISTS_NO_OVERWRITE')
 summary={'schema':'prd32_1103_visual_crop_paired_text_diagnostic_v1','scope':args.scope,
  'images':60,'canonical_instances':171,'requests_compared':len(pairs),
  'counts_TEXT_CHANGED_NOT_ACCURACY':c,
  'only_visual_input_differs':True,'original_focus_v1_unchanged':True,
  'baseline_full171_human_reviews_NOT_USED_FOR_MODEL':True,
  'holdout_not_used':True,'human_review_of_new_outputs_required':True,
  'official_acceptance':False,'publish_allowed':False}
 comparison=scope_output(args)/'paired_text_comparison.json'
 jwrite(comparison,summary)
 original=scope_output(args)
 with ZipFile(args.export,'w',ZIP_DEFLATED,compresslevel=6) as z:
  for f in [HERE/'README_FIRST_CN.md',HERE/'visual_spec.json',
            args.render_dir/'provenance.json',args.render_dir/'run_summary.json',
            args.render_dir/'visual_inputs.jsonl',original/'provenance.json',original/'run_summary.json',
            original/'qwen_results.jsonl',comparison]:
   if f.is_file():z.write(f,f.name if f.parent==HERE else f'{f.parent.name}/{f.name}')
  comparison_bytes='\n'.join(json.dumps(r,ensure_ascii=False) for r in pairs).encode('utf8')+b'\n'
  z.writestr('paired_text_rows.jsonl',comparison_bytes)
  # Preview a few only. This is *not* a blind evaluation or GT.
  from PIL import Image
  chosen=list(dict.fromkeys((p['case_id'],p['instance_id']) for p in pairs))[:12]
  idx={(r['case_id'],r['target_instance_id']):r for r in jlread(args.render_dir/'visual_inputs.jsonl')}
  for cid,iid in chosen:
   ip=idx[cid,iid]['composite_path']
   with Image.open(ip) as im:
    prev=im.convert('RGB');prev.thumbnail((800,760));
    import io
    b=io.BytesIO();prev.save(b,'JPEG',quality=78,optimize=True)
    z.writestr(f'preview/{cid}__{iid}.jpg',b.getvalue())
 print('VISUAL_CROP_UPLOAD_THIS',args.export,round(args.export.stat().st_size/1024**2,2),'MiB')
 print('COUNTS_TEXT_CHANGED_NOT_ACCURACY',c)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('command',choices=['preflight','render','run','export'])
 p.add_argument('--scope',choices=['smoke','full'],default='smoke')
 p.add_argument('--resume',action='store_true')
 p.add_argument('--raw',type=Path,default=ROOT/'PRD32_DEV60_V5_REAL_20261030_run01')
 p.add_argument('--input',type=Path,default=ROOT/'PRD32_FIXED_DEV_V2_60IMG_CANDIDATES_20261030/v5_image_only_inputs_60.jsonl')
 p.add_argument('--source',type=Path,default=ROOT/'PRD32_20261029_NextBatch_GTFree')
 p.add_argument('--focus-script',type=Path,default=ROOT/'PRD32_FOCUS_AB_PATCH_20261029/prd32_focus_ab.py')
 p.add_argument('--model-dir',type=Path,default=ROOT/'models/Qwen2.5-VL-7B-Instruct')
 p.add_argument('--snapshot',type=Path,default=ROOT/'PRD32_DEV60_QWEN_DIAGNOSTIC_SNAPSHOT_20261030')
 p.add_argument('--original-focus',type=Path,default=ROOT/'PRD32_DEV60_QWEN_FOCUS_V1_20261030_run01')
 p.add_argument('--render-dir',type=Path,default=ROOT/'PRD32_1103_VISUAL_CROP_RENDER_V1')
 p.add_argument('--smoke-output',type=Path,default=ROOT/'PRD32_1103_VISUAL_CROP_SMOKE_RUN01')
 p.add_argument('--full-output',type=Path,default=ROOT/'PRD32_1103_VISUAL_CROP_FULL_RUN01')
 p.add_argument('--export',type=Path)
 a=p.parse_args()
 if a.export is None:a.export=ROOT/f'PRD32_1103_VISUAL_CROP_{a.scope.upper()}_UPLOAD_THIS.zip'
 try:
  if a.command=='preflight':
   old,rows,_,_,_,_ = baseline_and_instances(a)
   print('VISUAL_CROP_PREFLIGHT_PASS',len(set(r['case_id'] for r in rows)),len(rows),len(smoke_keys()),'smoke cases')
   print('FOCUS_V1_PROMPTS_LOCKED; NO_GT_NO_HOLDOUT; GPU_NOT_LOADED')
  elif a.command=='render':render(a)
  elif a.command=='run':run(a)
  elif a.command=='export':export_scope(a)
 except Exception as ex:
  print('VISUAL_CROP_BLOCKED',type(ex).__name__,str(ex),file=sys.stderr)
  raise SystemExit(2)

if __name__=='__main__':main()
