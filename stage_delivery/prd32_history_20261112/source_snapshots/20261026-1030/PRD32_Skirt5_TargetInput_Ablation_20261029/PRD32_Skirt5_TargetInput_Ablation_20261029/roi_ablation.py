#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD 3.2 Skirt5: unchanged V2/V3 prompts, full marked image vs marked target crop.

Reads the immutable 5-image train-source frozen plan and original GPU predictions.
'preflight' checks provenance, hashes, generates ROI images WITHOUT GPU and GT input.
'gpu' runs original frozen prompts on ROI images only.
'score' compares stored full-image baseline to ROI results; never re-labels.
No holdout or val access; NOT independent acceptance or causal proof.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,statistics,sys,time
from pathlib import Path
from PIL import Image,ImageDraw

BASE=Path('/workspace')
GATE=BASE/'PRD32_20261029_Validation_Gate'
FROZEN=BASE/'PRD32_SKIRT5_FROZEN_20261029_run01'
OLD=BASE/'PRD32_SKIRT5_GPU_20261029_run01'/'predictions_pair.jsonl'
MODEL=BASE/'models/Qwen2.5-VL-7B-Instruct'
ALLOWED_ROOT=BASE/'fashion_data'
EXPECTED={
 'plan':'02afea075c7bf18346759136fd45f45fc08858e6e60b1123b897f8ef403743ce',
 'reference':'2b568999a8346149861411017ef00792dc48ba7ae05043967e746e771d1d322c',
 'baseline':'f6ea69dcd52897755a0dfb42984d7ac84f7e90327740ffcec0665d432e8c1be0',
 'prompt_v2':'4821bc893d412cffa3a7f8c13709d09b9f3ded05ccfb2a12b72be9298bad6e2b',
 'prompt_v3':'b6541858e4114ff4afb0ace15db7567b36952ccbca32fc59704df32b4807513c',
}
IDS=[f'skirt_diag_{i:03d}' for i in range(1,6)]
CATS={'top','pants','skirt','outerwear','dress','shoe','bag','accessory'}

def fail(msg):raise RuntimeError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for blk in iter(lambda:f.read(1<<20),b''):h.update(blk)
 return h.hexdigest()
def write_json(p,data):Path(p).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def read_json(p):return json.loads(Path(p).read_text(encoding='utf8'))
def read_jsonl(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf8').splitlines() if x.strip()]
def write_csv(p,rows,fieldnames):
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore');w.writeheader();w.writerows(rows)
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p)
 if spec is None or spec.loader is None:fail(f'Cannot import {p}')
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def load_gate(gate):
 p=gate/'run_and_score.py'
 if not p.is_file():fail('Missing original fixed Validation_Gate/run_and_score.py')
 return module('prd32_fixed_gate',p)

def validate(a):
 gate=load_gate(a.gate)
 plan,cfg=gate.verify_frozen(a.frozen,a.allowed_root,check_imgs=True)
 if [p['case_id'] for p in plan]!=IDS:fail('Unexpected cohort/order; do not use other images')
 if sha(a.frozen/'frozen_inference_plan.jsonl')!=EXPECTED['plan']:fail('Unexpected plan SHA; abort')
 if sha(a.frozen/'frozen_human_reference.jsonl')!=EXPECTED['reference']:fail('Unexpected frozen GT SHA; abort')
 if sha(a.baseline)!=EXPECTED['baseline']:fail('Unexpected baseline SHA; abort and report, do NOT overwrite files')
 baseline=read_jsonl(a.baseline)
 if len(baseline)!=5 or {b['case_id'] for b in baseline}!=set(IDS):fail('Baseline cases incomplete')
 bm={b['case_id']:b for b in baseline}
 for p in plan:
  b=bm[p['case_id']]
  if b['image_sha256']!=p['image_sha256'] or not all(b.get(v,{}).get('status')=='GENERATED_UNVERIFIED' for v in ('V2','V3')):
   fail('Baseline identity or status mismatch: '+p['case_id'])
 v2,v3=gate.load_helpers()
 hashes={'V2':hashlib.sha256(v2.TAG_PROMPT.encode()).hexdigest(),
         'V3':hashlib.sha256(v3.FULL_TAGS_PROMPT.encode()).hexdigest()}
 if hashes['V2']!=EXPECTED['prompt_v2'] or hashes['V3']!=EXPECTED['prompt_v3']:fail('Original prompts changed; abort')
 return gate,plan,bm,v2,v3,hashes

def roi_bounds(bounds,size,pad=.08):
 w,h=size;x1,y1,x2,y2=map(float,bounds)
 if not(0<=x1<x2<=w and 0<=y1<y2<=h):fail('Invalid bbox')
 dx=(x2-x1)*pad;dy=(y2-y1)*pad
 left=max(0,int(math.floor(x1-dx)));top=max(0,int(math.floor(y1-dy)))
 right=min(w,int(math.ceil(x2+dx)));bottom=min(h,int(math.ceil(y2+dy)))
 if right-left<18 or bottom-top<18:fail('Tiny ROI crop; manual review required')
 return left,top,right,bottom

def make_roi(original,bbox,dest,pad=.08):
 # Keep target cue identical (red rectangle); context changes via crop.
 with Image.open(original) as im:src=im.convert('RGB')
 l,t,r,b=roi_bounds(bbox,src.size,pad)
 crop=src.crop((l,t,r,b));x1,y1,x2,y2=bbox
 rect=(round(x1-l),round(y1-t),round(x2-l),round(y2-t))
 ImageDraw.Draw(crop).rectangle(rect,outline=(250,40,40),width=max(3,min(crop.size)//150))
 crop.save(dest,'JPEG',quality=93)
 return {'crop_xyxy':[l,t,r,b], 'local_bbox_xyxy':list(rect),
         'crop_width':crop.width,'crop_height':crop.height,
         'roi_area_fraction':round(crop.width*crop.height/src.width/src.height,6)}

def preflight(a):
 gate,plan,baseline,v2,v3,hashes=validate(a)
 if a.output.exists():fail('Output path exists; not overwriting: '+str(a.output))
 a.output.mkdir(parents=True);(a.output/'cropped_marked_images').mkdir()
 rows=[]
 for p in plan:
  dst=a.output/'cropped_marked_images'/(p['case_id']+'_roi_marked.jpg')
  info=make_roi(p['image_path'],p['bbox'],dst,pad=a.pad)
  rows.append({'case_id':p['case_id'],'source_image_sha256':p['image_sha256'],
               'crop_image_path':str(dst.resolve()),'crop_sha256':sha(dst),**info})
 m={'schema':'prd32_skirt5_roi_marked_input_v1','scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',
    'input_variant':'bbox_crop_8pct_padding_red_box' if a.pad==.08 else f'bbox_crop_{a.pad}_padding_red_box',
    'padding_fraction':a.pad,'frozen_plan_sha256':EXPECTED['plan'],
    'baseline_sha256':EXPECTED['baseline'],'original_prompts_sha256':hashes,
    'human_labels_sent_to_model':False,'source_labels_sent_to_model':False,
    'holdout_accessed':False,'cases':rows}
 write_json(a.output/'crop_manifest.json',m)
 write_json(a.output/'preflight_summary.json',{'status':'ROI_PREFLIGHT_OK_NO_GPU','case_count':len(rows),
    'max_roi_area_fraction':max(x['roi_area_fraction'] for x in rows),
    'min_roi_area_fraction':min(x['roi_area_fraction'] for x in rows),
    'original_prompt_hashes':hashes,'source':'TRAIN_PATH_ONLY','data_independence':False})
 print('ROI_PREFLIGHT_OK_NO_GPU',len(rows),'source images; check preview crop JPEGs before GPU')

def verified_crop_manifest(a,plan,prompt_sha):
 mp=a.prepared/'crop_manifest.json';m=read_json(mp)
 if m.get('schema')!='prd32_skirt5_roi_marked_input_v1' or m.get('frozen_plan_sha256')!=EXPECTED['plan'] or m.get('baseline_sha256')!=EXPECTED['baseline'] or m.get('original_prompts_sha256')!=prompt_sha:fail('Crop preflight identity mismatch')
 if len(m.get('cases',[]))!=5 or [r['case_id'] for r in m['cases']]!=IDS:fail('Crop cases changed')
 for p,r in zip(plan,m['cases']):
  cp=Path(r['crop_image_path']).resolve()
  if cp.parent!=(a.prepared/'cropped_marked_images').resolve() or not cp.is_file():fail('Crop location invalid')
  if r['source_image_sha256']!=p['image_sha256'] or sha(cp)!=r['crop_sha256']:fail('Crop identity changed')
  with Image.open(cp) as im:
   if im.size!=(r['crop_width'],r['crop_height']):fail('Crop pixel size modified')
 return m

def run(a):
 gate,plan,baseline,v2,v3,hashes=validate(a)
 m=verified_crop_manifest(a,plan,hashes)
 if a.output.exists():fail('GPU output exists, never overwrite: '+str(a.output))
 from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
 import torch
 if not (a.model/'config.json').is_file():fail('Local model missing')
 if not torch.cuda.is_available():fail('CUDA unavailable')
 free=torch.cuda.mem_get_info()[0]/1024**3
 if free<17:fail(f'Insufficient free VRAM {free:.2f} GiB (need >=17)')
 a.output.mkdir(parents=True)
 write_json(a.output/'gpu_run_config.json',{'schema':'prd32_skirt5_roi_run_v1',
    'mode':'ROI_ONLY_NO_BASELINE_RERUN','prompt_hashes':hashes,
    'crop_manifest_sha256':sha(a.prepared/'crop_manifest.json'),
    'baseline_sha256':EXPECTED['baseline'],'frozen_plan_sha256':EXPECTED['plan'],
    'model_path':str(a.model),'generation_max_new_tokens':170,
    'do_sample':False,'holdout_used':False,'human_gt_sent_to_model':False})
 model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
    str(a.model),local_files_only=True,device_map='auto',
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
 processor=AutoProcessor.from_pretrained(str(a.model),local_files_only=True,max_pixels=512*28*28)
 with (a.output/'roi_predictions_pair.jsonl').open('w',encoding='utf8') as f:
  for i,(p,roi) in enumerate(zip(plan,m['cases'])):
   output={'case_id':p['case_id'],'source_image_sha256':p['image_sha256'],
        'crop_sha256':roi['crop_sha256'],'crop_image_path':roi['crop_image_path'],
        'order':['V2','V3'] if i%2==0 else ['V3','V2']}
   rec={'case_id':p['case_id'],'image_sha256':p['image_sha256'],'image_path':p['image_path'],
        'target_instance_id':p['target_instance_id'],'target_bbox_xyxy':p['bbox'],'task':'tags'}
   try:
    for v in output['order']:
     prompt=v2.TAG_PROMPT if v=='V2' else v3.FULL_TAGS_PROMPT
     raw,timing=v2.run_model_case(model,processor,rec,Path(roi['crop_image_path']),prompt,task_limits={'tags':170})
     cat,valid,native,issues,canonical=gate.clean_tags(v2,v3,raw)
     output[v]={'status':'GENERATED_UNVERIFIED','category':cat,'canonical_tags':canonical,
      'schema_valid':valid,'native_json':native,'issues':issues,
      'raw_model_output':raw,'timing':timing}
   except Exception as e:
    for v in ('V2','V3'):
     if v not in output:output[v]={'status':'FAILED','category':None,'issues':[type(e).__name__+':'+str(e)[:180]]}
   f.write(json.dumps(output,ensure_ascii=False)+'\n');f.flush()
   print(f'[{i+1}/5] {p["case_id"]} crop V2={output["V2"].get("category")} V3={output["V3"].get("category")}',flush=True)
 print('ROI_RUN_DONE 5 paired, now call score',flush=True)

def get_correct(v,category):return v.get('status')=='GENERATED_UNVERIFIED' and v.get('category')==category

def score(a):
 gate,plan,old,v2,v3,hashes=validate(a)
 m=verified_crop_manifest(a,plan,hashes)
 if a.output.exists():fail('Score output path exists; never overwrite')
 cfg=read_json(a.roi_run/'gpu_run_config.json')
 if cfg.get('crop_manifest_sha256')!=sha(a.prepared/'crop_manifest.json') or cfg.get('prompt_hashes')!=hashes:fail('GPU config mismatch')
 pred_path=a.roi_run/'roi_predictions_pair.jsonl';new=read_jsonl(pred_path)
 if len(new)!=5 or [r['case_id'] for r in new]!=IDS:fail('ROI prediction cohort incomplete')
 reference={r['case_id']:r for r in gate.read_jsonl(a.frozen/'frozen_human_reference.jsonl')}
 rows=[]
 for p,roi,rn in zip(plan,m['cases'],new):
  cid=p['case_id'];gold=reference[cid]
  if gold['image_sha256']!=p['image_sha256'] or rn['source_image_sha256']!=p['image_sha256'] or rn['crop_sha256']!=roi['crop_sha256']:fail('ROI/GT mismatch: '+cid)
  if not all(rn.get(ver,{}).get('status')=='GENERATED_UNVERIFIED' for ver in ('V2','V3')):fail('Incomplete GPU inference for '+cid)
  s=bool(gold['scoreable'])
  item={'case_id':cid,'scoreable':s,'human_category':gold['category'],
        'overlap_other_garments':gold['overlap_other_garments'],
        'roi_area_fraction':roi['roi_area_fraction']}
  for ver in ('V2','V3'):
   b=old[cid][ver];n=rn[ver]
   item[ver+'_full_category']=b['category'];item[ver+'_roi_category']=n['category']
   item[ver+'_full_color']=(b.get('canonical_tags') or {}).get('primary_color')
   item[ver+'_roi_color']=(n.get('canonical_tags') or {}).get('primary_color')
   item[ver+'_full_correct']=get_correct(b,gold['category']) if s else ''
   item[ver+'_roi_correct']=get_correct(n,gold['category']) if s else ''
   item[ver+'_changed']=b['category']!=n['category']
   item[ver+'_change_group']=('EXCLUDED' if not s else 'IMPROVED' if not item[ver+'_full_correct'] and item[ver+'_roi_correct']
      else 'REGRESSED' if item[ver+'_full_correct'] and not item[ver+'_roi_correct']
      else 'STILL_CORRECT' if item[ver+'_full_correct'] and item[ver+'_roi_correct'] else 'STILL_WRONG')
  rows.append(item)
 scored=[r for r in rows if r['scoreable']]
 groups={}
 for key in ['V2','V3']:
  groups[key]={'n':len(scored),
    'full_correct':sum(r[key+'_full_correct'] for r in scored),
    'roi_correct':sum(r[key+'_roi_correct'] for r in scored),
    'improved':sum(r[key+'_change_group']=='IMPROVED' for r in scored),
    'regressed':sum(r[key+'_change_group']=='REGRESSED' for r in scored),
    'still_wrong':sum(r[key+'_change_group']=='STILL_WRONG' for r in scored)}
 sens=[r for r in scored if r['case_id']!='skirt_diag_002']
 sensitivity={key:{'n':len(sens),'full_correct':sum(r[key+'_full_correct'] for r in sens),
                   'roi_correct':sum(r[key+'_roi_correct'] for r in sens)} for key in ('V2','V3')}
 result={'schema':'prd32_skirt5_target_crop_ablation_v1',
     'scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY_NOT_INDEPENDENT',
     'input_comparison':'existing full marked image vs new marked target crop',
     'original_prompts_identical':True,'model_rerun_only_roi':True,
     'baseline_sha256':EXPECTED['baseline'],'roi_predictions_sha256':sha(pred_path),
     'frozen_plan_sha256':EXPECTED['plan'],'prompt_hashes':hashes,
     'n_images':5,'n_scoreable':len(scored),'results':groups,
     'sensitivity_excluding_case002':sensitivity,
     'confirmed_error_focus_cases':['skirt_diag_002','skirt_diag_004'],
     'note':'GT confirmed for earlier labels is AI-assisted; case002 remains visually disputed; ROI can remove helpful context. Category changes are not by themselves proof of binding correction. Posthoc visual review required.',
     'not_formal_acceptance':True,'holdout_used':False}
 a.output.mkdir(parents=True)
 write_json(a.output/'roi_comparison_summary.json',result)
 write_csv(a.output/'roi_paired_comparison.csv',rows,list(rows[0]))
 write_json(a.output/'roi_artifact_hashes.json',{'frozen_plan':sha(a.frozen/'frozen_inference_plan.jsonl'),
        'frozen_reference':sha(a.frozen/'frozen_human_reference.jsonl'),
        'baseline_predictions':sha(a.baseline),'crop_manifest':sha(a.prepared/'crop_manifest.json'),
        'roi_predictions':sha(pred_path)})
 print('ROI_SCORE_DONE',json.dumps({'n':len(scored),'V2':groups['V2'],'V3':groups['V3']},ensure_ascii=False))

def selftest():
 import tempfile
 from PIL import Image
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp);src=p/'in.png';dst=p/'out.jpg'
  Image.new('RGB',(100,100),(120,140,170)).save(src)
  d=make_roi(src,[25,20,75,80],dst,.08)
  assert dst.is_file() and d['crop_width']<100 and d['crop_height']<100
  assert d['local_bbox_xyxy'][0]>0 and d['local_bbox_xyxy'][1]>0
  assert sha(dst)==sha(dst)
  try:roi_bounds([5,5,5,30],(100,100))
  except RuntimeError:pass
  else:raise AssertionError('Empty bbox not blocked')
  assert get_correct({'status':'GENERATED_UNVERIFIED','category':'skirt'},'skirt')
  assert not get_correct({'status':'FAILED','category':'skirt'},'skirt')
 print('ROI_ABLATION_SELFTEST_PASS no GPU, no original image reads')

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
 sub.add_parser('selftest')
 for cmd in ('preflight','gpu','score'):
  s=sub.add_parser(cmd)
  s.add_argument('--gate',type=Path,default=GATE)
  s.add_argument('--frozen',type=Path,default=FROZEN)
  s.add_argument('--baseline',type=Path,default=OLD)
  s.add_argument('--allowed-root',type=Path,default=ALLOWED_ROOT)
  s.add_argument('--output',type=Path,required=True)
  if cmd=='preflight':s.add_argument('--pad',type=float,default=.08)
  if cmd in ('gpu','score'):s.add_argument('--prepared',type=Path,required=True)
  if cmd=='gpu':s.add_argument('--model',type=Path,default=MODEL)
  if cmd=='score':s.add_argument('--roi-run',type=Path,required=True)
 a=p.parse_args()
 if a.cmd=='selftest':selftest();return
 if a.cmd=='preflight':
  if not(0.03<=a.pad<=.15):fail('Padding must be between 0.03 and 0.15')
  preflight(a)
 elif a.cmd=='gpu':run(a)
 else:score(a)
if __name__=='__main__':main()
