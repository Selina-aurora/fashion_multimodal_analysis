#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run original V2/V3 prompts as frozen paired controls on an adjudicated NEW train-dev cohort.

GPU only in `run`; `preflight`, `score`, `rootcause-review` are GPU-free.
"""
from __future__ import annotations
import argparse,collections,csv,html,hashlib,importlib.util,io,json,math,statistics,time,sys
from pathlib import Path
from PIL import Image,ImageOps

HERE=Path(__file__).resolve().parent
PREP=HERE/'prepare_validation.py'
ROOT='/workspace/fashion_data'
BANNED={'holdout','sealed','fullval','val','valid','validation','test','testing','testset','eval','evaluation'}
CATS={'top','pants','skirt','outerwear','dress','shoe','bag','accessory'}

def err(msg):raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def read_jsonl(p):return [json.loads(l) for l in Path(p).read_text(encoding='utf8').splitlines() if l.strip()]
def write_json(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def write_csv(p,rows,cols):
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
def local_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
 return mod

def allowed(p,root):
 p=Path(p).resolve();root=Path(root).resolve()
 if not p.is_relative_to(root):return False
 seg={q.lower() for q in p.parts}
 return 'train' in seg and not any(x in BANNED or any(x.startswith(t+'_') or x.endswith('_'+t) for t in BANNED) for x in seg)

def verify_frozen(fro,allowed_root,check_imgs=True):
 root=Path(fro).resolve(); cfg=json.loads((root/'freeze.json').read_text(encoding='utf8'))
 planp=root/'frozen_inference_plan.jsonl'; rp=root/'frozen_human_reference.jsonl'
 if cfg.get('status')!='FROZEN_DEV_NOT_INDEPENDENT':err('Not an expected train-dev frozen plan')
 if sha(planp)!=cfg.get('plan_sha256') or sha(rp)!=cfg.get('human_reference_sha256'):err('Frozen file hash mismatch: abort')
 plan=read_jsonl(planp)
 if len(plan)!=cfg['total'] or not 1<=len(plan)<=50:err('Frozen count invalid')
 ids=set();hashes=set()
 for r in plan:
  cid=r.get('case_id');p=r.get('image_path')
  if cid in ids or r['image_sha256'] in hashes:err('Duplicate case/image')
  ids.add(cid);hashes.add(r['image_sha256'])
  if not allowed(p,allowed_root):err('Forbidden source split/path: '+str(p))
  if r.get('bbox_origin')!='historical_inventory_gt_pilot_NOT_V5_prediction':err('Unexpected bbox origin')
  b=r['bbox']
  if len(b)!=4 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in b) or not(0<=b[0]<b[2] and 0<=b[1]<b[3]):err('Invalid bbox: '+cid)
  if check_imgs:
   if not Path(p).is_file() or sha(p)!=r['image_sha256']:err('Image hash mismatch: '+cid)
   with Image.open(p) as img:
    w,h=img.size
    if w*h>45_000_000 or b[2]>w or b[3]>h:err('Image or box out of range: '+cid)
    if img.getexif().get(274,1) not in (1,None):err('Rotated EXIF image unsupported: '+cid)
    img.verify()
 return plan,cfg

def load_helpers():
 v2=local_module('prd32_original_v2_frozen',HERE/'frozen_helpers'/'prd32_integrated_pipeline_20261027.py')
 v3=local_module('prd32_original_v3_frozen',HERE/'frozen_helpers'/'prd32_v3_class_grounding.py')
 if not v2.TAG_PROMPT or not v3.FULL_TAGS_PROMPT:err('Missing frozen prompts')
 return v2,v3

def clean_tags(v2,v3,raw):
 # Same service-normalization logic used by V3, including invalid-pattern abstention.
 try:
  info,canonical=v3.interpret(v2,'full_tags',raw)
  cat=info.get('category')
  if cat not in CATS and cat is not None:cat=None
  return cat,info.get('schema_valid',False),info.get('native_json',False),info.get('issues',[]),canonical
 except Exception as exc:
  return None,False,False,['PARSE_EXCEPTION:'+type(exc).__name__],None

def preflight(a):
 plans,cfg=verify_frozen(a.frozen,a.allowed_root)
 v2,v3=load_helpers()
 summary={'schema':'prd32_newdev_fixed_pair_preflight_v1','status':'READY_NO_GPU',
  'case_count':len(plans),'v2_prompt_sha256':hashlib.sha256(v2.TAG_PROMPT.encode()).hexdigest(),
  'v3_prompt_sha256':hashlib.sha256(v3.FULL_TAGS_PROMPT.encode()).hexdigest(),
  'freeze_sha256':sha(Path(a.frozen)/'freeze.json'),'identities_checked':len(plans),
  'same_input_marking':'full_image_red_bbox','max_new_tokens':170,
  'human_gt_sent_to_model':False,'source_category_sent_to_model':False,
  'holdout_used':False,'scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY'}
 print('PREFLIGHT_OK',json.dumps(summary,ensure_ascii=False))
 if a.cmd=='preflight' and a.output:
  out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'preflight_summary.json',summary)
 return plans,summary,v2,v3

def run(a):
 plans,cfg,v2,v3=preflight(a)
 out=Path(a.output).resolve()
 if out.exists():err('Output exists: '+str(out))
 # No GPU loads before all image identities/paths and hashes are verified.
 from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
 import torch
 modelpath=Path(a.model)
 if not (modelpath/'config.json').is_file():err('Local model missing config.json')
 if not torch.cuda.is_available():err('CUDA not available')
 free=torch.cuda.mem_get_info()[0]/1024**3
 if free<17:err(f'Insufficient GPU memory {free:.2f} GiB; require >=17 GiB free')
 out.mkdir(parents=True);(out/'marked_images').mkdir()
 write_json(out/'run_config.json',{**cfg,'model':str(modelpath),
        'prompt_source':'frozen original V2/V3 20261027-28','model_load_started':True})
 start=time.perf_counter()
 model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(modelpath),local_files_only=True,device_map='auto',
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
 processor=AutoProcessor.from_pretrained(str(modelpath),local_files_only=True,max_pixels=512*28*28)
 load_s=round(time.perf_counter()-start,2)
 outputs=[]
 with (out/'predictions_pair.jsonl').open('w',encoding='utf8') as f:
  for i,p in enumerate(plans):
   cid=p['case_id'];msg={'case_id':cid,'image_sha256':p['image_sha256'],'scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',
      'bbox_origin':p['bbox_origin'],'order': ['V2','V3'] if i%2==0 else ['V3','V2']}
   inp={'case_id':cid,'image_path':p['image_path'],'image_sha256':p['image_sha256'],
       'target_bbox_xyxy':p['bbox'],'target_instance_id':p['target_instance_id'],'task':'tags'}
   try:
    pic=v2.make_marked_image(inp,out/'marked_images','marked_full_image',0.05)
    msg['marked_image']=str(pic)
    for key in msg['order']:
     prompt=v2.TAG_PROMPT if key=='V2' else v3.FULL_TAGS_PROMPT
     raw,t=v2.run_model_case(model,processor,inp,pic,prompt,task_limits={'tags':170})
     cat,valid,native,issues,canonical=clean_tags(v2,v3,raw)
     msg[key]={'status':'GENERATED_UNVERIFIED','category':cat,'canonical_tags':canonical,
           'schema_valid':valid,'native_json':native,'issues':issues,
           'raw_model_output':raw,'timing':t}
   except Exception as exc:
    for key in ['V2','V3']:
     if key not in msg:
      msg[key]={'status':'FAILED','category':None,'canonical_tags':None,'schema_valid':False,'native_json':False,
        'issues':['RUNTIME_ERROR:'+type(exc).__name__+':'+str(exc)[:150]]}
   f.write(json.dumps(msg,ensure_ascii=False)+'\n');f.flush();outputs.append(msg)
   print(f'[{i+1}/{len(plans)}] {cid} V2={msg["V2"]["category"]} V3={msg["V3"]["category"]}',flush=True)
 write_json(out/'run_summary.json',{'schema':'prd32_newdev_fixed_pair_run_v1','count':len(outputs),
  'complete_pairs':sum(all(p[k]['status']=='GENERATED_UNVERIFIED' for k in ['V2','V3']) for p in outputs),
  'model_load_seconds':load_s,'human_labels_in_model_input':False,'source_hint_in_model_input':False,
  'holdout_used':False,'scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY','model_config':str(modelpath),
  'frozen_plan_sha256':sha(Path(a.frozen)/'frozen_inference_plan.jsonl')})
 print('RUN_DONE',len(outputs),'paired model runs',flush=True)

def score(a):
 plan,cfg=verify_frozen(a.frozen,a.allowed_root)
 reference=read_jsonl(Path(a.frozen)/'frozen_human_reference.jsonl')
 pred=read_jsonl(a.predictions)
 if len(pred)!=len(plan) or {p['case_id'] for p in pred}!={p['case_id'] for p in plan}:err('Predictions must exactly cover frozen cohort')
 pri={r['case_id']:r for r in pred};ref={r['case_id']:r for r in reference}; report=[]
 for p in plan:
  cid=p['case_id'];v=pri[cid];r=ref[cid]
  if v.get('image_sha256')!=p['image_sha256'] or r.get('image_sha256')!=p['image_sha256']:err('SHA mismatch: '+cid)
  human=r['category'];s=bool(r['scoreable'])
  a2=v['V2'];a3=v['V3'];p2=a2.get('category');p3=a3.get('category')
  c2=s and a2.get('status')=='GENERATED_UNVERIFIED' and p2==human
  c3=s and a3.get('status')=='GENERATED_UNVERIFIED' and p3==human
  group='EXCLUDED' if not s else ('BOTH_CORRECT' if c2 and c3 else 'V2_ONLY' if c2 else 'V3_ONLY' if c3 else 'BOTH_WRONG')
  report.append({'case_id':cid,'image_sha256':p['image_sha256'],'human_category':human,
   'overlap':r['overlap_other_garments'],'boundary_case':r['boundary_case'],
   'evidence_level':r['evidence_level'],'scoreable':s,'v2_category':p2,'v3_category':p3,
   'v2_correct':c2 if s else '', 'v3_correct':c3 if s else '', 'pair_group':group,
   'v2_color':(a2.get('canonical_tags') or {}).get('primary_color'),
   'v3_color':(a3.get('canonical_tags') or {}).get('primary_color'),
   'v2_design':(a2.get('canonical_tags') or {}).get('visible_design'),
   'v3_design':(a3.get('canonical_tags') or {}).get('visible_design')})
 scoreable=[v for v in report if v['scoreable']]; n=len(scoreable)
 bygroup=collections.Counter(v['pair_group'] for v in report)
 v2correct=sum(v['v2_correct'] for v in scoreable);v3correct=sum(v['v3_correct'] for v in scoreable)
 def strata(key):
  groups={}
  for typ in sorted({r[key] for r in scoreable}):
   ss=[r for r in scoreable if r[key]==typ]
   groups[typ]={'n':len(ss),'v2_correct':sum(x['v2_correct'] for x in ss), 'v3_correct':sum(x['v3_correct'] for x in ss)}
  return groups
 def latencies(ver):
  a=[]
  for row in pred:
   ob=row.get(ver,{})
   if ob.get('status')=='GENERATED_UNVERIFIED':
    x=(ob.get('timing') or {}).get('end_to_end_ms')
    if isinstance(x,(int,float)):a.append(x)
  return {'n':len(a),'median_ms':round(statistics.median(a),2) if a else None,
      'p95_ms':round(sorted(a)[math.ceil(.95*len(a))-1],2) if a else None}
 summary={'schema':'prd32_newdev_fixed_pair_posthoc_score_v1','scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',
  'total_reviewed':len(report),'scoreable':n,'excluded':len(report)-n,
  'v2_match':v2correct,'v3_match':v3correct,
  'paired_outcomes':dict(bygroup),
  'by_overlap':strata('overlap'),'by_boundary':strata('boundary_case'),
  'by_human_category':strata('human_category'),
  'latency':{'V2':latencies('V2'),'V3':latencies('V3')},
  'not_independent':True,'cannot_claim_PRD_acceptance':True,
  'failure_cause_requires_posthoc_visual_annotation':True}
 out=Path(a.output)
 if out.exists():err('Output exists')
 out.mkdir(parents=True);write_json(out/'comparison_summary.json',summary)
 fields=list(report[0].keys()) if report else []
 write_csv(out/'paired_comparison.csv',report,fields)
 write_json(out/'frozen_artifact_checks.json',{'plan_sha256':cfg['plan_sha256'],'human_reference_sha256':cfg['human_reference_sha256'],
  'predictions_sha256':sha(a.predictions),'scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY'})
 print('SCORE_DONE',json.dumps({'scored':n,'v2_match':v2correct,'v3_match':v3correct,'paired':dict(bygroup)},ensure_ascii=False))
 return report,summary

def review_errors(a):
 plan,cfg=verify_frozen(a.frozen,a.allowed_root)
 ref={r['case_id']:r for r in read_jsonl(Path(a.frozen)/'frozen_human_reference.jsonl')}
 preds={r['case_id']:r for r in read_jsonl(a.predictions)}
 cases=[]
 for p in plan:
  cid=p['case_id'];rr=ref[cid];pr=preds[cid]
  if pr.get('image_sha256')!=p['image_sha256']:err('Image hash mismatched')
  if not rr['scoreable']:continue
  if pr['V2'].get('category')==rr['category'] and pr['V3'].get('category')==rr['category']:continue
  imagepath=Path(pr['marked_image'])
  # Created in existing paired run; ensure intended file resides inside its marked_images dir.
  if not imagepath.is_file() or imagepath.parent.name!='marked_images':err('Generated marked image not found:'+str(imagepath))
  with Image.open(imagepath) as img:
   small=img.copy();small.thumbnail((500,520));bio=io.BytesIO();small.save(bio,'JPEG',quality=78)
  imgurl='data:image/jpeg;base64,'+__import__('base64').b64encode(bio.getvalue()).decode('ascii')
  tags=[]
  for ver in ('V2','V3'):
   o=pr[ver];s=html.escape(json.dumps(o.get('canonical_tags'),ensure_ascii=False))
   tags.append(f'<div><b>{ver}: {html.escape(str(o.get("category")))}</b><pre>{s}</pre></div>')
  cases.append({'case_id':cid,'image_sha256':p['image_sha256'], 'human_category':rr['category'],
       'html':f'<article data-id="{cid}"><h2>{cid}</h2><div class="photos"><img src="{imgurl}" alt="目标红框图"><div>{"".join(tags)}</div></div>\n'
          '<div class="cols">'
          '<label>V2 识别对象<select class="v2_focus"><option value="">请选择</option><option value="TARGET">TARGET</option><option value="OTHER">OTHER</option><option value="MIXED">MIXED</option><option value="UNCERTAIN">UNCERTAIN</option></select></label>'
          '<label>V3 识别对象<select class="v3_focus"><option value="">请选择</option><option value="TARGET">TARGET</option><option value="OTHER">OTHER</option><option value="MIXED">MIXED</option><option value="UNCERTAIN">UNCERTAIN</option></select></label>'
          '<label>V2 根因<select class="v2_cause"><option value="">请选择</option><option value="NOT_ERROR">NOT_ERROR</option><option value="BINDING">BINDING</option><option value="CLASS_BOUNDARY">CLASS_BOUNDARY</option><option value="BOTH">BOTH</option><option value="UNCERTAIN">UNCERTAIN</option></select></label>'
          '<label>V3 根因<select class="v3_cause"><option value="">请选择</option><option value="NOT_ERROR">NOT_ERROR</option><option value="BINDING">BINDING</option><option value="CLASS_BOUNDARY">CLASS_BOUNDARY</option><option value="BOTH">BOTH</option><option value="UNCERTAIN">UNCERTAIN</option></select></label>'
          '<label>证据等级<select class="evidence"><option value="">请选择</option><option value="HIGH">HIGH</option><option value="MEDIUM">MEDIUM</option><option value="LOW">LOW</option></select></label>'
          '<label class="wide">视觉证据<textarea class="notes" placeholder="逐一说明目标颜色/结构与模型描述的服饰匹配关系；不要仅凭是否重叠归因为绑定错误"></textarea></label></div></article>'})
 if not cases:print('NO_ERROR_CASES');return
 meta=json.dumps([{k:v for k,v in q.items() if k!='html'} for q in cases],ensure_ascii=False)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PRD32 root cause — NEW cohort</title><style>body{font:15px/1.5 system-ui,'Microsoft YaHei',sans-serif;background:#edf1f5;color:#213248;margin:0}main{max-width:1150px;margin:auto;padding:20px}article{padding:22px;background:white;border:1px solid #d0dbe7;border-radius:12px;margin:18px 0}.photos{display:grid;grid-template-columns:1fr 1fr;gap:18px}.photos img{max-width:100%;max-height:550px;object-fit:contain}.cols{display:grid;grid-template-columns:1fr 1fr;gap:13px}.cols label{display:flex;flex-direction:column}.cols select,.cols textarea{padding:9px;border:1px solid #bac6d5;border-radius:7px}.wide{grid-column:span 2}.notes{height:80px}pre{white-space:pre-wrap}button{padding:12px 18px;background:#195fa4;color:white;border:0;border-radius:7px}@media(max-width:680px){.photos,.cols{grid-template-columns:1fr}}</style><main><h1>新开发样本：模型错误根因复核</h1><p>此页仅用于对已经发生的错误做事后归因。类别和实例归属要分别判断，不能用图片重叠推断模型一定认错对象。</p><button onclick="download()">导出 JSON</button><span id="stat"></span>'''+''.join(c['html'] for c in cases)+'''</main><script>const METADATA=__DATA__;function download(){const rows=[];for(const el of document.querySelectorAll('article[data-id]')){const m=METADATA.find(x=>x.case_id===el.dataset.id);const v=k=>el.querySelector('.'+k).value.trim();rows.push({...m,v2_focus:v('v2_focus'),v3_focus:v('v3_focus'),v2_root_cause:v('v2_cause'),v3_root_cause:v('v3_cause'),evidence_level:v('evidence'),reviewer_note:v('notes')});}const obj={schema:'prd32_newdev_rootcause_review_v1',scope:'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',rows};const a=document.createElement('a'),u=URL.createObjectURL(new Blob([JSON.stringify(obj,null,2)],{type:'application/json'}));a.href=u;a.download='PRD32_NewDev_RootCause_Review.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),2500);document.getElementById('stat').textContent=' '+rows.length+' 条';}</script></html>'''
 out=Path(a.output)
 if out.exists():err('Output exists')
 out.write_text(page.replace('__DATA__',meta.replace('</','<\\/')),encoding='utf8')
 print('ROOTCAUSE_REVIEW_READY',len(cases),'HTML',out)

def main():
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='cmd',required=True)
 for name in ('preflight','run','score','rootcause-review'):
  p=sub.add_parser(name);p.add_argument('--frozen',required=True);p.add_argument('--allowed-root',default=ROOT);p.add_argument('--output',required=True)
  if name=='run':p.add_argument('--model',default='/workspace/models/Qwen2.5-VL-7B-Instruct')
  if name in {'score','rootcause-review'}:p.add_argument('--predictions',required=True)
 a=parser.parse_args()
 if a.cmd=='preflight':preflight(a)
 elif a.cmd=='run':run(a)
 elif a.cmd=='score':score(a)
 else:review_errors(a)
if __name__=='__main__':main()
