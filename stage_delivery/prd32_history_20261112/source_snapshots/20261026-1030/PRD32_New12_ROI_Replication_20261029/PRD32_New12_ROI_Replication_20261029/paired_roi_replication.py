#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD32 New12: same-session full-red-box vs 8%-padded ROI-red-box, V2/V3 fixed prompts.

Only approved, frozen TRAIN-source development images. Never reads restricted splits.
No GT/source labels are sent to the model. Historical bboxes are NOT V5 model boxes.
For diagnostics only; not independent data, production gate, or PRD acceptance.
"""
from __future__ import annotations
import argparse,base64,csv,hashlib,html,importlib.util,io,json,math,statistics,sys,time
from pathlib import Path
from PIL import Image,ImageDraw

BASE=Path('/workspace')
GATE=BASE/'PRD32_20261029_Validation_Gate'
FROZEN=BASE/'PRD32_NEWCohort_FROZEN_20261029_run01'
ARCHIVED=BASE/'PRD32_NEWCohort_GPU_20261029_run01'/'predictions_pair.jsonl'
MODEL=BASE/'models/Qwen2.5-VL-7B-Instruct'
ALLOWED_ROOT=BASE/'fashion_data'
EXPECTED={
 'plan':'ba96d1becedeb7bdb3676a0496e1ce9efe89b12a6f34819ba5d97cf3bae2808a',
 'reference':'47b879e28abb37b4c70641cead06b0f8f8ed12674263a754bfc359817e0dda6c',
 'archived':'9760e61f75c1329670b35be55e5b65e78af1b5e7676fad73df8850ca2a2945c8',
 'prompt_v2':'4821bc893d412cffa3a7f8c13709d09b9f3ded05ccfb2a12b72be9298bad6e2b',
 'prompt_v3':'b6541858e4114ff4afb0ace15db7567b36952ccbca32fc59704df32b4807513c'
}
IDS=[f'fresh_dev_{i:03d}' for i in range(1,13)]


def fail(s):raise RuntimeError(s)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def loadj(path):return json.loads(Path(path).read_text('utf8'))
def loadjl(path):return [json.loads(x) for x in Path(path).read_text('utf8').splitlines() if x.strip()]
def dumpj(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def dumpcsv(path,rows):
 with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p)
 if spec is None or spec.loader is None:fail('Cannot import '+str(p))
 x=importlib.util.module_from_spec(spec);sys.modules[name]=x;spec.loader.exec_module(x);return x

def verify(a):
 gate=module('new12_original_gate',a.gate/'run_and_score.py')
 plan,_=gate.verify_frozen(a.frozen,a.allowed_root,check_imgs=True)
 if [x['case_id'] for x in plan]!=IDS:fail('Unexpected cases or order; abort')
 if sha(a.frozen/'frozen_inference_plan.jsonl')!=EXPECTED['plan']:fail('FROZEN plan SHA mismatch; abort')
 if sha(a.frozen/'frozen_human_reference.jsonl')!=EXPECTED['reference']:fail('FROZEN human reference SHA mismatch; abort')
 if sha(a.archived)!=EXPECTED['archived']:fail('ARCHIVED baseline SHA mismatch; abort')
 archived=loadjl(a.archived)
 if len(archived)!=12 or {r['case_id'] for r in archived}!=set(IDS):fail('Archived case count/IDs mismatch')
 old={r['case_id']:r for r in archived}
 for p in plan:
  r=old[p['case_id']]
  if r.get('image_sha256')!=p['image_sha256'] or any(r.get(v,{}).get('status')!='GENERATED_UNVERIFIED' for v in ('V2','V3')):
   fail('Archived identity/status mismatch: '+p['case_id'])
 v2,v3=gate.load_helpers()
 prompts={'V2':hashlib.sha256(v2.TAG_PROMPT.encode()).hexdigest(),
          'V3':hashlib.sha256(v3.FULL_TAGS_PROMPT.encode()).hexdigest()}
 if prompts['V2']!=EXPECTED['prompt_v2'] or prompts['V3']!=EXPECTED['prompt_v3']:
  fail('Original prompts altered; abort')
 return gate,plan,old,v2,v3,prompts

def roi_bbox(bbox,wh,pad=.08):
 w,h=wh;x1,y1,x2,y2=map(float,bbox)
 if not (0<=x1<x2<=w and 0<=y1<y2<=h):fail('Invalid image bbox')
 dx=(x2-x1)*pad;dy=(y2-y1)*pad
 b=(max(0,math.floor(x1-dx)),max(0,math.floor(y1-dy)),min(w,math.ceil(x2+dx)),min(h,math.ceil(y2+dy)))
 if b[2]-b[0]<18 or b[3]-b[1]<18:fail('ROI too small; inspect manually')
 return b

def crop_marked(srcpath,bbox,out,pad=.08):
 with Image.open(srcpath) as src:im=src.convert('RGB')
 l,t,r,b=roi_bbox(bbox,im.size,pad);cropped=im.crop((l,t,r,b))
 x1,y1,x2,y2=bbox
 # Same red rectangle convention as the preceding skirt5 ROI ablation.
 ImageDraw.Draw(cropped).rectangle((round(x1-l),round(y1-t),round(x2-l),round(y2-t)),
                                    outline=(250,40,40),width=max(3,min(cropped.size)//150))
 cropped.save(out,'JPEG',quality=93)
 return {'roi_xyxy':[l,t,r,b],'roi_area_fraction':round(cropped.width*cropped.height/(im.width*im.height),6),
         'width':cropped.width,'height':cropped.height}

def jpg_data_uri(imgpath):
 with Image.open(imgpath) as src:
  im=src.convert('RGB');im.thumbnail((480,460));buf=io.BytesIO();im.save(buf,format='JPEG',quality=75)
 return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')

def preflight(a):
 gate,plan,_,v2,v3,prompts=verify(a)
 if a.output.exists():fail('Output already exists; no overwrite: '+str(a.output))
 a.output.mkdir(parents=True);(a.output/'prepared_images').mkdir()
 meta=[];items=[]
 for p in plan:
  cid=p['case_id'];inp={'case_id':cid,'image_path':p['image_path'],'image_sha256':p['image_sha256'],
       'target_bbox_xyxy':p['bbox'],'target_instance_id':p['target_instance_id'],'task':'tags'}
  # Exact original full-frame marking, reused from frozen helper.
  full=Path(v2.make_marked_image(inp,a.output/'prepared_images','marked_full_image',0.05))
  if full.parent.resolve()!=(a.output/'prepared_images').resolve():fail('Unexpected full marker path')
  roi=a.output/'prepared_images'/f'{cid}_roi_marked.jpg'
  info=crop_marked(p['image_path'],p['bbox'],roi,a.pad)
  d={'case_id':cid,'image_sha256':p['image_sha256'],'full_path':str(full.resolve()),'full_sha256':sha(full),
      'roi_path':str(roi.resolve()),'roi_sha256':sha(roi),**info}
  meta.append(d)
  items.append('<article><h3>'+cid+'</h3><div class="pics"><figure><img src="'+jpg_data_uri(full)+'"><figcaption>FULL · historical frozen marked image</figcaption></figure><figure><img src="'+jpg_data_uri(roi)+'"><figcaption>ROI · 8% padding (not V5 predicted box)</figcaption></figure></div></article>')
 m={'schema':'prd32_new12_roi_same_session_preflight_v1','scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',
    'frozen_plan_sha256':EXPECTED['plan'],'archived_full_predictions_sha256':EXPECTED['archived'],
    'prompts':prompts,'padding':a.pad,'labels_in_model_input':False,'holdout_used':False,'cases':meta}
 dumpj(a.output/'images_manifest.json',m)
 page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>New12 Same-session ROI preview</title><style>body{font:15px system-ui;background:#edf1f7;margin:auto;max-width:1100px;padding:20px}article{background:white;border-radius:10px;margin:15px 0;padding:16px}.pics{display:grid;grid-template-columns:1fr 1fr;gap:14px}figure{margin:0;text-align:center}img{max-width:100%;max-height:470px}figcaption{font-size:12px;color:#667} @media(max-width:680px){.pics{grid-template-columns:1fr}}</style><h1>PRD 3.2 · new 12 images: FULL vs ROI</h1><p>Review all cropped boxes BEFORE GPU. Labels, model outputs and source category are hidden. TRAIN-source DEV only.</p>'+''.join(items)+'</html>'
 (a.output/'Review_Crop_Input_Before_GPU.html').write_text(page,encoding='utf8')
 dumpj(a.output/'preflight_summary.json',{'status':'READY_FOR_SAME_SESSION_GPU','n':len(meta),
    'max_roi_area_fraction':max(r['roi_area_fraction'] for r in meta),
    'min_roi_area_fraction':min(r['roi_area_fraction'] for r in meta),'heldout_opened':False})
 print('NEW12_ROI_PREFLIGHT_OK',len(meta),'images; review Review_Crop_Input_Before_GPU.html')

def verified_manifest(a,plan,prompts):
 m=loadj(a.prepared/'images_manifest.json')
 if m.get('schema')!='prd32_new12_roi_same_session_preflight_v1' or m.get('frozen_plan_sha256')!=EXPECTED['plan'] or m.get('archived_full_predictions_sha256')!=EXPECTED['archived'] or m.get('prompts')!=prompts or m.get('padding')!=.08:
  fail('Input manifest config mismatch')
 if len(m.get('cases',[]))!=12 or [r['case_id'] for r in m['cases']]!=IDS:fail('Input case identity/order changed')
 for p,r in zip(plan,m['cases']):
  if r['image_sha256']!=p['image_sha256']:fail('Source image SHA mismatch')
  for typ in ['full','roi']:
   f=Path(r[typ+'_path']).resolve()
   if f.parent!=(a.prepared/'prepared_images').resolve() or not f.is_file() or sha(f)!=r[typ+'_sha256']:
    fail('Full/ROI image modified or location changed: '+r['case_id']+' '+typ)
 return m

def run_gpu(a):
 gate,plan,_,v2,v3,prompts=verify(a);m=verified_manifest(a,plan,prompts)
 if a.output.exists():fail('GPU output already exists; no overwrite')
 import torch
 from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
 if not (a.model/'config.json').is_file():fail('Model config not found')
 if not torch.cuda.is_available():fail('CUDA unavailable')
 free=torch.cuda.mem_get_info()[0]/1024**3
 if free<17:fail(f'GPU free VRAM {free:.2f} GiB <17 GiB')
 a.output.mkdir(parents=True)
 dumpj(a.output/'run_config.json',{'schema':'prd32_new12_roi_same_session_run_v1',
     'images_manifest_sha256':sha(a.prepared/'images_manifest.json'),'frozen_plan_sha256':EXPECTED['plan'],
     'prompt_hashes':prompts,'model':str(a.model),'max_new_tokens':170,
     'full_vs_roi':'SAME_SESSION_TWO_INPUTS','gt_sent_to_model':False,'holdout_used':False})
 model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(a.model),local_files_only=True,device_map='auto',
     dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
     attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
 proc=AutoProcessor.from_pretrained(str(a.model),local_files_only=True,max_pixels=512*28*28)
 outf=a.output/'predictions_full_roi.jsonl'
 with outf.open('w',encoding='utf8') as f:
  for i,(p,im) in enumerate(zip(plan,m['cases'])):
   cid=p['case_id']
   # Balance FULL-first/ROI-first and V2/V3 order across cases.
   variants=['FULL','ROI'] if i%2==0 else ['ROI','FULL']
   versions=['V2','V3'] if (i//2)%2==0 else ['V3','V2']
   rec={'case_id':cid,'image_sha256':p['image_sha256'],'image_path':p['image_path'],
        'target_instance_id':p['target_instance_id'],'target_bbox_xyxy':p['bbox'],'task':'tags'}
   output={'case_id':cid,'image_sha256':p['image_sha256'],'full_sha256':im['full_sha256'],
           'roi_sha256':im['roi_sha256'],'variant_order':variants,'version_order':versions,'full':{},'roi':{}}
   for typ in variants:
    path=Path(im[typ.lower()+'_path'])
    for v in versions:
     prompt=v2.TAG_PROMPT if v=='V2' else v3.FULL_TAGS_PROMPT
     try:
      raw,timing=v2.run_model_case(model,proc,rec,path,prompt,task_limits={'tags':170})
      cat,valid,native,issues,tags=gate.clean_tags(v2,v3,raw)
      out={'status':'GENERATED_UNVERIFIED','category':cat,'canonical_tags':tags,
           'schema_valid':valid,'native_json':native,'issues':issues,
           'raw_model_output':raw,'timing':timing}
     except Exception as e:
      out={'status':'FAILED','category':None,'issues':[type(e).__name__+':'+str(e)[:180]]}
     output[typ.lower()][v]=out
   f.write(json.dumps(output,ensure_ascii=False)+'\n');f.flush()
   print(f'[{i+1}/12] {cid} FULL:{output["full"]["V2"].get("category")}/{output["full"]["V3"].get("category")} ROI:{output["roi"]["V2"].get("category")}/{output["roi"]["V3"].get("category")}',flush=True)
 print('NEW12_ROI_GPU_DONE 12 images x 2 variants x 2 prompts (48 calls)')

def isgood(obj,cat):return obj.get('status')=='GENERATED_UNVERIFIED' and obj.get('category')==cat

def make_posthoc_html(output_path, rows, records, manifest):
    """Blind pre-GPU review is separate; this post-hoc page displays predictions.

    It is for diagnosing changed/wrong model answers, NOT automated attribution.
    """
    by_id={r['case_id']:r for r in records};images={r['case_id']:r for r in manifest['cases']}
    interesting=[r for r in rows if any(r[v+'_group'] in ('IMPROVED','REGRESSED','STILL_WRONG')
                                         for v in ('V2','V3'))
                 or any(not r[v+'_archived_matches_full'] for v in ('V2','V3'))]
    if not interesting:
        return 0
    blocks=[];metadata=[]
    for item in interesting:
        cid=item['case_id'];pred=by_id[cid];pic=images[cid]
        rows_html=[]
        for v in ('V2','V3'):
            full=pred['full'][v];roi=pred['roi'][v]
            rows_html.append('<div class="pred"><h4>'+v+' · '+html.escape(str(item[v+'_group']))+'</h4>'
                '<p>FULL: <b>'+html.escape(str(full.get('category')))+'</b> · '+html.escape(str((full.get('canonical_tags') or {}).get('primary_color')))
                +'</p><p>ROI: <b>'+html.escape(str(roi.get('category')))+'</b> · '+html.escape(str((roi.get('canonical_tags') or {}).get('primary_color')))
                +'</p><details><summary>模型结构化输出</summary><pre>'+html.escape(json.dumps({'FULL':full.get('canonical_tags'),'ROI':roi.get('canonical_tags')},ensure_ascii=False,indent=2))+'</pre></details>'
                +'<label>'+v+' FULL 关注对象<select class="'+v+'_full_focus"><option value="">未审核</option><option>TARGET</option><option>OTHER</option><option>MIXED</option><option>UNCERTAIN</option></select></label>'
                +'<label>'+v+' ROI 关注对象<select class="'+v+'_roi_focus"><option value="">未审核</option><option>TARGET</option><option>OTHER</option><option>MIXED</option><option>UNCERTAIN</option></select></label></div>')
        blocks.append('<article data-cid="'+cid+'"><h3>'+cid+' · 审核类别：'+html.escape(str(item['human_category']))+(' (计分)' if item['scoreable'] else ' (排除)')+'</h3>'
            '<div class="photos"><img alt="完整红框图" src="'+jpg_data_uri(pic['full_path'])+'"><img alt="ROI红框图" src="'+jpg_data_uri(pic['roi_path'])+'"></div>'
            '<div class="preds">'+''.join(rows_html)+'</div><label>归因证据等级<select class="evidence"><option value="">未审核</option><option>HIGH</option><option>MEDIUM</option><option>LOW</option></select></label>'
            '<label>备注<textarea class="notes" placeholder="判断模型描述的是目标还是旁边的衣物；类别变化不能自动证明绑定已纠正"></textarea></label></article>')
        metadata.append({'case_id':cid,'image_sha256':pic['image_sha256'], 'human_category':item['human_category'],
                         'scoreable':item['scoreable'],'v2_group':item['V2_group'],'v3_group':item['V3_group']})
    js=json.dumps(metadata,ensure_ascii=False).replace('</','<\\/')
    doc=('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>New12 ROI · Posthoc audit</title>'
       '<style>body{max-width:1200px;margin:auto;padding:18px;background:#eef2f7;font:15px/1.5 system-ui}article{background:white;padding:15px;margin:15px 0;border-radius:12px}.photos,.preds{display:grid;grid-template-columns:1fr 1fr;gap:14px}img{width:100%;max-height:470px;object-fit:contain}pre{white-space:pre-wrap;font-size:11px}label{display:block;margin:10px 0}select,textarea{display:block;width:98%;padding:7px}textarea{min-height:65px}button{padding:12px;background:#23548f;color:white;border:0;border-radius:7px}@media(max-width:650px){.photos,.preds{grid-template-columns:1fr}}</style>'
       '<h1>PRD 3.2 · New12 Same-session ROI 错误复核</h1><p>人工审核与模型预测分离。这里是事后复核，不能仅靠类别变化认定实例绑定修复。图片为 TRAIN 来源开发诊断，目标框不是 V5 模型预测。</p>'
       '<button onclick="exportReview()">导出复核 JSON</button><span id="note"></span>'+''.join(blocks)
       +'<script>const META='+js+';function exportReview(){let rows=[];for(const el of document.querySelectorAll("article[data-cid]")){const m=META.find(x=>x.case_id===el.dataset.cid);const v=k=>el.querySelector("."+k).value;rows.push({...m,v2_full_focus:v("V2_full_focus"),v2_roi_focus:v("V2_roi_focus"),v3_full_focus:v("V3_full_focus"),v3_roi_focus:v("V3_roi_focus"),evidence_level:v("evidence"),reviewer_note:v("notes")});}const o={schema:"prd32_new12_roi_posthoc_focus_review_v1",scope:"TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY",rows};const u=URL.createObjectURL(new Blob([JSON.stringify(o,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=u;a.download="PRD32_New12_ROI_FocusReview.json";a.click();setTimeout(()=>URL.revokeObjectURL(u),2500);document.getElementById("note").textContent=" 导出 "+rows.length+" 条";}</script></html>')
    Path(output_path).write_text(doc,encoding='utf8')
    return len(interesting)

def score(a):
 gate,plan,old,v2,v3,prompts=verify(a);m=verified_manifest(a,plan,prompts)
 if a.output.exists():fail('Score output already exists; no overwrite')
 cfg=loadj(a.gpu_run/'run_config.json')
 if cfg.get('images_manifest_sha256')!=sha(a.prepared/'images_manifest.json') or cfg.get('prompt_hashes')!=prompts:
  fail('GPU config SHA/prompt mismatch')
 pp=a.gpu_run/'predictions_full_roi.jsonl';records=loadjl(pp)
 if len(records)!=12 or [r['case_id'] for r in records]!=IDS:fail('GPU results missing/changed')
 reference={r['case_id']:r for r in gate.read_jsonl(a.frozen/'frozen_human_reference.jsonl')}
 rows=[]
 for p,im,r in zip(plan,m['cases'],records):
  cid=p['case_id'];ref=reference[cid]
  if r['image_sha256']!=p['image_sha256'] or r['full_sha256']!=im['full_sha256'] or r['roi_sha256']!=im['roi_sha256'] or ref['image_sha256']!=p['image_sha256']:fail('Image/GT identity mismatch: '+cid)
  for typ in ('full','roi'):
   for v in ('V2','V3'):
    if r.get(typ,{}).get(v,{}).get('status')!='GENERATED_UNVERIFIED':fail('Incomplete inference: '+cid+' '+typ+' '+v)
  sc=bool(ref['scoreable']);gt=ref['category']
  item={'case_id':cid,'scoreable':sc,'human_category':gt,
        'overlap_other_garments':ref['overlap_other_garments'],'boundary_case':ref['boundary_case'],
        'evidence_level':ref['evidence_level'],'roi_area_fraction':im['roi_area_fraction']}
  for v in ('V2','V3'):
   bf=r['full'][v];ro=r['roi'][v];oldv=old[cid][v]
   item[v+'_archived_full_category']=oldv['category'];item[v+'_full_category']=bf['category'];item[v+'_roi_category']=ro['category']
   item[v+'_full_color']=(bf.get('canonical_tags') or {}).get('primary_color');item[v+'_roi_color']=(ro.get('canonical_tags') or {}).get('primary_color')
   item[v+'_full_correct']=isgood(bf,gt) if sc else ''
   item[v+'_roi_correct']=isgood(ro,gt) if sc else ''
   item[v+'_archived_matches_full']=oldv['category']==bf['category']
   item[v+'_group']=('EXCLUDED' if not sc else 'IMPROVED' if not item[v+'_full_correct'] and item[v+'_roi_correct']
      else 'REGRESSED' if item[v+'_full_correct'] and not item[v+'_roi_correct']
      else 'STILL_CORRECT' if item[v+'_full_correct'] and item[v+'_roi_correct'] else 'STILL_WRONG')
  rows.append(item)
 good=[r for r in rows if r['scoreable']]
 groups={}
 for v in ('V2','V3'):
  groups[v]={'n':len(good),'full_correct':sum(x[v+'_full_correct'] for x in good),
   'roi_correct':sum(x[v+'_roi_correct'] for x in good),
   'improved':sum(x[v+'_group']=='IMPROVED' for x in good),
   'regressed':sum(x[v+'_group']=='REGRESSED' for x in good),
   'both_wrong':sum(x[v+'_group']=='STILL_WRONG' for x in good),
   'archived_full_matches_current_full':sum(x[v+'_archived_matches_full'] for x in rows),
   'archived_full_n':len(rows)}
 def stratify(field):
  ans={}
  for name in sorted({x[field] for x in good}):
   a=[x for x in good if x[field]==name]
   ans[name]={v:{'n':len(a),'full_correct':sum(x[v+'_full_correct'] for x in a),
                'roi_correct':sum(x[v+'_roi_correct'] for x in a),
                'regressed':sum(x[v+'_group']=='REGRESSED' for x in a)} for v in ('V2','V3')}
  return ans
 summary={'schema':'prd32_new12_roi_same_session_score_v1','scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY_NOT_INDEPENDENT',
  'n_images':12,'n_scoreable':len(good),'n_excluded':12-len(good),
  'comparison':'SAME_SESSION_FULL_VS_ROI','roi_padding_fraction':.08,
  'frozen_plan_sha256':EXPECTED['plan'],'historical_baseline_sha256':EXPECTED['archived'],
  'gpu_predictions_sha256':sha(pp),'prompt_sha256':prompts,'results':groups,
  'by_human_category':stratify('human_category'), 'by_overlap':stratify('overlap_other_garments'),
  'by_boundary':stratify('boundary_case'), 'no_holdout':True,'not_independent':True,
  'not_PRD_acceptance':True,'bbox_origin':'historical_inventory_gt_pilot_NOT_V5_prediction',
  'caveat':'ROI may remove helpful context; changes in category alone do not prove instance-binding repair. Post-hoc visual review required.'}
 a.output.mkdir(parents=True);dumpj(a.output/'roi_replicated_summary.json',summary)
 dumpcsv(a.output/'roi_replicated_comparison.csv',rows)
 nreview=make_posthoc_html(a.output/'ROI_Change_RootCause_Review.html',rows,records,m)
 dumpj(a.output/'review_page_summary.json',{'cases_to_review':nreview,'review_is_posthoc':True,'scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY'})
 dumpj(a.output/'artifact_hashes.json',{'frozen_plan':sha(a.frozen/'frozen_inference_plan.jsonl'),
  'human_reference':sha(a.frozen/'frozen_human_reference.jsonl'),
  'archived_full_predictions':sha(a.archived),'same_session_predictions':sha(pp),
  'images_manifest':sha(a.prepared/'images_manifest.json')})
 print('NEW12_ROI_SCORE_DONE',json.dumps({'n':len(good),'results':groups},ensure_ascii=False))

def selftest():
 import tempfile
 with tempfile.TemporaryDirectory() as d:
  p=Path(d);src=p/'im.png';roi=p/'roi.jpg'
  Image.new('RGB',(200,120),(150,160,170)).save(src)
  m=crop_marked(src,[50,25,100,90],roi,.08)
  assert roi.is_file() and 0<m['roi_area_fraction']<1
  assert m['width']<200 and m['height']<120
  try:roi_bbox([50,20,50,90],(200,120))
  except RuntimeError:pass
  else:raise AssertionError('Should reject zero-sized box')
  assert isgood({'status':'GENERATED_UNVERIFIED','category':'skirt'},'skirt')
  assert not isgood({'status':'FAILED','category':'skirt'},'skirt')
  x={'V2_full_correct':True,'V2_roi_correct':False}
  assert (x['V2_full_correct'] and not x['V2_roi_correct'])
 print('NEW12_ROI_SELFTEST_PASS; synthetic only, no original images/model read')

def main():
 a=argparse.ArgumentParser(description=__doc__);s=a.add_subparsers(dest='command',required=True)
 s.add_parser('selftest')
 for op in ('preflight','gpu','score'):
  x=s.add_parser(op)
  x.add_argument('--gate',type=Path,default=GATE)
  x.add_argument('--frozen',type=Path,default=FROZEN)
  x.add_argument('--archived',type=Path,default=ARCHIVED)
  x.add_argument('--allowed-root',type=Path,default=ALLOWED_ROOT)
  x.add_argument('--output',type=Path,required=True)
  if op=='preflight':x.add_argument('--pad',type=float,default=.08)
  if op in ('gpu','score'):x.add_argument('--prepared',type=Path,required=True)
  if op=='gpu':x.add_argument('--model',type=Path,default=MODEL)
  if op=='score':x.add_argument('--gpu-run',type=Path,required=True)
 p=a.parse_args()
 if p.command=='selftest':selftest();return
 if p.command=='preflight':
  if p.pad!=.08:fail('This frozen extension only supports .08 padding for comparison')
  preflight(p)
 elif p.command=='gpu':run_gpu(p)
 else:score(p)
if __name__=='__main__':main()
