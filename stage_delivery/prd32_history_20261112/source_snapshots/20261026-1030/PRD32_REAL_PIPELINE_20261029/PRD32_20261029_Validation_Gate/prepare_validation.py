#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD32: source-protected, new-to-40 train-development cohort and blind review.

Never reads restricted-image bytes. This is NOT independent PRD acceptance.
"""
from __future__ import annotations
import argparse,base64,csv,hashlib,html,io,json,math,re,sys
from collections import Counter
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw

HERE=Path(__file__).resolve().parent
CATS=('top','pants','skirt','outerwear','dress','shoe','bag','accessory')
RESTRICTED={'holdout','sealed','fullval','val','valid','validation','test','testing','testset','eval','evaluation'}
FORBIDDEN_SUFFIXES={'.zip','.npy','.npz','.pth','.pkl','.pt','.json','.txt'}

def die(msg):raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def write_json(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def read_csv(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(path,rows,fields):
 with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def safe_image_path(source,project,data):
 if not source:return None,'EMPTY_SOURCE'
 p=(project/source).resolve(strict=False)
 if not p.is_relative_to(data):return None,'OUTSIDE_ALLOWED_DATA_ROOT'
 seg=[x.lower() for x in p.parts]
 if any(x in RESTRICTED or any(x.startswith(t+'_') or x.endswith('_'+t) for t in RESTRICTED) for x in seg):return None,'RESTRICTED_SPLIT'
 if 'train' not in seg:return None,'NO_EXPLICIT_TRAIN_SPLIT'
 if p.suffix.lower() not in {'.jpg','.jpeg','.png','.webp','.bmp'}:return None,'NOT_IMAGE_EXTENSION'
 if not p.is_file():return None,'MISSING_TRAIN_FILE'
 return p,''
def bbox_from(r):
 try:b=[float(r['bbox_'+n]) for n in ('x1','y1','x2','y2')]
 except (KeyError,ValueError,TypeError):return None
 if not all(map(math.isfinite,b)) or not(0<=b[0]<b[2] and 0<=b[1]<b[3]):return None
 return b
def preview(im_path,bbox,fullmax=600,cropmax=330):
 with Image.open(im_path) as raw:
  if raw.getexif().get(274,1) not in (1,None):raise ValueError('NONTRIVIAL_EXIF_ROTATION_NOT_SUPPORTED')
  im=raw.convert('RGB')
 w,h=im.size
 if w*h>45_000_000 or bbox[2]>w or bbox[3]>h:raise ValueError('BBOX_OUTSIDE_IMAGE_OR_OVERSIZE')
 with Image.open(im_path) as raw:
  im=raw.convert('RGB')
  full=im.copy(); draw=ImageDraw.Draw(full)
  draw.rectangle(bbox,outline=(235,35,35),width=max(3,min(w,h)//150))
  cropped=im.crop((math.floor(bbox[0]),math.floor(bbox[1]),math.ceil(bbox[2]),math.ceil(bbox[3])))
 def enc(image,largest):
  image.thumbnail((largest,largest));b=io.BytesIO();image.save(b,format='JPEG',quality=78,optimize=True)
  return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode('ascii')
 return enc(full,fullmax),enc(cropped,cropmax),(w,h)
def old40_hashes():
 rows=read_csv(HERE/'evidence'/'HumanEval40_Comparison.csv')
 h={r['image_sha256'].strip().lower() for r in rows}
 if len(rows)!=40 or len(h)!=40 or not all(re.fullmatch('[a-f0-9]{64}',x) for x in h):die('Bundled historical 40-cohort hash registry invalid')
 return h

def candidate_list(project,data,inventory,per_class,extra_hashes=None):
 seen={}; exclusions=[]; old=old40_hashes();extra=extra_hashes or set()
 for idx,r in enumerate(read_csv(inventory),1):
  category=(r.get('garment_category') or '').strip().lower()
  src=(r.get('source_image') or '').strip()
  path,code=safe_image_path(src,project,data)
  if category not in CATS:code=code or 'BAD_CLASS_HINT'
  b=bbox_from(r)
  if b is None:code=code or 'BAD_BBOX_METADATA'
  if code:
   exclusions.append({'row_index':idx,'reason':code});continue
  try:
   digest=sha(path)
  except OSError:
   exclusions.append({'row_index':idx,'reason':'TRAIN_HASH_READ_ERROR'});continue
  if digest in old:
   exclusions.append({'row_index':idx,'reason':'PREVIOUS_40_HASH_MATCH'});continue
  if digest in extra:
   exclusions.append({'row_index':idx,'reason':'ADDITIONAL_EXCLUSION_HASH_MATCH'});continue
  if digest in seen:
   exclusions.append({'row_index':idx,'reason':'DUPLICATE_IMAGE_HASH'});continue
  seen[digest]={'sha256':digest,'source_path':str(path),'source_relative':src,
           'bbox':b,'source_record_id':r.get('record_id',''),
           'source_garment_id':r.get('garment_id',''), 'inventory_hint_NOT_GT':category,'row_index':idx}
 groups={k:[] for k in CATS}
 for r in seen.values():
  groups[r['inventory_hint_NOT_GT']].append(r)
 chosen=[]
 for cat in CATS:
  # Repeatable hash order independent of file-manifest row order
  order=sorted(groups[cat],key=lambda r:hashlib.sha256(('20261029_FIXED_COHORT|'+r['sha256']).encode()).hexdigest())
  chosen.extend(order[:per_class])
 chosen.sort(key=lambda r:(CATS.index(r['inventory_hint_NOT_GT']),r['sha256']))
 for i,r in enumerate(chosen,1):r['case_id']=f'fresh_dev_{i:03d}'
 return chosen,exclusions,{x:len(v) for x,v in groups.items()},old

def build_review_html(selected,path):
 # Crucial: no source category or model prediction rendered to reviewer.
 body=[];meta=[]
 for r in selected:
  cid=r['case_id'];full=r['marked_preview'];crop=r['crop_preview']
  body.append(f'''<article data-cid="{cid}"><header><strong>{cid}</strong> <span>盲审 · 无来源标签/模型答案</span></header>
  <div class="pictures"><figure><img alt="红框原图" src="{full}"><figcaption>原图红框：目标位置与重叠情况</figcaption></figure><figure><img alt="局部框图" src="{crop}"><figcaption>局部视图：只用于确认细节，不是模型输入</figcaption></figure></div>
  <section class="fields">
  <label>人工目标类别<select class="human_category"><option value="">请选择</option>{''.join(f'<option value="{x}">{x}</option>' for x in (*CATS,'uncertain'))}</select></label>
  <label>框与目标对应？<select class="bbox_status"><option value="">请选择</option><option value="YES">YES</option><option value="NO">NO</option><option value="UNCERTAIN">UNCERTAIN</option></select></label>
  <label>是否能可靠识别类别？<select class="visually_identifiable"><option value="">请选择</option><option value="YES">YES</option><option value="NO">NO</option></select></label>
  <label>框内存在其他服饰？<select class="overlap_other_garments"><option value="">请选择</option><option value="YES">YES</option><option value="NO">NO</option><option value="UNCERTAIN">UNCERTAIN</option></select></label>
  <label>主要类别边界难点<select class="boundary_case"><option value="">请选择</option><option value="CLEAR">CLEAR</option><option value="TOP_OUTERWEAR">TOP_OUTERWEAR</option><option value="TOP_DRESS">TOP_DRESS</option><option value="SKIRT_PANTS">SKIRT_PANTS</option><option value="OTHER_BOUNDARY">OTHER_BOUNDARY</option></select></label>
  <label>证据等级<select class="evidence_level"><option value="">请选择</option><option value="HIGH">HIGH</option><option value="MEDIUM">MEDIUM</option><option value="LOW">LOW</option></select></label>
  <label class="wide">人工备注<textarea class="reviewer_note" placeholder="说明目标颜色、结构、服饰遮挡、类别边界等，不要查看模型预测"></textarea></label></section></article>''')
  meta.append({'case_id':cid,'image_sha256':r['sha256']})
 data=json.dumps(meta,ensure_ascii=False).replace('</','<\\/')
 doc='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PRD 3.2 NEW TRAIN-DEV · Blind Review</title>
<style>body{background:#eef2f7;color:#1c2a40;font:15px/1.5 system-ui,'Microsoft YaHei',sans-serif;margin:0}main{max-width:1140px;margin:auto;padding:22px}h1{font-size:24px}article{background:white;border:1px solid #cbd6e3;border-radius:14px;margin:20px 0;padding:16px}article header{display:flex;gap:14px;align-items:center}article header span{font-size:12px;color:#69788d}.pictures{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}.pictures figure{margin:0;background:#f5f7fa;padding:12px;text-align:center;border-radius:10px}.pictures img{max-width:100%;max-height:390px;object-fit:contain}.fields{display:grid;grid-template-columns:1fr 1fr;gap:9px}.fields label{display:flex;flex-direction:column;gap:4px;font-weight:550}.fields select,.fields textarea{border:1px solid #9caec3;border-radius:7px;padding:9px;font:inherit}.wide{grid-column:span 2}.fields textarea{height:70px}button{background:#1f508f;color:white;border:0;border-radius:7px;padding:12px 16px;cursor:pointer}.head{background:#162944;color:white;padding:18px;border-radius:12px}@media(max-width:650px){.pictures,.fields{grid-template-columns:1fr}.wide{grid-column:span 1}}</style></head><body><main><div class="head"><h1>PRD 3.2 新训练来源样本 · 盲审</h1><p>只用于开发诊断，不是独立 PRD 验收。请先标注目标及遮挡，再看预测；来源类别和模型预测未在本页展示。</p><button onclick="exportData()">导出人工审核 JSON</button> <span id="msg"></span></div>'''+''.join(body)+'''</main><script>const META=__META__;
function exportData(){const rows=[];for(const el of document.querySelectorAll('article[data-cid]')){const m=META.find(x=>x.case_id===el.dataset.cid);const val=k=>el.querySelector('.'+k).value.trim();rows.push({...m,human_category:val('human_category'),bbox_status:val('bbox_status'),visually_identifiable:val('visually_identifiable'),overlap_other_garments:val('overlap_other_garments'),boundary_case:val('boundary_case'),evidence_level:val('evidence_level'),reviewer_note:val('reviewer_note')});}const obj={schema:'prd32_new_train_dev_blind_review_v1',scope:'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',rows};const a=document.createElement('a');const u=URL.createObjectURL(new Blob([JSON.stringify(obj,null,2)],{type:'application/json'}));a.href=u;a.download='PRD32_NewDev_BlindReview_20261029.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),2500);document.getElementById('msg').textContent='已触发导出 '+rows.length+' 条';}</script></body></html>'''
 path.write_text(doc.replace('__META__',data),encoding='utf-8')

def do_prepare(a):
 project=Path(a.project_root).resolve();data=Path(a.data_root).resolve()
 inv=Path(a.inventory).resolve() if a.inventory else project/'configs/prd_8class_unified_manifest_v2.csv'
 if not inv.is_file():die(f'Inventory not found: {inv}')
 extras=set()
 if a.exclude_hashes:
  for line in Path(a.exclude_hashes).read_text(encoding='utf8').splitlines():
   v=line.strip().lower()
   if v and not re.fullmatch('[a-f0-9]{64}',v):die('Invalid additional exclusion hash line')
   if v:extras.add(v)
 chosen,excluded,pool_counts,old=candidate_list(project,data,inv,a.per_class,extras)
 out=Path(a.output).resolve()
 if out.exists():die(f'Output already exists: {out}')
 out.mkdir(parents=True)
 valid=[];fails=[]
 for r in chosen:
  try:
   # Read only explicitly allowed TRAIN-source image with passed SHA and new-to-40 check.
   full,crop,size=preview(Path(r['source_path']),r['bbox'])
   valid.append(dict(r,marked_preview=full,crop_preview=crop,width=size[0],height=size[1]))
  except Exception as exc:fails.append({'case_id':r['case_id'],'reason':str(exc)[:120]})
 if valid:build_review_html(valid,out/'Blind_Review_NewTrainDev.html')
 clean=[]
 for r in valid:
  clean.append({k:v for k,v in r.items() if not k.endswith('_preview')})
 with (out/'candidate_manifest.jsonl').open('w',encoding='utf8') as f:
  for r in clean:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 write_csv(out/'quarantined_metadata.csv',excluded,['row_index','reason'])
 write_json(out/'summary.json',{
  'schema':'prd32_new_train_source_development_preparation_v1',
  'status':'BLIND_REVIEW_READY' if valid else 'NO_NEW_TRAIN_DEV_CANDIDATES',
  'requested_per_class':a.per_class,'pool_counts_by_source_hint_NOT_GT':pool_counts,
  'selected_per_source_hint_NOT_GT':dict(Counter(x['inventory_hint_NOT_GT'] for x in valid)),
  'selected_count':len(valid),'selection_failures':fails,
  'excluded_metadata_count':len(excluded),
  'previous_40_image_hashes_excluded':len(old),
  'new_to_previous_40_hashes_only':True,
  'other_prior_data_split_overlap_cannot_be_proven':True,
  'source':'TRAIN_PATH_ONLY','holdout_images_opened':0,'model_calls':0,'official_accuracy':None,
  'eligibility':'TRAIN_SOURCE_DEVELOPMENT_ONLY_NOT_INDEPENDENT',
  'manifest_sha256':sha(inv),
  'candidate_manifest_sha256':sha(out/'candidate_manifest.jsonl')})
 print('PREPARE_OK',json.dumps({'selected':len(valid),'by_class_hint':dict(Counter(x['inventory_hint_NOT_GT'] for x in valid)),'preview_failures':len(fails),'out':str(out)},ensure_ascii=False))

def do_freeze(a):
 src=Path(a.prepared).resolve(); review=Path(a.review).resolve();out=Path(a.output).resolve()
 if out.exists():die('Output exists; do not overwrite frozen cohort')
 rows=[json.loads(l) for l in (src/'candidate_manifest.jsonl').read_text(encoding='utf8').splitlines() if l.strip()]
 cfg=json.loads((src/'summary.json').read_text(encoding='utf8'))
 if not rows or cfg.get('selected_count')!=len(rows):die('Missing or inconsistent prepared cohort')
 if sha(src/'candidate_manifest.jsonl')!=cfg['candidate_manifest_sha256']:die('Candidate manifest changed')
 obj=json.loads(review.read_text(encoding='utf8'))
 if obj.get('schema')!='prd32_new_train_dev_blind_review_v1':die('Review JSON has unexpected schema')
 rm={r.get('case_id'):r for r in obj.get('rows',[])}
 if len(rm)!=len(obj.get('rows',[])) or set(rm)!={r['case_id'] for r in rows}:die('Review rows incomplete/duplicate or from different cohort')
 cats=set(CATS);human=[];plan=[];exclusions=[]
 for p in rows:
  r=rm[p['case_id']]
  if r.get('image_sha256')!=p['sha256']:die('Review image SHA identity mismatch: '+p['case_id'])
  cat=r.get('human_category');b=r.get('bbox_status');ident=r.get('visually_identifiable');over=r.get('overlap_other_garments');bd=r.get('boundary_case');lev=r.get('evidence_level')
  if cat not in cats|{'uncertain'} or b not in {'YES','NO','UNCERTAIN'} or ident not in {'YES','NO'} or over not in {'YES','NO','UNCERTAIN'} or bd not in {'CLEAR','TOP_OUTERWEAR','TOP_DRESS','SKIRT_PANTS','OTHER_BOUNDARY'} or lev not in {'HIGH','MEDIUM','LOW'}:die('Missing/invalid manual review fields for '+p['case_id'])
  scoreable=cat in cats and b=='YES' and ident=='YES' and lev!='LOW'
  reason='' if scoreable else 'HUMAN_CATEGORY_UNCERTAIN_OR_INPUT_QUALITY_NOT_CONFIRMED'
  if not scoreable:exclusions.append({'case_id':p['case_id'],'reason':reason})
  human.append({'case_id':p['case_id'],'image_sha256':p['sha256'],'category':cat,
    'bbox_status':b,'visually_identifiable':ident,'overlap_other_garments':over,
    'boundary_case':bd,'evidence_level':lev,'reviewer_note':r.get('reviewer_note',''),
    'scoreable':scoreable,'exclusion_reason':reason})
  # Label-free plan. The model must never receive the human reference file.
  plan.append({'case_id':p['case_id'],'image_path':p['source_path'],'image_sha256':p['sha256'],
   'bbox':p['bbox'],'target_instance_id':p['source_garment_id'] or p['source_record_id'] or p['case_id'],
   'task':'tags','bbox_origin':'historical_inventory_gt_pilot_NOT_V5_prediction',
   'scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY'})
 out.mkdir(parents=True)
 for name,vals in [('frozen_inference_plan.jsonl',plan),('frozen_human_reference.jsonl',human)]:
  with (out/name).open('w',encoding='utf8') as f:
   for v in vals:f.write(json.dumps(v,ensure_ascii=False)+'\n')
 write_json(out/'freeze.json',{'schema':'prd32_new_train_dev_frozen_cohort_v1','status':'FROZEN_DEV_NOT_INDEPENDENT',
  'total':len(plan),'scoreable':sum(r['scoreable'] for r in human),'excluded':exclusions,
  'plan_sha256':sha(out/'frozen_inference_plan.jsonl'),
  'human_reference_sha256':sha(out/'frozen_human_reference.jsonl'),
  'blind_review_sha256':sha(review), 'candidate_manifest_sha256':sha(src/'candidate_manifest.jsonl'),
  'human_labels_must_not_be_loaded_during_inference':True,'holdout_used':False,'official_accuracy':None})
 print('FREEZE_OK',len(plan),'scoreable',sum(r['scoreable'] for r in human),'out',out)

def main():
 ap=argparse.ArgumentParser(description=__doc__);subs=ap.add_subparsers(dest='cmd',required=True)
 p=subs.add_parser('prepare');p.add_argument('--project-root',default='/workspace/fashion_multimodal_analysis');p.add_argument('--data-root',default='/workspace/fashion_data');p.add_argument('--inventory');p.add_argument('--per-class',type=int,default=4);p.add_argument('--exclude-hashes');p.add_argument('--output',required=True)
 p=subs.add_parser('freeze');p.add_argument('--prepared',required=True);p.add_argument('--review',required=True);p.add_argument('--output',required=True)
 a=ap.parse_args();
 if a.cmd=='prepare':
  if not 1<=a.per_class<=15:die('--per-class must be 1..15')
  do_prepare(a)
 else:do_freeze(a)
if __name__=='__main__':main()
