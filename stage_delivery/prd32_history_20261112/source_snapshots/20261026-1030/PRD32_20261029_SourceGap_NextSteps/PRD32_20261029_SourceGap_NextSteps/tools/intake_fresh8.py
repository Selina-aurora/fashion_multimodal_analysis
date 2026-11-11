#!/usr/bin/env python3
# coding: utf-8
"""PRD3.2 eight-class new-image intake: metadata, no model, no legacy images accessed.
Usage: python intake_fresh8.py --images /workspace/PRD32_NEW_DEV_IMAGES --output /workspace/PRD32_FRESH_DEV_INTAKE_run01 [--reference-csv old.csv]
"""
import argparse, csv, json, hashlib, base64, io, re
from collections import Counter, defaultdict
from pathlib import Path
from html import escape
from datetime import datetime, timezone
from PIL import Image, ImageOps

CATS = ['top','pants','skirt','outerwear','dress','shoe','bag','accessory']
PATS = ['','纯色','条纹','格纹','人字纹','波点','花卉印花','文字印花','字母印花','几何印花','动物纹','拼色','其他','未知']
EXTS={'.jpg','.jpeg','.png','.webp'}
RESTRICTED = re.compile(r'(?:holdout|sealed|testset|fullval|training|trainset|\\btrain\\b|\\bvalidation\\b)', re.I)

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def read_legacy_basenames(path):
 """Read CSV metadata only. Never open referenced pictures or follow source_manifest paths."""
 if not path:return set()
 p=Path(path)
 if not p.is_file():raise FileNotFoundError(f'reference CSV missing: {p}')
 names=set()
 with p.open('r',encoding='utf-8-sig',newline='') as f:
  reader=csv.DictReader(f)
  if 'source_image' not in reader.fieldnames:raise ValueError('reference CSV must have source_image')
  for row in reader:
   s=(row.get('source_image') or '').strip()
   if s:names.add(s.replace('\\','/').rsplit('/',1)[-1].casefold())
 return names

def make_thumb(p, max_width=400, max_height=520):
 with Image.open(p) as im:
  im=ImageOps.exif_transpose(im)
  im.thumbnail((max_width,max_height))
  out=io.BytesIO()
  if im.mode not in ('RGB','L'):im=im.convert('RGB')
  im.save(out,'JPEG',quality=76,optimize=True)
  return 'data:image/jpeg;base64,'+base64.b64encode(out.getvalue()).decode('ascii')

def scan(root, legacy_names, max_images=240, prior_hashes=None):
 prior_hashes=prior_hashes or set()
 if not root.is_dir():raise FileNotFoundError(f'Fresh images directory does not exist: {root}')
 if RESTRICTED.search(str(root)) or 'fashion_multimodal_analysis' in str(root):
  raise ValueError('Refusing legacy project/training/test/holdout directory; use dedicated NEW image folder')
 images=[]; paths=[]
 for cat in CATS:
  d=root/cat
  if not d.exists():continue
  for p in sorted(d.iterdir()):
   if p.is_symlink():continue
   if p.is_file() and p.suffix.lower() in EXTS:paths.append((cat,p))
 if len(paths)>max_images:raise ValueError(f'{len(paths)} images > max {max_images}; separate into batches')
 seen_sha={}; seen_names=Counter(p.name.casefold() for _,p in paths)
 for cat,p in paths:
  flags=[]
  if RESTRICTED.search(str(p)):flags.append('RESTRICTED_OR_TRAIN_PATH_NAME')
  if p.name.casefold() in legacy_names:flags.append('BASENAME_MATCHES_OLD_DATA')
  if seen_names[p.name.casefold()]>1:flags.append('DUPLICATE_BASENAME_IN_NEW_POOL')
  try:
   with Image.open(p) as im:
    w,h=ImageOps.exif_transpose(im).size
    im.verify if False else None
   if w<128 or h<128:flags.append('LOW_RESOLUTION')
   sha=digest(p)
   if sha in prior_hashes:flags.append('SHA_MATCH_PRIOR52_DO_NOT_USE')
   if sha in seen_sha:flags.append('DUPLICATE_IMAGE_CONTENT')
   else:seen_sha[sha]=p
   thumb=make_thumb(p)
  except Exception as e:
   w=h=0;sha='';thumb='';flags.append('IMAGE_UNREADABLE')
  rel=f'{cat}/{p.name}'
  images.append({'sample_id':f'newdev_{len(images)+1:04d}', 'relative_path':rel, 'declared_folder_category':cat,
                 'image_sha256':sha,'image_width':w,'image_height':h,'flag_codes':flags,
                 'provenance_status':'NOT_CLEARED','gt_status':'NOT_REVIEWED','thumbnail':thumb})
 return images

def build_html(records, output):
 # Static cards mean pictures visible even if JS fails. Filename and browser form contain no source-image path leakage.
 cards=[]
 for r in records:
  k=r['sample_id'];w=r['image_width'];h=r['image_height'];thumb=r['thumbnail']
  img=f'<img src="{thumb}" alt="{escape(k)}" width="320">' if thumb else '<span>图片损坏</span>'
  opts=''.join('<option value="'+s+'">'+(s or '请选择')+'</option>' for s in CATS)
  pats=''.join('<option value="'+s+'">'+(s or '请选择')+'</option>' for s in PATS)
  flags=escape(', '.join(r['flag_codes']) or '无自动风险标记（不代表来源已独立确认）')
  cards.append(f'''<article class="card" data-case="{k}"><div><h3>{k}</h3><div>{img}</div><p class="hint">{escape(r['relative_path'])} · {w}×{h}</p><p class="warning">{flags}</p></div>
<section><p><b>图片来源与数据划分：</b>必须人工核实；未确认不得进入独立准确率评估。</p>
<label>来源证明 <select class="provenance"><option value="UNVERIFIED">尚未核实</option><option value="VERIFIED_NEW_DEV">已核对：新开发图且未用于既有训练/测试</option><option value="RESTRICTED_OR_OVERLAP">受限/重复，隔离</option></select></label>
<label>人工目标类别 <select class="category"><option value="">请选择</option>{opts}</select></label>
<label>人工颜色 <input class="color" placeholder="黑色 / 白色 / 未知"></label>
<label>人工图案 <select class="pattern">{pats}</select></label>
<p>目标 bbox（原图像素坐标，x1/y1/x2/y2）。本页可直接在图片上辅助观察，但请以原图尺寸填写坐标。</p>
<div class="coords"><label>x1 <input class="x1" type="number" min="0" step="1"></label><label>y1 <input class="y1" type="number" min="0" step="1"></label><label>x2 <input class="x2" type="number" min="0" step="1"></label><label>y2 <input class="y2" type="number" min="0" step="1"></label></div>
<label>是否完成复核 <select class="review"><option value="PENDING">待审核</option><option value="VERIFIED">已核验</option><option value="REJECTED">隔离/不适用</option></select></label>
<label>备注 <textarea class="notes" placeholder="核实来源依据、目标位置、异常情况"></textarea></label></section></article>''')
 js=r'''const meta=__METADATA__;
function collect(){let result=[]; for(const card of document.querySelectorAll('.card')){
 const id=card.dataset.case;let m=meta.find(x=>x.sample_id===id);
 function val(cl){return card.querySelector('.'+cl).value.trim()}
 let obj={sample_id:id,relative_path:m.relative_path,image_sha256:m.image_sha256,image_width:m.image_width,image_height:m.image_height,
 declared_folder_category:m.declared_folder_category,source_verified:val('provenance'),human_category:val('category'),human_color:val('color'),human_pattern:val('pattern'),
 x1:val('x1'),y1:val('y1'),x2:val('x2'),y2:val('y2'),review_status:val('review'),notes:val('notes')};
 let nums=[obj.x1,obj.y1,obj.x2,obj.y2].map(x=>x===''?NaN:Number(x));
 obj.bbox_valid=nums.every(Number.isFinite)&&nums[0]>=0&&nums[1]>=0&&nums[2]>nums[0]&&nums[3]>nums[1]&&nums[2]<=m.image_width&&nums[3]<=m.image_height;
 obj.eligible_for_dev_only=(obj.review_status==='VERIFIED'&&obj.source_verified==='VERIFIED_NEW_DEV'&&obj.human_category!==''&&obj.bbox_valid&&m.flag_codes.length===0);
 result.push(obj); }return result;}
function download(name,content,mime){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([content],{type:mime}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),3000)}
function exportJSON(){download('PRD32_FRESH8_HUMAN_GT.json',JSON.stringify({schema:'prd32_newdev_gt_v1',scope:'development_not_independent_test',records:collect()},null,2),'application/json')}
function exportCSV(){let d=collect();let fields=Object.keys(d[0]||{});function cell(s){let t=String(s===null?'':s);return '"'+t.replace(/"/g,'""')+'"'}
 download('PRD32_FRESH8_HUMAN_GT.csv','\uFEFF'+[fields.map(cell).join(','),...d.map(x=>fields.map(f=>cell(x[f])).join(','))].join('\r\n'),'text/csv;charset=utf-8')}
function saveDraft(){try{localStorage.setItem('PRD32_FRESH8_'+location.pathname,JSON.stringify(collect()));document.getElementById('status').textContent='已保存到当前浏览器（仍应导出 JSON 备份）'}catch(e){document.getElementById('status').textContent='浏览器禁止暂存，请导出 JSON'}}
function restore(){let old;try{old=JSON.parse(localStorage.getItem('PRD32_FRESH8_'+location.pathname)||'[]')}catch(e){return}
 for(const row of old){const c=document.querySelector('[data-case="'+row.sample_id+'"]');if(!c)continue;
 for(const [name,key] of [['provenance','source_verified'],['category','human_category'],['color','human_color'],['pattern','human_pattern'],['x1','x1'],['y1','y1'],['x2','x2'],['y2','y2'],['review','review_status'],['notes','notes']]){const field=c.querySelector('.'+name); if(row[key]!==undefined)field.value=row[key]}}
 }
restore();
'''
 safe_meta=[{k:v for k,v in r.items() if k!='thumbnail'} for r in records]
 js=js.replace('__METADATA__',json.dumps(safe_meta,ensure_ascii=False).replace('</','<\\/'))
 html=f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PRD3.2 新开发八类样本人工审核</title>
<style>body{{font:15px system-ui,'Microsoft Yahei',sans-serif;background:#f5f7fb;color:#172337;margin:0}}main{{max-width:1250px;margin:auto;padding:20px}}header{{background:#17233c;color:white;padding:23px;border-radius:15px}}.card{{display:grid;grid-template-columns:340px 1fr;gap:22px;background:white;border:1px solid #dce4ee;border-radius:12px;padding:18px;margin:18px 0}}.card img{{max-width:100%;height:auto;border-radius:6px}}h3{{margin-top:0}}label{{display:block;margin:10px 0}}input,select,textarea{{padding:7px 9px;border:1px solid #cbd5e1;border-radius:7px;max-width:100%}}textarea{{min-width:70%;min-height:45px}}button{{padding:10px 15px;border:0;background:#2458ad;color:white;border-radius:8px;cursor:pointer;margin:10px 5px 10px 0}}.hint{{color:#718096;font-size:12px}}.warning{{color:#895c15;font-size:12px}}.coords{{display:flex;flex-wrap:wrap;gap:8px}}.coords input{{width:68px}}@media(max-width:800px){{.card{{grid-template-columns:1fr}}}}</style></head><body><main><header><h1>PRD 3.2 · 新八类开发样本人工审核</h1><p>新增样本登记，不是独立准确率或正式验收。若图片显示异常、来源不清或目标框不可信，请勿将样本评为可用。</p></header>
<button onclick="exportJSON()">导出 JSON 备份</button><button onclick="exportCSV()">导出 GT CSV</button><button onclick="saveDraft()">暂存当前审核</button><span id="status"></span>
{''.join(cards)}<p>所有图片均直接写入本地 HTML，关闭 JS 也可查看。填写坐标请参考原图真实像素大小。旧训练/holdout 数据不能用作新开发集。</p></main><script>{js}</script></body></html>'''
 output.write_text(html,encoding='utf-8')

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--images',required=True,help='Fresh dedicated folder with 8 class subdirectories')
 ap.add_argument('--output',required=True)
 ap.add_argument('--reference-csv',default=None,help='Optional legacy metadata-only CSV, do not open legacy images')
 ap.add_argument('--no-prior-hash-check',action='store_true',help='NOT RECOMMENDED: only for deliberately isolated synthetic tests')
 args=ap.parse_args()
 root=Path(args.images).resolve();out=Path(args.output).resolve()
 if out.exists():raise FileExistsError(f'Output directory exists: {out}. Change run suffix, do not overwrite')
 legacy=read_legacy_basenames(args.reference_csv)
 prior_hashes=set()
 if not args.no_prior_hash_check:
  from source_gap_audit import old_hashes,new_hashes,ROOT
  prior_hashes=old_hashes(ROOT/'evidence/HumanEval40_Comparison.csv')|new_hashes(ROOT/'evidence/new12_blind_review.json')
 rows=scan(root,legacy,prior_hashes=prior_hashes)
 if not rows:raise ValueError('No JPG/PNG/WEBP found in eight category folders under '+str(root))
 out.mkdir(parents=True)
 headers=['sample_id','relative_path','declared_folder_category','image_sha256','image_width','image_height','flag_codes','provenance_status','gt_status']
 with (out/'intake_candidates.csv').open('w',newline='',encoding='utf-8-sig') as f:
  wr=csv.DictWriter(f,fieldnames=headers);wr.writeheader();wr.writerows({k:json.dumps(r[k],ensure_ascii=False) if k=='flag_codes' else r[k] for k in headers} for r in rows)
 report={'schema':'prd32_fresh8_dev_intake_v1','status':'CANDIDATES_REQUIRE_MANUAL_GT_AND_PROVENANCE',
         'images_count':len(rows),'category_counts':dict(Counter(r['declared_folder_category'] for r in rows)),
         'flag_counts':dict(Counter(x for r in rows for x in r['flag_codes'])),
         'source_folder':str(root),'reference_csv':args.reference_csv,
         'approved_for_independent_accuracy':0,'model_inference_calls':0,'legacy_images_opened':0,'sealed_holdout_used':False,'prior52_image_sha_registry_checked':not args.no_prior_hash_check}
 (out/'intake_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 build_html(rows,out/'GT_review_STATIC.html')
 (out/'README.txt').write_text('本包仅是新图片收集与人工GT入口：请打开GT_review_STATIC.html，完成来源核验、类别颜色图案和bbox标注，并导出JSON与CSV；来源和训练重叠未独立证实前，不得作为正式独立评估。\n',encoding='utf-8')
 print('FRESH8_INTAKE_READY',out,'images=',len(rows),'category_counts=',report['category_counts']);return 0
if __name__=='__main__':raise SystemExit(main())
