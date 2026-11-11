#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD 3.2 existing dataset developer-only pool.

Conservatively selects records with an explicit TRAIN path in the actual original
source_image and physically present images. Does NOT turn training data into
independent validation data. Never opens holdout/test/fullval/validation images.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import csv
import hashlib
from html import escape
import io
import json
from pathlib import Path
import tempfile
from PIL import Image, ImageDraw, ImageOps

CLASSES = ('top','pants','skirt','outerwear','dress','shoe','bag','accessory')
PROTECTED = ('holdout', 'sealed', 'fullval', 'val', 'valid', 'validation', 'test', 'testing', 'testset')
GT_PATTERNS = ('','纯色','条纹','格纹','人字纹','波点','花卉印花','字母印花','文字印花','几何印花','动物纹','拼色','未知','其他')

def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False

def is_protected_segment(p: Path) -> bool:
    for part in p.parts:
        x=part.lower()
        # Strict component naming. e.g. 'train' is permitted; not arbitrary substring match.
        if x in PROTECTED or any(x.startswith(tag+'_') or x.endswith('_'+tag) for tag in PROTECTED):
            return True
    return False

def parse_bbox(row):
    try:
        bb=[float(row['bbox_'+v]) for v in ('x1','y1','x2','y2')]
        if not all(float('-inf') < z < float('inf') for z in bb): return None
        if bb[2]<=bb[0] or bb[3]<=bb[1] or bb[0]<0 or bb[1]<0: return None
        return bb
    except (ValueError,TypeError,KeyError):
        return None

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):h.update(chunk)
    return h.hexdigest()

def inspect_inventory(project:Path, allowed_data_root:Path, csv_path:Path):
    if not csv_path.is_file():raise FileNotFoundError(csv_path)
    with csv_path.open('r',encoding='utf-8-sig',newline='') as f:
        source_rows=list(csv.DictReader(f))
    records=[]
    for i,r in enumerate(source_rows,1):
        src=(r.get('source_image') or '').strip()
        cat=(r.get('garment_category') or '').strip().lower()
        dset=(r.get('source_dataset') or '').strip()
        path=(project / src).resolve(strict=False) if src else None
        reason=[]
        if cat not in CLASSES:reason.append('CLASS_NOT_8_ENUM')
        if not src or not path:reason.append('SOURCE_IMAGE_EMPTY')
        elif not path_is_relative_to(path,allowed_data_root):reason.append('OUTSIDE_ALLOWED_DATA_ROOT')
        elif is_protected_segment(path):reason.append('RESTRICTED_PATH_METADATA_ONLY')
        elif 'train' not in tuple(s.lower() for s in path.parts):reason.append('NOT_EXPLICIT_TRAIN_PATH_METADATA_ONLY')
        elif not path.is_file():reason.append('TRAIN_FILE_NOT_FOUND')
        bbox=parse_bbox(r)
        if bbox is None:reason.append('INVALID_BBOX_METADATA')
        row={'row_index':i,'record_id':r.get('record_id',''),
             'source_dataset':dset,'source_image':src,'resolved_image_path':str(path) if path else '',
             'garment_id':r.get('garment_id',''),'inventory_category_NOT_GT':cat,
             'bbox_x1':r.get('bbox_x1',''),'bbox_y1':r.get('bbox_y1',''),
             'bbox_x2':r.get('bbox_x2',''),'bbox_y2':r.get('bbox_y2',''),
             'source_manifest':r.get('source_manifest',''),
             'decision':'TRAIN_DEV_ONLY' if not reason else 'EXCLUDED',
             'reason_codes':';'.join(reason)}
        records.append(row)
    return records

def create_preview(path, bbox, max_wh=560):
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert('RGB')
        w,h=im.size
        # Return failure on bbox outside *actual* image; do not silently clamp it.
        if bbox[2]>w or bbox[3]>h:raise ValueError('BBOX_OUTSIDE_ACTUAL_IMAGE')
        im.thumbnail((max_wh,max_wh))
        dw,dh=im.size
        box=[round(bbox[0]*dw/w),round(bbox[1]*dh/h),round(bbox[2]*dw/w),round(bbox[3]*dh/h)]
        draw=ImageDraw.Draw(im)
        draw.rectangle(box,outline=(230,44,44),width=max(2,round(max(dw,dh)/150)))
        out=io.BytesIO();im.save(out,'JPEG',quality=75,optimize=True)
        return 'data:image/jpeg;base64,'+base64.b64encode(out.getvalue()).decode('ascii'),w,h

def select_review(records, max_per_class):
    selected=[]; counts=Counter(); seen_images=set()
    for r in sorted(records,key=lambda a:(CLASSES.index(a['inventory_category_NOT_GT']) if a['inventory_category_NOT_GT'] in CLASSES else 999,a['resolved_image_path'],a['record_id'])):
        c=r['inventory_category_NOT_GT']; p=r['resolved_image_path']
        if r['decision']!='TRAIN_DEV_ONLY' or counts[c]>=max_per_class or p in seen_images:continue
        selected.append(r); counts[c]+=1; seen_images.add(p)
    return selected

def html_review(rows,outpath):
    cards=[]; meta=[]
    for i,r in enumerate(rows,1):
        ident=f'train_dev_{i:03d}'
        label=escape(r['inventory_category_NOT_GT'])
        cid=escape(r['record_id'])
        cat_opts=''.join(f'<option value="{c}">{c}</option>' for c in CLASSES)
        pat_opts=''.join(f'<option value="{escape(p)}">{escape(p) or "请选择"}</option>' for p in GT_PATTERNS)
        cards.append(f'''<article class="card" data-case="{ident}"><div><p><b>{ident}</b> · 提议类别（不是人工 GT）：{label}</p><img src="{r['preview']}" alt="目标框预览"><div class="hint">原图 {r['width']}×{r['height']} · {cid}</div></div><div><p><b>限定用途：历史训练图开发诊断，不能参与独立验收。</b> 红框来自清单记录，需人工确认目标及 bbox。</p>
<label>目标与框核验 <select class="bbox_status"><option value="PENDING">待检查</option><option value="VALID">框与目标对应</option><option value="WRONG_TARGET">目标错误/框不准</option></select></label>
<label>人工类别 <select class="category"><option value="">请选择</option>{cat_opts}</select></label>
<label>人工主色 <input class="color" placeholder="例如 黑色；无法判断填 未知"></label>
<label>人工图案 <select class="pattern">{pat_opts}</select></label>
<label>审核状态 <select class="review_status"><option value="PENDING">待审核</option><option value="REVIEWED">已审核</option><option value="EXCLUDE">排除</option></select></label>
<label>备注 <textarea class="notes" placeholder="记录视觉混淆、分类或框质量问题"></textarea></label></div></article>''')
        meta.append({'sample_id':ident,'source_image':r['source_image'],'record_id':r['record_id'],'garment_id':r['garment_id'], 'image_sha256':r['sha256'],'candidate_category_NOT_GT':r['inventory_category_NOT_GT'],'bbox_xyxy':[float(r['bbox_x1']),float(r['bbox_y1']),float(r['bbox_x2']),float(r['bbox_y2'])],'source_split':'train','evaluation_scope':'TRAIN_DEV_DIAGNOSTIC_ONLY'})
    j=json.dumps(meta,ensure_ascii=False).replace('</','<\\/')
    html='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>PRD3.2 现有训练数据开发审核</title><style>body{background:#f2f5fa;color:#182339;font:15px system-ui,'Microsoft YaHei',sans-serif;margin:0}main{max-width:1180px;margin:auto;padding:20px}.card{background:white;display:grid;grid-template-columns:450px 1fr;gap:24px;padding:20px;border-radius:10px;margin:20px 0;border:1px solid #d7dfeb}img{width:100%;max-width:440px;border-radius:8px}label{display:block;margin:12px 0}select,input,textarea{padding:7px;min-width:200px;max-width:100%;border:1px solid #a9b9c9;border-radius:6px}textarea{min-width:80%;min-height:60px}button{background:#2855b7;color:white;border:0;padding:12px 17px;border-radius:8px;cursor:pointer}.hint{color:#69788b}header{background:#1a263e;color:white;padding:22px;border-radius:12px}@media(max-width:760px){.card{grid-template-columns:1fr}}</style></head><body><main><header><h1>PRD 3.2 · 八类现有 TRAIN 数据开发审核</h1><p>这些图片明确来自旧训练路径，只能用于开发调试/诊断，不能当作独立评估或声称 PRD 达标。</p></header><button onclick="exportGt()">导出开发审核 JSON</button><span id="status"></span>'''+''.join(cards)+'''</main><script>const metadata=__META__;
function exportGt(){let x=[];for(const card of document.querySelectorAll('.card')){const obj=metadata.find(z=>z.sample_id===card.dataset.case);const val=k=>card.querySelector('.'+k).value.trim();x.push({...obj,bbox_status:val('bbox_status'),human_category:val('category'),human_color:val('color'),human_pattern:val('pattern'),review_status:val('review_status'),notes:val('notes')})}const blob=new Blob([JSON.stringify({schema:'prd32_existing_train_dev_gt_v1',scope:'train_dev_diagnostic_only_NOT_INDEPENDENT',rows:x},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='PRD32_TRAIN_DEV_HUMAN_REVIEW.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),2000)}
</script></body></html>'''
    outpath.write_text(html.replace('__META__',j),encoding='utf-8')

def write_csv(p,rows):
    if not rows:return
    fields=list(rows[0].keys())
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description='Existing dataset TRAIN developer-only audit/preview. No protected images opened.')
    ap.add_argument('--project-root',default='/workspace/fashion_multimodal_analysis')
    ap.add_argument('--data-root',default='/workspace/fashion_data')
    ap.add_argument('--inventory',default=None)
    ap.add_argument('--output',required=True)
    ap.add_argument('--per-class',type=int,default=8)
    ap.add_argument('--metadata-only',action='store_true',help='No original images opened, just existence and metadata')
    a=ap.parse_args()
    project=Path(a.project_root).resolve(); data=Path(a.data_root).resolve()
    inventory=Path(a.inventory).resolve() if a.inventory else project/'configs/prd_8class_unified_manifest_v2.csv'
    out=Path(a.output).resolve()
    if out.exists():raise FileExistsError(f'Output exists, use a new run directory: {out}')
    if a.per_class<1 or a.per_class>30:raise ValueError('--per-class must be 1..30')
    records=inspect_inventory(project,data,inventory)
    chosen=select_review(records,a.per_class)
    usable=[]; failures=[]
    if not a.metadata_only:
        for r in chosen:
            bb=[float(r['bbox_'+v]) for v in ('x1','y1','x2','y2')]
            p=Path(r['resolved_image_path'])
            try:
                # Only selected, source-split explicitly TRAIN images are opened here.
                preview,w,h=create_preview(p,bb)
                r=dict(r,preview=preview,width=w,height=h,sha256=sha256(p))
                usable.append(r)
            except Exception as e:
                failures.append({'row_index':r['row_index'],'reason':type(e).__name__+': '+str(e)[:120]})
    out.mkdir(parents=True)
    write_csv(out/'inventory_decisions.csv',records)
    write_csv(out/'train_dev_selected.csv',[{k:v for k,v in r.items() if k not in ('preview',)} for r in usable])
    write_csv(out/'preview_failures.csv',failures)
    if usable:html_review(usable,out/'TRAIN_DEV_GT_REVIEW_STATIC.html')
    summary={'schema':'prd32_existing_dataset_dev_pool_v1','total_inventory_records':len(records),
             'train_dev_paths_present':sum(r['decision']=='TRAIN_DEV_ONLY' for r in records),
             'quarantined_records':sum(r['decision']!='TRAIN_DEV_ONLY' for r in records),
             'training_source_class_counts':dict(Counter(r['inventory_category_NOT_GT'] for r in records if r['decision']=='TRAIN_DEV_ONLY')),
             'selected_for_preview':len(chosen),'preview_created':len(usable),'preview_failures':failures,
             'metadata_only':bool(a.metadata_only), 'source_project':str(project),'data_root':str(data),
             'training_source_images_opened':0 if a.metadata_only else len(usable)+len(failures),
             'protected_images_opened':0, 'holdout_used':False, 'model_inference_calls':0,
             'approved_for_independent_accuracy':0,'scope':'TRAIN_DEV_DIAGNOSTIC_ONLY_NOT_OFFICIAL',
             'note':'Garment category in manifest is NOT assumed to be human-verified GT. Training images must not be reported as independent PRD acceptance.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('EXISTING_DATA_SCAN_DONE')
    print('total:',len(records),'train_dev_paths_present:',summary['train_dev_paths_present'],'quarantined:',summary['quarantined_records'])
    print('class_counts:',summary['training_source_class_counts'])
    print('review_previews:',len(usable),'out:',out)
    return 0

if __name__=='__main__':raise SystemExit(main())
