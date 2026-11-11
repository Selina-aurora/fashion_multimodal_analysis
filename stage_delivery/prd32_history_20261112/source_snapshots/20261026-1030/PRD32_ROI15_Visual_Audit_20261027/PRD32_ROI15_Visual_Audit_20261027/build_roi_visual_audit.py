#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD3.2: read-only, training-dev ROI visual audit for error-attribution.

Takes an already executed run_roi_paired.py output; never re-runs inference.
The generated HTML statically embeds full-image and actual ROI input images.
"""
import argparse, base64, csv, hashlib, html, io, json, math, statistics, sys
from collections import Counter
from pathlib import Path

ALLOWED_CATEGORIES={'top','pants','skirt','outerwear','dress','shoe','bag','accessory'}
ALLOWED_CAUSES={'MODEL_TARGET_BINDING','MODEL_CLASSIFICATION'}
FIELDS = ['target_visible', 'roi_single_item', 'target_type_clear', 'cause_revision', 'notes']

def read_jsonl(p):
    results=[]
    for n,line in enumerate(Path(p).open(encoding='utf-8-sig'),1):
        if line.strip():
            try: results.append(json.loads(line))
            except json.JSONDecodeError as e: raise ValueError(f'{p}, line {n}: {e}') from e
    return results

def file_sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def short_data_uri(img, max_side=760):
    from PIL import Image,ImageOps
    with Image.open(img) as im:
        im=ImageOps.exif_transpose(im).convert('RGB')
        im.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
        output=io.BytesIO();im.save(output,format='JPEG',quality=82,optimize=True)
    return 'data:image/jpeg;base64,'+base64.b64encode(output.getvalue()).decode('ascii')

def h(x): return html.escape(str(x) if x is not None else '',quote=True)

def build(run_dir, out, allowed_root, expected=15):
    from PIL import Image,ImageDraw
    run_dir=Path(run_dir).resolve();allowed_root=Path(allowed_root).resolve();out=Path(out).resolve()
    if not run_dir.is_dir(): raise FileNotFoundError(run_dir)
    if out.exists(): raise FileExistsError(f'Output exists, choose a new directory: {out}')
    plan=read_jsonl(run_dir/'case_plan.jsonl');pred=read_jsonl(run_dir/'roi_predictions.jsonl')
    if len(plan)!=expected or len(pred)!=expected: raise ValueError(f'Expected {expected} cases; got plan={len(plan)}, predictions={len(pred)}')
    by_id={r['case_id']:r for r in pred}
    if len(by_id)!=len(pred): raise ValueError('Duplicate prediction case_id')
    if len({p['case_id'] for p in plan})!=len(plan):raise ValueError('Duplicate plan case_id')
    if {p['case_id'] for p in plan}!=set(by_id):raise ValueError('case ID mismatch')
    summary=json.loads((run_dir/'summary.json').read_text(encoding='utf-8'))
    if summary.get('historical_GT_pilot_bbox_not_frozen_V5') is not True or summary.get('holdout_used') is not False:
        raise ValueError('Unexpected source or holdout status')
    pairs=[];excerpts=[]
    for i,p in enumerate(plan,1):
        r=by_id[p['case_id']]
        raw=Path(p['image_path']).resolve()
        if not raw.is_relative_to(allowed_root):raise ValueError(f'{p["case_id"]}: outside allowed training root')
        parts={x.lower() for x in raw.relative_to(allowed_root).parts}
        if 'train' not in parts or parts.intersection({'holdout','sealed','test','val','valid','validation','fullval'}):
            raise ValueError(f'{p["case_id"]}: not an unambiguously train-only path')
        if not raw.is_file() or file_sha256(raw)!=p['image_sha256']:
            raise ValueError(f'{p["case_id"]}: original image missing or SHA256 mismatch')
        roi=Path(r.get('roi_input','')).resolve()
        if not roi.is_relative_to(run_dir/'roi_marked_inputs') or not roi.is_file():
            raise ValueError(f'{p["case_id"]}: actual ROI file missing / path outside run directory')
        if r['cause'] not in ALLOWED_CAUSES or p['cause']!=r['cause']:raise ValueError('Cause mismatch')
        bbox=p['bbox'];x1,y1,x2,y2=map(float,bbox)
        with Image.open(raw) as im:
            im=im.convert('RGB');w,hh=im.size
            if not all(math.isfinite(x) for x in (x1,y1,x2,y2)) or not (0<=x1<x2<=w and 0<=y1<y2<=hh):
                raise ValueError(f'{p["case_id"]}: invalid box')
            marked=im.copy()
            ImageDraw.Draw(marked).rectangle((round(x1),round(y1),round(x2),round(y2)),outline='#ff3030',width=max(3,min(w,hh)//150))
            with io.BytesIO() as bf:
                marked.thumbnail((760,760),Image.Resampling.LANCZOS)
                marked.save(bf,'JPEG',quality=82,optimize=True)
                full_uri='data:image/jpeg;base64,'+base64.b64encode(bf.getvalue()).decode('ascii')
        roi_uri=short_data_uri(roi)
        with Image.open(roi) as im2:rw,rh=im2.size
        label=p.get('human_category_FOR_EVAL_ONLY')
        if label not in ALLOWED_CATEGORIES:raise ValueError('Invalid category in reviewed plan')
        actual=r.get('roi_category');orig=r.get('baseline_category')
        corrected=(actual==label and orig!=label)
        area_ratio=(x2-x1)*(y2-y1)/(w*hh)
        edge_ratio=min(x1,y1,w-x2,hh-y2)/max(w,hh)
        elapsed=(r.get('timing') or {}).get('end_to_end_ms')
        obj={'case_id':r['case_id'],'cause':r['cause'],'human_category_reviewed':label,'baseline_category':orig,'roi_category':actual,'corrected':corrected,
             'bbox_area_fraction':round(area_ratio,4),'bbox_nearest_image_edge_fraction':round(edge_ratio,4),'full_width':w,'full_height':hh,'roi_width':rw,'roi_height':rh,
             'roi_latency_ms':elapsed,'schema_valid':r.get('schema_valid'),'native_json':r.get('native_json'),'issues':' | '.join(r.get('issues') or []),
             'image_sha256':p['image_sha256'],'roi_sha256':file_sha256(roi)}
        excerpts.append(obj)
        status='已纠正' if corrected else '仍未纠正'
        klass='ok' if corrected else 'bad'
        pairs.append(f'''<section class="card" data-id="{h(r['case_id'])}">
           <div class="head"><h2>{i}. {h(r['case_id'])}</h2><span class="pill {klass}">{status}</span></div>
           <p class="meta">来源归因：{h(r['cause'])} ｜ 人工复核类别：<strong>{h(label)}</strong> ｜ 旧整图：{h(orig)} → ROI：<strong>{h(actual)}</strong> ｜ bbox占原图：{area_ratio:.1%}</p>
           <div class="images"><figure><img src="{full_uri}" alt="原图红框 {h(r['case_id'])}"/><figcaption>原图及原目标框（由原始坐标重绘）</figcaption></figure><figure><img src="{roi_uri}" alt="实际 ROI 输入 {h(r['case_id'])}"/><figcaption>实际送入 Qwen 的 ROI（不是重新生成的裁剪）</figcaption></figure></div>
           <p class="meta">bbox: {h([round(x,1) for x in bbox])} ｜ 原图 {w}×{hh} ｜ ROI {rw}×{rh} ｜ 模型原生 JSON: {h(r.get('native_json'))} ｜ 异常：{h(', '.join(r.get('issues') or []) or '无')}</p>
           <div class="fields">
           <label>红框是否准确覆盖目标？<select data-key="target_visible"><option value="">未审核</option><option value="YES">是</option><option value="PARTIAL">部分</option><option value="NO">否</option><option value="UNCERTAIN">不确定</option></select></label>
           <label>局部图是否仅包含目标服饰？<select data-key="roi_single_item"><option value="">未审核</option><option value="YES">是</option><option value="NO">否（有其他服饰/内搭）</option><option value="UNCERTAIN">不确定</option></select></label>
           <label>仅凭 ROI 能否可靠分辨目标类别？<select data-key="target_type_clear"><option value="">未审核</option><option value="YES">可以</option><option value="NO">不能</option><option value="UNCERTAIN">不确定</option></select></label>
           <label>复核后主因（不覆盖原始审核）<select data-key="cause_revision"><option value="">未审核</option><option value="CROP_MIXED_ITEMS">裁剪内混入其他服饰</option><option value="CROP_TRUNCATED">目标被截断/框不完整</option><option value="MODEL_CLASS_BIAS">目标清楚，模型仍错分</option><option value="VISUALLY_AMBIGUOUS">视觉无法可靠判断</option><option value="OTHER">其他/待查</option></select></label>
           </div><label class="notes">人工复核备注<textarea data-key="notes" placeholder="例如：目标是裤子，但ROI大部分可见内容仍是上衣"></textarea></label>
         </section>''')
    out.mkdir(parents=True)
    with (out/'roi15_measurements.csv').open('w',encoding='utf-8-sig',newline='') as f:
        dw=csv.DictWriter(f,fieldnames=list(excerpts[0]));dw.writeheader();dw.writerows(excerpts)
    stats={'scope':'DEVELOPMENT_ERROR_SELECTED_NOT_INDEPENDENT','source_run':str(run_dir),'source_summary_sha256':file_sha256(run_dir/'summary.json'),
           'n':len(excerpts),'corrected':sum(x['corrected'] for x in excerpts),'still_wrong':sum(not x['corrected'] for x in excerpts),
           'roi_predicted_top':sum(x['roi_category']=='top' for x in excerpts),'native_json_count':sum(x['native_json'] is True for x in excerpts),
           'schema_valid_count':sum(x['schema_valid'] is True for x in excerpts),
           'by_review_cause':{cause:{'n':sum(x['cause']==cause for x in excerpts),'corrected':sum(x['cause']==cause and x['corrected'] for x in excerpts)} for cause in sorted(ALLOWED_CAUSES)},
           'no_gpu_used':True,'holdout_used':False,'images_from_training_development_only':True}
    (out/'audit_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    js='''
const KEYS=['target_visible','roi_single_item','target_type_clear','cause_revision','notes'];
const STORE='prd32_roi15_visual_audit_20261027_v1';
function data(){const r=[];document.querySelectorAll('.card').forEach(c=>{const x={case_id:c.dataset.id};KEYS.forEach(k=>x[k]=c.querySelector(`[data-key="${k}"]`).value);r.push(x)});return {schema:'prd32_roi15_visual_review_v1',scope:'train_dev_not_independent',rows:r}}
function save(){try{localStorage.setItem(STORE,JSON.stringify(data()))}catch(e){} document.querySelector('#status').textContent='本地已暂存；请导出 JSON 备份。'}
function restore(){try{let p=JSON.parse(localStorage.getItem(STORE)||'null');if(!p||!Array.isArray(p.rows))return;for(let r of p.rows){let c=[...document.querySelectorAll('.card')].find(x=>x.dataset.id===r.case_id);if(!c)continue;for(let k of KEYS){const el=c.querySelector(`[data-key="${k}"]`);if(el && typeof r[k]==='string')el.value=r[k]}}}catch(e){} }
function dl(name,content,type){let a=document.createElement('a'),blob=new Blob([content],{type});a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
function exportJSON(){dl('PRD32_ROI15_visual_review.json',JSON.stringify(data(),null,2),'application/json')}
function esc(v){return '"'+String(v??'').replaceAll('"','""')+'"'}
function exportCSV(){const rows=[['case_id',...KEYS],...data().rows.map(r=>[r.case_id,...KEYS.map(k=>r[k])])];dl('PRD32_ROI15_visual_review.csv','\ufeff'+rows.map(r=>r.map(esc).join(',')).join('\\r\\n'),'text/csv;charset=utf-8')}
document.querySelectorAll('select,textarea').forEach(el=>el.addEventListener('change',save));restore();
'''
    doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"/><title>PRD 3.2 · ROI 15条视觉归因复核</title><style>
*{{box-sizing:border-box}}body{{background:#f3f6fb;color:#1b2736;margin:0;font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1280px;margin:auto;padding:24px}}header{{background:#1d2a45;color:white;padding:25px;border-radius:14px}}header h1{{margin:0;font-size:25px}}.meta{{color:#617188;font-size:13px}}header .meta{{color:#d7e2f6}}.toolbar{{display:flex;gap:10px;margin:22px 0;align-items:center;flex-wrap:wrap}}button{{font:inherit;background:#345fc2;color:white;border:0;border-radius:7px;padding:10px 17px;cursor:pointer}}button:hover{{filter:brightness(1.1)}}.card{{background:white;border:1px solid #dce3ee;border-radius:12px;padding:20px;margin:16px 0}}.head{{display:flex;align-items:center;justify-content:space-between;gap:12px}}h2{{font-size:18px;margin:0}}.pill{{padding:4px 10px;border-radius:12px;font-size:13px}}.ok{{background:#e9f5ee;color:#1b7c44}}.bad{{background:#f9efed;color:#9b3a2d}}.images{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:16px}}figure{{margin:0;background:#eef2f7;border-radius:10px;text-align:center;padding:12px}}figure img{{max-width:100%;height:390px;object-fit:contain}}figcaption{{font-size:12px;color:#586d87}}.fields{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}label{{font-size:13px;font-weight:600}}label select,label textarea{{display:block;width:100%;font:inherit;border-radius:7px;border:1px solid #bccae0;background:white;padding:8px;margin-top:5px}}label textarea{{min-height:64px}}.notes{{display:block;margin-top:12px}}@media(max-width:800px){{.images,.fields{{grid-template-columns:1fr}}figure img{{height:auto;max-height:380px}}}}
</style></head><body><main><header><h1>PRD 3.2 · ROI 15条实际图像复核</h1><p class="meta">开发集错误选样｜整图红框 vs 实际 Qwen ROI 输入｜只做错误归因，不是独立准确率</p><p>共有 {len(excerpts)} 条；ROI纠正 {stats['corrected']} 条、仍错误 {stats['still_wrong']} 条。本页所有卡片和图片直接内嵌，不依赖 JS 初始化。</p></header><div class="toolbar"><button onclick="exportJSON()">导出人工审核 JSON</button><button onclick="exportCSV()">导出 CSV</button><span id="status" class="meta">页面离线使用；选择后本地暂存，但请主动导出备份。</span></div><p class="meta">判断 ROI 是否仍混入其他衣物、目标是否被裁断以及类别是否有足够视觉证据；不要直接依据人工 GT 修改模型预测。原始 3.1 V5 冻结、sealed holdout 均未使用。</p>{''.join(pairs)}<script>{js}</script></main></body></html>'''
    (out/'ROI15_VISUAL_REVIEW_STATIC.html').write_text(doc,encoding='utf-8')
    print('ROI15_STATIC_REPORT_READY',json.dumps({'cards':len(excerpts),'corrected':stats['corrected'],'still_wrong':stats['still_wrong'],'output':str(out),'html_kb':round((out/'ROI15_VISUAL_REVIEW_STATIC.html').stat().st_size/1024,2)},ensure_ascii=False))

def selftest():
    from tempfile import TemporaryDirectory
    from PIL import Image
    with TemporaryDirectory() as td:
        td=Path(td);src=td/'fashion_data/raw/train/train/image';src.mkdir(parents=True)
        modelrun=td/'model_run';(modelrun/'roi_marked_inputs').mkdir(parents=True)
        img=src/'001.jpg';Image.new('RGB',(180,160),'white').save(img)
        roi=modelrun/'roi_marked_inputs'/'case_roi_marked.jpg';Image.new('RGB',(120,100),'white').save(roi)
        case={'case_id':'test_001','cause':'MODEL_CLASSIFICATION','human_category_FOR_EVAL_ONLY':'pants','baseline_category':'top','image_path':str(img),'image_sha256':file_sha256(img),'bbox':[20,20,140,135]}
        pred={'case_id':'test_001','cause':'MODEL_CLASSIFICATION','baseline_category':'top','roi_category':'pants','roi_input':str(roi),'schema_valid':True,'native_json':False,'issues':[],'timing':{'end_to_end_ms':1100}}
        (modelrun/'case_plan.jsonl').write_text(json.dumps(case)+'\n');(modelrun/'roi_predictions.jsonl').write_text(json.dumps(pred)+'\n')
        (modelrun/'summary.json').write_text(json.dumps({'historical_GT_pilot_bbox_not_frozen_V5':True,'holdout_used':False}))
        build(modelrun,td/'output',td/'fashion_data',expected=1)
        htmltext=(td/'output'/'ROI15_VISUAL_REVIEW_STATIC.html').read_text()
        assert htmltext.count('<section class="card"')==1
        assert htmltext.count('data:image/jpeg;base64,')==2
        assert 'function exportJSON' in htmltext and 'function exportCSV' in htmltext
        assert json.loads((td/'output'/'audit_summary.json').read_text())['corrected']==1
    print('ROI15_VISUAL_AUDIT_SELFTEST_PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--run-dir',default='/workspace/PRD32_BINDING_ROI_GPU_20261027_run01')
    p.add_argument('--allowed-root',default='/workspace/fashion_data')
    p.add_argument('--output',default='/workspace/PRD32_ROI15_VISUAL_AUDIT_20261027_run01')
    a=p.parse_args()
    if a.self_test:selftest()
    else:build(a.run_dir,a.output,a.allowed_root)
