#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build static-first visual audit for existing PRD3.2 TRAIN-only five-class cases.
Does NOT infer, read sealed data or calculate official accuracy. No human labels leaked into models.
"""
import argparse,base64,csv,hashlib,html,io,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image

CATS=('top','pants','skirt','outerwear','dress','shoe','bag','accessory')
PROTECTED=('holdout','sealed','fullval','validation','valid','val','test')

def check_train_image(path):
    p=Path(path).resolve()
    parts={v.lower() for v in p.parts}
    if 'train' not in parts or parts.intersection(PROTECTED) or not p.is_file():
        raise ValueError('Unsafe or missing training image; not opened: '+str(path))
    return p

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def csv_write(path, data):
    if not data:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]))
        writer.writeheader(); writer.writerows(data)

def percent(x):
    return round(x,2)

def make_thumbnail(im, width=610):
    im=im.convert('RGB')
    im.thumbnail((width,680),Image.Resampling.LANCZOS)
    stream=io.BytesIO();im.save(stream,format='JPEG',quality=80,optimize=True)
    return 'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode('ascii')

def build(args):
    pred_path=Path(args.predictions)
    selected_path=Path(args.selected)
    preds=[json.loads(line) for line in pred_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    with selected_path.open(encoding='utf-8-sig',newline='') as f: sel=list(csv.DictReader(f))
    if not preds or not sel:raise ValueError('Empty predictions/selected')
    if len({r['case_id'] for r in preds})!=len(preds):raise ValueError('duplicate case_id')
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    records=[]
    byhint=defaultdict(Counter)
    n_issue=Counter()
    for p in preds:
        cid=p['case_id']; idx=int(cid.rsplit('_',1)[-1])-1
        if idx<0 or idx>=len(sel):raise ValueError('case id / selection index mismatch: '+cid)
        s=sel[idx]
        if s.get('inventory_category_NOT_GT')!=p.get('source_hint_NOT_GT'):
            raise ValueError('source hint mismatch: '+cid)
        if p.get('source_split')!='train' or p.get('bbox_origin')!='historical_inventory_gt_pilot_NOT_V5_prediction':
            raise ValueError('not an approved train-dev source: '+cid)
        can=p.get('canonical_tags') or {}
        predicted=can.get('category')
        reference=p['source_hint_NOT_GT']
        byhint[reference][str(predicted)]+=1
        n_issue.update(p.get('issues') or [])
        records.append({'case_id':cid,'image_id':p.get('image_id',''), 'target_instance_id':p.get('target_instance_id',''),
                        'reference_hint_NOT_GT':reference, 'model_category':predicted or '',
                        'hint_match_NOT_ACCURACY':int(reference==predicted),
                        'model_color_UNVERIFIED':can.get('primary_color') or '',
                        'model_pattern_UNVERIFIED':can.get('pattern') or '',
                        'model_visible_design_UNVERIFIED':can.get('visible_design') or '',
                        'issues':';'.join(p.get('issues') or []),
                        'end_to_end_ms':(p.get('timing') or {}).get('end_to_end_ms',''),
                        'bbox_origin':p['bbox_origin']})
    csv_write(out/'all40_prediction_vs_source_hint_NOT_GT.csv', records)
    mismatches=[p for p in preds if (p.get('canonical_tags') or {}).get('category')!=p.get('source_hint_NOT_GT')]
    diagnostic=[];cards=[]
    selected_verified=0
    for p in mismatches:
        cid=p['case_id'];idx=int(cid.rsplit('_',1)[-1])-1;s=sel[idx]
        image=check_train_image(s['resolved_image_path'])
        if sha(image)!=p['input_image_sha256'] or s.get('sha256','')!=p['input_image_sha256']:
            raise ValueError('SHA mismatch before preview: '+cid)
        box=[float(s[f'bbox_{v}']) for v in ('x1','y1','x2','y2')]
        if not all(math.isfinite(v) for v in box):raise ValueError('non-finite bbox: '+cid)
        with Image.open(image) as src:
            if not (0<=box[0]<box[2]<=src.width and 0<=box[1]<box[3]<=src.height):
                raise ValueError('out-of-bounds bbox before crop: '+cid)
            dx=.05*(box[2]-box[0]); dy=.05*(box[3]-box[1]); crop=src.crop((max(0,int(box[0]-dx)),max(0,int(box[1]-dy)),min(src.width,int(box[2]+dx)),min(src.height,int(box[3]+dy))))
            roi_data=make_thumbnail(crop)
        marked=Path(p.get('marked_image',''))
        if not marked.is_file():raise FileNotFoundError('Missing marked image: '+str(marked))
        with Image.open(marked) as src:
            full_data=make_thumbnail(src)
        ref=p['source_hint_NOT_GT']; model=(p.get('canonical_tags') or {}).get('category')
        ed={**next(r for r in records if r['case_id']==cid),'source_image_sha256':p['input_image_sha256'],'bbox_xyxy':json.dumps(box)}
        diagnostic.append(ed)
        esc=lambda x:html.escape(str(x or ''),quote=True)
        options=''.join(f'<option value="{cat}">{cat}</option>' for cat in CATS)
        cards.append(f'''<section class="card" data-case="{esc(cid)}"><h2>{esc(cid)} · 来源候选 {esc(ref)} / 模型预测 {esc(model)}</h2>
<div class="meta">image_id: {esc(p.get('image_id'))} ｜ target: {esc(p.get('target_instance_id'))} ｜ bbox: {esc([round(x,1) for x in box])}</div>
<div class="imgs"><figure><img loading="lazy" src="{full_data}"><figcaption>红框整图：核对框有没有指错服饰</figcaption></figure>
<figure><img loading="lazy" src="{roi_data}"><figcaption>5% padding 裁剪：核对目标是否完整</figcaption></figure></div>
<p><b>模型字段：</b>颜色 {esc((p.get('canonical_tags') or {}).get('primary_color'))}，图案 {esc((p.get('canonical_tags') or {}).get('pattern'))}，设计 {esc((p.get('canonical_tags') or {}).get('visible_design'))}</p>
<div class="fields"><label>目标框绑定<select class="bbox"><option value="">未审核</option><option value="VALID">正确且足够完整</option><option value="WRONG_TARGET">框选错误目标</option><option value="INCOMPLETE">目标框不完整</option><option value="UNCERTAIN">无法判断</option></select></label>
<label>人工确认类别<select class="category"><option value="">未审核</option>{options}<option value="uncertain">无法判断</option></select></label>
<label>错误原因<select class="cause"><option value="">未审核</option><option value="MODEL_TARGET_BINDING">目标正确但模型看错对象</option><option value="MODEL_CLASSIFICATION">目标正确但分类错误</option><option value="SOURCE_LABEL_ERROR">历史来源类别不准确</option><option value="UPSTREAM_BBOX">来源目标框错误</option><option value="CATEGORY_AMBIGUITY">类别界限模糊</option><option value="UNCERTAIN">仍需核查</option></select></label>
<label class="wide">备注<textarea class="notes" placeholder="仅记录可见证据和可能原因"></textarea></label></div></section>''')
        selected_verified+=1
    csv_write(out/'mismatch19_review_queue.csv',diagnostic)
    stats={'schema':'prd32_train5_40case_diagnostic_v1','status':'SOURCE_HINT_CONCORDANCE_ONLY_NOT_VERIFIED_ACCURACY',
           'input_count':len(preds),'hint_match_count':sum(r['hint_match_NOT_ACCURACY'] for r in records),
           'hint_mismatch_count':len(mismatches),'model_categories':dict(Counter(r['model_category'] for r in records)),
           'hint_categories':dict(Counter(r['reference_hint_NOT_GT'] for r in records)),
           'confusion_source_hint_rows_vs_model_categories':{h:dict(v) for h,v in byhint.items()},
           'issue_counts':dict(n_issue),'native_json_count':sum(bool(x.get('native_json')) for x in preds),
           'service_schema_valid_count':sum(bool(x.get('schema_valid')) for x in preds),
           'end_to_end_ms_p50':round(statistics.median(float(r['end_to_end_ms']) for r in records),2),
           'image_cases_embedded':selected_verified,
           'warning':'Source hints NOT independent human GT; only train dev diagnosis. No Qwen inference or holdout access. Review mismatched bboxes before interpreting model errors.'}
    (out/'analysis_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # The review content is literal HTML, visible even when JS is disabled.
    script='''
function collect(){let rows=[];document.querySelectorAll('section.card').forEach(s=>{let get=k=>s.querySelector('.'+k).value;rows.push({case_id:s.dataset.case,bbox_status:get('bbox'),human_category:get('category'),cause:get('cause'),notes:get('notes'),review_status:(get('bbox')&&get('category')&&get('cause'))?'REVIEWED':'PENDING'});});return {schema:'prd32_train5_binding_diagnostic_review_v1',scope:'train_dev_diagnostic_only_NOT_INDEPENDENT',rows};}
function saveText(name,content,format){let b=new Blob([content],{type:format});let a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),2000);}
function exportJSON(){saveText('PRD32_train5_binding_review.json',JSON.stringify(collect(),null,2),'application/json;charset=utf-8');}
function exportCSV(){let rows=collect().rows;let keys=['case_id','bbox_status','human_category','cause','notes','review_status'];let q=v=>'"'+String(v??'').replace(/"/g,'""')+'"';let data=[keys.join(','),...rows.map(r=>keys.map(k=>q(r[k])).join(','))].join('\\r\\n');saveText('PRD32_train5_binding_review.csv','\\uFEFF'+data,'text/csv;charset=utf-8');}
'''
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PRD3.2 五类40图 · 错误目标绑定复核</title><style>
body{font:15px/1.55 system-ui,'Microsoft YaHei',sans-serif;background:#f3f5f9;color:#172034;margin:0}main{max-width:1140px;margin:24px auto;padding:0 12px}
header{background:#19304a;color:white;padding:24px;border-radius:14px}h1{margin:0 0 8px}.card{background:white;border:1px solid #d5dce6;border-radius:12px;padding:20px;margin:18px 0}h2{font-size:18px;margin:0 0 7px}.meta{color:#586879;font-size:12px}.imgs{display:grid;grid-template-columns:1fr 1fr;gap:10px;align-items:start}figure{margin:8px 0}img{width:100%;max-height:500px;object-fit:contain;background:#ecf0f4;border-radius:8px}figcaption{text-align:center;color:#586879;font-size:12px}.fields{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}label{display:flex;flex-direction:column;font-size:13px;gap:5px}select,textarea{padding:8px;font:inherit;border:1px solid #adbbcb;border-radius:6px;background:white}.wide{grid-column:1/-1}textarea{min-height:60px}button{cursor:pointer;border:0;border-radius:7px;padding:10px 16px;background:#2a60a5;color:white;margin:9px 7px 2px 0}.warn{background:#fff3d6;border-radius:7px;color:#5c400a;padding:10px;margin:12px 0}@media(max-width:730px){.imgs,.fields{grid-template-columns:1fr}}
</style></head><body><main><header><h1>PRD 3.2 五类开发诊断 · 19 条来源类别不一致复核</h1><p>本页使用已生成的红框整图与目标局部裁剪，不会运行模型。核对目标框、历史来源标签与模型输出的差异。</p><button onclick="exportJSON()">导出复核 JSON</button><button onclick="exportCSV()">导出复核 CSV</button></header>
<div class="warn">仅训练来源开发诊断，不能作为盲测 GT 或 PRD 独立验收。历史来源类别不是经本次确认的人工 GT；建议先核对 bbox，再判断分类错误。HTML 卡片在没有 JavaScript 时也会显示。</div>
'''+''.join(cards)+'<script>'+script+'</script></main></body></html>'
    (out/'TARGET_BINDING_REVIEW_STATIC.html').write_text(page,encoding='utf-8')
    print('TARGET_BINDING_DIAGNOSTIC_COMPLETE',json.dumps({'n':len(preds),'hint_mismatch':len(mismatches),'html_cards':len(cards),'output':str(out)},ensure_ascii=False))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions',required=True)
    ap.add_argument('--selected',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args();build(args)
if __name__=='__main__':main()
