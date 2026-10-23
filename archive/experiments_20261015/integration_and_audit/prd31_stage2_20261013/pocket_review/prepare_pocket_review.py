#!/usr/bin/env python3
"""Read-only source-image QA export for 24 unconfirmed Fashionpedia pocket proposals.

This tool NEVER changes GT, repository files, or review decisions. The export includes
both original photos and garment/part bounding box overlays to permit *human review*.
Requires Pillow; all other modules are standard library.
"""
import argparse
import base64
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
from PIL import Image, ImageDraw, ImageFont

ALLOWED = {"train", "val"}

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_path(project_root, relpath):
    """Resolve only expected Fashionpedia data files (never arbitrary local files)."""
    project_root = Path(project_root).resolve()
    base = (project_root.parent / 'fashion_data' / 'raw' / 'fashionpedia' / 'images').resolve()
    p = (project_root / relpath).resolve()
    if not p.is_relative_to(base) or p.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
        raise ValueError(f"Unexpected source path outside Fashionpedia: {relpath}")
    return p


def preflight(cases, project_root):
    findings = []
    bad = []
    existing = {}
    for case in cases:
        rid = case['record_id']
        try:
            if case['split'] not in ALLOWED:
                raise ValueError('Unknown split')
            p = source_path(project_root, case['source_image'])
            if not p.is_file():
                raise FileNotFoundError(str(p))
            actual = existing.setdefault(str(p), digest(p))
            expected = case['source_image_sha256'].lower()
            if expected != actual:
                raise ValueError(f'SHA256 MISMATCH expected={expected} actual={actual}')
            with Image.open(p) as im:
                im.verify()
            findings.append((case, p, actual))
        except Exception as e:
            bad.append({'record_id': rid, 'path': str(case.get('source_image','')), 'reason': str(e)})
    return findings, bad


def outline_image(p, case):
    with Image.open(p) as im:
        original = im.convert('RGB')
    original_w, original_h = original.size
    overlay = original.copy()
    draw = ImageDraw.Draw(overlay)
    target = case['garment_bbox']
    x0,y0,x1,y1 = target
    draw.rectangle((x0,y0,x1,y1), outline='#41f06b', width=max(3, min(original.size)//300))
    for i, part in enumerate(case['parts'], 1):
        x,y,w,h = part['part_bbox_xywh']
        draw.rectangle((x,y,x+w,y+h), outline='#ffcf39', width=max(3, min(original.size)//300))
        draw.text((max(0,x),max(0,y-15)),f'P{i}',fill='#ffcf39',stroke_width=2,stroke_fill='#202020')
    return original,overlay


def thumbnail_b64(im, max_dim=1050, quality=82):
    import io
    pic=im.copy(); pic.thumbnail((max_dim,max_dim))
    out=io.BytesIO(); pic.save(out,'JPEG',quality=quality,optimize=True)
    return 'data:image/jpeg;base64,'+base64.b64encode(out.getvalue()).decode()


def triage(cases):
    image_counts = Counter(x['source_image'] for x in cases)
    result=[]
    for c in cases:
        flags=[]
        if image_counts[c['source_image']]>1: flags.append('同一原图有多件候选衣物')
        if len(c['parts'])>1: flags.append('同一衣物多个口袋部件')
        for i,p in enumerate(c['parts'],1):
            x,y,w,h=p['part_bbox_xywh']
            if w*h < 450: flags.append(f'P{i}框面积较小')
            if min(w,h) < 13: flags.append(f'P{i}细长/窄框')
            covs=p.get('ranked_garment_coverages') or []
            if len(covs)>1 and covs[1]['part_coverage']>0.02: flags.append(f'P{i}存在其他衣物重叠')
        result.append({'record_id':c['record_id'],'split':c['split'],'category':c['category'],
        'image_id':c['image_id'],'source_image_sha256':c['source_image_sha256'],
        'num_parts':len(c['parts']),'triage_flags':'；'.join(dict.fromkeys(flags)),
        'review_status':'pending'})
    return result


def review_html(records, dataset_sha):
    # Client-side annotation; images display the originals and visually emphasized *proposals*.
    # Initial states remain pending. No predictions are used for evaluation here.
    encoded=json.dumps({'source_cases_sha256':dataset_sha,'records':records},ensure_ascii=False).replace('</','<\\/')
    return r'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>PRD 3.1.2 口袋关联 24 条人工审核</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:14px/1.55 system-ui,"Microsoft YaHei",sans-serif;color:#222;background:#f5f7f9;margin:0}
header{background:#172536;color:#fff;padding:14px 22px;position:sticky;top:0;z-index:3}
main{max-width:1280px;margin:auto;padding:16px}.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0}
button,select,textarea,input{font:inherit}button{background:#fff;border:1px solid #aeb9c5;padding:7px 12px;border-radius:7px;cursor:pointer}
button.primary{background:#234fc5;color:white;border-color:#234fc5}button:disabled{opacity:.35;cursor:default}
.panel{background:#fff;border:1px solid #dce3e9;border-radius:12px;padding:16px;margin:12px 0}
.pictures{display:grid;grid-template-columns:1fr 1fr;gap:12px}.pictures img{max-width:100%;max-height:65vh;object-fit:contain;background:#e0e5eb}
small,.muted{color:#657487}.part{padding:12px 0;border-top:1px solid #dce3e9}
.form{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.form label{display:flex;gap:5px;align-items:center}select{padding:7px;min-width:125px}
textarea{width:100%;min-height:52px;box-sizing:border-box;border:1px solid #c2cbd4;border-radius:6px;padding:7px}
.bad{color:#ac302c;font-weight:700}.ok{color:#16763e;font-weight:700}
code{font-size:12px} .legend{font-weight:700;background:#e8f2fc;padding:8px;border-radius:6px}
@media(max-width:750px){.pictures{grid-template-columns:1fr}}
</style></head><body>
<header><b>PRD 3.1.2 — 口袋关联人工审核 · 24条 / 42部件</b> <span id="progress"></span></header>
<main><div class="panel"><b>使用约定：</b>左为原始照片，右为部件提议叠加。<span style="color:green">绿色框=候选服装框</span>，
<span style="color:#b27800">黄色框=Fashionpedia口袋部件框</span>。它们是<strong>未人工确认的提议，不是GT。</strong>
需要对每个P号分别确认口袋是否属于绿色框目标衣物、部件框位置、口袋类型。
仅检查衣物归属/部件提议，不计算检测召回率。审核完导出JSON并上传。train/val严格保持分开。</div>
<div class="toolbar"><button onclick="move(-1)">上一条</button><select id="jump" onchange="jumpTo(+this.value)"></select><button onclick="move(1)">下一条</button>
<button class="primary" onclick="exportJSON()">导出人工审核JSON</button><button onclick="document.getElementById('restore').click()">导入此前导出的JSON（恢复）</button>
<input type="file" id="restore" accept=".json" style="display:none" onchange="restoreJSON(this.files[0])"/>
</div><section id="view" class="panel"></section><p class="muted">刷新前建议导出备份。该页面可离线使用，不向服务器发送图片或标注。</p></main>
<script id="dataset" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('dataset').textContent),rows=data.records;
let pos=0;const statusOpts=[['pending','待审核'],['yes','确认属实'],['no','确认不符'],['uncertain','无法可靠判断']];
const escapeHTML=s=>String(s??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const options=(v)=>statusOpts.map(([k,n])=>`<option value="${k}" ${v===k?'selected':''}>${n}</option>`).join('');
function progress(){let all=rows.reduce((n,r)=>n+r.parts.length,0);let done=rows.reduce((n,r)=>n+r.parts.filter(p=>['association','bbox','type'].every(k=>p.review[k]!=='pending')).length,0);
 document.getElementById('progress').textContent=`· 已完成单部件 ${done}/${all} · 记录 ${pos+1}/24`;
 let jump=document.getElementById('jump');jump.innerHTML=rows.map((r,i)=>`<option value="${i}" ${pos===i?'selected':''}>${String(i+1).padStart(2,'0')} ${r.record_id} (${r.split})</option>`).join('');}
function render(){const r=rows[pos];document.getElementById('view').innerHTML=`
 <h2>${pos+1}/24　${escapeHTML(r.record_id)}　<small>${r.split} / ${r.category} / ${r.parts.length}个口袋部件</small></h2>
 <div class="legend">原图 SHA256 ${escapeHTML(r.source_image_sha256)}　｜候选衣物 ID ${escapeHTML(r.native_garment_annotation_id)}</div>
 <div class="pictures"><div><h3>原图（不带框）</h3><img src="${r.raw_src}"/></div><div><h3>原生部件提议（待人审）</h3><img src="${r.overlay_src}"/></div></div>
 ${r.parts.map((p,i)=>`<div class="part"><b>部件 P${i+1} · ID ${p.id}</b>　<small>${escapeHTML(p.type)} · 框 ${p.bbox.join(', ')}</small>
 <div class="form"><label>属于绿色框衣物 <select onchange="change(${i},'association',this.value)">${options(p.review.association)}</select></label>
 <label>口袋框定位 <select onchange="change(${i},'bbox',this.value)">${options(p.review.bbox)}</select></label>
 <label>口袋类型 <select onchange="change(${i},'type',this.value)">${options(p.review.type)}</select></label></div>
 <textarea placeholder="遮挡、重叠、误关联或其他歧义（可空）" oninput="change(${i},'note',this.value)">${escapeHTML(p.review.note)}</textarea>
 </div>`).join('')}
 <label><b>本条总体备注</b><textarea oninput="rows[pos].case_note=this.value">${escapeHTML(r.case_note)}</textarea></label>`;
 progress();}
function change(i,k,v){rows[pos].parts[i].review[k]=v;progress()}
function move(d){pos=Math.max(0,Math.min(rows.length-1,pos+d));render()}
function jumpTo(i){pos=i;render()}
function exportJSON(){let out={schema_version:'prd312_pocket_human_review_v1',source_cases_sha256:data.source_cases_sha256,
 export_timestamp:new Date().toISOString(),records:rows.map(r=>({record_id:r.record_id,split:r.split,category:r.category,
 source_image_sha256:r.source_image_sha256,garment_annotation_id:r.native_garment_annotation_id,case_note:r.case_note,
 parts:r.parts.map(p=>({part_annotation_id:p.id,part_type_proposal:p.type,review:p.review}))}))};
 let blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);
 a.download='prd312_pocket24_HUMAN_REVIEW.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),2000)}
function restoreJSON(file){if(!file)return;let f=new FileReader();f.onload=()=>{try{let x=JSON.parse(f.result);if(x.source_cases_sha256!==data.source_cases_sha256)throw Error('候选文件哈希不一致，拒绝导入');
 let m=new Map((x.records||[]).map(r=>[r.record_id,r]));rows.forEach(r=>{let old=m.get(r.record_id);if(!old)return;
 r.case_note=old.case_note??'';let parts=new Map((old.parts||[]).map(p=>[p.part_annotation_id,p]));r.parts.forEach(p=>{let op=parts.get(p.id);if(op&&op.review)p.review=op.review})});render();alert('审核状态已从JSON恢复')
 }catch(e){alert('导入失败：'+e.message)}};f.readAsText(file)}
render();
</script></body></html>'''.replace('__DATA__',encoded)


def prepare(case_path,project_root,out_dir):
    case_path=Path(case_path); out_dir=Path(out_dir)
    cases=json.loads(case_path.read_text('utf8'))
    if len(cases)!=24 or len({x['record_id'] for x in cases})!=24:
        raise ValueError('Expected exactly 24 unique historic review cases')
    if sum(len(c['parts']) for c in cases)!=42:
        raise ValueError('Unexpected part count; stop and inspect source version')
    out_dir.mkdir(parents=True,exist_ok=True)
    entries,bad=preflight(cases,project_root)
    (out_dir/'preflight.json').write_text(json.dumps({'passed':not bad,'ready_cases':len(entries),'error_cases':bad,'source_cases_sha256':digest(case_path)},ensure_ascii=False,indent=2),'utf8')
    if bad:
        print(f'PREFLIGHT_FAILED {len(bad)} source images (no human review page created)')
        print('Please upload preflight.json'); return 2
    originals=out_dir/'originals';overlays=out_dir/'overlays';originals.mkdir(exist_ok=True);overlays.mkdir(exist_ok=True)
    records=[]
    for j,(c,p,sha) in enumerate(entries,1):
        raw,overlay=outline_image(p,c)
        original_dst=originals/f'{j:02d}_{c["record_id"]}{p.suffix.lower()}'
        overlay_dst=overlays/f'{j:02d}_{c["record_id"]}.jpg'
        shutil.copyfile(p,original_dst) # exact original raw bytes
        overlay.save(overlay_dst,quality=88,optimize=True)
        records.append({'record_id':c['record_id'],'split':c['split'],'category':c['category'],
           'native_garment_annotation_id':c['native_garment_annotation_id'],'source_image_sha256':sha,
           'raw_src':thumbnail_b64(raw),'overlay_src':thumbnail_b64(overlay),'case_note':'',
           'parts':[{'id':x['part_annotation_id'],'bbox':x['part_bbox_xywh'],
           'type':', '.join(a['name'] for a in x.get('part_attributes',[])),
           'review':{'association':'pending','bbox':'pending','type':'pending','note':''}} for x in c['parts']]})
    (out_dir/'prd312_pocket24_review_OFFLINE.html').write_text(review_html(records,digest(case_path)),encoding='utf8')
    with open(out_dir/'visual_triage_24_NOT_MODEL_ACCURACY.csv','w',newline='',encoding='utf-8-sig') as f:
        rows=triage(cases); writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out_dir/'run_summary.json').write_text(json.dumps({'status':'EXPORT_READY_PENDING_HUMAN_REVIEW','case_count':len(cases),'unique_source_images':len(set(x['source_image_sha256'] for x in cases)),
       'parts':sum(len(x['parts']) for x in cases),'train_cases':sum(x['split']=='train' for x in cases),
       'val_cases':sum(x['split']=='val' for x in cases),'source_cases_sha256':digest(case_path),
       'reviewed_parts':0,'GT_updated':False},ensure_ascii=False,indent=2),'utf8')
    import zipfile
    output_zip=out_dir.parent/(out_dir.name+'_UPLOAD_THIS.zip')
    with zipfile.ZipFile(output_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for fn in sorted(out_dir.rglob('*')):
            if fn.is_file():z.write(fn,fn.relative_to(out_dir))
    print('POCKET24_IMAGES_AND_OFFLINE_REVIEW_READY')
    print('UPLOAD_THIS_ZIP:',output_zip)
    print('REVIEW_HTML:',out_dir/'prd312_pocket24_review_OFFLINE.html')
    print('All inputs unchanged; review decisions all pending.')
    return 0


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cases',default=str(Path(__file__).parent/'inputs/review_cases.json'))
    ap.add_argument('--project-root',default='/workspace/fashion_multimodal_analysis')
    ap.add_argument('--out-dir',default='/workspace/prd312_pocket24_review_20261013')
    a=ap.parse_args();sys.exit(prepare(a.cases,a.project_root,a.out_dir))
if __name__=='__main__':main()
