#!/usr/bin/env python3
"""Build a blind, entirely offline, red-bbox manual review of 17 previously unreviewed train-source cases.

The reviewer never sees model guesses or source hints in the review page. This is development-only.
"""
import argparse
import base64
from collections import Counter
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATEGORIES = ["top", "pants", "skirt", "outerwear", "dress", "shoe", "bag", "accessory", "uncertain"]
YES_NO = ["YES", "NO", "UNCERTAIN"]

def registry():
    rows = json.loads((ROOT / 'all40_case_registry.json').read_text(encoding='utf-8'))
    assert len(rows) == 40 and len({r['case_id'] for r in rows}) == 40, '40-case manifest mismatch'
    selected = sorted((r for r in rows if r['priority'] == 'P2_NOT_HUMAN_REVIEWED'), key=lambda r: r['case_id'])
    assert len(selected) == 17, f'Expected exactly 17, found {len(selected)}'
    known19 = json.loads((ROOT/'previous_human_review_19.json').read_text(encoding='utf-8'))['rows']
    known4 = json.loads((ROOT/'previous_human_review_4.json').read_text(encoding='utf-8'))['rows']
    keys19 = {r['case_id'] for r in known19}; keys4 = {r['case_id'] for r in known4}
    assert len(keys19) == 19 and len(keys4) == 4 and not(keys19 & keys4)
    assert ({r['case_id'] for r in selected} | keys19 | keys4) == {r['case_id'] for r in rows}
    return selected

def build(output, image_root=None, test_only=False):
    selected = registry()
    cards=[]
    for r in selected:
        cid=r['case_id']
        orig=Path(r['marked_image'])
        pic=(Path(image_root) / orig.name) if image_root else orig
        if test_only:
            image='<div class="pic">SELF-TEST placeholder. No real image read.</div>'
        else:
            if not pic.is_file():
                raise FileNotFoundError(f'Marked image for {cid} not found: {pic}')
            if pic.stat().st_size > 15_000_000:
                raise ValueError(f'Unexpectedly large image {pic}')
            mime={'jpg':'image/jpeg','jpeg':'image/jpeg','png':'image/png','webp':'image/webp'}.get(pic.suffix.lower().lstrip('.'))
            if not mime:
                raise ValueError(f'Unsupported format: {pic}')
            b64=base64.b64encode(pic.read_bytes()).decode('ascii')
            image=f'<img alt="{cid} target image with red bbox" src="data:{mime};base64,{b64}" />'
        def select(name, opts):
            return f'<label>{name}<select data-field="{name}" required><option value="">请选择</option>'+''.join(f'<option value="{html.escape(a)}">{html.escape(a)}</option>' for a in opts)+'</select></label>'
        cards.append(f'''<section class="case" data-case="{html.escape(cid)}">
<h2>{html.escape(cid)}</h2><p>先看红框所指的目标本身。不要根据背景里其他服饰判断，也不要猜看不清的类别。</p>{image}
<div class="fields">{select('human_category',CATEGORIES)}{select('bbox_coverage',YES_NO)}{select('visually_identifiable',YES_NO)}{select('other_garments_visible',YES_NO)}</div>
<label>人工审核备注<textarea data-field="notes" rows="2" placeholder="记录可见证据、遮挡与不确定性；例如：目标上衣清晰，但框内有裙摆局部。"></textarea></label>
</section>''')
    css='''body{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1060px;margin:auto;padding:22px;background:#f3f6fa;color:#1c2b3a}
.top{position:sticky;top:0;background:white;padding:14px;border:1px solid #c9d4e0;border-radius:10px;z-index:5;display:flex;gap:18px;align-items:center;justify-content:space-between}
button{background:#185db1;border:0;border-radius:8px;padding:11px 18px;color:white;cursor:pointer;font-size:15px}
.case{background:white;padding:18px;border-radius:10px;margin:18px 0;border:1px solid #cfd8e3}
h2{margin:0 0 10px;font-size:18px}p{font-size:13px;color:#576980}img{display:block;margin:12px auto;max-width:100%;max-height:690px;object-fit:contain}
.pic{padding:60px;background:#eee}.fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
label{font-weight:600;font-size:13px;display:block;margin:12px 0}select,textarea{font-family:inherit;font-size:14px;display:block;width:100%;margin-top:6px;padding:8px;box-sizing:border-box;border:1px solid #b5c4d1;border-radius:6px}
@media(max-width:680px){.fields{grid-template-columns:1fr}}
'''
    js='''
function exportAll(){
 let rows=[], missing=[];
 document.querySelectorAll('.case').forEach(card=>{
  const row={case_id:card.dataset.case};
  card.querySelectorAll('[data-field]').forEach(el=>row[el.dataset.field]=el.value.trim());
  if(!row.human_category||!row.bbox_coverage||!row.visually_identifiable||!row.other_garments_visible) missing.push(row.case_id);
  rows.push(row);
 });
 if(missing.length){alert('请先完成所有17张图的四个选择字段。未完成：'+missing.join(', '));return;}
 const payload={schema:'prd32_20261028_new17_blind_visual_review_v1',scope:'train_source_dev_not_independent',
 note:'Blind review of the 17 previously unreviewed cases; prior 23 annotations untouched.',rows};
 const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json;charset=utf-8'});
 const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='PRD32_Remaining17_Human_Review_20261028.json';
 document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
'''
    webpage=f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><title>PRD 3.2 Remaining 17 Blind Review</title><style>{css}</style></head>
<body><h1>PRD 3.2 — 尚未人工审核的 17 条样本</h1><p>10.28 开发集人工复核。原始来源标签和所有 V2/V3/V4 预测均不在此页面显示，避免影响人工判断。仅用于训练来源开发诊断，非独立验收。</p>
<div class="top"><strong>17 张目标红框图 · 需全部填写</strong><button onclick="exportAll()">导出审核 JSON</button></div>
{''.join(cards)}<script>{js}</script></body></html>'''
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(webpage,encoding='utf-8')
    print(json.dumps({'status':'HTML_WRITTEN' if not test_only else 'SELF_TEST_OK','output':str(dest),'selected':len(selected),'embedded_images':0 if test_only else len(selected)},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--image-root');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    build(a.output,a.image_root,a.self_test)
