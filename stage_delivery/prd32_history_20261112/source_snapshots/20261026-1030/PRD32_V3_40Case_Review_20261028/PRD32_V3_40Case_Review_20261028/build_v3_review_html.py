#!/usr/bin/env python3
"""Generate a completely offline, read-only image review of the PRD32 V3 train-dev cohort.

Default: ONLY the four potential V3 regressions. --all: all 40 cases.
No model call, no file mutations except HTML output; no upload/remote services.
"""
import argparse
import base64
import html
import json
import mimetypes
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIORITY = {'P0_POSSIBLE_REGRESSION': 0,'P1_REVIEWED_RESIDUAL': 1,
            'P1_PROVENANCE_OR_AMBIGUITY': 2,'P2_NOT_HUMAN_REVIEWED': 3,
            'P2_PREVIOUSLY_REVIEWED': 4}
CATS = ['top','pants','skirt','outerwear','dress','shoe','bag','accessory','uncertain']


def make(args):
    cases = json.loads(Path(args.cases).read_text(encoding='utf-8'))
    assert len(cases) == 40 and len(set(x['case_id'] for x in cases)) == 40
    cases.sort(key=lambda x: (PRIORITY.get(x['priority'], 99),x['case_id']))
    if not args.all:
        cases = [x for x in cases if x['priority'] == 'P0_POSSIBLE_REGRESSION']
        assert len(cases) == 4
    cards = []
    missing = []
    for case in cases:
        cid = html.escape(case['case_id'])
        name = Path(case['marked_image']).name
        image_path = Path(args.image_root) / name if args.image_root else Path(case['marked_image'])
        if image_path.is_file() and not args.no_images:
            if image_path.stat().st_size > 15_000_000:
                raise ValueError('Unexpected image over 15MB: '+str(image_path))
            mimetype = mimetypes.guess_type(name)[0] or 'image/jpeg'
            if mimetype not in {'image/jpeg','image/png','image/webp'}:
                raise ValueError('Unsupported image type: '+name)
            encoded = base64.b64encode(image_path.read_bytes()).decode('ascii')
            pic = f'<img loading="lazy" class="photo" alt="{cid} original input with target rectangle" src="data:{mimetype};base64,{encoded}">'
        else:
            missing.append(str(image_path))
            pic = f'<div class="missing">Marked image unavailable: {html.escape(str(image_path))}</div>'
        detail = (f"Source hint (NOT GT): {case['source_hint_NOT_GT']}  |  "
                  f"V2: {case['v2_category']}  |  V3: {case['v3_category']}  |  "
                  f"category_only: {case['category_only']}  |  "
                  f"Prior human: {case['historical_review_category']} / {case['historical_review_cause']}")
        opts = '<option value="">-- Choose after visually reviewing --</option>' + ''.join(
            f'<option value="{z}">{z}</option>' for z in CATS)
        field = lambda key, options: (f'<label>{key}<select data-field="{key}"><option value="">-- optional --</option>' +
                                      ''.join(f'<option value="{z}">{z}</option>' for z in options) + '</select></label>')
        cards.append(f'''<section class="case" data-id="{cid}">
         <h2>{cid} <span class="badge">{html.escape(case['priority'])}</span></h2>
         <p class="guidance">Inspect the red rectangle target in the ORIGINAL image. Record the visual class before opening the predictions.</p>
         {pic}
         <div class="fields"><label><b>Human target category</b><select data-field="human_category">{opts}</select></label>
         {field('bbox_coverage',['YES','NO','UNCERTAIN'])}
         {field('visually_identifiable',['YES','NO','UNCERTAIN'])}
         {field('other_garments_visible',['YES','NO','UNCERTAIN'])}</div>
         <label class="notes">Evidence and uncertainty notes<textarea data-field="notes" rows="2" placeholder="e.g. Red box targets the yellow trousers, but white T-shirt overlaps the crop."></textarea></label>
         <details><summary>Reveal source/model history after recording your independent label</summary>
           <p>{html.escape(detail)}</p>
           <p>V3 tags: {html.escape(json.dumps(case['v3_tags'],ensure_ascii=False))}</p>
         </details></section>''')
    if missing and not args.no_images:
        raise FileNotFoundError(f'{len(missing)} marked image(s) missing; first: {missing[0]}')
    js='''
function exportReview(){
 const rows=[];
 document.querySelectorAll('.case').forEach(card=>{
  const obj={case_id:card.dataset.id};
  card.querySelectorAll('[data-field]').forEach(el=>{obj[el.dataset.field]=el.value.trim()});
  if(obj.human_category)rows.push(obj);
 });
 if(rows.length===0){alert('Select at least one human target category before export.');return;}
 const payload={schema:'prd32_v3_40_case_additive_visual_review_v1',scope:'train_dev_diagnostic_only_NOT_INDEPENDENT',
 note:'This review is ADDITIVE and does not overwrite frozen historical human annotations.',rows};
 const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json;charset=utf-8'});
 const url=URL.createObjectURL(blob); const a=document.createElement('a');
 a.href=url;a.download='PRD32_V3_40_Manual_Review_20261028.json';a.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
}
'''
    css='''
:root{font-family:Arial,"Microsoft YaHei",sans-serif;color:#17212c;background:#f6f8fa}
body{max-width:1140px;margin:auto;padding:25px}
h1{font-size:25px}p{line-height:1.5}.lead{color:#5a6877}.toolbar{position:sticky;top:0;background:#fff;border:1px solid #ccd5df;padding:12px 16px;border-radius:10px;z-index:2;display:flex;align-items:center;gap:15px}
button{background:#1465bc;color:white;padding:10px 18px;border:0;border-radius:7px;cursor:pointer;font-weight:bold}
.case{margin:18px 0;padding:20px;background:white;border:1px solid #d7dee6;border-radius:10px;box-shadow:0 1px 4px #ddd}.badge{font-size:11px;background:#ebeff5;padding:5px;border-radius:5px;color:#475569}
h2{margin-top:0;font-size:18px}.photo{max-width:100%;height:auto;max-height:650px;object-fit:contain;background:#eee;border-radius:7px;display:block;margin:auto}.missing{color:#a3312d;border:1px dashed;padding:25px}
.guidance{font-size:13px;color:#64748b}.fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:16px 0}
label{font-size:13px;font-weight:600;display:block}select,textarea{display:block;width:100%;box-sizing:border-box;padding:8px;border:1px solid #bfcbd8;border-radius:6px;background:#fff;margin-top:7px}details{margin-top:15px;border-top:1px solid #ddd;padding-top:11px;font-size:13px;color:#576779}summary{cursor:pointer;color:#0b5a9e}.notes{margin:10px 0}@media(max-width:720px){.fields{grid-template-columns:1fr}}
'''
    body='''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>PRD32 V3 40-Case Review</title><style>'''+css+'''</style></head><body>
 <h1>PRD 3.2 — V3 Source-Regressions Visual Review</h1><p class="lead">2026-10-28 • Training-source development cohort. NOT independent test or PRD acceptance. No hidden answer shown before review.</p>
 <div class="toolbar"><b>Review: '''+str(len(cases))+''' case(s)</b><button type="button" onclick="exportReview()">Export JSON</button><span>Start with four potential regressions; mark "uncertain" if the actual target class cannot be resolved.</span></div>'''+''.join(cards)+'''<script>'''+js+'''</script></body></html>'''
    target = Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(body,encoding='utf-8')
    print(json.dumps({'status':'HTML_WRITTEN','path':str(target),'cases':len(cases),'missing_images':len(missing),'embedded_images':len(cases)-len(missing)},ensure_ascii=False))

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cases',default=str(HERE/'V3_40_Case_Comparison.json'))
    p.add_argument('--output',required=True)
    p.add_argument('--all',action='store_true',help='Include all 40 instead of only 4 top-priority cases')
    p.add_argument('--image-root',default=None,help='Optional relocated marked_images directory')
    p.add_argument('--no-images',action='store_true',help='HTML smoke test only: omit photo image data')
    make(p.parse_args())
