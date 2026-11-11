#!/usr/bin/env python3
"""Build a single-file offline HTML review page on AutoDL, no GPU, no network or deps.

Usage: python build_visual_diagnosis.py --output /workspace/PRD32_RootCause18_Review.html
Inputs default to RootCause_18_ReviewSeed.json beside this script. Images are taken
from marked_image_autodl in the V3 full_tags run (actual marked original full image).
"""
import argparse,base64,json,pathlib,mimetypes,sys
HERE=pathlib.Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',default=str(HERE/'RootCause_18_ReviewSeed.json'))
    p.add_argument('--output',default=str(HERE/'RootCause_18_Visual_Review.html'))
    p.add_argument('--image-dir',default='',help='Optional directory containing marked images by basename')
    p.add_argument('--strict-images',action='store_true',help='Require all 18 images; fail before writing if missing')
    args=p.parse_args();data=json.loads(pathlib.Path(args.seed).read_text('utf-8'))
    n_images=0;missing=[]
    for r in data['rows']:
        src=pathlib.Path(r['marked_image_autodl'])
        if args.image_dir:
            cand=pathlib.Path(args.image_dir)/src.name
            if cand.is_file():src=cand
        if src.is_file():
            raw=src.read_bytes();tp=mimetypes.guess_type(src.name)[0] or 'image/jpeg'
            r['picture']='data:'+tp+';base64,'+base64.b64encode(raw).decode('ascii')
            n_images+=1
        else:
            r['picture']='';missing.append({'case_id':r['case_id'],'expected_image':str(src)})
    if args.strict_images and missing:
        print(json.dumps({'status':'IMAGES_MISSING','missing':missing},ensure_ascii=False,indent=2));sys.exit(2)
    embedded_data=json.dumps(data['rows'],ensure_ascii=False).replace('</','<\\/')
    page=r'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PRD 3.2 | Root-cause diagnosis | 2026-10-29</title>
<style>
:root{color-scheme:light;font-family:system-ui,"Microsoft Yahei",sans-serif;color:#17253a}body{max-width:1440px;margin:auto;padding:26px;background:#f6f8fc}h1{margin:0 0 5px;font-size:26px}h2{margin:6px 0 12px;font-size:19px}p{line-height:1.55}small,.muted{color:#586b81}.headline{background:#fff;padding:22px 26px;border:1px solid #d8e0eb;border-radius:16px;margin-bottom:20px}.toolbar{display:flex;flex-wrap:wrap;gap:9px;align-items:center;margin:15px 0}.btn{border:1px solid #1b4d85;background:#174b84;color:#fff;border-radius:7px;padding:9px 15px;cursor:pointer;font-weight:650}.btn.secondary{background:white;color:#174b84}.stats{display:flex;gap:10px;flex-wrap:wrap}.stat{background:#eef3fa;padding:8px 12px;border-radius:8px}.case{padding:21px 23px;background:white;border:1px solid #d9e0e8;border-radius:16px;margin-bottom:20px}.casehead{display:flex;justify-content:space-between;gap:20px;align-items:center}.casehead b{font-size:19px}.tag{display:inline-block;padding:5px 9px;border-radius:6px;background:#edf2fc;color:#285185;font-size:12px;margin-left:10px}.content{display:grid;grid-template-columns:minmax(300px,1fr) minmax(450px,1.25fr);gap:20px;margin-top:14px}.pic{background:#f2f5f9;border-radius:10px;min-height:260px;display:flex;justify-content:center;align-items:center;overflow:hidden}.pic img{max-width:100%;max-height:630px;object-fit:contain}.missing{padding:26px;color:#ad402d}.modelgrid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.model{border:1px solid #d7e0e9;border-radius:10px;padding:13px;background:#fcfdff;min-width:0}.model strong{font-size:17px}.err{color:#b42318}.ok{color:#197456}.attrs{line-height:1.8;font-size:14px}.evidence{background:#fff9e8;border-left:3px solid #dea041;padding:12px;margin-top:10px;font-size:14px}.original{background:#eff6ff;padding:10px;margin-top:8px;border-radius:8px;font-size:13px}.formrow{display:grid;grid-template-columns:repeat(2,minmax(240px,1fr));gap:13px;margin-top:16px}.field{display:flex;flex-direction:column;gap:5px;font-weight:650;font-size:13px}select,textarea{border:1px solid #b8c7d7;border-radius:8px;padding:10px;font:inherit;background:white;width:100%;box-sizing:border-box;font-weight:400}textarea{resize:vertical;min-height:70px}.full{grid-column:1/-1}.guide{font-size:14px;background:#edf4fc;padding:14px 20px;border-radius:10px}.guide strong{color:#1c4a7b}.footer{text-align:center;padding:25px;color:#738196;font-size:12px}@media(max-width:980px){.content{grid-template-columns:1fr}.formrow{grid-template-columns:1fr}.modelgrid{grid-template-columns:1fr}}</style></head><body>
<div class="headline"><h1>PRD 3.2 · V2 / V3 错误根因复核</h1><p>2026-10-29 | 18 条错误相关案例 | 仅训练来源开发集诊断，不是独立验收</p>
<div class="stats" id="stats"></div>
<div class="guide"><strong>判定先后顺序：</strong>先看红框所指真实目标，再分别看 V2/V3 的颜色、图案、设计究竟描述目标还是框内另一件服饰。<br>
<strong>实例绑定错位：</strong>有可见证据表明输出描述的主要是另一件衣物；<strong>类别标签混淆：</strong>颜色/结构描述目标本身，但类别归属错误；<strong>混合：</strong>两种证据同时存在；<strong>证据不足：</strong>不能确定模型究竟在描述哪件。<br>
<span style="color:#8c4b10">黄色“初步线索”仅辅助定位，不能作为最终根因。类别预测错误不等于绑定错误；红框混入其他衣服也不等于定位错误。</span></div>
<div class="toolbar"><button class="btn" onclick="exportJSON()">导出完整审核 JSON</button><button class="btn secondary" onclick="exportCSV()">导出 CSV</button><button class="btn secondary" onclick="expandAll()">展开全部案例</button><span class="muted">本页可离线使用；关闭前请导出结果。</span></div></div>
<div id="cards"></div><div class="footer">Sources: V2/V3 raw predictions · HumanEval40 · ROI15 revised review · historical19 · additional4. User entries are separate; original evidence is never overwritten.</div>
<script>const rows=__DATA__;const C={TARGET:'描述红框目标',OTHER:'描述另一件服饰',MIXED:'混合描述多件服饰',UNDETERMINED:'无法判定',UNREVIEWED:'未审核'};
const cause={UNREVIEWED:'未审核',NOT_ERROR:'该版本类别正确',BINDING:'实例绑定错误',CLASS_BOUNDARY:'类别标签混淆',BOTH:'绑定与类别混合',INSUFFICIENT:'证据不足'};
function e(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function opts(obj,val){return Object.entries(obj).map(([k,v])=>`<option value="${e(k)}" ${k===val?'selected':''}>${e(v)}</option>`).join('');}
function card(r){const id=r.case_id;const v2Wrong=r.v2_category!==r.human_category,v3Wrong=r.v3_category!==r.human_category;
 return `<section class="case" data-case="${id}"><div class="casehead"><div><b>${e(id)}</b><span class="tag">人工目标 ${e(r.human_category)}</span><span class="tag">${e(r.pair_group)}</span></div><span class="muted">原图 SHA256: ${e(r.image_sha256.slice(0,12))}…</span></div>
<div class="content"><div><div class="pic">${r.picture?`<img src="${r.picture}" alt="${e(id)} red-box original image"/>`:`<div class="missing">本机找不到红框原图。请在 AutoDL 上运行生成脚本，查看图片完整性。<br><small>${e(r.marked_image_autodl)}</small></div>`}</div><p class="muted">红框覆盖：${e(r.bbox_coverage)}；其他服饰：${e(r.garment_overlap_human)}；类别可辨：${e(r.visually_identifiable)}</p></div>
<div><div class="modelgrid"><div class="model"><strong>V2: <span class="${v2Wrong?'err':'ok'}">${e(r.v2_category)}</span></strong><div class="attrs">颜色：${e(r.v2_color||'null')}<br>图案：${e(r.v2_pattern||'null')}<br>可见设计：${e(r.v2_design||'null')}</div></div>
<div class="model"><strong>V3: <span class="${v3Wrong?'err':'ok'}">${e(r.v3_category)}</span></strong><div class="attrs">颜色：${e(r.v3_color||'null')}<br>图案：${e(r.v3_pattern||'null')}<br>可见设计：${e(r.v3_design||'null')}</div></div></div>
<div class="evidence"><b>初步线索（尚未确证）：</b> ${e(r.evidence_to_verify)}<br><b>历史人工初判：</b> ${e(r.historical_cause||'未记录')}</div>
<div class="original"><b>原人工备注：</b> ${e(r.manual_notes||'无')}</div></div></div>
<div class="formrow"><label class="field">V2 描述的衣物对象<select data-f="v2_focus">${opts(C,r.v2_focus)}</select></label><label class="field">V3 描述的衣物对象<select data-f="v3_focus">${opts(C,r.v3_focus)}</select></label>
<label class="field">V2 最终错误归因<select data-f="v2_root_cause">${opts(cause,r.v2_root_cause)}</select></label><label class="field">V3 最终错误归因<select data-f="v3_root_cause">${opts(cause,r.v3_root_cause)}</select></label>
<label class="field">证据可靠程度<select data-f="evidence_level">${opts({PENDING:'未审核',HIGH:'高：颜色+结构等可交叉核实',MEDIUM:'中：部分视觉线索支持',LOW:'低：难以区分',INSUFFICIENT:'证据不足'},r.evidence_level)}</select></label>
<label class="field">审核状态<select data-f="review_status">${opts({PENDING:'待审核',REVIEWED:'已审核',NEEDS_FOLLOWUP:'需要复核'},r.review_status)}</select></label>
<label class="field full">人工证据说明（写清“哪种颜色/结构对应哪件衣服”）<textarea data-f="reviewer_note" placeholder="例如：V3 的红色+裙摆指向红框内非目标红裙，目标本身为白色印花 T 恤。"></textarea></label></div></section>`;}
const host=document.getElementById('cards');host.innerHTML=rows.map(card).join('');
function gather(){return rows.map(r=>{const o={...r};delete o.picture;const card=document.querySelector(`[data-case="${r.case_id}"]`);card.querySelectorAll('[data-f]').forEach(el=>o[el.dataset.f]=el.value);return o;});}
function update(){let all=gather(),done=all.filter(r=>r.review_status==='REVIEWED').length,remaining=all.filter(r=>r.review_status==='PENDING').length;document.getElementById('stats').innerHTML=`<div class="stat">18 条待归因样本</div><div class="stat">已审核 ${done}/18</div><div class="stat">未审核 ${remaining}</div><div class="stat">视觉歧义或框问题另排除 5 条</div>`;}
function download(name,content,type){const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function exportJSON(){const data={schema:'prd32_rootcause_human_review_v1',date:'2026-10-29',scope:'TRAIN_DEV_DIAGNOSTIC_ONLY_NOT_INDEPENDENT',rows:gather()};download('PRD32_RootCause18_HumanReview_20261029.json',JSON.stringify(data,null,2),'application/json');}
function exportCSV(){const data=gather().map(x=>({case_id:x.case_id,human_category:x.human_category,v2_category:x.v2_category,v3_category:x.v3_category,v2_focus:x.v2_focus,v3_focus:x.v3_focus,v2_root_cause:x.v2_root_cause,v3_root_cause:x.v3_root_cause,evidence_level:x.evidence_level,review_status:x.review_status,reviewer_note:x.reviewer_note}));const keys=Object.keys(data[0]);let s=keys.join(',')+'\n'+data.map(row=>keys.map(k=>'"'+String(row[k]??'').replaceAll('"','""')+'"').join(',')).join('\n');download('PRD32_RootCause18_HumanReview_20261029.csv','\ufeff'+s,'text/csv;charset=utf-8');}
function expandAll(){document.querySelectorAll('.case').forEach(x=>x.style.display='block');window.scrollTo(0,0);}
host.addEventListener('change',update);host.addEventListener('input',update);update();</script></body></html>'''
    page=page.replace('__DATA__',embedded_data)
    target=pathlib.Path(args.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(page,encoding='utf-8')
    print(json.dumps({'status':'HTML_WRITTEN','cases':len(data['rows']),'embedded_images':n_images,'missing_images':len(missing),'output':str(target),'missing':missing[:5]},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
