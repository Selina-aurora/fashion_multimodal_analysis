import csv, json, base64, io, html, os, zipfile
from pathlib import Path
from PIL import Image, ImageOps
src=Path('/mnt/data')
base_rows=list(csv.DictReader((src/'review_sheet (1).csv').open(encoding='utf-8-sig',newline='')))
pre_rows=list(csv.DictReader((src/'PRD32_W7_Dev7_one_row_per_image_20261026.csv').open(encoding='utf-8-sig',newline='')))
preds={}
for row in base_rows:
    preds.setdefault(row['image_id'],[]).append(row)
pre={r['image_id']:r for r in pre_rows}
ids=['dev_02','dev_04','dev_05','dev_06','dev_08','dev_09','dev_12']
assert set(preds)==set(ids) and len(base_rows)==14 and set(pre)==set(ids)

# Candidate value suggestions, never pre-saved as confirmed GT.
suggestions={
'dev_02': {'category':'outerwear','color':'','pattern':'纯色','design':'长袖；前襟深色扣饰'},
'dev_04': {'category':'','color':'','pattern':'格纹','design':'翻领；长袖；前襟纽扣'},
'dev_05': {'category':'top','color':'','pattern':'条纹','design':'长袖；翻折袖口；前襟纽扣'},
'dev_06': {'category':'top','color':'白色','pattern':'纯色','design':'圆领；短袖'},
'dev_08': {'category':'top','color':'浅粉色','pattern':'纯色','design':'无袖；宽肩带；圆弧领口；横向分层'},
'dev_09': {'category':'outerwear','color':'','pattern':'人字纹','design':'翻领；前襟搭扣；翻折袖口'},
'dev_12': {'category':'outerwear','color':'黑色','pattern':'纯色','design':'西装翻领；长袖；敞开前襟'}
}

def encode_img(im):
    im=im.convert('RGB')
    im.thumbnail((720,860))
    buf=io.BytesIO(); im.save(buf,format='JPEG',quality=82,optimize=True)
    return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')

cases=[]
for id in ids:
    image=Image.open(src/f'{id}_focus.jpg')
    left=image.crop((8,48,495,655))
    right=image.crop((605,48,1100,655))
    rows=preds[id]
    assert len(rows)==2 and {r['input_variant'] for r in rows}=={'marked_full_image','reference_roi'}
    cases.append({
        'id':id,'image_left':encode_img(left),'image_right':encode_img(right),
        'original_prediction_rows':rows,'visual_precheck':pre[id],
        'suggested_draft':suggestions[id],
    })

# prevent literal closing script sequence from ending the JSON block
payload=json.dumps({'version':'PRD32_W7_DEV7_REVIEW_HTML_V1','generated':'2026-10-26','cases':cases,'original_fields':list(base_rows[0])},ensure_ascii=False).replace('</','<\\/')
page = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PRD 3.2 · W7 Dev7 人工审核工作台</title>
<style>
:root{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif;color:#192637;background:#f4f7fb;--blue:#2755b1;--line:#e3e9f1;--muted:#62758a;--shadow:0 3px 18px rgba(23,38,65,.04)}
*{box-sizing:border-box}body{margin:0}button,input,textarea,select{font:inherit}button{cursor:pointer}button:disabled{opacity:.55;cursor:not-allowed}.wrap{max-width:1600px;margin:0 auto;padding:24px 26px 44px}
.head{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;margin-bottom:20px}.eyebrow{font-size:12px;font-weight:800;color:var(--blue);letter-spacing:.12em;text-transform:uppercase}.head h1{font-size:28px;line-height:1.25;margin:7px 0 6px;letter-spacing:-.02em}.subtitle{color:var(--muted);font-size:13px;line-height:1.7;max-width:760px}.head-actions{display:flex;flex-wrap:wrap;gap:7px;justify-content:flex-end}
.btn{padding:10px 13px;border:1px solid #cdd8e8;border-radius:9px;background:white;color:#24364d;font-size:13px;font-weight:700;white-space:nowrap;transition:background .13s}.btn:hover{background:#eff5ff}.btn.primary{background:var(--blue);border-color:var(--blue);color:white}.btn.primary:hover{background:#194391}.btn.danger{color:#9d2a2a}.btn.small{padding:6px 9px;font-size:12px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:17px}.stat{border:1px solid var(--line);background:white;border-radius:12px;padding:14px 16px;box-shadow:var(--shadow)}.stat .small{font-size:12px;color:var(--muted)}.stat strong{font-size:23px;display:block;margin-top:3px;font-variant-numeric:tabular-nums}.stat .detail{font-size:11px;color:#7b8b9a;margin-top:3px}.bar{height:5px;border-radius:99px;background:#eaf0f7;margin-top:10px;overflow:hidden}.bar>div{height:100%;width:0;background:#338768;transition:width .2s}
.layout{display:grid;grid-template-columns:220px minmax(0,1fr);align-items:start;gap:16px}.nav,.main,.box{border:1px solid var(--line);border-radius:13px;background:#fff;box-shadow:var(--shadow)}.nav{padding:13px;position:sticky;top:12px}.nav h2{font-size:13px;margin:3px 7px 12px}.navtabs{display:flex;gap:5px;margin:0 4px 12px}.navtabs button{border:0;background:#edf3f9;border-radius:7px;padding:7px 8px;flex:1;font-size:11px;color:#526578}.navtabs button.active{background:#dceaff;color:#194997;font-weight:800}.nav-list{display:grid;gap:6px}.nav-case{border:1px solid transparent;background:none;text-align:left;padding:10px 10px;border-radius:9px;width:100%;display:flex;justify-content:space-between;gap:6px;align-items:center}.nav-case:hover{background:#f4f7fc}.nav-case.active{background:#e9f1ff;border-color:#b6cef8}.nav-case .id{font-weight:800;font-size:13px}.nav-case .topic{font-size:11px;color:var(--muted);margin-top:3px}.dot{border-radius:99px;font-size:10px;padding:4px 6px;background:#eef1f5;color:#708094;white-space:nowrap}.dot.verified{background:#dff6e9;color:#21754b}.dot.partial{background:#fff1d8;color:#905d13}.dot.uncertain{background:#fce8ea;color:#a13740}.nav-note{font-size:11px;line-height:1.65;color:var(--muted);margin:15px 6px 1px;padding-top:12px;border-top:1px solid var(--line)}
.main{padding:20px 23px;min-width:0}.case-head{display:flex;justify-content:space-between;gap:14px;align-items:flex-start}.case-head h2{margin:4px 0 7px;font-size:22px}.case-head .hint{font-size:12px;color:var(--muted)}.case-btns{display:flex;gap:7px;flex-wrap:wrap}.flagline{display:flex;gap:7px;flex-wrap:wrap;margin:13px 0 16px}.flag{border-radius:99px;border:1px solid #dfebf5;background:#f5f8fc;color:#49657f;font-size:11px;padding:5px 9px}.flag.warn{background:#fff6e7;color:#916019;border-color:#ffe3a8}.flag.bad{background:#fff0f2;color:#a4444b;border-color:#fad7dc}
.sec-title{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:25px 0 11px}.sec-title h3{font-size:15px;margin:0}.sec-title span{font-size:11px;color:var(--muted)}.pics{display:grid;grid-template-columns:1fr 1fr;gap:12px}.picture{min-width:0;border:1px solid var(--line);background:#fafcfe;border-radius:11px;overflow:hidden}.picture h4{padding:11px 13px;margin:0;font-size:12px;background:#f6f9fd;display:flex;justify-content:space-between}.picture img{display:block;width:100%;height:300px;object-fit:contain;cursor:zoom-in;padding:8px}.picture .desc{color:var(--muted);font-size:11px;padding:0 11px 11px}
.tablewrap{overflow:auto;border:1px solid var(--line);border-radius:10px}table{width:100%;border-collapse:collapse;font-size:12px;white-space:normal}th,td{padding:11px;border-bottom:1px solid #e9edf3;text-align:left;vertical-align:top}thead{background:#f7f9fc;color:#51657c}tr:last-child td{border-bottom:0}th:first-child{width:116px}.muted{color:var(--muted)}.mono{font-family:ui-monospace,Consolas,monospace}.em{font-weight:700}.issue{color:#a23a43;font-weight:700}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:12px}.suggest{background:#f5f8fe;border:1px solid #dbe7f8;border-radius:10px;padding:14px;font-size:12px;line-height:1.7;color:#435775}.suggest .tag{font-size:11px;text-transform:uppercase;color:#2c58a4;font-weight:800;letter-spacing:.07em}.suggest p{margin:7px 0 0}.suggest strong{color:#273a52}
.formgrid{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-top:14px}.field{display:flex;flex-direction:column;gap:6px;min-width:0}.field label{font-size:12px;font-weight:800}.field .tip{font-size:11px;color:var(--muted);line-height:1.5}.field input,.field select,.field textarea{background:#fff;width:100%;border:1px solid #cdd8e5;border-radius:8px;padding:10px;color:#263b51;outline:0;font-size:13px;min-width:0}.field textarea{resize:vertical;min-height:66px}.field input:focus,.field select:focus,.field textarea:focus{border-color:#5e8ae0;box-shadow:0 0 0 3px #e5efff}.twopart{display:grid;grid-template-columns:1fr 132px;gap:7px}.field.wide{grid-column:1/-1}.form-actions{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;margin-top:19px;padding-top:16px;border-top:1px solid var(--line);gap:12px}.minor{font-size:11px;color:var(--muted);line-height:1.65}.footer{color:#748297;font-size:11px;margin:18px auto;line-height:1.6}.callout{border-left:3px solid #4e7dc7;background:#f7faff;padding:10px 12px;color:#415a7d;font-size:11px;line-height:1.65;margin-top:13px;border-radius:4px}
#toast{position:fixed;bottom:22px;left:50%;transform:translateX(-50%);padding:11px 17px;color:white;background:#24374e;border-radius:9px;font-size:13px;box-shadow:0 8px 26px #0002;opacity:0;pointer-events:none;transition:opacity .25s;z-index:30;max-width:90vw}#toast.show{opacity:1}.lightbox{position:fixed;inset:0;background:rgba(12,20,34,.91);display:none;align-items:center;justify-content:center;z-index:40;padding:30px}.lightbox.open{display:flex}.lightbox img{max-width:94vw;max-height:90vh;object-fit:contain}.lightbox button{position:absolute;right:22px;top:18px;color:white;font-size:28px;background:none;border:0}
@media(max-width:1100px){.layout{grid-template-columns:175px 1fr}.nav{padding:9px}.main{padding:16px}.pics{grid-template-columns:1fr 1fr}.picture img{height:245px}.twopart{grid-template-columns:1fr}}
@media(max-width:750px){.wrap{padding:14px}.head{display:block}.head-actions{justify-content:flex-start;margin-top:15px}.stats{grid-template-columns:1fr 1fr}.layout{grid-template-columns:1fr}.nav{position:static}.nav-list{display:grid;grid-template-columns:repeat(4,minmax(0,1fr))}.nav-case{display:block;text-align:center;padding:7px 3px}.nav-case .topic{display:none}.nav-case .dot{margin:4px auto 0;display:inline-block}.pics,.two-col,.formgrid{grid-template-columns:1fr}.field.wide{grid-column:auto}.case-head{display:block}.case-btns{margin-top:10px}.picture img{height:auto;max-height:480px}.head h1{font-size:24px}}
</style>
</head>
<body>
<div class="wrap">
<header class="head"><div><div class="eyebrow">PRD 3.2 · DEVELOPMENT QUALITY REVIEW</div><h1>W7 Dev7 人工审核工作台</h1><div class="subtitle">7 张开发图片 · 14 条 Qwen2.5-VL-7B 预测 · 离线使用 · 自动本机暂存 · 支持导出 CSV / JSON。<strong>目视预审建议不等于经你核验的真实标签（GT）。</strong></div></div><div class="head-actions"><button class="btn" id="export7">导出 7 图 GT CSV</button><button class="btn" id="export14">导出 14 行完整 CSV</button><button class="btn primary" id="exportJSON">备份审核 JSON</button><button class="btn" id="importBtn">导入备份</button><input type="file" id="importFile" accept=".json,application/json" hidden></div></header>
<div class="stats"><div class="stat"><div class="small">图片总数</div><strong>7</strong><div class="detail">每张图片两种输入</div></div><div class="stat"><div class="small">已完成审核</div><strong id="countVerified">0</strong><div class="detail">须由审核者确认</div></div><div class="stat"><div class="small">可计分图片</div><strong id="countEligible">0</strong><div class="detail">全部字段及参考框已确认</div></div><div class="stat"><div class="small">审核进度</div><strong id="countProgress">0 / 7</strong><div class="bar"><div id="progressBar"></div></div></div></div>
<div class="layout"><aside class="nav"><h2>图片列表</h2><div class="navtabs"><button data-filter="all" class="active">全部</button><button data-filter="pending">待审</button><button data-filter="issues">重点</button></div><div class="nav-list" id="caseNav"></div><div class="nav-note">先选左侧图片，再核对模型预测和图片。保存仅在本浏览器本地进行；建议每次审核后导出 JSON 备份。图片和预测已包含在本 HTML 内，断网也可以审核。</div></aside>
<main class="main"><div class="case-head"><div><div class="eyebrow" id="casePosition">01 / 07</div><h2 id="caseTitle">dev_02</h2><div class="hint" id="caseSubtitle">按图片身份归档，整图 / ROI 共用一份人工标签</div></div><div class="case-btns"><button class="btn small" id="prevCase">← 上一张</button><button class="btn small" id="nextCase">下一张 →</button></div></div><div class="flagline" id="caseFlags"></div>
<div class="sec-title"><h3>原图标框与参考 ROI</h3><span>点击图片放大 · 红框为历史人工审核候选框，不是模型分割 GT</span></div>
<div class="pics"><div class="picture"><h4>整图标框 <span class="muted">marked_full_image</span></h4><img id="imgLeft" alt="整图标记候选框"><div class="desc">结合叠穿、包、背景检查目标绑定</div></div><div class="picture"><h4>参考区域裁剪 <span class="muted">reference_roi</span></h4><img id="imgRight" alt="参考 ROI"><div class="desc">检查目标是否完整，以及是否裁入其他服饰</div></div></div>
<div class="sec-title"><h3>两种输入的原始模型预测</h3><span>只读 · 不得直接当作 GT</span></div><div class="tablewrap"><table><thead><tr><th>字段</th><th>红框整图</th><th>参考 ROI</th></tr></thead><tbody id="compare"></tbody></table></div>
<div class="sec-title"><h3>辅助目视预审建议</h3><button class="btn small" id="applySuggestions">填入候选草稿（不自动确认）</button></div><div class="suggest" id="visualSummary"></div>
<div class="sec-title"><h3>你的人工审核结果</h3><span>只有你确认后才可标注 VERIFIED</span></div>
<div class="formgrid">
<div class="field"><label for="category">目标类别</label><div class="twopart"><select id="category"><option value="">请选择类别（尚未确认）</option><option value="top">top · 上衣</option><option value="pants">pants · 裤子</option><option value="skirt">skirt · 半身裙</option><option value="outerwear">outerwear · 外套</option><option value="dress">dress · 连衣裙</option><option value="shoe">shoe · 鞋</option><option value="bag">bag · 包</option><option value="accessory">accessory · 配饰</option></select><select id="categoryStatus" aria-label="类别确认状态"><option value="PENDING">未审核</option><option value="CONFIRMED">已确认</option><option value="UNCERTAIN">不确定</option></select></div><div class="tip">类别必须使用项目的八类定义；dev_04 尤其要复核。</div></div>
<div class="field"><label for="color">主颜色</label><div class="twopart"><input id="color" placeholder="例如：白色；无法判断则留空"><select id="colorStatus" aria-label="颜色确认状态"><option value="PENDING">未审核</option><option value="CONFIRMED">已确认</option><option value="UNCERTAIN">不确定</option></select></div><div class="tip">先统一颜色词表；尽量使用单一主颜色标签。</div></div>
<div class="field"><label for="pattern">图案 / 纹理</label><div class="twopart"><input id="pattern" list="patternHints" placeholder="纯色 / 条纹 / 格纹 / 人字纹…"><select id="patternStatus" aria-label="图案确认状态"><option value="PENDING">未审核</option><option value="CONFIRMED">已确认</option><option value="UNCERTAIN">不确定</option></select></div><div class="tip">可见织纹 ≠ 具体面料成分；装饰扣不能直接当作图案。</div></div>
<div class="field"><label for="design">可见设计特征</label><div class="twopart"><input id="design" placeholder="以中文分号隔开，如：翻领；长袖"><select id="designStatus" aria-label="设计确认状态"><option value="PENDING">未审核</option><option value="CONFIRMED">已确认</option><option value="UNCERTAIN">不确定</option></select></div><div class="tip">只填写能看到的结构/装饰，不接受“无”“无明显设计”。</div></div>
<div class="field"><label for="bbox">参考框是否可用于目标属性审核？</label><select id="bbox"><option value="PENDING">未复核</option><option value="TARGET_CORRECT">正确覆盖目标，适合 ROI 对照</option><option value="PARTIAL">部分覆盖 / 混有干扰，不适合直接计分</option><option value="INCORRECT">目标错误 / 严重偏移</option><option value="UNCERTAIN">无法确认</option></select><div class="tip">这只确认候选框可用性，不等于分割 IoU 验收。</div></div>
<div class="field"><label for="reviewStatus">整张图片审核状态</label><select id="reviewStatus"><option value="PENDING">PENDING · 未完成</option><option value="PARTIAL">PARTIAL · 部分已审核</option><option value="UNCERTAIN">UNCERTAIN · 存在未裁决问题</option><option value="VERIFIED">VERIFIED · 全部字段已确认</option></select><div class="tip">审核状态与单字段状态独立，VERIFIED 会进行完整性检查。</div></div>
<div class="field wide"><label for="notes">备注 / 错误归因</label><textarea id="notes" rows="3" placeholder="记录颜色歧义、类别定义、叠穿、图案与装饰区分、参考框风险等"></textarea><div class="tip">例如：outerwear/top 冲突，需要按八类规范裁定；模型设计字段遗漏无袖。</div></div>
</div><div class="callout" id="eligibility">GT 尚未全部核验。本页不计算 PRD 准确率。</div>
<div class="form-actions"><div class="minor" id="saveStatus">本地草稿将自动保存；本机浏览器缓存可能被清理，请导出 JSON 备份。</div><div class="case-btns"><button class="btn" id="clearOne">清空本图人工填写</button><button class="btn primary" id="saveNext">保存并下一张 →</button></div></div>
</main></div>
<div class="footer">PRD 3.2 · W7 Dev7 辅助人工审核（开发集）｜不访问远程网络｜不使用封存 holdout｜历史参考框并非真实分割 GT｜模型结果保留原值，人工审核另列。归档前请导出 CSV / JSON。</div></div>
<datalist id="patternHints"><option value="纯色"><option value="条纹"><option value="格纹"><option value="人字纹"><option value="波点"><option value="花卉印花"><option value="字母印花"><option value="几何纹"></datalist>
<div id="toast" role="status" aria-live="polite"></div><div class="lightbox" id="lightbox"><button aria-label="关闭预览" id="closeLightbox">×</button><img alt="放大预览" id="lightboxImg"></div>
<script id="embedded-data" type="application/json">__EMBEDDED_JSON__</script>
<script>
(()=>{'use strict';
const src=JSON.parse(document.getElementById('embedded-data').textContent);
const ids=src.cases.map(c=>c.id), lookup=Object.fromEntries(src.cases.map(c=>[c.id,c]));
const fieldNames=['category','color','pattern','design'];
const key='PRD32_DEV7_HUMAN_REVIEW_V1_20261026';
const mk=()=>({category:'',color:'',pattern:'',design:'',categoryStatus:'PENDING',colorStatus:'PENDING',patternStatus:'PENDING',designStatus:'PENDING',bbox:'PENDING',reviewStatus:'PENDING',notes:'',updatedAt:''});
let edits=Object.fromEntries(ids.map(id=>[id,mk()])); let index=0,filter='all';
const el=id=>document.getElementById(id), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const reviewCols=['category','color','pattern','design','categoryStatus','colorStatus','patternStatus','designStatus','bbox','reviewStatus','notes'];
const store=()=>{try{localStorage.setItem(key,JSON.stringify({version:1,edits}));el('saveStatus').textContent='自动保存于本机浏览器 · '+new Date().toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'});}catch(e){el('saveStatus').textContent='浏览器本地保存不可用！请立刻导出 JSON 备份。';}};
try{const old=JSON.parse(localStorage.getItem(key)||'null');if(old?.version===1)for(const id of ids)if(old.edits?.[id]){Object.assign(edits[id],Object.fromEntries(reviewCols.filter(k=>typeof old.edits[id][k]==='string').map(k=>[k,old.edits[id][k]])));edits[id].updatedAt=old.edits[id].updatedAt||'';}}catch(e){}
function toast(s){let t=el('toast');t.textContent=s;t.classList.add('show');clearTimeout(toast.handle);toast.handle=setTimeout(()=>t.classList.remove('show'),2500)}
function validGT(v){return v.reviewStatus==='VERIFIED'&&v.bbox==='TARGET_CORRECT'&&fieldNames.every(f=>v[f+'Status']==='CONFIRMED'&&!!String(v[f]||'').trim())&&!/^无$|^无明显设计$/.test(v.design.trim());}
function boxText(s){return s?esc(s):'<span class="muted">null / 空值</span>'}
function nav(){const n=el('caseNav');n.replaceChildren();for(const [i,id] of ids.entries()){const v=edits[id], pr=lookup[id].visual_precheck.visual_precheck_priority||'';if(filter==='pending'&&v.reviewStatus==='VERIFIED')continue;if(filter==='issues'&&!/HIGH/.test(pr))continue;const b=document.createElement('button');b.className='nav-case'+(i===index?' active':'');b.innerHTML=`<div><div class="id">${esc(id)}</div><div class="topic">${esc(lookup[id].original_prediction_rows[0].reference_category_not_gt)} · ${validGT(v)?'GT已确认':'开发审核'}</div></div><span class="dot ${v.reviewStatus==='VERIFIED'?'verified':v.reviewStatus==='PENDING'?'':v.reviewStatus==='UNCERTAIN'?'uncertain':'partial'}">${esc(v.reviewStatus)}</span>`;b.addEventListener('click',()=>{saveForm();index=i;render()});n.appendChild(b)}}
function stats(){const n=ids.filter(id=>edits[id].reviewStatus!=='PENDING').length;const verified=ids.filter(id=>edits[id].reviewStatus==='VERIFIED').length;const eligible=ids.filter(id=>validGT(edits[id])).length;el('countVerified').textContent=verified;el('countEligible').textContent=eligible;el('countProgress').textContent=n+' / 7';el('progressBar').style.width=(n/7*100)+'%'}
function render(){const c=lookup[ids[index]],p=c.visual_precheck,v=edits[c.id];el('casePosition').textContent=String(index+1).padStart(2,'0')+' / 07';el('caseTitle').textContent=c.id+' · 目标属性审核';el('caseSubtitle').textContent='历史类别提示：'+c.original_prediction_rows[0].reference_category_not_gt+'（非正式 GT）';el('imgLeft').src=c.image_left;el('imgRight').src=c.image_right;
let conflict=[];const [a,b]=c.original_prediction_rows;for(const k of ['pred_category','pred_primary_color','pred_pattern','pred_visible_design'])if(a[k]!==b[k])conflict.push(k);
el('caseFlags').innerHTML='<span class="flag">历史框候选 · 非 GT</span>'+(conflict.length?'<span class="flag warn">双视图预测有 '+conflict.length+' 个差异字段</span>':'<span class="flag">双视图预测完全一致（不代表正确）</span>')+(/HIGH/.test(p.visual_precheck_priority)?'<span class="flag bad">重点预审案例</span>':'')+'<span class="flag">正式属性 GT 尚未确认</span>';
const details=[['类别','pred_category'],['主颜色','pred_primary_color'],['图案','pred_pattern'],['可见设计','pred_visible_design'],['格式','native_json'],['问题','issues'],['端到端耗时','latency_ms']];el('compare').innerHTML=details.map(([title,key])=>'<tr><th>'+esc(title)+'</th>'+[a,b].map(r=>'<td class="'+(key==='issues'&&r[key]?'issue':'')+'">'+(key==='latency_ms'?esc(r[key])+' ms':boxText(r[key]))+'</td>').join('')+'</tr>').join('');
el('visualSummary').innerHTML='<div class="tag">ASSISTED VISUAL PRECHECK · NOT GROUND TRUTH</div>'+
'<p><strong>类别建议：</strong>'+esc(p.visual_precheck_category)+'　　<strong>主颜色建议：</strong>'+esc(p.visual_precheck_primary_color)+'</p>'+
'<p><strong>图案：</strong>'+esc(p.visual_precheck_pattern)+'　　<strong>设计特征：</strong>'+esc(p.visual_precheck_visible_design)+'</p>'+
'<p><strong>参考框初判：</strong>'+esc(p.visual_precheck_bbox)+'</p>'+
'<p><strong>需重点检查：</strong>'+esc(p.visual_precheck_notes)+'</p>';
for(const k of reviewCols)el(k).value=v[k]??'';
el('eligibility').textContent=validGT(v)?'本图片已具备“开发集内部计算属性指标”的基本人工标签条件（不代表 PRD 验收；仍需独立确认标签质量）。':'尚不可计分：请独立核对目标框与 4 项属性，并将各字段状态设为“已确认”、整图审核设为 VERIFIED。未核验项保持不确定/待审。';
nav();stats();el('prevCase').disabled=index===0;el('nextCase').disabled=index===ids.length-1;}
function saveForm(){const v=edits[ids[index]];for(const k of reviewCols)v[k]=el(k).value;v.updatedAt=new Date().toISOString();store();nav();stats();el('eligibility').textContent=validGT(v)?'本图片已有完整的开发 GT 候选，可以用于后续逐字段分析；不代表独立验证或 PRD 正式验收。':'尚不可计分：须核实全部属性与目标框。'}
for(const k of reviewCols){el(k).addEventListener(k==='notes'||['category','color','pattern','design'].includes(k)?'input':'change',()=>saveForm());if(k==='notes'||['category','color','pattern','design'].includes(k))el(k).addEventListener('change',()=>saveForm())}
el('applySuggestions').onclick=()=>{const x=lookup[ids[index]].suggested_draft;for(const f of fieldNames)if(x[f]){el(f).value=x[f];el(f+'Status').value='PENDING'}saveForm();toast('仅填入候选草稿；各字段仍为“未审核”，需要你亲自确认。')};
el('reviewStatus').addEventListener('change',()=>{if(el('reviewStatus').value==='VERIFIED'){const v=edits[ids[index]];if(!validGT(v)){el('reviewStatus').value='PARTIAL';saveForm();toast('尚缺完整已核验字段或正确参考框，已保持 PARTIAL。')}else toast('已标记 VERIFIED，请及时导出备份')}});
el('prevCase').onclick=()=>{saveForm();index=Math.max(0,index-1);render()};el('nextCase').onclick=()=>{saveForm();index=Math.min(ids.length-1,index+1);render()};el('saveNext').onclick=()=>{saveForm();if(index<ids.length-1){index++;render();toast('已自动保存，进入下一张')}else toast('已到最后一张，请导出 CSV / JSON 备份')};
el('clearOne').onclick=()=>{if(confirm('仅清空 '+ids[index]+' 的人工填写？不会修改原始模型预测。')){edits[ids[index]]=mk();store();render();toast('本图人工草稿已清空')}};
document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));nav()});
const q=s=>'"'+String(s??'').replace(/"/g,'""')+'"';
function csv(rows,cols){return '\ufeff'+cols.join(',')+'\r\n'+rows.map(r=>cols.map(c=>q(r[c])).join(',')).join('\r\n')+'\r\n'}
function exportBlob(filename,mime,content){const b=new Blob([content],{type:mime});const u=URL.createObjectURL(b);const a=document.createElement('a');a.href=u;a.download=filename;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),2500);}
function gtRow(id){const v=edits[id],c=lookup[id];return {image_id:id,source:'PRD32_W7_DEV7_HTML_REVIEW_V1',reference_category_not_gt:c.original_prediction_rows[0].reference_category_not_gt,human_review_status:v.reviewStatus,human_category_gt:v.categoryStatus==='CONFIRMED'?v.category:'',human_color_gt:v.colorStatus==='CONFIRMED'?v.color:'',human_pattern_gt:v.patternStatus==='CONFIRMED'?v.pattern:'',human_design_gt:v.designStatus==='CONFIRMED'?v.design:'',category_status:v.categoryStatus,color_status:v.colorStatus,pattern_status:v.patternStatus,design_status:v.designStatus,bbox_review_status:v.bbox,gt_scoring_eligible:validGT(v)?'TRUE':'FALSE',review_notes:v.notes,review_updated_at:v.updatedAt}}
const gtcols=['image_id','source','reference_category_not_gt','human_review_status','human_category_gt','human_color_gt','human_pattern_gt','human_design_gt','category_status','color_status','pattern_status','design_status','bbox_review_status','gt_scoring_eligible','review_notes','review_updated_at'];
el('export7').onclick=()=>{saveForm();exportBlob('PRD32_Dev7_human_GT_7images.csv','text/csv;charset=utf-8',csv(ids.map(gtRow),gtcols));toast('已导出每图一行 GT 草稿 CSV')};
el('export14').onclick=()=>{saveForm();const rows=ids.flatMap(id=>lookup[id].original_prediction_rows.map(r=>({...r,...gtRow(id)})));const cols=[...new Set([...src.original_fields,...gtcols])];exportBlob('PRD32_Dev7_prediction_GT_14rows.csv','text/csv;charset=utf-8',csv(rows,cols));toast('已导出 14 行模型预测 + 同图人工标签 CSV')};
el('exportJSON').onclick=()=>{saveForm();const data={schema:'prd32_w7_dev7_human_review_ui_v1',exported_at:new Date().toISOString(),input_sample_count:7,inference_predictions:14,review_source:'user_manually_entered',no_sealed_holdout:true,gt_scoring_rule:'VERIFIED, target correct, 4 fields confirmed',reviews:ids.map(id=>({...gtRow(id),draft_inputs:edits[id]}))};exportBlob('PRD32_Dev7_human_review_backup.json','application/json;charset=utf-8',JSON.stringify(data,null,2));toast('已导出审核 JSON 备份')};
el('importBtn').onclick=()=>el('importFile').click();el('importFile').onchange=async event=>{const f=event.target.files?.[0];if(!f)return;try{const data=JSON.parse(await f.text());if(data.schema!=='prd32_w7_dev7_human_review_ui_v1'||!Array.isArray(data.reviews)||data.reviews.length!==7||new Set(data.reviews.map(r=>r.image_id)).size!==7||data.reviews.some(r=>!ids.includes(r.image_id)||!r.draft_inputs))throw Error('不是本页面导出的七图审核备份');if(!confirm('导入备份将覆盖当前本机的七图人工填写。继续？'))return;for(const r of data.reviews){const v=mk();for(const k of reviewCols)if(typeof r.draft_inputs[k]==='string')v[k]=r.draft_inputs[k];v.updatedAt=r.draft_inputs.updatedAt||'';edits[r.image_id]=v}store();render();toast('已恢复 7 图审核备份')}catch(e){alert('导入失败：'+e.message)}event.target.value=''};
for(const [name,img] of [['imgLeft','imgLeft'],['imgRight','imgRight']])el(name).onclick=()=>{el('lightboxImg').src=el(img).src;el('lightbox').classList.add('open')};el('closeLightbox').onclick=()=>el('lightbox').classList.remove('open');el('lightbox').onclick=e=>{if(e.target.id==='lightbox')el('lightbox').classList.remove('open')};window.addEventListener('keydown',e=>{if(e.key==='Escape')el('lightbox').classList.remove('open')});
render();
})();
</script>
</body></html>'''
page=page.replace('__EMBEDDED_JSON__',payload)
filename=src/'PRD32_W7_Dev7_人工审核工作台_离线版.html'
filename.write_text(page,encoding='utf-8')
zip_path=src/'PRD32_W7_Dev7_HTML_离线审核包.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
 z.write(filename,arcname=filename.name)
 z.writestr('README_使用说明.txt','PRD3.2 七图人工审核页面\n1. 直接双击 HTML 用 Chrome/Edge 打开，无需网络和服务器。\n2. 每张图片查看红框整图和参考ROI，比较两组预测。\n3. 人工填写类别、颜色、图案、设计，逐字段设为已确认/不确定；审核参考框。\n4. 审核完成后务必导出 7 图 GT CSV、14 行完整 CSV 和 JSON 备份。\n5. 模型预测不等于GT，自动保存保存在当前浏览器本地，换设备前必须导出 JSON。\n6. 这是开发集审核，不代表PRD验收。\n')
print('HTML',filename,filename.stat().st_size,'ZIP',zip_path,zip_path.stat().st_size)
