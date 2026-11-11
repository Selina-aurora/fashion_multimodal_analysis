#!/usr/bin/env python3
"""Generate an image-embedded, fully offline root cause review form.

Run on AutoDL, where original marked images are stored. No model calls; no data leakage.
This makes no automatic claim that the model attended a specific object.
"""
import argparse,base64,csv,html,io,json,mimetypes,pathlib
from PIL import Image,ImageOps

ROOT=pathlib.Path(__file__).resolve().parents[1]

CSS='''
:root{font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;color:#1c2935;background:#f4f6f9}
*{box-sizing:border-box}body{max-width:1450px;margin:auto;padding:22px}header{margin-bottom:20px}
h1{margin:0;font-size:27px}h2{font-size:21px;margin:10px 0}p{line-height:1.65}
.notice{background:#fff6e8;border-left:4px solid #da9343;padding:12px 16px;border-radius:8px}
.metrics{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}.metric{padding:12px 20px;border:1px solid #dbe0e6;background:#fff;border-radius:12px;min-width:180px}
.metric strong{display:block;font-size:25px}.muted{color:#6a7786;font-size:13px}
.card{background:#fff;border:1px solid #dde3e8;border-radius:14px;margin:16px 0;padding:18px;box-shadow:0 1px 3px #0915260a}
.layout{display:grid;grid-template-columns:minmax(230px,380px) minmax(0,1fr);gap:22px}
@media(max-width:850px){.layout{grid-template-columns:1fr}}
.photo{max-height:550px;max-width:100%;width:auto;object-fit:contain;border:1px solid #dce2e7;border-radius:8px}
.empty{padding:50px 12px;background:#f6f6f6;border:1px dashed #b3bdc7;color:#667;text-align:center;line-height:1.6}
span.token{border:1px solid #cbd2da;border-radius:5px;padding:2px 6px;font-size:12px;background:#f4f6f8}
.guide{margin:10px 0;border-left:3px solid #a7b9c8;padding:6px 9px;background:#f8fafb}
.outputs{display:grid;grid-template-columns:1fr 1fr;gap:10px}@media(max-width:750px){.outputs{grid-template-columns:1fr}}
.output{background:#f6f8fa;padding:11px;border-radius:9px;line-height:1.6;font-size:14px}
select,textarea{display:block;width:100%;border-radius:7px;border:1px solid #becad5;background:#fff;padding:10px;font:inherit;margin:6px 0 10px}
textarea{height:80px;resize:vertical}.answer{border:1px solid #dbe2ea;padding:12px;border-radius:10px;margin-top:12px}
.answer.correct{background:#edf6f0;border-color:#c1dbca}.answer span.label{font-weight:700}
button{border:1px solid #bdc7d2;border-radius:8px;background:white;padding:9px 15px;cursor:pointer;font:inherit}
button.primary{background:#2b5e88;color:white;border:0}.actions{display:flex;gap:9px;flex-wrap:wrap;margin:14px 0;align-items:center;position:sticky;top:0;padding:12px;background:#f4f6f9eF;z-index:10}
small{color:#6a7786}label{font-weight:600;font-size:13px}
'''
JS='''
const CATS=[['UNREVIEWED','未审核'],['TARGET_BINDING','实例绑定错误：属性和类别指向另一件衣服'],['CLASS_CONFUSION','类别标签混淆：看对目标但类别判错'],['MIXED','混合原因：两类证据都有'],['INSUFFICIENT','证据不足：无法确定']];
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
const cases=JSON.parse(document.getElementById('source-json').textContent);
const drafts={};
function sel(cid,v,isCorrect){if(isCorrect)return '<div class="answer correct"><span class="label">'+v+'：模型类别正确，无需错误归因</span></div>';
const cur=drafts[cid]?.[v]||{}; const opts=CATS.map(([k,n])=>`<option value="${k}" ${cur.cause===k?'selected':''}>${n}</option>`).join('');
return `<div class="answer"><label>${v} 错误类型</label><select data-case="${cid}" data-version="${v}" data-field="cause">${opts}</select><label>${v} 判断证据：对应哪件衣服、颜色/领型/结构、为什么？</label><textarea data-case="${cid}" data-version="${v}" data-field="evidence" placeholder="至少写一个可见线索；看不出来请写证据不足">${esc(cur.evidence||'')}</textarea><label>${v} 证据置信度</label><select data-case="${cid}" data-version="${v}" data-field="confidence"><option value="" ${!cur.confidence?'selected':''}>请选择</option><option value="HIGH" ${cur.confidence==='HIGH'?'selected':''}>高：有可直接验证的颜色+结构证据</option><option value="MEDIUM" ${cur.confidence==='MEDIUM'?'selected':''}>中：有单项证据</option><option value="LOW" ${cur.confidence==='LOW'?'selected':''}>低：仍难确认</option></select></div>`;
}
function val(id,v,field,value){drafts[id]??={};drafts[id][v]??={};drafts[id][v][field]=value;updateSummary();}
function updateSummary(){let done=0,total=0;for(const x of cases){for(const v of ['V2','V3']){if(x.correct[v])continue;total++;const r=drafts[x.id]?.[v];if(r?.cause&&r.cause!=='UNREVIEWED')done++;}}
document.getElementById('done').textContent=`已审核 ${done}/${total} 次错误预测（18张图）`;}
function draw(){document.getElementById('cases').innerHTML=cases.map(x=>{
const src=x.image?`<img class="photo" src="${x.image}" alt="${esc(x.id)} red bbox image"/>`:'<div class="empty">当前环境未找到标注图片。<br/>请在 AutoDL 原图片仍存在的容器运行生成脚本，嵌入真实红框图。</div>';
const attr=(v)=>`<b>${esc(v.category)}</b> ｜ ${esc(v.primary_color||'无颜色')} ｜ ${esc(v.pattern||'无图案')} ｜ ${esc(v.visible_design||'无结构描述')}`;
return `<article class="card"><h2>${x.id}　<span class="token">人工：${esc(x.human)}</span> <span class="token">${esc(x.change)}</span></h2><div class="layout"><div>${src}<p class="muted">红框覆盖：${esc(x.bbox)}；框内其他衣服：${esc(x.other)}；ROI 仅目标：${esc(x.roi_single)}。位置正确不代表 ROI 没有混入其他衣物。</p></div><div><div class="outputs"><div class="output"><b>V2</b><br/>${attr(x.v2)}</div><div class="output"><b>V3</b><br/>${attr(x.v3)}</div></div><div class="guide"><b>原始人工备注：</b>${esc(x.human_notes||'未写备注')}<br/><b>ROI 复核：</b>${esc(x.roi_notes||'无记录')}<br/><b>已有归因（仅供参考）：</b>${esc(x.prior_cause||'无')}</div><div class="guide"><b>待验证的线索，不是最终结论：</b>${esc(x.triage)}<br/>${esc(x.reason)}</div>${sel(x.id,'V2',x.correct.V2)}${sel(x.id,'V3',x.correct.V3)}</div></div></article>`}).join('');
document.querySelectorAll('select[data-case],textarea[data-case]').forEach(el=>el.addEventListener('change',()=>val(el.dataset.case,el.dataset.version,el.dataset.field,el.value)));updateSummary();}
function exportReview(){const out={schema:'prd32_rootcause_human_adjudication_v1',scope:'train_source_dev_not_independent',cases:cases.map(x=>({case_id:x.id,human_category:x.human,version_results:Object.fromEntries(['V2','V3'].map(v=>{const p=drafts[x.id]?.[v]||{};return [v,{prediction:x[v.toLowerCase()].category,correct:x.correct[v],cause:x.correct[v]?'CORRECT':(p.cause||'UNREVIEWED'),confidence:x.correct[v]?'NA':(p.confidence||''),evidence:p.evidence||''}]}))}))};
const blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='PRD32_RootCause18_Adjudicated_20261029.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
document.getElementById('export').addEventListener('click',exportReview);
// never auto-submit to server or call any network APIs
window.addEventListener('beforeunload',e=>{const has=Object.keys(drafts).length>0;if(has){e.preventDefault();e.returnValue='';}});
draw();
'''

def img_data(s):
 if not s:return None
 p=pathlib.Path(s)
 if not p.is_file():return None
 try:
  with Image.open(p) as im:
   im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((900,900))
   b=io.BytesIO();im.save(b,format='JPEG',quality=76,optimize=True)
   return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode('ascii')
 except Exception as e:return None

def build(output,skip_images=False):
 records=json.load(open(ROOT/'RootCause_18Case_Detail.json',encoding='utf8'))
 arr=[];success=0
 for x in records:
  c=x['case'];pic=None if skip_images else img_data(c['marked_image_autodl'])
  if pic:success+=1
  arr.append(dict(id=c['case_id'],human=c['human_category'],change=c['change_type'],bbox=c['bbox_coverage'],other=c['other_garments_visible_full'],roi_single=c['roi_single_item'],image=pic,
    correct={'V2':c['v2_final_root_cause']=='CORRECT','V3':c['v3_final_root_cause']=='CORRECT'},
    v2=x['v2'],v3=x['v3'],human_notes=c['human_review_notes'],roi_notes=c['roi_review_notes'],
    prior_cause=c['historical_cause_UNVERIFIED'],triage=c['triage_hypothesis_NOT_FINAL'],reason=c['triage_reason_NOT_FINAL']))
 blob=json.dumps(arr,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
 header='''<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>PRD 3.2 Root Cause Review - 2026-10-29</title><style>'''+CSS+'''</style></head><body><header><h1>PRD 3.2 — 实例绑定 vs 类别混淆根因诊断</h1><p class="muted">2026-10-29 · 训练来源开发样本 · 未触碰封存集 · 不用于正式验收</p></header><div class="notice"><b>归因原则：</b>不要把“框内有其他服饰”直接判成实例绑定错误。只有模型输出的类别、颜色、图案或结构与另一件服饰相对应时，才有绑定证据。模型输出符合目标外观但类别错，才有类别混淆证据。无法确定则选“证据不足”。可先填部分并导出；本页不自动保存，导出前不要关闭。<br/><b>审查单位：</b>18 张图片中共有 25 次错误输出；V2 和 V3 分开标注。正确信息仅作参照。</div><div class="metrics"><div class="metric"><span class="muted">人工类别明确的图片</span><strong>35 / 40</strong></div><div class="metric"><span class="muted">至少一个版本出错</span><strong>18</strong></div><div class="metric"><span class="muted">待归因的错误输出</span><strong>25</strong></div><div class="metric"><span class="muted">图片实际嵌入</span><strong>'''+str(success)+''' / 18</strong></div></div><div class="actions"><button class="primary" id="export">导出当前人工归因 JSON</button><b id="done">已审核 0/25</b><small>支持部分完成后导出，不上传至任何服务器</small></div><div id="cases"></div><script type="application/json" id="source-json">'''+blob+'''</script><script>'''+JS+'''</script></body></html>'''
 pathlib.Path(output).write_text(header,encoding='utf8');print(json.dumps({'status':'HTML_WRITTEN','cases':len(arr),'embedded_images':success,'output':str(output)},ensure_ascii=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'RootCause18_Visual_Review.html'));p.add_argument('--no-images',action='store_true')
 a=p.parse_args();build(a.output,a.no_images)
