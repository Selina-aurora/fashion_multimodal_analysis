import csv,json,statistics,base64,html,datetime,zipfile,hashlib
from pathlib import Path
from collections import Counter
P=Path('/mnt/data'); gt_path=P/'PRD32_Dev7_human_GT_7images.csv'; preds_path=P/'PRD32_Dev7_prediction_GT_14rows.csv'
with gt_path.open(encoding='utf-8-sig',newline='') as f: gt=list(csv.DictReader(f))
with preds_path.open(encoding='utf-8-sig',newline='') as f: pred=list(csv.DictReader(f))
assert len(gt)==7 and len(pred)==14
G={g['image_id']:g for g in gt}; expected=['marked_full_image','reference_roi']; grouped={}
col_equiv={'black':'黑色','white':'白色','blue':'蓝色','红':'红色'}
pat_equiv={'格子':'格纹','无':'纯色'}
def norm_col(s):
 s=(s or '').strip();return col_equiv.get(s.lower(),s)
def norm_pat(s):
 s=(s or '').strip();return pat_equiv.get(s,s)
for r in pred:
 i=r['image_id'];g=G[i];v=r['input_variant']
 assert v in expected
 assert (i,v) not in grouped
 for k in ['human_category_gt','human_color_gt','human_pattern_gt','human_design_gt','human_review_status','gt_scoring_eligible']:
  assert r[k]==g[k],(i,k,r[k],g[k])
 assert r['human_review_status']=='VERIFIED' and r['gt_scoring_eligible'].upper()=='TRUE'
 assert all(g[c]=='CONFIRMED' for c in ['category_status','color_status','pattern_status','design_status'])
 assert r['bbox_review_status']=='TARGET_CORRECT'
 r['category_match']=r['pred_category']==g['human_category_gt']
 r['color_exact_match']=bool(norm_col(r['pred_primary_color'])) and norm_col(r['pred_primary_color'])==norm_col(g['human_color_gt'])
 r['pattern_exact_match']=bool(norm_pat(r['pred_pattern'])) and norm_pat(r['pred_pattern'])==norm_pat(g['human_pattern_gt'])
 r['design_nonstructural'] = 'non_structural_visible_design' in (r.get('issues') or '')
 r['latency_ms_float']=float(r['latency_ms'])
 r['pred_color_normalized']=norm_col(r['pred_primary_color'])
 r['gt_color_normalized']=norm_col(g['human_color_gt'])
 r['pred_pattern_normalized']=norm_pat(r['pred_pattern'])
 r['gt_pattern_normalized']=norm_pat(g['human_pattern_gt'])
 grouped[(i,v)]=r
for i in G:assert all((i,v) in grouped for v in expected)
metrics={}
for v in expected:
 rows=[grouped[(i,v)] for i in G]
 metrics[v]={
 'n_images':len(rows),
 'category_correct':sum(r['category_match'] for r in rows),
 'color_exact_correct':sum(r['color_exact_match'] for r in rows),
 'pattern_exact_correct':sum(r['pattern_exact_match'] for r in rows),
 'design_nonstructural_count':sum(r['design_nonstructural'] for r in rows),
 'latency_median_ms':round(statistics.median(r['latency_ms_float'] for r in rows),2),
 'latency_mean_ms':round(statistics.mean(r['latency_ms_float'] for r in rows),2),
 'native_json_count':sum(r['native_json'].lower()=='true' for r in rows),
 'parseable_json_count':sum(r['parseable'].lower()=='true' for r in rows),
 }
rows_audit=[]
for i in G:
 g=G[i]
 for v in expected:
  r=grouped[(i,v)]
  rows_audit.append(dict(image_id=i,variant=v,gt_category=g['human_category_gt'],pred_category=r['pred_category'],category_match=r['category_match'],gt_color=g['human_color_gt'],pred_color=r['pred_primary_color'],color_strict_match=r['color_exact_match'],gt_pattern=g['human_pattern_gt'],pred_pattern=r['pred_pattern'],pattern_strict_match=r['pattern_exact_match'],gt_design=g['human_design_gt'],pred_design=r['pred_visible_design'],nonstructural_design=r['design_nonstructural'],latency_ms=r['latency_ms'],native_json=r['native_json'],parseable=r['parseable'],review_status='VERIFIED_HUMAN_LABELS',acceptance_status='DEV_DIAGNOSTIC_ONLY'))
audit=P/'PRD32_Dev7_GT_scored_14rows.csv'
with audit.open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows_audit[0]));writer.writeheader();writer.writerows(rows_audit)
summary={
 'schema':'prd32_dev7_human_gt_scoring_v1',
 'status':'DEV_SCORE_COMPUTED_NOT_PRD_ACCEPTED',
 'source_gt':gt_path.name,'source_predictions':preds_path.name,
 'gt_records':7,'prediction_records':14,
 'gt_consistency_check':'PASS','gt_status':'VERIFIED_BY_USER_HTML','image_repeated_2x':True,
 'metrics_by_input_variant':metrics,
 'normalized_color_mappings':col_equiv,
 'normalized_pattern_mappings':pat_equiv,
 'color_matching':'strict after only explicit translations; differences in shade remain mismatches',
 'pattern_matching':'strict after 格子→格纹 and 无→纯色; blank prediction treated as unknown/miss',
 'design_accuracy':None,
 'design_accuracy_reason':'Multi-label design properties need an agreed ontology, human adjudication and token-by-token annotation; none reported as numeric accuracy.',
 'blind_test':False,
 'holdout_used':False,
 'full_prd_acceptance':False,
 'errors':'14 raw responses: 8 nonstructural design fields; 0 native JSON; 7-source images only top/outerwear',
 'cautions':['Manually confirmed labels are from a non-blind development workflow; 7 images, 14 correlated calls.',
             'Category coverage is top/outerwear only; not 8-class accuracy.',
             'Reference ROI uses historical human-reviewed candidate, not frozen 3.1 predictions.',
             'This dataset does not evaluate visual QA accuracy, copy satisfaction or serving throughput.',
             'Reported latency is not the full 3.1 + 3.2 pipeline.'],
}
summary_path=P/'PRD32_Dev7_GT_scoring_summary.json'
summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

# Standalone HTML with image previews and audit details.
css='''
:root{--ink:#14213d;--muted:#5e6a7f;--bg:#f4f7fb;--blue:#285ce5;--line:#e2e8f0;--card:#fff;--green:#11654d;--red:#ad3347;--amber:#976313}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;line-height:1.55}main{max-width:1300px;margin:0 auto;padding:30px 22px 70px}.eyebrow{letter-spacing:.13em;text-transform:uppercase;font-weight:750;font-size:12px;color:var(--blue)}h1{font-size:32px;line-height:1.18;margin:8px 0 9px;letter-spacing:-.02em}p.sub{color:var(--muted);margin:0 0 22px}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px}.label{font-size:12px;color:var(--muted);font-weight:750}.big{font-size:28px;font-weight:800;font-variant-numeric:tabular-nums;margin:7px 0 0}.mini{font-size:12px;color:var(--muted)}h2{font-size:21px;margin:29px 0 11px}h3{font-size:16px;margin:0 0 12px}.table-wrap{overflow-x:auto;border-radius:15px;border:1px solid var(--line);background:white}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:13px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:#f0f4fb;color:#47556d;font-weight:760}tr:last-child td{border-bottom:0}tr:hover td{background:#fafcff}.good{color:var(--green);font-weight:760}.bad{color:var(--red);font-weight:760}.neutral{color:var(--amber);font-weight:720}.tag{display:inline-block;background:#eef3ff;color:#234ca0;border-radius:20px;padding:3px 9px;font-weight:700;font-size:11px}.pill{display:inline-block;border-radius:20px;padding:2px 7px;font-size:11px;font-weight:750;background:#fbe8ec;color:var(--red)}.pill.ok{background:#e7f6f0;color:var(--green)}.case{display:grid;grid-template-columns:190px minmax(0,1fr);gap:16px;background:white;border:1px solid var(--line);border-radius:16px;padding:15px;margin:13px 0;break-inside:avoid}.photo{width:100%;height:280px;object-fit:contain;border:1px solid var(--line);border-radius:10px;background:#f9fafc}.predcols{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:11px}.predbox{background:#f8fafc;border:1px solid var(--line);border-radius:11px;padding:12px;font-size:13px}.fieldline{margin:5px 0}.fieldline b{color:#5c6779}.gt{background:#eaf2ff;border-radius:9px;padding:10px 12px;font-size:13px}.caption{font-size:12px;color:var(--muted)}.notes{border-left:4px solid #8096bd;padding:7px 16px;background:white;border-radius:7px;margin:13px 0}.bottom{font-size:12px;color:#697587;margin-top:26px}a{color:#285ce5}.footer{display:flex;justify-content:space-between;align-items:center;margin-top:14px}.button{background:#234ec6;color:white;border:none;border-radius:10px;padding:10px 15px;cursor:pointer;font-weight:750}details{border:1px solid var(--line);padding:12px 16px;border-radius:12px;background:white}details summary{cursor:pointer;font-weight:730}@media(max-width:900px){.kpis{grid-template-columns:repeat(2,1fr)}.case{grid-template-columns:1fr}.photo{height:340px}.predcols{grid-template-columns:1fr}}@media print{body{background:white}main{padding:0}.case{break-inside:avoid}.button{display:none}}
'''
E=lambda x:html.escape(str(x) if x is not None else '')
json_data=json.dumps(rows_audit,ensure_ascii=False)
# report table
trs=[]
for lab,v in [('红框整图','marked_full_image'),('参考 ROI','reference_roi')]:
 m=metrics[v];n=m['n_images']
 trs.append('<tr><td><b>'+lab+'</b></td>' + ''.join(f'<td>{m[k]}/{n} · {100*m[k]/n:.1f}%</td>' for k in ['category_correct','color_exact_correct','pattern_exact_correct'])+f'<td>{m["design_nonstructural_count"]}/{n}</td><td>{m["latency_median_ms"]:,.0f} ms</td></tr>')
case_html=[]
review_flag={
 'dev_02':'模型两次将套头上衣误判为 outerwear；黑色蝴蝶结应归装饰而非表面图案。当前 GT 类别为 top。',
 'dev_04':'模型两次误判 outerwear；ROI 的深绿色与确认的黑色不一致，颜色受光线影响可能较大。',
 'dev_05':'ROI 颜色为浅蓝色而 GT 是蓝色（严格评分不匹配）；两次设计字段都遗漏结构。',
 'dev_06':'ROI 识别出“圆领短袖”比整图“圆领”完整，但 ROI 未输出纯色图案标签。',
 'dev_08':'两种输入均输出“无”，遗漏无袖宽肩带、圆弧领口和横向分层。',
 'dev_09':'两次将前襟搭扣误写为“双排扣”，属设计语义错误；棕色与棕灰色在严格评分下不同。',
 'dev_12':'ROI 误写“双排扣”，整图漏掉结构设计；英文 black 归一化为黑色后计分。'
}
for i,g in G.items():
 p=P/(i+'_focus.jpg')
 pic=''
 if p.exists():
  data=base64.b64encode(p.read_bytes()).decode('ascii');pic=f'<img class="photo" alt="{E(i)} 可视化参考" src="data:image/jpeg;base64,{data}">'
 preds=[]
 for v in expected:
  r=grouped[(i,v)];caption='红框整图' if v=='marked_full_image' else '参考 ROI'
  def fmt(val,success):
   t=E(val) if str(val).strip() else '<i>未输出（空值）</i>'
   return f'<span class="{"good" if success else "bad"}">{t}</span>'
  preds.append(f'<div class="predbox"><div class="label">{caption} · {r["latency_ms_float"]:.0f} ms</div>'
    + f'<div class="fieldline"><b>类别：</b>{fmt(r["pred_category"],r["category_match"])}</div>'
    + f'<div class="fieldline"><b>颜色：</b>{fmt(r["pred_primary_color"],r["color_exact_match"])}</div>'
    + f'<div class="fieldline"><b>图案：</b>{fmt(r["pred_pattern"],r["pattern_exact_match"])}</div>'
    + f'<div class="fieldline"><b>设计：</b>{E(r["pred_visible_design"])} {"<span class=\"pill\">无效占位词</span>" if r["design_nonstructural"] else "<span class=\"pill ok\">格式未触发警报 ≠ 内容正确</span>"}</div></div>')
 case_html.append(f'<section class="case"><div>{pic}<div class="caption">{E(i)} · 原图/ROI 目视参考</div></div>'
  + f'<div><h3>{E(i)} <span class="tag">GT 已确认</span></h3>'
  +f'<div class="gt"><b>类别</b> {E(g["human_category_gt"])}　·　<b>颜色</b> {E(g["human_color_gt"])}　·　<b>图案</b> {E(g["human_pattern_gt"])}<br><b>设计</b> {E(g["human_design_gt"])}</div>'
  +'<div class="predcols">'+''.join(preds)+'</div>'
  +f'<div class="caption" style="margin-top:9px">诊断：{E(review_flag[i])}</div></div></section>')
html_report='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PRD 3.2 Dev7 人工GT评估报告</title><style>'''+css+'''</style></head><body><main><div class="eyebrow">PRD 3.2 / HUMAN GT AUDIT / 2026-10-26</div><h1>七图开发集 · 人工 GT 对照评估</h1><p class="sub">Qwen2.5-VL-7B-Instruct · RTX 4090 · 红框整图 vs 历史参考 ROI · <strong>开发诊断，不是 PRD 验收</strong></p><div class="kpis"><div class="card"><div class="label">人工确认图片</div><div class="big">7 / 7</div><div class="mini">同一图片两种推理输入</div></div><div class="card"><div class="label">类别准确率 · 两种输入均为</div><div class="big">71.4%</div><div class="mini">各 5/7 · 仅 top / outerwear</div></div><div class="card"><div class="label">原生 JSON 合规</div><div class="big">0 / 14</div><div class="mini">14/14 可解析，但输出有围栏</div></div><div class="card"><div class="label">设计字段异常</div><div class="big">8 / 14</div><div class="mini">无效占位词；还存在其他误识别</div></div></div>
<h2>1. 逐任务评分与延迟</h2><div class="table-wrap"><table><thead><tr><th>输入</th><th>类别</th><th>颜色严格匹配</th><th>图案严格匹配</th><th>设计无效占位</th><th>P50 延迟</th></tr></thead><tbody>'''+''.join(trs)+'''</tbody></table></div><div class="notes"><b>计分口径：</b>类别要求完全一致；颜色仅统一 black→黑色 等明确翻译，色深和混合色不自动视为相同；图案仅统一 格子→格纹、无→纯色。空字符串按模型未输出处理。设计是多标签信息，现阶段只列错误类型，不伪造整体设计准确率。14 次推理来自 7 张图，不是 14 张独立图片。</div><h2>2. 每张图片的模型输出和 GT</h2>'''+''.join(case_html)+'''<h2>3. 结论与下一阶段</h2><div class="card"><p><b>已确定的问题：</b>dev_02 与 dev_04 连续两种输入均发生 top→outerwear 类别错判；色彩在 ROI 中并未稳定改善；设计识别尤其是无袖、前襟搭扣、袖口等细粒度结构遗漏或幻觉较多。</p><p><b>工程优先级：</b>先冻结本版 GT 与预测结果，再建立统一属性词表（颜色层级 / 图案 vs 装饰 / 结构设计），修复 JSON 原生输出以及“无”等无效设计字段，使用同一 7 图作非盲开发回归；随后增加未用于调参的八类新验证样本和真正冻结 3.1 ROI 链路。</p><p><b>未评估：</b>3.2 开放式 QA 准确率、文案人工满意度、服务并发吞吐、3.1→3.2 完整端到端延迟；都不能凭本次 7 图的属性分数替代。</p></div><details><summary>来源、归一化规则与结果声明</summary><p>来源：用户于 2026-10-26 导出的 PRD32_Dev7_human_GT_7images.csv 与 PRD32_Dev7_prediction_GT_14rows.csv；7 条 GT 与 14 条预测已执行一一配对与标签一致性检查。</p><p>输入 reference_roi 来自历史人工审核的参考框候选，不是已冻结 3.1 的实际预测 ROI。GT 已按用户审核页面标记为 VERIFIED，但该开发集不盲测，不代表独立 PRD 验收。</p><p>更严格的评价还应有颜色容差规范、面料图案 ontology、逐结构多标签 GT、扩展独立样本与正式延迟测试协议。</p></details><div class="footer"><span class="bottom">本报告无需联网；图片已内嵌。版本：DEV7-GT-score-v1</span><button class="button" onclick="window.print()">打印 / 保存 PDF</button></div></main></body></html>'''
report_path=P/'PRD32_Dev7_人工GT评估报告_20261026.html'
report_path.write_text(html_report,encoding='utf-8')
zip_path=P/'PRD32_Dev7_GT_Evaluation_20261026.zip'
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in [report_path,audit,summary_path]:z.write(p,p.name)
print('metrics',json.dumps(metrics,ensure_ascii=False))
for p in [report_path,audit,summary_path,zip_path]:print(p,p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest()[:12])
