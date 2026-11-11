import csv,json,base64,html,statistics
from pathlib import Path
from collections import defaultdict
P=Path('/mnt/data')
GT={r['image_id']:r for r in csv.DictReader(open(P/'PRD32_Dev7_human_GT_7images.csv',encoding='utf-8-sig'))}
V1={(r['image_id'],r['input_variant']):r for r in map(json.loads,open(P/'predictions (1).jsonl',encoding='utf-8'))}
V2={(r['image_id'],r['input_variant']):r for r in map(json.loads,open(P/'predictions_v2.jsonl',encoding='utf-8'))}
S=json.load(open(P/'summary (7)(1).json',encoding='utf-8'))
ids=sorted(GT)
variants=['marked_full_image','reference_roi']
assert len(ids)==7 and len(V2)==14 and len(V1)==14
rows=[]
for id in ids:
 for variant in variants:
  gt=GT[id]; a=V1[id,variant]; b=V2[id,variant]
  p=a.get('parsed') or {}; q=b.get('canonical_service_json') or {}
  rows.append({'image_id':id,'variant':variant,'gt_category':gt['human_category_gt'],'v1_category':p.get('category'),'v2_category':q.get('category'),'gt_color':gt['human_color_gt'],'v1_color':p.get('primary_color'),'v2_color':q.get('primary_color'),'gt_pattern':gt['human_pattern_gt'],'v1_pattern':p.get('pattern'),'v2_pattern':q.get('pattern'),'gt_design':gt['human_design_gt'],'v1_design':p.get('visible_design'),'v2_design':q.get('visible_design'),'v2_issues':';'.join(b.get('issues',[])),'v1_ms':a.get('end_to_end_ms'),'v2_ms':b.get('end_to_end_ms'),'v2_native_json':b.get('native_json'),'gt_eligible':gt.get('gt_scoring_eligible','')})
with open(P/'PRD32_Dev7_V1_V2_逐图对照_20261026.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
E=lambda x: html.escape('—' if x is None else str(x),quote=True)
notes={
'dev_02':'类别已纠正；整图错误写“短袖”，ROI 写“V领”；蝴蝶结不应算面料图案；标准图案 GT 为“无”。',
'dev_04':'类别已纠正；V2 把可见翻领误写为“圆领”，设计信息仍不准确。',
'dev_05':'类别与条纹正确；V2 把长袖衬衫误写为“圆领;短袖”；颜色“浅蓝色”与人工主色“蓝色”严格不匹配。',
'dev_06':'类别、颜色、纯色、圆领短袖均与人工标签相符；保留作为正例对照。',
'dev_08':'整图识别到领口和肩带，但遗漏层叠设计；ROI 仍输出无效“无”，被服务端改成 null。',
'dev_09':'类别、纹理正确；把前中部三组搭扣错误写成“双排扣”，棕色与棕灰色严格不匹配。',
'dev_12':'重要回归：V1 类别 outerwear 正确，V2 两种输入变成 top；模型还写错圆领/短袖，漏西装翻领和敞开前襟。'
}
header='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PRD 3.2 V1-V2 开发集回归诊断</title><style>
:root{font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;color:#17253d;background:#f6f8fc}*{box-sizing:border-box}body{margin:0}main{max-width:1200px;margin:0 auto;padding:32px 20px 72px}h1{font-size:28px;margin:0 0 6px}h2{font-size:20px;margin:32px 0 16px}p{line-height:1.7}.muted{color:#64748b}.meta{font-size:13px;color:#65758b}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.stat{background:white;border:1px solid #dfe5ed;border-radius:13px;padding:16px}.stat b{display:block;font-size:23px;margin-top:8px}.warning{border-left:4px solid #d97706;background:#fffaf1;padding:13px 15px;border-radius:6px}.case{background:white;border:1px solid #dfe5ed;border-radius:14px;margin:15px 0;overflow:hidden}.case-head{display:flex;align-items:center;gap:14px;padding:15px 20px;background:#f9fbff}.case-head h3{margin:0}.case-body{display:grid;grid-template-columns:190px 1fr;gap:18px;padding:16px 20px 20px}.case img{width:100%;max-height:265px;object-fit:contain;background:#f5f5f5;border-radius:8px}.case table{width:100%;border-collapse:collapse;font-size:13px;table-layout:fixed}.case td,.case th{border-bottom:1px solid #e8eef5;padding:8px;vertical-align:top;overflow-wrap:anywhere;text-align:left}.case th{background:#f1f5fa}.case .note{font-size:13px;border-left:3px solid #4665ae;padding:8px 11px;background:#f6f9ff}.bad{color:#b91c1c;font-weight:600}.good{color:#15803d;font-weight:600}footer{margin:38px 0 0;color:#64748b;font-size:13px}@media(max-width:850px){.stats{grid-template-columns:1fr 1fr}.case-body{grid-template-columns:1fr}.case img{max-height:240px}}@media(max-width:420px){.stats{grid-template-columns:1fr}}</style></head><body><main><h1>PRD 3.2 · Dev7 V1 → V2 回归诊断</h1><p class="muted">7 张非盲开发图 · 红框整图和候选参考 ROI 各 7 条 · 基于原始模型响应、服务端规范输出及人工 GT</p><div class="stats"><div class="stat">类别准确率（两种视图）<b>71.4% → 85.7%</b></div><div class="stat">图案准确率（两种视图）<b>57.1/42.9% → 85.7%</b></div><div class="stat">原生 JSON（V2）<b>0 / 14</b></div><div class="stat">V2 设计字段格式异常<b>1 / 14</b></div></div><p class="warning"><b>特别注意：</b>V2 修正 dev_02、dev_04 的类别，但 dev_12 从 outerwear 退化为 top；设计字段虽然更符合格式，不等于设计内容准确。7 图、两类、非盲开发结果不能作为完整 PRD 验收依据。</p><h2>逐图回归对照</h2>'''
parts=[header]
for id in ids:
 gt=GT[id]
 image=P/f'{id}_focus.jpg'
 b64=base64.b64encode(image.read_bytes()).decode('ascii')
 parts.append(f'<section class="case"><div class="case-head"><h3>{E(id)}</h3><small class="muted">GT 类别：{E(gt["human_category_gt"])} · 人工审核状态：{E(gt["human_review_status"])}</small></div><div class="case-body"><div><img alt="{E(id)}参考区域" src="data:image/jpeg;base64,{b64}"></div><div><p class="note">{E(notes[id])}</p><table><thead><tr><th style="width:16%">输入</th><th style="width:19%">类别 V1 → V2</th><th style="width:19%">颜色 V1 → V2</th><th style="width:19%">图案 V1 → V2</th><th>设计 V1 → V2</th></tr></thead><tbody>')
 for variant in variants:
  a=V1[id,variant].get('parsed') or {};b=V2[id,variant].get('canonical_service_json') or {};raw=V2[id,variant].get('raw_parsed') or {}
  parts.append(f'<tr><td>{E(variant)}</td><td>{E(a.get("category"))} → <b>{E(b.get("category"))}</b></td><td>{E(a.get("primary_color"))} → <b>{E(b.get("primary_color"))}</b></td><td>{E(a.get("pattern"))} → <b>{E(b.get("pattern"))}</b></td><td>{E(a.get("visible_design"))} → <b>{E(b.get("visible_design"))}</b></td></tr>')
 parts.append('</tbody></table><p class="meta">人工 GT：颜色 '+E(gt['human_color_gt'])+'；图案 '+E(gt['human_pattern_gt'])+'；设计 '+E(gt['human_design_gt'])+'。<br>表中 V2 为服务端规范结果；原始 JSON 合规率另行统计，规范化不代表模型本身输出合规。</p></div></div></section>')
parts.append('''<h2>下一步建议</h2><p>① 把 dev_12 设为回归阻断案例，防止类别“outerwear → top”退化；② 对长袖/短袖、圆领/翻领、搭扣/双排扣使用显式可见证据判别；③ 正式区分设计字段格式合规率与多标签语义准确率；④ 在未见过的新开发样本上评估，再独立完成 QA、文案和延迟验收。</p><footer>本报告只读取已有文件生成，不触碰任何封存 holdout，也不改变冻结 PRD 3.1。V2 单次耗时是开发环境测量值，非服务级 P95/QPS。</footer></main></body></html>''')
out=P/'PRD32_Dev7_V1_V2_回归对照报告_20261026.html'
out.write_text(''.join(parts),encoding='utf-8')
print('wrote',out,out.stat().st_size,'csv rows',len(rows))
