import csv, json, statistics, html, base64, mimetypes
from pathlib import Path
from collections import Counter
D=Path('/mnt/data')
summary=json.loads((D/'summary (8).json').read_text(encoding='utf-8'))
v3=[json.loads(x) for x in (D/'predictions_v3.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
v2=[json.loads(x) for x in (D/'predictions_v2.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
with (D/'PRD32_Dev7_human_GT_7images.csv').open(encoding='utf-8-sig',newline='') as f: gt={r['image_id']:r for r in csv.DictReader(f)}
V2={(r['image_id'],r['input_variant']):r for r in v2}
V3={(r['image_id'],r['input_variant']):r for r in v3}
ids=['dev_02','dev_04','dev_05','dev_06','dev_08','dev_09','dev_12']
variants=('marked_full_image','reference_roi')
colors={'black':'黑色','white':'白色','blue':'蓝色'}
patterns={'无':'纯色','无图案':'纯色','solid':'纯色','格子':'格纹'}
def norm(x,m):
 if x is None:return None
 return m.get(str(x).strip().lower(),str(x).strip().lower())
def E(x):return html.escape(str(x if x is not None else '—'))
def shortjson(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'))
rows=[]
for iid in ids:
 for var in variants:
  z=V3[(iid,var)];p=z['parsed'] or {};d=p.get('design') or {};g=gt[iid]
  z2=V2[(iid,var)]; old=z2.get('canonical_service_json') or z2.get('raw_parsed') or {}
  raw_color_match=norm(p.get('primary_color'),colors)==norm(g['human_color_gt'],colors)
  raw_pattern_match=norm(p.get('pattern'),patterns)==norm(g['human_pattern_gt'],patterns)
  r={'image_id':iid,'variant':var,'gt_category':g['human_category_gt'],'v2_category':old.get('category'),'v3_raw_category':p.get('category'),
      'gt_color':g['human_color_gt'],'v2_color':old.get('primary_color'),'v3_raw_color':p.get('primary_color'),'v3_color_raw_match':raw_color_match,
      'gt_pattern':g['human_pattern_gt'],'v2_pattern':old.get('pattern'),'v3_raw_pattern':p.get('pattern'),'v3_pattern_raw_match':raw_pattern_match,
      'gt_design':g['human_design_gt'],'v2_design':old.get('visible_design'),'v3_design_json':shortjson(d),
      'v3_schema_valid':z['schema_valid'],'v3_native_json':z['raw_native_json'], 'v3_errors':';'.join(z['errors']),
      'v2_latency_ms':z2['end_to_end_ms'],'v3_latency_ms':z['end_to_end_ms'],
      'v3_output_type':'RAW_PARSED_UNSCORABLE' if not z['schema_valid'] else 'SCHEMA_VALID'}
  rows.append(r)
output_csv=D/'PRD32_Dev7_V3_独立诊断_14rows.csv'
with output_csv.open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
checks={}
for var in variants:
 rr=[r for r in rows if r['variant']==var]
 checks[var]={
  'schema_valid':sum(x['v3_schema_valid'] for x in rr),
  'raw_color_match_diagnostic_not_official':sum(x['v3_color_raw_match'] for x in rr),
  'raw_pattern_match_diagnostic_not_official':sum(x['v3_pattern_raw_match'] for x in rr),
  'invalid_case_ids':[x['image_id'] for x in rr if not x['v3_schema_valid']],
 }
report={'decision':'DO_NOT_PROMOTE_V3_KEEP_V2_BASELINE','comparison_scope':'Seven non-blind dev images, paired marked/ROI, human GT scoring-only',
        'official_metrics_source':'summary (8).json','additional_raw_field_diagnostics_not_official':checks,
        'critical_observations':['Model emits out-of-vocabulary shirt for category (dev_04 both and dev_05 marked).',
                                 'Model emits literal string null instead of JSON null in dev_12; schema also rejects strap.',
                                 'Design hallucination: backpack straps as clothing straps on dev_06; open front on dev_08.',
                                 'V3 design recall measures only named GT fields; no false-positive control, not F1.',
                                 'All outputs still Markdown-fenced rather than native JSON.'],
        'next_actions':['Archive V2 as stronger same-sample development baseline; V3 only as failed structured-attribute ablation.',
                        'Separate parsing/schema/semantic quality metrics and do not score invalid result as visual error.',
                        'Do not tune further against only these same 7 images; expand eight-class development images.',
                        'Evaluate QA and generated copy in a distinct test stage; optimize latency afterwards.']}
(D/'PRD32_Dev7_V3_诊断结论.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
CSS='''body{font-family:Arial,"Microsoft YaHei",sans-serif;background:#f5f7fb;color:#172334;margin:0;padding:32px;line-height:1.5}main{max-width:1120px;margin:auto}.tag{display:inline-block;background:#eef0f4;border-radius:100px;padding:4px 11px;color:#44516a;font-size:12px}.h{background:#fff;border:1px solid #e4e8f0;border-radius:14px;padding:21px 23px;margin:16px 0}h1{font-size:29px;margin:9px 0 4px}h2{font-size:20px;margin:0 0 11px}h3{font-size:16px;margin:8px 0 10px}.minor{color:#5c6677;font-size:13px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.metric{background:#f6f8fb;border-radius:10px;padding:12px}.big{font-size:25px;font-weight:750}.bad{color:#ab3e35}.ok{color:#23734e}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;border-bottom:1px solid #e7ebf2;text-align:left;vertical-align:top}th{background:#f4f6fa}.r{display:grid;grid-template-columns:175px 1fr;gap:18px;align-items:start}.r img{width:175px;max-height:230px;object-fit:contain;border:1px solid #e9eef4;border-radius:8px;background:#fff}.sgrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.sub{background:#fafbfe;border:1px solid #e8ecf5;border-radius:8px;padding:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;margin:7px 0}.hint{background:#fbf4ee;border-left:3px solid #c08044;padding:12px} @media(max-width:760px){body{padding:14px}.metrics{grid-template-columns:repeat(2,1fr)}.r{grid-template-columns:1fr}.r img{max-width:220px}.sgrid{grid-template-columns:1fr}}'''
parts=[f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PRD 3.2 V3 质量回归审计</title><style>{CSS}</style></head><body><main>
<span class="tag">PRD 3.2 · 开发集非盲对照</span><h1>V3 结构化输出回归审计</h1><p class="minor">基于用户提供的 V2/V3 模型输出、7 张人工 GT 与同一批 7 图；结果用于工程决策，不是 PRD 正式验收。</p>
<section class="h"><h2>版本决策：V3 暂不晋级</h2><div class="metrics"><div class="metric"><div class="minor">整图类别 V2 → V3</div><div class="big bad">85.7% → 57.1%</div></div><div class="metric"><div class="minor">ROI 类别 V2 → V3</div><div class="big bad">85.7% → 71.4%</div></div><div class="metric"><div class="minor">V3 严格结构合格</div><div class="big bad">9 / 14</div></div><div class="metric"><div class="minor">V3 原生 JSON</div><div class="big bad">0 / 14</div></div></div><p>V3 在领口、袖长等设计字段上产生了更细粒度的诊断信息，但引入非法类别、字符串“null”和错误归属，且推理时间增加。继续保留 V2 作为开发基线。</p></section>
<section class="h"><h2>原报告指标（严格 Schema 有效才计分）</h2><table><tr><th>指标</th><th>V2 整图</th><th>V3 整图</th><th>V2 ROI</th><th>V3 ROI</th></tr>''']
for label,k2,k3 in [('类别','category_acc','category_acc'),('颜色','color_exact_acc','color_acc'),('图案','pattern_exact_acc','pattern_acc')]:
 parts.append(f'<tr><td>{label}</td><td>{summary["v2_summary_metrics"]["marked_full_image"][k2]*100:.1f}%</td><td>{summary["v3_metrics"]["marked_full_image"][k3]*100:.1f}%</td><td>{summary["v2_summary_metrics"]["reference_roi"][k2]*100:.1f}%</td><td>{summary["v3_metrics"]["reference_roi"][k3]*100:.1f}%</td></tr>')
parts.append(f'''<tr><td>中位耗时</td><td>{summary['v2_summary_metrics']['marked_full_image']['median_latency_ms']:.0f} ms</td><td>{summary['v3_metrics']['marked_full_image']['median_latency_ms']:.0f} ms</td><td>{summary['v2_summary_metrics']['reference_roi']['median_latency_ms']:.0f} ms</td><td>{summary['v3_metrics']['reference_roi']['median_latency_ms']:.0f} ms</td></tr></table>
<p class="hint">计分说明：V3 评分脚本将不通过 Schema 的结果视为整条无效，因此例如某次颜色虽然预测正确，但若 category="shirt"（非法）仍计作该条颜色未命中。下方额外列出原始字段诊断；不可与正式准确率混用。</p><table><tr><th>仅看 V3 原始可解析字段</th><th>整图</th><th>ROI</th></tr><tr><td>颜色独立匹配（诊断）</td><td>{checks['marked_full_image']['raw_color_match_diagnostic_not_official']}/7</td><td>{checks['reference_roi']['raw_color_match_diagnostic_not_official']}/7</td></tr><tr><td>图案独立匹配（诊断）</td><td>{checks['marked_full_image']['raw_pattern_match_diagnostic_not_official']}/7</td><td>{checks['reference_roi']['raw_pattern_match_diagnostic_not_official']}/7</td></tr></table></section>
<section class="h"><h2>具体错误和归因</h2><p><b>协议问题：</b>dev_04 两种视图和 dev_05 整图输出 <code>category="shirt"</code>；dev_12 把 JSON null 写成字符串 "null"，而设计 strap 为字面字符串 null，导致 Schema 失败。所有 14 条原始输出仍不是直接可用的原生 JSON。</p><p><b>视觉语义问题：</b>dev_06 把背包肩带当成上衣肩带；dev_08 把无袖分层上衣误写成敞开前襟；dev_09 把搭扣解释成普通纽扣门襟；dev_12 类别仍错为 top。</p><p><b>指标范围：</b>设计召回为 GT 已明确写出的 21 项特征中的匹配数量，不计未提及字段的假阳性；它不是设计属性 Precision 或 F1。</p></section><h2>逐图 V2 / V3 原始预测对照</h2>''')
for iid in ids:
 g=gt[iid]; im=D/f'{iid}_focus.jpg';uri='data:image/jpeg;base64,'+base64.b64encode(im.read_bytes()).decode() if im.exists() else ''
 parts.append(f'<section class="h"><div class="r"><img src="{uri}" alt="{iid}"><div><h3>{iid}</h3><p class="minor"><b>GT 类别</b> {E(g["human_category_gt"])}　<b>颜色</b> {E(g["human_color_gt"])}　<b>图案</b> {E(g["human_pattern_gt"])}</p><p class="minor"><b>GT 可见设计</b> {E(g["human_design_gt"])}</p><div class="sgrid">')
 for var in variants:
  r=V3[(iid,var)];rr=V2[(iid,var)];old=rr.get('canonical_service_json') or rr.get('raw_parsed') or {}
  title='红框整图' if var=='marked_full_image' else '参考 ROI'
  status='<span class="ok">Schema 通过</span>' if r['schema_valid'] else '<span class="bad">Schema 不通过: '+E(','.join(r['errors']))+'</span>'
  parts.append(f'<div class="sub"><b>{title}</b> · {status}<div class="minor">V2 {rr["end_to_end_ms"]:.0f} ms / V3 {r["end_to_end_ms"]:.0f} ms</div><div class="minor">V2</div><pre>{E(shortjson(old))}</pre><div class="minor">V3 原始解析（即使无效也显示，非正式评分）</div><pre>{E(shortjson(r["parsed"]))}</pre></div>')
 parts.append('</div></div></div></section>')
parts.append('''<section class="h"><h2>下一步工作排序</h2><ol><li><b>冻结 V2 为当前质量基线</b>，V3 保留为不晋级消融结果。</li><li><b>分开统计协议合规和视觉属性</b>；在服务端对非法类别、字符串 null、与类别矛盾的设计属性直接拒绝，而不是悄悄修成正确答案。</li><li><b>扩大开发集到其他六类服饰</b>，不继续针对这 7 张非盲图调参。</li><li><b>补齐独立的开放式问答与商品文案评价</b>，再研究解码加速与服务端 P95/QPS。</li></ol></section><p class="minor">全部基于本地用户文件离线整理 · 冻结 3.1 不变 · holdout 未用 · 历史参考框并非 3.1 实际预测 ROI</p></main></body></html>''')
report_path=D/'PRD32_Dev7_V3_失败归因与版本决策_20261026.html';report_path.write_text(''.join(parts),encoding='utf-8')
import zipfile
zip_path=D/'PRD32_Dev7_V3_诊断包_20261026.zip'
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in [report_path,output_csv,D/'PRD32_Dev7_V3_诊断结论.json']:
  z.write(p,p.name)
print('created: ',[(p.name,p.stat().st_size) for p in [report_path,output_csv,zip_path]])
print('diagnostics: ',checks)
