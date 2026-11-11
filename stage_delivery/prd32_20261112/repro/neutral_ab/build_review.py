#!/usr/bin/env python3
'''Build offline user review HTML from real GPU JSONL records.'''
import argparse,json,csv,html,base64,collections
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',default='outputs/neutral_results.jsonl')
    ap.add_argument('--output',default='outputs/NEUTRAL_5_PAIRS_REVIEW.html')
    a=ap.parse_args()
    root=Path(__file__).resolve().parent
    records=root/a.results if not Path(a.results).is_absolute() else Path(a.results)
    if not records.is_file():raise SystemExit('No GPU results found. First run run_neutral_ab.py.')
    rows=[json.loads(s) for s in records.read_text(encoding='utf-8').splitlines() if s.strip()]
    look={(r['case_id'],r['arm']):r for r in rows}
    with (root/'manifest.csv').open(encoding='utf-8-sig',newline='') as f: manifest=list(csv.DictReader(f))
    out=root/a.output if not Path(a.output).is_absolute() else Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    cards=[]
    for c in manifest:
        pair=[]
        for arm,color in (('red','红框'),('cyan','青框')):
            r=look.get((c['case_id'],arm))
            p=root/c[f'{arm}_image']
            src='data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
            answer=html.escape(str(r['answer'])) if r else '<em>GPU 尚未生成该条结果</em>'
            meta=(html.escape(f"{r['duration_ms']} ms · {r['generated_tokens']} tokens") if r else '待运行')
            pair.append(f'<div class="arm"><strong>{color}（{arm.upper()}）</strong><img src="{src}" alt="{color}图像"/><p class="answer">{answer}</p><small>{meta}</small></div>')
        summary=html.escape(c['legacy_visual_note'])
        cards.append(f'''<section class="card" data-id="{c['case_id']}"><h3>{c['case_id']} · {html.escape(c['target'])} · {html.escape(c['question_id'])}</h3>
         <p class="q">{html.escape(c['prompt'])}</p><div class="pair">{''.join(pair)}</div>
         <details><summary>上一轮已确认的定性观察（非独立GT）</summary><p>{summary}</p></details>
         <label>复测结论 <select class="verdict"><option>待判定</option><option>红框更好</option><option>青框更好</option><option>两者相近且基本正确</option><option>两者均有实质错误</option><option>两者均拒答或证据不足</option><option>无法判断</option></select></label>
         <label>错误类型（可多选）<span class="checks">{''.join('<label><input type="checkbox" value="'+ x +'">'+x+'</label>' for x in ['目标绑定','类别串扰','Logo/文字遗漏','材质臆测','错误拒答','其他'])}</span></label>
         <label>人工备注 <textarea class="note" placeholder="记录哪句正确、哪句错误，以及可见证据"></textarea></label>
         <label><input type="checkbox" class="confirm"> 我已经逐图核对本条复测</label>
         </section>''')
    page='''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PRD32 Oracle8 中性提问配对复测审核</title><style>
    body{font:15px/1.5 system-ui,Arial,sans-serif;background:#f5f7fb;color:#182332;max-width:1100px;margin:20px auto;padding:12px} .card,#intro{background:white;border:1px solid #e0e6ed;border-radius:12px;padding:18px;margin:14px 0}h2{margin:0 0 4px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}.arm{display:flex;flex-direction:column;padding:10px;border:1px solid #ddd;border-radius:9px}.arm img{width:100%;max-height:440px;object-fit:contain}.answer{background:#f4f5f6;padding:9px;border-radius:5px;min-height:48px}.q{background:#edf3fc;padding:10px}.checks{display:flex;flex-wrap:wrap;gap:10px}.checks label{display:inline-block;margin-right:7px}label{display:block;margin-top:12px}select,textarea{font:inherit;padding:8px;max-width:100%;width:100%;box-sizing:border-box}textarea{min-height:60px}button{font:inherit;background:#1e5693;color:white;border:0;border-radius:8px;padding:11px;cursor:pointer}small{color:#666}details{margin-top:14px}#intro{position:sticky;top:0;z-index:4;box-shadow:0 2px 9px #0001}@media(max-width:640px){.pair{grid-template-columns:1fr}}
    </style></head><body><div id="intro"><h2>Oracle8 · 5 组标记颜色中性提问复测</h2><p>同一源图，仅框线像素变化；提问均不再包含“红框/青框”。模型答案来自 GPU JSONL；历史备注不是独立盲测 GT。状态：<b>SHADOW_BLOCKED</b>。</p><p id="prog">0/5 已审核</p><button onclick="exportCSV()">导出人工复核 CSV</button></div><main>'''+''.join(cards)+'''</main><script>
    document.querySelectorAll('.confirm').forEach(x=>x.addEventListener('change',()=>{document.querySelector('#prog').innerText=[...document.querySelectorAll('.confirm')].filter(a=>a.checked).length+'/5 已审核'}));
    function quote(s){return '"'+String(s??'').replaceAll('"','""').replaceAll('\\r',' ').replaceAll('\\n',' ')+'"'}
    function exportCSV(){const rows=[...document.querySelectorAll('section.card')].map(el=>({case_id:el.dataset.id,verdict:el.querySelector('.verdict').value,error_tags:[...el.querySelectorAll('.checks input:checked')].map(x=>x.value).join(';'),note:el.querySelector('.note').value,human_confirmed:el.querySelector('.confirm').checked,independent_blind_GT:false,production_release_allowed:false})); const k=Object.keys(rows[0]);const csv='\\uFEFF'+[k.map(quote).join(','),...rows.map(o=>k.map(a=>quote(o[a])).join(','))].join('\\r\\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const link=document.createElement('a');link.href=url;link.download='NEUTRAL_5_PAIRS_HUMAN_REVIEW.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),2000)}
    </script></body></html>'''
    out.write_text(page,encoding='utf-8')
    print(f'Review HTML: {out}; GPU rows {len(rows)}/10')
if __name__=='__main__':main()
