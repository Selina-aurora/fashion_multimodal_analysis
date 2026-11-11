#!/usr/bin/env python3
"""Stdlib-only offline replay over fixed DEV60. No GPU, no network, no writes to original."""
from __future__ import annotations
import argparse,csv,hashlib,html,json,collections,statistics
from pathlib import Path
from policy_vqa_only_v1 import evaluate,SCHEMA
HERE=Path(__file__).resolve().parent
AUDIT='INPUT_VQA_ONLY_171.jsonl'
REVIEW='INPUT_DEV60_USER_CONFIRMED_ASSISTED_171.csv'
FROZEN='INPUT_FROZEN_171.json'


def read_inputs():
    # Integrity-manifest only verifies archived input bytes; doesn't imply independent GT.
    hashes=json.loads((HERE/'INPUT_SHA256.json').read_text(encoding='utf-8'))
    for name,expected in hashes.items():
        actual=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError(f'Input modified: {name}')
    rows=[json.loads(x) for x in (HERE/AUDIT).read_text(encoding='utf8').splitlines() if x.strip()]
    with (HERE/REVIEW).open(encoding='utf-8-sig',newline='') as f: reviews=list(csv.DictReader(f))
    frozen=json.loads((HERE/FROZEN).read_text(encoding='utf8'))['targets']
    def k(x):return x['case_id']+'/'+x['instance_id']
    if len(rows)!=171 or len(reviews)!=171 or len(frozen)!=171:
        raise ValueError('Not exactly 171 records in each input')
    if len({k(x) for x in rows})!=171 or len({k(x) for x in reviews})!=171:
        raise ValueError('Duplicate instance IDs')
    if set(map(k,rows))!=set(map(k,reviews)) or set(map(k,rows))!=set(frozen):
        raise ValueError('Missing/unexpected instance in join')
    return rows,{k(x):x for x in reviews},frozen


def run(out:Path):
    rows,reviews,frozen=read_inputs()
    out.mkdir(exist_ok=True,parents=True)
    results=[]
    for r in rows:
        key=r['case_id']+'/'+r['instance_id']
        results.append(evaluate(r,reviews[key],frozen[key]))
    assert len(results)==171
    assert all(x['publish_allowed'] is False and not x['eligible_for_production'] for x in results)
    (out/'GUARD_SHADOW_171.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in results),encoding='utf-8')
    header=['review_key','question_type','primary_status','risk_reason_count','risk_reasons',
      'answer','latency_total_ms','generated_tokens','target_identity_review','mask_review',
      'vqa_quality_review','category_reference_status','publish_allowed','eligible_for_production',
      'dev_review_type']
    with (out/'GUARD_SHADOW_171.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=header);w.writeheader()
        for x in results:
            r={k:x.get(k,'') for k in header}
            r['risk_reasons']=';'.join(x['risk_reasons'])
            w.writerow(r)
    statuses=collections.Counter(x['primary_status'] for x in results)
    reasons=collections.Counter(reason for x in results for reason in x['risk_reasons'])
    summary={'schema':SCHEMA,'input_count':171,'matched_instance_count':len(results),
      'status_counts':dict(statuses),'reason_counts':dict(reasons),
      'any_risk_count':sum(bool(x['risk_reasons']) for x in results),
      'shadow_review_candidate_count':statuses['SHADOW_REVIEW_CANDIDATE'],
      'production_released_count':0,'original_three_task_guard_executed':False,
      'independent_test_used':False,'request_type':'color_only',
      'dev_audit_provenance':'USER_CHAT_CONFIRMED_AI_ASSISTED_NOT_INDEPENDENT_GT',
      'mean_vqa_only_ms':round(statistics.mean(float(x['latency_total_ms']) for x in results),3),
      'interpretation':'Shadow risk triage. Zero production releases. Not a PRD acceptance accuracy result.'}
    (out/'GUARD_SHADOW_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    table_rows=[]
    for r in results:
        tds=[r['review_key'],r['primary_status'],'; '.join(r['risk_reasons']),r['answer'],
             r['target_identity_review'],r['mask_review'],r['vqa_quality_review'],str(r['latency_total_ms'])]
        table_rows.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in tds)+'</tr>')
    template='''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>VQA-only DEV60 Guard Shadow 171</title>
<style>body{font-family:system-ui,'Microsoft YaHei',sans-serif;max-width:1480px;margin:auto;padding:24px;background:#f7f8fa;color:#182235}h1{font-size:25px}.lead{color:#526173}section{background:white;padding:18px;border-radius:12px;border:1px solid #e0e4ea;margin-bottom:16px}input,select{padding:10px;border:1px solid #cbd4df;border-radius:7px;font-size:14px}table{width:100%;border-collapse:collapse;font-size:12px}td,th{text-align:left;border-bottom:1px solid #eee;padding:8px;vertical-align:top;word-break:break-word}th{position:sticky;top:0;background:white}td:nth-child(3){min-width:260px}td:nth-child(4){min-width:90px}.scroll{max-height:70vh;overflow:auto}.alert{font-weight:bold;color:#8f2a2a}</style></head><body><h1>PRD 3.2 · VQA-only Guard Shadow · DEV60</h1>
<section><p class="alert">全部 171 条均禁止发布。此页仅是离线风险分诊，不是上线 Guard，也不是独立盲测准确率。</p><p class="lead">全部问题仍为固定颜色问答；历史审核是 AI 辅助并经聊天批量确认。原三任务 Guard 没有用于本实验。</p><p id="numbers"></p></section>
<section><input id="search" placeholder="搜索实例、状态、回答或风险" size="43" oninput="flt()"> <select id="status" onchange="flt()"><option value="">全部状态</option><option>BLOCKED_INTEGRITY</option><option>BLOCKED_TARGET_EVIDENCE</option><option>BLOCKED_ANSWER_REVIEW</option><option>SHADOW_REVIEW_CANDIDATE</option></select><div class="scroll"><table id="tbl"><thead><tr><th>实例</th><th>主状态</th><th>风险原因</th><th>回答</th><th>目标</th><th>Mask</th><th>VQA 质量审核</th><th>E2E ms</th></tr></thead><tbody>ROWS</tbody></table></div></section><script>function flt(){const q=document.getElementById('search').value.toLowerCase(),s=document.getElementById('status').value;let c=0;document.querySelectorAll('#tbl tbody tr').forEach(r=>{let ok=(!s||r.cells[1].textContent===s)&&r.textContent.toLowerCase().includes(q);r.hidden=!ok;if(ok)c++});document.getElementById('numbers').textContent='当前显示 '+c+' / 171 条'}flt();</script></body></html>'''
    (out/'GUARD_SHADOW_171_REVIEW.html').write_text(template.replace('ROWS','\n'.join(table_rows)),encoding='utf8')
    print('GUARD_SHADOW_V1_OFFLINE_REPLAY_PASS 171/171 ALL_RELEASE_FALSE')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=HERE/'output')
    args=ap.parse_args();run(args.output)
