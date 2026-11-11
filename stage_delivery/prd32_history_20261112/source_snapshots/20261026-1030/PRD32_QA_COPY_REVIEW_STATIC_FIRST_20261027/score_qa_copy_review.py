#!/usr/bin/env python3
"""Score PRD3.2 offline QA/copy human review export without fabricating missing judgments."""
import argparse
from collections import Counter
from pathlib import Path
import json

ALLOWED={'vqa':{'CORRECT','INCORRECT','JUSTIFIED_ABSTENTION','UNJUSTIFIED_ABSTENTION','UNCERTAIN'},
         'copy':{'SATISFIED','NOT_SATISFIED','UNCERTAIN'}}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--review',required=True,help='JSON exported by QA_COPY_review_FIXED.html')
    ap.add_argument('--case-index',default=str(Path(__file__).with_name('review_case_index.json')))
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    obj=json.loads(Path(a.review).read_text('utf8'))
    cases=json.loads(Path(a.case_index).read_text('utf8'))['cases']
    expected={x['case_id']:x for x in cases}
    if obj.get('schema')!='prd32_qa_copy_review_v1':raise ValueError('review export schema mismatch')
    scored={}; errors=[]
    for row in obj.get('results',[]):
        case_id=row.get('case_id')
        if case_id not in expected: errors.append(f'unknown case_id: {case_id}');continue
        if case_id in scored:errors.append(f'duplicate case_id: {case_id}');continue
        if row.get('task')!=expected[case_id]['task']:errors.append(f'task mismatch: {case_id}')
        scored[case_id]=row
    missing=set(expected)-set(scored)
    if missing: errors.append(f'missing exported records: {sorted(missing)}')
    result={'schema':'prd32_qa_copy_human_dev_metrics_v1','status':'DEV_REVIEW_ONLY_NOT_PRD_ACCEPTANCE',
            'total_cases':len(expected),'warnings':errors, 'tasks':{}}
    lines=['# PRD 3.2 QA / 商品文案人工评价（仅开发集）','']
    for task in ('vqa','copy'):
        exp=[k for k,v in expected.items() if v['task']==task]
        verified=[];pending=[];invalid=[]
        for case_id in exp:
            row=scored.get(case_id,{})
            verdict=row.get('verdict')
            if row.get('review_status')=='VERIFIED' and verdict in ALLOWED[task]:verified.append(row)
            else:pending.append(case_id)
            if verdict and verdict not in ALLOWED[task]:invalid.append(case_id)
        counts=Counter(x.get('verdict') for x in verified)
        scored_nonuncertain=[x for x in verified if x.get('verdict')!='UNCERTAIN']
        numer=counts['CORRECT'] if task=='vqa' else counts['SATISFIED']
        total=len(scored_nonuncertain)
        rate=round(numer/total,5) if total else None
        required=0.85
        result['tasks'][task]={'expected':len(exp),'verified':len(verified),'scored_nonuncertain':total,
                'pending_case_ids':pending,'invalid_verdict_case_ids':invalid,
                'verdict_counts':dict(counts),'exploratory_rate':rate,'target_rate':required,
                'dev_complete':len(verified)==len(exp) and len(pending)==0 and not invalid and counts['UNCERTAIN']==0,
                'note':'Exploratory only: small non-blind dev data. For QA, justified abstention is tracked separately, not counted as CORRECT.'}
        title='多模态问答' if task=='vqa' else '商品文案满意度'
        lines.extend([f'## {title}',f'- 审核完成：{len(verified)}/{len(exp)}；可计分：{total}；待审核：{len(pending)}。',
                      f'- 判定分布：{dict(counts)}。',
                      f'- 开发集探索性比例：{numer}/{total} = {rate:.1%}。' if rate is not None else '- 当前没有足够人工评分计算比例。',
                      f'- PRD 目标：≥{required:.0%}；**本开发集不具有正式验收效力**。',''])
    result['all_reviewed']=all(x['dev_complete'] for x in result['tasks'].values())
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'qa_copy_score_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    lines.insert(2,'**注意：未完成的判定不可填作正确或满意；计分样本过少，不可将开发集比例直接宣称达到 PRD 85%。**')
    lines.extend(['## 后续','- 用独立、覆盖更全面的图像与问题做正式评价，记录人工 GT、审核者、判定理由。',
                  '- 保留不确定/合理拒答比例，评估问答准确率时不得把合理拒答自动计为正确。',
                  '- 属性标签准确率需单独八类 GT 与字段级评价；本脚本不会生成这一指标。',''])
    (out/'QA_COPY_开发集人工评价摘要.md').write_text('\n'.join(lines),encoding='utf8')
    print('DEV_REVIEW_REPORT_CREATED',len(expected),'cases','all_reviewed=',result['all_reviewed'])

if __name__=='__main__':main()
