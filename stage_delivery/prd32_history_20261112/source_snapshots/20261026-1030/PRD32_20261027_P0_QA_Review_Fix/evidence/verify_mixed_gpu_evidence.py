#!/usr/bin/env python3
"""Read-only cross-file validation for PRD32 mixed P0 real-GPU fault-injection tests."""
import argparse
from collections import Counter
from pathlib import Path
import json
import sys


def read_jsonl(path):
    rows=[]
    with open(path, encoding='utf-8') as handle:
        for lineno,line in enumerate(handle,1):
            if line.strip():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f'{path}:{lineno}: invalid output JSONL: {e}')
    return rows


def validate(summary, decisions, predictions, service):
    errors=[]
    n=summary.get('source_requests')
    for name, rows in [('input_decisions',decisions),('predictions',predictions),('service_outputs',service)]:
        if len(rows)!=n: errors.append(f'{name} has {len(rows)} records, expected {n}')
    index=lambda r,k: (r.get('source_line'), r.get('case_id') if k!='decisions' else (r.get('identity') or {}).get('case_id'))
    pass_n=blocked_n=generated=blocked_without_model=blocked_null=0
    issue_counts=Counter()
    for i in range(min(len(decisions),len(predictions),len(service))):
        d,p,s=decisions[i],predictions[i],service[i]
        dk=index(d,'decisions')
        if dk!=index(p,'predictions') or dk!=index(s,'service'):
            errors.append(f'row {i+1}: source_line/case_id mismatch {dk}, {index(p,"predictions")}, {index(s,"service")}')
        dec=d.get('decision')
        codes=[issue.get('code') for issue in d.get('issues',[]) if isinstance(issue,dict)]
        issue_counts.update(codes)
        if dec=='PASS':
            pass_n+=1
            if p.get('status')!='GENERATED_NOT_ACCEPTED' or not s.get('inference_performed') or s.get('status')=='BLOCKED_INVALID_INPUT':
                errors.append(f'row {i+1} was PASS but not inferred as expected')
            else: generated+=1
            if s.get('payload') is None: errors.append(f'row {i+1} pass payload missing')
        elif dec=='BLOCKED':
            blocked_n+=1
            if p.get('status')!='SKIPPED_INVALID_INPUT': errors.append(f'row {i+1} blocked but prediction not skipped')
            if s.get('status')!='BLOCKED_INVALID_INPUT': errors.append(f'row {i+1} blocked but service status incorrect')
            if s.get('inference_performed') is False: blocked_without_model+=1
            else: errors.append(f'row {i+1} blocked but model inference flag present')
            if s.get('payload') is None: blocked_null+=1
            else: errors.append(f'row {i+1} blocked but payload is non-null')
        else: errors.append(f'row {i+1}: unexpected decision {dec!r}')
        if s.get('simulated_backend') is not False: errors.append(f'row {i+1}: simulated_backend not false')
    if pass_n!=summary.get('eligible_for_inference') or blocked_n!=summary.get('blocked_input_cases'):
        errors.append('recomputed PASS/BLOCKED disagree with summary')
    if generated!=summary.get('inference_completed') or summary.get('inference_failed')!=0:
        errors.append('inference count/failures disagree with summary')
    if summary.get('simulated_backend') is not False or summary.get('model_not_loaded') is not False:
        errors.append('summary does not indicate real GPU backend')
    if summary.get('test_mode')!='isolated_fault_injection_only':
        errors.append('test mode is not isolated fault injection')
    if summary.get('holdout_used') is not False or summary.get('frozen_prd31_modified') is not False:
        errors.append('frozen source or holdout safety status abnormal')
    if dict(Counter(s.get('status') for s in service))!=summary.get('output_status_counts'):
        errors.append('recomputed service status counts disagree with summary')
    return {'schema':'prd32_p0_gpu_evidence_crosscheck_v1','passed':not errors,'errors':errors,'checked_requests':n,
            'pass_count':pass_n,'blocked_count':blocked_n,'successful_real_inferences':generated,
            'blocked_without_model_call':blocked_without_model,'blocked_with_null_payload':blocked_null,
            'service_status_counts':dict(Counter(s.get('status') for s in service)),
            'error_code_occurrences':dict(sorted(issue_counts.items())),
            'note':'Error-code occurrences overlap across requests and cannot be summed as independent failed cases.',
            'scope':'Real GPU with 3 valid requests; 12 deliberate invalid inputs; not PRD acceptance.'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--summary',required=True)
    ap.add_argument('--decisions',required=True)
    ap.add_argument('--predictions',required=True)
    ap.add_argument('--service',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    src=json.loads(Path(a.summary).read_text('utf8'))
    report=validate(src,read_jsonl(a.decisions),read_jsonl(a.predictions),read_jsonl(a.service))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'crosscheck.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    if report['passed']:
        headline='PRD 3.2 P0 典型异常隔离：开发阶段真实 GPU 验证通过（非正式 PRD 验收）'
    else:headline='证据交叉核验未通过，需要检查源文件'
    text=f'''# {headline}

- 输入请求：{report['checked_requests']}，合法推理成功：{report['successful_real_inferences']}，异常隔离：{report['blocked_count']}。
- 被隔离请求未执行模型推理：{report['blocked_without_model_call']}/{report['blocked_count']}。
- 被隔离请求返回空值 payload：{report['blocked_with_null_payload']}/{report['blocked_count']}。
- 服务输出状态：{json.dumps(report['service_status_counts'],ensure_ascii=False)}。
- 原冻结数据未修改，封存 holdout 未使用；采用已冻结数据回放而非实时运行 3.1。
- 注意：error_counts 中一条请求可能对应多个错误代码；不能将错误计数之和当作独立错误请求数。
- 两条非法 JSON 行在输出里采用占位 case_id，不能据此宣称 NaN 框的专用规则单独通过；应在后续契约测试中独立覆盖。
- 结论边界：仅验证本次注入的典型坏输入和三条真实正常请求，无法证明所有未知异常、8 类属性准确率、QA/文案人工质量或生产延迟已达标。
'''
    if report['errors']:
        text+='\n## 核验错误\n'+'\n'.join('- '+e for e in report['errors'])+'\n'
    (out/'P0_证据核验报告.md').write_text(text,encoding='utf8')
    print('P0_GPU_EVIDENCE_CROSSCHECK_PASS' if report['passed'] else 'P0_GPU_EVIDENCE_CROSSCHECK_FAIL')
    print(json.dumps({k:report[k] for k in ('checked_requests','pass_count','blocked_count','successful_real_inferences','blocked_without_model_call','blocked_with_null_payload')},ensure_ascii=False))
    return 0 if report['passed'] else 2
if __name__=='__main__': sys.exit(main())
