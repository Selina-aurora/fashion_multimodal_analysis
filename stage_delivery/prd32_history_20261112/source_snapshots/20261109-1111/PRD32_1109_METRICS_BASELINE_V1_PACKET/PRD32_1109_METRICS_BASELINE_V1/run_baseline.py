#!/usr/bin/env python3
"""PRD 3.2 fixed-DEV60 baseline audit, purely offline and fail-closed.

Reads only explicitly named historical Qwen JSONL files. Does not call models,
scan directories, access holdout, overwrite outputs, or infer missing human GT.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

TASKS = ('tags', 'vqa', 'copy')
VARIANTS = ('baseline', 'focus_v1')
ALLOWED_CLASSES = {'top', 'pants', 'skirt', 'outerwear', 'dress', 'shoe', 'bag', 'accessory'}
FORBIDDEN = ('holdout', 'sealed', 'blind', 'independent_test')
BASE = Path(__file__).resolve().parent


def is_forbidden(path: Path) -> bool:
    return any(s in str(path).lower() for s in FORBIDDEN)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    data=[]
    with path.open('r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            row=json.loads(line)
            if not isinstance(row,dict):
                raise ValueError(f'Non-object JSONL at {path}:{i}')
            data.append(row)
    return data


def load_frozen() -> dict[tuple[str,str],dict]:
    with (BASE/'inputs/frozen_dev60_shadow_171.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=171:
        raise ValueError('Frozen DEV60 expected exactly 171 instances')
    result={}
    for r in rows:
        k=(r['case_id'],r['instance_id'])
        if k in result:
            raise ValueError(f'Duplicate frozen key {k}')
        result[k]=r
    if len({k[0] for k in result})!=60:
        raise ValueError('Frozen DEV60 expected exactly 60 images')
    return result


def load_confirmed() -> dict[tuple[str,str],dict]:
    with (BASE/'inputs/user_confirmed12_category_only.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=12:
        raise ValueError('Expected 12 chat-confirmed prior review rows')
    result={}
    for r in rows:
        k=(r['case_id'],r['instance_id'])
        if k in result:
            raise ValueError(f'Duplicate review key {k}')
        if r['review_status']!='USER_CONFIRMED_IN_CHAT':
            raise ValueError(f'Unconfirmed review cannot serve as category check: {k}')
        result[k]=r
    if sum(r['category_gt_is_determinate']=='YES' for r in result.values())!=9:
        raise ValueError('Expected nine determinate category labels, do not force unresolved GT')
    return result


def parsed_tag(row:dict)->dict:
    tag=row.get('tag_parse')
    if isinstance(tag,dict):
        x=tag.get('canonical_service_json')
        if tag.get('parseable') is True and tag.get('schema_valid') is True and isinstance(x,dict):
            return x
    # A live runtime result may have raw JSON without tag_parse. Historical
    # DEV60 logs are expected to have tag_parse, so do not invent labels here.
    return {}


def finite_timing(row:dict, field:str)->float|None:
    x=row.get('timing',{}).get(field) if isinstance(row.get('timing'),dict) else None
    if isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x) and x>=0:
        return float(x)
    return None


def percentile(xs:list[float],percent:float)->float|None:
    if not xs:return None
    ys=sorted(xs)
    rank=(len(ys)-1)*percent
    a=int(rank); b=min(a+1,len(ys)-1)
    return ys[a]+(ys[b]-ys[a])*(rank-a)


def describe(xs:list[float],threshold:float|None=None)->dict:
    if not xs:
        return {'n':0,'mean_ms':None,'median_ms':None,'p95_ms':None,'max_ms':None,'count_le_threshold':None}
    return {'n':len(xs),'mean_ms':round(statistics.mean(xs),3),
            'median_ms':round(percentile(xs,.5),3),'p95_ms':round(percentile(xs,.95),3),
            'max_ms':round(max(xs),3),
            'count_le_threshold':(sum(x<=threshold for x in xs) if threshold is not None else None)}


def validate_records(rows:list[dict],expected_variant:str,frozen:dict, strict_full:bool)->dict:
    index={};violations=Counter();cases=Counter()
    for i,r in enumerate(rows,1):
        k=(r.get('case_id'),r.get('target_instance_id'))
        task=r.get('task')
        if k not in frozen:raise ValueError(f'Unfrozen DEV60 instance at record {i}: {k}')
        if task not in TASKS:raise ValueError(f'Unknown task at record {i}: {task}')
        if r.get('model_variant')!=expected_variant:raise ValueError(f'Wrong variant at record {i}')
        key=(k,task)
        if key in index:raise ValueError(f'Duplicate instance-task record at record {i}: {key}')
        if r.get('image_sha256')!=frozen[k]['image_sha256']:
            raise ValueError(f'Image SHA mismatch at record {i}: {k}')
        if r.get('request_id')!=f'{k[0]}__{k[1]}__{task}':
            raise ValueError(f'Request ID mismatch at record {i}: {k} {task}')
        if r.get('publish_allowed') is not False:
            raise ValueError(f'Publish flag is not false at record {i}: {k} {task}')
        if r.get('status')!='GENERATED_REVIEW_REQUIRED':
            violations['unexpected_status']+=1
        if task=='tags' and not parsed_tag(r):violations['invalid_or_missing_tag_parse']+=1
        if not isinstance(r.get('raw_text'),str):violations['missing_response_text']+=1
        for name in ('preprocess_ms','generation_ms','end_to_end_ms'):
            if finite_timing(r,name) is None:violations['missing_or_bad_'+name]+=1
        if any(x in str(r.get('marked_image','')).lower() for x in FORBIDDEN):
            raise ValueError(f'Forbidden referenced source in {k}')
        index[key]=r
        cases[k]+=1
    incomplete=sum(len([task for task in TASKS if (k,task) not in index]) for k in frozen)
    if strict_full and (len(index)!=513 or len(cases)!=171 or incomplete):
        raise ValueError(f'Incomplete full DEV60 {expected_variant}: {len(index)} tasks, {len(cases)} instances, {incomplete} missing')
    return {'index':index,'unique_instances':len(cases),'record_count':len(index),'missing_task_count':incomplete,'issues':dict(violations)}


def category_eval(index:dict,review:dict)->dict:
    per=[]
    for key,r in review.items():
        if r['category_gt_is_determinate']!='YES':continue
        item=index.get((key,'tags'))
        if item is None:continue
        pred=parsed_tag(item).get('category')
        per.append({'case_id':key[0],'instance_id':key[1], 'human_category':r['category_gt'],
                    'qwen_category':pred or '', 'correct':int(pred==r['category_gt'])})
    return {'denominator':len(per),'correct':sum(x['correct'] for x in per),
            'category_only_accuracy':(round(sum(x['correct'] for x in per)/len(per),4) if per else None),
            'selection':'Nine previously user-confirmed category determinate HIGH-RISK instances, not random; NOT full attributes or PRD acceptance',
            'rows':per}


def evaluate(data:dict, frozen:dict, reviewed:dict, mode:str, sources:dict)->tuple[dict,list[dict],list[dict]]:
    metrics={}; review_rows=[]; comparison=[]
    indices={v:data[v]['index'] for v in VARIANTS}
    for variant in VARIANTS:
        idx=indices[variant]
        timings={}
        for task in TASKS:
            taskrows=[r for (k,t),r in idx.items() if t==task]
            timings[task]={
                'generation_ms':describe([x for r in taskrows if (x:=finite_timing(r,'generation_ms')) is not None],200 if task=='vqa' else None),
                'end_to_end_ms':describe([x for r in taskrows if (x:=finite_timing(r,'end_to_end_ms')) is not None]),
            }
        tags=[r for (k,t),r in idx.items() if t=='tags']
        vqa=[r for (k,t),r in idx.items() if t=='vqa']
        serial=[]; max_task=[]
        for key in frozen:
            if (key,'tags') not in idx or (key,'copy') not in idx:continue
            a=finite_timing(idx[(key,'tags')],'end_to_end_ms')
            b=finite_timing(idx[(key,'copy')],'end_to_end_ms')
            if a is not None and b is not None:
                serial.append(a+b);max_task.append(max(a,b))
        metrics[variant]={
            'records':data[variant]['record_count'],
            'instances_covered':data[variant]['unique_instances'],
            'missing_task_count':data[variant]['missing_task_count'],
            'validation_issues':data[variant]['issues'],
            'tag_schema_valid':sum(bool(parsed_tag(r)) for r in tags),
            'tag_count':len(tags),
            'vqa_visible_abstentions':sum(r.get('vqa_visible_abstention_marker') is True for r in vqa),
            'vqa_count':len(vqa),
            'task_timings':timings,
            'single_product_timing_proxies':{
                'sum_tags_copy_e2e_ms_sequential_ONLY_NOT_OBSERVED':describe(serial),
                'max_tags_copy_e2e_ms_MINIMUM_OF_OBSERVED_COMPONENTS_NOT_WALL_CLOCK':describe(max_task,100),
            },
            'category_only_user_confirmed_high_risk':category_eval(idx,reviewed),
        }
    for key,f in frozen.items():
        one={'case_id':key[0],'instance_id':key[1], 'image_sha256':f['image_sha256'],'v5_category_unverified':f['v5_category']}
        rev=reviewed.get(key)
        for variant in VARIANTS:
            t=indices[variant].get((key,'tags'))
            o=parsed_tag(t) if t else {}
            one[variant+'_category']=o.get('category','')
            one[variant+'_tag_schema_valid']=int(bool(o))
            one[variant+'_vqa_ms']=finite_timing(indices[variant][(key,'vqa')],'generation_ms') if (key,'vqa') in indices[variant] else ''
        one['baseline_focus_category_disagree']=int(bool(one['baseline_category'] and one['focus_v1_category'] and one['baseline_category']!=one['focus_v1_category']))
        one['category_human_confirmed']=rev['category_gt'] if rev and rev['category_gt_is_determinate']=='YES' else ''
        one['category_gt_unresolved']=int(bool(rev and rev['category_gt_is_determinate']!='YES'))
        comparison.append(one)
        base=indices['baseline']
        tag=base.get((key,'tags'));vqa=base.get((key,'vqa'));copy=base.get((key,'copy'))
        td=parsed_tag(tag) if tag else {}
        review_rows.append({
            'case_id':key[0],'instance_id':key[1], 'image_sha256':f['image_sha256'],
            'marked_baseline_image_path':str(tag.get('marked_image','')) if tag else '',
            'v5_category_UNVERIFIED':f['v5_category'],
            'baseline_category':td.get('category',''), 'baseline_color':td.get('primary_color',''),
            'baseline_pattern':td.get('pattern',''), 'baseline_design':td.get('visible_design',''),
            'vqa_question':str(vqa.get('question_text','')) if vqa else '',
            'vqa_prediction':str(vqa.get('raw_text','')) if vqa else '',
            'copy_prediction':str(copy.get('raw_text','')) if copy else '',
            'category_GT_if_confirmed':(rev['category_gt'] if rev and rev['category_gt_is_determinate']=='YES' else ''),
            'category_GT_status':('CONFIRMED_DETERMINATE' if rev and rev['category_gt_is_determinate']=='YES' else 'CONFIRMED_UNRESOLVED' if rev else 'UNREVIEWED'),
            'VISIBLE_COLOR_GT_TO_FILL':'','VQA_CORRECT_1_0_OR_NA_TO_FILL':'',
            'FULL_TAGS_CORRECT_1_0_OR_NA_TO_FILL':'',
            'COPY_SATISFACTION_1_TO_5_OR_NA_TO_FILL':'',
            'NEW_REVIEWER_CONFIRMED_YES_NO_TO_FILL':'NO',
            'REVIEW_NOTES_TO_FILL':'',
        })
    summary={
        'schema':'prd32_metrics_baseline_v1', 'mode':mode,
        'date':'2026-11-09','status':'HISTORICAL_OFFLINE_METRICS_NOT_PRD_ACCEPTANCE',
        'scope':{'frozen_images':60,'frozen_instances':171,
                 'this_run_is_full_DEV60':mode=='GPU_DEV60_FULL',
                 'human_category_gt_determinate_total':9,
                 'human_full_tag_gt_available':False,'human_vqa_gt_available':False,
                 'human_copy_satisfaction_available':False},
        'sources':sources,
        'metrics':metrics,
        'prd_targets':{
            'vqa_accuracy':'>=0.85 (NOT EVALUABLE: no independent VQA GT)',
            'vqa_generation_time_ms':'<=200 (historical per-task generation_ms; not full service latency)',
            'full_attribute_tag_accuracy':'>=0.90 (NOT EVALUABLE: full field GT missing)',
            'copy_human_satisfaction':'>=0.85 (NOT EVALUABLE: human rubric missing)',
            'single_product_generation_ms':'<=100 (NOT directly measured as product-level wall clock)',
        },
        'source_predictions_modified':False,'holdout_opened':False,'model_called':False,
        'production_approved':False,
    }
    return summary,comparison,review_rows


def csv_write(path:Path, rows:list[dict]):
    if not rows:raise ValueError('No rows to save')
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def render_markdown(s:dict)->str:
    title='FULL DEV60 historical records' if s['mode']=='GPU_DEV60_FULL' else 'TARGETED 32 high-risk instance fixture'
    lines=[
        '# PRD 3.2 Baseline Audit — 2026-11-09', '',
        f'**Scope:** {title}. Offline analysis only; not independent blind-test acceptance.',
        '', '## Task coverage and generation timing', '',
        '| Variant | Task | Records | Mean gen (ms) | P50 (ms) | P95 (ms) | Mean task E2E (ms) |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for v in VARIANTS:
        m=s['metrics'][v]
        for t in TASKS:
            a=m['task_timings'][t]['generation_ms'];b=m['task_timings'][t]['end_to_end_ms']
            lines.append(f"| {v} | {t} | {a['n']} | {a['mean_ms']} | {a['median_ms']} | {a['p95_ms']} | {b['mean_ms']} |")
    lines+=['','**Interpretation:** generation_ms excludes image preprocessing but includes decoding. Task E2E is logged per task; neither number is the complete user-facing service latency. The tags and copy tasks are different outputs. Their summed latency is only a serial proxy, not observed wall-clock product latency.',
            '', '## Existing nine category-only human confirmations (biased, high-risk sample)', '',
            '| Variant | Correct | Evaluated | Proxy accuracy |','|---|---:|---:|---:|']
    for v in VARIANTS:
        d=s['metrics'][v]['category_only_user_confirmed_high_risk']
        lines.append(f"| {v} | {d['correct']} | {d['denominator']} | {d['category_only_accuracy']} |")
    lines+=['', '**Not the PRD 90% full-attribute benchmark.** The nine labels are high-risk, non-random, and cover garment category only; no trustworthy color/pattern/design GT is available in this dataset.',
            '', '## VQA timing threshold (historical generation <=200 ms)','',
            '- VQA accuracy >=85%: **NOT MEASURABLE** without independent question-answer GT. Historical VQA questions mainly ask visible color; a full question-type benchmark is still required.',
            '- VQA generation <=200ms: measurable on the historical subset; report distribution and pass share, not only average.',
            '- Full attribute tags accuracy >=90%: **NOT MEASURABLE** until all required label fields have GT.',
            '- Copy human satisfaction >=85%: **NOT MEASURABLE** until an independently reviewed rubric is completed (suggested 4–5 / 5 as satisfactory, pending supervisor approval).',
            '- Single-product generation <=100ms: **NOT directly measured** in these independent tags / copy logs. Measure full product generation wall time after integration.',
            '', '## Next step','',
            'Fill manual_review_baseline_171.csv with human review of target GT, full tags, VQA responses, and copy satisfaction. Keep uncertain labels as NA; do not treat a model output as GT. Do not use Holdout until the candidate is frozen and authorized for blind testing.',
            '', 'DEV60 is a development set used repeatedly. These results do not establish unseen-data performance.',
           ]
    extra=['| Variant | Generation <=200 ms | Total VQA calls | Share |', '|---|---:|---:|---:|']
    for v in VARIANTS:
        x=s['metrics'][v]['task_timings']['vqa']['generation_ms']
        passed=x['count_le_threshold'];total=x['n']
        extra.append(f'| {v} | {passed} | {total} | {round(passed/total,4) if total else "NA"} |')
    at=lines.index('## VQA timing threshold (historical generation <=200 ms)')+2
    lines[at:at]=extra+['', '## PRD acceptance gaps', '']
    return '\n'.join(lines)+'\n'


def run(*, mode:str, baseline:Path|None, focus:Path|None, fixture:Path|None,out:Path)->dict:
    if is_forbidden(out):raise ValueError('Output cannot target Holdout/Blind/Sealed path')
    if out.exists():raise FileExistsError(f'Output exists; refusing overwrite: {out}')
    frozen=load_frozen();reviewed=load_confirmed()
    if mode=='GPU_DEV60_FULL':
        if baseline is None or focus is None:raise ValueError('Both DEV60 paths required')
        for p in (baseline,focus):
            if is_forbidden(p):raise ValueError(f'Holdout/Blind/Sealed input blocked: {p}')
            if 'PRD32_DEV60_QWEN_' not in str(p).upper() or p.name!='qwen_results.jsonl':
                raise ValueError(f'Only explicit DEV60 Qwen historical logs supported: {p}')
        if baseline.resolve()==focus.resolve():raise ValueError('Same baseline and focus path')
        expected=json.loads((BASE/'inputs/frozen_shadow_summary.json').read_text(encoding='utf-8'))['qwen_input_sha256']
        actual={'baseline':file_sha(baseline),'focus_v1':file_sha(focus)}
        for v in VARIANTS:
            if actual[v]!=expected[v]:raise ValueError(f'Immutable DEV60 historical log SHA differs for {v}: {actual[v]}')
        records={'baseline':read_jsonl(baseline),'focus_v1':read_jsonl(focus)}
        strict=True;source={'baseline_sha256':actual['baseline'],'focus_sha256':actual['focus_v1']}
    elif mode=='TARGETED_FIXTURE':
        if fixture is None or is_forbidden(fixture):raise ValueError('Safe fixture path required')
        all_rows=read_jsonl(fixture)
        records={v:[r for r in all_rows if r.get('model_variant')==v] for v in VARIANTS}
        strict=False;source={'fixture_sha256':file_sha(fixture),'derived_from':'32 prior high-risk instances; incomplete DEV60 by design'}
    else:raise ValueError(f'Unknown mode: {mode}')
    data={v:validate_records(records[v],v,frozen,strict) for v in VARIANTS}
    summary,comparison,review_rows=evaluate(data,frozen,reviewed,mode,source)
    out.mkdir(parents=True,exist_ok=False)
    (out/'baseline_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'baseline_report.md').write_text(render_markdown(summary),encoding='utf-8')
    csv_write(out/'per_instance_comparison_171.csv',comparison)
    csv_write(out/'manual_review_baseline_171.csv',review_rows)
    print(f"BASELINE_AUDIT_COMPLETE {mode} baseline={data['baseline']['record_count']} focus={data['focus_v1']['record_count']}")
    print(f'RESULTS_DIR {out.resolve()}')
    print('NOTE: NOT_PRD_ACCEPTANCE; NO_MODEL_CALLS; NO_HOLDOUT; NO_RELEASE')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',action='store_true',help='Local 32 high-risk instances fixture, not full DEV60')
    p.add_argument('--baseline',type=Path,default=Path('/workspace/PRD32_DEV60_QWEN_BASELINE_20261030_run01/qwen_results.jsonl'))
    p.add_argument('--focus',type=Path,default=Path('/workspace/PRD32_DEV60_QWEN_FOCUS_V1_20261030_run01/qwen_results.jsonl'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    run(mode='TARGETED_FIXTURE' if a.fixture else 'GPU_DEV60_FULL',baseline=a.baseline,focus=a.focus,
        fixture=BASE/'fixtures/targeted32_192.jsonl' if a.fixture else None,out=a.out)

if __name__=='__main__':main()
