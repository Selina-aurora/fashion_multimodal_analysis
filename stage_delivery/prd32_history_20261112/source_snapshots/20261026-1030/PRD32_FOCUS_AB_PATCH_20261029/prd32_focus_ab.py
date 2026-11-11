#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Opt-in GT-free PRD32 target-focus A/B GPU trial + strict provenance post-gate.

Never reads human review during inference/gating, never edits source or publishes.
Only TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY images, frozen V5 predictions and original helper.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_DEFAULT = Path('/workspace/PRD32_20261029_NextBatch_GTFree')
BASELINE_DEFAULT = Path('/workspace/PRD32_NEXTBATCH_GPU_20261029_run01/qwen_results.jsonl')
TRIAL_DEFAULT = Path('/workspace/PRD32_FOCUS_GPU_20261029_run01')
TASKS = ('tags','vqa','copy')
VQA_QUESTION = '红框内目标服饰的主要可见颜色是什么？'
# No human labels, category hints, class-specific corrections, or example answers.
TARGET_FOCUS = (
    '本次只分析红色矩形指向的一个独立服饰实例，不是在做整张图的服饰清单。'
    '矩形仅提示候选定位，不是服饰边界或类别真值；若它只包含衣物的一小块，'
    '或同时包含多件衣物，请根据框、衣物连续轮廓和穿着关系判断目标，'
    '无法可靠区分则保留不确定性，严禁把相邻服饰的颜色、花纹、领口或类别转移给目标。'
    'pants有分开的裤腿，skirt是无裤腿分叉的独立半身裙；'
    'top与outerwear主要依据独立外层叠穿结构，dress应为上衣与裙身连为一件。'
    '不要固定偏向任何类别。'
)
COPY_FOCUS = (
    '输出只允许描述目标服饰本身，不要出现红框、边框、框内、照片、图片、标注、定位等审核过程用语。'
    '仅描述可从目标可见部分支持的事实；不能根据触感、想象或其他服饰补充属性。'
    '如果目标实例不清晰，不要强行断言类别、款式或材质。'
)


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()


def hash_text(txt: str) -> str:
    return hashlib.sha256(txt.encode('utf-8')).hexdigest()


def readj(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def readjl(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return [json.loads(s) for s in f if s.strip()]


def writej(path,doc):
    Path(path).write_text(json.dumps(doc,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def writejl(path, rows):
    with Path(path).open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')


def verify_frozen_source(source:Path):
    hashes=readj(HERE/'frozen_source_hashes.json')
    for rel, expected in hashes.items():
        fp=Path(source)/rel
        if not fp.is_file():raise ValueError(f'FROZEN_SOURCE_MISSING:{rel}')
        if sha(fp)!=expected:raise ValueError(f'FROZEN_SOURCE_CHANGED:{rel}')
    return hashes


def load_source(source:Path):
    verify_frozen_source(source)
    s=str(source.resolve())
    if s not in sys.path:sys.path.insert(0,s)
    import v5_to_qwen_bridge as bridge
    import prd32_guard as guard
    import next_batch as nxt
    if Path(bridge.__file__).resolve().parent!=source.resolve():raise ValueError('BRIDGE_IMPORT_PATH_MISMATCH')
    return bridge,guard,nxt


def build_prompt(helper, rec, task):
    qrec={'task':task,'query':VQA_QUESTION,'length_chars':[20,80],
          'predicted_attribute_context':[]}
    prompt=helper.make_prompt(qrec,input_mode='marked_full_image',pass_unverified=False)
    prompt=TARGET_FOCUS+'\n'+prompt
    if task=='copy':prompt+='\n'+COPY_FOCUS
    return prompt


def strict_verify(rows):
    """Block misjoined triplets; request id & image/prompt evidence must agree.

    Compatible with legacy outputs; missing evidence is noted, never accepted as verified.
    """
    if not isinstance(rows,list) or not rows:raise ValueError('EMPTY_RESULTS')
    groups=defaultdict(dict)
    missing_evidence=[]
    for r in rows:
        if not isinstance(r,dict):raise ValueError('NON_OBJECT_ROW')
        cid,inst,task=r.get('case_id'),r.get('target_instance_id'),r.get('task')
        if not all(isinstance(v,str) and v and '__' not in v for v in (cid,inst,task)):
            raise ValueError('BAD_ID_OR_TASK')
        if task not in TASKS:raise ValueError('BAD_TASK')
        expected=f'{cid}__{inst}__{task}'
        if r.get('request_id')!=expected:raise ValueError(f'REQUEST_ID_MISMATCH:{expected}')
        if r.get('target_bbox_origin')!='prd31_v5_live_prediction':raise ValueError('UNVERIFIED_BBOX_ORIGIN')
        if r.get('human_label_or_gt_bbox_used') is not False:raise ValueError('GT_LEAKAGE')
        if r.get('publish_allowed') is not False:raise ValueError('UNSAFE_PUBLISH')
        if r.get('status')!='GENERATED_REVIEW_REQUIRED':raise ValueError('NON_SUCCESS_REQUEST')
        key=(cid,inst)
        if task in groups[key]:raise ValueError('DUPLICATE_REQUEST')
        groups[key][task]=r
        if not r.get('question_sha256') and task=='vqa':missing_evidence.append(f'{expected}:QUESTION_NOT_SAVED')
        if not r.get('marked_image_sha256'):
            missing_evidence.append(f'{expected}:MARKED_IMAGE_HASH_NOT_SAVED')
        elif r.get('marked_image') and Path(r['marked_image']).is_file():
            if sha(Path(r['marked_image']))!=r['marked_image_sha256']:
                raise ValueError(f'MARKED_IMAGE_FILE_HASH_MISMATCH:{expected}')
    for key,g in groups.items():
        if set(g)!=set(TASKS):raise ValueError(f'MISSING_TASKS:{key}')
        for field in ('v5_category_unverified','v5_predicted_score_uncalibrated','v5_predicted_bbox_xyxy','marked_image'):
            vals={json.dumps(r.get(field),sort_keys=True) for r in g.values()}
            if len(vals)!=1:raise ValueError(f'INSTANCE_CROSS_TASK_MISMATCH:{key}:{field}')
        if len({r.get('image_sha256') for r in g.values() if r.get('image_sha256')})>1:
            raise ValueError(f'INPUT_IMAGE_HASH_CROSS_TASK_MISMATCH:{key}')
        hashes={r.get('marked_image_sha256') for r in g.values() if r.get('marked_image_sha256')}
        if len(hashes)>1:raise ValueError(f'IMAGE_HASH_CROSS_TASK_MISMATCH:{key}')
        q=g['vqa']
        if q.get('question_text') is not None and hash_text(q['question_text'])!=q.get('question_sha256'):
            raise ValueError(f'QUESTION_HASH_MISMATCH:{key}')
    return {'instances':len(groups),'requests':len(rows),'missing_provenance_evidence':missing_evidence,
            'all_request_ids_bound':True,'all_triplets_complete':True}


def build_duplicate_groups(gate):
    """Do not auto merge: transitive bbox overlap is only a risk candidate."""
    by_case=defaultdict(dict)
    for r in gate['rows']:by_case[r['case_id']][r['instance_id']]=r
    output=[]
    for case,instances in sorted(by_case.items()):
        parent={k:k for k in instances}
        def root(x):
            while parent[x]!=x:
                parent[x]=parent[parent[x]];x=parent[x]
            return x
        for p in gate['duplicate_candidate_pairs']:
            if p['case_id']!=case:continue
            a,b=p['instance_a'],p['instance_b']
            if a not in parent or b not in parent:raise ValueError('DUPLICATE_PAIR_UNKNOWN_INSTANCE')
            parent[root(b)]=root(a)
        components=defaultdict(list)
        for k in instances:components[root(k)].append(k)
        for members in components.values():
            if len(members)<2:continue
            members.sort()
            output.append({'case_id':case,'member_ids':members,'size':len(members),
                           'status':'POSSIBLE_SAME_GARMENT_REVIEW_REQUIRED',
                           'auto_deduplicated':False,'primary_instance_selected':False,
                           'v5_categories':{k:instances[k]['v5_category'] for k in members}})
    return output


def audit_result(rows, bridge, guard, qwen_path):
    strict=strict_verify(rows)
    gate=guard.runtime_gate(rows,threshold=.8)
    if any(r.get('model_variant')=='focus_v1' for r in rows):
        if not all(r.get('model_variant')=='focus_v1' for r in rows):
            raise ValueError('MIXED_MODEL_VARIANTS')
        helper,_=bridge.load_helper()
        expected={task:hash_text(build_prompt(helper,{},task)) for task in TASKS}
        for r in rows:
            if r.get('prompt_sha256')!=expected[r['task']]:
                raise ValueError('FOCUS_PROMPT_HASH_MISMATCH:'+r['request_id'])
            if r['task']=='vqa' and r.get('question_text')!=VQA_QUESTION:
                raise ValueError('FOCUS_VQA_QUESTION_MISMATCH:'+r['request_id'])
    clusters=build_duplicate_groups(gate)
    hard=Counter(x for r in gate['rows'] for x in r['copy_block_reasons'])
    # Conservative service decision: never promote to publish, including class conflicts.
    per_instance=[]
    for r in gate['rows']:
        reasons=list(r['copy_block_reasons'])
        if r['duplicate_candidate']:reasons.append('POSSIBLE_DUPLICATE')
        if 'V5_QWEN_CATEGORY_DISAGREEMENT' in r['flags']:reasons.append('UNRESOLVED_CATEGORY_DISAGREEMENT')
        if 'TAGS_UNPARSEABLE' in r['flags']:reasons.append('UNPARSEABLE_TAGS')
        if not r['qwen_tag_category']:reasons.append('MISSING_QWEN_CATEGORY')
        per_instance.append({'case_id':r['case_id'],'instance_id':r['instance_id'],
            'strict_service_disposition':'BLOCKED_REVIEW_REQUIRED' if reasons else 'REVIEW_REQUIRED',
            'block_reasons':sorted(set(reasons)),'publish_allowed':False})
    out={'schema':'prd32_focus_trial_strict_audit_v1','scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',
         'source_sha256':sha(Path(qwen_path)),'strict_identity':strict,
         'duplicate_candidate_groups':clusters,'per_instance_strict_decisions':per_instance,'gate_counts':gate['counts'],
         'copy_hard_reason_counts':dict(hard),'instances':gate['instances'],'requests':gate['requests'],
         'auto_merge_performed':False,'publish_allowed':False,'official_PRD_acceptance':False,
         'note':'A diagnostic gate. Flag counts do not measure accuracy. Candidate clusters do not select a primary.'}
    return gate,out


def gpu(args):
    source=args.source.resolve();bridge,guard,nxt=load_source(source)
    if args.output.exists():raise FileExistsError('REFUSE_OVERWRITE:'+str(args.output))
    plan=readj(args.plan/'next_batch_plan.json')
    if plan.get('schema')!='prd32_nextbatch_gtfree_plan_v1' or plan.get('scope')!='TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY':
        raise ValueError('BAD_PLAN_SCHEMA_OR_SCOPE')
    if plan.get('tasks')!=list(TASKS) or plan.get('instance_policy')!='EVERY_V5_PREDICTION_IN_CHOSEN_IMAGES':
        raise ValueError('PLAN_INSTANCE_POLICY_CHANGED')
    if plan.get('human_gt_used') is not False:raise ValueError('GT_PLAN_NOT_ALLOWED')
    rec,summary=nxt.load_verified(args.prep,args.raw,args.gate,args.data_root)
    chosen=plan['case_ids']
    if not isinstance(chosen,list) or not chosen or len(chosen)!=len(set(chosen)):raise ValueError('BAD_CASE_IDS')
    if any(c in nxt.PREVIOUS_CASES for c in chosen):raise ValueError('OLD_REVIEW_CASE_EXCLUDED')
    if plan['v5_raw_sha256']!=summary['v5_predictions_sha256'] or plan['prepared_bridge_sha256']!=sha(args.prep/'gt_free_all_instances.jsonl'):
        raise ValueError('V5_OR_PREP_CHANGED')
    helper,helper_file=bridge.load_helper()
    if plan['frozen_prompt_helper_sha256']!=sha(helper_file):raise ValueError('PROMPT_HELPER_CHANGED')
    subset=sorted([r for r in rec if r['case_id'] in chosen],key=lambda r:(r['case_id'],r['target_instance_id']))
    if len({r['case_id'] for r in subset})!=len(chosen) or len(subset)!=plan['n_new_v5_instances'] or len(subset)*3!=plan['n_planned_qwen_requests']:
        raise ValueError('SELECTED_INSTANCE_SET_CHANGED')
    if len(subset)>12 or len(subset)*3>36:raise ValueError('DIAGNOSTIC_REQUEST_CAP_EXCEEDED')
    if not (args.model_dir/'config.json').is_file():raise FileNotFoundError('LOCAL_QWEN_CONFIG_MISSING')
    import os
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
    if torch.cuda.mem_get_info()[0]/1024**3<18:raise RuntimeError('REQUIRES_18_GiB_FREE_GPU_RAM')
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'marked_inputs').mkdir()
    prompt_hashes={t:hash_text(build_prompt(helper,{},t)) for t in TASKS}
    provenance={'schema':'prd32_focus_trial_provenance_v1','mode':'OPT_IN_AB_VARIANT_NOT_BASELINE',
        'scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY','GT_was_not_loaded':True,
        'helper_sha256':sha(helper_file),'focus_script_sha256':sha(Path(__file__)),
        'prompt_hashes':prompt_hashes,'tasks':list(TASKS),'case_ids':chosen,
        'instance_policy':'ALL_INSTANCES_IN_SELECTED_IMAGES','v5_predictions_sha256':summary['v5_predictions_sha256'],
        'publish_allowed':False,'official_PRD_acceptance':False}
    writej(args.output/'smoke_provenance.json',provenance)
    start=time.perf_counter()
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.model_dir),local_files_only=True,device_map='auto',dtype=dtype,attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    proc=AutoProcessor.from_pretrained(str(args.model_dir),local_files_only=True,max_pixels=512*28*28)
    load_s=round(time.perf_counter()-start,2)
    outputs=[];times=[]
    with (args.output/'qwen_results.jsonl').open('w',encoding='utf-8') as f:
        for item in subset:
            for task in TASKS:
                qrec={'image_path':item['image_path'],'image_sha256':item['image_sha256'],
                      'target_bbox_xyxy':item['target_bbox_xyxy'],'target_instance_id':item['target_instance_id'],
                      'task':task,'query':VQA_QUESTION,'length_chars':[20,80],
                      'predicted_attribute_context':[]}
                rid=f"{item['case_id']}__{item['target_instance_id']}__{task}"
                r={'request_id':rid,'case_id':item['case_id'],'task':task,'target_instance_id':item['target_instance_id'],
                   'v5_category_unverified':item['predicted_category_v5_unverified'],
                   'v5_predicted_bbox_xyxy':item['target_bbox_xyxy'],
                   'v5_predicted_score_uncalibrated':item['predicted_score_v5_uncalibrated'],
                   'target_bbox_origin':'prd31_v5_live_prediction','selection_policy':'ALL_INSTANCES_IN_SELECTED_IMAGES',
                   'human_label_or_gt_bbox_used':False,'publish_allowed':False,'quality_human_verified':False,
                   'status':'CASE_FAILED','model_variant':'focus_v1','image_sha256':item['image_sha256'],
                   'question_text':VQA_QUESTION if task=='vqa' else None,
                   'question_sha256':hash_text(VQA_QUESTION) if task=='vqa' else None}
                try:
                    marked=helper.make_marked_image(qrec,args.output/'marked_inputs','marked_full_image',.05)
                    p=build_prompt(helper,qrec,task)
                    raw,timing=helper.run_model_case(model,proc,qrec,marked,p,task_limits={'tags':170,'vqa':128,'copy':120})
                    if not isinstance(raw,str) or not raw.strip():raise ValueError('EMPTY_MODEL_OUTPUT')
                    r.update(status='GENERATED_REVIEW_REQUIRED',raw_text=raw,
                        prompt_sha256=hash_text(p),marked_image=str(marked),marked_image_sha256=sha(marked),timing=timing)
                    times.append(timing['end_to_end_ms'])
                    if task=='tags':
                        parsed=helper.parse_tag_output(raw);r['tag_parse']=parsed
                        if parsed.get('canonical_service_json') is not None:
                            canonical=parsed['canonical_service_json']
                            r['tag_bridge_candidate']=helper.bridge_tags_adapter(canonical)
                            if canonical.get('category') is not None and canonical['category']!=item['predicted_category_v5_unverified']:
                                r['review_flags']=['V5_VS_QWEN_CATEGORY_DISAGREEMENT_NOT_AUTOMATIC_CORRECTION']
                    if task=='vqa':r['vqa_visible_abstention_marker']=any(x in raw for x in ('无法确定','看不清'))
                    if task=='copy':
                        r['copy_review_flags']=[flag for flag,yes in (('DEBUG_MARKER_IN_COPY',bool(re.search('红框|红色.?框|框内|标注框|bbox|bounding box',raw,re.I))),('COPY_EXCEEDS_80_CHARS',len(raw.strip())>80)) if yes]
                except Exception as e:
                    r['error']=type(e).__name__+': '+str(e)
                outputs.append(r)
                f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
                print('FOCUS_QWEN_INSTANCE_DONE',len(outputs),'/',len(subset)*3,rid,r['status'],flush=True)
    summ={'schema':'prd32_gt_free_v5_qwen_smoke_v1','status':'COMPLETED_FOR_REVIEW' if all(r['status']=='GENERATED_REVIEW_REQUIRED' for r in outputs) else 'PARTIAL_FAILURES',
          'n_images':len(chosen),'n_predicted_instances_processed':len(subset),'n_requests':len(outputs),
          'successful_outputs':sum(r['status']=='GENERATED_REVIEW_REQUIRED' for r in outputs),
          'failed_outputs':sum(r['status']!='GENERATED_REVIEW_REQUIRED' for r in outputs),
          'v5_raw_sha256':summary['v5_predictions_sha256'],'model_load_seconds':load_s,
          'median_qwen_request_ms':round(statistics.median(times),2) if times else None,
          'task_counts':dict(Counter(r['task'] for r in outputs)),'target_selection_policy':'ALL_INSTANCES_IN_SELECTED_IMAGES',
          'human_reference_used':False,'publish_allowed':False,'official_prd_acceptance':False,
          'model_variant':'focus_v1'}
    writej(args.output/'gt_free_qwen_smoke_summary.json',summ)
    print('FOCUS_TRIAL_COMPLETED',summ['status'],len(outputs),'requests; no publication',flush=True)
    if summ['failed_outputs']:raise RuntimeError('PARTIAL_FAILURES_INSPECT_RESULTS_DO_NOT_SCORE')


def audit(args):
    bridge,guard,nxt=load_source(args.source.resolve())
    rows=readjl(args.qwen)
    gate,summary=audit_result(rows,bridge,guard,args.qwen)
    if args.output.exists():raise FileExistsError('REFUSE_OVERWRITE:'+str(args.output))
    args.output.mkdir(parents=True)
    writej(args.output/'strict_gate.json',summary)
    writej(args.output/'original_gate_unchanged.json',gate)
    # Reuse the untouched original review renderer; no labels are pre-filled.
    review_html=args.output/'Review_Focus_Or_Baseline.html'
    nxt.make_review_page(args.qwen.parent,rows,gate['rows'],review_html)
    html=review_html.read_text(encoding='utf-8')
    html=html.replace('<h1>PRD 3.2 新批次 · 全预测实例逐图审核</h1>',
        '<h1>PRD 3.2 A/B 开发诊断 · 逐实例人工审核</h1>'
        '<p>VQA 固定开发问题（新版结果另有 question_text + SHA）：'
        +VQA_QUESTION+'</p><p>此页导出仍是人工审核表，未人工确认前不得当作评测标签。</p>')
    review_html.write_text(html,encoding='utf-8')
    print('PRD32_STRICT_GATE_OK',json.dumps({'instances':summary['instances'],'counts':summary['gate_counts'],
         'duplicate_groups':len(summary['duplicate_candidate_groups']),
         'missing_evidence':len(summary['strict_identity']['missing_provenance_evidence'])},ensure_ascii=False),flush=True)


def compare(args):
    bridge,guard,nxt=load_source(args.source.resolve())
    old=readjl(args.baseline);new=readjl(args.trial)
    _,b=audit_result(old,bridge,guard,args.baseline)
    _,n=audit_result(new,bridge,guard,args.trial)
    bo={(r['case_id'],r['target_instance_id'],r['task']):r for r in old}
    no={(r['case_id'],r['target_instance_id'],r['task']):r for r in new}
    if set(bo)!=set(no):raise ValueError('BASELINE_TRIAL_REQUEST_SET_MISMATCH')
    changes=[]
    for key in sorted(bo):
        a,c=bo[key],no[key]
        for field in ('v5_category_unverified','v5_predicted_bbox_xyxy','v5_predicted_score_uncalibrated'):
            if a.get(field)!=c.get(field):raise ValueError(f'V5_PREDICTION_CHANGED:{key}:{field}')
        if a.get('raw_text')!=c.get('raw_text'):
            changes.append({'case_id':key[0],'instance_id':key[1],'task':key[2],
                            'baseline_output':a.get('raw_text'),'focus_output':c.get('raw_text')})
    res={'schema':'prd32_dev_focus_ab_comparison_v1','scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',
         'same_instance_task_set':True,'same_v5_prediction_inputs':True,
         'baseline_sha256':sha(args.baseline),'focus_sha256':sha(args.trial),
         'baseline_gate_counts':b['gate_counts'],'focus_gate_counts':n['gate_counts'],
         'baseline_duplicate_groups':len(b['duplicate_candidate_groups']),
         'focus_duplicate_groups':len(n['duplicate_candidate_groups']),
         'outputs_changed':len(changes),'per_request_changes':changes,
         'human_accuracy_not_measured':True,'human_re_review_required':True,
         'publish_allowed':False,'official_PRD_acceptance':False}
    if args.output.exists():raise FileExistsError('REFUSE_OVERWRITE:'+str(args.output))
    args.output.mkdir(parents=True)
    writej(args.output/'focus_ab_comparison.json',res)
    print('FOCUS_AB_COMPARISON_OK',json.dumps({'changes':len(changes),'baseline':b['gate_counts'],'focus':n['gate_counts']},ensure_ascii=False),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    for name in ('audit','gpu','compare'):
        s=sub.add_parser(name);s.add_argument('--source',type=Path,default=SOURCE_DEFAULT)
        s.add_argument('--output',type=Path,required=True)
        if name=='audit':s.add_argument('--qwen',type=Path,required=True)
        if name=='compare':
            s.add_argument('--baseline',type=Path,default=BASELINE_DEFAULT)
            s.add_argument('--trial',type=Path,default=TRIAL_DEFAULT/'qwen_results.jsonl')
        if name=='gpu':
            s.add_argument('--plan',type=Path,default=Path('/workspace/PRD32_NEXTBATCH_PREP_20261029_run01'))
            s.add_argument('--raw',type=Path,default=Path('/workspace/PRD32_V5_REAL_20261029_run01'))
            s.add_argument('--gate',type=Path,default=Path('/workspace/PRD32_V5_LIVE_PREFLIGHT_20261029_run01'))
            s.add_argument('--data-root',type=Path,default=Path('/workspace/fashion_data'))
            s.add_argument('--prep',type=Path,default=Path('/workspace/PRD32_V5_QWEN_PREP_20261029_run01'))
            s.add_argument('--model-dir',type=Path,default=Path('/workspace/models/Qwen2.5-VL-7B-Instruct'))
    args=p.parse_args(argv)
    try:
        {'gpu':gpu,'audit':audit,'compare':compare}[args.cmd](args)
    except Exception as exc:
        print('PRD32_FOCUS_BLOCKED',type(exc).__name__+': '+str(exc),file=sys.stderr)
        raise SystemExit(2)


if __name__=='__main__':main()
