#!/usr/bin/env python3
# coding: utf-8
"""PRD 3.2 fixed-cohort, ground-truth-free paired Baseline/Focus V1 regression.

No source edits, no GT in inference, no automatic strategy promotion, no holdout.
Derived from the user-supplied 2026-10-29 helper/Focus pipeline contracts.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

TASKS = ('tags', 'vqa', 'copy')
DEFAULT_FREEZE = Path('/workspace/PRD32_FIXED_DEV_FREEZE_20261030')
DEFAULT_SOURCE = Path('/workspace/PRD32_20261029_NextBatch_GTFree')
DEFAULT_FOCUS_SCRIPT = Path('/workspace/PRD32_FOCUS_AB_PATCH_20261029/prd32_focus_ab.py')
DEFAULT_PREP = Path('/workspace/PRD32_V5_QWEN_PREP_20261029_run01')
DEFAULT_RAW = Path('/workspace/PRD32_V5_REAL_20261029_run01')
DEFAULT_MODEL = Path('/workspace/models/Qwen2.5-VL-7B-Instruct')
QUESTION = '红框内目标服饰的主要可见颜色是什么？'
EXPECTED = {'v5_to_qwen_bridge.py':'affee7342646d808b4f0214f135d60d2d2611db920655c5e064268092e2dd02d',
'next_batch.py':'08b35135223b49e2a2a7bbb448cdad3a4ce90a6a1373e57af29d1119102489f6',
'prd32_guard.py':'deeaaea28d9f5e80805da013ef23689fc0df95ee8a6514c583beb468b766fc0f',
'frozen_helpers/prd32_integrated_pipeline_20261027.py':'90dda6974dc533aaae30fe1343095864ee74d536713ad80ce2006617286fa6b4'}
DEBUG = re.compile(r'红框|红色.?框|红色边框|框内|标注框|框选|\bbbox\b|bounding\s+box',re.I)
ABSTAIN = re.compile(r'无法确定|无法判断|看不清|无法辨认|不能确定|不确定')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()

def text_sha(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def jread(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def jlread(p):
    with Path(p).open(encoding='utf-8-sig') as f:return [json.loads(line) for line in f if line.strip()]
def jwrite(p,obj):Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ValueError('CANNOT_LOAD_MODULE:'+str(path))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def frozen_inputs(freeze,source,focus_script,prep,raw,verify_images=True):
    f=jread(freeze/'freeze.json')
    if f.get('schema')!='prd32_20261030_fixed_dev_freeze_v1':raise ValueError('BAD_FREEZE_SCHEMA')
    if f.get('scope')!='TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY' or f.get('holdout_used') is not False or f.get('gt_or_human_labels_in_inference') is not False:raise ValueError('UNSAFE_FREEZE_SCOPE')
    for rel,sha in EXPECTED.items():
        fp=source/rel
        if not fp.is_file() or digest(fp)!=sha or f['frozen_original_source_sha256'].get(rel)!=sha:raise ValueError('SOURCE_CHANGED:'+rel)
    if f.get('focus_trial_script_sha256')!=digest(focus_script):raise ValueError('FOCUS_SCRIPT_CHANGED')
    if digest(freeze/'frozen_v5_instances.jsonl')!=f['frozen_instance_manifest_sha256']:raise ValueError('INSTANCE_MANIFEST_CHANGED')
    if digest(freeze/'frozen_request_ids.jsonl')!=f['request_manifest_sha256']:raise ValueError('REQUEST_MANIFEST_CHANGED')
    if digest(prep/'gt_free_all_instances.jsonl')!=f['prepared_instance_manifest_sha256']:raise ValueError('ORIGINAL_PREP_CHANGED')
    if digest(raw/'v5_raw_predictions.jsonl')!=f['v5_sha256']:raise ValueError('V5_RAW_CHANGED')
    rows=jlread(freeze/'frozen_v5_instances.jsonl')
    requests=jlread(freeze/'frozen_request_ids.jsonl')
    if rows!=sorted(rows,key=lambda r:(r['case_id'],r['target_instance_id'])):raise ValueError('INSTANCE_ORDER_CHANGED')
    if len(rows)!=f['instances'] or len(set(r['case_id'] for r in rows))!=f['images'] or len(rows)!=31 or f['images']!=12:raise ValueError('COHORT_SIZE_CHANGED')
    if len(requests)!=len(rows)*3 or f.get('tasks')!=list(TASKS) or len(requests)!=93:raise ValueError('REQUEST_COUNT_CHANGED')
    known={(r['case_id'],r['target_instance_id']):r for r in rows}
    if len(known)!=len(rows):raise ValueError('DUPLICATE_INSTANCES')
    expected_ids={f'{r["case_id"]}__{r["target_instance_id"]}__{task}' for r in rows for task in TASKS}
    actual_ids={r['request_id'] for r in requests}
    if len(actual_ids)!=len(requests) or actual_ids!=expected_ids:raise ValueError('REQUEST_SET_CHANGED')
    if f.get('qwen_vqa_question')!=QUESTION or f.get('vqa_question_sha256')!=text_sha(QUESTION):raise ValueError('QUESTION_CHANGED')
    for q in requests:
        if q['task'] not in TASKS or q.get('vqa_question')!=(QUESTION if q['task']=='vqa' else None):raise ValueError('QUESTION_OR_TASK_MISMATCH')
    from pathlib import Path as P
    seen_images={}
    for r in rows:
        if r.get('target_bbox_origin')!='prd31_v5_live_prediction' or r.get('human_reference_used') is not False:raise ValueError('GT_ORIGIN_IN_INSTANCE')
        if r.get('scope')!='TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY':raise ValueError('BAD_INSTANCE_SCOPE')
        path=P(r['image_path'])
        if not path.is_absolute() or any(x.lower() in {'holdout','sealed','validation','test','testing','testset','eval'} for x in path.parts):raise ValueError('RESTRICTED_IMAGE_PATH')
        if verify_images:
            if not path.is_file():raise FileNotFoundError('IMAGE_NOT_FOUND:'+str(path))
            if r['image_sha256']!=seen_images.setdefault(str(path),digest(path)):raise ValueError('IMAGE_HASH_CHANGED:'+str(path))
    return f,rows

def load_runtime(source,focus_script):
    bridge=import_file('prd32_bridge_1030',source/'v5_to_qwen_bridge.py')
    helper,_=bridge.load_helper()
    focus=import_file('prd32_focus_1030',focus_script)
    return bridge,helper,focus

def build_prompt(helper,focus,record,task,variant):
    q={'task':task,'query':QUESTION,'length_chars':[20,80],'predicted_attribute_context':[]}
    if variant=='baseline':return helper.make_prompt(q,input_mode='marked_full_image',pass_unverified=False)
    return focus.build_prompt(helper,q,task)

def check(args):
    f,rows=frozen_inputs(args.freeze,args.source,args.focus_script,args.prep,args.raw)
    _,helper,focus=load_runtime(args.source,args.focus_script)
    prompts={v:{t:text_sha(build_prompt(helper,focus,{},t,v)) for t in TASKS} for v in ('baseline','focus_v1')}
    print('FIXED_COHORT_PREFLIGHT_OK',f['images'],'images',len(rows),'instances',len(rows)*3,'requests per variant')
    print('PROMPT_SHA256',json.dumps(prompts,ensure_ascii=False))
    print('ALL_IMAGE_SHA256_VERIFIED; SOURCE_AND_REQUEST_MANIFEST_VERIFIED')
    print('NOT_INDEPENDENT_NOT_OFFICIAL_ACCEPTANCE')
    return f,rows

def run(args):
    if args.output.exists():raise FileExistsError('REFUSE_OVERWRITE:'+str(args.output))
    f,rows=check(args)
    if not (args.model_dir/'config.json').is_file():raise FileNotFoundError('MODEL_CONFIG_MISSING')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
    if torch.cuda.mem_get_info()[0]/1024**3<18:raise RuntimeError('REQUIRES_AT_LEAST_18_GIB_FREE_GPU')
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    random.seed(20261030)
    torch.manual_seed(20261030)
    torch.cuda.manual_seed_all(20261030)
    _,helper,focus=load_runtime(args.source,args.focus_script)
    psha={task:text_sha(build_prompt(helper,focus,{},task,args.variant)) for task in TASKS}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'marked_inputs').mkdir()
    jwrite(args.output/'provenance.json',{'schema':'prd32_fixed_dev_run_20261030_v1','variant':args.variant,
       'scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY','freeze_sha256':digest(args.freeze/'freeze.json'),
       'instances_sha256':f['frozen_instance_manifest_sha256'],'requests_sha256':f['request_manifest_sha256'],
       'v5_sha256':f['v5_sha256'],'focus_script_sha256':f['focus_trial_script_sha256'],
       'prompt_hashes':psha,'vqa_question_sha256':text_sha(QUESTION),'images':12,'instances':31,
       'planned_requests':93,'no_gt_in_inference':True,'holdout_used':False,'publish_allowed':False,'official_acceptance':False})
    start=time.perf_counter()
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.model_dir),local_files_only=True,
       device_map='auto',torch_dtype=dtype,attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    proc=AutoProcessor.from_pretrained(str(args.model_dir),local_files_only=True,max_pixels=512*28*28)
    load_s=round(time.perf_counter()-start,2)
    good=0;failed=0;times=[]
    with (args.output/'qwen_results.jsonl').open('w',encoding='utf-8') as out:
        for rec in rows:
            for task in TASKS:
                rid=f'{rec["case_id"]}__{rec["target_instance_id"]}__{task}'
                qrec={'image_path':rec['image_path'],'image_sha256':rec['image_sha256'],
                      'target_bbox_xyxy':rec['target_bbox_xyxy'],'target_instance_id':rec['target_instance_id'],
                      'task':task,'query':QUESTION,'length_chars':[20,80],'predicted_attribute_context':[]}
                record={'case_id':rec['case_id'],'target_instance_id':rec['target_instance_id'],
                    'request_id':rid,'task':task,'model_variant':args.variant,
                    'status':'CASE_FAILED','v5_category_unverified':rec['predicted_category_v5_unverified'],
                    'v5_predicted_bbox_xyxy':rec['target_bbox_xyxy'],
                    'v5_predicted_score_uncalibrated':rec['predicted_score_v5_uncalibrated'],
                    'target_bbox_origin':'prd31_v5_live_prediction','image_sha256':rec['image_sha256'],
                    'selection_policy':'ALL_FROZEN_V5_PREDICTED_INSTANCES','human_label_or_gt_bbox_used':False,
                    'quality_human_verified':False,'publish_allowed':False,
                    'question_text':QUESTION if task=='vqa' else None,
                    'question_sha256':text_sha(QUESTION) if task=='vqa' else None}
                try:
                    marked=helper.make_marked_image(qrec,args.output/'marked_inputs','marked_full_image',.05)
                    prompt=build_prompt(helper,focus,qrec,task,args.variant)
                    raw,timing=helper.run_model_case(model,proc,qrec,marked,prompt,task_limits={'tags':170,'vqa':128,'copy':120})
                    if not isinstance(raw,str) or not raw.strip():raise ValueError('EMPTY_OUTPUT')
                    record.update(status='GENERATED_REVIEW_REQUIRED',raw_text=raw,timing=timing,
                       marked_image=str(marked),marked_image_sha256=digest(marked),prompt_sha256=text_sha(prompt))
                    if task=='tags':
                        parsed=helper.parse_tag_output(raw);record['tag_parse']=parsed
                        if parsed.get('canonical_service_json') is not None:
                            canonical=parsed['canonical_service_json']
                            record['tag_bridge_candidate']=helper.bridge_tags_adapter(canonical)
                            if canonical.get('category') is not None and canonical['category']!=rec['predicted_category_v5_unverified']:
                                record['review_flags']=['V5_VS_QWEN_CATEGORY_DISAGREEMENT_NOT_AUTOMATIC_CORRECTION']
                    if task=='vqa':record['vqa_visible_abstention_marker']=bool(ABSTAIN.search(raw))
                    if task=='copy':record['copy_review_flags']=['DEBUG_MARKER_IN_COPY'] if DEBUG.search(raw) else []
                    good+=1;times.append(timing['end_to_end_ms'])
                except Exception as err:
                    record['error']=type(err).__name__+': '+str(err);failed+=1
                out.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');out.flush()
                print('FIXED_DEV_REQUEST',good+failed,'/93',rid,record['status'],flush=True)
    report={'schema':'prd32_fixed_dev_gpu_summary_20261030_v1','variant':args.variant,'images':12,'instances':31,
       'requests':good+failed,'successful':good,'failed':failed,'model_load_seconds':load_s,
       'median_request_ms':round(statistics.median(times),2) if times else None,
       'status':'COMPLETE_FOR_REVIEW' if good==93 and failed==0 else 'PARTIAL_FAILURES',
       'publish_allowed':False,'official_prd_acceptance':False}
    jwrite(args.output/'run_summary.json',report)
    print('FIXED_DEV_RUN',report['status'],'SUCCESS',good,'FAILED',failed)
    if failed:raise RuntimeError('PARTIAL_FAILURES_INSPECT_ERROR_BEFORE_RETRY')

def validate_run(path,f,known,variant):
    provenance=jread(path/'provenance.json')
    if provenance.get('variant')!=variant or provenance.get('instances_sha256')!=f['frozen_instance_manifest_sha256'] or provenance.get('requests_sha256')!=f['request_manifest_sha256']:raise ValueError('COHORT_PROVENANCE_MISMATCH:'+variant)
    rows=jlread(path/'qwen_results.jsonl')
    if len(rows)!=93:raise ValueError('INCOMPLETE_REQUESTS:'+variant)
    got={}
    for r in rows:
        cid,iid,task=r.get('case_id'),r.get('target_instance_id'),r.get('task')
        k=(cid,iid,task)
        if (cid,iid) not in known or task not in TASKS or k in got:raise ValueError('UNEXPECTED_OR_DUPLICATE_REQUEST:'+str(k))
        if r.get('request_id')!=f'{cid}__{iid}__{task}' or r.get('status')!='GENERATED_REVIEW_REQUIRED':raise ValueError('INVALID_REQUEST_OR_STATUS:'+str(k))
        rec=known[(cid,iid)]
        for rf,mf in [('v5_predicted_bbox_xyxy','target_bbox_xyxy'),('v5_category_unverified','predicted_category_v5_unverified'),('v5_predicted_score_uncalibrated','predicted_score_v5_uncalibrated'),('image_sha256','image_sha256')]:
            if r.get(rf)!=rec[mf]:raise ValueError('CHANGED_FROZEN_INPUT:'+str(k)+':'+rf)
        if r.get('human_label_or_gt_bbox_used') is not False or r.get('publish_allowed') is not False:raise ValueError('UNSAFE_PROVENANCE:'+str(k))
        if r.get('prompt_sha256')!=provenance['prompt_hashes'][task]:raise ValueError('PROMPT_HASH_MISMATCH:'+str(k))
        if task=='vqa' and (r.get('question_text')!=QUESTION or r.get('question_sha256')!=text_sha(QUESTION)):raise ValueError('QUESTION_MISMATCH:'+str(k))
        if not isinstance(r.get('marked_image_sha256'),str) or len(r['marked_image_sha256'])!=64:raise ValueError('MISSING_MARKED_IMAGE_HASH:'+str(k))
        im=r.get('marked_image')
        if im and Path(im).is_file() and digest(im)!=r['marked_image_sha256']:raise ValueError('MARKED_IMAGE_CHANGED:'+str(k))
        got[k]=r
    if set(got)!={(cid,iid,t) for cid,iid in known for t in TASKS}:raise ValueError('NOT_FULL_FROZEN_REQUEST_COHORT')
    return got

def stats(rows):
    grouped=defaultdict(dict)
    for (cid,iid,task),row in rows.items():grouped[(cid,iid)][task]=row
    counts=Counter()
    for trip in grouped.values():
        parsed=(trip['tags'].get('tag_parse') or {}).get('canonical_service_json')
        if not isinstance(parsed,dict):counts['tags_unparseable']+=1
        elif parsed.get('category')!=trip['tags'].get('v5_category_unverified'):counts['v5_qwen_class_disagreements']+=1
        if DEBUG.search(trip['copy'].get('raw_text','')):counts['copy_debug_text']+=1
        if ABSTAIN.search(trip['vqa'].get('raw_text','')):counts['vqa_visible_abstentions']+=1
    return {'instances':len(grouped),**{k:counts[k] for k in ['tags_unparseable','v5_qwen_class_disagreements','copy_debug_text','vqa_visible_abstentions']}}

def compare(args):
    f,records=frozen_inputs(args.freeze,args.source,args.focus_script,args.prep,args.raw)
    known={(r['case_id'],r['target_instance_id']):r for r in records}
    old=validate_run(args.baseline,f,known,'baseline')
    new=validate_run(args.focus,f,known,'focus_v1')
    for k in old:
        if old[k]['marked_image_sha256']!=new[k]['marked_image_sha256']:
            raise ValueError('BASELINE_FOCUS_MARKED_IMAGE_MISMATCH:'+str(k))
    if args.output.exists():raise FileExistsError('REFUSE_OVERWRITE:'+str(args.output))
    args.output.mkdir(parents=True)
    per=[]
    for cid,iid in sorted(known):
        trip_old={t:old[(cid,iid,t)] for t in TASKS};trip_new={t:new[(cid,iid,t)] for t in TASKS}
        per.append({'case_id':cid,'instance_id':iid,'v5_class_NOT_GT':known[(cid,iid)]['predicted_category_v5_unverified'],
             'tags_changed':trip_old['tags']['raw_text']!=trip_new['tags']['raw_text'],
             'vqa_changed':trip_old['vqa']['raw_text']!=trip_new['vqa']['raw_text'],
             'copy_changed':trip_old['copy']['raw_text']!=trip_new['copy']['raw_text'],
             'baseline_tags':trip_old['tags']['raw_text'],'focus_tags':trip_new['tags']['raw_text'],
             'baseline_vqa':trip_old['vqa']['raw_text'],'focus_vqa':trip_new['vqa']['raw_text'],
             'baseline_copy':trip_old['copy']['raw_text'],'focus_copy':trip_new['copy']['raw_text']})
    with (args.output/'paired_instances.csv').open('w',newline='',encoding='utf-8-sig') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(per[0]));writer.writeheader();writer.writerows(per)
    sa,sb=stats(old),stats(new)
    report={'schema':'prd32_fixed_dev_paired_comparison_20261030_v1',
        'scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY','matched_instances':31,'matched_requests':93,
        'baseline_file_sha256':digest(args.baseline/'qwen_results.jsonl'),
        'focus_file_sha256':digest(args.focus/'qwen_results.jsonl'),
        'same_instances_images_bboxes_questions_and_marked_input':True,
        'baseline_diagnostics_NOT_ACCURACY':sa,'focus_diagnostics_NOT_ACCURACY':sb,
        'human_qualitative_review_required':True,'no_main_strategy_change':True,
        'holdout_used':False,'publish_allowed':False,'official_prd_acceptance':False,
        'notes':['V5 category conflicts are not accuracy without GT','duplicate predictions can inflate apparent n','12 images are too few for stable conclusions']}
    jwrite(args.output/'comparison.json',report)
    print('FIXED_COHORT_PAIRED_COMPARE_OK',json.dumps({'baseline':sa,'focus':sb},ensure_ascii=False))
    print('NOT_ACCURACY_NOT_PRD_ACCEPTANCE')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for cmd in ('preflight','run','compare'):
        s=sub.add_parser(cmd)
        for arg,default in [('freeze',DEFAULT_FREEZE),('source',DEFAULT_SOURCE),('focus-script',DEFAULT_FOCUS_SCRIPT),('prep',DEFAULT_PREP),('raw',DEFAULT_RAW)]:
            s.add_argument('--'+arg,type=Path,default=default)
        if cmd=='run':
            s.add_argument('--variant',choices=('baseline','focus_v1'),required=True)
            s.add_argument('--output',type=Path,required=True)
            s.add_argument('--model-dir',type=Path,default=DEFAULT_MODEL)
        if cmd=='compare':
            s.add_argument('--baseline',type=Path,required=True)
            s.add_argument('--focus',type=Path,required=True)
            s.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    try:
        {'preflight':check,'run':run,'compare':compare}[args.command](args)
    except Exception as e:
        print('FIXED_COHORT_BLOCKED',type(e).__name__+': '+str(e),file=sys.stderr)
        raise SystemExit(2)

if __name__=='__main__':main()
