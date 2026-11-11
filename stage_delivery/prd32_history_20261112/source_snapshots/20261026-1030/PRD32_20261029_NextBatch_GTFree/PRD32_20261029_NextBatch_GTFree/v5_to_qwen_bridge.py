#!/usr/bin/env python3
# coding: utf-8
"""GT-free V5 raw predictions -> Qwen tags/VQA/copy smoke.
No human reference, pilot boxes, oracle IoU, or category labels are accessed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

CKPT_SHA = 'c5ba795225349b1c521aac0517ac7c493d9818dcc17d558ac4a3f6f8669452f0'
CLASSES = ['top','pants','skirt','outerwear','dress','shoe','bag','accessory']
RESTRICTED = {'holdout','sealed','fullval','val','valid','validation','test','testing','testset','eval','evaluation','devtest'}
DEFAULT_RAW='/workspace/PRD32_V5_REAL_20261029_run01'
DEFAULT_GATE='/workspace/PRD32_V5_LIVE_PREFLIGHT_20261029_run01'
DEFAULT_DATA='/workspace/fashion_data'
DEFAULT_PREP='/workspace/PRD32_V5_QWEN_PREP_20261029_run01'
DEFAULT_GPU='/workspace/PRD32_V5_QWEN_SMOKE_20261029_run01'
DEFAULT_MODEL='/workspace/models/Qwen2.5-VL-7B-Instruct'
ALLOWED_TASKS=['tags','vqa','copy']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def readj(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def readjl(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def writej(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def writejl(p,rows):
    with Path(p).open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def restriction(part):
    value=part.lower()
    return value in RESTRICTED or any(value.startswith(x+'_') or value.endswith('_'+x) for x in RESTRICTED)
def approved_image(name,root):
    p=Path(name)
    if not p.is_absolute() or any(restriction(x) for x in p.parts):raise ValueError('RESTRICTED_OR_NONABSOLUTE_IMAGE_PATH')
    q=p.resolve(strict=False); base=Path(root).resolve()
    if not q.is_relative_to(base) or any(restriction(x) for x in q.parts):raise ValueError('RESTRICTED_RESOLVED_OR_OUTSIDE_ROOT')
    if 'train' not in [v.lower() for v in q.parts]:raise ValueError('TRAIN_COMPONENT_REQUIRED')
    if not q.is_file():raise FileNotFoundError(q)
    if q.suffix.lower() not in {'.png','.jpg','.jpeg','.webp'}:raise ValueError('UNSUPPORTED_IMAGE_TYPE')
    return q

def validbox(b,w,h):
    return isinstance(b,list) and len(b)==4 and all(type(x) in (float,int) and math.isfinite(x) for x in b) and 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h

def validated(raw,gate,data_root):
    from PIL import Image
    raw=Path(raw);gate=Path(gate)
    s=readj(raw/'v5_real_summary.json'); run=readj(raw/'run_manifest.json')
    predpath=raw/'v5_raw_predictions.jsonl'
    if s.get('status')!='V5_RAW_INFERENCE_COMPLETED' or s.get('checkpoint_sha256')!=CKPT_SHA or s.get('v5_predictions_sha256')!=sha(predpath):
        raise ValueError('V5 raw prediction identity or status mismatch')
    if run.get('schema')!='prd32_v5_real_predictions_run_v1' or run.get('checkpoint_sha256')!=CKPT_SHA or run.get('human_gt_used_for_inference') is not False:
        raise ValueError('Not an approved GT-free V5 run')
    if run.get('score_threshold')!=0.4 or run.get('mask_threshold')!=0.5:raise ValueError('Frozen V5 thresholds changed')
    gatepath=gate/'v5_image_only_inputs.jsonl'
    if run.get('input_manifest_sha256')!=sha(gatepath):raise ValueError('Input manifest not tied to V5 run')
    g=readj(gate/'v5_live_preflight_summary.json')
    if g.get('ready_for_live_v5_invocation') is not True or g.get('checkpoint',{}).get('actual_sha256')!=CKPT_SHA:raise ValueError('Source gate not ready')
    imgs=readjl(gatepath); rows=readjl(predpath)
    if len(imgs)!=12 or len(rows)!=12 or s.get('n_images')!=12:raise ValueError('Unexpected V5 image count')
    expected={x['case_id']:x for x in imgs};seen=set();output=[]
    if len(expected)!=12:raise ValueError('Duplicate image-only IDs')
    for row in rows:
        allowed_row={'schema','case_id','image_sha256','source_image_path','image_size','segmentation_ms','instances','instance_count','human_target_instance_selected','review_image','scope'}
        if set(row)!=allowed_row:raise ValueError('V5 raw image result fields differ from approved schema')
        if row.get('schema')!='prd32_v5_live_raw_image_result_v1' or row.get('scope')!='TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY' or row.get('human_target_instance_selected') is not False:
            raise ValueError('Unexpected raw record schema or target policy')
        cid=row['case_id']
        if cid not in expected or cid in seen:raise ValueError('Unknown or duplicate case ID')
        seen.add(cid)
        src=expected[cid]
        if src.get('human_reference_withheld') is not True or src.get('do_not_use_historical_gt_bbox') is not True:
            raise ValueError('Image-only source not GT-free')
        image=approved_image(src['image_path'],data_root)
        if str(image)!=str(Path(row['source_image_path']).resolve()) or sha(image)!=row['image_sha256'] or row['image_sha256']!=src['image_sha256']:
            raise ValueError(cid+': input image SHA or path mismatch')
        with Image.open(image) as img:
            if list(img.size)!=row['image_size']:raise ValueError('Image dimensions differ')
            w,h=img.size
        if row.get('instance_count')!=len(row.get('instances',[])):raise ValueError('instance_count mismatch')
        if not isinstance(row.get('instances'),list):raise ValueError('Malformed instances')
        local=set()
        for item in row['instances']:
            if set(item)-{'instance_id','predicted_category','predicted_class_id','score','bbox_xyxy','bbox_coordinate_space','mask_path','mask_coordinate_space','prediction_origin'}:
                raise ValueError(cid+': unexpected prediction fields, potential annotation leakage')
            iid=item['instance_id']; cls=item['predicted_category']; b=item['bbox_xyxy']
            if iid in local or not isinstance(iid,str) or not iid.startswith('pred_'):raise ValueError('Bad instance ID')
            local.add(iid)
            if cls not in CLASSES or CLASSES.index(cls)+1!=item['predicted_class_id']:raise ValueError('V5 class ID mismatch')
            if not validbox(b,w,h) or item.get('bbox_coordinate_space')!='original_image':raise ValueError('Invalid bbox')
            if type(item.get('score')) not in (int,float) or not math.isfinite(item['score']) or not .4<=item['score']<=1.:raise ValueError('Invalid score')
            if item.get('prediction_origin')!='V5_CHECKPOINT_LIVE_MODEL_OUTPUT_NOT_GT':raise ValueError('Non-V5 prediction origin')
            mask=item.get('mask_path')
            if mask is not None:
                if not isinstance(mask,str) or Path(mask).name!=mask or (raw/mask).resolve().parent!=raw.resolve() or not (raw/mask).is_file():raise ValueError('Mask filename invalid or missing')
            output.append({'schema':'prd32_gt_free_predicted_instance_v1','case_id':cid,'image_sha256':src['image_sha256'],
              'image_path':str(image),'image_size':[w,h],'target_instance_id':iid,
              'target_bbox_xyxy':b,'target_bbox_origin':'prd31_v5_live_prediction',
              'predicted_category_v5_unverified':cls,'predicted_score_v5_uncalibrated':item['score'],
              'mask_path_from_v5':str((raw/mask).resolve()) if mask else None,
              'segment_ms_per_image':row['segmentation_ms'],'selection_policy':'ALL_PREDICTED_INSTANCES_NO_TARGET_ORACLE',
              'not_user_selected_target':True,'human_reference_used':False,'scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY'})
    if seen!=set(expected):raise ValueError('Missing predicted image')
    if len(output)!=s.get('n_predicted_instances'):raise ValueError('Predicted instance total mismatch')
    return output,s

def prepare(raw,gate,data_root,output):
    out=Path(output)
    if out.exists():raise FileExistsError('No overwrite: '+str(out))
    records,summary=validated(raw,gate,data_root)
    out.mkdir(parents=True)
    writejl(out/'gt_free_all_instances.jsonl',records)
    fieldnames=['case_id','target_instance_id','predicted_category_v5_unverified','predicted_score_v5_uncalibrated','target_bbox_xyxy','segment_ms_per_image','selection_policy']
    with (out/'gt_free_instances.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader()
        for row in records:w.writerow({k:row[k] for k in fieldnames})
    # Review thumbnails are copied to avoid requiring personal annotations or oracle CSV.
    from shutil import copyfile
    cards=[]; raw=Path(raw)
    for cid in sorted(set(r['case_id'] for r in records)):
        source=raw/(cid+'_v5_review.jpg');target=out/(cid+'_v5_review.jpg')
        if source.is_file():copyfile(source,target)
        kinds=[r for r in records if r['case_id']==cid]
        lines=', '.join(f"{html.escape(r['target_instance_id'])} ({r['predicted_category_v5_unverified']}, {r['predicted_score_v5_uncalibrated']:.3f})" for r in kinds)
        cards.append(f'<section><h2>{cid}</h2><img src="{target.name}" alt="V5 predicted all instances"><p>{lines}</p></section>' if target.exists() else f'<section><h2>{cid}</h2><p>{lines}</p></section>')
    hdoc='<!doctype html><html lang="zh"><meta charset="utf-8"><title>V5 All Predicted Instances</title><style>body{font:15px/1.5 system-ui;background:#f5f7fb;margin:20px auto;max-width:1050px}section{background:white;border:1px solid #dae1e7;border-radius:12px;padding:18px;margin:14px 0}img{max-width:100%;max-height:580px;object-fit:contain}h2{font-size:18px}</style><h1>V5 真实预测实例 · GT-free 审核</h1><p>红框、类别均来自 V5。未使用人工框/人工类别选择实例。请勿将 pred_0000 默认解释为目标。</p>'+''.join(cards)+'</html>'
    (out/'V5_Predicted_Instances_Review.html').write_text(hdoc,encoding='utf-8')
    result={'schema':'prd32_gt_free_v5_to_qwen_preparation_v1','status':'READY_FOR_DEVELOPMENT_SMOKE','scope':'TRAIN_SOURCE_DEV_NOT_INDEPENDENT',
            'n_images':12,'n_predicted_instances':len(records),'v5_raw_predictions_sha256':summary['v5_predictions_sha256'],
            'all_instances_manifest_sha256':sha(out/'gt_free_all_instances.jsonl'),
            'categories_v5_unverified':dict(Counter(r['predicted_category_v5_unverified'] for r in records)),
            'target_selection_policy':'ALL_INSTANCES','operational_user_target_selector_implemented':False,
            'human_gt_or_posthoc_bbox_used':False,'qwen_calls':0,'official_prd_acceptance':False}
    writej(out/'bridge_preparation_summary.json',result)
    print('GT_FREE_BRIDGE_READY',json.dumps({'images':12,'predicted_instances':len(records),'human_oracle_used':False},ensure_ascii=False),flush=True)
    return result

def load_helper():
    p=Path(__file__).resolve().parent/'frozen_helpers/prd32_integrated_pipeline_20261027.py'
    spec=importlib.util.spec_from_file_location('frozen_prd32_prompt_helper',p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod,p

def gpu(prep,raw,gate,data_root,output,model_dir,case_ids,tasks):
    out=Path(output); prep=Path(prep)
    if out.exists():raise FileExistsError('No overwrite: '+str(out))
    prior=readj(prep/'bridge_preparation_summary.json')
    if prior.get('status')!='READY_FOR_DEVELOPMENT_SMOKE' or prior.get('human_gt_or_posthoc_bbox_used') is not False:raise ValueError('GT-free preflight not ready')
    records,source_summary=validated(raw,gate,data_root)
    if source_summary['v5_predictions_sha256']!=prior['v5_raw_predictions_sha256'] or sha(prep/'gt_free_all_instances.jsonl')!=prior['all_instances_manifest_sha256']:
        raise ValueError('Prepared bridge changed')
    prepared=readjl(prep/'gt_free_all_instances.jsonl')
    if records!=prepared:raise ValueError('Bridge and V5 live predictions differ')
    selected={x.strip() for x in case_ids.split(',') if x.strip()}
    if not selected:raise ValueError('Explicit case IDs required')
    if not selected.issubset({r['case_id'] for r in prepared}):raise ValueError('Unknown case ID')
    chosen=[r for r in prepared if r['case_id'] in selected]
    chosen.sort(key=lambda x:(x['case_id'],x['target_instance_id']))
    tasklist=[x.strip() for x in tasks.split(',') if x.strip()]
    if not tasklist or any(x not in ALLOWED_TASKS for x in tasklist) or len(tasklist)!=len(set(tasklist)):
        raise ValueError('Unsupported/duplicated tasks')
    # Every V5 instance in each chosen image is used; no oracle or hidden first-ROI selection.
    if len(chosen)>12 or len(chosen)*len(tasklist)>36:raise ValueError('Diagnostic smoke exceeds 36 requests; reduce case IDs')
    model_dir=Path(model_dir)
    if not (model_dir/'config.json').is_file():raise FileNotFoundError('Local Qwen model absent')
    helper,helper_file=load_helper()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this smoke')
    free_gb=torch.cuda.mem_get_info()[0]/1024**3
    if free_gb<18:raise RuntimeError('Require at least 18GiB free GPU memory, have %.1f'%free_gb)
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    out.mkdir(parents=True)
    (out/'marked_inputs').mkdir()
    writej(out/'smoke_provenance.json',{'scope':'TRAIN_SOURCE_DEV_ONLY','V5_raw_sha256':source_summary['v5_predictions_sha256'],
            'prepared_manifest_sha256':prior['all_instances_manifest_sha256'],'helper_sha256':sha(helper_file),
            'tags_prompt_sha256':hashlib.sha256(helper.TAG_PROMPT.encode('utf-8')).hexdigest(),
            'tasks':tasklist,'case_ids':sorted(selected),'instance_policy':'ALL_INSTANCES_IN_SELECTED_IMAGES',
            'GT_was_not_loaded':True,'user_target_identification_proven':False})
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    start=time.perf_counter()
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_dir),local_files_only=True,device_map='auto',dtype=dtype,attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    processor=AutoProcessor.from_pretrained(str(model_dir),local_files_only=True,max_pixels=512*28*28)
    load_seconds=round(time.perf_counter()-start,2)
    outputs=[]
    with (out/'qwen_results.jsonl').open('w',encoding='utf-8') as fout:
        for rec in chosen:
            for task in tasklist:
                qrec={'image_path':rec['image_path'],'image_sha256':rec['image_sha256'],
                      'target_bbox_xyxy':rec['target_bbox_xyxy'],'target_instance_id':rec['target_instance_id'],
                      'task':task,'query':'红框内目标服饰的主要可见颜色是什么？','length_chars':[20,80],
                      'predicted_attribute_context':[]}
                outcome={'request_id':rec['case_id']+'__'+rec['target_instance_id']+'__'+task,
                         'case_id':rec['case_id'],'task':task,'target_instance_id':rec['target_instance_id'],
                         'v5_category_unverified':rec['predicted_category_v5_unverified'],
                         'v5_predicted_bbox_xyxy':rec['target_bbox_xyxy'],
                         'v5_predicted_score_uncalibrated':rec['predicted_score_v5_uncalibrated'],
                         'target_bbox_origin':'prd31_v5_live_prediction','selection_policy':'ALL_INSTANCES_IN_SELECTED_IMAGES',
                         'human_label_or_gt_bbox_used':False,'publish_allowed':False,'quality_human_verified':False,
                         'status':'CASE_FAILED'}
                try:
                    image_file=helper.make_marked_image(qrec,out/'marked_inputs','marked_full_image',.05)
                    prompt=helper.make_prompt(qrec,input_mode='marked_full_image',pass_unverified=False)
                    rawtxt,timing=helper.run_model_case(model,processor,qrec,image_file,prompt,task_limits={'tags':170,'vqa':128,'copy':120})
                    if not rawtxt.strip():raise ValueError('EMPTY_MODEL_OUTPUT')
                    outcome.update(status='GENERATED_REVIEW_REQUIRED',raw_text=rawtxt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),marked_image=str(image_file),timing=timing)
                    if task=='tags':
                        parsed=helper.parse_tag_output(rawtxt)
                        outcome['tag_parse']=parsed
                        if parsed.get('canonical_service_json') is not None:
                            canonical=parsed['canonical_service_json']
                            outcome['tag_bridge_candidate']=helper.bridge_tags_adapter(canonical)
                            if canonical.get('category') is not None and canonical['category']!=rec['predicted_category_v5_unverified']:
                                outcome['review_flags']=['V5_VS_QWEN_CATEGORY_DISAGREEMENT_NOT_AUTOMATIC_CORRECTION']
                    elif task=='vqa':
                        outcome['vqa_visible_abstention_marker']=('无法确定' in rawtxt or '看不清' in rawtxt)
                    elif task=='copy':
                        flags=[]
                        if any(term in rawtxt for term in ('红框','方框','标注框','bbox','bounding box')):flags.append('DEBUG_MARKER_IN_COPY')
                        if len(rawtxt.strip())>80:flags.append('COPY_EXCEEDS_80_CHARS')
                        outcome['copy_review_flags']=flags
                except Exception as e:
                    outcome['error']=type(e).__name__+': '+str(e)
                outputs.append(outcome)
                fout.write(json.dumps(outcome,ensure_ascii=False,allow_nan=False)+'\n');fout.flush()
                print('QWEN_INSTANCE_DONE',len(outputs),'/',len(chosen)*len(tasklist),outcome['request_id'],outcome['status'],flush=True)
    timings=[r['timing']['end_to_end_ms'] for r in outputs if 'timing' in r]
    result={'schema':'prd32_gt_free_v5_qwen_smoke_v1','status':'COMPLETED_FOR_REVIEW' if all(r['status']!='CASE_FAILED' for r in outputs) else 'PARTIAL_FAILURES',
            'n_images':len(selected),'n_predicted_instances_processed':len(chosen),'n_requests':len(outputs),
            'successful_outputs':sum(r['status']!='CASE_FAILED' for r in outputs),
            'failed_outputs':sum(r['status']=='CASE_FAILED' for r in outputs),
            'v5_raw_sha256':source_summary['v5_predictions_sha256'],'model_load_seconds':load_seconds,
            'median_qwen_request_ms':round(statistics.median(timings),2) if timings else None,
            'task_counts':dict(Counter(r['task'] for r in outputs)),
            'target_selection_policy':'ALL_INSTANCES_IN_SELECTED_IMAGES',
            'user_target_identification_proven':False,'human_reference_used':False,
            'publish_allowed':False,'official_prd_acceptance':False}
    writej(out/'gt_free_qwen_smoke_summary.json',result)
    print('GT_FREE_QWEN_SMOKE_DONE',json.dumps({'requests':len(outputs),'failures':result['failed_outputs']},ensure_ascii=False),flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest='cmd',required=True)
    a=sp.add_parser('prepare')
    a.add_argument('--raw',default=DEFAULT_RAW);a.add_argument('--gate',default=DEFAULT_GATE);a.add_argument('--data-root',default=DEFAULT_DATA);a.add_argument('--output',default=DEFAULT_PREP)
    b=sp.add_parser('gpu')
    b.add_argument('--raw',default=DEFAULT_RAW);b.add_argument('--gate',default=DEFAULT_GATE);b.add_argument('--data-root',default=DEFAULT_DATA)
    b.add_argument('--prepare',default=DEFAULT_PREP);b.add_argument('--output',default=DEFAULT_GPU);b.add_argument('--model-dir',default=DEFAULT_MODEL)
    b.add_argument('--case-ids',default='fresh_dev_001,fresh_dev_004',help='All V5 instances in these images, NOT oracle target selection')
    b.add_argument('--tasks',default='tags,vqa,copy')
    args=p.parse_args()
    try:
        if args.cmd=='prepare':prepare(args.raw,args.gate,args.data_root,args.output)
        else:gpu(args.prepare,args.raw,args.gate,args.data_root,args.output,args.model_dir,args.case_ids,args.tasks)
    except Exception as e:
        print('GT_FREE_BRIDGE_BLOCKED',type(e).__name__+': '+str(e),file=sys.stderr)
        sys.exit(2)
if __name__=='__main__':main()
