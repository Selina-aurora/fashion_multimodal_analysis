#!/usr/bin/env python3
# coding: utf-8
"""Frozen V5 Mask R-CNN raw inference on 12 source-verified TRAIN DEV images.

Inference reads ONLY the GT-free image manifest. Human annotations are forbidden here.
Reuses original project builder + predict_timed; no GT boxes, no Qwen, no fine-tuning.
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys, time, statistics
from pathlib import Path
from collections import Counter

CHECKPOINT_REL = 'outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth'
CKPT_SHA = 'c5ba795225349b1c521aac0517ac7c493d9818dcc17d558ac4a3f6f8669452f0'
CLASS_TO_ID = {'top':1,'pants':2,'skirt':3,'outerwear':4,'dress':5,'shoe':6,'bag':7,'accessory':8}
ID_TO_CLASS = {v:k for k,v in CLASS_TO_ID.items()}
RESTRICTED = {'holdout','sealed','fullval','val','valid','validation','test','testing','testset','eval','evaluation','devtest'}
DEFAULT_GATE = '/workspace/PRD32_V5_LIVE_PREFLIGHT_20261029_run01'
DEFAULT_PROJECT = '/workspace/fashion_multimodal_analysis'
DEFAULT_DATA = '/workspace/fashion_data'
DEFAULT_OUTPUT = '/workspace/PRD32_V5_REAL_20261029_run01'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def readj(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def readjl(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def writej(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def writejl(path,rows):
    with Path(path).open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def restricted_name(seg):
    s=seg.lower()
    return any(s==x or s.startswith(x+'_') or s.endswith('_'+x) for x in RESTRICTED)

def verified_source_path(filename,data_root):
    if not isinstance(filename,str) or not filename.startswith('/'):
        raise ValueError('Input image path must be absolute')
    p=Path(filename)
    if any(restricted_name(q) for q in p.parts):raise ValueError('RESTRICTED_SPLIT_LEXICAL')
    rp=p.resolve(strict=False);rr=Path(data_root).resolve(strict=False)
    if any(restricted_name(q) for q in rp.parts):raise ValueError('RESTRICTED_SPLIT_RESOLVED')
    if not rp.is_relative_to(rr):raise ValueError('OUTSIDE_APPROVED_DATA_ROOT')
    if 'train' not in [x.lower() for x in rp.parts]:raise ValueError('TRAIN_COMPONENT_REQUIRED')
    if rp.suffix.lower() not in ('.jpg','.jpeg','.png','.webp'):raise ValueError('UNSUPPORTED_IMAGE_FORMAT')
    if not rp.is_file():raise ValueError('APPROVED_TRAIN_FILE_MISSING')
    return rp

def validate_inputs(gate_dir,data_root,project,check_hash=True):
    gate=Path(gate_dir)
    summary=readj(gate/'v5_live_preflight_summary.json')
    if summary.get('schema')!='prd32_v5_live_preflight_v1' or summary.get('stage')!='CHECKPOINT_AND_SOURCE_PROVENANCE_ONLY' or summary.get('ready_for_live_v5_invocation') is not True:
        raise ValueError('Frozen preflight is not ready')
    c=summary.get('checkpoint',{})
    cp=(Path(project).resolve()/CHECKPOINT_REL).resolve()
    if str(cp)!=str(Path(c.get('path','')).resolve()) or c.get('expected_sha256')!=CKPT_SHA or c.get('actual_sha256')!=CKPT_SHA or c.get('hash_verified_now') is not True:
        raise ValueError('Checkpoint record differs from known freeze')
    if not cp.is_file():raise ValueError('Checkpoint not found')
    if check_hash and sha(cp)!=CKPT_SHA:raise ValueError('Checkpoint changed after preflight')
    rows=readjl(gate/'v5_image_only_inputs.jsonl')
    if len(rows)!=12 or summary.get('new_train_source_candidates_verified')!=12:raise ValueError('Expected 12 approved TRAIN images')
    ids=set(); digests=set();checked=[]
    for row in rows:
        allowed={'case_id','image_path','image_sha256','requested_output','scope','do_not_use_historical_gt_bbox','human_reference_withheld'}
        if set(row)!=allowed:raise ValueError('GT-free image-only manifest schema mismatch: refuse extra bbox/category fields')
        cid=row.get('case_id'); digest=row.get('image_sha256')
        if not isinstance(cid,str) or not cid.startswith('fresh_dev_') or cid in ids:raise ValueError('Invalid or duplicated case ID')
        if not isinstance(digest,str) or len(digest)!=64 or digest in digests:raise ValueError('Invalid or duplicated image hash')
        if row.get('requested_output')!='PRD31_V5_RAW_PREDICTIONS_ONLY' or row.get('scope')!='TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY' or row.get('do_not_use_historical_gt_bbox') is not True or row.get('human_reference_withheld') is not True:
            raise ValueError('Unapproved source/input policy')
        p=verified_source_path(row['image_path'],data_root)
        if sha(p)!=digest:raise ValueError(f'{cid} TRAIN_IMAGE_SHA256_CHANGED')
        ids.add(cid);digests.add(digest)
        checked.append({'case_id':cid,'image_path':str(p),'image_sha256':digest})
    return checked, cp, summary

def run_preflight(a):
    checked,cp,s=validate_inputs(a.gate,a.data_root,a.project,check_hash=True)
    print('V5_LIVE_INPUTS_OK',json.dumps({'count':len(checked),'checkpoint_sha256':CKPT_SHA,'human_bbox_or_category_present':False,'gpu_calls':0},ensure_ascii=False),flush=True)

def run_gpu(a):
    import os
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    if a.score_threshold!=0.4 or a.mask_threshold!=0.5:raise ValueError('Frozen protocol requires score=0.4, mask=0.5')
    checked,cp,pre=validate_inputs(a.gate,a.data_root,a.project,check_hash=True)
    out=Path(a.output)
    if out.exists():raise ValueError('Refusing to overwrite run directory: '+str(out))
    from PIL import Image,ImageDraw
    import numpy as np
    import torch
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable: do not silently run CPU and claim GPU timing')
    sys.path.insert(0,str(Path(a.project).resolve()/'src'))
    from fashion_multimodal_analysis.segmentation.modeling import build_evaluation_model
    from fashion_multimodal_analysis.evaluation.runtime import predict_timed
    from fashion_multimodal_analysis.common.schema import CLASS_TO_ID as native_ids
    if native_ids!=CLASS_TO_ID:raise ValueError('Project class IDs differ from expected freeze')
    print('V5_CHECKPOINT_VERIFIED: loading original Mask R-CNN only',flush=True)
    # Known, SHA-verified checkpoint. The historical project stores a full training state.
    checkpoint=torch.load(str(cp),map_location='cpu',weights_only=False)
    if not isinstance(checkpoint,dict) or checkpoint.get('class_to_id')!=CLASS_TO_ID or 'model_state_dict' not in checkpoint:
        raise ValueError('V5 state dict/class order mismatch')
    saved=checkpoint.get('args',{})
    msize=int(saved.get('min_size',640));msmax=int(saved.get('max_size',1024))
    if not (32<=msize<=2048 and msize<=msmax<=4096):raise ValueError('Unsafe model input dimensions in checkpoint')
    model=build_evaluation_model(msize,msmax)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    del checkpoint
    torch.cuda.empty_cache()
    device=torch.device('cuda')
    model.to(device).eval()
    out.mkdir(parents=True)
    writej(out/'run_manifest.json',{'schema':'prd32_v5_real_predictions_run_v1','input_manifest_sha256':sha(Path(a.gate)/'v5_image_only_inputs.jsonl'),
      'checkpoint_sha256':CKPT_SHA,'min_size':msize,'max_size':msmax,'score_threshold':a.score_threshold,
      'mask_threshold':a.mask_threshold,'model_builder':'fashion_multimodal_analysis.segmentation.modeling.build_evaluation_model',
      'predictor':'fashion_multimodal_analysis.evaluation.runtime.predict_timed','scope':'TRAIN_SOURCE_DEV_ONLY_NOT_INDEPENDENT',
      'human_gt_used_for_inference':False,'model_predictions_not_human_verified':True})
    allrows=[]; time_ms=[]
    for n,row in enumerate(checked,1):
        from PIL import ImageOps
        start=time.perf_counter()
        with Image.open(row['image_path']) as src:
            # Project uses PIL RGB conversion; non-default EXIF must be checked manually.
            if src.getexif().get(274,1) not in (1,None):raise ValueError('Image EXIF rotation unsupported: '+row['case_id'])
            im=src.convert('RGB')
        if im.width*im.height>45_000_000:raise ValueError('Image too large')
        boxes,labels,scores,masks,ms=predict_timed(im,model,device,a.score_threshold,a.mask_threshold)
        timeround=round(float(ms),2)
        time_ms.append(timeround)
        entries=[];canvas=im.copy();draw=ImageDraw.Draw(canvas)
        for i,(box,label,confidence,mask) in enumerate(zip(boxes,labels,scores,masks)):
            b=list(map(float,box)); lab=int(label)
            if lab not in ID_TO_CLASS or not all(math.isfinite(x) for x in b):continue
            x1,y1,x2,y2=b
            b=[max(0.,x1),max(0.,y1),min(float(im.width),x2),min(float(im.height),y2)]
            if not (b[0]<b[2] and b[1]<b[3]):continue
            if np.asarray(mask).shape!=(im.height,im.width):raise ValueError('Mask coordinate shape mismatch')
            instance_id=f'pred_{i:04d}'
            mask_file=None
            if len(entries)<a.max_masks:
                mask_file=f'{row["case_id"]}_{instance_id}_mask.png'
                Image.fromarray((np.asarray(mask)>0).astype('uint8')*255).save(out/mask_file)
            records={'instance_id':instance_id,'predicted_category':ID_TO_CLASS[lab], 'predicted_class_id':lab,
             'score':round(float(confidence),6),'bbox_xyxy':[round(x,3) for x in b],
             'bbox_coordinate_space':'original_image','mask_path':mask_file,
             'mask_coordinate_space':'original_image' if mask_file else 'not_saved',
             'prediction_origin':'V5_CHECKPOINT_LIVE_MODEL_OUTPUT_NOT_GT'}
            entries.append(records)
            draw.rectangle(b,outline=(245,39,39),width=max(2,min(im.size)//160))
            draw.text((b[0],b[1]),instance_id+' '+ID_TO_CLASS[lab]+' '+str(round(float(confidence),2)),fill=(240,30,30))
        canvas.thumbnail((1100,1100)); canvas.save(out/f'{row["case_id"]}_v5_review.jpg',quality=84)
        allrows.append({'schema':'prd32_v5_live_raw_image_result_v1','case_id':row['case_id'],'image_sha256':row['image_sha256'],
          'source_image_path':row['image_path'],'image_size':[im.width,im.height],
          'segmentation_ms':timeround,'instances':entries,'instance_count':len(entries),
          'human_target_instance_selected':False,'review_image':f'{row["case_id"]}_v5_review.jpg',
          'scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY'})
        writejl(out/'v5_raw_predictions.jsonl',allrows)
        print(f'V5_IMAGE_DONE {n}/{len(checked)} {row["case_id"]} instances={len(entries)} segment_ms={timeround}',flush=True)
    summary={'schema':'prd32_v5_real_prediction_summary_v1','status':'V5_RAW_INFERENCE_COMPLETED','n_images':len(allrows),
      'n_images_with_no_instances':sum(not x['instances'] for x in allrows),
      'n_predicted_instances':sum(len(x['instances']) for x in allrows),
      'segment_latency_median_ms':round(statistics.median(time_ms),2),
      'latency_is_single_pass_no_warmup':True,'checkpoint_sha256':CKPT_SHA,
      'v5_predictions_sha256':sha(out/'v5_raw_predictions.jsonl'),
      'categories_from_v5_predictions':dict(Counter(v['predicted_category'] for x in allrows for v in x['instances'])),
      'human_reference_used':False,'qwen_calls':0,'official_prd_acceptance':False,
      'note':'All instances are real frozen-checkpoint predictions. No human target mapping was used during inference.'}
    writej(out/'v5_real_summary.json',summary)
    print('V5_REAL_INFERENCE_DONE',json.dumps({'images':len(allrows),'instances':summary['n_predicted_instances'],'output':str(out)},ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode',required=True)
    for cmd in ('preflight','gpu'):
        a=sub.add_parser(cmd)
        a.add_argument('--gate',default=DEFAULT_GATE)
        a.add_argument('--project',default=DEFAULT_PROJECT)
        a.add_argument('--data-root',default=DEFAULT_DATA)
        a.add_argument('--output',default=DEFAULT_OUTPUT)
        a.add_argument('--score-threshold',type=float,default=0.4)
        a.add_argument('--mask-threshold',type=float,default=0.5)
        a.add_argument('--max-masks',type=int,default=8)
    a=p.parse_args()
    try:
        if a.mode=='preflight':run_preflight(a)
        elif a.mode=='gpu':run_gpu(a)
    except Exception as e:
        print('V5_REAL_BLOCKED',type(e).__name__+': '+str(e),file=sys.stderr)
        sys.exit(2)
if __name__=='__main__':main()
