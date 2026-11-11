#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROI counterpart of fixed V2 tagged-full-image TRAIN DEV diagnostics.

All source photos are from the existing approved-for-dev TRAIN pool, never from
sealed holdout. Reviewer judgment is loaded solely for selecting case IDs and
posthoc comparisons; it is NEVER included in the model prompt or image.
This is developer error-analysis, not independently measured PRD accuracy.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, statistics, time
from collections import Counter
from pathlib import Path

FIVE = {'top','pants','skirt','outerwear','dress','shoe','bag','accessory'}
CAUSES = {'MODEL_TARGET_BINDING', 'MODEL_CLASSIFICATION'}
DEFAULT_SELECTED='/workspace/PRD32_EXISTING_DATA_REVIEW_20261027_run01/train_dev_selected.csv'
DEFAULT_ORIGINAL='/workspace/PRD32_TRAIN5_GPU_ALL40_20261027_run01/predictions.jsonl'
DEFAULT_MODEL='/workspace/models/Qwen2.5-VL-7B-Instruct'
DEFAULT_ALLOWED='/workspace/fashion_data'
HERE=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def loadj(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def dumpj(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n', encoding='utf-8')

def jsonl(path):
    with Path(path).open('r',encoding='utf-8-sig') as f:
        return [json.loads(x) for x in f if x.strip()]

def putline(f,obj):
    f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n');f.flush()

def loadhelper(p):
    spec=importlib.util.spec_from_file_location('prd32_fixed_v2_helper', p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def is_valid_train(path,root):
    p=Path(path).resolve();root=Path(root).resolve()
    if not p.is_relative_to(root):return False
    parts={s.lower() for s in p.relative_to(root).parts}
    return 'train' in parts and not parts.intersection({'holdout','sealed','test','valid','validation','fullval','val'})

def roi_red_box(source,box,dest,pad=0.05):
    from PIL import Image,ImageDraw
    x1,y1,x2,y2=box
    with Image.open(source) as im:
        im=im.convert('RGB');w,h=im.size
        if not (0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('BBOX_OUT_OF_ACTUAL_IMAGE')
        dx=(x2-x1)*pad;dy=(y2-y1)*pad
        a,b,c,d=max(0,math.floor(x1-dx)),max(0,math.floor(y1-dy)),min(w,math.ceil(x2+dx)),min(h,math.ceil(y2+dy))
        crop=im.crop((a,b,c,d))
        ImageDraw.Draw(crop).rectangle((round(x1-a),round(y1-b),round(x2-a),round(y2-b)),outline=(250,40,40),width=max(3,min(im.size)//150))
        crop.save(dest,quality=93)
    return dest

def build_plan(selected,review,original,allowed_root,full_verify=True):
    from PIL import Image
    with Path(selected).open('r',encoding='utf-8-sig',newline='') as f: selected_rows=list(csv.DictReader(f))
    re=loadj(review)
    if re.get('scope')!='train_dev_diagnostic_only_NOT_INDEPENDENT': raise ValueError('Unexpected review scope')
    human={r['case_id']:r for r in re['rows']}
    predictions={r['case_id']:r for r in jsonl(original)}
    if len(predictions)!=40:raise ValueError('Expected the frozen 40-case baseline; do not switch experiments')
    plans=[];problems=[]
    for cid,r in human.items():
        if r.get('cause') not in CAUSES:continue
        if r.get('review_status')!='REVIEWED' or r.get('bbox_status')!='VALID':raise ValueError(f'{cid}: review not eligible')
        ix=int(cid.split('_')[-1])-1
        if ix<0 or ix>=len(selected_rows):raise ValueError(f'{cid}: absent from source rows')
        sel=selected_rows[ix]; pred=predictions.get(cid)
        if not pred or pred.get('status')!='GENERATED_UNVERIFIED' or not isinstance(pred.get('canonical_tags'),dict): raise ValueError(f'{cid}: missing baseline prediction')
        path=Path(sel.get('resolved_image_path','')).resolve()
        if not is_valid_train(path,allowed_root):raise ValueError(f'{cid}: restricted or external source path')
        if sel.get('decision')!='TRAIN_DEV_ONLY':raise ValueError(f'{cid}: bad source decision')
        if sel.get('inventory_category_NOT_GT') not in FIVE:raise ValueError(f'{cid}: bad inventory label')
        box=[float(sel[f'bbox_{k}']) for k in ('x1','y1','x2','y2')]
        if any(not math.isfinite(x) for x in box):raise ValueError(f'{cid}: nonfinite bbox')
        if not path.is_file():raise FileNotFoundError(path)
        with Image.open(path) as im:
            w,h=im.size
            if w*h>45_000_000:raise ValueError(f'{cid}: oversized image')
            if not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):raise ValueError(f'{cid}: invalid bbox')
            im.verify()
        s=sel.get('sha256','').lower().strip()
        if len(s)!=64 or (full_verify and sha(path)!=s):raise ValueError(f'{cid}: image hash changed or invalid')
        if pred.get('input_image_sha256')!=s:raise ValueError(f'{cid}: baseline image hash disagrees')
        if pred.get('source_hint_NOT_GT')!=sel['inventory_category_NOT_GT']:raise ValueError(f'{cid}: source hint changed')
        if r['human_category'] not in FIVE:raise ValueError(f'{cid}: invalid reviewed class')
        if pred.get('canonical_tags',{}).get('category')==r['human_category']: raise ValueError(f'{cid}: expected reviewed wrong baseline category')
        plans.append({'case_id':cid,'cause':r['cause'],'human_category_FOR_EVAL_ONLY':r['human_category'],
            'baseline_category':pred['canonical_tags']['category'],'source_hint_NOT_GT':sel['inventory_category_NOT_GT'],
            'image_path':str(path),'image_sha256':s,'bbox':box,'target_instance_id':str(sel.get('garment_id') or sel.get('record_id') or cid),
            'review_notes':r.get('notes','')})
    if len(plans)!=15:raise ValueError(f'Expected 8 binding and 7 classification errors, got {len(plans)}')
    counts=Counter(p['cause'] for p in plans)
    if counts!={'MODEL_TARGET_BINDING':8,'MODEL_CLASSIFICATION':7}:raise ValueError('Review group counts changed')
    return plans

def run(args, dry):
    plan=build_plan(args.selected,args.review,args.original_predictions,args.allowed_root)
    if args.subset!='all':plan=[p for p in plan if p['cause']==('MODEL_TARGET_BINDING' if args.subset=='binding' else 'MODEL_CLASSIFICATION')]
    out=Path(args.output)
    if out.exists():raise FileExistsError(f'Output exists; choose new run name: {out}')
    out.mkdir(parents=True)
    mod=loadhelper(Path(args.helper))
    expected_prompt_sha=hashlib.sha256(mod.TAG_PROMPT.encode('utf-8')).hexdigest()
    original_config=Path(args.original_predictions).parent/'run_config.json'
    if original_config.is_file():
        prior=loadj(original_config).get('v2_prompt_sha256')
        if prior and prior!=expected_prompt_sha:raise ValueError('V2 prompt hash mismatch with baseline; do not run comparison')
    dumpj(out/'run_design.json',{'purpose':'visual_target_binding_ablation_train_dev_only','scope':'NOT_INDEPENDENT','baseline':'marked full image red rectangle','counterpart':'ROI 5% padding red rectangle','v2_prompt_sha256':expected_prompt_sha,'source_selected_sha256':sha(args.selected),'source_review_sha256':sha(args.review),'baseline_predictions_sha256':sha(args.original_predictions),'case_count':len(plan),'human_gt_passed_to_model':False,'subset':args.subset,'holdout_used':False,'model':args.model})
    with (out/'case_plan.jsonl').open('w',encoding='utf-8') as f:
        for r in plan:putline(f,r)
    if dry:
        print('PAIRED_ROI_PREFLIGHT_OK',json.dumps({'cases':len(plan),'by_cause':dict(Counter(x['cause'] for x in plan)),'mode':'NO_GPU','output':str(out)},ensure_ascii=False),flush=True)
        return
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available():raise RuntimeError('No GPU CUDA available')
    free=torch.cuda.mem_get_info()[0]/2**30
    if free<17:raise RuntimeError(f'Insufficient free VRAM {free:.1f}GiB; inspect GPU processes first')
    img_dir=out/'roi_marked_inputs';img_dir.mkdir()
    started=time.perf_counter();model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,local_files_only=True,device_map='auto',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    processor=AutoProcessor.from_pretrained(args.model,local_files_only=True,max_pixels=512*28*28)
    load_sec=round(time.perf_counter()-started,2)
    records=[]
    with (out/'roi_predictions.jsonl').open('w',encoding='utf-8') as f:
        for ix,p in enumerate(plan,1):
            res={'case_id':p['case_id'],'cause':p['cause'],'review_human_category_FOR_POSTHOC_ONLY':p['human_category_FOR_EVAL_ONLY'],'baseline_category':p['baseline_category'],'roi_category':None,'status':'RUNTIME_FAILED','model_called':False,'timing':None,'issues':[],'raw_model_output':None}
            try:
                img=roi_red_box(p['image_path'],p['bbox'],img_dir/(p['case_id']+'_roi_marked.jpg'),args.pad)
                rec={'task':'tags','case_id':p['case_id'],'image_path':p['image_path'],'image_sha256':p['image_sha256'],'target_bbox_xyxy':p['bbox'],'target_instance_id':p['target_instance_id']}
                raw,t=mod.run_model_case(model,processor,rec,img,mod.TAG_PROMPT,task_limits={'tags':args.max_new_tokens})
                parsed=mod.parse_tag_output(raw);can=parsed.get('canonical_service_json')
                if can and isinstance(can.get('pattern'),str) and can['pattern'] not in {'纯色','条纹','格纹','人字纹','波点','花卉印花','文字印花','字母印花','几何印花','动物纹','拼色'}:
                    can=dict(can);can['pattern']=None;parsed.setdefault('issues',[]).append('NONSTANDARD_PATTERN_ABSTAIN')
                res.update(raw_model_output=raw,roi_category=can.get('category') if can else None,roi_canonical_tags=can,issues=parsed.get('issues',[]),schema_valid=parsed.get('schema_valid',False),native_json=parsed.get('raw_native_json',False),status='GENERATED_UNVERIFIED',model_called=True,timing=t,roi_input=str(img))
                res['category_changed']=res['baseline_category']!=res['roi_category']
                res['roi_matches_review_human']=res['roi_category']==p['human_category_FOR_EVAL_ONLY']
            except Exception as ex:
                res['issues'].append(f'{type(ex).__name__}:{str(ex)[:200]}')
            putline(f,res);records.append(res)
            print(f'[{ix}/{len(plan)}] {p["case_id"]}: {res["status"]} => {res["roi_category"]}',flush=True)
    groups={}
    for cause in sorted(set(p['cause'] for p in plan)):
        x=[r for r in records if r['cause']==cause];success=[r for r in x if r['status']=='GENERATED_UNVERIFIED']
        groups[cause]={'n':len(x),'inference_success':len(success),'roi_agrees_reviewed_human':sum(r.get('roi_matches_review_human',False) for r in success),'category_changed':sum(r.get('category_changed',False) for r in success)}
    times=[r['timing']['end_to_end_ms'] for r in records if isinstance(r.get('timing'),dict)]
    summary={'schema':'prd32_dev_roi_target_binding_ablation_v1','status':'DEV_ROI_ABLATION_COMPLETED_NOT_PRD_ACCEPTED','requested':len(plan),'inference_completed':sum(r['model_called'] for r in records),'failed':sum(r['status']=='RUNTIME_FAILED' for r in records),'by_review_cause':groups,'median_roi_ms':round(statistics.median(times),2) if times else None,'model_load_seconds':load_sec,'human_gt_passed_to_model':False,'historical_GT_pilot_bbox_not_frozen_V5':True,'holdout_used':False,'baseline_independent_accuracy':False}
    dumpj(out/'summary.json',summary)
    print('PAIRED_ROI_DEV_DONE',json.dumps(summary,ensure_ascii=False),flush=True)

def self_test():
    from PIL import Image
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as td:
        td=Path(td); root=td/'fashion_data'/'raw'/'train'/'train'/'image';root.mkdir(parents=True)
        im=root/'fake.jpg';Image.new('RGB',(160,120),'white').save(im)
        dest=td/'roi.jpg';roi_red_box(im,[25,20,115,100],dest)
        with Image.open(dest) as result:
            assert result.size[0]<160 and result.size[1]<120
        assert is_valid_train(im,td/'fashion_data')
        assert not is_valid_train(td/'fashion_data'/'raw'/'test'/'foo.jpg',td/'fashion_data')
        assert len(sha(im))==64
        mod=loadhelper(HERE/'prd32_integrated_pipeline_20261027.py')
        assert mod.parse_tag_output('```json\n{"category":"top","primary_color":"白色","pattern":"无","visible_design":"圆领"}\n```')['parseable']
    print('ROI_PAIR_SELF_TEST_PASS')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--selected',default=DEFAULT_SELECTED);p.add_argument('--original-predictions',default=DEFAULT_ORIGINAL)
    p.add_argument('--review',default=str(HERE/'PRD32_train5_binding_review.json'))
    p.add_argument('--helper',default=str(HERE/'prd32_integrated_pipeline_20261027.py'))
    p.add_argument('--allowed-root',default=DEFAULT_ALLOWED);p.add_argument('--model',default=DEFAULT_MODEL)
    p.add_argument('--subset',choices=['binding','classification','all'],default='all')
    p.add_argument('--pad',type=float,default=.05);p.add_argument('--max-new-tokens',type=int,default=170)
    p.add_argument('--output',default='/workspace/PRD32_TRAIN5_PAIRED_ROI_20261027_run01')
    args=p.parse_args()
    if args.self_test:self_test()
    else:
        if not 0<=args.pad<=0.2:raise ValueError('pad must lie in 0..0.2')
        run(args,args.dry_run)
if __name__=='__main__': main()
