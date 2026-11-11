#!/usr/bin/env python3
# coding: utf-8
"""PRD3.2 TRAIN-only 5-class DEV diagnostics. NOT independent PRD acceptance.

Uses the *already generated* 40-image train_dev_selected.csv. Model receives
only marked source images (never the inventory category or human GT); source
bbox is from historical inventory/GT-pilot and is NOT claimed as frozen V5.

Subcommands: preflight (no GPU), run (Qwen GPU), score (human GT-only, no GPU).
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, statistics, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

CATS = ('top','pants','skirt','outerwear','dress','shoe','bag','accessory')
PROTECTED = ('holdout','sealed','fullval','validation','valid','val','test')
SELECTED_DEFAULT = '/workspace/PRD32_EXISTING_DATA_REVIEW_20261027_run01/train_dev_selected.csv'
V1_DEFAULT = '/workspace/prd32_integrated_pipeline_20261027.py'
MODEL_DEFAULT = '/workspace/models/Qwen2.5-VL-7B-Instruct'

def utc(): return datetime.now(timezone.utc).isoformat()
def jwrite(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def append_jsonl(f,v): f.write(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def safe_path(p, allowed_root):
    root=Path(allowed_root).resolve(); path=Path(p).resolve()
    if not path.is_relative_to(root):return False
    parts={x.lower() for x in path.parts}
    if 'train' not in parts:return False
    if parts.intersection(PROTECTED):return False
    return True

def prepare(selected_path,allowed_root):
    from PIL import Image
    path=Path(selected_path)
    if not path.is_file():raise FileNotFoundError(path)
    with path.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    if not rows:raise ValueError('No selected rows. Finish existing pool preview first.')
    result=[];seen_images=set();decision=[]
    for ix,r in enumerate(rows,1):
        issues=[]; image_path=r.get('resolved_image_path','')
        cat=r.get('inventory_category_NOT_GT','')
        if r.get('decision')!='TRAIN_DEV_ONLY':issues.append('NOT_TRAIN_DEV')
        if cat not in CATS:issues.append('CATEGORY_HINT_NOT_ALLOWED')
        if not image_path or not safe_path(image_path,allowed_root):issues.append('NOT_VERIFIED_TRAIN_SOURCE')
        if image_path in seen_images:issues.append('REUSED_IMAGE_IN_THIS_BATCH')
        seen_images.add(image_path)
        box=[]
        try:
            box=[float(r[f'bbox_{v}']) for v in ('x1','y1','x2','y2')]
            if any(not math.isfinite(x) for x in box) or not (0<=box[0]<box[2] and 0<=box[1]<box[3]):issues.append('INVALID_BBOX')
        except (KeyError,TypeError,ValueError):issues.append('INVALID_BBOX')
        p=Path(image_path)
        if not issues:
            try:
                if not p.is_file():issues.append('MISSING_IMAGE')
                else:
                    with Image.open(p) as im:
                        w,h=im.size
                        if w*h>45_000_000:issues.append('IMAGE_TOO_LARGE')
                        if box[2]>w or box[3]>h:issues.append('BBOX_OUTSIDE_REAL_IMAGE')
                        im.verify()
                    hs=r.get('sha256','').lower().strip()
                    if not hs:issues.append('MISSING_PRIOR_IMAGE_SHA256')
                    elif sha256(p)!=hs:issues.append('SOURCE_SHA_CHANGED')
            except (OSError,ValueError) as e:issues.append('IMAGE_UNREADABLE_'+type(e).__name__)
        sample=f'train_dev_{ix:03d}'
        obj={'case_id':sample,'task':'tags','source_split':'train','evaluation_scope':'TRAIN_DEV_DIAGNOSTIC_ONLY',
             'image_path':image_path,'image_id':p.stem,'target_instance_id':str(r.get('garment_id') or r.get('record_id') or sample),
             'image_sha256':r.get('sha256'),'target_bbox_xyxy':box,
             'bbox_origin':'historical_inventory_gt_pilot_NOT_V5_prediction',
             'hint_category_NOT_GT':cat,'source_record_id':r.get('record_id','')}
        decision.append({'case_id':sample,'status':'PASS' if not issues else 'BLOCKED','issues':issues})
        if not issues: result.append(obj)
    return result,decision

def make_helper_record(row):
    """Adapter for the existing 3.2 V1 image helper; only verified train-dev row fields.

    Do not forward the inventory category/GT to the model.
    """
    required = ('case_id','image_path','image_sha256','target_bbox_xyxy','target_instance_id')
    for key in required:
        if not row.get(key):
            raise ValueError(f'Missing verified input field before model: {key}')
    return {'task':'tags','case_id':row['case_id'],
            'image_path':row['image_path'],
            'image_sha256':row['image_sha256'],
            'target_bbox_xyxy':row['target_bbox_xyxy'],
            'target_instance_id':row['target_instance_id']}


def get_v1(helperpath):
    import importlib.util
    p=Path(helperpath)
    if not p.is_file():raise FileNotFoundError(p)
    spec=importlib.util.spec_from_file_location('prd32_v2_fixed_prompt_helper',p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def preflight(args):
    rows,decisions=prepare(args.selected,args.allowed_root)
    summary={'schema':'prd32_train_five_class_dev_preflight_v1','selected_source':args.selected,
        'input_rows':len(decisions),'eligible':len(rows),'blocked':len(decisions)-len(rows),
        'by_source_hint_NOT_GT':dict(Counter(x['hint_category_NOT_GT'] for x in rows)),
        'source_is_historical_training':True,'bbox_origin':'inventory_GT_pilot_not_frozen_V5',
        'human_gt_not_used':True,'independent_accuracy_eligible':False,'holdout_accessed':False,
        'model_loaded':False}
    if args.output:
        o=Path(args.output)
        if o.exists():raise FileExistsError(o)
        o.mkdir(parents=True)
        jwrite(o/'preflight_summary.json',summary)
        with (o/'input_decisions.jsonl').open('w',encoding='utf-8') as f:
            for row in decisions:append_jsonl(f,row)
    print('TRAIN_DEV_PREFLIGHT',json.dumps(summary,ensure_ascii=False),flush=True)
    if not rows:raise ValueError('No eligible train-dev inputs')
    return rows,summary

def run(args):
    prepared,summary=preflight(argparse.Namespace(selected=args.selected,allowed_root=args.allowed_root,output=None))
    if args.max_cases is not None:prepared=prepared[:args.max_cases]
    if not prepared:raise ValueError('No selected inputs')
    o=Path(args.output)
    if o.exists():raise FileExistsError(o)
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available():raise RuntimeError('GPU CUDA unavailable')
    model_dir=Path(args.model)
    if not (model_dir/'config.json').is_file():raise FileNotFoundError(model_dir/'config.json')
    free=torch.cuda.mem_get_info()[0]/1024**3
    if free<17:raise RuntimeError(f'Need about 17GiB free GPU for this model; only {free:.2f} GiB. Inspect existing GPU processes.')
    v1=get_v1(args.v1_helper)
    o.mkdir(parents=True);imgroot=o/'marked_images';imgroot.mkdir()
    jwrite(o/'run_config.json',{'schema':'prd32_train5_dev_config_v1','scope':'TRAIN_DEV_DIAGNOSTIC_ONLY',
           'source_selected_sha256':sha256(args.selected),'source':args.selected,'model':args.model,
           'v2_prompt_sha256':hashlib.sha256(v1.TAG_PROMPT.encode('utf-8')).hexdigest(),
           'max_new_tokens':args.max_new_tokens,'max_cases':args.max_cases,
           'source_bbox_is_not_v5_prediction':True,'human_gt_passed_to_model':False,
           'holdout_used':False,'prd31_modified':False})
    start=time.perf_counter()
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_dir),local_files_only=True,
                  device_map='auto',dtype=dtype,attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    processor=AutoProcessor.from_pretrained(str(model_dir),local_files_only=True,max_pixels=512*28*28)
    load_s=round(time.perf_counter()-start,2)
    good=[];status=Counter();ms=[]
    with (o/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for ix,row in enumerate(prepared,1):
            result={'case_id':row['case_id'],'task':'tags','image_id':row['image_id'],
              'target_instance_id':row['target_instance_id'],'bbox_origin':row['bbox_origin'],
              'source_split':'train','input_image_sha256':row['image_sha256'],
              'source_hint_NOT_GT':row['hint_category_NOT_GT'],
              'status':'STARTED','raw_model_output':None,'canonical_tags':None,'issues':[],
              'accuracy_evaluated':False,'inference_performed':False}
            try:
                rec=make_helper_record(row)
                marked=v1.make_marked_image(rec,imgroot,'marked_full_image',0.05)
                raw,t=v1.run_model_case(model,processor,rec,marked,v1.TAG_PROMPT,
                                      task_limits={'tags':args.max_new_tokens})
                p=v1.parse_tag_output(raw)
                can=p.get('canonical_service_json')
                if can:
                    from copy import deepcopy
                    can=deepcopy(can)
                    val=can.get('pattern')
                    allowed={'纯色','条纹','格纹','人字纹','波点','花卉印花','文字印花','字母印花','几何印花','动物纹','拼色'}
                    if val is not None and val not in allowed:
                        can['pattern']=None;p['issues'].append('NONSTANDARD_PATTERN_ABSTAIN')
                result.update(status='GENERATED_UNVERIFIED',raw_model_output=raw,canonical_tags=can,
                   issues=p.get('issues',[]),native_json=p.get('raw_native_json',False),
                   schema_valid=p.get('schema_valid',False),timing=t,inference_performed=True,
                   marked_image=str(marked))
                ms.append(t['end_to_end_ms'])
            except Exception as e:
                result.update(status='RUNTIME_FAILED',issues=[type(e).__name__+':'+str(e)[:100]])
            append_jsonl(f,result);good.append(result);status[result['status']]+=1
            print(f'[{ix}/{len(prepared)}] {row["case_id"]}: {result["status"]}',flush=True)
    summ={'schema':'prd32_train5_tags_dev_run_v1','status':'TRAIN_DEV_RUN_COMPLETED_NOT_PRD_ACCEPTED' if any(x['inference_performed'] for x in good) else 'TRAIN_DEV_RUN_ALL_FAILED',
         'inputs':len(prepared),'model_load_seconds':load_s,'status_counts':dict(status),
         'median_end_to_end_ms':round(statistics.median(ms),2) if ms else None,
         'generated':sum(x['inference_performed'] for x in good),'native_json_count':sum(x.get('native_json',False) for x in good),
         'strict_schema_count':sum(x.get('schema_valid',False) for x in good),
         'verified_accuracy':None,'scope':'TRAIN_DEV_DIAGNOSTIC_ONLY','model_input_GT':False,
         'bbox_from_frozen_V5':False,'independent_PRD_acceptance':False,'holdout_used':False}
    jwrite(o/'summary.json',summ)
    print('TRAIN_DEV_INFERENCE_COMPLETE',json.dumps(summ,ensure_ascii=False),flush=True)
    if not summ['generated']:
        raise RuntimeError('All selected inference cases failed; inspect predictions.jsonl. No QA/accuracy metrics are valid.')

def score(args):
    p=Path(args.predictions);g=Path(args.human_review)
    pred=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
    user=json.loads(g.read_text(encoding='utf-8-sig'))
    if user.get('scope')!='train_dev_diagnostic_only_NOT_INDEPENDENT':
        raise ValueError('Expected exported TRAIN DEV human review JSON, not independent test')
    gt={x['sample_id']:x for x in user['rows']}
    breakdown=defaultdict(lambda:Counter())
    evaluated=0;excluded=0
    for r in pred:
        row=gt.get(r['case_id']);can=r.get('canonical_tags')
        if not row or row.get('review_status')!='REVIEWED' or row.get('bbox_status')!='VALID' or not isinstance(can,dict):
            excluded+=1;continue
        evaluated+=1
        for field,pkey,hkey in [('category','category','human_category'),('color','primary_color','human_color'),('pattern','pattern','human_pattern')]:
            actual=(row.get(hkey) or '').strip();candidate=can.get(pkey)
            if actual and actual not in ('未知','无法确定','不确定'):
                breakdown[field]['denominator']+=1
                if isinstance(candidate,str) and candidate.strip()==actual:breakdown[field]['correct']+=1
        breakdown['by_class:'+str(row.get('human_category',''))]['n']+=1
    out={'schema':'prd32_train5_human_score_dev_v1','reviewed_with_bbox_valid':evaluated,'excluded':excluded,
         'metric':{k:{**dict(v),'accuracy':round(v['correct']/v['denominator'],4) if v.get('denominator') else None} for k,v in breakdown.items()},
         'explicit_warning':'TRAIN DATA DEV DIAGNOSTIC ONLY. Not independent 8-class or PRD acceptance.'}
    jwrite(args.output,out);print('TRAIN_DEV_SCORE_COMPLETE',json.dumps(out,ensure_ascii=False))

def selftest():
    from PIL import Image
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as root:
        p=Path(root); data=p/'fashion_data'/'raw'/'train'/'train'/'image';data.mkdir(parents=True)
        Image.new('RGB',(100,100),'blue').save(data/'1.jpg')
        good={'decision':'TRAIN_DEV_ONLY','resolved_image_path':str(data/'1.jpg'),
            'inventory_category_NOT_GT':'top','bbox_x1':5,'bbox_y1':5,'bbox_x2':80,'bbox_y2':90,
            'sha256':sha256(data/'1.jpg'),'record_id':'test1','garment_id':'item1'}
        bad={**good,'resolved_image_path':str(p/'suspicious'/'test'/'2.jpg'),'garment_id':'item2'}
        selection=p/'selected.csv'
        with selection.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=good.keys());w.writeheader();w.writerows([good,bad])
        got,decision=prepare(selection,p/'fashion_data')
        assert len(got)==1 and decision[0]['status']=='PASS' and decision[1]['status']=='BLOCKED',decision
        assert got[0]['bbox_origin']=='historical_inventory_gt_pilot_NOT_V5_prediction'
        rec=make_helper_record(got[0])
        assert rec['image_sha256']==sha256(data/'1.jpg')
        # Exact prior failure occurred within the imported V1 image helper.
        helper=get_v1(Path(__file__).parent/'prd32_integrated_pipeline_20261027.py')
        mark_dir=p/'marked';mark_dir.mkdir()
        marked=helper.make_marked_image(rec,mark_dir,'marked_full_image',0.05)
        assert marked.is_file(), 'Bridge image helper failed'
        assert not safe_path(p/'fashion_data'/'raw'/'holdout'/'train'/'x.jpg',p/'fashion_data')
    print('TRAIN_DEV_FIVECLASS_SELF_TEST_PASS')

def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest='command',required=True)
    s=sub.add_parser('self-test')
    s=sub.add_parser('preflight');s.add_argument('--selected',default=SELECTED_DEFAULT);s.add_argument('--allowed-root',default='/workspace/fashion_data');s.add_argument('--output',required=True)
    s=sub.add_parser('run');s.add_argument('--selected',default=SELECTED_DEFAULT);s.add_argument('--allowed-root',default='/workspace/fashion_data');s.add_argument('--output',required=True)
    s.add_argument('--v1-helper',default=V1_DEFAULT);s.add_argument('--model',default=MODEL_DEFAULT)
    s.add_argument('--max-new-tokens',type=int,default=170);s.add_argument('--max-cases',type=int,default=None)
    s=sub.add_parser('score');s.add_argument('--predictions',required=True);s.add_argument('--human-review',required=True);s.add_argument('--output',required=True)
    a=ap.parse_args()
    if a.command=='self-test':selftest()
    elif a.command=='preflight':preflight(a)
    elif a.command=='run':run(a)
    elif a.command=='score':score(a)
    return 0
if __name__=='__main__':raise SystemExit(main())
