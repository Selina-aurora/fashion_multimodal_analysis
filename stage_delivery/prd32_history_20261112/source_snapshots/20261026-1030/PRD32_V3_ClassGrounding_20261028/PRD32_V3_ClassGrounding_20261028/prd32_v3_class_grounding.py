#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD 3.2, 2026-10-28: controlled TARGET CLASS prompt ablation.

- STRICTLY training-source development data only, no holdout/val.
- Original V2 run is read-only and SHA checked; no baseline rerun.
- Human/source labels never enter the model prompt or model record.
- Use V2 identical full-image + 5%-padding red rectangle for fair prompt comparison.
- First stage category_only; optional stage full_tags. NOT PRD acceptance.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,re,statistics,time
from pathlib import Path
from collections import Counter

HERE=Path(__file__).resolve().parent
DEFAULT_SELECTED='/workspace/PRD32_EXISTING_DATA_REVIEW_20261027_run01/train_dev_selected.csv'
DEFAULT_BASELINE='/workspace/PRD32_TRAIN5_GPU_ALL40_20261027_run01/predictions.jsonl'
DEFAULT_MODEL='/workspace/models/Qwen2.5-VL-7B-Instruct'
CATS={'top','pants','skirt','outerwear','dress','shoe','bag','accessory'}
BANNED={'holdout','val','validation','fullval','test','sealed','holdout_sealed','sealed_do_not_open'}
ROOT='/workspace/fashion_data'

# This prompt contains NO per-case ground-truth category or inventory/source class hint.
# Class-first instructions: return abstention if truly unresolvable, never default to top.
CLASS_RULES=(
    '你是服饰目标类别判别器。图中的红色矩形是必须识别的目标位置，红框内外可能有多件衣服。'
    '必须以红框指向的目标物件为依据，不要把红框覆盖到的上衣、内搭、背包或背景自动当作目标。'
    '先在内部观察目标的外形与相对身体部位（上身、腰腿、裙摆、全身、外层、鞋、包、配饰）。'
    '若红框所指是腰部以下的下装，绝对不要仅因同时看到一件上衣就输出top。'
    '若目标是外层开襟的夹克、西装外套或大衣，应考虑outerwear，不要因为里面有T恤就归为top。'
    '类别定义：top=T恤、衬衫、卫衣、针织上衣、背心等单独上装；'
    'pants=长裤、短裤或其他有裤腿结构的下装；skirt=裙装下半身、无连衣上身结构；'
    'dress=连衣裙、具有相连上衣和裙身的单件服饰；'
    'outerwear=外套、夹克、西服、大衣等外层服饰；shoe=鞋；bag=包；accessory=配饰。'
    '注意上下装同框、重叠、遮挡时，以红框的目标为准；不要依据背景或人物上另一件衣服推断。'
    '若可确认是下装但无法可靠区分裙子与裤子，可以输出JSON null，不能退回到top。'
    '不能看到充分证据时输出JSON null；不得凭来源标注或外部先验猜测。'
)
CLASS_ONLY_PROMPT=CLASS_RULES + (
    '只输出一个JSON对象，严格仅包含一个键category，其值只能是'
    '"top","pants","skirt","outerwear","dress","shoe","bag","accessory"或JSON null。'
    '不要解释、代码围栏、额外文字和额外字段。'
)
FULL_TAGS_PROMPT=CLASS_RULES + (
    '在内部优先确定目标category，再分析属于同一个目标的可见属性，绝不读取旁边另一件衣服的颜色或设计。'
    '只输出一个JSON对象，键必须且只能为category、primary_color、pattern、visible_design。'
    'category只能取top,pants,skirt,outerwear,dress,shoe,bag,accessory或JSON null；'
    'primary_color只能是可见的中文颜色或JSON null；'
    'pattern只能使用纯色、条纹、格纹、人字纹、波点、花卉印花、文字印花、字母印花、'
    '几何印花、动物纹、拼色之一，无法确定填JSON null；'
    'visible_design只能填写目标上可见的领型、袖型、裤型、裙摆、开襟、扣合、肩带、装饰等短词，'
    '多个以分号分隔，不清楚填JSON null。'
    '纽扣属于装饰，不是pattern；不得编造材质、工艺、品牌、功能。'
    '不要解释、Markdown围栏、额外文字及多余键。'
)
PATTERNS={'纯色','条纹','格纹','人字纹','波点','花卉印花','文字印花','字母印花','几何印花','动物纹','拼色'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rows_jsonl(p):
    vals=[]
    for i,line in enumerate(Path(p).read_text(encoding='utf-8-sig').splitlines(),1):
        if line.strip():
            j=json.loads(line)
            if not isinstance(j,dict):raise ValueError(f'JSONL {p} row {i} not object')
            vals.append(j)
    return vals

def allowed_image(p,root):
    r=Path(root).resolve(); p=Path(p).resolve()
    return p.is_relative_to(r) and 'train' in [x.lower() for x in p.parts] and not BANNED.intersection({x.lower() for x in p.parts})

def load_helper():
    path=HERE/'prd32_integrated_pipeline_20261027.py'
    spec=importlib.util.spec_from_file_location('prd32_v2_readonly_helper',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def read_review(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if obj.get('scope')!='train_dev_diagnostic_only_NOT_INDEPENDENT':raise ValueError('Unexpected review scope')
    if obj.get('schema')!='prd32_train5_binding_diagnostic_review_v1':raise ValueError('Unexpected human review schema')
    rows=obj['rows'];m={r['case_id']:r for r in rows}
    if len(m)!=len(rows):raise ValueError('Duplicate human review case IDs')
    return m

def visual_review(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if obj.get('scope')!='train_dev_not_independent':raise ValueError('Unexpected visual review scope')
    rows=obj['rows'];m={r['case_id']:r for r in rows}
    if len(m)!=len(rows):raise ValueError('Duplicate visual review IDs')
    return m

def build(args, check_files=True):
    from PIL import Image
    with open(args.selected,'r',newline='',encoding='utf-8-sig') as f: selected=list(csv.DictReader(f))
    baseline=rows_jsonl(args.baseline)
    bm={r.get('case_id'):r for r in baseline};review=read_review(args.review);vis=visual_review(args.visual_review)
    if len(selected)!=40 or len(bm)!=40 or len(baseline)!=40 or len({r['case_id'] for r in baseline})!=40:
        raise ValueError('Expected exactly the 40-case verified original train-dev cohort, not a changed cohort')
    if not all(r.get('case_id')==f'train_dev_{i:03d}' for i,r in enumerate(baseline,1)):
        raise ValueError('Unexpected baseline case order or IDs')
    if args.subset=='reviewed15':
        targets={k for k,v in review.items() if v.get('review_status')=='REVIEWED' and v.get('bbox_status')=='VALID'
                 and v.get('cause') in {'MODEL_TARGET_BINDING','MODEL_CLASSIFICATION'} and v.get('human_category') in CATS}
        if len(targets)!=15:raise ValueError(f'Expected 15 known errors, got {len(targets)}; do not silently change cohort')
    else: targets=set(bm)
    plans=[];seen=set()
    for i,r in enumerate(selected,1):
        cid=f'train_dev_{i:03d}'
        if cid not in targets:continue
        pred=bm[cid]
        p=Path(r.get('resolved_image_path','')).resolve()
        if r.get('decision')!='TRAIN_DEV_ONLY' or not allowed_image(p,args.allowed_root):raise ValueError(f'{cid}: prohibited source path/split')
        if r.get('inventory_category_NOT_GT') not in CATS:raise ValueError(f'{cid}: unsupported source hint')
        if str(p) in seen:raise ValueError(f'{cid}: duplicate source image')
        seen.add(str(p))
        if not p.is_file():raise FileNotFoundError(p)
        try:bbox=[float(r[f'bbox_{v}']) for v in ('x1','y1','x2','y2')]
        except Exception as e:raise ValueError(f'{cid}: malformed bbox') from e
        with Image.open(p) as img:
            w,h=img.size
            if w*h>45_000_000 or not all(math.isfinite(x) for x in bbox) or not (0<=bbox[0]<bbox[2]<=w and 0<=bbox[1]<bbox[3]<=h):
                raise ValueError(f'{cid}: invalid bbox or oversized image')
            img.verify()
        digest=sha(p)
        if digest != str(r.get('sha256','')).lower():raise ValueError(f'{cid}: image hash differs from selected manifest')
        if digest != str(pred.get('input_image_sha256','')).lower():raise ValueError(f'{cid}: image hash differs from V2 baseline')
        if pred.get('status')!='GENERATED_UNVERIFIED' or pred.get('task')!='tags' or pred.get('bbox_origin')!='historical_inventory_gt_pilot_NOT_V5_prediction':
            raise ValueError(f'{cid}: not a valid V2 train-dev tag baseline')
        base_cat=(pred.get('canonical_tags') or {}).get('category')
        if base_cat is not None and base_cat not in CATS:raise ValueError(f'{cid}: V2 baseline bad category')
        plans.append({'case_id':cid,'image_path':str(p),'image_sha256':digest,'bbox':bbox,
          'target_instance_id':str(r.get('garment_id') or r.get('record_id') or cid),
          'image_id':pred.get('image_id'), 'baseline_category':base_cat,
          'source_hint_NOT_GT':r.get('inventory_category_NOT_GT')})
    if len(plans)!=len(targets):raise ValueError('Selection/baseline case identity mismatch')
    # Human labels and visually ambiguous status are intentionally NOT placed into model record/plan.
    return plans,review,vis

def parse_category(raw):
    s=raw.strip();native=s.startswith('{') and s.endswith('}')
    s=re.sub(r'^```(?:json)?\s*','',s,count=1,flags=re.I)
    if s.rstrip().endswith('```'):s=s.rstrip()[:-3].strip()
    try:obj=json.loads(s)
    except Exception as e:return {'category':None,'schema_valid':False,'native_json':False,'issues':['JSON_PARSE_ERROR']}
    if not isinstance(obj,dict) or set(obj)!={'category'}:
        return {'category':None,'schema_valid':False,'native_json':native,'issues':['CATEGORY_SCHEMA_KEYS']}
    category=obj['category']
    if category is not None and category not in CATS:
        return {'category':None,'schema_valid':False,'native_json':native,'issues':['ILLEGAL_CATEGORY']}
    return {'category':category,'schema_valid':True,'native_json':native,'issues':[]}

def prompt_for_mode(mode):return CLASS_ONLY_PROMPT if mode=='category_only' else FULL_TAGS_PROMPT

def interpret(helper,mode,raw):
    if mode=='category_only':return parse_category(raw),None
    v=helper.parse_tag_output(raw)
    can=v.get('canonical_service_json')
    issues=list(v.get('issues') or [])
    if can and isinstance(can.get('pattern'),str) and can['pattern'] not in PATTERNS:
        can=dict(can);can['pattern']=None;issues.append('NONSTANDARD_PATTERN_ABSTAIN')
    return {'category':can.get('category') if can else None,'schema_valid':v['schema_valid'],'native_json':v['raw_native_json'],'issues':issues},can

def analyze(predictions,baseline,review,vis,mode):
    byid={p['case_id']:p for p in predictions}
    eligible=[];ambiguous=[];all_verified=[]
    for cid,v in review.items():
        if cid not in byid or v.get('bbox_status')!='VALID' or v.get('human_category') not in CATS or v.get('review_status')!='REVIEWED':continue
        if v.get('cause') not in {'MODEL_TARGET_BINDING','MODEL_CLASSIFICATION'}:continue
        rv=vis.get(cid,{})
        if rv.get('target_type_clear')=='NO': ambiguous.append(cid);continue
        if rv.get('target_type_clear')!='YES':continue
        eligible.append(cid)
    pairs=[]
    for cid in eligible:
        p=byid[cid]
        human=review[cid]['human_category']
        v2=baseline[cid]['canonical_tags']['category']
        v3=p.get('category')
        pairs.append({'case_id':cid,'human_reviewed_category_POSTHOC_ONLY':human,'v2_category':v2,'v3_category':v3,
                      'v2_match_human':v2==human,'v3_match_human':v3==human,'v3_issues':p.get('issues')})
    finished=[x for x in predictions if x.get('status')=='GENERATED_UNVERIFIED']
    confused=Counter(x.get('category') for x in finished)
    base_cnt=Counter(x.get('baseline_category') for x in predictions)
    result={'schema':'prd32_v3_class_grounding_dev_summary_v1','status':'DEV_DIAGNOSTIC_NOT_PRD_ACCEPTED',
          'mode':mode,'count':len(predictions),'model_success':len(finished),
          'format_valid':sum(x.get('schema_valid',False) for x in finished),
          'native_json':sum(x.get('native_json',False) for x in finished),
          'v2_top_rate_in_selected':round(base_cnt['top']/len(predictions),4) if predictions else None,
          'v3_top_rate_in_selected':round(confused['top']/len(finished),4) if finished else None,
          'v3_category_counts':{str(k):v for k,v in confused.items()},
          'posthoc_review_scored':len(pairs),'ambiguous_review_excluded':sorted(ambiguous),
          'v2_matches_review_on_scored':sum(x['v2_match_human'] for x in pairs),
          'v3_matches_review_on_scored':sum(x['v3_match_human'] for x in pairs),
          'paired':{'old_wrong_new_correct':sum(not x['v2_match_human'] and x['v3_match_human'] for x in pairs),
                    'old_correct_new_wrong':sum(x['v2_match_human'] and not x['v3_match_human'] for x in pairs),
                    'both_wrong':sum(not x['v2_match_human'] and not x['v3_match_human'] for x in pairs),
                    'both_correct':sum(x['v2_match_human'] and x['v3_match_human'] for x in pairs)},
          'scope':'TRAIN_DEV_DIAGNOSTIC_ONLY', 'selected_historical_error_cases':True,
          'human_labels_in_model_prompt':False,'source_hint_in_model_prompt':False,
          'bbox_origin':'HISTORICAL_GT_PILOT_NOT_FROZEN_V5','holdout_used':False,
          'formal_accuracy':None,'full_PRD_acceptance':False}
    times=[x.get('timing',{}).get('end_to_end_ms') for x in finished if isinstance(x.get('timing'),dict)]
    times=[t for t in times if isinstance(t,(int,float))]
    result['median_latency_ms']=round(statistics.median(times),2) if times else None
    return result,pairs

def preflight(args, write_files=False):
    plans,review,vis=build(args)
    prompt=prompt_for_mode(args.mode)
    checks={'schema':'prd32_v3_train_dev_preflight_v1','status':'PREFLIGHT_OK','subset':args.subset,
      'mode':args.mode,'selected_cases':len(plans),'case_ids':[p['case_id'] for p in plans],
      'image_hashes_verified':True,'input_method':'marked_full_image_5percent_padding',
      'selected_manifest_sha256':sha(args.selected),'v2_baseline_sha256':sha(args.baseline),
      'binding_review_sha256':sha(args.review),'visual_review_sha256':sha(args.visual_review),
      'prompt_sha256':hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
      'model_not_loaded':True,'human_labels_sent_to_model':False,'source_hint_sent_to_model':False,
      'holdout_used':False,'historical_GT_pilot_not_frozen_V5':True}
    if write_files:
        o=Path(args.output)
        if o.exists():raise FileExistsError(o)
        o.mkdir(parents=True)
        write(o/'preflight_summary.json',checks)
        with (o/'case_plan.jsonl').open('w',encoding='utf-8') as f:
            for r in plans:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print('PREFLIGHT_OK',json.dumps({k:checks[k] for k in ['subset','mode','selected_cases','image_hashes_verified','model_not_loaded']},ensure_ascii=False),flush=True)
    return plans,review,vis,checks

def run(args):
    plans,review,vis,checks=preflight(args)
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    modeldir=Path(args.model)
    if not (modeldir/'config.json').exists():raise FileNotFoundError(modeldir/'config.json')
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU missing')
    free=torch.cuda.mem_get_info()[0]/(1024**3)
    if free<17:raise RuntimeError(f'Need >=17GiB free VRAM; only {free:.2f}GiB')
    helper=load_helper()
    prompt=prompt_for_mode(args.mode)
    out.mkdir(parents=True); (out/'marked_images').mkdir()
    write(out/'run_config.json',{**checks,'model':args.model,'max_tokens':args.max_tokens,'model_not_loaded':False,'prompt_text':prompt})
    t0=time.perf_counter()
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(modeldir),local_files_only=True,
            device_map='auto',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    processor=AutoProcessor.from_pretrained(str(modeldir),local_files_only=True,max_pixels=512*28*28)
    load_s=round(time.perf_counter()-t0,2)
    prior={x['case_id']:x for x in rows_jsonl(args.baseline)}
    results=[]
    with (out/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for i,plan in enumerate(plans,1):
            cid=plan['case_id']
            res={'case_id':cid,'status':'RUNTIME_FAILED','category':None,'baseline_category':plan['baseline_category'],
                 'schema_valid':False,'native_json':False,'issues':[],'raw_model_output':None,'canonical_tags':None,
                 'timing':None,'input_image_sha256':plan['image_sha256'],'scope':'TRAIN_DEV_DIAGNOSTIC_ONLY'}
            try:
                # Deliberate no source_hint, human_category, review cause or baseline category in model input.
                modelrec={'case_id':cid,'task':'tags','image_path':plan['image_path'],
                    'image_sha256':plan['image_sha256'],'target_bbox_xyxy':plan['bbox'],
                    'target_instance_id':plan['target_instance_id']}
                img=helper.make_marked_image(modelrec,out/'marked_images','marked_full_image',0.05)
                raw,t=helper.run_model_case(model,processor,modelrec,img,prompt,task_limits={'tags':args.max_tokens})
                parsed,can=interpret(helper,args.mode,raw)
                res.update(status='GENERATED_UNVERIFIED',category=parsed['category'],canonical_tags=can,
                    raw_model_output=raw,timing=t,issues=parsed['issues'],schema_valid=parsed['schema_valid'],
                    native_json=parsed['native_json'],marked_image=str(img))
            except Exception as ex:
                res['issues']=[f'RUNTIME_ERROR:{type(ex).__name__}:{str(ex)[:180]}']
            f.write(json.dumps(res,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
            results.append(res)
            print(f'[{i}/{len(plans)}] {cid} | {res["status"]} | {res["category"]}',flush=True)
    summ,pairs=analyze(results,prior,review,vis,args.mode)
    summ['model_load_seconds']=load_s
    summ['failed']=sum(x['status']!='GENERATED_UNVERIFIED' for x in results)
    write(out/'summary.json',summ)
    with (out/'reviewed_pairs.jsonl').open('w',encoding='utf-8') as f:
        for p in pairs:f.write(json.dumps(p,ensure_ascii=False)+'\n')
    print('V3_DEV_DONE',json.dumps({k:summ[k] for k in ['mode','count','model_success','posthoc_review_scored','ambiguous_review_excluded','paired','v3_top_rate_in_selected']},ensure_ascii=False),flush=True)

def self_test():
    from tempfile import TemporaryDirectory
    from PIL import Image
    helper=load_helper()
    with TemporaryDirectory() as td:
        root=Path(td); imd=root/'fashion_data'/'raw'/'train'/'train'/'image';imd.mkdir(parents=True)
        image=imd/'one.jpg'; Image.new('RGB',(120,160),'white').save(image)
        assert allowed_image(image,root/'fashion_data')
        assert not allowed_image(root/'fashion_data'/'raw'/'test'/'one.jpg',root/'fashion_data')
        assert len(sha(image))==64
        m={'task':'tags','case_id':'x','image_path':str(image),'image_sha256':sha(image),'target_bbox_xyxy':[20,20,80,90],'target_instance_id':'item_1'}
        marked=helper.make_marked_image(m,root,'marked_full_image',0.05)
        assert marked.is_file()
        assert parse_category('```json\n{"category":"pants"}\n```')['category']=='pants'
        assert parse_category('{"category":"shirt"}')['schema_valid'] is False
        assert parse_category('{"category":null}')['schema_valid'] is True
        assert 'source_hint' not in CLASS_ONLY_PROMPT and 'human_category' not in FULL_TAGS_PROMPT
        a=[{'case_id':'x','status':'GENERATED_UNVERIFIED','category':'pants','baseline_category':'top','schema_valid':True,'native_json':True,'timing':{'end_to_end_ms':1000}}]
        b={'x':{'canonical_tags':{'category':'top'}}}
        review={'x':{'bbox_status':'VALID','review_status':'REVIEWED','cause':'MODEL_TARGET_BINDING','human_category':'pants'}}
        summary,pairs=analyze(a,b,review,{'x':{'target_type_clear':'YES'}},'category_only')
        assert summary['paired']['old_wrong_new_correct']==1
        summary,pairs=analyze(a,b,review,{'x':{'target_type_clear':'NO'}},'category_only')
        assert summary['posthoc_review_scored']==0 and summary['ambiguous_review_excluded']==['x']
        assert interpret(helper,'full_tags','```json\n{"category":"dress","primary_color":"白色","pattern":"无","visible_design":"无袖"}\n```')[1]['pattern'] is None
    print('V3_CLASS_GROUNDING_SELF_TEST_PASS')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('self-test')
    for name in ('preflight','run'):
        p=sub.add_parser(name)
        p.add_argument('--selected',default=DEFAULT_SELECTED)
        p.add_argument('--baseline',default=DEFAULT_BASELINE)
        p.add_argument('--review',default=str(HERE/'binding_review_19.json'))
        p.add_argument('--visual-review',default=str(HERE/'roi15_human_review_corrected.json'))
        p.add_argument('--allowed-root',default=ROOT)
        p.add_argument('--subset',choices=['reviewed15','all40'],default='reviewed15')
        p.add_argument('--mode',choices=['category_only','full_tags'],default='category_only')
        p.add_argument('--model',default=DEFAULT_MODEL)
        p.add_argument('--max-tokens',type=int,default=None)
        p.add_argument('--output',required=True)
    args=ap.parse_args()
    if args.command!='self-test' and args.max_tokens is None:
        args.max_tokens=90 if args.mode=='category_only' else 170
    if args.command=='self-test':self_test()
    elif args.command=='preflight':preflight(args,write_files=True)
    elif args.command=='run':run(args)
if __name__=='__main__':main()
