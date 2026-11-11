#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD 3.2, 2026-10-28: symmetric TARGET INSTANCE focused V4 prompt ablation.

- STRICTLY training-source development data only, no holdout/val.
- Original V2 run is read-only and SHA checked; no baseline rerun.
- Human/source labels never enter the model prompt or model record.
- Use V2 identical full-image + 5%-padding red rectangle for fair prompt comparison.
- Run reviewed19 first (18 scored, one ambiguous); optionally all40. NOT PRD acceptance.
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
    '你是服饰实例目标类别判别器，不是全图服饰清单分类器。'
    '红色矩形对应本次要识别的目标服饰实例，但矩形内部可能有其他服饰局部重叠。'
    '先结合完整原图判断哪一件独立服饰在位置、边缘、轮廓和覆盖范围上与红框目标最匹配；'
    '明确区分目标服饰和仅仅穿过红框的其他服饰局部。'
    '一个矩形里出现绿色裙子、红裙、裤腿、T恤、外套内搭，并不意味着它们就是目标；'
    '不要因为某个干扰服饰颜色鲜艳、纹理清晰或者只看局部像裤装，就忽略红框主要目标。'
    '上衣为红框主体且仅有裙子或裤子一角进入框时，仍应选择top；'
    '裤装、裙装或外套为红框主体而有上衣或内搭进入框时，应选择相应的目标类别。'
    '这条规则是双向对称的；绝对不要固定偏向top、pants、skirt或者其他任何类别。'
    '基于该目标的连续形状、服饰功能结构和相对穿着位置判别，不能只凭某一小块颜色或者衣边。'
    'top=独立T恤、衬衫、卫衣、针织衫、背心等上衣；'
    'pants=有裤腿分叉的短裤或长裤；skirt=独立半身裙；'
    'dress=上衣和裙身连接成一件的连衣裙；outerwear=叠穿在内搭之外的夹克、西装、大衣等。'
    'shoe=鞋；bag=包；accessory=配饰。'
    'outerwear与top主要看是否存在独立外层结构、开襟前片或叠穿证据；'
    'dress与top主要看是否存在相连裙身，不能仅因为没拍全裙摆就直接猜top。'
    '确实无法依据可见证据区分类别时输出JSON null，不得为了凑完整度猜类别。'
)
CLASS_ONLY_PROMPT=CLASS_RULES + (
    '只返回单个JSON对象，键必须且只能有category，值只能是'
    '"top","pants","skirt","outerwear","dress","shoe","bag","accessory"或JSON null。'
    '不要解释、Markdown、代码围栏和额外字符。'
)
FULL_TAGS_PROMPT=CLASS_RULES + (
    '先判别红框所指目标类别，再只针对同一个目标生成可见属性；'
    '严禁借用框内其他衣物的颜色、图案或款式。'
    '只输出单个JSON对象，键必须且只能有category、primary_color、pattern、visible_design。'
    'category取八类之一或JSON null，primary_color取可见中文颜色或JSON null，'
    'pattern只允许纯色、条纹、格纹、人字纹、波点、花卉印花、文字印花、字母印花、'
    '几何印花、动物纹、拼色之一或JSON null。'
    'visible_design只写同一目标可见的领型、袖型、裤型、裙摆、开襟、扣合、肩带或装饰短语；'
    '多个短语用分号隔开，不清楚用JSON null；纽扣按装饰处理，不能编造面料、工艺、品牌。'
    '不要解释、代码围栏、额外字符。'
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
    spec=importlib.util.spec_from_file_location('prd32_v4_readonly_helper',path)
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

def extra_review(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if obj.get('schema')!='prd32_v3_40_case_additive_visual_review_v1' or obj.get('scope')!='train_dev_diagnostic_only_NOT_INDEPENDENT':
        raise ValueError('Unexpected new human review scope/schema')
    rows=obj.get('rows')
    if not isinstance(rows,list) or len(rows)!=4 or {x.get('case_id') for x in rows}!={'train_dev_001','train_dev_003','train_dev_005','train_dev_008'}:
        raise ValueError('Expected exactly four reviewed top cases; do not change this cohort')
    for x in rows:
        if (x.get('human_category')!='top' or x.get('bbox_coverage')!='YES' or
            x.get('visually_identifiable')!='YES' or x.get('other_garments_visible')!='YES'):
            raise ValueError('Unexpected reviewed top-case annotation: '+str(x.get('case_id')))
    return {x['case_id']:x for x in rows}

def verified_v3(path, base):
    xs=rows_jsonl(path)
    if len(xs)!=40 or len({x.get('case_id') for x in xs})!=40:
        raise ValueError('V3 full_tags baseline must contain exact original 40 cases')
    m={x['case_id']:x for x in xs}
    if set(m)!=set(base):raise ValueError('V3/V2 cohorts differ')
    for cid,x in m.items():
        if (x.get('status')!='GENERATED_UNVERIFIED' or
            x.get('input_image_sha256')!=base[cid].get('input_image_sha256')):
            raise ValueError(f'{cid}: V3 output/hash mismatch')
        cat=x.get('category')
        if cat is not None and cat not in CATS:raise ValueError(f'{cid}: bad V3 category')
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
    extra=extra_review(args.extra_review)
    verified_v3(args.v3_full_tags,bm)
    if args.subset=='reviewed19':
        targets={k for k,v in review.items() if v.get('review_status')=='REVIEWED' and v.get('bbox_status')=='VALID'
                 and v.get('cause') in {'MODEL_TARGET_BINDING','MODEL_CLASSIFICATION'} and v.get('human_category') in CATS}
        if len(targets)!=15 or (set(extra)&targets):raise ValueError('Expected 15 historical cases and four disjoint top reviews')
        targets.update(extra)
        if len(targets)!=19:raise ValueError('Expected 19 focused cases')
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

def analyze(predictions,baseline,review,vis,mode,v3_full_tags,extra):
    byid={p['case_id']:p for p in predictions}
    v3=verified_v3(v3_full_tags,baseline)
    refs={}; ambiguous=[]
    for cid,v in review.items():
        if cid not in byid or v.get('bbox_status')!='VALID' or v.get('human_category') not in CATS or v.get('review_status')!='REVIEWED':continue
        if v.get('cause') not in {'MODEL_TARGET_BINDING','MODEL_CLASSIFICATION'}:continue
        rv=vis.get(cid,{})
        if rv.get('target_type_clear')=='NO':ambiguous.append(cid);continue
        if rv.get('target_type_clear')!='YES':continue
        refs[cid]=(v['human_category'],'historical14')
    if len([x for x in refs if x.startswith('train_dev_')])!=14:
        raise ValueError('Expected 14 clearly reviewed historical error cases')
    for cid,v in extra.items():
        if cid in byid:
            if cid in refs:raise ValueError('Duplicate human reference')
            refs[cid]=(v['human_category'],'new_regression4')
    pairs=[]
    for cid in sorted(refs):
        h,origin=refs[cid]
        old=(baseline[cid].get('canonical_tags') or {}).get('category')
        prev=v3[cid].get('category')
        now=byid[cid].get('category')
        pairs.append({'case_id':cid,'reviewed_human_category_POSTHOC_ONLY':h,'review_set':origin,
            'v2_category':old,'v3_full_tags_category':prev,'v4_category':now,
            'v2_correct':old==h,'v3_full_tags_correct':prev==h,'v4_correct':now==h,
            'v4_schema_valid':byid[cid].get('schema_valid',False),
            'v4_issues':byid[cid].get('issues')})
    def compare(oldkey):
        return {'old_wrong_new_correct':sum(not x[oldkey] and x['v4_correct'] for x in pairs),
            'old_correct_new_wrong':sum(x[oldkey] and not x['v4_correct'] for x in pairs),
            'both_wrong':sum(not x[oldkey] and not x['v4_correct'] for x in pairs),
            'both_correct':sum(x[oldkey] and x['v4_correct'] for x in pairs)}
    done=[x for x in predictions if x.get('status')=='GENERATED_UNVERIFIED']
    categories=Counter(x.get('category') for x in done)
    summary={
        'schema':'prd32_v4_symmetric_target_focus_dev_summary_v1',
        'status':'DEV_DIAGNOSTIC_NOT_PRD_ACCEPTED',
        'mode':mode,'count':len(predictions),'model_success':len(done),
        'format_valid':sum(bool(x.get('schema_valid')) for x in done),
        'native_json':sum(bool(x.get('native_json')) for x in done),
        'v4_category_counts':{str(k):v for k,v in categories.items()},
        'v2_top_rate_in_selected':round(sum(baseline[x['case_id']]['canonical_tags']['category']=='top' for x in predictions)/len(predictions),4),
        'v3_top_rate_in_selected':round(sum(v3[x['case_id']].get('category')=='top' for x in predictions)/len(predictions),4),
        'v4_top_rate_in_selected':round(categories['top']/len(done),4) if done else None,
        'human_scored':len(pairs),'expected_scored_for_reviewed19':18,
        'historical_ambiguous_excluded':sorted(set(ambiguous)),
        'v2_matches_human':sum(x['v2_correct'] for x in pairs),
        'v3_full_tags_matches_human':sum(x['v3_full_tags_correct'] for x in pairs),
        'v4_matches_human':sum(x['v4_correct'] for x in pairs),
        'v2_to_v4':compare('v2_correct'),'v3_to_v4':compare('v3_full_tags_correct'),
        'new_regression4_v4_correct':sum(x['v4_correct'] for x in pairs if x['review_set']=='new_regression4'),
        'historical14_v4_correct':sum(x['v4_correct'] for x in pairs if x['review_set']=='historical14'),
        'scope':'TRAIN_DEV_DIAGNOSTIC_ONLY','selected_historical_error_cases':len(predictions)==19,
        'human_labels_in_model_prompt':False,'source_hint_in_model_prompt':False,
        'bbox_origin':'HISTORICAL_GT_PILOT_NOT_FROZEN_V5','holdout_used':False,
        'formal_accuracy':None,'full_PRD_acceptance':False,
        'warning':'Do not pick a winning prompt based on the same repeatedly tuned 18 training-source cases.'}
    times=[x.get('timing',{}).get('end_to_end_ms') for x in done if isinstance(x.get('timing'),dict)]
    times=[t for t in times if isinstance(t,(int,float))]
    summary['median_latency_ms']=round(statistics.median(times),2) if times else None
    return summary,pairs

def preflight(args, write_files=False):
    plans,review,vis=build(args)
    prompt=prompt_for_mode(args.mode)
    checks={'schema':'prd32_v4_train_dev_preflight_v1','status':'PREFLIGHT_OK','subset':args.subset,
      'mode':args.mode,'selected_cases':len(plans),'case_ids':[p['case_id'] for p in plans],
      'image_hashes_verified':True,'input_method':'marked_full_image_5percent_padding',
      'selected_manifest_sha256':sha(args.selected),'v2_baseline_sha256':sha(args.baseline),
      'binding_review_sha256':sha(args.review),'visual_review_sha256':sha(args.visual_review),
      'additional_review_sha256':sha(args.extra_review),'v3_full_tags_sha256':sha(args.v3_full_tags),
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
    summ,pairs=analyze(results,prior,review,vis,args.mode,args.v3_full_tags,extra_review(args.extra_review))
    summ['model_load_seconds']=load_s
    summ['failed']=sum(x['status']!='GENERATED_UNVERIFIED' for x in results)
    write(out/'summary.json',summ)
    with (out/'reviewed_pairs.jsonl').open('w',encoding='utf-8') as f:
        for p in pairs:f.write(json.dumps(p,ensure_ascii=False)+'\n')
    print('V4_DEV_DONE',json.dumps({k:summ[k] for k in ['mode','count','model_success','human_scored','historical_ambiguous_excluded','v3_to_v4','v4_top_rate_in_selected']},ensure_ascii=False),flush=True)

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
        assert '双向对称' in CLASS_RULES
        assert extra_review(HERE/'manual_review4_20261028.json')['train_dev_001']['human_category']=='top'
        assert parse_category('{"category":null}')['schema_valid']
    print('V4_TARGET_FOCUS_SELF_TEST_PASS')

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
        p.add_argument('--extra-review',default=str(HERE/'manual_review4_20261028.json'))
        p.add_argument('--v3-full-tags',default='/workspace/PRD32_V3_TAG40_GPU_20261028_run01/predictions.jsonl')
        p.add_argument('--allowed-root',default=ROOT)
        p.add_argument('--subset',choices=['reviewed19','all40'],default='reviewed19')
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
