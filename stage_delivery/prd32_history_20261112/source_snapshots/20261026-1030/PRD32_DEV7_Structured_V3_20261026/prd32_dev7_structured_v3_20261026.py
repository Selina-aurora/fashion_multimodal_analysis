#!/usr/bin/env python3
"""PRD 3.2 Dev7 V3 -- structured garment design, QA-safe development regression.

Never uses human GT in model prompts. Reads *only* the previously screened Dev7.
The historical reference ROI is a human-review candidate, not frozen 3.1 output.
JSON returned by the model is independently distinguished from service-canonical JSON.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

DEV_IDS = ('dev_02','dev_04','dev_05','dev_06','dev_08','dev_09','dev_12')
VARIANTS = ('marked_full_image','reference_roi')
CATEGORIES = ('top','pants','skirt','outerwear','dress','shoe','bag','accessory')
STRUCTURE = ('neckline','sleeve','strap','front','details')
VOCAB = {
    'neckline': ('圆领','圆弧领','花瓣领','V领','翻领','西装翻领'),
    'sleeve': ('无袖','短袖','长袖'),
    'strap': ('宽肩带','细肩带'),
    'front': ('纽扣门襟','搭扣','双排扣','敞开前襟'),
    'details': ('蝴蝶结','翻折袖口','横向分层'),
}
PATTERN_ALIASES = {'无':'纯色','无图案':'纯色','solid':'纯色','条形纹':'条纹','格子':'格纹',
                   '格子纹':'格纹','striped':'条纹','plaid':'格纹','herringbone':'人字纹',
                   '波点印花':'波点'}
COLOR_ALIASES = {'black':'黑色','white':'白色','blue':'蓝色','pink':'粉红色',
                 '红':'红色','黑':'黑色','白':'白色','蓝':'蓝色'}
OUTPUT_KEYS = {'category','primary_color','pattern','design'}

PROMPT = (
    '你是服饰实例级属性标注器。只分析红框指向或裁剪图中心的主要目标服饰；'
    '不要把内搭、外搭、包、裤子、背景或头发的属性算在目标上。'
    '请独立判断服饰类别，不存在任何人工类别提示。'
    'top包括T恤、衬衫、卫衣、套头针织衫、背心；'
    'outerwear仅为西装外套、夹克、大衣等外层服装；不可因为长袖或深色就判outerwear。'
    'category限定top,pants,skirt,outerwear,dress,shoe,bag,accessory，不能确定时用null。'
    'primary_color只写中文主颜色（例如黑色、白色、玫红色、浅蓝色）；看不清填null。'
    'pattern只描述衣料表面的可见图案，例如纯色、条纹、格纹、人字纹、花卉、文字印花；'
    '扣子、蝴蝶结、搭扣等立体结构和装饰不是面料图案，无法判断写null。'
    'design必须是含neckline,sleeve,strap,front,details五键的对象：'
    'neckline只可取圆领、圆弧领、花瓣领、V领、翻领、西装翻领或null；'
    'sleeve只可取无袖、短袖、长袖或null；'
    'strap只可取宽肩带、细肩带或null；'
    'front只可取纽扣门襟、搭扣、双排扣、敞开前襟或null；'
    'details必须是数组，仅可包含蝴蝶结、翻折袖口、横向分层，无可证实细节用空数组。'
    '仅标视觉能够证实的特征；被头发遮挡、画面外、无法识别时必须写null，禁止靠衣服名称猜。'
    '例如双排扣必须看见明确两列扣子；圆领不得代替翻领。'
    '禁止推测面料成分、工艺、品牌、用途和风格。'
    '只输出一个合法JSON对象，键严格为category,primary_color,pattern,design；'
    '不需要markdown代码围栏、解释或额外字段。'
)


def blocked(path):
    return any(z in str(path).lower() for z in ('sealed','holdout'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for p in iter(lambda:f.read(1<<20),b''):h.update(p)
    return h.hexdigest()


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def validate_sources(manifest,gt_path,validate_images=True):
    if blocked(manifest) or blocked(gt_path):raise ValueError('Do not access sealed holdout')
    if not manifest.is_file() or not gt_path.is_file():raise FileNotFoundError('Manifest or GT absent')
    rows, gt=csv_rows(manifest),csv_rows(gt_path)
    if len(rows)!=7 or len(gt)!=7 or {r['image_id'] for r in rows}!={r['image_id'] for r in gt} or {r['image_id'] for r in rows}!=set(DEV_IDS):
        raise ValueError('Dev7 identity/row count mismatch')
    gm={r['image_id']:r for r in gt}
    for r in rows:
        k=r['image_id']; g=gm[k]
        if r.get('status')!='REFERENCE_CANDIDATE_VISUAL_REVIEW_REQUIRED' or r.get('geometry_valid','').lower()!='true' or r.get('identity_errors'):
            raise ValueError('Bad source identity or reference geometry: '+k)
        if g.get('human_review_status')!='VERIFIED' or g.get('gt_scoring_eligible','').lower()!='true' or g.get('bbox_review_status')!='TARGET_CORRECT':
            raise ValueError('Human review not approved: '+k)
        if any(g.get(f'{field}_status')!='CONFIRMED' for field in ('category','color','pattern','design')):
            raise ValueError('Human GT not complete: '+k)
        if g.get('human_category_gt') not in CATEGORIES:raise ValueError('Bad human class: '+k)
        box=json.loads(r['reference_bbox_xyxy'])
        if not isinstance(box,list) or len(box)!=4 or not all(isinstance(z,(int,float)) and math.isfinite(z) for z in box):
            raise ValueError('Bad bbox: '+k)
        if validate_images:
            path=Path(r['image_path'])
            if blocked(path) or not path.is_file() or digest(path)!=r['image_sha256']:
                raise ValueError('Image missing, sealed or SHA256 mismatch: '+k)
            from PIL import Image
            with Image.open(path) as im: width,height=im.size
            x1,y1,x2,y2=box
            if not (0<=x1<x2<=width and 0<=y1<y2<=height):raise ValueError('bbox beyond image: '+k)
    return rows,gm


def parse_raw(raw):
    text=raw.strip()
    raw_native=text.startswith('{') and text.endswith('}')
    peeled=text
    if text.startswith('```'):
        peeled=re.sub(r'^```(?:json)?\s*\n','',text,count=1,flags=re.I)
        if peeled.rstrip().endswith('```'):peeled=peeled.rstrip()[:-3].strip()
    try: item=json.loads(peeled)
    except Exception as ex:
        return dict(raw_native_json=False,parseable=False,schema_valid=False,parsed=None,
                    canonical_service_json=None,errors=['json_parse:'+str(ex)])
    errors=[]
    if not isinstance(item,dict) or set(item)!=OUTPUT_KEYS:
        errors.append('root_keys_mismatch')
    else:
        if item.get('category') is not None and item.get('category') not in CATEGORIES:
            errors.append('unknown_category')
        for x in ('primary_color','pattern'):
            if item[x] is not None and not isinstance(item[x],str):errors.append('type_'+x)
        ds=item['design']
        if not isinstance(ds,dict) or set(ds)!=set(STRUCTURE):errors.append('design_schema_mismatch')
        else:
            for k in STRUCTURE[:-1]:
                if ds[k] is not None and ds[k] not in VOCAB[k]:errors.append('out_of_vocab_'+k)
            d=ds['details']
            if not isinstance(d,list) or len(d)!=len(set(map(str,d))) or not all(isinstance(x,str) and x in VOCAB['details'] for x in d):
                errors.append('invalid_details')
    valid=not errors
    # Canonical service JSON is a serialization of validated model content, not evidence of native JSON.
    clean=item if valid else None
    return dict(raw_native_json=bool(raw_native and valid),parseable=True,schema_valid=valid,
                parsed=item,canonical_service_json=clean,errors=errors)


def normalized_color(text):
    if text is None:return None
    x=str(text).strip().lower()
    return COLOR_ALIASES.get(x,x)


def normalized_pattern(text):
    if text is None:return None
    x=str(text).strip().lower()
    return PATTERN_ALIASES.get(x,x)


def labeled_design_from_human(g):
    """Extract only explicitly named GT features. Omitted GT fields are NOT negatives."""
    s=g.get('human_design_gt','')
    # Generic mappings apply to every image and never enter prompts.
    norms={'neckline':None,'sleeve':None,'strap':None,'front':None,'details':[]}
    for name, rule in [('西装翻领','西装翻领'),('花瓣型领口','花瓣领'),('花瓣形领口','花瓣领'),('圆弧花瓣型领口','花瓣领'),
                       ('圆弧形领口','圆弧领'),('圆弧领口','圆弧领'),('圆领','圆领'),('V领','V领'),('翻领','翻领')]:
        if name in s:
            norms['neckline']=rule;break
    for x in ('无袖','短袖','长袖'):
        if x in s:norms['sleeve']=x;break
    for x in ('宽肩带','细肩带'):
        if x in s:norms['strap']=x;break
    if '敞开前襟' in s:norms['front']='敞开前襟'
    elif '双排扣' in s:norms['front']='双排扣'
    elif '搭扣' in s:norms['front']='搭扣'
    elif '纽扣门襟' in s or '前襟纽扣' in s or '前中部纽扣门襟' in s:norms['front']='纽扣门襟'
    for needle,typ in [('蝴蝶结','蝴蝶结'),('翻折袖口','翻折袖口'),('袖口翻折','翻折袖口'),('横向分层','横向分层')]:
        if needle in s and typ not in norms['details']:norms['details'].append(typ)
    return norms


def summarize(results,gt):
    scores={};scored=[]
    for variant in VARIANTS:
        subset=[r for r in results if r.get('input_variant')==variant]
        if not subset:continue
        hit=Counter();lat=[];detail_tp=0;detail_n=0
        for r in subset:
            g=gt[r['image_id']];p=r.get('canonical_service_json') or {}
            cc=p.get('category')==g['human_category_gt']
            co=normalized_color(p.get('primary_color'))==normalized_color(g['human_color_gt'])
            pa=normalized_pattern(p.get('pattern'))==normalized_pattern(g['human_pattern_gt'])
            hit.update({'category':int(cc),'color':int(co),'pattern':int(pa),'native':int(r.get('raw_native_json',False)),
                        'valid':int(r.get('schema_valid',False))})
            if r.get('end_to_end_ms') is not None:lat.append(float(r['end_to_end_ms']))
            ds=p.get('design') if isinstance(p.get('design'),dict) else {}
            target=labeled_design_from_human(g)
            outcomes={}
            for field in STRUCTURE[:-1]:
                if target[field] is not None:
                    ok=ds.get(field)==target[field]
                    outcomes[field]=ok
                    detail_n+=1;detail_tp+=int(ok)
                else:outcomes[field]='GT_UNANNOTATED'
            actual=set(ds.get('details',[])) if isinstance(ds.get('details'),list) else set()
            if target['details']:
                outcomes['details_expected']=target['details']
                outcomes['details_recalled']=sorted(actual & set(target['details']))
                detail_n+=len(target['details']);detail_tp+=len(actual & set(target['details']))
            scored.append(dict(image_id=r['image_id'],input_variant=variant,gt_category=g['human_category_gt'],
                pred_category=p.get('category'),category_match=cc,gt_color=g['human_color_gt'],pred_color=p.get('primary_color'),
                color_match=co,gt_pattern=g['human_pattern_gt'],pred_pattern=p.get('pattern'),pattern_match=pa,
                gt_design_text=g['human_design_gt'],model_design=json.dumps(ds,ensure_ascii=False),
                gt_design_known=json.dumps(target,ensure_ascii=False),known_design_match=json.dumps(outcomes,ensure_ascii=False),
                schema_valid=r.get('schema_valid'),native_json=r.get('raw_native_json'),end_to_end_ms=r.get('end_to_end_ms'),
                errors=';'.join(r.get('errors',[]))))
        n=len(subset)
        scores[variant]=dict(n=n,category_correct=hit['category'],category_acc=round(hit['category']/n,4),
            color_correct=hit['color'],color_acc=round(hit['color']/n,4),pattern_correct=hit['pattern'],pattern_acc=round(hit['pattern']/n,4),
            raw_native_json_count=hit['native'],schema_valid_count=hit['valid'],
            gt_named_design_recall=round(detail_tp/detail_n,4) if detail_n else None,
            gt_named_design_hits=detail_tp,gt_named_design_fields=detail_n,
            median_latency_ms=round(statistics.median(lat),2) if lat else None)
    return scores,scored


def csv_write(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',default='/workspace/PRD32_W7_DEV7_SCREEN_V4_20261026/dev7_candidate_manifest.csv')
    ap.add_argument('--gt',default='/workspace/PRD32_DEV7_V2/PRD32_Dev7_human_GT_7images.csv')
    ap.add_argument('--v2',default='/workspace/PRD32_W7_DEV7_QUALITY_V2_20261026/predictions_v2.jsonl')
    ap.add_argument('--model-id',default='/workspace/models/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--output',default='/workspace/PRD32_W7_DEV7_STRUCTURED_V3_20261026')
    ap.add_argument('--max-new-tokens',type=int,default=190)
    ap.add_argument('--max-pixels',type=int,default=401408)
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--offline-selftest',action='store_true')
    args=ap.parse_args()
    if args.max_new_tokens<80 or args.max_pixels<1024:ap.error('Invalid token/pixel budget')
    manifest,gtpath,v2path,modelpath,out=map(Path,(args.manifest,args.gt,args.v2,args.model_id,args.output))
    if any(blocked(p) for p in (manifest,gtpath,v2path,modelpath,out)):ap.error('Sealed/holdout path forbidden')
    if args.offline_selftest:
        demo='```json\n{"category":"top","primary_color":"黑色","pattern":"纯色","design":{"neckline":"圆领","sleeve":"短袖","strap":null,"front":null,"details":[]}}\n```'
        parsed=parse_raw(demo)
        assert parsed['schema_valid'] and not parsed['raw_native_json']
        assert parse_raw(demo.replace('```json\n','').replace('\n```',''))['raw_native_json']
        wrong=demo.replace('"短袖"','"随便填的袖子"')
        assert not parse_raw(wrong)['schema_valid']
        print('V3_OFFLINE_SELFTEST_OK')
        return
    manifest_rows, gt=validate_sources(manifest,gtpath,validate_images=True)
    if not v2path.is_file():raise FileNotFoundError('V2 regression baseline absent: '+str(v2path))
    v2=read_jsonl(v2path)
    if len(v2)!=14 or {(r['image_id'],r['input_variant']) for r in v2}!={(k,v) for k in DEV_IDS for v in VARIANTS}:
        raise ValueError('V2 baseline pair check failed')
    prompt_sha=hashlib.sha256(PROMPT.encode('utf-8')).hexdigest()
    print('V3_INPUT_PREFLIGHT_OK: 7 verified image SHA256, human GT separated, V2 pairs 14',flush=True)
    if args.dry_run:
        print(json.dumps({'DRY_RUN_OK':True,'n_images':7,'planned_inference_calls':14,
                          'model_path':str(modelpath),'v2_baseline':str(v2path),'prompt_sha256':prompt_sha,
                          'reference_bbox_is_candidate':True,'sealed_holdout_accessed':False},ensure_ascii=False,indent=2))
        return
    if out.exists():raise FileExistsError('Refuse to overwrite prior results: '+str(out))
    if not modelpath.is_dir():raise FileNotFoundError('Local Qwen path missing: '+str(modelpath))
    from PIL import Image,ImageDraw
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    if not torch.cuda.is_available():raise RuntimeError('No CUDA device')
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    out.mkdir(parents=True)
    shutil.copy2(manifest,out/'input_manifest_snapshot.csv')
    shutil.copy2(gtpath,out/'human_gt_for_scoring_only_snapshot.csv')
    shutil.copy2(v2path,out/'v2_baseline_snapshot.jsonl')
    (out/'prompt_v3.txt').write_text(PROMPT+'\n',encoding='utf-8')
    im_paths=[]
    for row in manifest_rows:
        case=row['image_id'];box=json.loads(row['reference_bbox_xyxy']);x1,y1,x2,y2=box
        with Image.open(row['image_path']) as f:im=f.convert('RGB')
        marked=im.copy();ImageDraw.Draw(marked).rectangle(tuple(round(z) for z in box),outline=(255,50,35),width=max(3,min(im.size)//150))
        a=out/(case+'_marked.jpg');marked.save(a,quality=92)
        px,py=.04*(x2-x1),.04*(y2-y1)
        crop=(max(0,math.floor(x1-px)),max(0,math.floor(y1-py)),min(im.width,math.ceil(x2+px)),min(im.height,math.ceil(y2+py)))
        b=out/(case+'_reference_roi.jpg');im.crop(crop).save(b,quality=92)
        im_paths.append((row,{'marked_full_image':a,'reference_roi':b}))
    print('Loading Qwen from local weights ...',flush=True)
    load_start=time.perf_counter()
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(modelpath),local_files_only=True,
           device_map='auto',dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    processor=AutoProcessor.from_pretrained(str(modelpath),local_files_only=True,max_pixels=args.max_pixels)
    load_seconds=round(time.perf_counter()-load_start,3)
    def infer(p,text):
        msg=[{'role':'user','content':[{'type':'image','image':p.resolve().as_uri()},{'type':'text','text':text}]}]
        torch.cuda.synchronize();t0=time.perf_counter()
        formatted=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
        imgs,vids=process_vision_info(msg)
        x=processor(text=[formatted],images=imgs,videos=vids,padding=True,return_tensors='pt').to(model.device)
        torch.cuda.synchronize();t1=time.perf_counter()
        with torch.inference_mode():
            seq=model.generate(**x,max_new_tokens=args.max_new_tokens,do_sample=False)
        torch.cuda.synchronize();t2=time.perf_counter()
        output=processor.batch_decode(seq[:,x.input_ids.shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        return output,dict(preprocess_ms=round(1000*(t1-t0),2),generation_ms=round(1000*(t2-t1),2),
                           end_to_end_ms=round(1000*(t2-t0),2),generated_tokens=int(seq.shape[1]-x.input_ids.shape[1]))
    print('GPU warmup',flush=True)
    infer(im_paths[0][1]['marked_full_image'],'只用一个中文词回答：图片里有什么服饰？')
    records=[]
    # Predictions are persisted after every request to survive terminal disconnects/other failures.
    dest=out/'predictions_v3.jsonl'
    with dest.open('w',encoding='utf-8') as stream:
        for row,path_map in im_paths:
            for variant in VARIANTS:
                raw,timing=infer(path_map[variant],PROMPT)
                data=parse_raw(raw)
                rec=dict(image_id=row['image_id'],image_sha256=row['image_sha256'],input_variant=variant,
                         image_sent=str(path_map[variant]),raw_response=raw,**data,**timing,
                         gt_was_not_in_prompt=True,model_category_not_guided=True,
                         reference_roi_type='historical_review_candidate_not_frozen_3_1',
                         prompt_sha256=prompt_sha,not_prd_acceptance=True)
                records.append(rec)
                stream.write(json.dumps(rec,ensure_ascii=False)+'\n');stream.flush()
                pd=data.get('canonical_service_json') or {}
                print(f"{row['image_id']} | {variant} | {timing['end_to_end_ms']:.0f}ms | "
                      f"category={pd.get('category')} | errors={data['errors']}",flush=True)
    v3_scores,scored=summarize(records,gt)
    csv_write(out/'scored_v3_14rows.csv',scored)
    # Read V2 summary as-is; the V2 design free-text scores are not directly comparable to the V3 structured schema.
    old_summary=v2path.parent/'summary.json'
    v2_metrics=None
    if old_summary.is_file():
        with old_summary.open(encoding='utf-8') as f:v2_metrics=json.load(f).get('v2_metrics')
    v2_classes={r['image_id']:{ } for r in v2}
    for r in v2:v2_classes[r['image_id']][r['input_variant']]=(r.get('canonical_service_json') or {}).get('category')
    regressions=[];recoveries=[]
    for r in records:
        vid=r['input_variant'];case=r['image_id'];true=gt[case]['human_category_gt']
        was=v2_classes[case].get(vid)==true
        now=(r.get('canonical_service_json') or {}).get('category')==true
        if was and not now:regressions.append(case+'|'+vid)
        if not was and now:recoveries.append(case+'|'+vid)
    report=dict(schema='prd32_dev7_structured_v3',status='DEV_V3_COMPLETED_NOT_PRD_ACCEPTED',
        n_images=7,n_inference_calls=14,model_id=str(modelpath),model_load_seconds=load_seconds,
        v2_summary_metrics=v2_metrics,v3_metrics=v3_scores,
        category_regressions_from_v2=regressions,category_recoveries_from_v2=recoveries,
        raw_native_json_count=sum(bool(r['raw_native_json']) for r in records),
        strict_schema_valid_count=sum(bool(r['schema_valid']) for r in records),
        service_json_count=sum(r['canonical_service_json'] is not None for r in records),
        generated_text_type='V3 structured visual attributes only',
        caveats=['7 non-blind development images, two classes only',
                 'Historical reference boxes are not 3.1 frozen predicted ROI or certified segmentation GT',
                 'GT was scoring-only; structured design recall counts only GT-named fields and does not imply full precision/F1',
                 'No QA, merchandising copy satisfaction, eight-class generalization, formal service P95 or QPS tested',
                 'Model native strict JSON and service-canonical JSON are separate metrics'],
        sealed_holdout_used=False,frozen_prd31_modified=False,full_prd_acceptance=False,prompt_sha256=prompt_sha)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('V3_COMPLETED:',out,flush=True)

if __name__=='__main__':main()
