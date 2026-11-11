#!/usr/bin/env python3
"""PRD 3.2 Dev7 V2: ontology-constrained visible-attribute regression.

- Uses the SAME seven development images and historical reference candidate boxes as V1.
- No per-case category/GT/old predictions go into prompts.
- Keeps raw model text, reports native JSON separately from service canonical JSON.
- GT used only *after* inference for development scoring. No holdout/3.1 writes.
- Reference ROI is NOT the frozen 3.1 model ROI or a certified segmentation GT.
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

HERE = Path(__file__).resolve().parent
DEV_IDS = ("dev_02", "dev_04", "dev_05", "dev_06", "dev_08", "dev_09", "dev_12")
VARIANTS = ("marked_full_image", "reference_roi")
KEYS = {"category", "primary_color", "pattern", "visible_design"}
CATEGORIES = {"top", "pants", "skirt", "outerwear", "dress", "shoe", "bag", "accessory"}
BAD_DESIGN = {"无", "无设计", "无明显设计", "无装饰", "无明显装饰", "无图案", "没有", "未知", "无特殊设计", "none", "unknown", "n/a"}
SHADES = {"black":"黑色", "white":"白色", "blue":"蓝色", "red":"红色", "pink":"粉红色",
          "红":"红色", "黑":"黑色", "白":"白色", "蓝":"蓝色"}
PATTERNS = {"格子":"格纹", "格子纹":"格纹", "无":"纯色", "无图案":"纯色", "solid":"纯色",
            "striped":"条纹", "plaid":"格纹", "herringbone":"人字纹", "polka dots":"波点"}

# The same eight-category label ontology for ALL images; never case-specific.
PROMPT = (
    "任务：只分析目标服饰本身；红框或裁剪范围可能还含有内搭、裤子、背包和背景。"
    "整图时只分析红框指示的服饰；裁剪时只分析画面中央最主要的一件服饰。"
    "请独立判断类别；任何外部类别提示都不是可靠标签。"
    "类别规则：top 包含T恤、衬衫、卫衣、针织套头衫、背心；"
    "outerwear 包含作为外层穿着的西装外套、夹克、大衣等；"
    "仅因衣服长袖、宽松、有装饰或颜色深，不可判为outerwear。"
    "dress 为连衣裙，其余类别有 pants、skirt、shoe、bag、accessory。"
    "只返回一个 JSON 对象，严格四个键：category、primary_color、pattern、visible_design。"
    "category 只能为 top,pants,skirt,outerwear,dress,shoe,bag,accessory 或 null。"
    "primary_color 用中文具体主色（例如黑色、白色、玫红色、浅蓝色）；看不清填 null。"
    "pattern 只描述面料表面图案：纯色、条纹、格纹、人字纹、波点、花卉、文字印花或其他可见图案；"
    "蝴蝶结、纽扣、搭扣等立体装饰不得放入pattern；确实看不清填null。"
    "visible_design 只列图中看得到的领口、衣袖、肩带、门襟、扣合方式、层叠结构或装饰，"
    "用中文短词以分号分隔；如圆领;短袖。看不清则用null。"
    "不要写‘无’‘无设计’‘无明显设计’之类占位语，不要根据服装名称猜结构。"
    "不能编造材质成分、品牌、功能或风格。不要Markdown围栏、前后解释或多余字段。"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def restricted(path) -> bool:
    return any(k in str(path).lower() for k in ("holdout", "sealed"))


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check_files(manifest: Path, gt_path: Path, images: bool):
    if any(restricted(p) for p in (manifest, gt_path)):
        raise ValueError("Sealed/holdout path not allowed")
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing Dev7 manifest: {manifest}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing human GT CSV: {gt_path}")
    data, gt_rows = read_csv(manifest), read_csv(gt_path)
    if len(data) != 7 or {r['image_id'] for r in data} != set(DEV_IDS):
        raise ValueError("Manifest must contain exactly the known Dev7 IDs")
    if len(gt_rows) != 7 or {r['image_id'] for r in gt_rows} != set(DEV_IDS):
        raise ValueError("GT CSV must contain exactly the same 7 IDs")
    gts = {r['image_id']:r for r in gt_rows}
    for r in data:
        case = r['image_id']
        if (r.get('status') != 'REFERENCE_CANDIDATE_VISUAL_REVIEW_REQUIRED'
                or r.get('identity_errors','').strip()
                or r.get('geometry_valid','').lower() != 'true'):
            raise ValueError("Identity/geometry gate failed: " + case)
        gt = gts[case]
        if gt.get('human_review_status') != 'VERIFIED' or gt.get('gt_scoring_eligible','').lower() != 'true':
            raise ValueError("GT not verified for: " + case)
        if any(gt.get(field + '_status') != 'CONFIRMED' for field in ('category','color','pattern','design')):
            raise ValueError("GT field not confirmed for: " + case)
        if gt.get('human_category_gt') not in CATEGORIES:
            raise ValueError('Invalid human GT category: ' + case)
        if gt.get('bbox_review_status') != 'TARGET_CORRECT':
            raise ValueError('BBox not visually approved in uploaded GT: ' + case)
        box = json.loads(r['reference_bbox_xyxy'])
        if not isinstance(box,list) or len(box)!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in box):
            raise ValueError("Invalid bbox: " + case)
        if not images:
            continue
        p = Path(r['image_path'])
        if restricted(p) or not p.is_file() or sha256_file(p) != r['image_sha256']:
            raise ValueError("Image missing, restricted, or hash mismatch: " + case)
        from PIL import Image
        with Image.open(p) as im:
            w,h = im.size
        x1,y1,x2,y2 = box
        if not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h:
            raise ValueError("BBox out of bounds: " + case)
    return data, gts


def decode_json(raw: str):
    txt = raw.strip()
    native = txt.startswith('{') and txt.endswith('}')
    peeled = re.sub(r'^```(?:json)?\s*\n', '', txt, count=1, flags=re.I)
    if peeled != txt and peeled.rstrip().endswith('```'):
        peeled = peeled.rstrip()[:-3].strip()
    try:
        parsed = json.loads(peeled)
    except (ValueError, TypeError) as exc:
        return dict(parseable=False,native_json=native,schema_valid=False,raw_parsed=None,
                    canonical_service_json=None,issues=['json_parse_error:'+str(exc)])
    errors = []
    if not isinstance(parsed,dict) or set(parsed) != KEYS:
        errors.append('keys_mismatch')
        return dict(parseable=True,native_json=native,schema_valid=False,raw_parsed=parsed,
                    canonical_service_json=None,issues=errors)
    if parsed['category'] is not None and parsed['category'] not in CATEGORIES:
        errors.append('category_not_in_8_classes')
    for key in ('primary_color','pattern','visible_design'):
        if parsed[key] is not None and not isinstance(parsed[key],str):
            errors.append('invalid_type:'+key)
    structural = not any(x.startswith('invalid_type:') or x in ('keys_mismatch','category_not_in_8_classes') for x in errors)
    canonical = None
    if structural:
        canonical = dict(parsed)
        design = canonical.get('visible_design')
        if isinstance(design,str) and design.strip().lower() in BAD_DESIGN:
            errors.append('non_structural_visible_design')
            canonical['visible_design'] = None # Safe abstention, NOT a correct design prediction.
        elif isinstance(design,str) and not design.strip():
            errors.append('empty_visible_design')
            canonical['visible_design'] = None
        pattern = canonical.get('pattern')
        if isinstance(pattern,str) and any(q in pattern.lower() for q in ('蝴蝶结','纽扣','搭扣','zipper','蕾丝','lace','棉','针织')):
            errors.append('pattern_ontology_review_required')
            canonical['pattern'] = None  # Do not force guessed fabric/ornament into pattern.
        for k in ('primary_color','pattern','visible_design'):
            if isinstance(canonical[k],str) and not canonical[k].strip():
                canonical[k]=None
        # A safe serialization is not an independent model completion.
    return dict(parseable=True,native_json=native,schema_valid=structural,raw_parsed=parsed,
                canonical_service_json=canonical,issues=errors)


def normalize_color(x):
    if x is None: return None
    s=str(x).strip().lower()
    return SHADES.get(s,s) if s else None


def normalize_pattern(x):
    if x is None: return None
    s=str(x).strip().lower()
    return PATTERNS.get(s,s) if s else None


def scores(predictions, gt_map):
    by_variant = {}
    scored_rows=[]
    for v in VARIANTS:
        items = [r for r in predictions if r['input_variant'] == v]
        if not items: continue
        hits=Counter(); issues=Counter()
        for r in items:
            gt=gt_map[r['image_id']]
            p=r.get('canonical_service_json')
            if not isinstance(p,dict): p={}
            cc = p.get('category') == gt['human_category_gt']
            co = normalize_color(p.get('primary_color')) == normalize_color(gt['human_color_gt'])
            pa = normalize_pattern(p.get('pattern')) == normalize_pattern(gt['human_pattern_gt'])
            if cc: hits['category']+=1
            if co: hits['color']+=1
            if pa: hits['pattern']+=1
            if r.get('native_json'):hits['native_json']+=1
            if r.get('parseable'):hits['parseable']+=1
            if 'non_structural_visible_design' in r.get('issues',[]):hits['non_structural_design']+=1
            issues.update(r.get('issues',[]))
            scored_rows.append({
                'image_id':r['image_id'],'input_variant':v,'gt_category':gt['human_category_gt'],
                'pred_category':p.get('category'), 'category_match':cc,
                'gt_color':gt['human_color_gt'],'pred_color':p.get('primary_color'),'color_strict_match':co,
                'gt_pattern':gt['human_pattern_gt'],'pred_pattern':p.get('pattern'),'pattern_strict_match':pa,
                'gt_design':gt['human_design_gt'],'pred_design':p.get('visible_design'),
                'design_accuracy':'NOT_SCORED_MULTILABEL','invalid_design': 'non_structural_visible_design' in r.get('issues',[]),
                'raw_native_json':bool(r.get('native_json')),'parseable':bool(r.get('parseable')),
                'latency_ms':r.get('end_to_end_ms'),'issues':';'.join(r.get('issues',[]))
            })
        n=len(items)
        lat=[float(r['end_to_end_ms']) for r in items if r.get('end_to_end_ms') is not None]
        by_variant[v] = {
            'n_images':n,'category_correct':hits['category'],'category_acc':round(hits['category']/n,4),
            'color_exact_correct':hits['color'],'color_exact_acc':round(hits['color']/n,4),
            'pattern_exact_correct':hits['pattern'],'pattern_exact_acc':round(hits['pattern']/n,4),
            'nonstructural_design_count':hits['non_structural_design'],
            'raw_native_json_count':hits['native_json'],'parseable_count':hits['parseable'],
            'median_latency_ms':round(statistics.median(lat),2) if lat else None,
            'issue_counts':dict(issues)
        }
    return by_variant,scored_rows


def old_baseline(path:Path):
    if not path.is_file():
        return []
    with path.open(encoding='utf-8') as f:
        rows=[json.loads(line) for line in f if line.strip()]
    return [dict(image_id=r['image_id'],input_variant=r['input_variant'],
                 canonical_service_json=r.get('parsed'),issues=r.get('issues',[]),
                 end_to_end_ms=r.get('end_to_end_ms'),native_json=r.get('native_json'),parseable=r.get('parseable')) for r in rows]


def write_csv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',default='/workspace/PRD32_W7_DEV7_SCREEN_V4_20261026/dev7_candidate_manifest.csv')
    ap.add_argument('--gt',default=str(HERE/'PRD32_Dev7_human_GT_7images.csv'))
    ap.add_argument('--baseline',default='/workspace/PRD32_W7_DEV7_BATCH_V1_20261026/predictions.jsonl')
    ap.add_argument('--model-id',default='/workspace/models/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--output',default='/workspace/PRD32_W7_DEV7_QUALITY_V2_20261026')
    ap.add_argument('--max-new-tokens',type=int,default=110)
    ap.add_argument('--max-pixels',type=int,default=401408)
    ap.add_argument('--dry-run',action='store_true',help='Validate everything without loading GPU or writing files')
    ap.add_argument('--offline-selftest',action='store_true',help='Test output parser and baseline scoring without GPU/images')
    args=ap.parse_args()
    if args.max_new_tokens<32 or args.max_pixels<1024:
        ap.error('Invalid generation/image budget')
    manifest,gt_path,baseline_path,model_path,out=map(Path,(args.manifest,args.gt,args.baseline,args.model_id,args.output))
    if any(restricted(x) for x in (manifest,gt_path,baseline_path,model_path,out)):
        ap.error('Sealed/holdout path forbidden')
    if args.offline_selftest:
        demo='```json\n{"category":"top","primary_color":"黑色","pattern":"纯色","visible_design":"无"}\n```'
        val=decode_json(demo)
        assert val['parseable'] and val['canonical_service_json']['visible_design'] is None
        assert 'non_structural_visible_design' in val['issues']
        assert not val['native_json']
        demo2='{"category":"top","primary_color":"玫红色","pattern":"纯色","visible_design":"长袖;圆领"}'
        assert decode_json(demo2)['native_json']
        if gt_path.is_file() and baseline_path.is_file():
            _, gt = check_files(manifest,gt_path,images=False)
            baseline_score,_=scores(old_baseline(baseline_path),gt)
            print(json.dumps({'OFFLINE_SELFTEST_OK':True,'v1_baseline':baseline_score},ensure_ascii=False,indent=2))
        else:print('OFFLINE_SELFTEST_OK (parser only; GT/baseline unavailable)')
        return
    rows,gt=check_files(manifest,gt_path,images=True)
    baseline=old_baseline(baseline_path)
    base_metrics,_=scores(baseline,gt) if baseline else ({},[])
    print('DEV7_INPUT_PREFLIGHT_OK: 7 image SHA256 + human GT verified; reference bbox is still a historical candidate',flush=True)
    if args.dry_run:
        print(json.dumps({'DRY_RUN_OK':True,'gt_images':len(gt),'baseline_available':bool(baseline),
                          'baseline_scores':base_metrics,'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest()},ensure_ascii=False,indent=2))
        return
    if out.exists():raise FileExistsError('Result directory exists, will not overwrite: '+str(out))
    if not model_path.is_dir():raise FileNotFoundError('Local model missing: '+str(model_path))
    from PIL import Image,ImageDraw
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    out.mkdir(parents=True)
    shutil.copy2(manifest,out/'input_manifest_snapshot.csv')
    shutil.copy2(gt_path,out/'human_gt_snapshot.csv')
    (out/'prompt_v2.txt').write_text(PROMPT,encoding='utf-8')
    images=[]
    for row in rows:
        with Image.open(row['image_path']) as fi:im=fi.convert('RGB')
        box=json.loads(row['reference_bbox_xyxy']);x1,y1,x2,y2=box
        marked=im.copy();ImageDraw.Draw(marked).rectangle(tuple(round(v) for v in box),outline=(255,45,35),width=max(3,min(im.size)//150))
        a=out/(row['image_id']+'_marked.jpg');marked.save(a,quality=92)
        pad_x,pad_y=.04*(x2-x1),.04*(y2-y1)
        crop_box=(max(0,math.floor(x1-pad_x)),max(0,math.floor(y1-pad_y)),min(im.width,math.ceil(x2+pad_x)),min(im.height,math.ceil(y2+pad_y)))
        b=out/(row['image_id']+'_reference_roi.jpg');im.crop(crop_box).save(b,quality=92)
        images.append((row,{'marked_full_image':a,'reference_roi':b}))
    print('Loading local Qwen2.5-VL-7B (no model download)...',flush=True)
    tload=time.perf_counter()
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path),local_files_only=True,
          device_map='auto',dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    processor=AutoProcessor.from_pretrained(str(model_path),local_files_only=True,max_pixels=args.max_pixels)
    model_load_s=round(time.perf_counter()-tload,3)
    def infer(image_path,text):
        msg=[{'role':'user','content':[{'type':'image','image':image_path.resolve().as_uri()},
                                          {'type':'text','text':text}]}]
        torch.cuda.synchronize();t0=time.perf_counter()
        formatted=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
        pics,vids=process_vision_info(msg)
        x=processor(text=[formatted],images=pics,videos=vids,padding=True,return_tensors='pt').to(model.device)
        torch.cuda.synchronize();t1=time.perf_counter()
        with torch.inference_mode():
            ids=model.generate(**x,max_new_tokens=args.max_new_tokens,do_sample=False)
        torch.cuda.synchronize();t2=time.perf_counter()
        response=processor.batch_decode(ids[:,x.input_ids.shape[1]:],skip_special_tokens=True,
                   clean_up_tokenization_spaces=False)[0]
        return response,dict(preprocess_ms=round((t1-t0)*1000,2),generation_ms=round((t2-t1)*1000,2),
                             end_to_end_ms=round((t2-t0)*1000,2),generated_tokens=int(ids.shape[1]-x.input_ids.shape[1]))
    print('Warm-up...',flush=True)
    infer(images[0][1]['marked_full_image'],'只用一个中文词回答：这是什么类型的服饰？')
    result=[]
    for row,paths in images:
        for v in VARIANTS:
            raw,timing=infer(paths[v],PROMPT)
            parsed=decode_json(raw)
            result.append(dict(image_id=row['image_id'],image_sha256=row['image_sha256'],input_variant=v,
                               image_sent=str(paths[v]),raw_response=raw,**parsed,**timing,
                               prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
                               model_category_not_guided=True,gt_not_in_prompt=True,
                               reference_bbox_type='historical_human_review_candidate_not_segmentation_gt',
                               is_prd_acceptance=False))
            print(f"{row['image_id']} | {v} | {timing['end_to_end_ms']:.0f}ms | "
                  f"category={parsed.get('canonical_service_json',{}).get('category') if parsed.get('canonical_service_json') else None} | "
                  f"issues={parsed['issues']}",flush=True)
    with (out/'predictions_v2.jsonl').open('w',encoding='utf-8') as f:
        for r in result:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    new_metrics,scored=scores(result,gt)
    write_csv(out/'scored_v2_14rows.csv',scored)
    by={r['image_id']:{ } for r in result}
    for r in result:by[r['image_id']][r['input_variant']]=r.get('canonical_service_json')
    disagreements=[k for k,v in by.items() if v.get(VARIANTS[0]) != v.get(VARIANTS[1])]
    summary={
        'schema':'prd32_dev7_quality_v2','status':'DEV_V2_COMPLETED_NOT_PRD_ACCEPTED',
        'model':str(model_path),'gpu':torch.cuda.get_device_name(0),'dtype':'bfloat16','attention':'sdpa',
        'gt_source':str(gt_path),'gt_usage':'SCORING_ONLY_NOT_PROMPT','model_load_seconds':model_load_s,
        'images':7,'inference_calls':len(result),'v1_baseline':base_metrics,'v2_metrics':new_metrics,
        'cross_view_disagreement_ids':disagreements,
        'raw_native_json_count':sum(r['native_json'] for r in result),
        'canonical_service_json_count':sum(r['canonical_service_json'] is not None for r in result),
        'design_quality_not_scored':True,
        'accuracy_scope':'same 7 non-blind development images; top/outerwear only',
        'reference_bbox_scope':'historical candidate, not frozen PRD 3.1 predicted ROI',
        'sealed_holdout_used':False,'frozen_prd31_modified':False,
        'qa_and_copy_evaluated':False,'full_prd_acceptance':False,
        'limitations':['n=7 correlated dev images; no independent holdout','no formal design multi-label ontology scoring',
                       'single-run latency not formal service P95/QPS','canonical JSON serialization does not improve native model compliance']}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('V2_COMPLETED:',out,flush=True)

if __name__=='__main__':main()
