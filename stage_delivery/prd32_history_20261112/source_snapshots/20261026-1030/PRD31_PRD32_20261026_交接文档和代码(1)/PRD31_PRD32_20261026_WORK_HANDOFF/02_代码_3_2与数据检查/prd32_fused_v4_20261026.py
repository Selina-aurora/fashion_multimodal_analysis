#!/usr/bin/env python3
"""PRD 3.2 V4 fused-output development diagnostic; no PRD acceptance claims.

Reuse frozen engine-neutral 3.1 image+target bbox without changing 3.1.
In one multimodal generation, request QA color, four tags, and product copy.
Compare ROI crop to full image with highlighted predicted bbox.
Explicit --target-category is HUMAN-PROVIDED for target binding, not category GT.
No sealed/holdout, no accuracy/QPS/PRD-acceptance assertion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

CATEGORIES = ('top','pants','skirt','outerwear','dress','shoe','bag','accessory')
CN = {'top':'上衣','pants':'裤子','skirt':'半身裙','outerwear':'外套',
      'dress':'连衣裙','shoe':'鞋','bag':'包','accessory':'配饰'}
TAG_FIELDS = ('category','primary_color','pattern','visible_design','uncertain_fields')
TOP_FIELDS = ('qa_color','tags','marketing_copy')
UNCERTAIN_FIELDS = {'primary_color','pattern','visible_design'}


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_fused(raw: str, manual_category: str) -> dict:
    """Assess raw model output; normalize fence only for canonical service JSON.

    Refuse to silently repair semantic contradictions or fabricate null fields.
    """
    raw=raw.strip()
    raw_native=raw.startswith('{') and raw.endswith('}') and '```' not in raw
    text=raw
    if re.match(r'^```(?:json)?\s*\n', raw, flags=re.I) and raw.endswith('```'):
        text=re.sub(r'^```(?:json)?\s*\n', '', raw, count=1, flags=re.I)
        text=text[:-3].strip()
    result={
        'model_raw_is_native_json': raw_native,
        'parseable': False, 'structural_valid': False, 'semantic_consistent': False,
        'errors': [], 'parsed': None, 'canonical_service_json': None,
    }
    try:
        parsed=json.loads(text)
    except (ValueError, TypeError) as exc:
        result['errors'].append('json_parse: ' + str(exc))
        return result
    result['parsed']=parsed
    result['parseable']=True
    if not isinstance(parsed,dict) or set(parsed)!=set(TOP_FIELDS):
        result['errors'].append('top_level_keys_mismatch')
        return result
    qa,tags,copy=parsed['qa_color'],parsed['tags'],parsed['marketing_copy']
    if not isinstance(tags,dict) or set(tags)!=set(TAG_FIELDS):
        result['errors'].append('tag_keys_mismatch')
        return result
    if not isinstance(qa,str) or not qa.strip():
        result['errors'].append('qa_not_nonempty_string')
    if not isinstance(copy,str) or not copy.strip() or len(copy)>60:
        result['errors'].append('copy_missing_or_over_60_characters')
    if tags.get('category')!=manual_category:
        result['errors'].append('category_not_equal_manual_target')
    for f in ('primary_color','pattern','visible_design'):
        if tags.get(f) is not None and not isinstance(tags[f],str):
            result['errors'].append('invalid_tag_type:'+f)
    uncertain=tags.get('uncertain_fields')
    if not isinstance(uncertain,list) or any(
        not isinstance(s,str) or s not in UNCERTAIN_FIELDS for s in uncertain
    ) or (isinstance(uncertain,list) and len(uncertain)!=len(set(map(str,uncertain)))):
        result['errors'].append('bad_uncertain_fields')
    result['structural_valid']=not result['errors']
    if result['errors']:
        return result
    uncertain_set=set(uncertain)
    for f in UNCERTAIN_FIELDS:
        if tags[f] is None and f not in uncertain_set:
            result['errors'].append('null_not_marked_uncertain:'+f)
        if tags[f] is not None and f in uncertain_set:
            result['errors'].append('non_null_marked_uncertain:'+f)
    if tags['primary_color'] is not None:
        # Match simple color QA, allowing a trailing '色'. Not a semantic truth oracle.
        def norm(s):return re.sub(r'[\s。,.，！!]', '', s).strip()
        if norm(qa) not in {norm(tags['primary_color']), norm(str(tags['primary_color'])+'色')}:
            result['errors'].append('qa_and_tag_color_disagree')
    elif qa not in ('无法确定','不确定','看不清'):
        result['errors'].append('qa_should_refuse_when_color_null')
    result['semantic_consistent']=not result['errors']
    if result['semantic_consistent']:
        result['canonical_service_json']=json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
    return result


def pct(xs, q):
    xs=sorted(xs); pos=(len(xs)-1)*q; lo=math.floor(pos); hi=math.ceil(pos)
    return xs[lo]+(xs[hi]-xs[lo])*(pos-lo)


def make_prompt(cat: str, image_type: str) -> str:
    location='红框内' if image_type=='marked_full_image' else '图中'
    return (
        f'请只分析{location}的{CN[cat]}（类别{cat}）；目标类别已由人工指定，不需要重新分类。'
        '目标可能与其他衣服叠穿；严禁把内搭、外搭、肩包、鞋子、背景的颜色、图案或袖长算到目标上。'
        '仅凭可见图像作答，不猜测面料、工艺、品牌、风格或功能。'
        '只返回一个原生JSON对象，且无Markdown、解释或代码围栏。'
        '顶层必须且只包含qa_color、tags、marketing_copy三个键。'
        'qa_color应为目标自身主颜色名称，无法确定写“无法确定”；'
        'tags必须且只包含category、primary_color、pattern、visible_design、uncertain_fields五个键；'
        f'category必须是已人工指定的字符串“{cat}”；'
        'primary_color应为目标自身主颜色字符串或JSON null；'
        'pattern应为目标自身图案字符串或JSON null；'
        'visible_design应为目标自身结构设计字符串或JSON null；'
        'uncertain_fields应为字符串数组；'
        'marketing_copy应为60字以内仅描述目标自身的中文客观文案。'
        'primary_color、pattern、visible_design 三个字段：不确定就填null，且字段名加入uncertain_fields；'
        '确定并填了值则绝对不要加入uncertain_fields。category是人工给定的，不要放入uncertain_fields。'
        'qa_color须与tags.primary_color一致；颜色为null时qa_color写“无法确定”。'
        '文案只写目标本身可见的特点，不写搭配的其他物品，不写“百搭”“复古”等主观评价。'
    )


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bridge-input', default='/workspace/prd31_stage_freeze_20261026/bridge_inputs.jsonl')
    p.add_argument('--case-index',type=int,default=0)
    p.add_argument('--target-category',choices=CATEGORIES,required=True,
                   help='Human-specified category for diagnostic only, never counted as predicted classification')
    p.add_argument('--model-id',default='/workspace/models/Qwen2.5-VL-7B-Instruct')
    p.add_argument('--output',default='/workspace/PRD32_FUSED_V4_20261026')
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--roi-padding',type=float,default=0.06)
    p.add_argument('--max-pixels',type=int,default=401408)
    p.add_argument('--max-new-tokens',type=int,default=160)
    args=p.parse_args()
    if not 1<=args.repeats<=5:p.error('repeats must be 1..5')
    if not 0<=args.roi_padding<=0.3:p.error('roi-padding must be within 0..0.3')
    if not 64<=args.max_new_tokens<=512:p.error('max-new-tokens must be 64..512')
    bridge=Path(args.bridge_input).resolve(strict=True)
    if any(s in str(bridge).lower() for s in ('sealed','holdout')):
        raise ValueError('sealed/holdout bridge is excluded')
    rows=[json.loads(s) for s in bridge.read_text(encoding='utf-8').splitlines() if s.strip()]
    if not 0<=args.case_index<len(rows):p.error('case-index out of range')
    row=rows[args.case_index]
    if row.get('schema')!='prd32_engine_neutral_input_v1':
        raise ValueError('Unexpected 3.1 bridge schema')
    source=Path(row['image_path']).resolve(strict=True)
    if any(s in str(source).lower() for s in ('sealed','holdout')):
        raise ValueError('sealed/holdout image excluded')
    if sha256_file(source)!=row.get('image_sha256'):
        raise ValueError('source image SHA256 mismatch')
    box=row.get('target_bbox_xyxy')
    if not isinstance(box,list) or len(box)!=4 or not all(type(x) in (float,int) and math.isfinite(x) for x in box):
        raise ValueError('Invalid bbox')
    from PIL import Image,ImageDraw
    with Image.open(source) as im: img=im.convert('RGB')
    w,h=img.size;x1,y1,x2,y2=box
    if not 0<=x1<x2<=w or not 0<=y1<y2<=h:raise ValueError('bbox outside original image')
    dx=(x2-x1)*args.roi_padding;dy=(y2-y1)*args.roi_padding
    roi=(max(0,math.floor(x1-dx)),max(0,math.floor(y1-dy)),min(w,math.ceil(x2+dx)),min(h,math.ceil(y2+dy)))
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    report_path=out/f'fused_case_{args.case_index:02d}.json'
    crop_path=out/f'roi_case_{args.case_index:02d}.jpg'
    marked_path=out/f'marked_case_{args.case_index:02d}.jpg'
    if any(x.exists() for x in (report_path,crop_path,marked_path)):
        raise FileExistsError('Output already exists; use a fresh directory to retain previous results')
    img.crop(roi).save(crop_path,quality=94)
    marked=img.copy();pen=ImageDraw.Draw(marked)
    pen.rectangle(tuple(map(round,box)),outline=(255,45,35),width=max(3,int(min(w,h)/180)))
    marked.save(marked_path,quality=94)
    
    print('[1/4] Loading local model...',flush=True)
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
    from qwen_vl_utils import process_vision_info
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU unavailable')
    free,_=torch.cuda.mem_get_info()
    if free<18*1024**3:raise RuntimeError(f'GPU free {free/1024**3:.2f}GiB < 18GiB')
    model_path=Path(args.model_id)
    local=model_path.is_dir()
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
        device_map='auto',attn_implementation='sdpa',local_files_only=local,
        low_cpu_mem_usage=True).eval()
    processor=AutoProcessor.from_pretrained(args.model_id,min_pixels=128*28*28,
                   max_pixels=args.max_pixels,local_files_only=local)
    torch.cuda.reset_peak_memory_stats()

    def predict(image_path, variant, warmup=False):
        prompt=('图片里有衣服吗？回答是或否。' if warmup else make_prompt(args.target_category,variant))
        message=[{'role':'user','content':[{'type':'image','image':image_path.as_uri()},
                                           {'type':'text','text':prompt}]}]
        torch.cuda.synchronize();t0=time.perf_counter()
        chat=processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
        images,videos=process_vision_info(message)
        inputs=processor(text=[chat],images=images,videos=videos,
                         padding=True,return_tensors='pt').to(model.device)
        torch.cuda.synchronize();t1=time.perf_counter()
        with torch.inference_mode():
            gen=model.generate(**inputs,max_new_tokens=8 if warmup else args.max_new_tokens,do_sample=False)
        torch.cuda.synchronize();t2=time.perf_counter()
        text=processor.batch_decode(gen[:,inputs['input_ids'].shape[1]:],
                                    skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        if warmup:return None
        return {
            'variant':variant,'request':prompt,'raw_response':text,
            'input_tokens_including_visual':int(inputs['input_ids'].shape[1]),
            'generated_tokens':int(gen.shape[1]-inputs['input_ids'].shape[1]),
            'preprocess_ms':round((t1-t0)*1000,2),
            'generation_ms':round((t2-t1)*1000,2),
            'end_to_end_ms':round((t2-t0)*1000,2),
            **parse_fused(text,args.target_category),
        }
    print('[2/4] GPU warmup...',flush=True)
    predict(crop_path,'roi_target_only',warmup=True)
    print(f'[3/4] Fused output, 2 input variants x {args.repeats} repeats...',flush=True)
    variants=[('roi_target_only',crop_path),('marked_full_image',marked_path)]
    measurements=[]
    for rep in range(args.repeats):
        sequence=variants if rep%2==0 else list(reversed(variants))
        for variant,path in sequence:
            r=predict(path,variant)
            r['repeat_index']=rep+1
            measurements.append(r)
            print(f"  {variant} rep{rep+1} {r['end_to_end_ms']}ms: "
                  f"parse={r['parseable']}, semantics={r['semantic_consistent']}, raw_native={r['model_raw_is_native_json']}",flush=True)
    aggregate=[]
    for variant,_ in variants:
        subset=[r for r in measurements if r['variant']==variant]
        times=[r['end_to_end_ms'] for r in subset]
        aggregate.append({'variant':variant,'n':len(times),
                          'p50_ms':round(statistics.median(times),2),
                          'p95_interpolated_ms':round(pct(times,.95),2),
                          'native_json_count':sum(r['model_raw_is_native_json'] for r in subset),
                          'parseable_count':sum(r['parseable'] for r in subset),
                          'semantic_consistent_count':sum(r['semantic_consistent'] for r in subset),
                          'canonical_service_json_count':sum(bool(r['canonical_service_json']) for r in subset)})
    result={'schema':'prd32_fused_dev_v4','status':'DIAGNOSTIC_COMPLETED_NOT_ACCEPTED',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'bridge_case_id':row.get('case_id'),'manual_target_category':args.target_category,
            'target_category_human_supplied_not_model_accuracy':True,
            'image_sha256_verified':True,'bbox_source':'3.1 frozen predicted bbox, not GT',
            'source_bbox_xyxy':box,'roi_xyxy':roi,'model_id':args.model_id,
            'variants':[name for name,_ in variants],
            'gpu':torch.cuda.get_device_name(0),'gpu_peak_allocated_gib':round(torch.cuda.max_memory_allocated()/1024**3,3),
            'aggregate':aggregate,'measurements':measurements,
            'limitations':['Single dev image, n too small for PRD accuracy/latency/QPS acceptance.',
                           'Manual category provided in prompt: not classification evidence.',
                           'Raw output may contain Markdown fences; canonical service JSON is labeled separately.',
                           'Deterministic structural consistency is NOT verified factual accuracy or copy satisfaction.',
                           'Fused single request changes task architecture and is not directly equivalent to independent latency thresholds.',
                           'Inputs from 3.1 are predictions; sample may be contaminated by layered garments.',
                           'Sealed holdout not accessed; full 3.1+3.2 e2e latency not measured.']}
    report_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'[4/4] {report_path} status={result["status"]}',flush=True)

if __name__=='__main__':main()
