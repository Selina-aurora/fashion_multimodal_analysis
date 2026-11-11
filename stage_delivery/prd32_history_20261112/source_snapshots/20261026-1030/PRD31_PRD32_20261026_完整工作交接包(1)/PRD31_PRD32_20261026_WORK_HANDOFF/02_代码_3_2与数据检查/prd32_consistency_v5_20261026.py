#!/usr/bin/env python3
"""PRD 3.2 V5: source-of-truth/target-binding development experiment.

Runs two prompting modes on ROI vs marked full image for a SINGLE selected
unsealed bridge request, using locally installed Qwen2.5-VL. It doesn't edit
3.1 weights or input files. Explicit target category is HUMAN-GIVEN for this
controlled diagnosis; cannot count toward category classification accuracy.

Mode A: three visual properties only. The service derives the simple COLOR QA
and a cautious factual description from those same properties.
Mode B: the same properties plus the VLM's candidate merchant copy. The VLM
copy is kept as UNVERIFIED (not substituted into the service description).

The generated QA is scoped to the COLOR question, not general PRD 3.2 VQA.
No GT accuracy, satisfaction, concurrency/QPS, or end-to-end PRD acceptance.
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

CATEGORIES = ('top', 'pants', 'skirt', 'outerwear', 'dress', 'shoe', 'bag', 'accessory')
CN = dict(zip(CATEGORIES, ('上衣','裤子','半身裙','外套','连衣裙','鞋','包','配饰')))
ATTR_KEYS = ('primary_color', 'pattern', 'visible_design')
KNOWN_NON_DESIGNS = ('无明显设计', '无设计', '没有设计', '无图案', '无装饰', '纯色', 'solid', 'plain')


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_attributes(raw: str, mode: str) -> dict:
    """Remove optional Markdown wrapping for parsing, but REPORT that change.

    No guessing, filling, or reassigning model-provided values when invalid.
    """
    raw = raw.strip()
    is_native = raw.startswith('{') and raw.endswith('}') and '```' not in raw
    text = raw
    if re.match(r'^```(?:json)?\s*\n', raw, flags=re.I) and raw.endswith('```'):
        text = re.sub(r'^```(?:json)?\s*\n', '', raw, count=1, flags=re.I)[:-3].strip()
    result = {'raw_native_json': is_native, 'parseable': False,
              'attributes_valid': False, 'errors': [], 'parsed': None}
    try:
        obj = json.loads(text)
    except (ValueError, TypeError) as exc:
        result['errors'].append('json_parse:' + str(exc))
        return result
    result['parseable'] = True
    result['parsed'] = obj
    allowed = set(ATTR_KEYS) | ({'marketing_copy'} if mode == 'tags_and_copy' else set())
    if not isinstance(obj, dict) or set(obj) != allowed:
        result['errors'].append('unexpected_keys')
        return result
    for key in ATTR_KEYS:
        value = obj[key]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            result['errors'].append('invalid_type_or_empty:' + key)
    if mode == 'tags_and_copy':
        copy = obj['marketing_copy']
        if not isinstance(copy, str) or not copy.strip() or len(copy) > 60:
            result['errors'].append('invalid_model_copy')
    design = obj['visible_design']
    if isinstance(design, str) and design.strip().casefold() in KNOWN_NON_DESIGNS:
        result['errors'].append('non_structural_visible_design')
    result['attributes_valid'] = not result['errors']
    return result


def build_service_output(obj: dict, target_category: str, max_copy_chars: int = 60) -> dict:
    """Derive logically consistent COLOR QA and conservative copy from tags.

    This is deterministic service post-processing, NOT separate generative VQA
    or human-reviewed merchandising copy. No human factual verification.
    """
    tags = {'category': target_category, **{k: obj[k] for k in ATTR_KEYS}}
    tags['uncertain_fields'] = [k for k in ATTR_KEYS if tags[k] is None]
    color = tags['primary_color']
    qa = color if color is not None else '无法确定'
    title = CN[target_category]
    if color:
        title = color.strip() + title
    clauses = [title]
    if tags['pattern']:
        clauses.append(tags['pattern'].strip() + '图案' if tags['pattern'].strip() not in {'纯色', '素色'} else '纯色设计')
    if tags['visible_design']:
        clauses.append(tags['visible_design'].strip())
    # Conservative, factual template: no novel composition, brand, feel, or function.
    candidate = '，'.join(clauses) + '。'
    if len(candidate) > max_copy_chars:
        candidate = candidate[:max_copy_chars-1].rstrip('，、 ') + '。'
    output = {'qa_color': qa, 'tags': tags, 'marketing_copy': candidate}
    return output


def prompt_for(category: str, marked: bool, mode: str) -> str:
    loc = '红框所指的目标' if marked else '这张裁剪图中的目标'
    base = (
        f'你正在识别{loc}，其类别在本次诊断中由人工指定为{category}（{CN[category]}）。'
        '画面可能包含叠穿衣物或包等其他物品，只分析这个类别的目标本身。'
        '不要把其他衣物的颜色、图案或袖子算到目标上。'
        '不能确定的可见属性填JSON null；不要猜面料成分、加工工艺、品牌、功能和风格。'
        'primary_color为目标的主颜色；pattern为目标面料表面可见的图案（确实无图案可填纯色，否则null）；'
        'visible_design只填可见的服装结构特征（如领口结构、衣袖结构、肩带、口袋等），'
        '看不清就填null，不得用“无图案”“无装饰”“无明显设计”填这个字段。'
        'JSON只包含要求的键，字段的值只能是中文短字符串或null。'
        '只返回JSON对象，不要代码围栏、说明或其他内容。'
    )
    if mode == 'tags_only':
        return base + '键恰好为primary_color、pattern、visible_design。'
    return (base + '键恰好为primary_color、pattern、visible_design、marketing_copy。'
            'marketing_copy为不超过60个汉字的客观商品描述，只提到已确认的目标本身可见特点。'
            '不得添加对其他服饰或配饰的描述。')


def percentile(values, q):
    vals = sorted(values)
    pos = (len(vals)-1)*q
    lo, hi = math.floor(pos), math.ceil(pos)
    return vals[lo] + (vals[hi]-vals[lo]) * (pos-lo)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bridge-input', default='/workspace/prd31_stage_freeze_20261026/bridge_inputs.jsonl')
    ap.add_argument('--case-index', type=int, default=0)
    ap.add_argument('--model-id', default='/workspace/models/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--target-category', choices=CATEGORIES, required=True,
                    help='Manually supplied target category for development diagnosis; NOT classification GT')
    ap.add_argument('--output', default='/workspace/PRD32_CONSISTENCY_V5_20261026')
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--roi-padding', type=float, default=0.06)
    ap.add_argument('--max-pixels', type=int, default=401408)
    ap.add_argument('--max-new-tokens', type=int, default=128)
    args = ap.parse_args()
    if not 1 <= args.repeats <= 5: ap.error('--repeats must be between 1 and 5')
    if not 0 <= args.roi_padding <= 0.3: ap.error('--roi-padding must be within 0..0.3')
    if not 64 <= args.max_new_tokens <= 256: ap.error('--max-new-tokens must be within 64..256')
    bridge = Path(args.bridge_input).resolve(strict=True)
    if any(x in str(bridge).lower() for x in ('sealed', 'holdout')):
        raise ValueError('Sealed/holdout inputs are explicitly excluded')
    rows = [json.loads(line) for line in bridge.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not 0 <= args.case_index < len(rows): ap.error('--case-index outside bridge input')
    row = rows[args.case_index]
    if row.get('schema') != 'prd32_engine_neutral_input_v1':
        raise ValueError('Unexpected 3.1 bridge schema')
    source = Path(row['image_path']).resolve(strict=True)
    if any(x in str(source).lower() for x in ('sealed', 'holdout')):
        raise ValueError('Sealed/holdout image excluded')
    if file_sha256(source) != row.get('image_sha256'):
        raise ValueError('Source image identity mismatch')
    box = row.get('target_bbox_xyxy')
    if not isinstance(box, list) or len(box) != 4 or not all(type(v) in (int, float) and math.isfinite(v) for v in box):
        raise ValueError('Invalid 3.1 predicted bounding box')
    from PIL import Image, ImageDraw
    with Image.open(source) as im: img = im.convert('RGB')
    w, h = img.size; x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError('Bounding box outside original image')
    dx, dy = (x2-x1)*args.roi_padding, (y2-y1)*args.roi_padding
    crop = (max(0,math.floor(x1-dx)), max(0,math.floor(y1-dy)),
            min(w,math.ceil(x2+dx)), min(h,math.ceil(y2+dy)))
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_path = out/f'consistency_case_{args.case_index:02}.json'
    roi_path = out/f'roi_case_{args.case_index:02}.jpg'
    marked_path = out/f'marked_case_{args.case_index:02}.jpg'
    if any(p.exists() for p in (result_path, roi_path, marked_path)):
        raise FileExistsError('Output file already exists; use new directory to preserve previous run')
    img.crop(crop).save(roi_path, quality=94)
    marked = img.copy()
    ImageDraw.Draw(marked).rectangle(tuple(map(round,box)), outline=(255,45,35), width=max(3,min(w,h)//180))
    marked.save(marked_path, quality=94)

    print('[1/4] Loading local model...', flush=True)
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU not available')
    free, _ = torch.cuda.mem_get_info()
    if free < 18 * 1024**3:
        raise RuntimeError(f'GPU has only {free/1024**3:.1f} GiB free; need >=18 GiB before loading')
    local = Path(args.model_id).is_dir()
    if not local:
        raise ValueError('V5 requires an existing local model directory; no network download')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
        device_map='auto', attn_implementation='sdpa', low_cpu_mem_usage=True,
        local_files_only=True).eval()
    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=128*28*28,
                                               max_pixels=args.max_pixels, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()

    def infer(path: Path, variant: str, mode: str, warmup=False):
        q = '图片里有服饰吗？只回答是或否。' if warmup else prompt_for(args.target_category, variant=='marked_full_image', mode)
        messages=[{'role':'user','content':[{'type':'image','image':path.as_uri()},
                                            {'type':'text','text':q}]}]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        chat=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, vids=process_vision_info(messages)
        inputs=processor(text=[chat], images=imgs, videos=vids, padding=True, return_tensors='pt').to(model.device)
        torch.cuda.synchronize(); t1=time.perf_counter()
        with torch.inference_mode():
            gen=model.generate(**inputs, max_new_tokens=8 if warmup else args.max_new_tokens, do_sample=False)
        torch.cuda.synchronize(); t2=time.perf_counter()
        text=processor.batch_decode(gen[:,inputs['input_ids'].shape[1]:],
                                     skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        if warmup:return None
        checked=parse_attributes(text, mode)
        service=build_service_output(checked['parsed'],args.target_category) if checked['attributes_valid'] else None
        candidate=checked['parsed'].get('marketing_copy') if (checked['attributes_valid'] and mode == 'tags_and_copy') else None
        return {'variant':variant,'mode':mode,'prompt':q,'raw_response':text,
                'prompt_tokens_including_visual':int(inputs['input_ids'].shape[1]),
                'generated_tokens':int(gen.shape[1]-inputs['input_ids'].shape[1]),
                'preprocess_ms':round((t1-t0)*1000,2),
                'generation_ms':round((t2-t1)*1000,2),
                'end_to_end_ms':round((t2-t0)*1000,2),
                **checked,'service_result':service,
                'service_result_is_postprocessed_not_separate_model_qa':service is not None,
                'model_copy_candidate_unverified':candidate}

    print('[2/4] Prewarming CUDA and image pipeline...', flush=True)
    infer(roi_path,'roi_target_only','tags_only',warmup=True)
    pairs=[('roi_target_only', roi_path),('marked_full_image', marked_path)]
    modes=['tags_only','tags_and_copy']
    print(f'[3/4] 2 image variants x 2 prompting modes x {args.repeats} repeats',flush=True)
    measurements=[]
    ordered=[(name,path,mode) for name,path in pairs for mode in modes]
    for rep in range(args.repeats):
        seq=ordered if rep%2==0 else list(reversed(ordered))
        for name,path,mode in seq:
            r=infer(path,name,mode)
            r['repeat_index']=rep+1
            measurements.append(r)
            print(f"  {name} {mode} rep={rep+1}: {r['end_to_end_ms']}ms "
                  f"parse={r['parseable']} valid={r['attributes_valid']} "
                  f"service={r['service_result'] is not None}",flush=True)
    aggregate=[]
    for name,_ in pairs:
        for mode in modes:
            group=[r for r in measurements if r['variant']==name and r['mode']==mode]
            times=[r['end_to_end_ms'] for r in group]
            aggregate.append({'variant':name,'mode':mode,'n':len(times),
                              'p50_ms':round(statistics.median(times),2),
                              'p95_interpolated_ms':round(percentile(times,.95),2),
                              'raw_native_count':sum(r['raw_native_json'] for r in group),
                              'parseable_count':sum(r['parseable'] for r in group),
                              'attributes_valid_count':sum(r['attributes_valid'] for r in group),
                              'service_result_count':sum(r['service_result'] is not None for r in group)})
    report={'schema':'prd32_consistency_dev_v5',
            'status':'DIAGNOSTIC_COMPLETED_NOT_ACCEPTED',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'bridge_case_id':row.get('case_id'),'case_index':args.case_index,
            'target_category':args.target_category,
            'target_category_manually_given_not_classification_gt':True,
            'input_image_sha256_verified':True,
            'bbox_source':'frozen_3_1_prediction_not_ground_truth',
            'input_bbox_xyxy':box,'roi_xyxy':list(crop),
            'image_variants':{'roi_target_only':str(roi_path),
                              'marked_full_image':str(marked_path)},
            'model_id':args.model_id, 'dtype':str(next(model.parameters()).dtype),
            'gpu':torch.cuda.get_device_name(0),
            'gpu_peak_allocated_gib':round(torch.cuda.max_memory_allocated()/1024**3,3),
            'aggregate':aggregate, 'measurements':measurements,
            'limitations':[
                'The human-given category does not measure automatic category recognition.',
                'The QA in service_result answers only the color question and is derived from tags, not a separate generative QA.',
                'The service copy is a deterministic template derived from predicted tags; it is not human-rated merchandising copy.',
                'The model-generated copy candidate is unverified and not automatically approved.',
                'Canonical JSON is a service serialization, not native model JSON. Native compliance reported separately.',
                'Any model field may still be visually wrong; no human GT accuracy evaluated.',
                'Single image and low repetitions are not PRD latency, QPS, or end-to-end acceptance.',
                'Sealed holdout is not used. Frozen 3.1 inputs are read-only.'
            ]}
    result_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'[4/4] Report: {result_path}\nSTATUS={report["status"]}',flush=True)


if __name__ == '__main__':
    main()
