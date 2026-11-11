#!/usr/bin/env python3
"""Read-only PRD 3.2 development batch for unsealed frozen bridge inputs.

One target per distinct image, starting from a prepared manifest. Category is
predicted without human hints. A review sheet is created, but accuracy is NEVER
computed without manually reviewed GT. Existing inputs and 3.1 weights remain
untouched. Does not implement PRD 3.2 general VQA or marketing-copy acceptance.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

CATEGORIES = {'top', 'pants', 'skirt', 'outerwear', 'dress', 'shoe', 'bag', 'accessory'}
FIELDS = ('category', 'primary_color', 'pattern', 'visible_design')
BAD_DESIGN = {'无', '无明显设计', '无设计', '没有设计', '无图案', '无装饰', '纯色', 'solid', 'plain', 'none', 'n/a', '未知', '不确定'}
GT_FIELDS = ['human_review_status', 'human_target_category', 'human_primary_color', 'human_pattern', 'human_visible_design', 'target_binding_correct', 'notes']
PROMPT = (
    '图中红色框是上一模块预测的目标范围，不是人工真值，也不是干净的掩码。'
    '仅识别红框所指的目标服饰实例，框内可能叠穿其他衣服或有包等物品；'
    '不要把其他物品的颜色、图案、袖子当作目标属性。若目标无法唯一确定，则相关字段写null。'
    '输出一个JSON对象，必须且仅有category、primary_color、pattern、visible_design四个字段；'
    'category只能是top,pants,skirt,outerwear,dress,shoe,bag,accessory或null；'
    'primary_color是目标主要可见颜色（中文短词）；pattern是目标本身图案（例如纯色、条纹），不确定用null；'
    'visible_design是清楚可见的结构设计（例如肩带、无袖、口袋），看不清用null。'
    '不要把“无”“无明显设计”“纯色”等填入visible_design；不要编造面料、工艺、品牌、风格。'
    '不得用Markdown围栏或解释文字。')

def blocked(path):
    return any(t in str(path).lower() for t in ('holdout', 'sealed'))

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            digest.update(b)
    return digest.hexdigest()

def parse_model_output(raw):
    txt = raw.strip()
    native = txt.startswith('{') and txt.endswith('}') and '```' not in txt
    if txt.startswith('```') and txt.endswith('```'):
        txt = re.sub(r'^```(?:json)?\s*', '', txt, count=1, flags=re.I)[:-3].strip()
    result = {'native_json': native, 'parseable': False, 'schema_valid': False, 'semantic_warnings': [], 'parsed': None}
    try:
        obj = json.loads(txt)
    except (ValueError, TypeError) as exc:
        result['semantic_warnings'].append('parse_error:' + str(exc))
        return result
    result['parseable'] = True
    result['parsed'] = obj
    if not isinstance(obj, dict) or set(obj) != set(FIELDS):
        result['semantic_warnings'].append('field_set_invalid')
        return result
    if obj['category'] is not None and obj['category'] not in CATEGORIES:
        result['semantic_warnings'].append('category_enum_invalid')
    for k in FIELDS[1:]:
        v = obj[k]
        if v is not None and (not isinstance(v, str) or not v.strip()):
            result['semantic_warnings'].append('value_type_invalid:' + k)
    des = obj['visible_design']
    if isinstance(des, str) and des.strip().casefold() in BAD_DESIGN:
        result['semantic_warnings'].append('non_structural_visible_design')
    result['schema_valid'] = not result['semantic_warnings']
    return result

def build_selection(manifest, limit, distinct=True):
    with manifest.open('r', newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Manifest empty. Generate it from frozen bridge first.')
    selected, seen = [], set()
    for row in rows:
        path = row.get('image_path', '')
        if blocked(path):
            raise ValueError('Sealed/holdout reference detected in manifest. Aborting.')
        key = row.get('image_sha256', '') if distinct else row.get('target_key', '')
        if not key:
            raise ValueError('Missing image SHA or target key')
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= limit:
            break
    return rows, selected

def preflight_row(row):
    path = Path(row['image_path'])
    if not path.is_file():
        raise FileNotFoundError(f'Missing image: {path}')
    if sha256(path) != row['image_sha256']:
        raise ValueError(f'Image hash mismatch: {path}')
    box = json.loads(row['predicted_target_bbox_xyxy'])
    if not isinstance(box, list) or len(box) != 4 or not all(type(x) in (int,float) and math.isfinite(x) for x in box):
        raise ValueError('Invalid prediction bbox')
    return path, box

def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--manifest', type=Path, default=Path('/workspace/PRD32_DEV_20261026/dev_manifest.csv'))
    a.add_argument('--model-id', type=Path, default=Path('/workspace/models/Qwen2.5-VL-7B-Instruct'))
    a.add_argument('--output', type=Path, default=Path('/workspace/PRD32_DEV_BATCH_V1_20261026'))
    a.add_argument('--limit', type=int, default=8)
    a.add_argument('--max-pixels', type=int, default=401408)
    a.add_argument('--max-new-tokens', type=int, default=96)
    a.add_argument('--dry-run', action='store_true', help='Verify manifest/images without loading GPU model')
    a.add_argument('--allow-same-image-targets', action='store_true', help='Use multiple targets per image')
    args = a.parse_args()
    if not 1 <= args.limit <= 100: a.error('--limit must be 1..100')
    if not 32 <= args.max_new_tokens <= 256: a.error('--max-new-tokens must be 32..256')
    if any(blocked(x) for x in (args.manifest, args.output, args.model_id)):
        a.error('Sealed/holdout paths are forbidden')
    if not args.manifest.is_file(): a.error('Manifest missing: run preparation script first')
    total, selected = build_selection(args.manifest, args.limit, not args.allow_same_image_targets)
    print(json.dumps({'manifest_rows':len(total),'selected_targets':len(selected),
                      'unique_images_total':len(set(r['image_sha256'] for r in total)),
                      'selection':'one_per_image' if not args.allow_same_image_targets else 'one_per_target',
                      'gt_prepopulated':sum(r.get('human_review_status') not in ('', 'NOT_REVIEWED') for r in selected)},
                     ensure_ascii=False,indent=2), flush=True)
    prepared = [(r,*preflight_row(r)) for r in selected]
    print('PREFLIGHT_OK: images exist and SHA256 match; no holdout accessed',flush=True)
    if args.dry_run:
        print('DRY_RUN_ONLY: no model loaded, no results written',flush=True)
        return
    if not args.model_id.is_dir(): a.error('Local model directory missing; will not fetch model over network')
    if args.output.exists(): a.error('Output directory exists: preserve runs, use a new output path')
    args.output.mkdir(parents=True, exist_ok=False)
    from PIL import Image, ImageDraw
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    free, _ = torch.cuda.mem_get_info()
    if free < 18 * 1024**3: raise RuntimeError(f'Only {free/1024**3:.2f}GiB GPU free, require >=18GiB')
    print('Loading local Qwen2.5-VL-7B once...',flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_id), device_map='auto', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        attn_implementation='sdpa', low_cpu_mem_usage=True, local_files_only=True).eval()
    processor = AutoProcessor.from_pretrained(str(args.model_id), min_pixels=128*28*28,
                                              max_pixels=args.max_pixels, local_files_only=True)
    def infer(image_file, question, max_tokens):
        messages=[{'role':'user','content':[{'type':'image','image':image_file.as_uri()},
                                            {'type':'text','text':question}]}]
        torch.cuda.synchronize(); t0=time.perf_counter()
        chat=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        imgs,vids=process_vision_info(messages)
        data=processor(text=[chat],images=imgs,videos=vids,padding=True,return_tensors='pt').to(model.device)
        torch.cuda.synchronize(); t1=time.perf_counter()
        with torch.inference_mode():
            tokens=model.generate(**data,max_new_tokens=max_tokens,do_sample=False)
        torch.cuda.synchronize(); t2=time.perf_counter()
        raw=processor.batch_decode(tokens[:,data['input_ids'].shape[1]:],skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)[0]
        return raw, {'preprocess_ms':round((t1-t0)*1000,2), 'generation_ms':round((t2-t1)*1000,2),
                     'end_to_end_ms':round((t2-t0)*1000,2),
                     'input_tokens_including_visual':int(data['input_ids'].shape[1]),
                     'generated_tokens':int(tokens.shape[1]-data['input_ids'].shape[1])}
    print('Preparing images and warming up...',flush=True)
    image_paths=[]
    for i,(r, source, box) in enumerate(prepared):
        with Image.open(source) as im: img=im.convert('RGB')
        w,h=img.size
        x1,y1,x2,y2=box
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError('Predicted bbox outside image: ' + str(source))
        marked=img.copy()
        ImageDraw.Draw(marked).rectangle(tuple(int(round(v)) for v in box),outline=(255,45,35),width=max(3,min(w,h)//170))
        p=args.output/f'case_{i:02}_marked.jpg'
        marked.save(p,quality=90)
        image_paths.append(p)
    infer(image_paths[0],'图片中是否有服饰？只回答是或否。',8)
    results=[]
    with (args.output/'predictions.jsonl').open('w',encoding='utf-8') as log:
        for i, ((r,source,box), marked) in enumerate(zip(prepared,image_paths)):
            try:
                raw,timing=infer(marked,PROMPT,args.max_new_tokens)
                parsed=parse_model_output(raw)
                record={'target_key':r['target_key'],'image_id':r['image_id'],
                        'target_instance_id':r['target_instance_id'],
                        'image_sha256':r['image_sha256'], 'bbox_source':'frozen_3_1_prediction',
                        'target_category_hint_used':False,'source_image':str(source),
                        'marked_image':str(marked), 'raw_response':raw, **timing, **parsed,
                        'human_gt_checked':False}
            except Exception as exc:
                record={'target_key':r['target_key'], 'image_id':r['image_id'],
                        'human_gt_checked':False, 'inference_error':repr(exc)}
            results.append(record)
            log.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
            print(f'{i+1}/{len(prepared)} {r["target_key"]}: '
                  f'{record.get("end_to_end_ms", "ERROR")}ms '
                  f'schema_valid={record.get("schema_valid", False)}',flush=True)
    times=[r['end_to_end_ms'] for r in results if 'end_to_end_ms' in r]
    summary={'status':'DEV_BATCH_COMPLETED_NOT_ACCEPTED',
             'created_utc':datetime.now(timezone.utc).isoformat(),
             'n_selected':len(prepared), 'n_inference_success':len(times),
             'n_schema_valid':sum(r.get('schema_valid',False) for r in results),
             'n_native_json':sum(r.get('native_json',False) for r in results),
             'median_end_to_end_ms':round(statistics.median(times),2) if times else None,
             'gpu_peak_allocated_gib':round(torch.cuda.max_memory_allocated()/1024**3,3),
             'accuracy':None,'human_copy_satisfaction':None,'qps':None,
             'warnings':['No accuracy without independent human GT',
                         'One image/target per development input; not sealed holdout',
                         '3.1 predicted bounding box and model category not ground truth',
                         'No general VQA or merchant-copy generation tested',
                         'No system throughput, service concurrency, or full 3.1+3.2 latency measured']}
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    columns=['target_key','image_id','source_image','marked_image','pred_category','pred_primary_color',
             'pred_pattern','pred_visible_design','native_json','schema_valid','end_to_end_ms',
             'human_review_status','human_target_category','human_primary_color',
             'human_pattern','human_visible_design','target_binding_correct','notes']
    with (args.output/'review_sheet.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for record in results:
            obj=record.get('parsed') if isinstance(record.get('parsed'),dict) else {}
            w.writerow({'target_key':record['target_key'],'image_id':record['image_id'],
                'source_image':record.get('source_image',''),'marked_image':record.get('marked_image',''),
                **{'pred_'+k:obj.get(k) for k in FIELDS},
                'native_json':record.get('native_json',False),
                'schema_valid':record.get('schema_valid',False),
                'end_to_end_ms':record.get('end_to_end_ms',''),
                'human_review_status':'NOT_REVIEWED'})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('Review sheet:',args.output/'review_sheet.csv',flush=True)

if __name__=='__main__':
    main()
