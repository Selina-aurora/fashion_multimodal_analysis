import csv, hashlib, json, re, statistics, time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('/workspace')
SRC = ROOT / 'PRD32_DEV_20261026/dev_manifest.csv'
OUT = ROOT / 'PRD32_BATCH3_20261026'
MODEL = ROOT / 'models/Qwen2.5-VL-7B-Instruct'
STATUS = {
    'new1022_0003': ('USE_WITH_OVERLAP', 'Prior development case; dress overlap with striped top and bag'),
    'new1022_0004': ('SKIP_BAD_BBOX', 'BBox covers a narrow waist band rather than a full garment'),
    'new1022_0011': ('USE_PARTIAL', 'Light blue dress; crossed straps; incomplete edges'),
    'new1022_0018': ('USE_OCCLUDED', 'Gray-white button-front printed garment; hair occlusion; class needs review'),
}
if OUT.exists():
    raise SystemExit('Output exists; do not overwrite: ' + str(OUT))
with SRC.open(newline='', encoding='utf-8-sig') as f:
    source_rows = list(csv.DictReader(f))
if {r['image_id'] for r in source_rows} != set(STATUS) or len(source_rows) != 4:
    raise SystemExit('Manifest changed; inspect before selecting targets')
selected = []
for row in source_rows:
    tag, note = STATUS[row['image_id']]
    row['visual_screening_status'] = tag
    row['visual_screening_notes'] = note
    if any(v in row['image_path'].lower() for v in ('sealed', 'holdout')):
        raise SystemExit('Sealed holdout is forbidden')
    fpath = Path(row['image_path'])
    if not fpath.is_file() or hashlib.sha256(fpath.read_bytes()).hexdigest() != row['image_sha256']:
        raise SystemExit('Image missing or hash mismatch: ' + str(fpath))
    if tag.startswith('USE_'):
        selected.append(row)
OUT.mkdir(parents=True)
keys = list(source_rows[0].keys())
for name, rows in [('screening_all4.csv',source_rows), ('development_selected3.csv', selected)]:
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
print('SCREENING_OK: 4 reviewed visually, 3 diagnostic candidates, 1 bad bbox excluded',flush=True)
print('NOTE: screenshot review is not full ground-truth annotation',flush=True)

from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable')
if not MODEL.is_dir():
    raise SystemExit('Local model missing; refusing to download')
print('Loading local model once...',flush=True)
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(MODEL),local_files_only=True,
         device_map='auto',dtype=torch.bfloat16,attn_implementation='sdpa').eval()
processor=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True,max_pixels=401408)
prompt=('请只识别红色矩形内的目标服饰，框内可能还有内搭、包或头发遮挡。'
 '不得将其他服饰的属性算到目标上。只输出一个 JSON 对象，键恰好为 '
 'category, primary_color, pattern, visible_design。category 只可为 '
 'top,pants,skirt,outerwear,dress,shoe,bag,accessory 或 null。'
 '其他字段用简短中文或 null；visible_design 仅描述肩带、袖子、纽扣、领口等可见结构，'
 '不得写“无”“无明显设计”。未知就 null。不编造面料成分和工艺。')

def infer(img_path):
    messages=[{'role':'user','content':[{'type':'image','image':img_path.as_uri()},
                                        {'type':'text','text':prompt}]}]
    torch.cuda.synchronize(); t0=time.perf_counter()
    txt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    imgs, vids=process_vision_info(messages)
    enc=processor(text=[txt],images=imgs,videos=vids,padding=True,return_tensors='pt').to(model.device)
    torch.cuda.synchronize(); t1=time.perf_counter()
    with torch.inference_mode():
        ids=model.generate(**enc,max_new_tokens=96,do_sample=False)
    torch.cuda.synchronize(); t2=time.perf_counter()
    raw=processor.batch_decode(ids[:,enc.input_ids.shape[1]:],skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)[0]
    return raw,round((t2-t0)*1000,2),round((t1-t0)*1000,2),round((t2-t1)*1000,2)

records=[]
for ix,row in enumerate(selected):
    path=Path(row['image_path'])
    with Image.open(path) as im:
        img=im.convert('RGB')
    bbox=json.loads(row['predicted_target_bbox_xyxy'])
    x1,y1,x2,y2=bbox
    if not (0<=x1<x2<=img.width and 0<=y1<y2<=img.height):
        raise RuntimeError('Invalid bbox for ' + row['image_id'])
    marked=img.copy()
    ImageDraw.Draw(marked).rectangle(tuple(round(v) for v in bbox),outline=(255,35,35),width=4)
    target=OUT/f'case_{ix:02}_{row["image_id"]}.jpg'
    marked.save(target,quality=92)
    raw,ms,pre,gen=infer(target)
    native=raw.strip().startswith('{') and raw.strip().endswith('}')
    cleaned=re.sub(r'^```(?:json)?\s*','',raw.strip(),flags=re.I).removesuffix('```').strip()
    try:
        parsed=json.loads(cleaned)
    except json.JSONDecodeError:
        parsed=None
    valid=isinstance(parsed,dict) and set(parsed)=={'category','primary_color','pattern','visible_design'}
    invalid_design=isinstance(parsed,dict) and str(parsed.get('visible_design')).strip() in {'无','无明显设计','无设计','无图案'}
    rec={'image_id':row['image_id'],'target_key':row['target_key'],
         'screening_status':row['visual_screening_status'],'raw_response':raw,'parsed':parsed,
         'native_json':native,'structure_valid':valid,'invalid_design':invalid_design,
         'latency_ms':ms,'preprocess_ms':pre,'generation_ms':gen,
         'accuracy':None,'human_gt_verified':False,'marked_image':str(target)}
    records.append(rec)
    print(f'{ix+1}/{len(selected)} {row["image_id"]}: {ms} ms, parsed={valid}',flush=True)
with (OUT/'predictions.jsonl').open('w',encoding='utf-8') as f:
    for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
summary={'status':'BATCH3_DEV_ONLY_NOT_PRD_ACCEPTED','n_input':4,'n_screened_out':1,
         'n_inferred':len(records),'model':str(MODEL),'mean_latency_ms':round(statistics.mean(r['latency_ms'] for r in records),2),
         'n_structural_json':sum(r['structure_valid'] for r in records),
         'n_native_json':sum(r['native_json'] for r in records),
         'n_invalid_design':sum(r['invalid_design'] for r in records),
         'accuracy':None,'gt_verified':False,'holdout_used':False,
         'warning':'One previously seen dev case, no GT, not PRD accuracy/latency acceptance'}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
