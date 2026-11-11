import argparse, csv, hashlib, json, re, statistics, time
from pathlib import Path
from datetime import datetime, timezone

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def safe(p):
    if any(s in str(p).lower() for s in ('sealed','holdout')):
        raise ValueError('SEALED/HOLDOUT paths are forbidden')

ap=argparse.ArgumentParser()
ap.add_argument('--manifest',default='/workspace/PRD32_DEV_20261026/dev_manifest.csv')
ap.add_argument('--model-id',default='/workspace/models/Qwen2.5-VL-7B-Instruct')
ap.add_argument('--out',default='/workspace/PRD32_BATCH_V1_20261026')
ap.add_argument('--limit',type=int,default=8)
ap.add_argument('--dry-run',action='store_true')
a=ap.parse_args()
for p in (a.manifest,a.model_id,a.out):safe(p)
with open(a.manifest,encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
if not rows: raise ValueError('dev_manifest.csv is empty')
selected=[]; seen=set()
for r in rows:
    safe(r['image_path'])
    if r['image_sha256'] not in seen:
        seen.add(r['image_sha256']);selected.append(r)
    if len(selected)>=a.limit:break
print('Manifest rows:',len(rows),'unique images:',len({x['image_sha256'] for x in rows}),
      'selected:',len(selected),flush=True)
for r in selected:
    im=Path(r['image_path']);safe(im)
    if not im.is_file() or sha(im)!=r['image_sha256']:
        raise ValueError('Missing/changed image: '+str(im))
    box=json.loads(r['predicted_target_bbox_xyxy'])
    if not isinstance(box,list) or len(box)!=4 or not all(isinstance(v,(int,float)) for v in box):
        raise ValueError('Invalid bbox')
print('PREFLIGHT_OK: image hashes verified; no sealed test read',flush=True)
if a.dry_run: raise SystemExit(0)
out=Path(a.out)
if out.exists():raise FileExistsError('Choose a fresh --out directory: '+str(out))
if not Path(a.model_id).is_dir():raise FileNotFoundError('Local model missing')
import torch
from PIL import Image,ImageDraw
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
free,_=torch.cuda.mem_get_info()
if free<18*1024**3:raise RuntimeError('Need >=18 GiB free GPU memory')
out.mkdir(parents=True)
imgs=[]
for i,r in enumerate(selected):
    with Image.open(r['image_path']) as im:img=im.convert('RGB')
    w,h=img.size;x1,y1,x2,y2=json.loads(r['predicted_target_bbox_xyxy'])
    if not (0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('bbox outside image')
    ImageDraw.Draw(img).rectangle(tuple(round(z) for z in (x1,y1,x2,y2)),
                                  outline=(255,45,35),width=max(3,min(w,h)//160))
    p=out/f'marked_{i:02}.jpg';img.save(p);imgs.append(p)
print('Loading local model once...',flush=True)
m=Qwen2_5_VLForConditionalGeneration.from_pretrained(
    a.model_id,dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map='auto',attn_implementation='sdpa',local_files_only=True).eval()
proc=AutoProcessor.from_pretrained(a.model_id,min_pixels=128*28*28,
                                    max_pixels=401408,local_files_only=True)

def infer(img,prompt,limit):
    msgs=[{'role':'user','content':[{'type':'image','image':img.as_uri()},
                                   {'type':'text','text':prompt}]}]
    torch.cuda.synchronize();t0=time.perf_counter()
    chat=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    visuals,videos=process_vision_info(msgs)
    data=proc(text=[chat],images=visuals,videos=videos,return_tensors='pt').to(m.device)
    torch.cuda.synchronize();t1=time.perf_counter()
    with torch.inference_mode(): gen=m.generate(**data,max_new_tokens=limit,do_sample=False)
    torch.cuda.synchronize();t2=time.perf_counter()
    response=proc.batch_decode(gen[:,data['input_ids'].shape[1]:],skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)[0]
    return response,round((t2-t0)*1000,2),round((t1-t0)*1000,2)

prompt=('红框是3.1预测的目标范围，并非人工真值。只分析红框所指的那件服饰；'
        '叠穿的内搭、包和其他服饰不能算作该目标的属性。无法唯一确定则用null。'
        '只输出JSON对象，且恰好包含category、primary_color、pattern、visible_design四个键。'
        'category只能为top,pants,skirt,outerwear,dress,shoe,bag,accessory或null。'
        '其他三个字段分别为主要可见颜色、目标自身图案、可见结构设计；不确定用null。'
        'visible_design禁止填“无”“无明显设计”“纯色”等非结构描述。'
        '不要猜面料成分、工艺、品牌、风格，不要Markdown代码围栏或解释。')

infer(imgs[0],'图中有没有衣服？只回答有或没有。',8)
results=[]
with (out/'predictions.jsonl').open('w',encoding='utf-8') as f:
    for i,(r,img) in enumerate(zip(selected,imgs)):
        raw,ms,prep=infer(img,prompt,96)
        native=raw.strip().startswith('{') and raw.strip().endswith('}')
        cleaned=re.sub(r'^```(?:json)?\s*','',raw.strip(),flags=re.I)
        if cleaned.endswith('```'):cleaned=cleaned[:-3].strip()
        try: obj=json.loads(cleaned)
        except ValueError:obj=None
        fields={'category','primary_color','pattern','visible_design'}
        schema=(isinstance(obj,dict) and set(obj)==fields and
                obj['category'] in (None,'top','pants','skirt','outerwear','dress','shoe','bag','accessory') and
                all(obj[k] is None or (isinstance(obj[k],str) and obj[k].strip()) for k in fields-{'category'}))
        design=obj.get('visible_design') if isinstance(obj,dict) else None
        bad=bool(isinstance(design,str) and design.strip().lower() in
                 {'无','无明显设计','无设计','没有设计','纯色','无图案','无装饰','solid','plain'})
        record={'target_key':r['target_key'],'image_id':r['image_id'],
                'source_image':r['image_path'],'marked_image':str(img),
                'category_human_supplied':False,'raw':raw,'parsed':obj,
                'native_json':native,'schema_valid':bool(schema),
                'visible_design_semantically_suspect':bad,
                'latency_ms':ms,'preprocess_ms':prep,'human_gt_checked':False}
        f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();results.append(record)
        print(f'{i+1}/{len(selected)}  {ms} ms  schema={schema}  design_issue={bad}',flush=True)

with (out/'review_sheet.csv').open('w',encoding='utf-8-sig',newline='') as f:
    cols=['target_key','image_id','marked_image','pred_category','pred_primary_color',
          'pred_pattern','pred_visible_design','latency_ms','schema_valid',
          'human_review_status','human_target_category','human_primary_color',
          'human_pattern','human_visible_design','target_binding_correct','notes']
    w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
    for r in results:
        obj=r['parsed'] if isinstance(r['parsed'],dict) else {}
        w.writerow({'target_key':r['target_key'],'image_id':r['image_id'],
          'marked_image':r['marked_image'],'pred_category':obj.get('category'),
          'pred_primary_color':obj.get('primary_color'),'pred_pattern':obj.get('pattern'),
          'pred_visible_design':obj.get('visible_design'),'latency_ms':r['latency_ms'],
          'schema_valid':r['schema_valid'],'human_review_status':'NOT_REVIEWED'})
summary={'status':'DEV_BATCH_ONLY_NOT_PRD_ACCEPTANCE','samples':len(results),
         'native_json':sum(r['native_json'] for r in results),
         'schema_valid':sum(r['schema_valid'] for r in results),
         'suspect_design':sum(r['visible_design_semantically_suspect'] for r in results),
         'p50_latency_ms':statistics.median(r['latency_ms'] for r in results),
         'accuracy':None,'copy_satisfaction':None,'qps':None,
         'note':'GT not reviewed; category is model-predicted; no general VQA/copy test; sealed holdout untouched'}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
