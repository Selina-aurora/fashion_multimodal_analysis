import csv,json,hashlib,re,time,statistics
from pathlib import Path
from PIL import Image,ImageDraw
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

manifest=Path('/workspace/PRD32_DEV_20261026/dev_manifest.csv')
model_path='/workspace/models/Qwen2.5-VL-7B-Instruct'
out=Path('/workspace/PRD32_DEV_BATCH_20261026')
if out.exists():raise FileExistsError('Output exists; use a new output directory')
with manifest.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
if not rows:raise ValueError('Empty dev manifest')
selected=[];seen=set()
for r in rows:
    if any(z in r['image_path'].lower() for z in ('sealed','holdout')):
        raise ValueError('Sealed holdout forbidden')
    if r['image_sha256'] not in seen:
        seen.add(r['image_sha256']); selected.append(r)
    if len(selected)==8:break
print('total rows',len(rows),'distinct selected images',len(selected),flush=True)
for r in selected:
    p=Path(r['image_path']);h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=r['image_sha256']:raise ValueError('SHA mismatch: '+str(p))
if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
free,_=torch.cuda.mem_get_info()
if free<18*1024**3:raise RuntimeError('Not enough free GPU memory')
out.mkdir(parents=True)
marked=[]
for i,r in enumerate(selected):
    with Image.open(r['image_path']) as im:im=im.convert('RGB')
    coords=json.loads(r['predicted_target_bbox_xyxy']);w,h=im.size
    x1,y1,x2,y2=coords
    if not (0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('Invalid bbox')
    ImageDraw.Draw(im).rectangle(tuple(map(round,coords)),outline=(255,45,35),width=4)
    p=out/f'marked_{i:02}.jpg';im.save(p);marked.append(p)
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map='auto',attn_implementation='sdpa',local_files_only=True).eval()
processor=AutoProcessor.from_pretrained(model_path,max_pixels=401408,local_files_only=True)

def run(img,prompt,tokens):
    msg=[{'role':'user','content':[{'type':'image','image':img.as_uri()},
                                   {'type':'text','text':prompt}]}]
    torch.cuda.synchronize();t0=time.perf_counter()
    inp=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
    images,videos=process_vision_info(msg)
    x=processor(text=[inp],images=images,videos=videos,return_tensors='pt').to(model.device)
    torch.cuda.synchronize();t1=time.perf_counter()
    with torch.inference_mode():y=model.generate(**x,max_new_tokens=tokens,do_sample=False)
    torch.cuda.synchronize();t2=time.perf_counter()
    ans=processor.batch_decode(y[:,x['input_ids'].shape[1]:],skip_special_tokens=True)[0]
    return ans,round((t2-t0)*1000,2),round((t1-t0)*1000,2)

prompt=('红框是预测目标范围，仅分析该目标服饰本身，不把叠穿内搭、背包等其他物品算入。'
        '只返回JSON对象，字段恰好为category、primary_color、pattern、visible_design。'
        'category只允许top,pants,skirt,outerwear,dress,shoe,bag,accessory或null。'
        '颜色、图案、结构设计看不清则填null，不能把“无”“无明显设计”当结构设计。'
        '不要猜测面料、工艺、风格或品牌；不要Markdown围栏。')
run(marked[0],'图中是否有服饰？只回答是或否',8)
results=[]
with (out/'predictions.jsonl').open('w',encoding='utf-8') as f:
    for i,(r,img) in enumerate(zip(selected,marked)):
        raw,ms,pre=run(img,prompt,96)
        clean=re.sub(r'^```(?:json)?\s*','',raw.strip(),flags=re.I)
        if clean.endswith('```'):clean=clean[:-3].strip()
        try:parsed=json.loads(clean)
        except ValueError:parsed=None
        keys={'category','primary_color','pattern','visible_design'}
        valid=(isinstance(parsed,dict) and set(parsed)==keys and
               parsed['category'] in (None,'top','pants','skirt','outerwear','dress','shoe','bag','accessory') and
               all(parsed[k] is None or isinstance(parsed[k],str) and bool(parsed[k].strip())
                   for k in keys-{'category'}))
        design=parsed.get('visible_design') if isinstance(parsed,dict) else None
        if isinstance(design,str) and design.strip() in ('无','无明显设计','无设计','无图案','纯色'):
            valid=False
        record={'target_key':r['target_key'],'image_id':r['image_id'],
                'marked_image':str(img),'raw':raw,'parsed':parsed,
                'native_json':raw.strip().startswith('{') and raw.strip().endswith('}'),
                'validated_structure_only':bool(valid),'latency_ms':ms,'preprocess_ms':pre,
                'human_gt_checked':False}
        f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();results.append(record)
        print(i+1,'/',len(selected),'latency_ms',ms,'structure_valid',valid,flush=True)
with (out/'review_sheet.csv').open('w',newline='',encoding='utf-8-sig') as f:
    names=['target_key','marked_image','pred_category','pred_primary_color','pred_pattern',
           'pred_visible_design','human_review_status','human_target_category',
           'human_primary_color','human_pattern','human_visible_design','notes']
    w=csv.DictWriter(f,fieldnames=names);w.writeheader()
    for r in results:
        p=r['parsed'] if isinstance(r['parsed'],dict) else {}
        w.writerow({'target_key':r['target_key'],'marked_image':r['marked_image'],
                    'pred_category':p.get('category'),'pred_primary_color':p.get('primary_color'),
                    'pred_pattern':p.get('pattern'),'pred_visible_design':p.get('visible_design'),
                    'human_review_status':'NOT_REVIEWED'})
summary={'status':'DEVELOPMENT_ONLY_NOT_PRD_ACCEPTED','n':len(results),
         'structure_valid':sum(r['validated_structure_only'] for r in results),
         'median_latency_ms':statistics.median(x['latency_ms'] for x in results),
         'accuracy':None,'copy_satisfaction':None,'qps':None,
         'note':'No human GT yet; no general QA or marketing copy evaluation; no sealed holdout used'}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
