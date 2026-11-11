#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD3.2 DEV ONLY: audit historical instance masks; compare SAME-session ROI vs mask-guided ROI.

NO production or PRD accuracy claims. Historical GT-pilot instance mask is PRIVILEGED
localization information, not frozen V5 segmentation; never pass class labels as text
or claim end-to-end accuracy. Protected holdout/test paths are never opened.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, io, json, math, statistics, time, base64, html
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_ROI_RUN = '/workspace/PRD32_BINDING_ROI_GPU_20261027_run01'
DEFAULT_PROJECT = '/workspace/fashion_multimodal_analysis'
DEFAULT_INVENTORY = '/workspace/fashion_multimodal_analysis/configs/prd_8class_unified_manifest_v2.csv'
DEFAULT_ALLOWED_DATA = '/workspace/fashion_data'
DEFAULT_MODEL = '/workspace/models/Qwen2.5-VL-7B-Instruct'
FORBIDDEN = {'holdout','sealed','test','testing','fullval','val','valid','validation','testset'}
CLASSES = {'top','pants','skirt','outerwear','dress','shoe','bag','accessory'}


def jread(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def jwrite(p, o): Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def readjl(p):
    with Path(p).open(encoding='utf-8-sig') as f:return [json.loads(s) for s in f if s.strip()]
def writejl(p, rows):
    with Path(p).open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def beneath(p, root):
    return Path(p).resolve().is_relative_to(Path(root).resolve())
def is_train_image(p, allowed):
    p=Path(p).resolve(); allowed=Path(allowed).resolve()
    if not p.is_relative_to(allowed):return False
    parts={s.lower() for s in p.relative_to(allowed).parts}
    return 'train' in parts and not (parts & FORBIDDEN)
def allowed_mask(p,project):
    p=Path(p).resolve(); proj=Path(project).resolve()
    if not p.is_relative_to(proj):return False
    segments={s.lower() for s in p.relative_to(proj).parts}
    if segments & FORBIDDEN or any('holdout' in x or 'sealed' in x for x in segments):return False
    # Only historical derived instance masks within the local project outputs.
    return p.is_relative_to(proj/'outputs') and p.suffix.lower() in ('.png','.webp','.tif','.tiff')
def resolve_project_path(project,val):
    if not isinstance(val,str) or not val.strip():return None
    return (Path(project)/val.strip()).resolve()
def finite_bbox(x):
    if not isinstance(x,list) or len(x)!=4:return False
    return all(isinstance(q,(float,int)) and not isinstance(q,bool) and math.isfinite(q) for q in x) and 0<=x[0]<x[2] and 0<=x[1]<x[3]
def bbox_match(a,b,tol=2.0):return all(abs(float(x)-float(y))<=tol for x,y in zip(a,b))
def box_of(row):
    try:return [float(row['bbox_'+k]) for k in ('x1','y1','x2','y2')]
    except (ValueError,KeyError,TypeError):return []
def crop_rect(box,w,h,pad=.05):
    x1,y1,x2,y2=box;dx=(x2-x1)*pad;dy=(y2-y1)*pad
    return (max(0,math.floor(x1-dx)),max(0,math.floor(y1-dy)),
            min(w,math.ceil(x2+dx)),min(h,math.ceil(y2+dy)))

def decode_mask(mask_path,image_wh,bbox):
    """Return full-canvas boolean mask (strictly validated); NEVER auto-invert."""
    w,h=image_wh
    with Image.open(mask_path) as im:
        if im.width*im.height>50_000_000:raise ValueError('MASK_OVERSIZED')
        mode=im.mode
        if mode=='RGBA':
            alpha=np.asarray(im.getchannel('A'))
            if alpha.min()<128 and alpha.max()>128:
                raw=alpha>127
            else:
                rgb=np.asarray(im.convert('RGB')).astype('float32')
                raw=rgb.mean(axis=2)>127
        elif mode in ('P','1'):
            raw=np.asarray(im)>0
        elif mode in ('L','LA'):
            raw=np.asarray(im.convert('L'))>127
        elif mode=='RGB':
            rgb=np.asarray(im).astype('float32')
            # No colored semantic label / multicolor map; only grayscale masks.
            if (np.max(rgb,axis=2)-np.min(rgb,axis=2)>15).mean()>.01:raise ValueError('MASK_NOT_BINARY_GRAYSCALE')
            raw=rgb.mean(axis=2)>127
        else:raise ValueError('MASK_MODE_UNSUPPORTED:'+mode)
        mh,mw=raw.shape
    if (mw,mh)==(w,h):
        canvas=raw
        placement='FULL_IMAGE'
    else:
        x1,y1,x2,y2=bbox
        bw=round(x2)-round(x1);bh=round(y2)-round(y1)
        if abs(mw-bw)>2 or abs(mh-bh)>2:raise ValueError(f'MASK_DIMENSION_UNSUPPORTED:{mw}x{mh}_image:{w}x{h}_bbox:{bw}x{bh}')
        canvas=np.zeros((h,w),dtype=bool)
        ix,iy=round(x1),round(y1)
        if ix<0 or iy<0 or ix+mw>w or iy+mh>h:raise ValueError('MASK_PATCH_OUTSIDE_IMAGE')
        canvas[iy:iy+mh,ix:ix+mw]=raw
        placement='BBOX_LOCAL'
    if canvas.sum()<25:raise ValueError('MASK_FOREGROUND_TOO_SMALL')
    bx1,by1,bx2,by2 = [round(a) for a in bbox]
    inside=canvas[max(0,by1):min(h,by2),max(0,bx1):min(w,bx2)]
    fg=int(canvas.sum());inside_fg=int(inside.sum())
    if inside_fg<25 or inside_fg/fg<.97:raise ValueError('MASK_LEAKS_OUTSIDE_TARGET_BBOX')
    area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1]);coverage=inside_fg/max(1,area)
    if not 0.015<=coverage<=.985:raise ValueError(f'MASK_BBOX_COVERAGE_IMPLAUSIBLE:{coverage:.3f}')
    return canvas, {'mask_mode':mode,'mask_placement':placement,'foreground_pixels':fg,'bbox_coverage':round(coverage,4),'inside_fraction':round(inside_fg/fg,4)}

def masked_roi_from_existing(original_roi, mask_bool, bbox, image_wh, output, pad=.05):
    """Preserve exactly the existing ROI pixels; neutralize ONLY outside target mask.
    We do not create a new image crop with different pixel framing/resolution.
    """
    w,h=image_wh; a,b,c,d=crop_rect(bbox,w,h,pad)
    with Image.open(original_roi) as im:
        im=im.convert('RGB'); roi=np.asarray(im).copy()
    if (roi.shape[1],roi.shape[0])!=(c-a,d-b):
        raise ValueError(f'ROI_CROP_PIXEL_DIM_MISMATCH:{roi.shape[1]}x{roi.shape[0]} != {c-a}x{d-b}')
    mask_roi=mask_bool[b:d,a:c]
    if not mask_roi.any():raise ValueError('MASK_NOT_INSIDE_ROI')
    # Light neutral background preserves framing but suppresses other garments.
    roi[~mask_roi]=np.array([216,216,216],dtype=np.uint8)
    img=Image.fromarray(roi,'RGB')
    box_rel=(round(bbox[0]-a),round(bbox[1]-b),round(bbox[2]-a),round(bbox[3]-b))
    ImageDraw.Draw(img).rectangle(box_rel,outline=(250,40,40),width=max(3,min(w,h)//150))
    img.save(output,'JPEG',quality=94)
    return output

def to_data_uri(path, max_w=420):
    with Image.open(path) as im:
        im=im.convert('RGB');im.thumbnail((max_w,430))
        b=io.BytesIO();im.save(b,format='JPEG',quality=77,optimize=True)
    return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode('ascii')

def render_audit_html(rows, out):
    cards=[]
    for r in rows:
        status='READY' if r['decision']=='READY' else 'QUARANTINED'
        desc=html.escape(','.join(r['reason_codes']) or 'Mask 与目标身份、尺寸、几何核验通过')
        images=''
        if status=='READY':
            a=to_data_uri(r['original_roi']);b=to_data_uri(r['mask_guided_roi'])
            images=f'<div class="pair"><figure><img src="{a}"><figcaption>原有 ROI（A）</figcaption></figure><figure><img src="{b}"><figcaption>Mask 引导 ROI（B）</figcaption></figure></div>'
        cards.append(f'<article><h2>{html.escape(r["case_id"])} · {status}</h2><p>人工复核的 ROI 混入其他服饰：{html.escape(str(r["roi_single_item"]))}；原 ROI 预测：{html.escape(str(r["old_roi_category"]))}</p><p>{desc}</p>{images}</article>')
    doc='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PRD3.2 Mask 目标隔离预检</title><style>body{background:#f4f6fa;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif;color:#1b293b;margin:0}main{max-width:1190px;margin:auto;padding:23px}header{background:#193054;color:white;padding:20px;border-radius:12px}article{background:white;border:1px solid #dce4ec;border-radius:12px;padding:18px;margin:17px 0}h2{font-size:17px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:13px}figure{background:#edf2f7;padding:10px;border-radius:10px;text-align:center;margin:0}img{max-width:100%;max-height:440px}figcaption{font-size:12px;color:#52647a}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style></head><body><main><header><h1>PRD 3.2 · GT-pilot Mask 身份核验</h1><p>仅 TRAIN 开发错误选样；A/B 图像静态内嵌，非 GPU 推理结果；人工 GT 不会传给模型。此 Mask 不是冻结 V5 预测。</p></header>'''+''.join(cards)+'</main></body></html>'
    Path(out).write_text(doc,encoding='utf-8')

def audit(args):
    out=Path(args.output).resolve()
    if out.exists():raise FileExistsError(f'Output exists; use new name: {out}')
    run_dir=Path(args.roi_run_dir).resolve();project=Path(args.project_root).resolve();data_root=Path(args.allowed_data_root).resolve()
    if not beneath(args.inventory,project):raise ValueError('INVENTORY_OUTSIDE_PROJECT')
    review=jread(args.review)
    if review.get('scope')!='train_dev_not_independent':raise ValueError('REVIEW_SCOPE_MISMATCH')
    human={r['case_id']:r for r in review['rows']}
    plans=readjl(run_dir/'case_plan.jsonl'); preds={r['case_id']:r for r in readjl(run_dir/'roi_predictions.jsonl')}
    if len(plans)!=15 or len(human)!=15 or len(preds)!=15 or len({r['case_id'] for r in plans})!=15 or set(human)!=set(preds) or set(human)!={r['case_id'] for r in plans}:raise ValueError('EXPECTED_SAME_15_CASE_IDS')
    with Path(args.inventory).open(encoding='utf-8-sig',newline='') as f: inventory=list(csv.DictReader(f))
    idx=defaultdict(list)
    for row in inventory:
        src=resolve_project_path(project,row.get('source_image',''))
        if src and is_train_image(src,data_root):
            for token in {row.get('garment_id',''),row.get('record_id','')}:
                if token:idx[(str(src),token)].append(row)
    out.mkdir(parents=True); (out/'masked_inputs').mkdir()
    decisions=[];ready=[]
    for p in plans:
        cid=p['case_id']; reasons=[]
        previous=preds.get(cid);review_row=human.get(cid)
        record={'case_id':cid,'decision':'QUARANTINED','reason_codes':reasons,
            'roi_single_item':(review_row or {}).get('roi_single_item'),
            'review_cause':(review_row or {}).get('cause_revision'),
            'old_roi_category':(previous or {}).get('roi_category'),
            'reviewed_category_EVAL_ONLY':p.get('human_category_FOR_EVAL_ONLY'),
            'historical_GT_mask_not_V5':True}
        source=Path(p.get('image_path','')).resolve();bbox=p.get('bbox')
        if not review_row or not previous or previous.get('status')!='GENERATED_UNVERIFIED':reasons.append('REVIEW_OR_BASELINE_MISSING')
        if not finite_bbox(bbox):reasons.append('BBOX_INVALID')
        if not is_train_image(source,data_root):reasons.append('NOT_EXPLICIT_TRAIN_IMAGE')
        if not source.is_file():reasons.append('IMAGE_MISSING')
        if reasons:decisions.append(record);continue
        if sha(source)!=p['image_sha256']:reasons.append('SOURCE_SHA_MISMATCH')
        origin=Path(previous.get('roi_input','')).resolve()
        if not beneath(origin,run_dir/'roi_marked_inputs') or not origin.is_file():reasons.append('OLD_ROI_MISSING_OR_OUTSIDE_RUN')
        if reasons:decisions.append(record);continue
        with Image.open(source) as im: im_size=im.size
        if not (0<=bbox[0]<bbox[2]<=im_size[0] and 0<=bbox[1]<bbox[3]<=im_size[1]):reasons.append('BOX_OUTSIDE_ACTUAL_IMAGE')
        if reasons:decisions.append(record);continue
        matches=[r for r in idx.get((str(source),p['target_instance_id']),[]) if finite_bbox(box_of(r)) and bbox_match(bbox,box_of(r))]
        # Identical duplicate manifest rows must still be disambiguated rather than guessed.
        if len(matches)!=1:
            reasons.append('INSTANCE_MATCH_NOT_UNIQUE' if matches else 'INSTANCE_MASK_ROW_NOT_FOUND')
            decisions.append(record);continue
        matched=matches[0]
        mask_path=resolve_project_path(project,matched.get('mask_path',''))
        if mask_path is None or not allowed_mask(mask_path,project):reasons.append('MASK_PATH_NOT_ALLOWED_OR_EMPTY')
        elif not mask_path.is_file():reasons.append('MASK_FILE_NOT_FOUND')
        if reasons:decisions.append(record);continue
        try:
            mb,stats=decode_mask(mask_path,im_size,bbox)
            dst=out/'masked_inputs'/(cid+'_mask_guided.jpg')
            masked_roi_from_existing(origin,mb,bbox,im_size,dst,pad=args.pad)
        except Exception as ex:
            reasons.append(str(ex)[:125] if isinstance(ex,ValueError) else type(ex).__name__+':'+str(ex)[:125]);decisions.append(record);continue
        record.update(decision='READY',image_path=str(source),image_sha256=sha(source),bbox=bbox,
            target_instance_id=p['target_instance_id'],original_roi=str(origin),original_roi_sha256=sha(origin),
            mask_guided_roi=str(dst),mask_guided_sha256=sha(dst),mask_path=str(mask_path),mask_sha256=sha(mask_path),
            mask_stats=stats,source_inventory_record_id=matched.get('record_id',''),source_dataset=matched.get('source_dataset',''))
        ready.append(record);decisions.append(record)
    writejl(out/'mask_decisions.jsonl',decisions); writejl(out/'ready_cases.jsonl',ready)
    ready_mix=sum(x['roi_single_item']=='NO' for x in ready)
    summary={'schema':'prd32_mask_guided_paired_dev_v1','status':'MASK_SOURCE_AUDIT_NOT_PRD_ACCEPTED',
        'source_roi_case_count':len(plans),'eligible_mask_cases':len(ready), 'quarantined':len(plans)-len(ready),
        'ready_with_other_garments_in_old_roi':ready_mix,
        'failure_codes':dict(Counter(reason.split(':',1)[0] for r in decisions for reason in r['reason_codes'])),
        'historical_GT_pilot_mask_not_frozen_V5':True,'source_scope':'TRAIN_DEV_ERROR_SELECTED',
        'no_gpu_used':True,'holdout_used':False,'human_gt_passed_to_model':False,
        'input_manifest_sha256':sha(args.inventory),'source_roi_plan_sha256':sha(run_dir/'case_plan.jsonl'),
        'corrected_visual_review_sha256':sha(args.review),
        'model_helper_sha256':sha(args.helper),'pad':args.pad}
    jwrite(out/'audit_summary.json',summary)
    render_audit_html(decisions,out/'MASK_GUIDED_VISUAL_CHECK_STATIC.html')
    print('MASK_AUDIT_COMPLETE',json.dumps({'eligible':len(ready),'quarantined':len(plans)-len(ready),'total':len(plans),'output':str(out)},ensure_ascii=False),flush=True)
    return summary

def loadhelper(p):
    spec=importlib.util.spec_from_file_location('prd32maskfixedv2',str(p));mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod);return mod

def model_run(args):
    source=Path(args.preflight_dir).resolve(); pre=jread(source/'audit_summary.json')
    if pre['status']!='MASK_SOURCE_AUDIT_NOT_PRD_ACCEPTED':raise ValueError('INCORRECT_PREFLIGHT')
    if pre['historical_GT_pilot_mask_not_frozen_V5'] is not True or pre['source_scope']!='TRAIN_DEV_ERROR_SELECTED':raise ValueError('UNSAFE_PROVENANCE')
    rows=readjl(source/'ready_cases.jsonl')
    if not rows:raise ValueError('NO_ELIGIBLE_MASKS; inspect audit_summary.json before GPU')
    if len(rows)!=pre['eligible_mask_cases']:raise ValueError('READY_COUNT_MISMATCH')
    ids=[r['case_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('DUPLICATE_CASE_IDS')
    if args.limit>0:rows=rows[:args.limit]
    out=Path(args.output).resolve()
    if out.exists():raise FileExistsError(f'Output already exists: {out}')
    mod=loadhelper(args.helper)
    if sha(args.helper)!=pre['model_helper_sha256']:raise ValueError('PROMPT_HELPER_CHANGED')
    for r in rows:
        if not is_train_image(r['image_path'],args.allowed_data_root):raise ValueError('PROTECTED_OR_EXTERNAL_SOURCE')
        if not allowed_mask(r['mask_path'],args.project_root):raise ValueError('UNSAFE_MASK_PATH')
        for path,key in [(r['image_path'],'image_sha256'),(r['mask_path'],'mask_sha256'),(r['original_roi'],'original_roi_sha256'),(r['mask_guided_roi'],'mask_guided_sha256')]:
            if not Path(path).is_file() or sha(path)!=r[key]:raise ValueError('ASSET_CHANGED_OR_MISSING:'+r['case_id']+':'+key)
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available():raise RuntimeError('CUDA_NOT_AVAILABLE')
    free_gib=torch.cuda.mem_get_info()[0]/2**30
    if free_gib<17:raise RuntimeError(f'GPU_MEMORY_LOW:{free_gib:.2f}GiB')
    out.mkdir(parents=True)
    start=time.perf_counter()
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,local_files_only=True,device_map='auto',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    processor=AutoProcessor.from_pretrained(args.model,local_files_only=True,max_pixels=512*28*28)
    load_sec=round(time.perf_counter()-start,2)
    results=[]
    with (out/'paired_predictions.jsonl').open('w',encoding='utf-8') as f:
        for idx,r in enumerate(rows,1):
            result={'case_id':r['case_id'],'baseline_roi_category_old_FOR_REFERENCE':r['old_roi_category'],
                'reviewed_category_FOR_POSTHOC_ONLY':r['reviewed_category_EVAL_ONLY'],
                'roi_mixed_others_human':r['roi_single_item'], 'mask_stats':r['mask_stats'],
                'status':'RUNTIME_FAILED','A_roi':None,'B_mask_guided':None}
            try:
                request={'task':'tags','case_id':r['case_id'],'image_path':r['image_path'],'image_sha256':r['image_sha256'],
                    'target_bbox_xyxy':r['bbox'],'target_instance_id':r['target_instance_id']}
                for key,image_path in [('A_roi',r['original_roi']),('B_mask_guided',r['mask_guided_roi'])]:
                    # No reviewed label, source label, or original model category in the request/prompt.
                    raw,timing=mod.run_model_case(model,processor,request,Path(image_path),mod.TAG_PROMPT,task_limits={'tags':args.max_new_tokens})
                    parsed=mod.parse_tag_output(raw)
                    tags=parsed.get('canonical_service_json')
                    if tags and tags.get('pattern') not in (None,'纯色','条纹','格纹','人字纹','波点','花卉印花','文字印花','字母印花','几何印花','动物纹','拼色'):
                        tags=dict(tags);tags['pattern']=None;parsed.setdefault('issues',[]).append('NONSTANDARD_PATTERN_ABSTAIN')
                    result[key]={'category':tags.get('category') if tags else None,'canonical_tags':tags,
                        'raw_model_output':raw,'native_json':parsed.get('raw_native_json'),
                        'schema_valid':parsed.get('schema_valid'),'issues':parsed.get('issues',[]),'timing':timing}
                result['status']='GENERATED_UNVERIFIED'
                human=r['reviewed_category_EVAL_ONLY']
                result['A_matches_review']=result['A_roi']['category']==human
                result['B_matches_review']=result['B_mask_guided']['category']==human
            except Exception as ex:
                result['failure_type']=type(ex).__name__;result['failure_detail']=str(ex)[:220]
            results.append(result);f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
            print(f'[{idx}/{len(rows)}] {r["case_id"]}: {result["status"]} A={result["A_roi"]["category"] if result["A_roi"] else "-"} B={result["B_mask_guided"]["category"] if result["B_mask_guided"] else "-"}',flush=True)
    good=[r for r in results if r['status']=='GENERATED_UNVERIFIED']
    paired={'A_wrong_B_correct':sum(not r['A_matches_review'] and r['B_matches_review'] for r in good),
            'A_correct_B_wrong':sum(r['A_matches_review'] and not r['B_matches_review'] for r in good),
            'A_correct_B_correct':sum(r['A_matches_review'] and r['B_matches_review'] for r in good),
            'A_wrong_B_wrong':sum(not r['A_matches_review'] and not r['B_matches_review'] for r in good)}
    summary={'schema':'prd32_mask_guided_paired_dev_results_v1', 'status':'DEV_MASK_PAIRED_DONE_NOT_PRD_ACCEPTED',
        'mask_eligible':len(rows),'both_conditions_completed':len(good),'failed':len(rows)-len(good),'pairs':paired,
        'observed_A_correct':sum(r['A_matches_review'] for r in good),'observed_B_correct':sum(r['B_matches_review'] for r in good),
        'median_A_ms':round(statistics.median([r['A_roi']['timing']['end_to_end_ms'] for r in good]),2) if good else None,
        'median_B_ms':round(statistics.median([r['B_mask_guided']['timing']['end_to_end_ms'] for r in good]),2) if good else None,
        'model_load_seconds':load_sec,'human_gt_passed_to_model':False,'historical_GT_pilot_mask_not_frozen_V5':True,
        'development_error_selection_only':True,'sealed_holdout_used':False,'full_prd_acceptance':False}
    jwrite(out/'summary.json',summary)
    print('MASK_PAIRED_DEV_DONE',json.dumps(summary,ensure_ascii=False),flush=True)

def self_test():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as t:
        root=Path(t);proj=root/'project';dr=root/'fashion_data';imgdir=dr/'raw'/'train'/'train'/'image';imgdir.mkdir(parents=True)
        (proj/'outputs'/'garment_instances'/'gt_pilot').mkdir(parents=True)
        im=Image.new('RGB',(160,120),(200,210,220));ip=imgdir/'001.jpg';im.save(ip)
        bb=[35,20,125,110]
        mm=np.zeros((120,160),dtype=np.uint8);mm[27:103,44:116]=255
        mask_path=proj/'outputs'/'garment_instances'/'gt_pilot'/'001_mask.png';Image.fromarray(mm,'L').save(mask_path)
        with Image.open(ip) as inp: pixels=inp.size
        mask,stats=decode_mask(mask_path,pixels,bb)
        assert stats['inside_fraction']==1 and stats['mask_placement']=='FULL_IMAGE'
        roi=root/'roi.jpg';a,b,c,d=crop_rect(bb,160,120);im.crop((a,b,c,d)).save(roi)
        masked=root/'guided.jpg';masked_roi_from_existing(roi,mask,bb,pixels,masked)
        assert Image.open(roi).size==Image.open(masked).size
        assert is_train_image(ip,dr) and not is_train_image(dr/'raw/test/a.jpg',dr)
        assert allowed_mask(mask_path,proj)
        try:
            imask=np.ones((120,160),dtype=np.uint8)*255
            bad=root/'all_white.png';Image.fromarray(imask).save(bad)
            decode_mask(bad,pixels,bb)
            raise AssertionError('bad all-white mask accepted')
        except ValueError:pass
        # Deliberately mismatched mask dimensions must fail closed.
        badshape=root/'badshape.png';Image.new('L',(45,45),255).save(badshape)
        try:decode_mask(badshape,pixels,bb);raise AssertionError('bad dimensions accepted')
        except ValueError:pass
        html_path=root/'test.html';render_audit_html([{'case_id':'x','decision':'READY','reason_codes':[],'roi_single_item':'NO','old_roi_category':'top','original_roi':str(roi),'mask_guided_roi':str(masked)}],html_path)
        assert '<img src="data:image/jpeg;base64,' in html_path.read_text()
        mod=loadhelper(HERE/'prd32_integrated_pipeline_20261027.py')
        assert mod.parse_tag_output('```json\n{"category":"top","primary_color":"白色","pattern":"无","visible_design":"圆领"}\n```')['parseable']
    print('MASK_GUIDED_SELF_TEST_PASS')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    cmd=ap.add_subparsers(dest='cmd',required=True)
    au=cmd.add_parser('audit',help='No GPU. Verify 15 historical GT-pilot masks, prepare A/B previews')
    au.add_argument('--roi-run-dir',default=DEFAULT_ROI_RUN);au.add_argument('--project-root',default=DEFAULT_PROJECT)
    au.add_argument('--inventory',default=DEFAULT_INVENTORY);au.add_argument('--allowed-data-root',default=DEFAULT_ALLOWED_DATA)
    au.add_argument('--review',default=str(HERE/'roi15_human_review_CORRECTED.json'))
    au.add_argument('--helper',default=str(HERE/'prd32_integrated_pipeline_20261027.py'))
    au.add_argument('--pad',type=float,default=.05);au.add_argument('--output',required=True)
    ru=cmd.add_parser('run',help='Real GPU same-session paired baseline ROI vs mask guided')
    ru.add_argument('--preflight-dir',required=True);ru.add_argument('--helper',default=str(HERE/'prd32_integrated_pipeline_20261027.py'))
    ru.add_argument('--allowed-data-root',default=DEFAULT_ALLOWED_DATA);ru.add_argument('--project-root',default=DEFAULT_PROJECT)
    ru.add_argument('--model',default=DEFAULT_MODEL);ru.add_argument('--limit',type=int,default=0);ru.add_argument('--max-new-tokens',type=int,default=170)
    ru.add_argument('--output',required=True)
    cmd.add_parser('self-test')
    args=ap.parse_args()
    if args.cmd=='self-test':self_test()
    elif args.cmd=='audit':
        if not (0<=args.pad<=.2):raise ValueError('PADDING_OUT_OF_RANGE')
        audit(args)
    else:
        if args.limit<0 or args.limit>15:raise ValueError('INVALID_LIMIT')
        model_run(args)
if __name__=='__main__':main()
