#!/usr/bin/env python3
"""DEV60-only frozen V5 mask contour visual preflight, NO model inference.

Outputs paired historical Focus V1 red-box images and newly drawn original-image
V5-mask contours for manual identity review. It NEVER declares masks semantic GT,
never modifies upstream assets, and never calls V5/Qwen/Guard for inference.
"""
from __future__ import annotations

import argparse, csv, hashlib, html, json, math, os, re, shutil, sys, traceback, zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DEFAULT_RAW=Path('/workspace/PRD32_DEV60_V5_REAL_20261030_run01')
DEFAULT_IMAGES=Path('/workspace/fashion_data')
RAW_SHA='688cb801daebd0484e8740c877c021eca86c83f362857d4fc982d02fb50f2710'
CKPT_SHA='c5ba795225349b1c521aac0517ac7c493d9818dcc17d558ac4a3f6f8669452f0'
APPROVED_CLASSES={'top','pants','skirt','outerwear','dress','shoe','bag','accessory'}
RESTRICTED={'holdout','sealed','blind','independent_test','testset','fullval','validation'}
ID_PATTERN=re.compile(r'^fresh_dev_(\d{3})/pred_(\d{4})$')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):h.update(c)
    return h.hexdigest()


def ensure_approved(path):
    if any(any(token in part.lower() for token in RESTRICTED) for part in Path(path).parts):
        raise ValueError(f'Restricted/holdout path: {path}')


def load_keys(path):
    ensure_approved(path)
    with Path(path).open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    if len(rows)!=45:raise ValueError(f'Expected exactly 45 keys, got {len(rows)}')
    got=set()
    for r in rows:
        key=r['review_key'];m=ID_PATTERN.fullmatch(key)
        if m is None or not 1 <=int(m.group(1))<=60:raise ValueError('Non-DEV60 key: '+key)
        if key in got:raise ValueError('duplicate key '+key)
        got.add(key)
        if (r['case_id']+'/'+r['instance_id'])!=key:raise ValueError('key fields inconsistent')
        if r['bbox_quality_triage'] not in ('CLEAR','PARTIAL','MIXED'):raise ValueError('Unknown bbox quality')
        ref=ROOT/'reference_marked'/r['reference_marked_name']
        if ref.name!=key.replace('/','__')+'.jpg' or sha(ref)!=r['reference_marked_sha256']:
            raise ValueError('Historical marked image hash/name mismatch: '+key)
    return rows


def load_frozen(raw_dir):
    ensure_approved(raw_dir)
    raw=raw_dir/'v5_raw_predictions.jsonl'
    if sha(raw)!=RAW_SHA:raise ValueError('V5 frozen raw JSONL SHA-256 mismatch')
    summ=json.loads((raw_dir/'v5_real_summary.json').read_text())
    manifest=json.loads((raw_dir/'run_manifest.json').read_text())
    if (summ.get('n_images'),summ.get('n_predicted_instances'))!=(60,171):
        raise ValueError('DEV60 frozen count mismatch')
    if manifest.get('checkpoint_sha256')!=CKPT_SHA or summ.get('checkpoint_sha256')!=CKPT_SHA:
        raise ValueError('V5 checkpoint identity mismatch')
    if manifest.get('score_threshold')!=0.4 or manifest.get('mask_threshold')!=0.5 or manifest.get('human_gt_used_for_inference') is not False:
        raise ValueError('V5 frozen settings mismatch')
    records=[json.loads(x) for x in raw.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    if len(records)!=60:raise ValueError('Expected 60 image records')
    allpred={};count=0
    for k,rec in enumerate(records,1):
        case=f'fresh_dev_{k:03d}'
        if rec.get('case_id')!=case:raise ValueError('Frozen image ordering mismatch')
        for j,ins in enumerate(rec['instances']):
            iid=f'pred_{j:04d}';count+=1
            if ins.get('instance_id')!=iid or ins.get('predicted_category') not in APPROVED_CLASSES:
                raise ValueError('V5 instance metadata mismatch')
            allpred[case+'/'+iid]=(rec,ins)
    if count!=171:raise ValueError('Not 171 frozen V5 instances')
    return allpred


def resolve_source_image(source_path,image_root,expected_sha):
    ensure_approved(source_path);ensure_approved(image_root)
    base=Path(image_root).resolve(strict=True)
    src=Path(source_path).resolve(strict=True)
    if not src.is_relative_to(base) or 'train' not in [p.lower() for p in src.parts]:
        raise ValueError('V5 source image not under approved training directory')
    if src.suffix.lower() not in ('.jpg','.jpeg','.png','.webp'):
        raise ValueError('unapproved source image type')
    if sha(src)!=expected_sha:raise ValueError('Original source image bytes changed')
    return src


def bbox_iou(a,b):
    a=[float(x) for x in a];b=[float(x) for x in b]
    if any(not math.isfinite(x) for x in a+b):raise ValueError('nonfinite bbox')
    def area(z):return max(0,z[2]-z[0])*max(0,z[3]-z[1])
    inter=area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])])
    den=area(a)+area(b)-inter
    return inter/den if den else 0


def assess_mask(binary,bbox):
    import numpy as np
    a=np.asarray(binary,dtype=bool)
    if a.ndim!=2 or not a.size:raise ValueError('Bad mask shape')
    h,w=a.shape
    count=int(a.sum())
    if not count:raise ValueError('Empty mask')
    ys,xs=np.where(a)
    mb=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
    if len(bbox)!=4:raise ValueError('bbox not four values')
    b=[float(z) for z in bbox]
    if any(not math.isfinite(z) for z in b) or b[0]>=b[2] or b[1]>=b[3]:raise ValueError('bad frozen bbox')
    x0=max(0,math.floor(b[0]-2));y0=max(0,math.floor(b[1]-2))
    x1=min(w,math.ceil(b[2]+2));y1=min(h,math.ceil(b[3]+2))
    inside=int(a[y0:y1,x0:x1].sum()) if x0<x1 and y0<y1 else 0
    ratio=count/(h*w)
    outside=(count-inside)/count
    biou=bbox_iou(mb,b)
    flags=[]
    if ratio<0.0001:flags.append('MASK_TINY')
    if ratio>0.7:flags.append('MASK_HUGE')
    if outside>0.2:flags.append('MASK_OUTSIDE_FROZEN_BBOX_OVER20PCT')
    if biou<0.2:flags.append('MASK_AND_FROZEN_BBOX_WEAK_IOU')
    return {'mask_pixel_count':count,'image_pixel_fraction':round(ratio,6),
            'mask_bbox_xyxy':mb,'bbox_overlap_iou':round(biou,5),
            'mask_pixel_outside_bbox_fraction':round(outside,5),
            'geometry_flags':flags}


def draw_contour(source_rgb,binary):
    from PIL import Image, ImageFilter
    import numpy as np
    original=np.asarray(source_rgb.convert('RGB')).copy()
    if original.shape[:2]!=binary.shape:raise ValueError('Mask does not align with source RGB image')
    gray=Image.fromarray((binary.astype('uint8')*255),mode='L')
    outer=np.asarray(gray.filter(ImageFilter.MaxFilter(5)))>0
    inner=np.asarray(gray.filter(ImageFilter.MinFilter(5)))>0
    edge=outer & ~inner
    if not edge.any():raise ValueError('No contour pixels produced')
    original[edge]=[255,0,0] # red contour: keep same marking color family as historical bbox
    return Image.fromarray(original,'RGB')


def page_html(results):
    # HTML kept outside the data to avoid unescaped markup or accidental JS injection.
    cards=[]
    for i,r in enumerate(results,1):
        key=html.escape(r['review_key']);s=html.escape(r['status'])
        bb=html.escape(str(r.get('bbox_quality_triage','')))
        flags=', '.join(r.get('geometry_flags',[])) or 'no automatic geometry flags'
        if r['status']=='READY_FOR_MANUAL_VISUAL_CHECK':
            imgs=(f'<figure><img loading="lazy" src="reference_marked/{html.escape(r["filename"])}"><figcaption>历史 Focus V1 红框输入（原归档字节）</figcaption></figure>'
                  f'<figure><img loading="lazy" src="candidate_mask_contour/{html.escape(r["filename"])}"><figcaption>原图 + V5 Mask 红色轮廓（新候选）</figcaption></figure>')
        else:
            imgs=f'<p class="failed">本实例无法安全生成候选：{html.escape(r.get("failure_reason",""))}</p>'
        cards.append(f'<section class="item" data-key="{key}"><h3>{i:02d}. {key}</h3><p>红框关系初审：{bb} · 状态：{s} · 形状风险：{html.escape(flags)}</p>'
                     f'<div class="pair">{imgs}</div><label>人工对照结论：<select class="decision"><option value="UNREVIEWED">未核对</option><option value="MASK_TARGET_MATCH">轮廓对应目标</option><option value="MASK_TARGET_WRONG">Mask 错目标/混入其他目标</option><option value="UNCERTAIN">不确定，暂缓</option></select></label><input class="note" placeholder="备注（可选）"></section>')
    css='''body{font:15px/1.6 system-ui,sans-serif;color:#152235;margin:25px auto;max-width:1200px;padding:0 18px}header{position:sticky;top:0;background:#f8fafc;padding:12px;border:1px solid #d6dde5;z-index:1}button{border:0;background:#1f5c99;color:white;padding:10px 18px;border-radius:8px;cursor:pointer}.item{border:1px solid #d5dce4;border-radius:12px;padding:16px;margin:16px 0}.pair{display:flex;gap:14px;flex-wrap:wrap}.pair figure{flex:1 1 300px;margin:0}.pair img{width:100%;max-height:420px;object-fit:contain;background:#f1f4f7}.pair figcaption{text-align:center;color:#59677a;font-size:13px}h3{margin:3px 0}select,input{padding:6px;margin:6px}.failed{color:#be123c}footer{color:#566;border-top:1px solid #ddd;padding:20px}'''
    js='''function exportCSV(){const heads=['review_key','visual_mask_review_status','reviewer_note'];const rows=[heads];for(const el of document.querySelectorAll('.item')){const q=s=>String(s).replace(/"/g,'""');rows.push([el.dataset.key,el.querySelector('.decision').value,el.querySelector('.note').value].map(x=>'"'+q(x)+'"'));}const data='\\ufeff'+rows.map(r=>r.join(',')).join('\\r\\n');const blob=new Blob([data],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='FOCUS45_MASK_CONTOUR_VISUAL_REVIEW_PENDING.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}'''
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DEV60 Focus45 V5 Mask 轮廓核验</title><style>'+css+'</style><body><header><strong>PRD 3.2 · 45 条 Mask 轮廓对照预检</strong><div>注意：此处仅是视觉预检，不是新模型输出，尚未证明 Mask 语义目标正确；红框基线与候选可能存在图像编码差异，不能作为正式 A/B 精度证据。</div><button onclick="exportCSV()">导出我的审核 CSV</button></header>'+''.join(cards)+'<footer>45 条既有归因已由用户聊天整体确认；本轮 Mask 轮廓仍须单独核对。版本疑点继续保留；不读取 Holdout，不调用 Qwen。</footer><script>'+js+'</script></body></html>'


def run(args):
    import numpy as np
    from PIL import Image
    raw_dir=Path(args.raw_dir); image_root=Path(args.image_root)
    for p in (raw_dir,image_root,ROOT,args.output):ensure_approved(p)
    keys=load_keys(ROOT/'KEYS_ONLY_INPUT_45.csv')
    predicted=load_frozen(raw_dir)
    out=Path(args.output)
    if out.exists():raise FileExistsError('Refuse to overwrite: '+str(out))
    out.mkdir(parents=True)
    (out/'reference_marked').mkdir()
    (out/'candidate_mask_contour').mkdir()
    results=[]
    for r in keys:
        key=r['review_key'];case=r['case_id'];iid=r['instance_id']
        result={'review_key':key,'case_id':case,'instance_id':iid,'filename':r['reference_marked_name'],
                'bbox_quality_triage':r['bbox_quality_triage'], 'status':'ERROR_NEEDS_REVIEW',
                'source':'FROZEN_V5_DEV60_NOT_GT','qwen_called':False,'eligible_for_qwen_ab':False,
                'producer_mask_identity_status':'FROZEN_PATH_GEOMETRY_AUDITED_NOT_LIVE_REVERIFIED'}
        try:
            rec,ins=predicted[key]
            source=resolve_source_image(rec['source_image_path'],image_root,rec['image_sha256'])
            mask_path=raw_dir/f'{case}_{iid}_mask.png'
            ensure_approved(mask_path)
            with Image.open(source) as f: rgb=f.convert('RGB')
            if list(rgb.size)!=list(rec['image_size']):raise ValueError('Source image dimension mismatch')
            with Image.open(mask_path) as f:
                if f.format!='PNG':raise ValueError('Mask not PNG')
                if f.size!=rgb.size:raise ValueError('Mask vs image dimension mismatch')
                mask=np.asarray(f.convert('L'))>127
            geometry=assess_mask(mask,ins['bbox_xyxy'])
            candidate=draw_contour(rgb,mask)
            dest=out/'candidate_mask_contour'/r['reference_marked_name']
            candidate.save(dest,format='JPEG',quality=94,subsampling=0)
            ref=ROOT/'reference_marked'/r['reference_marked_name']
            shutil.copyfile(ref,out/'reference_marked'/ref.name)
            result.update(geometry)
            result.update({'status':'READY_FOR_MANUAL_VISUAL_CHECK',
                'candidate_sha256':sha(dest),'historical_focus_marked_sha256':sha(ref),
                'frozen_mask_png_sha256':sha(mask_path),'image_sha256':sha(source),
                'frozen_category':ins['predicted_category'],'frozen_bbox_xyxy':ins['bbox_xyxy'],
                'image_size':rec['image_size']})
            if r['bbox_quality_triage']!='CLEAR':
                result.setdefault('geometry_flags',[]).append('PRIOR_PARTIAL_OR_MIXED_TARGET')
        except Exception as exc:
            result['failure_reason']=f'{type(exc).__name__}: {exc}'
        results.append(result)
    (out/'visual_preview_status_45.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'visual_preview_status_45.csv').open('w',newline='',encoding='utf-8-sig') as f:
        cols=['review_key','status','bbox_quality_triage','frozen_category','geometry_flags','bbox_overlap_iou','mask_pixel_outside_bbox_fraction','source','candidate_sha256','frozen_mask_png_sha256','producer_mask_identity_status','failure_reason']
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader()
        for r in results:w.writerow({**r,'geometry_flags':';'.join(r.get('geometry_flags',[]))})
    (out/'REVIEW_FOCUS45_MASK_CONTOUR_PREVIEW.html').write_text(page_html(results),encoding='utf-8')
    status_count=Counter(r['status'] for r in results)
    report={'schema':'prd32_dev60_focus45_mask_contour_visual_preflight_v1',
      'status':'VISUAL_PREVIEW_ONLY_NO_QWEN_NO_GUARD_NOT_VALIDATED',
      'date':'2026-11-10','n_selected':len(keys), 'counts':dict(status_count),
      'geometry_risk_count':sum(bool(r.get('geometry_flags')) for r in results),
      'qwen_calls':0,'holdout_accessed':False,'release_allowed':False,
      'notes':['Source image + frozen mask geometry checked where available; frozen file name is not independent live producer authentication',
               'Original historical marked input and new contour candidate may have different JPEG encoding. This is NOT formal A/B model validation',
               'Mask may segment wrong object; mandatory separate human visual check before any Qwen trial',
               'Cause labels user chat batch-confirmed; no independent blind validation'],
      'frozen_v5_raw_sha256':RAW_SHA,'frozen_v5_checkpoint_sha256':CKPT_SHA}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    z=out.with_suffix('.zip')
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
        for p in sorted(out.rglob('*')):
            if p.is_file(): f.write(p,p.relative_to(out.parent).as_posix())
    print('FOCUS45_MASK_CONTOUR_VISUAL_PREFLIGHT_DONE',json.dumps(status_count,ensure_ascii=False),flush=True)
    print('NO_NEW_QWEN_REQUESTS; HUMAN_REVIEW_REQUIRED; HOLDOUT_NOT_ACCESSED',flush=True)
    print('UPLOAD_THIS:',z,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw-dir',default=str(DEFAULT_RAW))
    parser.add_argument('--image-root',default=str(DEFAULT_IMAGES))
    parser.add_argument('--output',default=str(Path('/workspace')/('PRD32_1110_FOCUS45_MASK_CONTOUR_PREVIEW_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))))
    run(parser.parse_args())
