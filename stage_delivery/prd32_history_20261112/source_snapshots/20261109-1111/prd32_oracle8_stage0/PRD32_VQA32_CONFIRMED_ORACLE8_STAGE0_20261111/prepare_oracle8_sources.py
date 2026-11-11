#!/usr/bin/env python3
"""DEV60-only, source SHA-verified Oracle8 annotation preparation. No model calls."""
from __future__ import annotations
import argparse,base64,csv,hashlib,html,json,math,shutil,sys,zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
RAW_DIR=Path('/workspace/PRD32_DEV60_V5_REAL_20261030_run01')
IMAGE_ROOT=Path('/workspace/fashion_data')
OUT=Path('/workspace/PRD32_ORACLE8_SOURCE_EXPORT_20261111')
UPLOAD=Path('/workspace/PRD32_ORACLE8_SOURCE_IMAGES_UPLOAD_THIS.zip')
TARGETS=HERE/'ORACLE8_TARGET_SELECTION.json'
MANIFEST=HERE/'full_dev60_171_manifest_v1.json'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load_plan():
    d=json.loads(TARGETS.read_text(encoding='utf8'))
    m=json.loads(MANIFEST.read_text(encoding='utf8'))
    chosen=d.get('targets',[])
    if len(chosen)!=8 or len({(t['case_id'],t['instance_id']) for t in chosen})!=8:
        raise RuntimeError('expected 8 distinct fixed DEV60 targets')
    if d.get('holdout_used') is not False or d.get('production_release') is not False:
        raise RuntimeError('unsafe fixed plan')
    if m.get('target_count')!=171 or m.get('source_image_count')!=60:
        raise RuntimeError('bad frozen full manifest')
    ref={(r['case_id'],r['instance_id']):r for r in m['targets']}
    for t in chosen:
        if (t['case_id'],t['instance_id']) not in ref:
            raise RuntimeError('target absent from DEV60 frozen manifest')
        if t['oracle_approved_for_comparison']!='NO' or t['oracle_target_status']!='NOT_ANNOTATED':
            raise RuntimeError('oracle must begin unannotated')
    return chosen,ref


def preflight():
    chosen,ref=load_plan()
    root=IMAGE_ROOT.resolve(strict=True)
    if not (RAW_DIR/'v5_raw_predictions.jsonl').is_file():
        raise FileNotFoundError('frozen DEV60 V5 raw predictions unavailable')
    raw=[json.loads(x) for x in (RAW_DIR/'v5_raw_predictions.jsonl').read_text(encoding='utf8').splitlines() if x.strip()]
    if len(raw)!=60:
        raise RuntimeError('frozen DEV60 must have 60 source records')
    rec={x['case_id']:x for x in raw}
    data=[]
    from PIL import Image
    for t in chosen:
        case,inst=t['case_id'],t['instance_id'];r=rec[case]
        actual=Path(r['source_image_path']).resolve(strict=True)
        # Source must stay in the approved training-image root, never holdout.
        if not actual.is_relative_to(root) or 'train' not in [v.lower() for v in actual.parts]:
            raise RuntimeError('source image outside approved training directory: '+case)
        if actual.suffix.lower() not in ('.png','.jpeg','.jpg','.webp'):
            raise RuntimeError('unknown source image type')
        exp=ref[case,inst]['source_image_sha256']
        if sha(actual)!=exp or r['image_sha256']!=exp:
            raise RuntimeError('source image SHA256 mismatch: '+case)
        idx=int(inst.rsplit('_',1)[1]);ins=r['instances'][idx]
        if ins['instance_id']!=inst or ins['predicted_category']!=t['v5_frozen_category']:
            raise RuntimeError('frozen V5 instance identity/category drift: '+case+'/'+inst)
        with Image.open(actual) as im:w,h=im.size
        if (w,h)!=tuple(r['image_size']):
            raise RuntimeError('image dimensions drift')
        b=[float(x) for x in ins['bbox_xyxy']]
        if len(b)!=4 or not all(map(math.isfinite,b)) or not (0<=b[0]<b[2]<=w+1 and 0<=b[1]<b[3]<=h+1):
            raise RuntimeError('invalid bbox '+case+'/'+inst)
        mask=RAW_DIR/f'{case}_{inst}_mask.png'
        if not mask.is_file():raise FileNotFoundError('frozen V5 mask missing: '+str(mask))
        with Image.open(mask) as im:
            if im.size!=(w,h):raise RuntimeError('mask dimensions drift')
        data.append({**t,'image_path':str(actual),'source_image_sha256':exp,
                     'source_image_width':w,'source_image_height':h,'v5_bbox_xyxy':b,
                     'frozen_mask_path':str(mask),'frozen_mask_sha256':sha(mask)})
    return data


def markup(records):
    from PIL import Image,ImageDraw
    items=[]
    for r in records:
        case,inst=r['case_id'],r['instance_id']
        dst=OUT/'original_images'/f'{case}_{inst}{Path(r["image_path"]).suffix.lower()}'
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(r['image_path'],dst)
        if sha(dst)!=r['source_image_sha256']:raise RuntimeError('copy hash mismatch')
        r['exported_original_rel']=str(dst.relative_to(OUT))
        with Image.open(dst) as im:
            source=im.convert('RGB')
            # Keep a separate reference overlay, never overwrite the original image.
            mask=Image.open(r['frozen_mask_path']).convert('L')
            import PIL.ImageChops
            tint=Image.new('RGB',source.size,(0,120,220))
            overlay=Image.blend(source,tint,0.45)
            binary=mask.point(lambda x:255 if x>127 else 0)
            source.paste(overlay,(0,0),binary)
            d=ImageDraw.Draw(source)
            d.rectangle(r['v5_bbox_xyxy'],outline=(255,50,60),width=max(2,round(source.width/300)))
            prev=OUT/'mask_overlays'/f'{case}_{inst}_MASK_NOT_GT.jpg'
            prev.parent.mkdir(exist_ok=True,parents=True)
            source.save(prev,quality=86)
        r['mask_overlay_rel']=str(prev.relative_to(OUT))
        imgbytes=dst.read_bytes()
        mime='image/png' if dst.suffix.lower()=='.png' else ('image/webp' if dst.suffix.lower()=='.webp' else 'image/jpeg')
        items.append({k:r[k] for k in ('case_id','instance_id','question_id','cohort','observed_qa_issue','diagnostic_purpose','assisted_dev_reference_category_not_blind_gt','v5_frozen_category','source_image_sha256','source_image_width','source_image_height','v5_bbox_xyxy')}|{'src':'data:'+mime+';base64,'+base64.b64encode(imgbytes).decode('ascii')})
    (OUT/'ORACLE8_VERIFIED_SOURCE_MANIFEST.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
    jsdata=json.dumps(items,ensure_ascii=False).replace('</','<\\/')
    s=r'''<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><title>DEV60 Oracle8 目标框人工复核</title>
<style>body{font-family:system-ui,'Microsoft Yahei',sans-serif;margin:24px auto;max-width:1000px;background:#f6f8fa;color:#152134;padding:0 12px}article{background:white;border:1px solid #dce1e5;border-radius:14px;padding:16px;margin:16px 0}h1{font-size:24px}h2{font-size:18px}.muted{color:#596575;font-size:13px}.v{display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap}canvas{max-width:100%;height:auto;touch-action:none;cursor:crosshair;border:1px solid #aeb8c4;border-radius:6px}.ctrl{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}button,select{font:inherit;padding:7px 12px;border-radius:8px;border:1px solid #8798a9;background:#f9fafb;cursor:pointer}.danger{color:#a12b32}textarea{width:100%;min-height:60px;box-sizing:border-box;padding:8px;margin-top:8px}.legend{font-size:13px;margin-top:8px}</style></head>
<body><h1>Oracle8｜人工目标框复核</h1><p>红框是 <b>V5 冻结预测框</b>，绿框是你明确确认或绘制的框。<b>不要把当前预测框自动当成人工 Oracle</b>。可直接在原图上拖动绘制绿色框；如果目标不存在，选择“伪实例”。完成后导出 JSON 并上传。未审核记录可保留空白。</p><p class="muted">本文件仅用于 DEV60 开发实验。图片来自源图 SHA256 校验；不包含 Holdout，不进行 GPU 推理，不允许自动发布。</p><div id="items"></div><button id="export">导出本轮 Oracle8 审核 JSON</button><script>const data=__DATA__;
const states=data.map(()=>({status:'not_reviewed',oracle_bbox_xyxy:null,notes:''}));
function draw(c,i,im,temp=null){let ctx=c.getContext('2d'),w=c.width,h=c.height;ctx.clearRect(0,0,w,h);ctx.drawImage(im,0,0,w,h);let b=data[i].v5_bbox_xyxy;ctx.lineWidth=Math.max(2,w/250);ctx.strokeStyle='#ff3046';ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);let o=temp||states[i].oracle_bbox_xyxy;if(o){ctx.strokeStyle='#10a858';ctx.strokeRect(o[0],o[1],o[2]-o[0],o[3]-o[1])}}
function coords(c,e){let b=c.getBoundingClientRect();return [Math.round(Math.max(0,Math.min(c.width,(e.clientX-b.left)/b.width*c.width))),Math.round(Math.max(0,Math.min(c.height,(e.clientY-b.top)/b.height*c.height)))]}
data.forEach((r,i)=>{const art=document.createElement('article');art.innerHTML=`<h2>${i+1}. ${r.case_id}/${r.instance_id} · ${r.cohort==='critical'?'高风险':'对照'}</h2><div class="muted">参考类别：${r.assisted_dev_reference_category_not_blind_gt}（仅开发集辅助审核，非盲测 GT）｜任务 ${r.question_id}</div><p>${r.observed_qa_issue}。<span class="muted">${r.diagnostic_purpose}</span></p>`;let c=document.createElement('canvas');c.width=r.source_image_width;c.height=r.source_image_height;let im=new Image();im.onload=()=>draw(c,i,im);im.src=r.src;art.appendChild(c);const info=document.createElement('div');info.className='legend';info.textContent='红：V5预测框；绿：人工明确确认或新画框；请不要在目标混合/遮挡时猜测。';art.appendChild(info);
const controls=document.createElement('div');controls.className='ctrl';const b=document.createElement('button');b.textContent='人工确认：V5 框就是正确目标';b.onclick=()=>{states[i].status='confirmed';states[i].oracle_bbox_xyxy=[...r.v5_bbox_xyxy];s.value='confirmed';draw(c,i,im)};controls.appendChild(b);
const s=document.createElement('select');[['not_reviewed','尚未审核'],['confirmed','确认正确目标（需绿框）'],['ambiguous','无法确定目标'],['invalid','不是独立服饰目标/伪实例']].forEach(([v,t])=>{let op=document.createElement('option');op.value=v;op.textContent=t;s.appendChild(op)});s.onchange=()=>{states[i].status=s.value;if(s.value!=='confirmed')states[i].oracle_bbox_xyxy=null;draw(c,i,im)};controls.appendChild(s);art.appendChild(controls);
let start=null;c.onpointerdown=e=>{start=coords(c,e);c.setPointerCapture(e.pointerId)};c.onpointermove=e=>{if(!start)return;let p=coords(c,e);draw(c,i,im,[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.max(start[0],p[0]),Math.max(start[1],p[1])])};c.onpointerup=e=>{if(!start)return;let p=coords(c,e),rect=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.max(start[0],p[0]),Math.max(start[1],p[1])];start=null;if(rect[2]-rect[0]<3||rect[3]-rect[1]<3)return;states[i].status='confirmed';states[i].oracle_bbox_xyxy=rect;s.value='confirmed';draw(c,i,im)};
let note=document.createElement('textarea');note.placeholder='可选：写下目标区域、被遮挡部位或为何无法确认';note.oninput=()=>states[i].notes=note.value;art.appendChild(note);document.querySelector('#items').appendChild(art)});
document.querySelector('#export').onclick=()=>{let result={schema:'DEV60_ORACLE8_MANUAL_BBOX_ANNOTATION_V1',origin:'manual_editor_on_SHA_verified_training_image',not_independent_blind_gt:true,holdout_used:false,production_release:false,annotations:data.map((r,i)=>({case_id:r.case_id,instance_id:r.instance_id,question_id:r.question_id,source_image_sha256:r.source_image_sha256,source_image_size:[r.source_image_width,r.source_image_height],v5_bbox_xyxy:r.v5_bbox_xyxy,oracle_target_status:states[i].status,oracle_bbox_xyxy:states[i].oracle_bbox_xyxy,notes:states[i].notes}))};let blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='DEV60_ORACLE8_HUMAN_BBOX_REVIEW.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1500)};
</script></body></html>'''
    (OUT/'ORACLE8_ANNOTATION.html').write_text(s.replace('__DATA__',jsdata),encoding='utf8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=['package-check','export-source-images']);args=ap.parse_args()
    chosen,ref=load_plan()
    assert all(x['oracle_target_status']=='NOT_ANNOTATED' for x in chosen)
    if args.mode=='package-check':
        print('ORACLE8_PACKAGE_CHECK_PASS: 8 fixed targets; no Oracle GT, no GPU, no holdout');return
    data=preflight();OUT.mkdir(parents=True,exist_ok=True);markup(data)
    (OUT/'README_EXPORT.md').write_text('## 操作\n打开 ORACLE8_ANNOTATION.html，逐条绘制或确认目标框。导出的 JSON 是人工开发集标注，不是独立盲测 GT。确认后上传给助手，再决定是否执行 Qwen Oracle A/B。\n',encoding='utf8')
    with zipfile.ZipFile(UPLOAD,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(OUT))
    with zipfile.ZipFile(UPLOAD) as z:
        bad=z.testzip();assert bad is None
        assert len([x for x in z.namelist() if x.startswith('original_images/')])==8
    print('ORACLE8_SOURCE_EXPORT_PASS; SHA_verified=8; correct_bbox_annotations=0; no_GPU; no_holdout')
    print('UPLOAD_THIS:',UPLOAD)

if __name__=='__main__':main()
