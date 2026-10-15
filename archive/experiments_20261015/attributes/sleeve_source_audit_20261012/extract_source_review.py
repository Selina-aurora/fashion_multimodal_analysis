#!/usr/bin/env python3
# coding: utf-8
"""Read-only source-image audit for 10/12 sleeve candidate review.

No training, no model weights, no writes to the repository or frozen data.
Generated output is a diagnostic package, NOT approved training labels.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, os, shutil, sys, zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE=Path(__file__).resolve().parent
EXPECTED_TRAIN='e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f'
EXPECTED_VAL='39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93'
IMPORTANT=['v5_train_01719','v5_train_00286','v5_train_00681','v5_train_01748']
NAME_MAP={'shoulder':'S','elbow':'E','wrist':'W','cuff':'C'}
COLORS={'shoulder':(240,67,72),'elbow':(28,167,122),'wrist':(61,105,230),'cuff':(222,72,188)}

def read(name):return json.loads((HERE/name).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def csvrows(path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_json(path,body):path.write_text(json.dumps(body,ensure_ascii=False,indent=2),encoding='utf-8')
def src_path(root,relative):return (root/relative).resolve()

def find_font(size):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']:
        if Path(p).exists():return ImageFont.truetype(p,size)
    return ImageFont.load_default()

def marked(im,record,garment):
    canvas=im.copy(); pen=ImageDraw.Draw(canvas)
    w,h=canvas.size
    # A thin outline of the target garment appears only in OVERLAY image, never original.
    bounds=garment['garment_bbox']; rw=w/garment['original_width']; rh=h/garment['original_height']
    xy=tuple(int(x*rw) if i%2==0 else int(x*rh) for i,x in enumerate(bounds))
    pen.rectangle(xy,outline=(255,201,32),width=max(2,round(w/400)))
    pts=record.get('points') or {}
    if all(k in pts for k in ('shoulder','elbow','wrist')):
        pen.line([(round(pts[k][0]*w),round(pts[k][1]*h)) for k in ('shoulder','elbow','wrist')],fill=(16,180,175),width=max(2,round(w/280)))
    radius=max(6,round(w/105));font=find_font(max(13,round(w/45)))
    for k,(nx,ny) in pts.items():
        if k not in COLORS:continue
        x,y=round(nx*w),round(ny*h); color=COLORS[k]
        pen.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color,outline=(255,255,255),width=max(2,radius//4))
        pen.text((x+radius+3,y-radius-3),NAME_MAP[k],font=font,fill=color,stroke_width=2,stroke_fill=(255,255,255))
    return canvas

def make_contact(pairs,out_path,prefix):
    if not pairs:return
    fw,fh=330,410; cols=4;rows=math.ceil(len(pairs)/cols)
    canvas=Image.new('RGB',(fw*cols,fh*rows),'white'); pen=ImageDraw.Draw(canvas)
    font=find_font(16)
    for i,(image_path,record) in enumerate(pairs):
        x=(i%cols)*fw;y=(i//cols)*fh
        text=f'{i+1:02d} {record["record_id"]} | {record["state"]} | {record.get("current_class", "")} '
        pen.text((x+6,y+7),text,fill=(25,40,55),font=font)
        with Image.open(image_path) as im:
            tile=ImageOps.contain(im.convert('RGB'),(fw-12,fh-42))
            canvas.paste(tile,(x+(fw-tile.width)//2,y+32))
    canvas.save(out_path,quality=92)


def offline_check():
    m=read('candidate_manifest.json');a=read('ORIGINAL_user_review_20261012.json');q=read('source_qa_previous.json');s=read('parent_split_index.json')
    by_id={c['record_id']:c for c in m['cases']}
    assert len(m['cases'])==len(by_id)==24
    assert len(a['annotations'])==24
    assert len(s['records'])==44
    assert Counter(i['experiment_split'] for i in s['records'])=={'train':32,'dev':12}
    assert hashlib.sha256((HERE/'ORIGINAL_user_review_20261012.json').read_bytes()).hexdigest()==q['source_review_sha256']
    assert set(by_id)=={r['record_id'] for r in a['annotations']}
    for r in a['annotations']:
        c=by_id[r['record_id']]
        assert r['source_image_sha256']==c['source_image_sha256']
        assert r['garment_id']==c['garment_id']
        assert r['source_image']==c['source_image']
    qa_by_id={e['annotation']['record_id']:e['qa'] for e in q['annotations']}
    assert len(qa_by_id)==24
    assert Counter(e['qa_status'] for e in qa_by_id.values())=={'PRELIM_PASS_SOURCE_PENDING':7,'RECHECK_REQUIRED':9,'FINALIZED_NONTRAINABLE':2,'EXCLUDED_FROM_SUPPLEMENT':6}
    print('PACKAGE_CHECK_OK 24 cases / 7 prelim / 9 recheck / 32 train + 12 dev / original JSON sha matched')
    return m,a,q,s,qa_by_id


def main():
    ap=argparse.ArgumentParser(description='Read-only source audit, no training')
    ap.add_argument('--project-root',default='/workspace/fashion_multimodal_analysis',type=Path)
    ap.add_argument('--out-dir',type=Path,default=Path('/workspace'))
    ap.add_argument('--check-package',action='store_true')
    opt=ap.parse_args()
    m,review,oldqa,oldsplit,qa_by_id=offline_check()
    if opt.check_package:return 0
    root=opt.project_root.resolve();dest=opt.out_dir.resolve()
    cases={c['record_id']:c for c in m['cases']}
    annotations={a['record_id']:a for a in review['annotations']}
    selection=[a for a in review['annotations'] if qa_by_id[a['record_id']]['qa_status'] in ('PRELIM_PASS_SOURCE_PENDING','RECHECK_REQUIRED')]
    selection.sort(key=lambda a: (0 if a['record_id'] in IMPORTANT else 1,0 if qa_by_id[a['record_id']]['qa_status']=='RECHECK_REQUIRED' else 1,a['record_id']))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=dest/f'sleeve_source_audit_20261012_{stamp}'
    out.mkdir(parents=True,exist_ok=False)
    (out/'original_no_boxes').mkdir();(out/'point_overlays').mkdir()
    errors=[];passes=[];issues=[];checks={};shacache={}
    train_path=root/'configs/prd_8class_train_v5_balanced150_20261008.csv'
    val_path=root/'configs/prd_8class_val_v2_full_targets.csv'
    for name,path,expected in [('train_csv',train_path,EXPECTED_TRAIN),('val_csv',val_path,EXPECTED_VAL)]:
        try:
            digest=sha(path);checks[name]={'file':str(path),'sha256':digest,'matches_frozen':digest==expected}
            if digest!=expected:errors.append(f'{name}: frozen CSV hash mismatch; abort training until resolved')
        except Exception as e:errors.append(f'{name}: {type(e).__name__}: {e}')
    if errors:
        status={'status':'BLOCKED_FROZEN_SOURCE','ready_for_training':False,'errors':errors,'checks':checks}
        write_json(out/'audit_report.json',status)
        print('BLOCKED: frozen train/validation CSV unavailable or changed; did not read any source images')
        for e in errors:print(e)
        print('REPORT_DIR:',out)
        return 2
    trainrows={r['record_id']:r for r in csvrows(train_path)}
    valrows=csvrows(val_path)
    def getsha(path):
        if path not in shacache:shacache[path]=sha(path)
        return shacache[path]
    def hashed_set(rows,purpose):
        values=set()
        for r in rows:
            if not r.get('source_image'):
                errors.append(purpose+' missing source_image');continue
            p=src_path(root,r['source_image'])
            try:values.add(getsha(p))
            except Exception as e:errors.append(purpose+' '+str(p)+' : '+str(e))
        return values
    validation_hashes=hashed_set(valrows,'validation source')
    prior=read('exclude_previous.json')
    old_hashes=hashed_set(prior,'previously reviewed source')
    devhashes={r['source_image_sha256'] for r in oldsplit['records'] if r['experiment_split']=='dev'}
    checks['validation_images_hashed']=len(validation_hashes)
    checks['previously_reviewed_images_hashed']=len(old_hashes)
    checks['dev_split_images_known']=len(devhashes)
    candidate_shas=[]
    for c in m['cases']:
        rid=c['record_id'];path=src_path(root,c['source_image']);r=trainrows.get(rid)
        if not r:errors.append(f'{rid} missing from train CSV');continue
        try:
            rowbox=[float(r['bbox_'+k]) for k in ('x1','y1','x2','y2')]
            if rowbox != [float(v) for v in c['garment_bbox']]: errors.append(rid+' target bbox differs from frozen train')
            for k,c_key in [('source_image','source_image'),('garment_id','garment_id'),('garment_category','category')]:
                if r.get(k)!=c[c_key]:errors.append(rid+' '+k+' differs from frozen train')
            dg=getsha(path); candidate_shas.append(dg)
            if dg != c['source_image_sha256']:errors.append(rid+' original image SHA mismatch')
            if dg in validation_hashes:errors.append(rid+' source overlaps val images')
            if dg in devhashes:errors.append(rid+' source overlaps old dev split')
            if dg in old_hashes:errors.append(rid+' overlaps previous review images')
        except Exception as e:errors.append(rid+' '+str(e))
    if len(candidate_shas)!=len(set(candidate_shas)):errors.append('same original image present in multiple candidates')
    # Preview only after integrity checks. Wrong hashes are not silently accepted.
    for idx,a in enumerate(selection,1):
        rid=a['record_id'];c=cases[rid];path=src_path(root,c['source_image'])
        if any(e.startswith(rid) for e in errors):
            issues.append({'record_id':rid,'error':'source source-mismatch; preview withheld'});continue
        try:
            with Image.open(path) as i:
                original=i.convert('RGB');width,height=original.size
            if (width,height)!=(c['original_width'],c['original_height']):
                errors.append(f'{rid} original size mismatch: {width}x{height}');continue
            copy_path=out/'original_no_boxes'/f'{idx:02d}_{rid}{path.suffix.lower()}'
            shutil.copyfile(path,copy_path)
            im=original.copy();im.thumbnail((1400,1400));overlay=marked(im,a,c)
            overlay_path=out/'point_overlays'/f'{idx:02d}_{rid}.jpg';overlay.save(overlay_path,quality=93)
            passes.append({'record_id':rid,'source_sha256':c['source_image_sha256'],'original_path':str(copy_path.relative_to(out)),'overlay_path':str(overlay_path.relative_to(out)),'original_dimensions':[width,height],'qa_status':qa_by_id[rid]['qa_status'],'original_ratio':a.get('ratio'),'original_state':a['state'],'original_class':a.get('current_class'),'note':qa_by_id[rid]['qa_reason_zh']})
            print(f'IMAGE_OK {idx:02d}/16 {rid} {qa_by_id[rid]["qa_status"]}',flush=True)
        except Exception as e: errors.append(f'{rid} image export: {str(e)}')
    pairs_orig=[];pairs_overlay=[]
    for a in passes:
        record=annotations[a['record_id']]
        pairs_orig.append((out/a['original_path'],record))
        pairs_overlay.append((out/a['overlay_path'],record))
    make_contact(pairs_orig,out/'contact_original_no_boxes.jpg','original')
    make_contact(pairs_overlay,out/'contact_point_overlays.jpg','overlay')
    full_ok=not errors and len(passes)==16
    report={'status':'SOURCE_AUDIT_PASSED_VISUAL_REVIEW_STILL_NEEDED' if full_ok else 'SOURCE_AUDIT_INCOMPLETE',
            'ready_for_training':False,'raw_training_labels_added':0,'reviewed_images':len(passes),'expected_images':16,
            'checks':checks,'errors':errors,'images':passes,
            'limitations':['These are reference exports, not training-ready labels','Manual visual check of points and garment semantics required','Even a verified source cannot automatically resolve 9 QA ambiguities','Keep frozen dev/validation data unchanged']}
    write_json(out/'audit_report.json',report)
    (out/'README_请先看.txt').write_text('10.12 sleeve source QA, no training. Original images contain NO yellow boxes. Point overlays contain only QA guides.\nAttach this ZIP in ChatGPT for final semantic and source-level check. Do not copy any records to model training yet.\n',encoding='utf-8')
    # Keep final ZIP under 38MiB, omit ORIGINAL backups if needed. Previews always remain.
    def buildzip():
        p=out.with_suffix('.zip');files=[f for f in out.rglob('*') if f.is_file()]
        with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for f in files:z.write(f,arcname=str(f.relative_to(out)))
        return p
    archive=buildzip()
    if archive.stat().st_size>38*1024*1024:
        for f in (out/'original_no_boxes').glob('*'):f.unlink()
        errors.append('Original full-size image files omitted to keep ZIP within 38 MiB; only overlays left')
        # Don't falsely state original included; regenerate unboxed 1400px thumbnails.
        for idx,a in enumerate(passes,1):
            src=src_path(root,cases[a['record_id']]['source_image']);im=Image.open(src).convert('RGB');im.thumbnail((1400,1400));p=out/'original_no_boxes'/f'{idx:02d}_{a["record_id"]}.jpg';im.save(p,quality=90)
            a['original_path']=str(p.relative_to(out))
        report['errors']=errors;report['status']='SOURCE_AUDIT_PREVIEW_ONLY';report['images']=passes
        write_json(out/'audit_report.json',report);archive.unlink();archive=buildzip()
    if archive.stat().st_size>38*1024*1024:
        print('ERROR: ZIP exceeds 38MiB. Files available in directory:',out)
        return 3
    print('AUDIT_STATUS:',report['status']);print('READY_FOR_TRAINING: FALSE')
    print('UPLOAD_THIS_ZIP:',archive);print('ZIP_BYTES:',archive.stat().st_size)
    return 0 if full_ok else 2

if __name__=='__main__':sys.exit(main())
