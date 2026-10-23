"""Materialize unseen TRAIN candidates for visual screening, never train a model."""
import argparse, csv, hashlib, io, json, math, shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

HERE=Path(__file__).resolve().parent
TRAIN_SHA='e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f'
VAL_SHA='39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,data): Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def materialize(root,out,train_rows,val_rows,candidates,excluded):
    """No candidates may overlap previous review images or validation, even via renamed copies."""
    cache={}
    def source_sha(source):
        path=(root/source).resolve()
        if path not in cache:
            if not path.is_file():raise FileNotFoundError(f'Original image missing: {path}')
            cache[path]=sha(path)
        return cache[path]
    exclusion_hashes={source_sha(r['source_image']) for r in excluded}
    for r in excluded:
        if r.get('source_image_sha256') and source_sha(r['source_image'])!=r['source_image_sha256']:
            raise ValueError('Previously reviewed source changed: '+r['record_id'])
    val_hashes={source_sha(r['source_image']) for r in val_rows}
    seen=set();kept=[];rejected=[]
    for c in sorted(candidates,key=lambda r:(r['draft_sleeve_length'],r['record_id'])):
        rid=c['record_id'];r=train_rows.get(rid)
        if not r:raise ValueError('Candidate absent from frozen train manifest: '+rid)
        box=[float(r['bbox_'+k]) for k in ('x1','y1','x2','y2')]
        if box!=c['garment_bbox'] or r['source_image']!=c['source_image'] or r['garment_id']!=c['garment_id'] or r['garment_category']!=c['category']:
            raise ValueError('Candidate differs from frozen target: '+rid)
        if c['split']!='train':raise ValueError('Non-training candidate')
        digest=source_sha(c['source_image'])
        reason='validation_image_overlap' if digest in val_hashes else 'previous_review_image' if digest in exclusion_hashes else 'duplicate_candidate_image' if digest in seen else None
        if reason:rejected.append({'record_id':rid,'reason':reason});continue
        seen.add(digest)
        with Image.open(root/c['source_image']) as im:
            image=im.convert('RGB');ow,oh=image.size
        a,b,cc,dd=box
        if not all(math.isfinite(v) for v in box) or not (0<=a<cc<=ow and 0<=b<dd<=oh):
            raise ValueError('Invalid bbox: '+rid)
        image.thumbnail((1400,1400))
        w,h=image.size;sx,sy=w/ow,h/oh
        pen=ImageDraw.Draw(image)
        pen.rectangle([a*sx,b*sy,cc*sx,dd*sy],outline=(255,210,0),width=3)
        number=len(kept)+1
        path=out/'review_images'/f'{number:02d}_{rid}.jpg'
        image.save(path,quality=94)
        kept.append({**c,'experiment_split':'train','source_image_sha256':digest,
                     'native_draft_label_NOT_confirmed':c['draft_sleeve_length'],
                     'review_status':'pending','confirmed_training_label':False,
                     'original_width':ow,'original_height':oh,'review_width':w,'review_height':h,
                     'review_image':str(path.relative_to(out)),'review_image_sha256':sha(path),
                     'review_image_role':'yellow garment box for identification only, never model input'})
        print(f"prepared {number}: {rid} ({c['draft_sleeve_length']}, pending review)",flush=True)
    if not kept:raise ValueError('No independent training candidates remain')
    return kept,rejected,{'previous_review_unique_images':len(exclusion_hashes),'validation_unique_images':len(val_hashes)}

def contacts(out,kept):
    for start in range(0,len(kept),12):
        page=Image.new('RGB',(1200,1230),'white');draw=ImageDraw.Draw(page)
        for j,r in enumerate(kept[start:start+12]):
            x=(j%4)*300;y=(j//4)*410
            draw.text((x+5,y+5),f"{start+j+1:02d} {r['record_id']}",fill='black')
            draw.text((x+5,y+23),'native hint: '+r['draft_sleeve_length'],fill='black')
            with Image.open(out/r['review_image']) as im:
                tile=ImageOps.contain(im,(290,360));page.paste(tile,(x+5,y+45))
        page.save(out/f'contact_{start//12+1:02d}.jpg',quality=90)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,default=Path('/workspace/fashion_multimodal_analysis'));args=ap.parse_args()
    root=args.project_root.resolve()
    for file,digest in read(HERE/'input_hashes.json').items():
        if sha(HERE/file)!=digest:raise ValueError('Bundled input changed: '+file)
    trainfile=root/'configs/prd_8class_train_v5_balanced150_20261008.csv'
    valfile=root/'configs/prd_8class_val_v2_full_targets.csv'
    if sha(trainfile)!=TRAIN_SHA or sha(valfile)!=VAL_SHA:raise ValueError('Frozen train/validation CSV differs; stop and report before changing anything')
    train={r['record_id']:r for r in csv_rows(trainfile)}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=root/'reports/reruns/sleeve_supplement_1012'/stamp
    (out/'review_images').mkdir(parents=True,exist_ok=False)
    pool=read(HERE/'candidate_pool.json');excluded=read(HERE/'exclude_previous.json')
    kept,rejected,checks=materialize(root,out,train,csv_rows(valfile),pool,excluded)
    contacts(out,kept)
    manifest={'status':'PENDING_VISUAL_SCREENING','training_performed':False,'new_confirmed_labels':0,
              'selection':'All remaining native-hint cap/elbow candidates, after record and image-level isolation; no predictions used',
              'cases':kept,'rejected':rejected,'isolation_checks':checks,
              'counts_by_native_hint':dict(Counter(r['draft_sleeve_length'] for r in kept)),
              'fixed_old_split_sha256':sha(HERE/'parent_split_index.json'),
              'train_csv_sha256':sha(trainfile),'val_csv_sha256':sha(valfile)}
    dump(out/'candidate_manifest.json',manifest)
    for name in ['parent_split_index.json','exclude_previous.json','input_hashes.json','prepare_candidates.py','README.md']:
        shutil.copy2(HERE/name,out/name)
    archive=shutil.make_archive(str(out)+'_candidates','zip',root_dir=out)
    print('PREPARED_NATIVE_HINT_COUNTS:',manifest['counts_by_native_hint'],flush=True)
    print('NEW_CONFIRMED_LABELS: 0 (these are candidates only)',flush=True)
    print('UPLOAD_THIS_ZIP:',archive,flush=True)
    convenience=Path('/workspace/sleeve_candidates_1012.zip')
    if root.parent==Path('/workspace') and not convenience.exists():
        shutil.copy2(archive,convenience)
        print('EASY_DOWNLOAD:',convenience,flush=True)
    elif convenience.exists():
        print('Existing /workspace/sleeve_candidates_1012.zip preserved; download the timestamped UPLOAD_THIS_ZIP above.',flush=True)

if __name__=='__main__':main()
