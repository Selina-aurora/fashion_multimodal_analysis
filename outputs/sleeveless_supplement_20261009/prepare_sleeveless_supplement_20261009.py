"""Export eight TRAIN-only sleeveless-category candidates for human confirmation.
The category hint is a draft; no classifier training starts before confirmation.
"""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import time
import zipfile
from collections import Counter

TRAIN_SHA='e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f'
VAL_SHA='39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93'
REVIEW_SHA='767b79a4fdb1e1c91ede245c6f59f240cb0a23baddb1aaa33ca0014fda82ada4'
SLEEVELESS_CATEGORIES={'vest','sling','vest_dress','sling_dress'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    from PIL import Image,ImageDraw
    from fashion_multimodal_analysis.common.paths import project_root,resolve_path
    root=project_root();here=Path(__file__).resolve().parent
    reviewed=here/'sleeve_training_review_confirmed_20261009.json'
    if sha(reviewed)!=REVIEW_SHA:raise ValueError('Confirmed 40-target review changed')
    old=json.loads(reviewed.read_text())
    if len(old['cases'])!=40 or len({c['record_id'] for c in old['cases']})!=40 or not all(c['review_status']=='confirmed' for c in old['cases']):raise ValueError('Expected 40 confirmed targets')
    usable=[c for c in old['cases'] if c['sleeve_length']!='unknown' and c['confirmed_training_label']]
    if len(usable)!=37:raise ValueError('Expected 37 usable reviewed targets')
    train=resolve_path('configs/prd_8class_train_v5_balanced150_20261008.csv');val=resolve_path('configs/prd_8class_val_v2_full_targets.csv')
    if sha(train)!=TRAIN_SHA or sha(val)!=VAL_SHA:raise ValueError('Frozen data manifests changed')
    def read(path):
        with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
    rows=read(train);validation=read(val);by_id={r['record_id']:r for r in rows}
    used_hashes={c['source_image_sha256'] for c in old['cases']}
    used_hashes.update(sha(resolve_path(r['source_image'])) for r in validation)
    for c in old['cases']:
        row=by_id[c['record_id']];box=[float(row['bbox_'+k]) for k in ('x1','y1','x2','y2')]
        if c['split']!='train' or c['garment_bbox']!=box or c['source_image_sha256']!=sha(resolve_path(row['source_image'])):raise ValueError('Reviewed training target changed')
    pool=sorted([r for r in rows if r['source_dataset']=='DeepFashion2' and r.get('fine_category') in SLEEVELESS_CATEGORIES and r['garment_category'] in {'top','dress'}],key=lambda r:r['record_id'])
    random.Random(20261009).shuffle(pool)
    selected=[]
    def take(candidates,limit):
        added=0
        for r in candidates:
            digest=sha(resolve_path(r['source_image']))
            if digest in used_hashes:continue
            selected.append((r,digest));used_hashes.add(digest);added+=1
            if added==limit:break
    for category in ('top','dress'):take([r for r in pool if r['garment_category']==category],4)
    if len(selected)<8:take(pool,8-len(selected))
    if len(selected)!=8:raise ValueError('Not enough distinct training images for eight candidates')
    out=root/'reports/reruns/recovery_20261009'/('sleeveless_supplement_'+time.strftime('%Y%m%d_%H%M%S')+f'_{time.time_ns()%1000000:06d}')
    out.mkdir(parents=True,exist_ok=False);(out/'review_images').mkdir()
    cases=[]
    for number,(r,digest) in enumerate(selected,1):
        original=Image.open(resolve_path(r['source_image'])).convert('RGB');box=[float(r['bbox_'+k]) for k in ('x1','y1','x2','y2')]
        image=original.copy();ImageDraw.Draw(image).rectangle(box,outline='yellow',width=max(2,image.width//220));image.thumbnail((950,1000))
        image.save(out/'review_images'/f'{number:02d}_{r["record_id"]}.jpg',quality=92)
        crop=original.crop(tuple(int(v) for v in box));crop.thumbnail((650,700))
        def uri(im):
            b=io.BytesIO();im.save(b,format='JPEG',quality=88)
            return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()
        cases.append({'record_id':r['record_id'],'split':'train','source_dataset':'DeepFashion2','category':r['garment_category'],
                      'fine_category':r['fine_category'],'source_image':r['source_image'],'source_image_sha256':digest,
                      'garment_id':r['garment_id'],'garment_bbox':box,'draft_sleeve_length':'sleeveless',
                      'review_status':'pending','confirmed_training_label':False,
                      'image':uri(image),'crop':uri(crop)})
    payload={'purpose':'TRAIN-only sleeveless category supplement; weak draft requires visual confirmation',
             'labels':['sleeveless','cap','short','elbow','three_quarter','long'],'association_thresholds':None,'cases':cases}
    template=(here/'review_template.html').read_text()
    (out/'sleeveless_review_8.html').write_text(template.replace('__DATA__',json.dumps(payload,ensure_ascii=False)),encoding='utf-8')
    dump(out/'sleeveless_candidates.json',{**payload,'cases':[{k:v for k,v in c.items() if k not in {'image','crop'}} for c in cases]})
    dump(out/'confirmed_previous_40.json',old)
    dump(out/'summary.json',{'status':'SLEEVELESS_SUPPLEMENT_REQUIRES_REVIEW','previous_reviewed_targets':40,
                            'previous_usable_targets':37,'previous_label_counts':dict(Counter(c['sleeve_length'] for c in usable)),
                            'supplement_candidates':8,'supplement_category_counts':dict(Counter(c['category'] for c in cases)),
                            'source_sha_exclusion':'All validation sources and all prior 40 reviewed sources excluded by exact hash',
                            'training_started':False,'formal_prd_acceptance':False})
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(out))
    print('SUPPLEMENT_READY: eight pending train targets, no training',flush=True)
    print('UPLOAD_THIS_ZIP:',archive,flush=True)


if __name__=='__main__':main()
