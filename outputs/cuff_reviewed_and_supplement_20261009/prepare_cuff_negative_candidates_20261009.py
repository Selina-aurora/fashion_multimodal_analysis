"""Export twelve candidate sleeveless training garments for visual review; no inferred truth."""
import importlib.util
import json
import base64
import hashlib
from pathlib import Path
from datetime import datetime
import zipfile
from collections import Counter


def main():
    from PIL import Image
    from fashion_multimodal_analysis.common.paths import project_root,resolve_path
    root=project_root();here=Path(__file__).resolve().parent
    original=root/'prepare_cuff_annotation_20261009.py'
    if not original.is_file():raise SystemExit('缺少原导出脚本 prepare_cuff_annotation_20261009.py，请放回项目根目录。')
    spec=importlib.util.spec_from_file_location('cuff_export_base',original);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    accepted=json.loads((here/'cuff_reviewed_20261009.json').read_text())
    if len(accepted['cases'])!=36 or any(c['review_status']!='reviewed' for c in accepted['cases']):raise ValueError('Reviewed batch incomplete')
    excluded_hashes={c['source_image_sha256'] for c in accepted['cases']}
    excluded_paths={resolve_path(c['source_image_reference']).resolve() for c in accepted['cases']}
    train=resolve_path('configs/prd_8class_train_v5_balanced150_20261008.csv');val=resolve_path('configs/prd_8class_val_v2_full_targets.csv');core=resolve_path('benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv')
    expected={train:'e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f',val:'39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93',core:'c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89'}
    for p,h in expected.items():
        if base.sha(p)!=h:raise ValueError('Manifest changed: '+str(p))
    protected={resolve_path(r['source_image']).resolve() for p in (val,core) for r in base.rows(p)}
    print('Checking protected source hashes...',flush=True)
    banned_hashes={base.sha(p) for p in protected}|excluded_hashes;used_paths=set();used_hashes=set();cases=[];all_rows=base.rows(train)
    strata=[('top','Fashionpedia',{'vest'}),('top','DeepFashion2',{'vest','sling'}),('dress','DeepFashion2',{'sling_dress','vest_dress'})]
    for category,dataset,fine in strata:
        pool=[r for r in all_rows if r['garment_category']==category and r['source_dataset']==dataset and r['fine_category'] in fine]
        pool.sort(key=lambda r:(0 if r['fine_category']=='sling_dress' else 1,-min(float(r['bbox_width']),float(r['bbox_height'])),r['record_id']))
        n=0
        for r in pool:
            source=resolve_path(r['source_image']).resolve()
            if source in protected or source in excluded_paths or source in used_paths:continue
            h=base.sha(source)
            if h in banned_hashes or h in used_hashes:continue
            with Image.open(source) as im:w,height=im.size
            box=[float(r['bbox_'+k]) for k in ('x1','y1','x2','y2')]
            if not(0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=height):raise ValueError('Invalid native garment box')
            if (w,height)!=(int(float(r['image_width'])),int(float(r['image_height']))):raise ValueError('Native image dimension mismatch')
            raw=source.read_bytes();name=r['record_id']+source.suffix.lower();mime='image/png' if source.suffix.lower()=='.png' else 'image/jpeg'
            cases.append({'record_id':r['record_id'],'source_dataset':dataset,'category':category,'fine_category':r['fine_category'],'source_image_reference':r['source_image'],'garment_id':r['garment_id'],'garment_bbox':box,'image_width':w,'image_height':height,'source_image_sha256':h,'source_file_relative':'source_images/'+name,'presence':'pending','cuff_boxes':[],'review_status':'pending','image_data':'data:'+mime+';base64,'+base64.b64encode(raw).decode()})
            used_paths.add(source);used_hashes.add(h);n+=1
            if n==4:break
        if n<4:raise ValueError(f'Insufficient isolated candidates: {category}/{dataset}: {n}')
    output=root/'reports/reruns/recovery_20261009'/('cuff_negative_candidates_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));(output/'source_images').mkdir(parents=True)
    for c in cases:(output/c['source_file_relative']).write_bytes(base64.b64decode(c['image_data'].split(',',1)[1]))
    data={'status':'CANDIDATES_PENDING_VISUAL_REVIEW_NOT_NEGATIVE_TRUTH','selection_sha256':hashlib.sha256(json.dumps([(c['record_id'],c['source_image_sha256']) for c in cases]).encode()).hexdigest(),'role':'training_side_candidate_pool_not_independent_test','definition':accepted['definition'],'script_sha256':base.sha(__file__),'reviewed_count':0,'cases':cases,'limitations':['Fine-category names select candidates only; do not establish cuff absence.','Excludes val/Core and previous 36 images by resolved path and exact file hash.','No visually similar or re-encoded image duplicate check.','Target garment only; visible sleeves from another layer must not be labeled as target cuffs.','No split or independent test set established.']}
    html=base.HTML.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')).replace('袖口训练侧标注小批次','无袖候选补充：需要目视确认')
    (output/'cuff_negative_candidates.html').write_text(html)
    (output/'selection.json').write_text(json.dumps({**data,'cases':[{k:v for k,v in c.items() if k!='image_data'} for c in cases]},ensure_ascii=False,indent=2))
    (output/'summary.json').write_text(json.dumps({'images':12,'reviewed':0,'status':data['status'],'strata':dict(Counter(c['category']+'/'+c['source_dataset'] for c in cases)),'protected_path_count':len(protected),'previous_batch_excluded':36},ensure_ascii=False,indent=2))
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in output.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(output))
    print('RESULT_ZIP='+str(archive),flush=True)


if __name__=='__main__':main()
