"""Materialize reviewed target-garment crops and aligned masks. CPU only; no training."""
import json
import math
import csv
import hashlib
from pathlib import Path
from datetime import datetime
import zipfile


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    import numpy as np
    from PIL import Image
    from fashion_multimodal_analysis.common.paths import project_root,resolve_path
    from fashion_multimodal_analysis.image_processing.masks import reconstruct_full_mask
    here=Path(__file__).resolve().parent;root=project_root()
    manifest=here/'cuff_pilot_manifest.json';data=json.loads(manifest.read_text());audit=json.loads((here/'DATA_CHECK.json').read_text())
    if sha(manifest)!=audit['manifest_sha256']:raise ValueError('Pilot manifest changed')
    train=resolve_path('configs/prd_8class_train_v5_balanced150_20261008.csv');val=resolve_path('configs/prd_8class_val_v2_full_targets.csv');core=resolve_path('benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv')
    expected={train:'e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f',val:'39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93',core:'c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89'}
    for p,h in expected.items():
        if sha(p)!=h:raise ValueError('Original manifest changed: '+str(p))
    native={r['record_id']:r for r in read_rows(train)}
    protected={resolve_path(r['source_image']).resolve() for p in (val,core) for r in read_rows(p)}
    print('Checking protected file hashes...',flush=True);protected_hashes={sha(p) for p in protected}
    output=root/'reports/reruns/recovery_20261009'/('cuff_expanded_data_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));output.mkdir(parents=True,exist_ok=False)
    records=[];hashes=set()
    for case in data['cases']:
        if case['pilot_split']=='excluded_ambiguous':continue
        if case['review_status']!='reviewed' or case['presence'] not in ('present','absent'):raise ValueError('Unreviewed training case')
        row=native[case['record_id']];source=resolve_path(row['source_image']).resolve();h=sha(source)
        if source in protected or h in protected_hashes or h in hashes:raise ValueError('Protected/duplicate source: '+case['record_id'])
        if h!=case['source_image_sha256']:raise ValueError('Source bytes changed')
        native_box=[float(row['bbox_'+k]) for k in ('x1','y1','x2','y2')]
        if native_box!=case['garment_bbox'] or row['garment_category']!=case['category']:raise ValueError('Native target differs')
        image=Image.open(source).convert('RGB');w,height=image.size
        if [w,height]!=[case['image_width'],case['image_height']]:raise ValueError('Image dimensions changed')
        native_bounds=[int(math.floor(native_box[0])),int(math.floor(native_box[1])),int(math.ceil(native_box[2])),int(math.ceil(native_box[3]))]
        # Uniform context policy; no cuff coordinates enter crop generation.
        # Reuse the same 10% policy with predicted ROIs in any later inference adapter.
        dx=(native_box[2]-native_box[0])*.1;dy=(native_box[3]-native_box[1])*.1
        x1=max(0,int(math.floor(native_box[0]-dx)));y1=max(0,int(math.floor(native_box[1]-dy)))
        x2=min(w,int(math.ceil(native_box[2]+dx)));y2=min(height,int(math.ceil(native_box[3]+dy)))
        if not(0<=x1<x2<=w and 0<=y1<y2<=height):raise ValueError('Invalid target crop')
        full=np.asarray(reconstruct_full_mask(resolve_path(row['mask_path']),(w,height),tuple([int(v) for v in native_box])))>0
        if full.shape!=(height,w) or not np.any(full):raise ValueError('Invalid native garment mask')
        crop=image.crop((x1,y1,x2,y2));mask=full[y1:y2,x1:x2]
        if not mask.any():raise ValueError('Empty target mask crop')
        boxes=[];box_audit=[]
        for b in case['cuff_boxes']:
            clipped=[max(x1,b[0]),max(y1,b[1]),min(x2,b[2]),min(y2,b[3])]
            if clipped[2]<=clipped[0] or clipped[3]<=clipped[1]:raise ValueError('Cuff completely outside native garment crop')
            retention=(clipped[2]-clipped[0])*(clipped[3]-clipped[1])/((b[2]-b[0])*(b[3]-b[1]))
            if retention<.999999:raise ValueError('Cuff box extends outside fixed crop; manual review needed: '+case['record_id'])
            local=[clipped[0]-x1,clipped[1]-y1,clipped[2]-x1,clipped[3]-y1]
            assert [local[0]+x1,local[1]+y1,local[2]+x1,local[3]+y1]==clipped
            boxes.append(local);a,b1,c,d=[int(v) for v in local]
            box_audit.append({'source_box':b,'local_box':local,'source_box_area_retained':retention,'native_garment_mask_fraction_in_box':float(mask[b1:d,a:c].mean())})
        if (case['presence']=='present')!=bool(boxes):raise ValueError('Presence/boxes inconsistent')
        folder=output/case['pilot_split']/case['record_id'];folder.mkdir(parents=True)
        crop.save(folder/'image.png');Image.fromarray(mask.astype('uint8')*255).save(folder/'garment_mask.png')
        record={'record_id':case['record_id'],'split':case['pilot_split'],'presence':case['presence'],'category':case['category'],'source_dataset':case['source_dataset'],'source_image_sha256':h,'native_crop_xyxy':native_bounds,'input_crop_xyxy':[x1,y1,x2,y2],'context_padding_fraction':.1,'crop_size':list(crop.size),'image':str((folder/'image.png').relative_to(output)),'garment_mask':str((folder/'garment_mask.png').relative_to(output)),'cuff_boxes_xyxy_local':boxes,'box_audit':box_audit,'image_sha256':sha(folder/'image.png'),'garment_mask_sha256':sha(folder/'garment_mask.png')}
        records.append(record);hashes.add(h);(folder/'target.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
        print(f'{len(records)}/76 {case["record_id"]}',flush=True)
    assert len(records)==76
    (output/'dataset.json').write_text(json.dumps({'status':'REVIEWED_CUFF_PILOT_MATERIALIZED','manifest_sha256':sha(manifest),'role':data['role'],'task':data['task'],'definition':data['definition'],'limitations':data['limitations'],'records':records},ensure_ascii=False,indent=2))
    summary={'status':'DATA_READY_TRAINING_NOT_STARTED','cases':76,'train_images':67,'development_val_images':9,'positive_images':54,'negative_images':22,'cuff_boxes':sum(len(r['cuff_boxes_xyxy_local']) for r in records),'ambiguous_excluded':20,'protected_source_files_checked':len(protected),'source_hashes_unique':len(hashes),'crop_coordinate_roundtrip_checked':True,'context_padding_fraction':.1,'crop_policy_uses_cuff_labels':False,'model_weights_modified':False,'low_mask_overlap_boxes_are_diagnostic_not_auto_relabeling':True}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(output/'cuff_pilot_manifest.json').write_bytes(manifest.read_bytes())
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in output.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(output))
    print('DATA_ZIP='+str(archive),flush=True)
    return output


if __name__=='__main__':main()
