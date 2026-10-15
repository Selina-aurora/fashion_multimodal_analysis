"""全量训练掩码检查并导出训练侧复核样本，仅使用 CPU，不改标签或权重。

使用训练器相同的 mask 重建函数。框/掩码几何异常只是审核线索，不能
证明语义标签错误；小尺寸阈值也不是 PRD 验收标准。原训练清单只读。
"""
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import csv
import hashlib
import json
import zipfile
import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def mask_geometry(array,box):
    ys,xs=np.nonzero(array)
    if len(xs)==0:return {'mask_pixels':0,'tight_bbox':None,'outside_bbox_fraction':None,'bbox_maskbox_iou':None}
    tight=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
    x1,y1,x2,y2=box
    inside=int(array[y1:y2,x1:x2].sum())
    inter=max(0,min(x2,tight[2])-max(x1,tight[0]))*max(0,min(y2,tight[3])-max(y1,tight[1]))
    union=(x2-x1)*(y2-y1)+(tight[2]-tight[0])*(tight[3]-tight[1])-inter
    return {'mask_pixels':len(xs),'tight_bbox':tight,
            'outside_bbox_fraction':(len(xs)-inside)/len(xs),'bbox_maskbox_iou':inter/max(1,union)}


def main():
    from fashion_multimodal_analysis.common.paths import project_root,resolve_path
    from fashion_multimodal_analysis.image_processing.masks import reconstruct_full_mask
    from fashion_multimodal_analysis.common.schema import CLASS_TO_ID
    root=project_root()
    train=resolve_path('configs/prd_8class_train_v5_balanced150_20261008.csv')
    val=resolve_path('configs/prd_8class_val_v2_full_targets.csv')
    core=resolve_path('benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv')
    expected='e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f'
    if sha(train)!=expected:raise SystemExit('训练清单不是昨日原 V5 清单，未开始检查。')
    rows=read_csv(train)
    if len(rows)!=2299 or len({r['source_image'] for r in rows})!=625:
        raise SystemExit('训练清单数量不符。')
    protected={str(resolve_path(r['source_image'])) for p in (val,core) for r in read_csv(p)}
    overlap=sorted({str(resolve_path(r['source_image'])) for r in rows}&protected)
    if overlap:raise SystemExit(f'训练与验证/Core按规范化路径重叠：{len(overlap)}，未导出样本。')
    output=root/'reports/reruns/recovery_20261009'/('training_mask_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=False)
    dimensions={}; records=[]; prepared={}
    for index,row in enumerate(rows,1):
        result={k:row[k] for k in ('record_id','source_dataset','source_image','garment_id','garment_category','fine_category','mask_path')}
        result.update(status='checked',error='',mask_representation='',mask_pixels='',bbox_area_ratio='',mask_area_ratio='',bbox_maskbox_iou='',outside_bbox_fraction='',estimated_resized_min_side='',review_flags='',bbox_json='',tight_mask_bbox_json='')
        flags=[]
        try:
            source=resolve_path(row['source_image']);mask_path=resolve_path(row['mask_path'])
            if source not in dimensions:
                with Image.open(source) as image: dimensions[source]=image.size
            iw,ih=dimensions[source]
            if row['garment_category'] not in CLASS_TO_ID:raise ValueError('Unknown PRD category')
            if (int(float(row['image_width'])),int(float(row['image_height'])))!=(iw,ih):
                flags.append('manifest_image_dimensions_differ')
            raw_box=[int(float(row['bbox_'+k])) for k in ('x1','y1','x2','y2')]
            a,b,c,d=raw_box
            a=max(0,min(iw-1,a));b=max(0,min(ih-1,b));c=max(a+1,min(iw,c));d=max(b+1,min(ih,d))
            box=[a,b,c,d]
            if box!=raw_box:flags.append('trainer_clips_bbox')
            with Image.open(mask_path) as opened:
                raw_size=opened.size
                raw_nonempty=bool(np.any(np.asarray(opened.convert('L'))>0))
            if not raw_nonempty:raise ValueError('Empty stored mask')
            if raw_size==(iw,ih):representation='full_image'
            elif raw_size==(c-a,d-b):representation='bbox_crop'
            else:
                representation='resized_bbox_crop'
                flags.append('trainer_resizes_mask_to_bbox_review')
            mask=np.asarray(reconstruct_full_mask(mask_path,(iw,ih),tuple(box)))>0
            if mask.shape!=(ih,iw):raise ValueError('Reconstructed mask has wrong dimensions')
            geometry=mask_geometry(mask,box)
            if geometry['mask_pixels']==0:raise ValueError('Empty reconstructed mask')
            if geometry['outside_bbox_fraction']>.01:flags.append('over_1pct_mask_outside_bbox_review')
            if geometry['bbox_maskbox_iou']<.5:flags.append('bbox_and_tight_maskbox_iou_below_0_5_review')
            # 只估算 torchvision min640/max1024 缩放，不声称测到内部真实张量尺寸。
            scale=min(640/min(iw,ih),1024/max(iw,ih))
            min_side=min(c-a,d-b)*scale
            if min_side<16:flags.append('estimated_min_side_below_16px')
            result.update(mask_representation=representation,mask_pixels=geometry['mask_pixels'],
                          bbox_area_ratio=(c-a)*(d-b)/(iw*ih),mask_area_ratio=geometry['mask_pixels']/(iw*ih),
                          bbox_maskbox_iou=geometry['bbox_maskbox_iou'],outside_bbox_fraction=geometry['outside_bbox_fraction'],
                          estimated_resized_min_side=min_side,bbox_json=json.dumps(box),
                          tight_mask_bbox_json=json.dumps(geometry['tight_bbox']))
            prepared[row['record_id']]=(row,source,mask_path,box,(iw,ih))
        except (OSError,ValueError,KeyError,TypeError) as e:
            result.update(status='error',error=f'{type(e).__name__}: {e}')
        result['review_flags']=';'.join(flags);records.append(result)
        if index%200==0 or index==len(rows):print(f'AUDIT {index}/{len(rows)}',flush=True)

    grouped=defaultdict(list)
    for r in records:grouped[(r['garment_category'],r['fine_category'])].append(r)
    stats=[]
    for (category,fine),group in sorted(grouped.items()):
        valid=[r for r in group if r['status']=='checked']
        stats.append({'category':category,'fine_category':fine,'instances':len(group),'checked':len(valid),
                      'errors':len(group)-len(valid),'flagged_instances':sum(bool(r['review_flags']) for r in valid),
                      'estimated_min_side_below_16px':sum(r['estimated_resized_min_side']<16 for r in valid),
                      'median_mask_area_ratio':float(np.median([r['mask_area_ratio'] for r in valid])) if valid else ''})
    # 先选几何审核线索，再按训练侧类别覆盖取最小实例；不使用测试标签选训练样本。
    geometric=[r for r in records if r['status']=='checked' and any(k in r['review_flags'] for k in
               ('outside_bbox','maskbox_iou','resizes_mask','dimensions_differ','clips_bbox'))]
    selected=sorted(geometric,key=lambda r:r['record_id'])[:8]
    seen={r['record_id'] for r in selected}
    priorities=[('accessory','glove'),('accessory','watch'),('accessory','sock'),('bag','bag, wallet'),
                ('skirt','skirt'),('outerwear','long_sleeve_outwear'),('outerwear','short_sleeve_outwear'),
                ('outerwear','cardigan'),('dress','vest_dress'),('dress','long_sleeve_dress'),
                ('top','long_sleeve_top'),('top','vest')]
    for key in priorities:
        group=sorted((r for r in grouped.get(key,[]) if r['status']=='checked'),key=lambda r:(r['mask_area_ratio'],r['record_id']))
        for r in [r for r in group if r['record_id'] not in seen][:2]:selected.append(r);seen.add(r['record_id'])
    reviews=[]
    for index,r in enumerate(selected[:32],1):
        row,source,mask_path,box,size=prepared[r['record_id']]
        with Image.open(source) as opened:image=opened.convert('RGB')
        mask=reconstruct_full_mask(mask_path,size,tuple(box))
        overlay=Image.blend(image,Image.new('RGB',size,'lime'),.3)
        view=Image.composite(overlay,image,mask)
        draw=ImageDraw.Draw(view);draw.rectangle(box,outline='red',width=3)
        view.thumbnail((1100,1100))
        folder=output/f'review_{index:02d}';folder.mkdir()
        view.save(folder/'mask_overlay.jpg',quality=92)
        original=image.copy();original.thumbnail((1100,1100));original.save(folder/'source.jpg',quality=92)
        crop=image.crop(box);crop.thumbnail((768,768));crop.save(folder/'object_crop.jpg',quality=92)
        exported=dict(r,review_folder=folder.name,source_sha256=sha(source),mask_sha256=sha(mask_path),
                      review_status='pending_not_frozen_gt')
        reviews.append(exported)
        (folder/'record.json').write_text(json.dumps(exported,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'rows':len(records),'images':len(dimensions),'checked':sum(r['status']=='checked' for r in records),
             'errors':sum(r['status']=='error' for r in records),
             'mask_representation_counts':dict(Counter(r['mask_representation'] for r in records if r['status']=='checked')),
             'review_flag_counts':dict(Counter(flag for r in records for flag in r['review_flags'].split(';') if flag)),
             'training_review_images':len(reviews),'normalized_path_overlap_with_val_core':0,
             'train_manifest_sha256':sha(train),'labels_modified':False,'model_inference_run':False,
             'notes':['Flags are review cues, not proven annotation errors.','No source-polygon pixel comparison or semantic accuracy computed.',
                      'Sampled reviews are training-side only; normalized-path isolation does not prove cross-path byte/content deduplication.',
                      'Small-object dimension is an approximate resize calculation, not PRD acceptance.']}
    write_csv(output/'per_instance_mask_audit.csv',records);write_csv(output/'fine_category_stats.csv',stats)
    (output/'review_cases.json').write_text(json.dumps(reviews,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(output.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(output))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('UPLOAD_THIS_ZIP:',archive,flush=True)


if __name__=='__main__':main()
