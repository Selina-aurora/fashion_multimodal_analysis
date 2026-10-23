"""Frozen V3 cuff holdout evaluation; no training, tuning or model promotion.
Source images are segmentation validation images, not an unseen end-to-end test.
"""
import json,hashlib,csv,math,time,zipfile
from pathlib import Path

BASE_SHA='c5ba795225349b1c521aac0517ac7c493d9818dcc17d558ac4a3f6f8669452f0'
CUFF_SHA='52f8d78ffa032b9135eea8ba8ea511e942e2f12cd793f49bbca324f720a91c84'
BASE='outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth'
CUFF='reports/reruns/recovery_20261009/cuff_expanded_train_v3_20261009_052743_791119/checkpoint_best.pth'
ANNOTATION_SHA='e955462cdc2bb9cd4b1b4bf3b961750eeb7c1660882b7eb3f5f32a892a09e62a'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_json(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def iou(a,b):
    area=lambda q:max(0,q[2]-q[0])*max(0,q[3]-q[1])
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));union=area(a)+area(b)-inter
    return inter/union if union else 0.
def match(pred,truth):
    links=[[j for j in sorted(range(len(truth)),key=lambda j:iou(p,truth[j]),reverse=True)if iou(p,truth[j])>=.5]for p in pred];assigned={}
    def augment(i,seen):
        for j in links[i]:
            if j in seen:continue
            seen.add(j)
            if j not in assigned or augment(assigned[j],seen):assigned[j]=i;return True
        return False
    for i in range(len(pred)):augment(i,set())
    tp=len(assigned);return {'tp':tp,'fp':len(pred)-tp,'fn':len(truth)-tp,'matches':[{'prediction_index':i,'truth_index':j,'iou':iou(pred[i],truth[j])}for j,i in assigned.items()]}
def metrics(cases):
    tp=sum(c['tp']for c in cases);fp=sum(c['fp']for c in cases);fn=sum(c['fn']for c in cases)
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp)if tp+fp else 0.,'recall':tp/(tp+fn)if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else 0.,'negative_images':sum(not c['truth_boxes']for c in cases),'negative_images_with_fp':sum(not c['truth_boxes']and c['fp']>0 for c in cases)}
def crop_bounds(box,w,h):
    x,y,xx,yy=box;dx=.1*(xx-x);dy=.1*(yy-y)
    return [max(0,math.floor(x-dx)),max(0,math.floor(y-dy)),min(w,math.ceil(xx+dx)),min(h,math.ceil(yy+dy))]
def main():
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from fashion_multimodal_analysis.common.paths import project_root,resolve_path
    from fashion_multimodal_analysis.common.schema import CLASS_TO_ID
    from fashion_multimodal_analysis.segmentation.modeling import build_evaluation_model
    from fashion_multimodal_analysis.image_processing.masks import reconstruct_full_mask
    here=Path(__file__).resolve().parent;root=project_root();annotations=here/'cuff_holdout_reviewed.json'
    if sha(annotations)!=ANNOTATION_SHA:raise ValueError('Frozen annotations changed')
    data=json.loads(annotations.read_text());cases=data['cases'];usable=[c for c in cases if c['presence']!='ambiguous']
    if len(cases)!=18 or len(usable)!=15 or sum(len(c['cuff_boxes'])for c in usable)!=21:raise ValueError('Frozen case counts changed')
    if not all(c['review_status']=='reviewed'for c in cases):raise ValueError('Unreviewed cases')
    base=root/BASE;cuff_path=root/CUFF
    if sha(base)!=BASE_SHA or sha(cuff_path)!=CUFF_SHA:raise ValueError('Frozen checkpoint changed')
    val=resolve_path('configs/prd_8class_val_v2_full_targets.csv')
    if sha(val)!='39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93':raise ValueError('Original val manifest changed')
    with val.open(encoding='utf-8-sig',newline='')as f:rows={r['record_id']:r for r in csv.DictReader(f)}
    if not torch.cuda.is_available():raise RuntimeError('Run on GPU in py312')
    segmenter=build_evaluation_model(640,1024);cp=torch.load(base,map_location='cpu',weights_only=False);segmenter.load_state_dict(cp['model_state_dict'],strict=True);del cp;segmenter.to('cuda').eval()
    cuff=build_evaluation_model(640,1024);cuff.roi_heads.box_predictor=FastRCNNPredictor(cuff.roi_heads.box_predictor.cls_score.in_features,2)
    cuff.roi_heads.mask_roi_pool=None;cuff.roi_heads.mask_head=None;cuff.roi_heads.mask_predictor=None
    old=cuff.backbone.body.conv1;cuff.backbone.body.conv1=torch.nn.Conv2d(4,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
    cuff.transform.image_mean=[.485,.456,.406,0.];cuff.transform.image_std=[.229,.224,.225,1.]
    cp=torch.load(cuff_path,map_location='cpu',weights_only=False);cuff.load_state_dict(cp['model_state_dict'],strict=True);del cp;cuff.to('cuda').eval()
    out=root/'reports/reruns/recovery_20261009'/f'cuff_holdout_results_{time.strftime("%Y%m%d_%H%M%S")}_{time.time_ns()%1000000:06d}';out.mkdir(parents=True);(out/'overlays').mkdir()
    results={'native_roi':[],'predicted_roi':[]}
    def detect(rgb,mask,bounds):
        x,y,xx,yy=bounds;arr=np.concatenate([rgb[y:yy,x:xx],mask[y:yy,x:xx,None].astype(np.float32)],axis=2)
        pred=cuff([torch.from_numpy(arr).permute(2,0,1).contiguous().to('cuda')])[0];keep=(pred['scores']>=.4)&(pred['labels']==1)
        bb=pred['boxes'][keep].cpu().tolist();ss=pred['scores'][keep].cpu().tolist()
        return [[q[0]+x,q[1]+y,q[2]+x,q[3]+y]for q in bb],ss
    with torch.inference_mode():
        for c in usable:
            row=rows[c['record_id']];source=resolve_path(row['source_image']);box=[float(row['bbox_'+k])for k in ['x1','y1','x2','y2']]
            if sha(source)!=c['source_image_sha256']or box!=c['garment_bbox']or row['garment_category']!=c['category']:raise ValueError('Native target changed')
            picture=Image.open(source).convert('RGB');w,h=picture.size
            if [w,h]!=[c['image_width'],c['image_height']]:raise ValueError('Image dimensions changed')
            rgb=np.asarray(picture,dtype=np.float32).copy()/255.;truth=c['cuff_boxes'];native_mask=np.asarray(reconstruct_full_mask(resolve_path(row['mask_path']),(w,h),tuple(int(v)for v in box)))>0
            if native_mask.shape!=(h,w)or not native_mask.any():raise ValueError('Invalid original garment mask')
            bounds=crop_bounds(box,w,h);bb,ss=detect(rgb,native_mask,bounds)
            native={'record_id':c['record_id'],'category':c['category'],'presence':c['presence'],'truth_boxes':truth,'predicted_boxes':bb,'scores':ss,'crop_xyxy':bounds,'gt_boxes_not_fully_inside_crop':sum(not(bounds[0]<=q[0]and bounds[1]<=q[1]and bounds[2]>=q[2]and bounds[3]>=q[3])for q in truth),**match(bb,truth)};results['native_roi'].append(native)
            pred=segmenter([torch.from_numpy(rgb).permute(2,0,1).contiguous().to('cuda')])[0];ids=[i for i,s in enumerate(pred['scores'].cpu().tolist())if s>=.4];allboxes=pred['boxes'].cpu().tolist();chosen=max(ids,key=lambda i:iou(allboxes[i],box))if ids else None
            bb=[];ss=[];bounds=None;roi=None;giou=0.;status='NO_GARMENT_PREDICTION'
            if chosen is not None:
                roi=allboxes[chosen];giou=iou(roi,box);status='GARMENT_IOU_BELOW_0.5'if giou<.5 else'GARMENT_CLASS_MISMATCH'if int(pred['labels'][chosen])!=CLASS_TO_ID[c['category']]else'MATCHED_PREDICTED_TARGET'
                if status=='MATCHED_PREDICTED_TARGET':
                    bounds=crop_bounds(roi,w,h);mask=pred['masks'][chosen,0].cpu().numpy()>=.5;bb,ss=detect(rgb,mask,bounds)
            result={'record_id':c['record_id'],'category':c['category'],'presence':c['presence'],'truth_boxes':truth,'predicted_boxes':bb,'scores':ss,'target_association_status':status,'garment_iou':giou,'predicted_garment_box':roi,'crop_xyxy':bounds,**match(bb,truth)};results['predicted_roi'].append(result)
            for role,item in [('native_roi',native),('predicted_roi',result)]:
                im=picture.copy();pen=ImageDraw.Draw(im);lw=max(2,w//250)
                for q in truth:pen.rectangle(q,outline='lime',width=lw)
                if role=='predicted_roi'and roi:pen.rectangle(roi,outline='yellow',width=lw)
                for q,s in zip(item['predicted_boxes'],item['scores']):pen.rectangle(q,outline='red',width=lw);pen.text((q[0],q[1]),f'{s:.2f}',fill='red')
                im.save(out/'overlays'/f'{c["record_id"]}_{role}.jpg')
            print(f'{len(results["native_roi"])}/15 {c["record_id"]}: native TP={native["tp"]} FP={native["fp"]}; predicted {status} TP={result["tp"]} FP={result["fp"]}',flush=True)
    report={'status':'FROZEN_CUFF_HOLDOUT_EVALUATION_COMPLETED','candidate_epoch':27,'candidate_sha256':CUFF_SHA,'base_sha256':BASE_SHA,'annotations_sha256':ANNOTATION_SHA,'selected_images':18,'usable_images':15,'positive_images':12,'negative_images':3,'gt_cuff_boxes':21,'ambiguous_excluded':[c['record_id']for c in cases if c['presence']=='ambiguous'],'confidence_threshold':.4,'matching_iou':.5,'native_roi_metrics':metrics(results['native_roi']),'predicted_roi_metrics':metrics(results['predicted_roi']),'matched_predicted_targets':sum(c['target_association_status']=='MATCHED_PREDICTED_TARGET'for c in results['predicted_roi']),'unmatched_negative_targets':sum(not c['truth_boxes']and c['target_association_status']!='MATCHED_PREDICTED_TARGET'for c in results['predicted_roi']),'cases':results,'formal_prd_acceptance':False,'limitations':['These images were not used for cuff training or tuning; original segmentation used them for validation.','Not a blind end-to-end benchmark.','Labels apply to the specified garment only, not all garments in the image.','Ground truth garment association is for evaluation only; predicted crops/masks never use cuff labels.','Unmatched positive targets remain cuff misses; unmatched negatives do not establish absence success.','Fixed checkpoint/thresholds; no selection of another epoch or threshold on this holdout.'],'overlay_legend':'green=confirmed cuff; red=prediction; yellow=predicted garment'}
    write_json(out/'summary.json',report);write_json(out/'reviewed_annotations.json',data)
    with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED)as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(out))
    print('UPLOAD ZIP: '+str(out.with_suffix('.zip')),flush=True)
if __name__=='__main__':main()
