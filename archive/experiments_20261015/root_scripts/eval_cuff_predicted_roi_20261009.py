#!/usr/bin/env python3
"""Development diagnostic of cuff candidate with V5 predicted garment ROI/mask.
Target association uses garment GT only for benchmark matching, never crop generation.
Not an independent blind test, full-image cuff metric, or formal PRD promotion.
"""
import argparse,json,math,sys,time,zipfile
from pathlib import Path
for folder in [Path.cwd()/'outputs/cuff_expanded_train_20261009',Path(__file__).resolve().parent/'outputs/cuff_expanded_train_20261009',Path(__file__).resolve().parent/'cuff_expanded_train_20261009']:
    if (folder/'train_cuff_expanded_v3.py').is_file():
        sys.path.insert(0,str(folder));break
from train_cuff_expanded_v3 import check_dataset,sha,match_boxes,summarize,write_json,iou,BASE_SHA

def padded_crop(box,w,h):
    x,y,xx,yy=box;dx=.1*(xx-x);dy=.1*(yy-y)
    return [max(0,math.floor(x-dx)),max(0,math.floor(y-dy)),min(w,math.ceil(xx+dx)),min(h,math.ceil(yy+dy))]

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-dir',type=Path);args=ap.parse_args()
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from fashion_multimodal_analysis.segmentation.modeling import build_evaluation_model
    from fashion_multimodal_analysis.common.paths import resolve_path
    from fashion_multimodal_analysis.common.schema import CLASS_TO_ID
    if not torch.cuda.is_available():raise RuntimeError('Run in GPU py312 environment')
    if args.run_dir:run=args.run_dir.resolve()
    else:
        runs=sorted(Path('reports/reruns/recovery_20261009').glob('cuff_expanded_train_v3_*/summary.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        if not runs:raise FileNotFoundError('No completed expanded V3 run')
        run=runs[0].parent
    conf=json.loads((run/'run_config.json').read_text());summary=json.loads((run/'summary.json').read_text());data=Path(conf['dataset_dir']);_,groups=check_dataset(data)
    manifest=json.loads((data/'cuff_pilot_manifest.json').read_text());sourcecases={c['record_id']:c for c in manifest['cases']}
    base=Path(conf['base_checkpoint']);weight=run/'checkpoint_best.pth'
    if sha(base)!=BASE_SHA or sha(weight)!=summary['weight_files']['checkpoint_best.pth']['sha256']:raise ValueError('Checkpoint changed')
    segmenter=build_evaluation_model(640,1024);cp=torch.load(base,map_location='cpu',weights_only=False);segmenter.load_state_dict(cp['model_state_dict'],strict=True);del cp;segmenter.to('cuda').eval()
    cuff=build_evaluation_model(640,1024);cuff.roi_heads.box_predictor=FastRCNNPredictor(cuff.roi_heads.box_predictor.cls_score.in_features,2)
    cuff.roi_heads.mask_roi_pool=None;cuff.roi_heads.mask_head=None;cuff.roi_heads.mask_predictor=None
    old=cuff.backbone.body.conv1;cuff.backbone.body.conv1=torch.nn.Conv2d(4,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
    cuff.transform.image_mean=[.485,.456,.406,0.];cuff.transform.image_std=[.229,.224,.225,1.]
    cp=torch.load(weight,map_location='cpu',weights_only=False);cuff.load_state_dict(cp['model_state_dict'],strict=True);del cp;cuff.to('cuda').eval()
    out=Path('reports/reruns/recovery_20261009')/f'cuff_predicted_roi_eval_{time.strftime("%Y%m%d_%H%M%S")}_{time.time_ns()%1000000:06d}';out.mkdir(parents=True,exist_ok=False)
    (out/'overlays').mkdir();cases=[];control=[]
    with torch.inference_mode():
        for r in groups['development_val']:
            c=sourcecases[r['record_id']];source=resolve_path(c['source_image_reference'])
            if sha(source)!=c['source_image_sha256']:raise ValueError('Source changed')
            image=Image.open(source).convert('RGB');rgb=np.asarray(image,dtype=np.float32).copy()/255.;w,h=image.size
            def cuff_predict(rgb_array,mask):
                inp=torch.from_numpy(np.concatenate([rgb_array,mask[:,:,None]],axis=2)).permute(2,0,1).contiguous().to('cuda')
                pred=cuff([inp])[0];keep=(pred['scores']>=.4)&(pred['labels']==1)
                return pred['boxes'][keep].cpu().tolist(),pred['scores'][keep].cpu().tolist()
            native_rgb=np.asarray(Image.open(data/r['image']).convert('RGB'),dtype=np.float32).copy()/255.
            native_mask=(np.asarray(Image.open(data/r['garment_mask']).convert('L'))>0).astype(np.float32)
            bb,ss=cuff_predict(native_rgb,native_mask);ox,oy,_,_=r['input_crop_xyxy'];control_boxes=[[b[0]+ox,b[1]+oy,b[2]+ox,b[3]+oy]for b in bb]
            truth=c['cuff_boxes'];control.append({'record_id':r['record_id'],'truth_boxes':truth,'predicted_boxes':control_boxes,'scores':ss,**match_boxes(control_boxes,truth)})
            result=segmenter([torch.from_numpy(rgb).permute(2,0,1).contiguous().to('cuda')])[0]
            candidates=[i for i,s in enumerate(result['scores'].cpu().tolist())if s>=.4]
            candidate_boxes=result['boxes'].cpu().tolist();chosen=max(candidates,key=lambda i:iou(candidate_boxes[i],c['garment_bbox']))if candidates else None
            predicted=[];scores=[];bounds=None;roi_box=None;roi_score=None;roi_iou=0.;roi_label=None;status='NO_GARMENT_PREDICTION'
            if chosen is not None:
                roi_box=candidate_boxes[chosen];roi_score=float(result['scores'][chosen]);roi_iou=iou(roi_box,c['garment_bbox']);roi_label=int(result['labels'][chosen])
                status='GARMENT_IOU_BELOW_0.5'if roi_iou<.5 else 'GARMENT_CLASS_MISMATCH'if roi_label!=CLASS_TO_ID[c['category']]else 'MATCHED_PREDICTED_TARGET'
                if status=='MATCHED_PREDICTED_TARGET':
                    bounds=padded_crop(roi_box,w,h);x,y,xx,yy=bounds
                    mask=(result['masks'][chosen,0,y:yy,x:xx].cpu().numpy()>=.5).astype(np.float32)
                    bb,scores=cuff_predict(rgb[y:yy,x:xx],mask)
                    predicted=[[b[0]+x,b[1]+y,b[2]+x,b[3]+y]for b in bb]
            row={'record_id':r['record_id'],'presence':c['presence'],'category':c['category'],'target_association_status':status,'predicted_garment_box':roi_box,'garment_score':roi_score,'garment_iou':roi_iou,'garment_label_id':roi_label,'input_crop_xyxy':bounds,'truth_boxes':truth,'predicted_boxes':predicted,'scores':scores,**match_boxes(predicted,truth)};cases.append(row)
            pic=image.copy();pen=ImageDraw.Draw(pic);lw=max(2,w//250)
            for b in truth:pen.rectangle(b,outline='lime',width=lw)
            if roi_box:pen.rectangle(roi_box,outline='yellow',width=lw)
            for b,s in zip(predicted,scores):pen.rectangle(b,outline='red',width=lw);pen.text((b[0],b[1]),f'{s:.2f}',fill='red')
            pic.save(out/'overlays'/f'{r["record_id"]}.jpg');print(f'{r["record_id"]}: {status} TP={row["tp"]} FP={row["fp"]} FN={row["fn"]}',flush=True)
    expected=json.loads((run/'best_validation.json').read_text())['metrics'];control_metrics=summarize(control)
    for k in ['tp','fp','fn']:
        if control_metrics[k]!=expected[k]:raise RuntimeError(f'Native ROI control did not reproduce saved metrics: {control_metrics} vs {expected}')
    report={'status':'DEVELOPMENT_PREDICTED_ROI_DIAGNOSTIC_COMPLETED','candidate_checkpoint':str(weight.resolve()),'candidate_sha256':sha(weight),'selected_epoch':summary['best_epoch'],'native_roi_control':control_metrics,'predicted_roi_target_conditioned_metrics':summarize(cases),'matched_garment_targets':sum(c['target_association_status']=='MATCHED_PREDICTED_TARGET'for c in cases),'total_targets':len(cases),'unmatched_negative_targets':sum(not c['truth_boxes']and c['target_association_status']!='MATCHED_PREDICTED_TARGET'for c in cases),'cases':cases,'control_cases':control,'formal_prd_acceptance':False,'limitations':['Same small development set used for model selection; not independent generalization evidence.','GT garment box/class used only for evaluation target association, not predicted input crop or mask.','Unmatched garment targets count as missed cuffs for positive targets.','Unmatched negatives are reported separately; cannot claim successful absence detection.','Metrics cover the annotated target garment only; unannotated other garment cuffs are not evaluated.','Standalone candidate diagnostic, not a production pipeline integration.'],'overlay_legend':'green=confirmed cuff; red=predicted cuff; yellow=predicted garment'}
    write_json(out/'summary.json',report)
    with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED)as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(out))
    print(f'UPLOAD ZIP: {out.with_suffix(".zip")}',flush=True)

if __name__=='__main__':main()
