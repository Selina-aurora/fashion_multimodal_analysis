#!/usr/bin/env python3
"""Read-only cuff pilot diagnosis. No training or model promotion.
Requires run_cuff_pilot_train_20261009.py in the same directory.
GT-box classification is an oracle diagnostic, never a prediction metric.
"""
import argparse,json,time,zipfile
from pathlib import Path
from run_cuff_pilot_train_20261009 import check_dataset,match_boxes,summarize,write_json,sha

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-dir',type=Path);args=ap.parse_args()
    import torch
    import numpy as np
    from PIL import Image,ImageDraw
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.ops import box_iou
    from fashion_multimodal_analysis.segmentation.modeling import build_evaluation_model
    if not torch.cuda.is_available():raise RuntimeError('Run in GPU environment')
    if args.run_dir:run=args.run_dir.resolve()
    else:
        runs=sorted(Path('reports/reruns/recovery_20261009').glob('cuff_pilot_train_*/summary.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        if not runs:raise FileNotFoundError('No training output found')
        run=runs[0].parent
    conf=json.loads((run/'run_config.json').read_text());data=Path(conf['dataset_dir']);_,groups=check_dataset(data)
    original=json.loads((run/'summary.json').read_text())
    out=run.parent/f'cuff_pilot_diagnosis_{time.strftime("%Y%m%d_%H%M%S")}_{time.time_ns()%1000000:06d}';out.mkdir(exist_ok=False)
    model=build_evaluation_model(640,1024)
    model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,2)
    model.roi_heads.mask_roi_pool=None;model.roi_heads.mask_head=None;model.roi_heads.mask_predictor=None
    old=model.backbone.body.conv1;model.backbone.body.conv1=torch.nn.Conv2d(4,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
    model.transform.image_mean=[.485,.456,.406,0.];model.transform.image_std=[.229,.224,.225,1.]
    model.roi_heads.score_thresh=.001
    model.to('cuda').eval();summary={'role':'diagnostic only; threshold .4 remains the reported operating point','run_dir':str(run),'dataset_json_sha256':sha(data/'dataset.json'),'checkpoints':{}}
    with torch.inference_mode():
        for name in ['checkpoint_best.pth','checkpoint_last.pth']:
            weight=run/name
            if sha(weight)!=original['weight_files'][name]['sha256']:raise ValueError('Checkpoint hash mismatch')
            cp=torch.load(weight,map_location='cpu',weights_only=False);model.load_state_dict(cp['model_state_dict'],strict=True);del cp
            report={};checkpoint_out=out/name.replace('.pth','');checkpoint_out.mkdir()
            for split,records in groups.items():
                all_cases=[];coverage=[];oracle=[]
                for r in records:
                    rgb=np.asarray(Image.open(data/r['image']).convert('RGB'),dtype=np.float32).copy()/255.
                    mask=(np.asarray(Image.open(data/r['garment_mask']).convert('L'))>0).astype(np.float32)
                    tensor=torch.from_numpy(np.concatenate([rgb,mask[:,:,None]],axis=2)).permute(2,0,1).contiguous().to('cuda')
                    gt=torch.tensor(r['cuff_boxes_xyxy_local'],dtype=torch.float32,device='cuda').reshape(-1,4)
                    images,targets=model.transform([tensor],[{'boxes':gt,'labels':torch.ones(len(gt),dtype=torch.int64,device='cuda')}])
                    features=model.backbone(images.tensors);proposals,_=model.rpn(images,features,None)
                    resized_gt=targets[0]['boxes']
                    max_iou=box_iou(resized_gt,proposals[0]).max(dim=1).values.cpu().tolist()if len(gt)and len(proposals[0])else [0.]*len(gt)
                    coverage.extend(max_iou)
                    oracle_scores=[]
                    if len(gt):
                        pooled=model.roi_heads.box_roi_pool(features,[resized_gt],images.image_sizes)
                        logits,_=model.roi_heads.box_predictor(model.roi_heads.box_head(pooled))
                        oracle_scores=logits.softmax(dim=1)[:,1].cpu().tolist();oracle.extend(oracle_scores)
                    detections,_=model.roi_heads(features,proposals,images.image_sizes,None)
                    pred=model.transform.postprocess(detections,images.image_sizes,[tensor.shape[-2:]])[0]
                    boxes=pred['boxes'].cpu().tolist();scores=pred['scores'].cpu().tolist()
                    c={'record_id':r['record_id'],'truth_boxes':gt.cpu().tolist(),'predicted_boxes':boxes,'scores':scores,'rpn_best_iou_per_gt':max_iou,'oracle_gt_box_cuff_scores':oracle_scores};all_cases.append(c)
                    if split=='development_val':
                        picture=Image.open(data/r['image']).convert('RGB');pen=ImageDraw.Draw(picture)
                        for b in c['truth_boxes']:pen.rectangle(b,outline='lime',width=3)
                        for b,s in list(zip(boxes,scores))[:10]:pen.rectangle(b,outline='red',width=2);pen.text((b[0],b[1]),f'{s:.3f}',fill='red')
                        picture.save(checkpoint_out/f'{r["record_id"]}_top10.jpg')
                metrics={}
                for threshold in [.4,.1,.05,.01]:
                    cases=[]
                    for c in all_cases:
                        retained=[b for b,s in zip(c['predicted_boxes'],c['scores'])if s>=threshold]
                        cases.append({'truth_boxes':c['truth_boxes'],**match_boxes(retained,c['truth_boxes'])})
                    metrics[str(threshold)]=summarize(cases)
                report[split]={'operating_point_0.4':metrics['0.4'],'diagnostic_thresholds_only':metrics,'rpn_gt_coverage_iou_0.5':sum(v>=.5 for v in coverage),'gt_boxes':len(coverage),'rpn_best_iou_mean':sum(coverage)/len(coverage)if coverage else None,'oracle_gt_box_cuff_score_mean':sum(oracle)/len(oracle)if oracle else None}
                write_json(checkpoint_out/f'{split}_cases.json',all_cases)
                print(f'{name} {split}: {json.dumps(report[split])}',flush=True)
            summary['checkpoints'][name]=report
    summary['limitations']=['Train results measure fitting, not generalization.','Development threshold comparisons are diagnostic, not acceptance.','GT ROI classification uses cuff labels only to isolate classification behavior.','RPN coverage uses ground truth for measurement only.','Prediction overlays show up to ten boxes >= .001; these are not operating-point predictions.','No weights or annotations modified.']
    write_json(out/'summary.json',summary)
    with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED)as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(out))
    print(f'UPLOAD ZIP: {out.with_suffix(".zip")}',flush=True)

if __name__=='__main__':main()
