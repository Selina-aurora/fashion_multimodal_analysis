#!/usr/bin/env python3
"""V3 expanded target-garment cuff BOX experiment; development validation, not PRD acceptance.
Download into /workspace/fashion_multimodal_analysis and run with PYTHONPATH=src.
RGB plus target mask; 10% context crop is prepared independently of cuff labels.
No cuff mask training, SAM refinement, formal model promotion, or blind test.
"""
import argparse
import hashlib
import json
import math
import random
import time
import zipfile
from pathlib import Path

MANIFEST_SHA = '94ad937c80a9ecc099d96ff0f5d833f05bcd22a9a3f58de3ce7f8d994261fb58'
BASE_SHA = 'c5ba795225349b1c521aac0517ac7c493d9818dcc17d558ac4a3f6f8669452f0'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def check_dataset(root):
    from PIL import Image
    d = json.loads((root/'dataset.json').read_text())
    if d['manifest_sha256'] != MANIFEST_SHA or sha(root/'cuff_pilot_manifest.json') != MANIFEST_SHA:
        raise ValueError('Frozen manifest hash mismatch')
    records = d['records']; groups = {s: [] for s in ['train', 'development_val']}
    for r in records:
        groups[r['split']].append(r)
        if r['presence'] not in ['present', 'absent']: raise ValueError('Ambiguous sample included')
        for field, hf in [('image','image_sha256'), ('garment_mask','garment_mask_sha256')]:
            p=root/r[field]
            if sha(p)!=r[hf] or Image.open(p).size!=tuple(r['crop_size']): raise ValueError(f'Corrupt sample: {r["record_id"]}')
        if r['context_padding_fraction']!=0.1: raise ValueError('Crop policy mismatch')
        if (r['presence']=='present') != bool(r['cuff_boxes_xyxy_local']): raise ValueError('Label mismatch')
        for b in r['box_audit']:
            x,y,_,_=r['input_crop_xyxy']; q=b['source_box']
            if b['source_box_area_retained']<.999999 or b['local_box']!=[q[0]-x,q[1]-y,q[2]-x,q[3]-y]: raise ValueError('Clipped or shifted cuff')
        for x1,y1,x2,y2 in r['cuff_boxes_xyxy_local']:
            w,h=r['crop_size']
            if not (0<=x1<x2<=w and 0<=y1<y2<=h): raise ValueError('Invalid box')
    if [len(groups[s]) for s in groups] != [67,9]: raise ValueError('Split count mismatch')
    hashes=[{r['source_image_sha256'] for r in groups[s]} for s in groups]
    if len(hashes[0])!=67 or len(hashes[1])!=9 or hashes[0]&hashes[1]: raise ValueError('Duplicate image / split overlap')
    if sum(len(r['cuff_boxes_xyxy_local']) for r in records)!=98: raise ValueError('Box count mismatch')
    if sorted(hashes[1])!=['08d9b1b4aa091a8070c8286dcd4e38af3329d73749fad31bffcc16281013d7c1', '0bf3598b752c26f60f53a9069c0daabc91b3b313c74d626134599079871c2dab', '0f3dacae99c6dcb74e1cb1dfdb3b3ef9e8c2c6435f636ce1ebd27b0060c68cd0', '6c1faa6f3fa57132543f18769fc7a0b3fed698f76adb25a074aa5d6cd28c66d5', 'ad048c59fb04f42bf703b5344d9928d461aad6c925af43f114cdffaf26294013', 'b17c91aa9f687020190918874d3b858ac16210849853bafc4db2883ce9a4028f', 'd9aac98bd3fa22494fd98de318f5ab0e7e54c0ffe0b63a990e129afd0d00242c', 'd9d0675eb6291def49d9dba41be99d1acdb0d0cdc078bcb9c9dee90c816c27e7', 'efa0b4274ccfa3102dad25e86f95b924cc714ab6618fc7d6e3ca8a2ed35d86a2']: raise ValueError('Frozen development images changed')
    return d,groups

def flip_boxes(boxes,width):
    return [[width-b[2],b[1],width-b[0],b[3]]for b in boxes]

def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda q:max(0,q[2]-q[0])*max(0,q[3]-q[1])
    union=area(a)+area(b)-inter
    return inter/union if union else 0.0

def match_boxes(pred, truth, threshold=.5):
    # Maximum-cardinality bipartite matching: each prediction and GT used once.
    links=[[j for j in sorted(range(len(truth)),key=lambda j:iou(p,truth[j]),reverse=True) if iou(p,truth[j])>=threshold] for p in pred]
    assigned={}
    def augment(i,seen):
        for j in links[i]:
            if j in seen: continue
            seen.add(j)
            if j not in assigned or augment(assigned[j],seen): assigned[j]=i; return True
        return False
    for i in range(len(pred)): augment(i,set())
    pairs=[{'prediction_index':i,'truth_index':j,'iou':iou(pred[i],truth[j])}for j,i in assigned.items()]
    tp=len(pairs)
    return {'tp':tp,'fp':len(pred)-tp,'fn':len(truth)-tp,'matches':pairs}

def summarize(cases):
    tp=sum(c['tp']for c in cases);fp=sum(c['fp']for c in cases);fn=sum(c['fn']for c in cases)
    precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.
    negatives=[c for c in cases if not c['truth_boxes']]
    return {'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'negative_images':len(negatives),'negative_images_with_fp':sum(c['fp']>0 for c in negatives)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-dir',type=Path)
    ap.add_argument('--checkpoint',type=Path,default=Path('outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth'))
    ap.add_argument('--epochs',type=int,default=40)
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--no-augmentation',action='store_true',help='Optional same-data control run; defaults to train-only augmentation')
    args=ap.parse_args()
    if args.epochs<1: ap.error('--epochs must be positive')
    root=Path.cwd()
    if args.dataset_dir:
        data_root=args.dataset_dir.resolve()
    else:
        candidates=sorted(root.glob('reports/reruns/recovery_20261009/cuff_pilot_data_*/dataset.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        candidates=[p for p in candidates if json.loads(p.read_text()).get('manifest_sha256')==MANIFEST_SHA]
        if not candidates: raise FileNotFoundError('No completed dataset found; pass --dataset-dir /path/to/folder/containing/dataset.json')
        data_root=candidates[0].parent
    dataset,groups=check_dataset(data_root)
    print(f'DATA OK: {data_root}\ntrain=67 development_val=9 cuff_boxes=98; no split overlap',flush=True)
    if args.check_only: return
    if sha(args.checkpoint)!=BASE_SHA: raise ValueError('Original V5 checkpoint hash mismatch; will not initialize another model')
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from fashion_multimodal_analysis.segmentation.modeling import build_evaluation_model
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable: run in the GPU py312 environment')
    seed=20261009;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device('cuda')
    stamp=time.strftime('%Y%m%d_%H%M%S')+f'_{time.time_ns()%1000000:06d}'
    out=root/'reports/reruns/recovery_20261009'/f'cuff_expanded_train_v4_roiadapt_{stamp}'
    out.mkdir(parents=True,exist_ok=False)
    config={'task':'target_garment_conditioned_cuff_boxes','dataset_dir':str(data_root),'manifest_sha256':MANIFEST_SHA,'base_checkpoint':str(args.checkpoint.resolve()),'base_checkpoint_sha256':BASE_SHA,'dataset_json_sha256':sha(data_root/'dataset.json'),'epochs':args.epochs,'seed':seed,'input':'RGB + binary target garment mask; full RGB context retained','context_padding_fraction':.1,'min_size':640,'max_size':1024,'batch_size':1,'workers':0,'confidence_threshold':.4,'matching_iou':.5,'optimizer':'SGD lr=.005 momentum=.9 weight_decay=.0005','lr_milestones':[25,35],'roi_batch_size_per_image':64,'roi_positive_fraction':.5,'trainable_resnet':['conv1','layer3','layer4'],'variant':'v3 expanded training; train-only horizontal flips and mild RGB brightness/contrast; unchanged development split','augmentation_enabled':not args.no_augmentation,'roi_adaptation':dataset.get('roi_adaptation'), 'training_variant':'v4 native/predicted training-view mixture; validation native inputs unchanged','selection':'highest development_val F1; ties retain earlier epoch','limitations':dataset.get('limitations'),'role':'controlled pilot using native target garment ROI/mask; not end-to-end prediction or blind PRD acceptance'}
    write_json(out/'run_config.json',config)
    model=build_evaluation_model(640,1024)
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(cp['model_state_dict'],strict=True);del cp
    model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,2)
    model.roi_heads.mask_roi_pool=None; model.roi_heads.mask_head=None; model.roi_heads.mask_predictor=None
    old=model.backbone.body.conv1
    stem=torch.nn.Conv2d(4,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
    with torch.no_grad(): stem.weight[:,:3].copy_(old.weight);stem.weight[:,3:].zero_()
    model.backbone.body.conv1=stem
    model.transform.image_mean=[.485,.456,.406,0.];model.transform.image_std=[.229,.224,.225,1.]
    # Reduce background ROI dominance; permit adaptation of late visual features.
    model.roi_heads.fg_bg_sampler.batch_size_per_image=64
    model.roi_heads.fg_bg_sampler.positive_fraction=.5
    for name,p in model.backbone.body.named_parameters():
        p.requires_grad=name.startswith(('conv1.','layer3.','layer4.'))
    model.to(device)
    class CuffDataset(torch.utils.data.Dataset):
        def __init__(self,records,augment=False,mix_views=False): self.records=records;self.augment=augment;self.mix_views=mix_views
        def __len__(self): return len(self.records)
        def __getitem__(self,index):
            r=self.records[index]
            if self.mix_views and 'native_view' in r and random.random()<.5: r=r['native_view']
            rgb=np.asarray(Image.open(data_root/r['image']).convert('RGB'),dtype=np.float32).copy()/255.
            mask=(np.asarray(Image.open(data_root/r['garment_mask']).convert('L'))>0).astype(np.float32)
            image=torch.from_numpy(np.concatenate([rgb,mask[:,:,None]],axis=2)).permute(2,0,1).contiguous()
            boxes=torch.tensor(r['cuff_boxes_xyxy_local'],dtype=torch.float32).reshape(-1,4)
            if self.augment:
                if random.random()<.5:
                    image=torch.flip(image,dims=[2]);boxes=torch.tensor(flip_boxes(boxes.tolist(),image.shape[2]),dtype=torch.float32).reshape(-1,4)
                brightness=random.uniform(.85,1.15);contrast=random.uniform(.85,1.15)
                rgb=image[:3];mean=rgb.mean(dim=(1,2),keepdim=True)
                image[:3]=((rgb-mean)*contrast+mean).mul(brightness).clamp(0,1)
            target={'boxes':boxes,'labels':torch.ones(len(boxes),dtype=torch.int64),'image_id':torch.tensor([index]),'area':(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]),'iscrowd':torch.zeros(len(boxes),dtype=torch.int64)}
            return image,target
    trainset=CuffDataset(groups['train'],augment=not args.no_augmentation,mix_views=True);train_evalset=CuffDataset(groups['train']);valset=CuffDataset(groups['development_val'])
    loader=torch.utils.data.DataLoader(trainset,batch_size=1,shuffle=True,num_workers=0,collate_fn=lambda b:tuple(zip(*b)))
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.005,momentum=.9,weight_decay=.0005)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[25,35],gamma=.1)
    @torch.inference_mode()
    def evaluate(epoch,draw=False,split='development_val'):
        evalset=valset if split=='development_val' else train_evalset
        model.eval();cases=[]
        if draw: (out/'best_overlays').mkdir(exist_ok=True)
        for index,r in enumerate(groups[split]):
            im,target=evalset[index]; pred=model([im.to(device)])[0]
            keep=(pred['scores']>=.4)&(pred['labels']==1)
            boxes=pred['boxes'][keep].cpu().tolist();scores=pred['scores'][keep].cpu().tolist();truth=target['boxes'].tolist()
            result=match_boxes(boxes,truth)
            case={'record_id':r['record_id'],'presence':r['presence'],'truth_boxes':truth,'predicted_boxes':boxes,'scores':scores,**result};cases.append(case)
            if draw:
                picture=Image.open(data_root/r['image']).convert('RGB');pen=ImageDraw.Draw(picture)
                for box in truth: pen.rectangle(box,outline='lime',width=3)
                for box,score in zip(boxes,scores): pen.rectangle(box,outline='red',width=3);pen.text((box[0],box[1]),f'{score:.2f}',fill='red')
                picture.save(out/'best_overlays'/f'{r["record_id"]}.jpg')
        return {'epoch':epoch,'metrics':summarize(cases),'cases':cases,'overlay_legend':'green=confirmed cuff; red=prediction at score >= .4'}
    initial=evaluate(0);write_json(out/'initial_untrained_cuff_head.json',initial)
    history=[];best=-1.;best_epoch=None;started=time.time()
    try:
        for epoch in range(1,args.epochs+1):
            model.train();total=0.;components={}
            for images,targets in loader:
                losses=model([im.to(device)for im in images],[{k:v.to(device)for k,v in t.items()}for t in targets]);loss=sum(losses.values())
                for k,v in losses.items(): components[k]=components.get(k,0.)+v.item()
                if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at epoch {epoch}')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters()if p.requires_grad],10.);optimizer.step();total+=loss.item()
            report=evaluate(epoch);metrics=report['metrics'];row={'epoch':epoch,'avg_loss':total/len(loader),'lr':optimizer.param_groups[0]['lr'],**metrics};row['loss_components']={k:v/len(loader)for k,v in components.items()}
            if epoch==1 or epoch%5==0 or epoch==args.epochs:
                fit=evaluate(epoch,split='train');row['train_fit_metrics']=fit['metrics'];write_json(out/f'train_fit_epoch_{epoch:02d}.json',fit)
                print(f'TRAIN FIT epoch {epoch}: {fit["metrics"]}',flush=True)
            history.append(row)
            scheduler.step()
            write_json(out/'history.json',history)
            write_json(out/f'val_epoch_{epoch:02d}.json',report)
            save={'model_state_dict':model.state_dict(),'epoch':epoch,'config':config,'metrics':metrics}
            if metrics['f1']>best:
                best=metrics['f1'];best_epoch=epoch;torch.save(save,out/'checkpoint_best.pth');write_json(out/'best_validation.json',evaluate(epoch,draw=True))
            torch.save(save,out/'checkpoint_last.pth')
            print(f'epoch {epoch:02d}/{args.epochs} loss={row["avg_loss"]:.4f} val TP={metrics["tp"]} FP={metrics["fp"]} FN={metrics["fn"]} P={metrics["precision"]:.3f} R={metrics["recall"]:.3f} F1={metrics["f1"]:.3f}',flush=True)
        status='PILOT_TRAINING_COMPLETED'
    except KeyboardInterrupt:
        status='INTERRUPTED';print('Interrupted: completed-epoch checkpoints retained.',flush=True)
    summary={'status':status,'completed_epochs':len(history),'best_epoch':best_epoch,'best_development_f1':best if best_epoch else None,'elapsed_minutes':(time.time()-started)/60,'output_dir':str(out),'formal_prd_acceptance':False,'weight_files':{p.name:{'path':str(p),'sha256':sha(p)}for p in out.glob('*.pth')},'limitations':config['role'],'result_archive_excludes_weights':True}
    write_json(out/'summary.json',summary)
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED)as z:
        for p in sorted(out.rglob('*')):
            if p.is_file()and p.suffix!='.pth': z.write(p,p.relative_to(out))
    print(f'TRAINING ZIP: {archive}\nWEIGHTS kept on GPU: {out}\nDevelopment metrics only; original V5 files unchanged.',flush=True)
    return out

if __name__=='__main__': main()
