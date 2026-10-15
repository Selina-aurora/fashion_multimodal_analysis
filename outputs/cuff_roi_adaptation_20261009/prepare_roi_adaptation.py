"""Prepare train-side V5 predicted inputs, preserving labels and validation inputs."""
import json,math,shutil,time,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'outputs/cuff_expanded_train_20261009'))
from train_cuff_expanded_v3 import check_dataset,sha,write_json,iou,BASE_SHA

def main():
    import numpy as np
    import torch
    from PIL import Image
    from fashion_multimodal_analysis.common.paths import resolve_path
    from fashion_multimodal_analysis.common.schema import CLASS_TO_ID
    from fashion_multimodal_analysis.segmentation.modeling import build_evaluation_model
    runs=sorted(Path('reports/reruns/recovery_20261009').glob('cuff_expanded_train_v3_*/run_config.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    if not runs:raise FileNotFoundError('Original V3 run not found')
    config=json.loads(runs[0].read_text());source_data=Path(config['dataset_dir']);d,groups=check_dataset(source_data)
    if sha(source_data/'dataset.json')!=config['dataset_json_sha256']:raise ValueError('Source dataset changed')
    manifest=json.loads((source_data/'cuff_pilot_manifest.json').read_text());cases={c['record_id']:c for c in manifest['cases']}
    base=Path(config['base_checkpoint'])
    if sha(base)!=BASE_SHA:raise ValueError('Original V5 checkpoint changed')
    if not torch.cuda.is_available():raise RuntimeError('Run on GPU')
    model=build_evaluation_model(640,1024);cp=torch.load(base,map_location='cpu',weights_only=False);model.load_state_dict(cp['model_state_dict'],strict=True);del cp;model.to('cuda').eval()
    out=source_data.parent/f'cuff_roi_adapt_data_{time.strftime("%Y%m%d_%H%M%S")}_{time.time_ns()%1000000:06d}'
    shutil.copytree(source_data,out);audits=[];adapted=0
    with torch.inference_mode():
        for r in d['records']:
            if r['split']!='train':continue
            native=dict(r);c=cases[r['record_id']];source=resolve_path(c['source_image_reference'])
            if sha(source)!=c['source_image_sha256']:raise ValueError('Source image changed')
            image=Image.open(source).convert('RGB');w,h=image.size;rgb=np.asarray(image,dtype=np.float32).copy()/255.
            pred=model([torch.from_numpy(rgb).permute(2,0,1).contiguous().to('cuda')])[0]
            ids=[i for i,s in enumerate(pred['scores'].cpu().tolist())if s>=.4];bb=pred['boxes'].cpu().tolist();chosen=max(ids,key=lambda i:iou(bb[i],c['garment_bbox']))if ids else None
            reason='NO_TARGET_PROPOSAL';bounds=None
            if chosen is not None:
                target=bb[chosen];reason='LOW_GARMENT_IOU'if iou(target,c['garment_bbox'])<.5 else 'WRONG_GARMENT_CLASS'if int(pred['labels'][chosen])!=CLASS_TO_ID[c['category']]else 'READY'
                if reason=='READY':
                    x,y,xx,yy=target;dx=.1*(xx-x);dy=.1*(yy-y);bounds=[max(0,math.floor(x-dx)),max(0,math.floor(y-dy)),min(w,math.ceil(xx+dx)),min(h,math.ceil(yy+dy))];x,y,xx,yy=bounds
                    # Labels check coverage only; they do not determine crop size or origin.
                    if not all(x<=q[0]<q[2]<=xx and y<=q[1]<q[3]<=yy for q in c['cuff_boxes']):reason='PART_BOX_OUTSIDE_FIXED_PREDICTED_CROP'
                    mask=(pred['masks'][chosen,0,y:yy,x:xx].cpu().numpy()>=.5).astype('uint8')
                    if not mask.any():reason='EMPTY_PREDICTED_GARMENT_MASK'
                    if reason=='READY':
                        folder=out/'train_predicted'/r['record_id'];folder.mkdir(parents=True);image.crop((x,y,xx,yy)).save(folder/'image.png');Image.fromarray(mask*255).save(folder/'garment_mask.png')
                        local=[[q[0]-x,q[1]-y,q[2]-x,q[3]-y]for q in c['cuff_boxes']]
                        audit=[]
                        for q,b in zip(c['cuff_boxes'],local):
                            a,b1,cc,dd=map(int,b);audit.append({'source_box':q,'local_box':b,'source_box_area_retained':1.,'native_garment_mask_fraction_in_box':float(mask[b1:dd,a:cc].mean()),'mask_fraction_role':'predicted_mask_diagnostic'})
                        r.update({'image':str((folder/'image.png').relative_to(out)),'garment_mask':str((folder/'garment_mask.png').relative_to(out)),'image_sha256':sha(folder/'image.png'),'garment_mask_sha256':sha(folder/'garment_mask.png'),'input_crop_xyxy':bounds,'crop_size':[xx-x,yy-y],'cuff_boxes_xyxy_local':local,'box_audit':audit,'native_view':native,'training_input_role':'mixed_native_and_V5_predicted_views'});adapted+=1
            audits.append({'record_id':r['record_id'],'status':reason,'predicted_crop_xyxy':bounds,'fallback_native':reason!='READY'});print(f'{len(audits)}/67 {r["record_id"]}: {reason}',flush=True)
    provenance={'adapted_training_images':adapted,'native_only_fallback_training_images':67-adapted,'train_view_mixing_probability':.5,'source_dataset_json_sha256':sha(source_data/'dataset.json'),'base_segmenter_sha256':BASE_SHA,'validation_native_views_unchanged':True,'target_association':'ground truth associates training garment only; predicted box determines crop','fallback_policy':'unmatched class/box, incomplete part coverage, empty mask: keep native training view without changing labels','audit':audits}
    if not adapted:raise RuntimeError('No suitable predicted training views; will not train duplicate experiment')
    d['roi_adaptation']=provenance;write_json(out/'dataset.json',d);write_json(out/'roi_adaptation_audit.json',provenance)
    check_dataset(out)
    assert [r for r in d['records']if r['split']=='development_val']==groups['development_val']
    print(f'ADAPTED TRAIN IMAGES: {adapted}/67; original development inputs unchanged',flush=True)
    return out
