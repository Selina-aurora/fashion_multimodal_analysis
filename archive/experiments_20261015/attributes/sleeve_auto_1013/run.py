"""Oct13 exploratory pipeline: predicted masks + automatic pose, followed by GT evaluation."""
import argparse,csv,hashlib,json,math,subprocess,sys,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from geometry import infer_instances,iou

HERE=Path(__file__).resolve().parent
TRAIN='configs/prd_8class_train_v5_balanced150_20261008.csv'
VAL='configs/prd_8class_val_v2_full_targets.csv'
CORE='benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv'
EXPECTED={TRAIN:'e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f',VAL:'39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93',CORE:'c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89'}
CLASS_TO_ID={'top':1,'pants':2,'skirt':3,'outerwear':4,'dress':5,'shoe':6,'bag':7,'accessory':8}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(p,obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def resolve(root,p): return (root/Path(p)).resolve()

def preflight(root,a,out):
    hashes={}
    for rel,expected in EXPECTED.items():
        path=root/rel
        if not path.is_file(): raise FileNotFoundError('Required frozen file: '+str(path))
        hashes[rel]=sha(path)
        if hashes[rel]!=expected: raise ValueError('Frozen CSV hash mismatch: '+rel)
    images=json.loads((HERE/'images.json').read_text())
    core=set((HERE/'core400_sha256.txt').read_text().split())
    if len(core)!=400: raise ValueError('Core400 hash list is incomplete')
    if len(images)!=8 or len({r['sha256'] for r in images})!=8: raise ValueError('Expected eight unique images')
    for r in images:
        path=resolve(root,r['path'])
        if r['sha256'] in core: raise ValueError('Core400 overlap: '+r['image_id'])
        if not path.is_file(): raise FileNotFoundError(path)
        if sha(path)!=r['sha256']: raise ValueError('Raw image hash mismatch: '+str(path))
        r['path']=str(path)
    for path in (a.pose_python,a.pose_weights,a.checkpoint):
        if not Path(path).is_file(): raise FileNotFoundError(path)
    write(out/'images_inference_only.json',images)
    meta={'purpose':'8-image development feasibility, NOT blind validation','input':'original images only; predicted garment bbox/mask',
          'gt_used_in_inference':False,'GT_matching':'evaluation only after predictions.json saved',
          'frozen_csv_sha256_before':hashes,'pose_weights_sha256':sha(a.pose_weights),'segmentation_checkpoint':a.checkpoint,
          'segmentation_weights_sha256':sha(a.checkpoint),'script_sha256':{p.name:sha(p) for p in HERE.glob('*.py')},
          'parameters':{'seg_score':.4,'mask_threshold':.5,'pose_score':.25,'joint_confidence':.25,'evaluation_bbox_iou':.5},
          'acceptance_thresholds':'NOT AGREED; exploratory metrics only; PCK@0.05 diagonal is a diagnostic'}
    write(out/'run_metadata.json',meta)
    return images,meta

def segment(root,images,a,out):
    import torch
    from torchvision.transforms.functional import to_tensor
    sys.path.insert(0,str(root/'src'))
    from fashion_multimodal_analysis.segmentation.modeling import build_evaluation_model
    from fashion_multimodal_analysis.segmentation.training.checkpoint_state import validate_checkpoint
    # User-owned existing training checkpoint. No downloaded/untrusted pickle is loaded.
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=json.loads((HERE/'reference_config.json').read_text())
    validate_checkpoint(cp,EXPECTED[TRAIN],EXPECTED[VAL],settings=cfg['checkpoint_settings'])
    saved=cp['args']; model=build_evaluation_model(int(saved['min_size']),int(saved['max_size']))
    model.load_state_dict(cp['model_state_dict'],strict=True);del cp
    device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device).eval()
    (out/'masks').mkdir(); allrows=[]
    for image in images:
        im=Image.open(image['path']).convert('RGB')
        with torch.inference_mode(): pred=model([to_tensor(im).to(device)])[0]
        instances=[]
        for j,(box,label,score) in enumerate(zip(pred['boxes'],pred['labels'],pred['scores'])):
            score=float(score);category={v:k for k,v in CLASS_TO_ID.items()}.get(int(label),'unknown')
            if score<.4: continue
            mask=(pred['masks'][j,0].detach().cpu().numpy()>=.5).astype('uint8')*255
            if mask.shape!=(im.height,im.width): raise ValueError('Predicted mask has wrong image scale')
            name=f"masks/{image['image_id']}_{j}.png";Image.fromarray(mask).save(out/name)
            instances.append({'instance_id':j,'bbox':box.detach().cpu().tolist(),'category':category,'score':score,'mask_file':name})
        allrows.append({'image_id':image['image_id'],'instances':instances})
        print('segment',image['image_id'],'instances',len(instances),flush=True)
    del model
    if device=='cuda': torch.cuda.empty_cache()
    write(out/'segmentation_all_predictions.json',allrows)
    return allrows

def evaluate(root,images,predictions,out):
    # This function alone sees the manual side, points, ratio and garment GT box.
    labels=json.loads((HERE/'evaluation_labels.json').read_text())['annotations']
    with (root/TRAIN).open(encoding='utf-8-sig',newline='') as f: source={r['record_id']:r for r in csv.DictReader(f)}
    byid={r['image_id']:r for r in predictions}; paths={r['image_id']:r['path'] for r in images}
    rows=[];(out/'visuals').mkdir()
    for gt in labels:
        rid=gt['record_id']; prediction=byid[rid]; w,h=prediction['size']; diag=math.hypot(w,h)
        target=source[rid];box=[float(target['bbox_'+key]) for key in ('x1','y1','x2','y2')]
        candidates=sorted([(iou(box,p['bbox']),p) for p in prediction['instances']],key=lambda x:x[0],reverse=True)
        row={'record_id':rid,'gt_side':gt['side'],'gt_ratio':gt['ratio'],'matched_instance_id':None,'bbox_iou':None,
             'predicted_category':None,'gt_category':target['garment_category'],'person_id':None,'anatomical_side':None,
             'joint_error_mean_diag':None,'joint_pck_005':None,'cuff_error_px':None,'cuff_error_diag':None,
             'cuff_error_arm':None,'ratio_pred':None,'ratio_abs_error':None,'status':'segmentation_miss'}
        match=None; arm=None
        if candidates and candidates[0][0]>=.5:
            if len(candidates)>1 and candidates[1][0]>=.5 and candidates[0][0]-candidates[1][0]<.05:
                row['status']='ambiguous_instance_evaluation_match'
            else:
                overlap,match=candidates[0];row.update(matched_instance_id=match['instance_id'],bbox_iou=overlap,
                      predicted_category=match['category'],person_id=match['predicted_person_id'],status=match['association_failure'] or 'annotated_side_unavailable')
                arms=[v for v in match['arms'] if v['image_side']==gt['side']]
                if len(arms)==1:
                    arm=arms[0];row['anatomical_side']=arm['anatomical_side'];row['status']=arm['failure_reason'] or 'predicted'
                    errors=[]
                    for name in ('shoulder','elbow','wrist'):
                        truth=np.asarray(gt['points'][name])*[w,h]
                        err=float(np.linalg.norm(np.asarray(arm['points'][name])-truth)/diag)
                        row[name+'_error_diag']=err;errors.append(err)
                    row['joint_error_mean_diag']=float(np.mean(errors));row['joint_pck_005']=float(np.mean(np.asarray(errors)<=.05))
                    if arm['auto_cuff'] is not None:
                        err=float(np.linalg.norm(np.asarray(arm['auto_cuff'])-np.asarray(gt['points']['cuff'])*[w,h]))
                        row.update(cuff_error_px=err,cuff_error_diag=err/diag,cuff_error_arm=err/gt['arm_pixels'],
                                   ratio_pred=arm['ratio_pred_or_null'],ratio_abs_error=abs(arm['ratio_pred_or_null']-gt['ratio']))
        rows.append(row)
        im=Image.open(paths[rid]).convert('RGB');d=ImageDraw.Draw(im)
        for inst in prediction['instances']:
            d.rectangle(inst['bbox'],outline='orange',width=2)
            d.text((inst['bbox'][0],max(0,inst['bbox'][1]-12)),f"pred {inst['instance_id']} {inst['category']}",fill='orange')
            for ar in inst['arms']:
                q=[tuple(ar['points'][k]) for k in ('shoulder','elbow','wrist')]
                d.line(q,fill='cyan',width=3)
                if ar['auto_cuff'] is not None:
                    x,y=ar['auto_cuff'];d.ellipse((x-5,y-5,x+5,y+5),outline='red',width=3)
        # GT is drawn only after predictions are frozen, solely for visual evaluation.
        d.rectangle(box,outline='lime',width=2)
        for name,point in gt['points'].items():
            x,y=np.asarray(point)*[w,h];d.ellipse((x-4,y-4,x+4,y+4),outline='lime',width=2)
        d.text((5,5),'PRED: orange/cyan/red   GT: green   '+row['status'],fill='red')
        im.thumbnail((1200,1200));im.save(out/'visuals'/f'{rid}.jpg')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'per_case.csv').open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(rows)
    valid=[r for r in rows if r['ratio_pred'] is not None]; joints=[r for r in rows if r['joint_error_mean_diag'] is not None]
    def avg(rs,key): return float(np.mean([r[key] for r in rs])) if rs else None
    from collections import Counter
    summary={'n_images':8,'n_gt_annotated_arms':len(rows),'n_bbox_matches':sum(r['matched_instance_id'] is not None for r in rows),
             'n_associated_annotated_arms':len(joints),'n_ratio_outputs':len(valid),'ratio_coverage':len(valid)/len(rows),
             'status_counts':dict(Counter(r['status'] for r in rows)),
             'conditional_joint_mean_error_diag':avg(joints,'joint_error_mean_diag'),'conditional_joint_pck_005_diag':avg(joints,'joint_pck_005'),
             'conditional_cuff_mean_error_px':avg(valid,'cuff_error_px'),'conditional_cuff_mean_error_diag':avg(valid,'cuff_error_diag'),
             'conditional_cuff_mean_error_arm':avg(valid,'cuff_error_arm'),'conditional_ratio_MAE':avg(valid,'ratio_abs_error'),
             'conditional_ratio_RMSE':float(np.sqrt(np.mean([r['ratio_abs_error']**2 for r in valid]))) if valid else None,
             'n_category_errors_on_matched_instances':sum(r['matched_instance_id'] is not None and r['predicted_category']!=r['gt_category'] for r in rows),
             'scope':'8 previously reviewed development examples; no blind-test or teacher acceptance claim',
             'warning':'Report conditional errors WITH coverage. Both arms predicted; manual side is used only for evaluation matching. Person association accuracy still needs per-image review.',
             'gt_labels_sha256':sha(HERE/'evaluation_labels.json'),'predictions_sha256':sha(out/'predictions.json')}
    write(out/'summary.json',summary)
    with (out/'manual_review.csv').open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.writer(f);wr.writerow(['record_id','correct_garment','correct_person','correct_arm','cuff_is_real_sleeve_boundary','failure_stage','notes'])
        for row in rows: wr.writerow([row['record_id'],'','','','','',''])
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/workspace/fashion_multimodal_analysis')
    p.add_argument('--pose-python',default='/workspace/sleeve_pose_sandbox/.venv/bin/python')
    p.add_argument('--pose-weights',default='/workspace/sleeve_pose_sandbox/yolo11n-pose.pt')
    p.add_argument('--checkpoint',default='outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth')
    a=p.parse_args();root=Path(a.root).resolve();a.checkpoint=str(resolve(root,a.checkpoint))
    out=root/'reports/reruns'/('sleeve_auto_1013_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True,exist_ok=False);success=False
    try:
        images,meta=preflight(root,a,out)
        # Neither worker receives evaluation labels or GT boxes/masks.
        with (out/'pose.log').open('w') as log:
            subprocess.run([a.pose_python,str(HERE/'pose_worker.py'),'--images',str(out/'images_inference_only.json'),
                            '--weights',a.pose_weights,'--output',str(out/'pose_all_predictions.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
        poses={r['image_id']:r for r in json.loads((out/'pose_all_predictions.json').read_text())}
        segmentations=segment(root,images,a,out); predictions=[]
        for seg in segmentations:
            pose=poses[seg['image_id']]
            result=infer_instances(seg['instances'],pose['people'],pose['size'],lambda name:np.asarray(Image.open(out/name))>0)
            predictions.append({'image_id':seg['image_id'],'size':pose['size'],'instances':result})
        write(out/'predictions.json',predictions) # Freeze before accessing any manual annotations.
        evaluate(root,images,predictions,out)
        meta['frozen_csv_sha256_after']={rel:sha(root/rel) for rel in EXPECTED}
        if meta['frozen_csv_sha256_after']!=meta['frozen_csv_sha256_before']: raise RuntimeError('Frozen input changed during run')
        meta['completed']=True;write(out/'run_metadata.json',meta);success=True
    except Exception:
        error=traceback.format_exc();(out/'ERROR.txt').write_text(error);print(error,file=sys.stderr)
    finally:
        archive=out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for file in sorted(out.rglob('*')):
                if file.is_file():z.write(file,str(file.relative_to(out)))
        print(('DONE' if success else 'FAILED')+'; please return ZIP: '+str(archive),flush=True)
    return 0 if success else 1
if __name__=='__main__':sys.exit(main())
