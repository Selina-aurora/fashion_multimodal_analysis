#!/usr/bin/env python3
"""Run frozen V2 pose/segmentation + immutable V3 DEV_V1 proposal on new images.

No V3 GT loaded and no thresholds tuned. Output directory and original project are
kept separate. Predictions cover every segmentation instance and both arms.
"""
import argparse,collections,csv,hashlib,json,shutil,subprocess,sys,traceback,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'legacy_v2'))
from geometry import infer_instances
import run_full as v2
from geometry_v3_proposals import propose_endpoints

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def null(reason):return {'status':'abstain','reason':reason,'A_outer_px':None,'B_inner_px':None,'midpoint_px':None,'metrics':{}}
def check_cohort(cohort):
 mf=cohort/'holdout_images_FROZEN.json';prov=json.loads((cohort/'cohort_provenance.json').read_text())
 if sha(mf)!=prov['selected_file_manifest_sha256']:raise ValueError('Cohort manifest modified!')
 if sha(HERE/'geometry_v3_proposals.py')!=prov['sampling_inference_rule_sha256']:raise ValueError('V3 code modified since cohort freeze!')
 images=json.loads(mf.read_text())
 if len(images)!=prov['selected_n'] or len({x['sha256'] for x in images})!=len(images):raise ValueError('Cohort count/uniqueness changed')
 for x in images:
  im=Path(x['path'])
  if sha(im)!=x['sha256']:raise ValueError('Image changed: '+str(im))
 return images,prov

def build_v3(v2pred, pose_rows, mask_dir):
 poses={p['image_id']:p for p in pose_rows}; outputs=[]
 for image in v2pred:
  pose=poses[image['image_id']]
  byid={p['person_id']:p for p in pose['people']}
  items=[]
  for ins in image['instances']:
   arms=[];pid=ins['predicted_person_id']; person=byid.get(pid)
   for arm in ins['arms']:
    if person is None:pro=null('no_person_association')
    else:
     xy=np.asarray(person['xy'],float); conf=np.asarray(person['conf'],float)
     if xy.shape[0]<7 or conf.shape[0]<7 or min(conf[5:7])<.25 or not np.isfinite(xy[5:7]).all():
      pro=null('unreliable_torso_center')
     else:
      mask=np.asarray(Image.open(mask_dir/ins['mask_file']).convert('L'))>0
      pro=propose_endpoints(mask,arm,float(xy[5:7,0].mean()))
    arms.append({'anatomical_side':arm['anatomical_side'],'image_side':arm['image_side'],
                 'v2_cuff_px':arm['auto_cuff'],'v2_ratio':arm['ratio_pred_or_null'],'v3_proposal':pro})
   items.append({'instance_id':ins['instance_id'],'category':ins['category'],
    'predicted_person_id':ins['predicted_person_id'],'mask_file':ins['mask_file'],
    'bbox':ins['bbox'],'arms':arms})
  outputs.append({'image_id':image['image_id'],'size':image['size'],'instances':items})
 return outputs

def contact(images,indir,output,overlay=None):
 tiles=[]
 pred={p['image_id']:p for p in overlay} if overlay else {}
 for x in images:
  orig=Image.open(indir/x['source_name']).convert('RGB')
  if x['image_id'] in pred:
   d=ImageDraw.Draw(orig)
   for ins in pred[x['image_id']]['instances']:
    bbox=ins['bbox'];d.rectangle(bbox,outline='orange',width=2)
    for ar in ins['arms']:
     pro=ar['v3_proposal'];p=pro.get('midpoint_px')
     if p:
      px,py=p;d.ellipse((px-5,py-5,px+5,py+5),fill='yellow')
     elif pro.get('A_outer_px'):
      px,py=pro['A_outer_px'];d.ellipse((px-5,py-5,px+5,py+5),fill='red')
  im=ImageOps.contain(orig,(210,215))
  t=Image.new('RGB',(230,245),'white');t.paste(im,((230-im.width)//2,25))
  ImageDraw.Draw(t).text((8,7),x['image_id'],fill='black');tiles.append(t)
 cols=5;rows=(len(tiles)+cols-1)//cols
 board=Image.new('RGB',(230*cols,245*rows),'white')
 for i,t in enumerate(tiles):board.paste(t,((i%cols)*230,(i//cols)*245))
 board.save(output,quality=85)

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--root',default='/workspace/fashion_multimodal_analysis')
 ap.add_argument('--out-dir',default='/workspace/sleeve_v3_holdout20_20261013')
 ap.add_argument('--pose-python',default='/workspace/sleeve_pose_sandbox/.venv/bin/python')
 ap.add_argument('--pose-weights',default='/workspace/sleeve_pose_sandbox/yolo11n-pose.pt')
 ap.add_argument('--checkpoint',default='outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth')
 a=ap.parse_args();r=Path(a.root).resolve();cohort=Path(a.out_dir).resolve()
 images,prov=check_cohort(cohort)
 # Guard against rerunning in place; preserve all prior outputs on failure.
 results=cohort/'predictions_frozen'
 if results.exists():raise FileExistsError('Predictions already exist: '+str(results))
 cp=Path(a.checkpoint)
 if not cp.is_absolute():cp=r/cp
 for f in [a.pose_python,a.pose_weights,str(cp)]:
  if not Path(f).is_file():raise FileNotFoundError('Required model/environment file not found: '+f)
 for name,want in v2.EXPECTED.items():
  if sha(r/name)!=want:raise ValueError('Original frozen manifest changed: '+name)
 results.mkdir(parents=True,exist_ok=False)
 try:
  raw=results/'original_images';raw.mkdir()
  converted=[]
  for x in images:
   # Unique source filename is not guaranteed; copy by holdout ID to avoid collisions.
   ext=Path(x['path']).suffix.lower()
   basename=x['image_id']+ext
   shutil.copyfile(x['path'],raw/basename)
   converted.append({**x,'source_name':basename})
  save(results/'images_inference_only.json',images)
  # YOLO uses its isolated virtualenv (OpenCV/headless dependencies remain isolated).
  with (results/'pose.log').open('w') as logfile:
   subprocess.run([a.pose_python,str(HERE/'legacy_v2/pose_worker.py'),
    '--images',str(results/'images_inference_only.json'),
    '--weights',str(a.pose_weights),'--output',str(results/'pose_all_predictions.json')],
    stdout=logfile,stderr=subprocess.STDOUT,check=True)
  poses=json.loads((results/'pose_all_predictions.json').read_text())
  # Strict reuse of the exact V2 segmentation routine/checkpoint settings.
  args=argparse.Namespace(checkpoint=str(cp))
  segs=v2.segment(r,images,args,results)
  bypose={x['image_id']:x for x in poses}
  v2p=[]
  for seg in segs:
   p=bypose[seg['image_id']]
   inf=infer_instances(seg['instances'],p['people'],p['size'],lambda f:np.asarray(Image.open(results/f))>0)
   v2p.append({'image_id':seg['image_id'],'size':p['size'],'instances':inf})
  save(results/'predictions_v2_ALL.json',v2p)
  v3p=build_v3(v2p,poses,results)
  save(results/'predictions_v3_ALL_INSTANCES_BOTH_ARMS.json',v3p)
  frozen_sha=sha(results/'predictions_v3_ALL_INSTANCES_BOTH_ARMS.json')
  all_arms=[a['v3_proposal'] for p in v3p for i in p['instances'] for a in i['arms']]
  summary={'cohort_images':len(images),'predicted_clothing_instances':sum(len(p['instances']) for p in v3p),
   'all_predicted_arms':len(all_arms),'proposal_statuses':dict(collections.Counter(x['status'] for x in all_arms)),
   'image_count_with_any_full_proposal':sum(any(ar['v3_proposal']['status']=='proposal_requires_review' for i in p['instances'] for ar in i['arms']) for p in v3p),
   'rules_frozen_sha256':sha(HERE/'geometry_v3_proposals.py'),
   'predictions_frozen_sha256':frozen_sha,
   'important':'No annotations used or produced. This is NOT an endpoint accuracy metric or PRD validation pass.'}
  save(results/'blind_prediction_summary.json',summary)
  contact(converted,raw,results/'ORIGINALS_blind_review_contact.jpg')
  contact(converted,raw,results/'PREDICTIONS_after_freeze_contact.jpg',v3p)
  checks={x['image_id']:{'original_sha256':x['sha256'],'copied_sha256':sha(raw/x['source_name'])} for x in converted}
  if any(v['original_sha256']!=v['copied_sha256'] for v in checks.values()):raise RuntimeError('Review images differ from source images!')
  save(results/'frozen_provenance.json',{'selection_manifest_sha256':sha(cohort/'holdout_images_FROZEN.json'),
   'train_val_core_csv_sha256':v2.EXPECTED,'pose_weight_sha256':sha(a.pose_weights),'seg_checkpoint_sha256':sha(cp),
   'v3_code_sha256':sha(HERE/'geometry_v3_proposals.py'),'predictions_sha256':frozen_sha,
   'n_images':len(images),'no_gt_used':True,'originals_checked':checks,'model_retrained':False})
  # Pack separately: a concise machine-readable result and full original images.
  for name, include_original in [('UPLOAD_01_PREDICTIONS.zip',False),('UPLOAD_02_ORIGINALS.zip',True)]:
   dest=cohort/name
   with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    if include_original:
     for f in sorted(raw.iterdir()):z.write(f,'original_images/'+f.name)
     for f in [results/'ORIGINALS_blind_review_contact.jpg',cohort/'holdout_images_FROZEN.json',cohort/'cohort_provenance.json']:
      z.write(f,f.name)
    else:
     for f in sorted(results.rglob('*')):
      if f.is_file() and raw not in f.parents and f.name!='ORIGINALS_blind_review_contact.jpg':
       z.write(f,f.relative_to(results))
     for f in [cohort/'holdout_images_FROZEN.json',cohort/'cohort_provenance.json']:
      z.write(f,f.name)
  print('FROZEN_PREDICTIONS_SUCCESS',json.dumps(summary,ensure_ascii=False))
  print('UPLOAD_THIS_ZIP:',cohort/'UPLOAD_01_PREDICTIONS.zip')
  print('UPLOAD_THIS_ZIP:',cohort/'UPLOAD_02_ORIGINALS.zip')
 except Exception:
  (results/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
  print((results/'ERROR.txt').read_text(),file=sys.stderr)
  raise
if __name__=='__main__':main()
