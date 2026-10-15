#!/usr/bin/env python3
"""Export HUMAN REVIEW previews for Stage14. Never auto-label GT.

Always inspect original imagery before deciding whether a region is present.
Stage14's tolerance parity is NOT semantic accuracy.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys,traceback,zipfile
from pathlib import Path
from collections import Counter
BASE=Path(__file__).resolve().parent
ROOT_DEFAULT=Path('/workspace/fashion_multimodal_analysis')
OUTROOT=Path('/workspace/prd31_stage15_review_exports_20261014')
sys.path.insert(0,str(BASE))
import stage6_helpers as h
import run_stage14 as stage14
import strict_batch_engine as engine

ROOTREF=BASE.parent/'reference'
FIELDS=[
 'case','region','source_image','source_sha256','predicted_garment_category',
 'prediction_status','stage14_prediction_status','status_matches_stage14',
 'prediction_box_original_xyxy','predicted_mask_area_px','model_score',
 'original_image_preview','garment_plain_preview','prediction_overlay_preview',
 'human_region_presence_yes_no_uncertain',
 'human_localization_correct_coarse_wrong_missed_uncertain_not_applicable',
 'human_bbox_xyxy_if_drawn','human_mask_annotation_path_if_drawn',
 'human_reviewer','human_review_notes',
]

def load_reference():
 expected=json.loads((ROOTREF/'selection.json').read_text(encoding='utf-8'))['cases']
 with (ROOTREF/'final_region_parity.csv').open(encoding='utf-8-sig',newline='') as f:
  statuses={(x['case'],x['region']):x['production_status'] for x in csv.DictReader(f)}
 selection=json.loads((ROOTREF/'stage14_selection_audit.json').read_text(encoding='utf-8'))
 return expected,statuses,selection

def save_jpeg(img,path,limit=1100):
 p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
 result=img.convert('RGB').copy();result.thumbnail((limit,limit))
 result.save(p,'JPEG',quality=82,optimize=True)

def draw_preview(c,pred,region):
 from PIL import Image,ImageDraw
 import numpy as np
 crop=c['crop'].convert('RGB').copy()
 overlay=np.asarray(crop).copy()
 if pred.get('status')=='detected' and pred.get('mask') is not None:
  raw=np.asarray(pred['mask'])
  if raw.shape!= (c['image'].height,c['image'].width):
   raise ValueError(f'Mask size mismatch {c["case"]}/{region}: {raw.shape}')
  x1,y1,x2,y2=c['origin'];inner=raw[y1:y2,x1:x2]>0
  if inner.shape != overlay.shape[:2]:raise ValueError('Crop/mask coordinate mismatch')
  overlay=overlay.astype('float32')
  overlay[inner]=0.55*overlay[inner]+0.45*np.array([40,210,115],dtype='float32')
  overlay=np.clip(overlay,0,255).astype('uint8')
 im=Image.fromarray(overlay,'RGB');dr=ImageDraw.Draw(im)
 if pred.get('bbox') is not None:
  x1,y1,x2,y2=c['origin'];bb=[float(v) for v in pred['bbox']]
  dr.rectangle((bb[0]-x1,bb[1]-y1,bb[2]-x1,bb[3]-y1),outline=(255,58,50),width=max(2,im.width//170))
 dr.rectangle((0,0,min(im.width,310),min(im.height,33)),fill=(10,10,10))
 dr.text((5,5),f'{region} | {pred.get("status","unknown")}',fill='white')
 return im

def archive(out):
 dest=out.parent/f'prd31_stage15_{out.name}_UPLOAD_THIS.zip'
 with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(out.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(out))
 with zipfile.ZipFile(dest) as z:
  if z.testzip() is not None:raise ValueError('Invalid output ZIP')
 return dest

def run(args):
 root=args.root.resolve();outroot=args.out_root.resolve()
 if root==outroot or root in outroot.parents or outroot==root.parent:
  raise ValueError('Output directory must be outside project repository')
 out=h.new_out(outroot)
 audit={'status':'STARTED','project_files_modified':False,'formal_semantic_accuracy':False}
 rc=0
 try:
  os.environ.setdefault('HF_HOME','/workspace/hf_cache')
  os.environ.setdefault('HF_HUB_CACHE','/workspace/hf_cache/hub')
  os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
  sys.path.insert(0,str(root/'src'))
  cfg,previous=h.check_inputs(root,out)
  if not cfg.get('strict_region_grounding'):raise RuntimeError('Strict grounding disabled')
  candidates,selection_audit=stage14.build_selection(root,previous,12)
  expected,statuses,reference_audit=load_reference()
  if len(expected)!=12:raise ValueError('Reference count not 12')
  for exp in expected:
   case_id=int(exp['id'].split('_')[-1]);candidate=candidates[case_id-1]
   s=h.filehash(candidate['source_image'])
   if s!=candidate['source_sha256']:raise RuntimeError('Input image mutated')
   # GPU Stage14 selection hashes must match the ones preserved in its output.
   old=reference_audit['candidate_image_sha256'][case_id-1]
   if s!=old['source_sha256'] or candidate['source_image']!=old['source_image']:
    raise RuntimeError(f'Stage14 image changed: {exp["id"]}')
  audit['selection']={'reference_cases':len(expected),'candidate_count':len(candidates),
                      'unique_val_images':selection_audit['unique_val_images'],
                      'path_exclusion_only':True}
  if args.dry_run:
   audit['status']='DRY_RUN_PASS'
  else:
   from fashion_multimodal_analysis.integration.model_backends import ModelBackends
   backend=ModelBackends(cfg,device='cuda')
   cases,skipped=engine.choose_cases(backend,candidates,12,4)
   if skipped:raise RuntimeError('Stage14 selection changed: skipped '+str(skipped))
   if len(cases)!=len(expected):raise RuntimeError('Stage14 model ROI count changed')
   rows=[];mismatches=[]
   for c,exp in zip(cases,expected):
    if c['case']!=exp['id'] or c['roi'].category!=exp['class'] or list(c['crop'].size)!=exp['crop_size'] or c['regions']!=exp['regions']:
     raise RuntimeError(f'Stage14 reproduced ROI does not match: {exp["id"]}')
    num=int(c['case'].split('_')[-1]);src=candidates[num-1]
    im_id=c['case'];imgdir=out/'images'
    save_jpeg(c['image'],imgdir/f'{im_id}_original.jpg',limit=1300)
    save_jpeg(c['crop'],imgdir/f'{im_id}_garment_plain.jpg',limit=1000)
    for region in c['regions']:
     ans,lat_ms=backend.ground(c['image'],c['roi'],region)
     status=ans.get('status','unknown');expected_status=statuses[(im_id,region)]
     matched=status==expected_status
     if not matched:mismatches.append(f'{im_id}/{region}: stage14 {expected_status} != replay {status}')
     prev=f'images/{im_id}_{region}_prediction.jpg'
     save_jpeg(draw_preview(c,ans,region),out/prev,limit=1000)
     import numpy as np
     binary=(np.asarray(ans.get('mask'))>0) if ans.get('mask') is not None else None
     box=ans.get('bbox')
     rows.append({'case':im_id,'region':region,'source_image':src['source_image'],
                  'source_sha256':src['source_sha256'],'predicted_garment_category':c['roi'].category,
                  'prediction_status':status,'stage14_prediction_status':expected_status,
                  'status_matches_stage14':matched,
                  'prediction_box_original_xyxy':json.dumps(box,ensure_ascii=False) if box is not None else '',
                  'predicted_mask_area_px':int(binary.sum()) if binary is not None else '',
                  'model_score':ans.get('score',''),
                  'original_image_preview':f'images/{im_id}_original.jpg',
                  'garment_plain_preview':f'images/{im_id}_garment_plain.jpg',
                  'prediction_overlay_preview':prev,
                  'human_region_presence_yes_no_uncertain':'',
                  'human_localization_correct_coarse_wrong_missed_uncertain_not_applicable':'',
                  'human_bbox_xyxy_if_drawn':'','human_mask_annotation_path_if_drawn':'',
                  'human_reviewer':'','human_review_notes':''})
    print('EXPORTED',im_id,c['roi'].category,flush=True)
   h.write_csv(out/'review_manifest_FILL_MANUALLY.csv',rows,FIELDS)
   if len(rows)!=48:raise RuntimeError(f'Unexpected number of tasks: {len(rows)}')
   h.write_json(out/'replay_mismatches.json',{'mismatches':mismatches})
   audit.update({'status':'STAGE15_REVIEW_ASSETS_EXPORTED','cases':len(cases),
                 'region_review_tasks':len(rows),'model_status_distribution':dict(Counter(x['prediction_status'] for x in rows)),
                 'replay_status_matches_stage14':len(rows)-len(mismatches),
                 'replay_status_mismatches':len(mismatches),
                 'human_reviews_completed':0,
                 'accuracy_acceptance':False,
                 'note':'Model predictions are not human GT. Input selection is validation-manifest/development-only and only top/dress four region types.'})
   (out/'HOW_TO_REVIEW.md').write_text("""# Manual review guidelines\n\n1. Open each `*_garment_plain.jpg` together with `*_original.jpg`. Use the prediction overlay only AFTER deciding which physical region should be present.\n2. For each row of `review_manifest_FILL_MANUALLY.csv`, fill `human_region_presence_yes_no_uncertain`, then `human_localization_correct_coarse_wrong_missed_uncertain_not_applicable`, and notes. Do not auto-approve because the A/B model outputs agreed.\n3. If a region is absent in reality, mark presence=no and localization=not_applicable, even if the model says detected; separately describe the false positive.\n4. If the region is invisible, occluded, too small or semantically unclear, mark uncertain; do not guess coordinates.\n5. Where confident, annotate the human region bbox in original image coordinates using `human_bbox_xyxy_if_drawn` and optionally a human mask file; this script DOES NOT produce that GT.\n6. Only after human labels and case definitions have been reviewed should any localization accuracy metric be computed. This task is not representative of all eight garment classes or all eight local regions.\n7. If `replay_status_matches_stage14` is not 48/48, inspect `replay_mismatches.json` before comparing against Stage14 results.\n""",encoding='utf-8')
 except Exception as e:
  audit['status']='STAGE15_FAILED';audit['error']=repr(e)
  (out/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');print('ERROR',repr(e),flush=True);rc=2
 finally:
  h.write_json(out/'run_audit.json',audit)
  dest=archive(out);print('STATUS:',audit['status'],flush=True)
  print('UPLOAD_THIS_ZIP:',dest,flush=True)
 return rc

def main():
 p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true')
 p.add_argument('--root',type=Path,default=ROOT_DEFAULT)
 p.add_argument('--out-root',type=Path,default=OUTROOT)
 raise SystemExit(run(p.parse_args()))
if __name__=='__main__':main()
