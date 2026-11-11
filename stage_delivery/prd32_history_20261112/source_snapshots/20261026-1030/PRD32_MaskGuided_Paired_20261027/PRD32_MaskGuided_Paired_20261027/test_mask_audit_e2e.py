#!/usr/bin/env python3
"""No-GPU end-to-end synthetic provenance test; does NOT reference real user images."""
from __future__ import annotations
import csv, json, tempfile, hashlib
from pathlib import Path
from PIL import Image
import numpy as np
from argparse import Namespace
import prd32_mask_guided_paired as m

def run():
  with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);project=root/'fashion_multimodal_analysis'
    images=root/'fashion_data/raw/train/train/image';images.mkdir(parents=True)
    roidir=root/'roi_run';(roidir/'roi_marked_inputs').mkdir(parents=True)
    maskdir=project/'outputs/garment_instances/gt_pilot';maskdir.mkdir(parents=True)
    manifest=[];plans=[];preds=[];reviews=[]
    bbox=[20.,10.,100.,85.]
    for i in range(15):
      cid=f'train_dev_{i+1:03d}';img=images/f'{i:03d}.jpg'
      Image.new('RGB',(120,100),(195,190,180)).save(img)
      mask=np.zeros((100,120),dtype=np.uint8);mask[15:80,26:93]=255
      mask_path=maskdir/f'{i:03d}_mask.png';Image.fromarray(mask).save(mask_path)
      a,b,c,d=m.crop_rect(bbox,120,100)
      roi=roidir/'roi_marked_inputs'/(cid+'_roi_marked.jpg')
      Image.open(img).crop((a,b,c,d)).save(roi)
      plans.append({'case_id':cid,'image_path':str(img),'image_sha256':m.sha(img),'bbox':bbox,
                    'target_instance_id':f'garment{i}','human_category_FOR_EVAL_ONLY':'pants'})
      preds.append({'case_id':cid,'status':'GENERATED_UNVERIFIED','roi_category':'top','roi_input':str(roi)})
      reviews.append({'case_id':cid,'roi_single_item':'NO','cause_revision':'CROP_MIXED_ITEMS'})
      manifest.append({'record_id':f'rec{i}','garment_id':f'garment{i}',
                      'source_image':str(img.relative_to(project.parent)),'source_dataset':'DeepFashion2',
                      'mask_path':str(mask_path.relative_to(project)),
                      **{f'bbox_{k}':bbox[j] for j,k in enumerate(('x1','y1','x2','y2'))}})
    # Source paths are relative to project, so explicitly ../fashion_data.
    for r in manifest:r['source_image']='../fashion_data/'+r['source_image'].split('fashion_data/',1)[-1]
    m.writejl(roidir/'case_plan.jsonl',plans)
    m.writejl(roidir/'roi_predictions.jsonl',preds)
    review=root/'review.json';m.jwrite(review,{'scope':'train_dev_not_independent','rows':reviews})
    inventory=project/'configs/prd_8class_unified_manifest_v2.csv';inventory.parent.mkdir(parents=True)
    with inventory.open('w',newline='',encoding='utf-8-sig') as f:
      w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
    # One missing instance mask, one wrong instance size should quarantine, other 13 pass.
    (maskdir/'001_mask.png').unlink()
    Image.new('L',(15,15),255).save(maskdir/'002_mask.png')
    output=root/'audit_output'
    args=Namespace(output=str(output),roi_run_dir=str(roidir),project_root=str(project),inventory=str(inventory),
                   allowed_data_root=str(root/'fashion_data'),review=str(review),
                   helper=str(m.HERE/'prd32_integrated_pipeline_20261027.py'),pad=.05)
    result=m.audit(args)
    assert result['source_roi_case_count']==15, result
    assert result['eligible_mask_cases']==13, result
    assert result['quarantined']==2, result
    assert result['failure_codes']['MASK_FILE_NOT_FOUND']==1,result
    assert result['failure_codes']['MASK_DIMENSION_UNSUPPORTED']==1,result
    ready=m.readjl(output/'ready_cases.jsonl'); assert len(ready)==13
    assert all(Path(z['mask_guided_roi']).exists() and len(z['mask_guided_sha256'])==64 for z in ready)
    page=(output/'MASK_GUIDED_VISUAL_CHECK_STATIC.html').read_text()
    assert page.count('<article>')==15
    assert page.count('data:image/jpeg;base64,')==26
    assert m.jread(output/'audit_summary.json')['holdout_used'] is False
    print('MASK_AUDIT_SYNTHETIC_E2E_PASS',{'ready':13,'quarantined':2,'static_cards':15})

if __name__=='__main__':run()
