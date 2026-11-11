#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only DEV60 mask-to-V5 bbox pixel-level geometry checks.
Run on GPU host holding the original 171 PNGs; NEVER reads holdout paths.
Does NOT infer garment semantic identity and does NOT edit any files.
Requires pillow; writes its own CSV/JSON only.
"""
import argparse, csv, json, math
from pathlib import Path
try:from PIL import Image
except ImportError:Image=None
ROOT=Path(__file__).resolve().parent

def safe_dev_path(path):
 p=str(path).lower().replace('\\','/')
 return not any(token in p for token in ('holdout','sealed','w7_holdout','blind_test','test_set','/test/'))

def inspect_mask_record(im,pred,mask_root,margin=3.0):
 if Image is None:raise RuntimeError('Pillow not installed')
 cid=im['case_id'];iid=pred['instance_id'];source=pred['mask_path']
 if not safe_dev_path(source):raise ValueError('refuse holdout mask path')
 if Path(source).name!=f'{cid}_{iid}_mask.png':
  return {'case_id':cid,'instance_id':iid,'status':'MASK_FILENAME_CONTRACT_FAIL','mask_path':source}
 # Use basename only, avoiding any accidental traversal from a stored path.
 mask_path=Path(mask_root)/Path(source).name
 if not mask_path.is_file():return {'case_id':cid,'instance_id':iid,'status':'MASK_NOT_FOUND','mask_path':str(mask_path)}
 w,h=map(int,im['image_size']);bbox=[float(v) for v in pred['bbox_xyxy']]
 if not 0<=bbox[0]<bbox[2]<=w+1 or not 0<=bbox[1]<bbox[3]<=h+1:
  return {'case_id':cid,'instance_id':iid,'status':'INVALID_V5_BBOX','mask_path':str(mask_path)}
 with Image.open(mask_path) as src:
  if src.size!=(w,h):
   return {'case_id':cid,'instance_id':iid,'status':'MASK_DIMENSION_MISMATCH','mask_path':str(mask_path),'mask_size':str(src.size),'expected_size':str((w,h))}
  gray=src.convert('L');tight=gray.point(lambda x:255 if x>0 else 0).getbbox()
  if tight is None:return {'case_id':cid,'instance_id':iid,'status':'EMPTY_MASK','mask_path':str(mask_path)}
  mbox=tuple(map(float,tight))
  # A mask may be slightly inset relative to model's bbox. Require bbox to enclose mask,
  # NOT exact equality; no semantic or segmentation-quality claim from this test.
  enclosed=(bbox[0]-margin<=mbox[0] and bbox[1]-margin<=mbox[1] and
             bbox[2]+margin>=mbox[2] and bbox[3]+margin>=mbox[3])
  cx=(mbox[0]+mbox[2])/2.;cy=(mbox[1]+mbox[3])/2.
  centered=(bbox[0]-margin<=cx<=bbox[2]+margin and bbox[1]-margin<=cy<=bbox[3]+margin)
  stat='GEOMETRY_COMPATIBLE' if enclosed and centered else 'MASK_BBOX_GEOMETRY_SUSPECT'
  return {'case_id':cid,'instance_id':iid,'status':stat,'mask_path':str(mask_path),
          'mask_tight_bbox':json.dumps(list(tight)),'v5_bbox_xyxy':json.dumps(bbox),
          'mask_inside_bbox':str(enclosed),'mask_center_inside_bbox':str(centered)}

def run(mask_root,manifest=ROOT/'input/v5_raw_predictions.jsonl',output_dir=ROOT/'output/mask_geometry_probe'):
 mask_root=Path(mask_root).resolve(); manifest=Path(manifest).resolve();output_dir=Path(output_dir).resolve()
 if not safe_dev_path(mask_root) or not safe_dev_path(manifest):raise ValueError('holdout-like path prohibited')
 if output_dir==mask_root or output_dir==manifest.parent or mask_root in output_dir.parents or output_dir in mask_root.parents:
  raise ValueError('output path overlaps input')
 output_dir.mkdir(parents=True,exist_ok=True)
 images=[json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines() if s.strip()]
 if len(images)!=60 or any('DEV' not in x.get('scope','').upper() for x in images):raise ValueError('DEV60 cohort only')
 rows=[inspect_mask_record(im,pred,mask_root) for im in images for pred in im['instances']]
 if len(rows)!=171:raise ValueError('DEV60 171-instance signature changed')
 fields=['case_id','instance_id','status','mask_path','mask_tight_bbox','v5_bbox_xyxy','mask_inside_bbox','mask_center_inside_bbox','mask_size','expected_size']
 with (output_dir/'mask_geometry_171.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 from collections import Counter
 st=dict(Counter(r['status'] for r in rows))
 summary={'scope':'DEV60_ONLY_MASK_GEOMETRY_READONLY','mask_root':str(mask_root),'counts':st,'raw_masks_modified':0,'not_a_segmentation_accuracy_metric':True,'holdout_opened':False}
 (output_dir/'mask_geometry_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 return summary
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--mask-root',required=True)
 parser.add_argument('--output-dir',default=str(ROOT/'output/mask_geometry_probe'))
 args=parser.parse_args();print(json.dumps(run(args.mask_root,output_dir=args.output_dir),ensure_ascii=False,indent=2))
