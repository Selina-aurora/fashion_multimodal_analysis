#!/usr/bin/env python3
"""Synthetic check: never open val/test/holdout image bytes, no GPU."""
from __future__ import annotations
import csv,hashlib,json,tempfile
from pathlib import Path
from source_gap_audit import audit,sha

def write_csv(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 fields=['record_id','source_image','garment_id','garment_category','bbox_x1','bbox_y1','bbox_x2','bbox_y2']
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
 with tempfile.TemporaryDirectory() as d:
  root=Path(d);project=root/'project';data=root/'fashion_data';config=project/'configs';config.mkdir(parents=True)
  train=data/'raw/train/train/image';train.mkdir(parents=True)
  val=data/'raw/val/image';val.mkdir(parents=True)
  for name in ['unused_top','old_skirt','new12_outer','unused_skirt']:
   (train/(name+'.jpg')).write_bytes(('fake_image_'+name).encode())
  (val/'restricted.jpg').write_bytes(b'PROTECTED: MUST NEVER HASH')
  old_real=sha(train/'old_skirt.jpg');n12_real=sha(train/'new12_outer.jpg')
  # Deterministic but complete 40 and 12 hash-registry fixtures
  oldh=[old_real]+[hashlib.sha256(f'old{i}'.encode()).hexdigest() for i in range(39)]
  newh=[n12_real]+[hashlib.sha256(f'new{i}'.encode()).hexdigest() for i in range(11)]
  oldcsv=root/'old40.csv'
  with oldcsv.open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=['image_sha256']);w.writeheader();w.writerows({'image_sha256':x} for x in oldh)
  newjs=root/'new12.json';newjs.write_text(json.dumps({'rows':[{'image_sha256':x} for x in newh]}),encoding='utf-8')
  def rec(n,cat,loc,x2='10'):
   return dict(record_id=n,source_image=loc,garment_id=n,garment_category=cat,bbox_x1='1',bbox_y1='1',bbox_x2=x2,bbox_y2='10')
  rows=[rec('a','top','../fashion_data/raw/train/train/image/unused_top.jpg'),
        rec('b','skirt','../fashion_data/raw/train/train/image/old_skirt.jpg'),
        rec('c','outerwear','../fashion_data/raw/train/train/image/new12_outer.jpg'),
        rec('d','skirt','../fashion_data/raw/train/train/image/unused_skirt.jpg'),
        rec('e','dress','../fashion_data/raw/val/image/restricted.jpg'),
        rec('f','pants','../fashion_data/raw/train/train/image/unused_top.jpg',x2='0'),
        rec('g','outerwear','../fashion_data/raw/train/train/image/nope.jpg')]
  inventory=config/'prd_8class_unified_manifest_v2.csv';write_csv(inventory,rows)
  out=root/'output';summary=audit(project.resolve(),data.resolve(),inventory,oldcsv,newjs,out)
  assert summary['post_exclusion_distinct_image_hashes_all_categories']==2,summary
  assert summary['by_category_source_hint_NOT_GT']['skirt']['SAME_SHA_AS_PRIOR40']==1
  assert summary['by_category_source_hint_NOT_GT']['outerwear']['SAME_SHA_AS_PRIOR12']==1
  assert summary['by_category_source_hint_NOT_GT']['dress']['RESTRICTED_SPLIT']==1
  assert summary['allowed_train_hash_reads']==4
  assert summary['quarantined_image_reads']==0
  assert len(list(csv.DictReader((out/'exclusion_reason_audit.csv').open(encoding='utf-8-sig'))))==7
  try:audit(project.resolve(),data.resolve(),inventory,oldcsv,newjs,out)
  except FileExistsError:pass
  else:raise AssertionError('Audit must not overwrite existing output')
  print('SOURCE_GAP_SYNTHETIC_TEST_PASS 7 rows; allowed hashes only; old40/new12 excluded; restricted split never opened')
if __name__=='__main__':main()
