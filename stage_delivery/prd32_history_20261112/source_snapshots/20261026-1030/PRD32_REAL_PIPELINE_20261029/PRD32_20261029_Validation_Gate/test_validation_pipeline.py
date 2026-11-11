#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""End-to-end synthetic test. Never loads Qwen/CUDA and never uses real user data."""
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image
import csv,hashlib,json,os,sys
import prepare_validation as pv
import run_and_score as rs
from types import SimpleNamespace

def main():
 with TemporaryDirectory() as td:
  root=Path(td);project=root/'project';data=root/'fashion_data';train=data/'raw'/'train'/'train'/'image';train.mkdir(parents=True)
  (data/'raw'/'holdout'/'image').mkdir(parents=True)
  items=[];cats=['top','pants','skirt','outerwear','dress']
  for k,c in enumerate(cats):
   for j in range(3):
    img=train/f'img_{c}_{j}.jpg';Image.new('RGB',(140,120),(40+k*25,30+j*45,80)).save(img)
    items.append({'record_id':f'r_{k}_{j}','source_image':os.path.relpath(img,project),'garment_category':c,
     'garment_id':f'item_{j}','bbox_x1':10,'bbox_y1':20,'bbox_x2':105,'bbox_y2':110})
  # duplicate image hash with separate ID and unsafe holdout path
  items.append(dict(items[0],record_id='duplicate_new_id',garment_id='another_item'))
  items.append({'record_id':'holdout_guard','source_image':'../fashion_data/raw/holdout/image/secret.jpg',
      'garment_category':'dress','bbox_x1':0,'bbox_y1':0,'bbox_x2':10,'bbox_y2':15})
  inv=project/'configs'/'prd_8class_unified_manifest_v2.csv';inv.parent.mkdir(parents=True)
  with inv.open('w',newline='',encoding='utf-8-sig') as f:
   w=csv.DictWriter(f,fieldnames=items[0].keys());w.writeheader();w.writerows(items)
  prepare=root/'prepared'
  pv.do_prepare(SimpleNamespace(project_root=str(project),data_root=str(data),inventory=str(inv),per_class=2,exclude_hashes=None,output=str(prepare)))
  summ=json.loads((prepare/'summary.json').read_text())
  assert summ['selected_count']==10,summ
  assert (prepare/'Blind_Review_NewTrainDev.html').is_file()
  assert 'RESTRICTED_SPLIT' in (prepare/'quarantined_metadata.csv').read_text(encoding='utf-8-sig')
  pl=[json.loads(l) for l in (prepare/'candidate_manifest.jsonl').read_text().splitlines()]
  assert len(pl)==len({p['sha256'] for p in pl})
  reviews=[]
  for i,r in enumerate(pl):
   reviews.append({'case_id':r['case_id'],'image_sha256':r['sha256'],
      'human_category':'uncertain' if i==0 else r['inventory_hint_NOT_GT'],
      'bbox_status':'YES','visually_identifiable':'NO' if i==0 else 'YES',
      'overlap_other_garments':'YES' if i%2 else 'NO', 'boundary_case':'CLEAR',
      'evidence_level':'HIGH','reviewer_note':'synthetic generated image'})
  rpath=root/'human.json';pv.write_json(rpath,{'schema':'prd32_new_train_dev_blind_review_v1','rows':reviews})
  frozen=root/'frozen';pv.do_freeze(SimpleNamespace(prepared=str(prepare),review=str(rpath),output=str(frozen)))
  fmeta=json.loads((frozen/'freeze.json').read_text());assert fmeta['scoreable']==9
  pp,cfg=rs.verify_frozen(frozen,data);assert len(pp)==10
  fake=root/'fake_run';(fake/'marked_images').mkdir(parents=True)
  ps=[]
  for i,p in enumerate(pp):
   img=fake/'marked_images'/f'{i:03d}.jpg';Image.new('RGB',(80,80),(44,66,88)).save(img)
   expected=reviews[i]['human_category'];pr={}
   for ver in ('V2','V3'):
    predicted='pants' if i%2==0 else expected
    if ver=='V3' and i%2==0:predicted='top'
    pr[ver]={'status':'GENERATED_UNVERIFIED','category':predicted,
          'canonical_tags':{'category':predicted,'primary_color':'蓝色','pattern':'纯色','visible_design':None},
          'schema_valid':True,'native_json':False,
          'timing':{'end_to_end_ms':float(1000+i*10)}}
   ps.append({'case_id':p['case_id'],'image_sha256':p['image_sha256'],
          'marked_image':str(img),**pr})
  fp=root/'predictions_pair.jsonl';fp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in ps))
  cmp=root/'score';report,summary=rs.score(SimpleNamespace(frozen=str(frozen),allowed_root=str(data),predictions=str(fp),output=str(cmp)))
  assert summary['scoreable']==9
  page=root/'rootcause.html';rs.review_errors(SimpleNamespace(frozen=str(frozen),allowed_root=str(data),predictions=str(fp),output=str(page)))
  assert page.is_file() and '导出 JSON' in page.read_text()
  # Tamper test: hash mismatch causes score preflight to refuse
  pp[0]['image_sha256']='0'*64
  (frozen/'frozen_inference_plan.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pp))
  try:rs.verify_frozen(frozen,data)
  except ValueError as exc:assert 'hash' in str(exc).lower()
  else:raise AssertionError('Tampered freeze was not rejected')
  # Split-scope protection test (never accesses restricted image bytes)
  x,c=pv.safe_image_path('../fashion_data/raw/holdout/image/secret.jpg',project,data)
  assert x is None and c=='RESTRICTED_SPLIT'
  print('SYNTHETIC_E2E_PASS: 10 prepared; 9 scored; protected split blocked; frozen SHA tamper blocked; rootcause review generated')
if __name__=='__main__':main()
