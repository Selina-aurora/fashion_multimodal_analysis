#!/usr/bin/env python3
"""Offline test of 12-case preflight and score. No user data, model or GPU."""
from __future__ import annotations
import importlib.util,sys,hashlib,json,tempfile
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('new12_roi_tool',HERE/'paired_roi_replication.py')
app=importlib.util.module_from_spec(spec);sys.modules[spec.name]=app;spec.loader.exec_module(app)
v2path=Path('/mnt/data/PRD32_20261029_Validation_Gate/frozen_helpers/prd32_integrated_pipeline_20261027.py')
# The test works from anywhere; use a minimal faithful rendering stub (real helper is checked in deployment).
class FakeV2:
 @staticmethod
 def make_marked_image(record,outdir,mode,pad):
  from PIL import ImageDraw
  with Image.open(record['image_path']) as im:im=im.convert('RGB')
  b=record['target_bbox_xyxy'];ImageDraw.Draw(im).rectangle(tuple(b),outline=(250,40,40),width=3)
  dest=outdir/(record['case_id']+'_full.jpg');im.save(dest,'JPEG',quality=93);return dest

def main():
 with tempfile.TemporaryDirectory() as d:
  p=Path(d);imgs=p/'train';imgs.mkdir();fr=p/'frozen';fr.mkdir();gpu=p/'gpu';gpu.mkdir()
  (fr/'frozen_inference_plan.jsonl').write_text('synthetic frozen plan\n')
  (fr/'frozen_human_reference.jsonl').write_text('synthetic frozen refs\n')
  archived=p/'archived.jsonl';archived.write_text('synthetic archived predictions\n')
  plan=[];old={};refs=[]
  for i,cid in enumerate(app.IDS):
   file=imgs/f'{cid}.png';Image.new('RGB',(170,140),(100+i*8,120+i*2,150)).save(file)
   hashv=app.sha(file)
   plan.append({'case_id':cid,'image_sha256':hashv,'image_path':str(file),
                'target_instance_id':'inst1','bbox':[40,30,100,105]})
   old[cid]={'V2':{'category':'top'},'V3':{'category':'top'}}
   refs.append({'case_id':cid,'image_sha256':hashv,'scoreable':i not in (4,7,11),
                'category':'top' if i not in (4,7,11) else 'uncertain',
                'overlap_other_garments':'YES' if i%2 else 'NO','boundary_case':'CLEAR','evidence_level':'HIGH'})
  fakegate=SimpleNamespace(read_jsonl=lambda unused:refs)
  hashes={'V2':app.EXPECTED['prompt_v2'],'V3':app.EXPECTED['prompt_v3']}
  orig_verify=app.verify
  app.verify=lambda unused:(fakegate,plan,old,FakeV2,SimpleNamespace(),hashes)
  try:
   pre=p/'prepared';prea=SimpleNamespace(output=pre,pad=.08,frozen=fr,archived=archived)
   app.preflight(prea)
   a=SimpleNamespace(prepared=pre)
   m=app.verified_manifest(a,plan,hashes)
   assert len(m['cases'])==12 and (pre/'Review_Crop_Input_Before_GPU.html').is_file()
   preds=[]
   for i,(r,im) in enumerate(zip(plan,m['cases'])):
    preds.append({'case_id':r['case_id'],'image_sha256':r['image_sha256'],'full_sha256':im['full_sha256'],
                  'roi_sha256':im['roi_sha256'],
                  'full':{'V2':{'status':'GENERATED_UNVERIFIED','category':'top','canonical_tags':{'primary_color':'白色'}},
                          'V3':{'status':'GENERATED_UNVERIFIED','category':'top','canonical_tags':{'primary_color':'白色'}}},
                  'roi':{'V2':{'status':'GENERATED_UNVERIFIED','category':'pants' if i==0 else 'top','canonical_tags':{'primary_color':'黑色' if i==0 else '白色'}},
                         'V3':{'status':'GENERATED_UNVERIFIED','category':'pants' if i==1 else 'top','canonical_tags':{'primary_color':'黑色' if i==1 else '白色'}}}})
   predp=gpu/'predictions_full_roi.jsonl';predp.write_text(''.join(json.dumps(r)+'\n' for r in preds))
   app.dumpj(gpu/'run_config.json',{'images_manifest_sha256':app.sha(pre/'images_manifest.json'),'prompt_hashes':hashes})
   scorepath=p/'score'
   app.score(SimpleNamespace(prepared=pre,gpu_run=gpu,output=scorepath,gate=None,frozen=fr,archived=archived))
   s=app.loadj(scorepath/'roi_replicated_summary.json')
   assert s['n_images']==12 and s['n_scoreable']==9
   assert s['results']['V2']['regressed']==1 and s['results']['V3']['regressed']==1
   assert len((scorepath/'roi_replicated_comparison.csv').read_text(encoding='utf-8-sig').splitlines())==13
   print('SYNTHETIC_NEW12_PREP_SCORE_PASS, 12 prepared / 9 scoreable / 2 regressions correctly reported')
  finally:app.verify=orig_verify
if __name__=='__main__':main()
