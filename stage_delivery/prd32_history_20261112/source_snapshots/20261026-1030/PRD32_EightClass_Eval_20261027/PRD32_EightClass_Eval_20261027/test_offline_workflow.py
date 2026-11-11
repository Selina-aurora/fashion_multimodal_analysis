#!/usr/bin/env python3
"""End-to-end synthetic TEST. NOT eight-class results or PRD acceptance."""
import json, tempfile, subprocess, sys, csv
from pathlib import Path
from PIL import Image
from prd32_eightclass_eval import shasum

def go():
 script=Path(__file__).parent/'prd32_eightclass_eval.py'
 with tempfile.TemporaryDirectory() as d:
  r=Path(d);img=r/'new_image.png';Image.new('RGB',(96,105),'white').save(img)
  base={'case_id':'new01','image_id':'new01','image_path':str(img),
        'image_sha256':shasum(img),'target_instance_id':'inst01',
        'target_bbox_xyxy':[6,6,91,99], 'target_bbox_origin':'prd31_v5_prediction',
        'predicted_category':'top','source_split':'dev_new'}
  rows=[base,dict(base,case_id='training_case',image_id='im_train',target_instance_id='train01',source_split='train'),
        dict(base,case_id='badbox',image_id='bad_image',target_instance_id='bb01',target_bbox_xyxy=[-1,2,55,99])]
  src=r/'src.jsonl';src.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')
  def call(*x):return subprocess.run([sys.executable,str(script),*x],text=True,capture_output=True,check=True)
  call('prepare','--input',str(src),'--output',str(r/'p'))
  o=json.loads((r/'p'/'summary.json').read_text());assert o['eligible']==1 and o['blocked']==2,o
  assert 'DRY_RUN_OK' in call('run','--input',str(r/'p'/'eligible_dev.jsonl'),'--output',str(r/'not_created'),'--dry-run').stdout
  assert not (r/'not_created').exists()
  mock={'case_id':'new01','image_id':'new01','target_instance_id':'inst01','status':'GENERATED_UNVERIFIED',
        'canonical_tags':{'category':'top','primary_color':'白色','pattern':'纯色','visible_design':'圆领'}}
  (r/'pred.jsonl').write_text(json.dumps(mock,ensure_ascii=False)+'\n',encoding='utf-8')
  call('review','--predictions',str(r/'pred.jsonl'),'--manifest',str(r/'p'/'eligible_dev.jsonl'),'--output',str(r/'review.html'))
  html=(r/'review.html').read_text();assert '<article class=' in html and 'data:image/jpeg;base64,' in html
  gt={'schema':'prd32_8class_gt_dev_v1','results':[{'case_id':'new01','review_status':'VERIFIED',
           'gt_category':'top','gt_primary_color':'白色','gt_pattern':'纯色'}]}
  (r/'gt.json').write_text(json.dumps(gt,ensure_ascii=False))
  call('score','--predictions',str(r/'pred.jsonl'),'--gt',str(r/'gt.json'),'--output',str(r/'score.json'))
  z=json.loads((r/'score.json').read_text());assert z['fields']['category']['correct']==1
 print('SYNTHETIC_OFFLINE_E2E_PASS: 1 valid, train+bbox rejected, dryrun, static review, dev score')
if __name__=='__main__':go()
