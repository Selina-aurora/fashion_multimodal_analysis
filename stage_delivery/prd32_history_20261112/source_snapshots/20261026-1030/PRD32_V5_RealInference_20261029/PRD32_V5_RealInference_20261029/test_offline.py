#!/usr/bin/env python3
"""Synthetic/no-GPU tests for protected source paths, identity, and posthoc scoring."""
import hashlib, json, os, tempfile
from pathlib import Path
from PIL import Image
import v5_real_inference as v5
import v5_posthoc_evaluate as post

def writej(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False),encoding='utf8')
def writejl(p,rows):p.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf8')
def main():
 with tempfile.TemporaryDirectory() as t:
  root=Path(t);data=root/'data';proj=root/'project';train=data/'raw/train';train.mkdir(parents=True)
  cp=proj/v5.CHECKPOINT_REL;cp.parent.mkdir(parents=True);cp.write_bytes(b'fake checksum only no model load')
  saved=v5.CKPT_SHA;v5.CKPT_SHA=v5.sha(cp)
  gate=root/'gate';gate.mkdir(); rows=[]
  for i in range(12):
   p=train/f'img_{i:02d}.png';Image.new('RGB',(50,50),(i*11,4,3)).save(p)
   rows.append({'case_id':f'fresh_dev_{i+1:03d}','image_path':str(p),'image_sha256':v5.sha(p),
    'requested_output':'PRD31_V5_RAW_PREDICTIONS_ONLY','scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',
    'do_not_use_historical_gt_bbox':True,'human_reference_withheld':True})
  writejl(gate/'v5_image_only_inputs.jsonl',rows)
  writej(gate/'v5_live_preflight_summary.json',{'schema':'prd32_v5_live_preflight_v1','stage':'CHECKPOINT_AND_SOURCE_PROVENANCE_ONLY',
    'ready_for_live_v5_invocation':True,'checkpoint':{'path':str(cp),'expected_sha256':v5.CKPT_SHA,'actual_sha256':v5.CKPT_SHA,'hash_verified_now':True},
    'new_train_source_candidates_verified':12})
  ok,_,_=v5.validate_inputs(gate,data,proj,True);assert len(ok)==12
  bad=data/'raw/val/x.jpg';bad.parent.mkdir();bad.write_bytes(b'NEVER_READ')
  try:v5.verified_source_path(str(bad),data);assert False
  except ValueError as e:assert 'RESTRICTED' in str(e)
  sym=train/'linked.png';sym.symlink_to(bad)
  try:v5.verified_source_path(str(sym),data);assert False
  except ValueError as e:assert 'RESTRICTED' in str(e)
  rows[0]['bbox']=[0,0,10,10];writejl(gate/'v5_image_only_inputs.jsonl',rows)
  try:v5.validate_inputs(gate,data,proj);assert False
  except ValueError as e:assert 'schema' in str(e)
  rows[0].pop('bbox');writejl(gate/'v5_image_only_inputs.jsonl',rows)
  rows[0]['image_sha256']='a'*64;writejl(gate/'v5_image_only_inputs.jsonl',rows)
  try:v5.validate_inputs(gate,data,proj);assert False
  except ValueError as e:assert 'SHA' in str(e)
  rows[0]['image_sha256']=v5.sha(rows[0]['image_path']);writejl(gate/'v5_image_only_inputs.jsonl',rows)
  # Check frozen posthoc join; this cannot affect inference because different script/process.
  frozen=root/'frozen';frozen.mkdir()
  plans=[{'case_id':r['case_id'],'image_sha256':r['image_sha256'],'bbox':[5,5,30,30],
    'bbox_origin':'historical_inventory_gt_pilot_NOT_V5_prediction'} for r in rows]
  humans=[{'case_id':r['case_id'],'image_sha256':r['image_sha256'],'category':'top','scoreable':i<9,
    'overlap_other_garments':'NO'} for i,r in enumerate(rows)]
  writejl(frozen/'frozen_inference_plan.jsonl',plans);writejl(frozen/'frozen_human_reference.jsonl',humans)
  writej(frozen/'freeze.json',{'status':'FROZEN_DEV_NOT_INDEPENDENT',
    'plan_sha256':v5.sha(frozen/'frozen_inference_plan.jsonl'),'human_reference_sha256':v5.sha(frozen/'frozen_human_reference.jsonl')})
  raw=root/'raw';raw.mkdir(); predictions=[]
  for r in rows:
   predictions.append({'case_id':r['case_id'],'image_sha256':r['image_sha256'],
    'instances':[{'instance_id':'pred_0000','predicted_category':'top','score':.9,'bbox_xyxy':[6,6,30,30]}]})
  writejl(raw/'v5_raw_predictions.jsonl',predictions)
  writej(raw/'v5_real_summary.json',{'status':'V5_RAW_INFERENCE_COMPLETED','v5_predictions_sha256':v5.sha(raw/'v5_raw_predictions.jsonl')})
  post.analyze(frozen,raw,root/'eval')
  result=post.j(root/'eval/v5_posthoc_summary.json')
  assert result['scorable_human_cases']==9 and result['bbox_and_class_match_at_0_5']==9
  assert post.box_iou([0,0,10,10],[5,5,15,15])==25/175
  v5.CKPT_SHA=saved
 print('V5_SYNTHETIC_TESTS_PASS: source guard, symlink, GT-free schema, hash tamper, posthoc join')
if __name__=='__main__':main()
