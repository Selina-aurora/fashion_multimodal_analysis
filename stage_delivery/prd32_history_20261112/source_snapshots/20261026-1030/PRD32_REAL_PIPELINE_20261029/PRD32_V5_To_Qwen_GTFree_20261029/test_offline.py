#!/usr/bin/env python3
# Offline end-to-end smoke: provenance, no hidden GT, path guards, chosen instances.
from pathlib import Path
from PIL import Image
import tempfile, json
import v5_to_qwen_bridge as gate

def w(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False)+'\n',encoding='utf-8')
def wl(p,rows):p.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')

def run():
 with tempfile.TemporaryDirectory() as d:
  p=Path(d); data=p/'fashion_data'/'source'/'train';data.mkdir(parents=True)
  raw=p/'raw';raw.mkdir();g=p/'source_gate';g.mkdir()
  samples=[];preds=[]
  for n in range(1,13):
   cid=f'fresh_dev_{n:03d}';image=data/(cid+'.jpg');Image.new('RGB',(150,120),(n*12,40,90)).save(image)
   digest=gate.sha(image)
   samples.append({'case_id':cid,'image_path':str(image),'image_sha256':digest,'requested_output':'PRD31_V5_RAW_PREDICTIONS_ONLY',
    'scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY','do_not_use_historical_gt_bbox':True,'human_reference_withheld':True})
   entries=[{'instance_id':'pred_0000','predicted_category':'top','predicted_class_id':1,'score':.95,
             'bbox_xyxy':[4.0,6.0,70.0,80.0],'bbox_coordinate_space':'original_image','mask_path':None,
             'mask_coordinate_space':'not_saved','prediction_origin':'V5_CHECKPOINT_LIVE_MODEL_OUTPUT_NOT_GT'}]
   if n==4:
    entries.append({'instance_id':'pred_0001','predicted_category':'pants','predicted_class_id':2,'score':.75,
             'bbox_xyxy':[55.0,50.0,110.0,115.0],'bbox_coordinate_space':'original_image','mask_path':None,
             'mask_coordinate_space':'not_saved','prediction_origin':'V5_CHECKPOINT_LIVE_MODEL_OUTPUT_NOT_GT'})
   preds.append({'schema':'prd32_v5_live_raw_image_result_v1','case_id':cid,'image_sha256':digest,
        'source_image_path':str(image),'image_size':[150,120],'segmentation_ms':33.0,
        'instances':entries,'instance_count':len(entries),'human_target_instance_selected':False,
        'review_image':cid+'_v5_review.jpg','scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY'})
  wl(g/'v5_image_only_inputs.jsonl',samples)
  w(g/'v5_live_preflight_summary.json',{'ready_for_live_v5_invocation':True,'checkpoint':{'actual_sha256':gate.CKPT_SHA}})
  wl(raw/'v5_raw_predictions.jsonl',preds)
  w(raw/'v5_real_summary.json',{'status':'V5_RAW_INFERENCE_COMPLETED','checkpoint_sha256':gate.CKPT_SHA,
    'v5_predictions_sha256':gate.sha(raw/'v5_raw_predictions.jsonl'),'n_images':12,'n_predicted_instances':13})
  w(raw/'run_manifest.json',{'schema':'prd32_v5_real_predictions_run_v1','checkpoint_sha256':gate.CKPT_SHA,
    'human_gt_used_for_inference':False,'score_threshold':.4,'mask_threshold':.5,
    'input_manifest_sha256':gate.sha(g/'v5_image_only_inputs.jsonl')})
  check,out=gate.validated(raw,g,p/'fashion_data')
  assert len(check)==13 and all(x['human_reference_used'] is False for x in check)
  result=gate.prepare(raw,g,p/'fashion_data',p/'prepared')
  assert result['n_predicted_instances']==13
  assert (p/'prepared/V5_Predicted_Instances_Review.html').is_file()
  assert sum(x['case_id']=='fresh_dev_004' for x in gate.readjl(p/'prepared/gt_free_all_instances.jsonl'))==2
  helper,_=gate.load_helper()
  sample={'task':'tags','query':'颜色','image_path':check[0]['image_path'],
          'image_sha256':check[0]['image_sha256'],'target_instance_id':'pred_0000','target_bbox_xyxy':check[0]['target_bbox_xyxy'],
          'predicted_attribute_context':[]}
  assert 'category' in helper.make_prompt(sample,input_mode='marked_full_image',pass_unverified=False)
  assert helper.parse_tag_output('{"category":"top","primary_color":"白色","pattern":"纯色","visible_design":"圆领"}')['schema_valid']
  tmp=p/'tamper';tmp.mkdir();
  hacked=[dict(x) for x in preds];hacked[0]=dict(hacked[0]);hacked[0]['human_category']='top'
  wl(raw/'v5_raw_predictions.jsonl',hacked)
  s=gate.readj(raw/'v5_real_summary.json');s['v5_predictions_sha256']=gate.sha(raw/'v5_raw_predictions.jsonl');w(raw/'v5_real_summary.json',s)
  try:gate.validated(raw,g,p/'fashion_data');raise AssertionError('Unsafe GT field accepted')
  except ValueError as e:assert 'fields' in str(e)
  wl(raw/'v5_raw_predictions.jsonl',preds)
  bad=p/'fashion_data'/'source'/'val'/'x.jpg';bad.parent.mkdir();bad.write_bytes(b'NOT_READ')
  try:gate.approved_image(str(bad),p/'fashion_data');raise AssertionError('Restricted split accepted')
  except ValueError:pass
  link=data/'symlink.jpg';link.symlink_to(bad)
  try:gate.approved_image(str(link),p/'fashion_data');raise AssertionError('Unsafe symlink accepted')
  except ValueError:pass
  print('V5_GT_FREE_QWEN_BRIDGE_TEST_PASS: 12 images, all-instance bridge, no GT, tamper/path protection, original prompts')
if __name__=='__main__':run()
