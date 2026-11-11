import csv, json, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from identity_class_guard_v3 import route_case, assert_frozen_metadata, extract_tags_category, ACTIONS
from mask_binding_contract import inspect_mask_record,safe_dev_path

class TestIdentityClassGuard(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.v5=[json.loads(s) for s in (ROOT/'input/v5_raw_predictions.jsonl').read_text().splitlines() if s.strip()]
  cls.qwen=[json.loads(s) for s in (ROOT/'input/focus_v1_qwen_results_513.jsonl').read_text().splitlines() if s.strip()]

 def test_01_fixed_dataset_and_no_holdout(self):
  self.assertEqual(len(self.v5),60);self.assertEqual(sum(len(x['instances']) for x in self.v5),171)
  self.assertTrue(all(x['scope']=='TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY' for x in self.v5))

 def test_02_three_outputs_per_instance(self):
  self.assertEqual(len(self.qwen),513)
  from collections import Counter
  q=Counter((x['case_id'],x['target_instance_id']) for x in self.qwen)
  self.assertTrue(all(v==3 for v in q.values()));self.assertEqual(len(q),171)

 def test_03_metadata_matched(self):
  im=self.v5[0]; pred=im['instances'][0]
  tag=next(x for x in self.qwen if x['case_id']==im['case_id'] and x['target_instance_id']==pred['instance_id'] and x['task']=='tags')
  self.assertEqual(assert_frozen_metadata(im,pred,tag),[])

 def test_04_tampered_id_rejected(self):
  im=self.v5[0];pred=im['instances'][0]
  tag=next(x for x in self.qwen if x['case_id']==im['case_id'] and x['target_instance_id']==pred['instance_id'] and x['task']=='tags')
  altered=dict(tag,target_instance_id='pred_wrong')
  self.assertIn('instance_id',assert_frozen_metadata(im,pred,altered))

 def test_05_tampered_bbox_rejected(self):
  im=self.v5[0];pred=im['instances'][0]
  tag=next(x for x in self.qwen if x['case_id']==im['case_id'] and x['target_instance_id']==pred['instance_id'] and x['task']=='tags')
  bbox=list(tag['v5_predicted_bbox_xyxy']);bbox[0]+=15
  self.assertIn('bbox_value',assert_frozen_metadata(im,pred,dict(tag,v5_predicted_bbox_xyxy=bbox)))

 def test_06_human_bbox_rejected(self):
  im=self.v5[0];pred=im['instances'][0]
  tag=next(x for x in self.qwen if x['case_id']==im['case_id'] and x['target_instance_id']==pred['instance_id'] and x['task']=='tags')
  self.assertIn('gt_or_human_bbox_used',assert_frozen_metadata(im,pred,dict(tag,human_label_or_gt_bbox_used=True)))

 def test_07_categories_are_distinct(self):
  self.assertEqual(ACTIONS['SPATIAL'],'BLOCK_AND_DIAGNOSE_VISUAL_IDENTITY')
  self.assertNotEqual(ACTIONS['SEMANTIC'],ACTIONS['UPSTREAM_MIXED'])

 def test_08_full_run_30_and_93(self):
  with tempfile.TemporaryDirectory() as td:
   s=route_case(ROOT,Path(td)/'sidecar');self.assertEqual(s['tags_v5_class_conflicts'],93)
   self.assertEqual(s['reviewed_30_user_chat_attested_AI_prefill'],30)
   self.assertEqual(s['reviewed_30_intersection_shadow_alias'],0)
   self.assertEqual(s['published_categories'],0)

 def test_09_counts_of_root_causes(self):
  with tempfile.TemporaryDirectory() as td:
   s=route_case(ROOT,Path(td)/'sidecar')
  x=s['prior_ai_root_cause_distribution']
  self.assertEqual(x,{'MIXED_EVIDENCE':3,'SEMANTIC':6,'SPATIAL':14,'UNCERTAIN':1,'UPSTREAM_MIXED':6})

 def test_10_no_publish_and_no_move(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/'sidecar';route_case(ROOT,out)
   with (out/'dev60_171_identity_class_routing.csv').open(encoding='utf-8-sig') as f:
    rows=list(csv.DictReader(f))
   self.assertEqual(len(rows),171)
   self.assertTrue(all(x['qwen_tag_category_publish_allowed']=='NO' and x['qwen_record_modified']=='NO' for x in rows))

 def test_11_shadow_nonoverlap(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/'sidecar';route_case(ROOT,out)
   with (out/'dev60_171_identity_class_routing.csv').open(encoding='utf-8-sig') as f:
    rows=list(csv.DictReader(f))
   self.assertTrue(all(not(x['user_chat_attested_30']=='YES' and x['shadow_alias_candidate']=='YES') for x in rows))

 def test_12_mask_holdout_disallowed(self):
  self.assertFalse(safe_dev_path('/workspace/W7_HOLDOUT_SEALED/xx.png'))
  self.assertTrue(safe_dev_path('/workspace/PRD32_DEV60_V5_REAL_20261030_run01'))

 def test_13_mask_contract_synthetic(self):
  from PIL import Image,ImageDraw
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);im={'case_id':'fresh_dev_001','image_size':[20,20]}
   pred={'instance_id':'pred_0000','mask_path':'fresh_dev_001_pred_0000_mask.png','bbox_xyxy':[4,4,12,12]}
   img=Image.new('L',(20,20),0);ImageDraw.Draw(img).rectangle([5,5,11,11],fill=255)
   img.save(p/pred['mask_path'])
   self.assertEqual(inspect_mask_record(im,pred,p)['status'],'GEOMETRY_COMPATIBLE')
   pred['bbox_xyxy']=[15,15,19,19]
   self.assertEqual(inspect_mask_record(im,pred,p)['status'],'MASK_BBOX_GEOMETRY_SUSPECT')

if __name__=='__main__':unittest.main()
