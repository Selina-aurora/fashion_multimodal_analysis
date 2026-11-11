import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
SRC=Path(__file__).resolve().parents[1]/'prd32_freeze_dev_20261030.py'
spec=importlib.util.spec_from_file_location('freeze_dev',SRC)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TestFixedDev(unittest.TestCase):
 def test_bbox_full_overlap(self):self.assertAlmostEqual(m.bbox_iou([0,0,10,10],[0,0,10,10]),1)
 def test_bbox_partial_overlap(self):self.assertAlmostEqual(m.bbox_iou([0,0,10,10],[5,5,15,15]),25/175)
 def test_bbox_disjoint(self):self.assertEqual(m.bbox_iou([0,0,1,1],[2,2,3,3]),0)
 def test_bbox_invalid(self):self.assertEqual(m.bbox_iou([0,0,float('nan'),1],[0,0,1,1]),0)
 def test_sha_matches(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a';p.write_text('abc');self.assertEqual(m.sha_file(p),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
 def test_existing_pair_partial(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'q.jsonl'
   row={'case_id':'x','target_instance_id':'pred_0000','task':'tags','request_id':'x__pred_0000__tags',
        'status':'GENERATED_REVIEW_REQUIRED','human_label_or_gt_bbox_used':False,'publish_allowed':False,
        'v5_predicted_bbox_xyxy':[1,2,3,4],'v5_category_unverified':'top','v5_predicted_score_uncalibrated':.9}
   p.write_text(json.dumps(row)+'\n')
   ref={('x','pred_0000'):{'target_bbox_xyxy':[1,2,3,4],'predicted_category_v5_unverified':'top','predicted_score_v5_uncalibrated':.9}}
   r=m.validate_existing_pair(p,ref);self.assertEqual(r['status'],'PARTIAL_COVERAGE');self.assertEqual(r['missing_requests'],2)
 def test_existing_pair_tampered(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'q.jsonl'
   row={'case_id':'x','target_instance_id':'pred_0000','task':'tags','request_id':'wrong',
        'status':'GENERATED_REVIEW_REQUIRED','human_label_or_gt_bbox_used':False,'publish_allowed':False,
        'v5_predicted_bbox_xyxy':[1,2,3,4],'v5_category_unverified':'top','v5_predicted_score_uncalibrated':.9}
   p.write_text(json.dumps(row)+'\n')
   ref={('x','pred_0000'):{'target_bbox_xyxy':[1,2,3,4],'predicted_category_v5_unverified':'top','predicted_score_v5_uncalibrated':.9}}
   r=m.validate_existing_pair(p,ref);self.assertEqual(r['status'],'INCOMPATIBLE_WITH_FIXED_COHORT')
 def test_missing_history(self):
  r=m.validate_existing_pair('/no/such/path.jsonl',{});self.assertEqual(r['status'],'NOT_FOUND')
if __name__=='__main__':unittest.main()
