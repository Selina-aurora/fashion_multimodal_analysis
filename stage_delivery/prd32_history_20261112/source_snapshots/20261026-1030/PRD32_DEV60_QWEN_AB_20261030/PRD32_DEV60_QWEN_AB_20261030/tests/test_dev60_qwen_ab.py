import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
S=Path(__file__).resolve().parents[1]/'dev60_qwen_ab.py'
spec=importlib.util.spec_from_file_location('dev60',S);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TestDev60AB(unittest.TestCase):
 def test_bound_expected_inputs(self):
  self.assertEqual(m.DEV60_INPUT,'c629ff0ca557a0c57c7b97c93cf389502f9660de63a934463cb1f5e8e722752f')
  self.assertEqual(m.V5_HASH,'688cb801daebd0484e8740c877c021eca86c83f362857d4fc982d02fb50f2710')
 def test_total_requests(self):
  self.assertEqual(171*len(m.TASKS),513)
 def test_no_human_gt_in_snapshot(self):
  fake=[{'case_id':'fresh_dev_001','target_instance_id':'pred_0000'}]
  self.assertEqual(m.expected_keys(fake),[('fresh_dev_001','pred_0000','tags'),('fresh_dev_001','pred_0000','vqa'),('fresh_dev_001','pred_0000','copy')])
 def test_reject_holdout_train_path(self):
  with self.assertRaises(ValueError):m.safe_path('/workspace/fashion_data/holdout/train/pic.jpg',require=False)
 def test_reject_validation(self):
  with self.assertRaises(ValueError):m.safe_path('/workspace/fashion_data/raw/val/train/img.jpg',require=False)
 def test_allow_train_path(self):
  p=m.safe_path('/workspace/fashion_data/raw/train/train/image/0001.jpg',require=False)
  self.assertEqual(p.name,'0001.jpg')
 def test_reject_outside_root(self):
  with self.assertRaises(ValueError):m.safe_path('/root/work/train/img.jpg',require=False)
 def test_debug_marker_detects_missing_1030_case(self):
  self.assertIsNotNone(m.DEBUG.search('红色矩形指向一件鞋'))
 def test_dont_flag_normal_color_red(self):
  self.assertIsNone(m.DEBUG.search('这是一条红色连衣裙'))
 def test_prompt_core_sha_pinned(self):
  path=S.parents[0]/'reference_12img_runner_READ_ONLY.py'
  self.assertEqual(m.sha(path),m.CORE_SHA)
 def test_wrong_source_pins_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises((ValueError,FileNotFoundError)):
    m.source_pins(Path(d),Path(d)/'focus.py')
 def test_empty_prefix_valid(self):
  self.assertTrue(m.check_prefix([],[],[], 'baseline'))
 def test_incomplete_bad_request_rejected(self):
  with self.assertRaises(ValueError):m.check_prefix([{'case_id':'xxx'}],[('fresh_dev_001','pred_0000','tags')],[], 'baseline')
 def test_reject_mismatched_question(self):
  self.assertNotEqual(m.tsha('其它问题'),m.tsha(m.QUESTION))
 def test_sample_model_output_schema(self):
  with tempfile.TemporaryDirectory() as d:
   img=Path(d)/'im.jpg';img.write_bytes(b'marked')
   r={'case_id':'fresh_dev_001','target_instance_id':'pred_0000','task':'vqa',
   'request_id':'fresh_dev_001__pred_0000__vqa','status':'GENERATED_REVIEW_REQUIRED','model_variant':'baseline',
   'image_sha256':'a'*64,'v5_predicted_bbox_xyxy':[1,2,10,20],'v5_category_unverified':'top',
   'human_label_or_gt_bbox_used':False,'prompt_sha256':m.PROMPT_SHA['baseline']['vqa'],
   'marked_image_sha256':m.sha(img),'marked_image':str(img),'raw_text':'白色',
   'publish_allowed':False,'question_sha256':m.tsha(m.QUESTION)}
   rows=[{'case_id':'fresh_dev_001','target_instance_id':'pred_0000','image_sha256':'a'*64,'target_bbox_xyxy':[1,2,10,20], 'predicted_category_v5_unverified':'top'}]
   self.assertTrue(m.check_prefix([r],[('fresh_dev_001','pred_0000','vqa')],rows,'baseline'))
   r['raw_text']=' '
   with self.assertRaises(ValueError):m.check_prefix([r],[('fresh_dev_001','pred_0000','vqa')],rows,'baseline')
 def test_actual_uploaded_raw_manifest_offline(self):
  actual=Path('/mnt/data/dev60_v5_unpacked_local_run')
  inp=Path('/mnt/data/PRD32_DEV60_STAGE1E_LOCAL_EVIDENCE/v5_image_only_inputs_60.jsonl')
  if actual.is_dir() and inp.is_file():
   rows,cats,dups=m.read_and_validate_raw(actual,inp,check_images=False)
   self.assertEqual(len(rows),171)
   self.assertEqual(sum(cats.values()),171)
   self.assertEqual(len(cats),8)
   self.assertEqual(len(dups),20)
if __name__=='__main__':unittest.main()
