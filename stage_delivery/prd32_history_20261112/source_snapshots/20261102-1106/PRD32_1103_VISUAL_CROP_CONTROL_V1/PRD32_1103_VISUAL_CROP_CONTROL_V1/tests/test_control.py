from pathlib import Path
import hashlib,importlib.util,json,os,tempfile,unittest
from PIL import Image,ImageDraw
P=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('run_visual_crop_control',P/'run_visual_crop_control.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ContractTests(unittest.TestCase):
 def test_original_runner_pin(self):self.assertEqual(m.sha(P/'reference_dev60_qwen_ab.py'),m.LEGACY_SHA)
 def test_canonical_lists_pinned(self):
  for f,h in [('smoke24_input_only.jsonl',m.SMOKE_SHA),('instances171_input_only.jsonl',m.FIXED171_SHA),('requests513_input_only.jsonl',m.FIXED513_SHA)]:
   self.assertEqual(m.sha(P/f),h)
 def test_171_canonical_list_no_gt(self):
  a=m.jlread(P/'instances171_input_only.jsonl');self.assertEqual(len(a),171)
  self.assertEqual(len({(x['case_id'],x['target_instance_id']) for x in a}),171)
  self.assertTrue(all(x['human_reference_used'] is False for x in a))
  self.assertFalse(any('gt' in k.lower() or 'human_label' in k.lower() for x in a for k in x))
 def test_canonical_513(self):
  rows=m.jlread(P/'instances171_input_only.jsonl')
  s=m.validate_fixed_lists(rows);self.assertEqual(len(s),24)
 def test_smoke_key_unique(self):self.assertEqual(len(set(m.smoke_keys())),24)
 def test_smoke_planned_72(self):
  a=m.jlread(P/'instances171_input_only.jsonl')
  self.assertEqual(len(m.planned_scope_keys('smoke',a)),72)
 def test_full_planned_513(self):
  a=m.jlread(P/'instances171_input_only.jsonl')
  self.assertEqual(len(m.planned_scope_keys('full',a)),513)
 def test_smoke_is_subset_of_full(self):
  a=m.jlread(P/'instances171_input_only.jsonl')
  self.assertTrue(set(m.planned_scope_keys('smoke',a)).issubset(set(m.planned_scope_keys('full',a))))
 def test_crop_rect_padding(self):self.assertEqual(m.target_crop_box([20,20,40,40],(100,100)),(17,17,43,43))
 def test_crop_clip(self):self.assertEqual(m.target_crop_box([0,0,80,80],(80,80)),(0,0,80,80))
 def test_crop_invalid(self):
  with self.assertRaises(ValueError):m.target_crop_box([30,30,20,40],(100,100))
 def test_render_equal_left_and_expected_right(self):
  with tempfile.TemporaryDirectory() as tmp:
   tmp=Path(tmp);src=tmp/'source.png';marked=tmp/'marked.png';out=tmp/'target.png'
   Image.new('RGB',(300,400),(20,100,200)).save(src)
   pic=Image.new('RGB',(240,320),(40,50,60));ImageDraw.Draw(pic).rectangle((20,30,200,290),outline=(255,0,0),width=3);pic.save(marked)
   v=m.make_composite(marked,src,[50,50,250,350],out,m.sha(src),m.sha(marked))
   self.assertEqual(v['composite_size'],[384,320])
   with Image.open(out) as o:
    self.assertEqual(o.crop((0,0,240,320)).tobytes(),pic.tobytes())
    self.assertEqual(o.size,(384,320))
 def test_render_rejects_changed_sha(self):
  with tempfile.TemporaryDirectory() as tmp:
   tmp=Path(tmp);src=tmp/'source.png';marked=tmp/'marked.png';out=tmp/'target.png'
   Image.new('RGB',(300,400)).save(src);Image.new('RGB',(300,400)).save(marked)
   with self.assertRaises(ValueError):m.make_composite(marked,src,[20,20,30,30],out,'0'*64,m.sha(marked))
 def test_prompt_only_full_visual_diff(self):
  self.assertTrue(m.CROP_SPEC['no_text_or_class_labels_in_overlay'])
  self.assertTrue(m.CROP_SPEC['baseline_full_marked_image_unchanged_left'])
 def test_scope_invalid(self):
  with self.assertRaises(ValueError):m.planned_scope_keys('bad',[])
 def test_resume_records(self):
  rr=[{'case_id':'fresh_dev_001','target_instance_id':'pred_0000','composite_sha256':'a','original_marked_sha256':'b'}]
  good={'case_id':'fresh_dev_001','target_instance_id':'pred_0000','request_id':'fresh_dev_001__pred_0000__tags',
    'task':'tags','model_variant':m.VARIANT,'status':'GENERATED_REVIEW_REQUIRED','composite_sha256':'a','original_marked_sha256':'b',
    'raw_text':'xxx','human_label_or_gt_bbox_used':False,'publish_allowed':False}
  m.check_prefix([good],[('fresh_dev_001','pred_0000','tags')],rr)
  with self.assertRaises(ValueError):m.check_prefix([dict(good,composite_sha256='c')],[('fresh_dev_001','pred_0000','tags')],rr)

if __name__=='__main__':unittest.main()
