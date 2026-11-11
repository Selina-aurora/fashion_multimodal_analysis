# Offline contract tests. No GPU or source image files needed.
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import run_idv1_control as m

HERE=Path(__file__).parent

class TestIDV1(unittest.TestCase):
 def test_frozen_lists_and_smoke(self):
  f=m.jload(HERE/'fixed_manifest.json')
  for n,h in f['pinned_files'].items(): self.assertEqual(m.file_sha(HERE/n),h)
  self.assertEqual(m.file_sha(HERE/'smoke24_input_only.jsonl'),m.PINNED_SMOKE_SHA)
  inst=m.jlread(HERE/'instances171_input_only.jsonl')
  req=m.jlread(HERE/'requests513_input_only.jsonl')
  smoke=m.jlread(HERE/'smoke24_input_only.jsonl')
  self.assertEqual((len(inst),len(req),len(smoke)),(171,513,24))
  keys=[(r['case_id'],r['target_instance_id'],t) for r in inst for t in m.TASKS]
  self.assertEqual([r['request_id'] for r in req],['__'.join(k) for k in keys])
  chosen={(r['case_id'],r['target_instance_id']) for r in smoke}
  self.assertEqual(len(chosen),24)
  self.assertEqual(len(m.scope_keys('smoke',inst,chosen)),72)
  self.assertEqual(len(m.scope_keys('full',inst,chosen)),513)
  self.assertEqual(len({r['image_sha256'] for r in inst}),60)
 def test_canonical_generator(self):
  inst=m.jlread(HERE/'requests513_input_only.jsonl')
  template=m.jload(HERE/'candidate_addenda.json')['ID-V1']
  with tempfile.TemporaryDirectory() as t:
   import subprocess,sys
   p=Path(t)/'delta.jsonl'
   result=subprocess.run([sys.executable,str(HERE/'preflight_and_generate_delta.py'),
     '--variant','ID-V1','--out',str(p)],check=True,capture_output=True,text=True)
   self.assertIn('FIXED_DEV60_171_513_INPUT_GATE_PASS',result.stdout)
   generated=m.jlread(p)
  self.assertEqual(len(generated),513)
  for r,d in zip(inst,generated):
   self.assertEqual(d['request_id'],r['request_id'])
   expected='\n\n'+template.format(instance_id=r['target_instance_id'])
   self.assertEqual(d['append_to_original_focus_v1_prompt'],expected)
   self.assertFalse(d['human_reference_used'])
 def test_resume_prefix_reject_changed_target_or_prompt(self):
  k=('fresh_dev_001','pred_0000','tags')
  prompt='ORIGINAL FOCUS'
  old=Mock(PROMPT_SHA={'focus_v1':{'tags':m.text_sha(prompt)}})
  delta={'append_to_original_focus_v1_prompt':'\n\nID-V1'}
  before={'image_sha256':'abc','marked_image_sha256':'def'}
  rec={'request_id':'__'.join(k),'model_variant':m.VARIANT,'status':'GENERATED_REVIEW_REQUIRED',
    'image_sha256':'abc','marked_image_sha256':'def','base_prompt_sha256':m.text_sha(prompt),
    'final_prompt_sha256':m.text_sha(prompt+delta['append_to_original_focus_v1_prompt']),
    'human_label_or_gt_bbox_used':False,'publish_allowed':False,
    'raw_text':'ok','original_prompt':prompt}
  m.check_existing_prefix([rec],[k],{k:before},{k:delta},old)
  rec['final_prompt_sha256']='0'*64
  with self.assertRaises(ValueError): m.check_existing_prefix([rec],[k],{k:before},{k:delta},old)
 def test_reference_sha(self):
  self.assertEqual(m.file_sha(HERE/'reference_dev60_qwen_ab.py'),m.PINNED_LEGACY_SHA)

if __name__=='__main__':unittest.main(verbosity=2)
