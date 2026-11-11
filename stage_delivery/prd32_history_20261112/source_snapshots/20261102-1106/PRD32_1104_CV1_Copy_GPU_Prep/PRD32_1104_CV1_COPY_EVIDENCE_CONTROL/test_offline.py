import hashlib,json,unittest
from pathlib import Path
import importlib.util
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('cv1',P/'run_cv1_copy_control.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class TestCV1Offline(unittest.TestCase):
 def test_frozen_static_and_count(self):
  manifest,rows,requests,smoke,delta=m.check_static()
  self.assertEqual((len(rows),len(requests),len(smoke)),(171,513,24))
 def test_scope_copy_only(self):
  _,rows,_,smoke,_=m.check_static()
  smoke_keys=m.ordered_keys('smoke',rows,smoke);allkeys=m.ordered_keys('full',rows,smoke)
  self.assertEqual((len(smoke_keys),len(allkeys)),(24,171))
  self.assertEqual({k[-1] for k in smoke_keys+allkeys},{'copy'})
 def test_no_gt_terms_in_delta(self):
  _,rows,_,smoke,delta=m.check_static()
  self.assertEqual(delta['human_gt_used'],False)
  self.assertNotIn('ground_truth',delta['append_to_locked_focus_v1_copy_prompt'])
 def test_no_overwrite_resume_check(self):
  self.assertTrue(callable(m.check_prior))
 def test_reference_hash(self):
  self.assertEqual(m.sha(P/'reference_dev60_qwen_ab.py'),m.PINNED_LEGACY_SHA)
 def test_cv1_does_not_inherit_idv1_prompt(self):
  delta=m.readj(P/'cv1_copy_delta.json')['append_to_locked_focus_v1_copy_prompt']
  self.assertNotIn('ID-V1',delta)
  self.assertNotIn('S-V1',delta)
 def test_pinned_prompts(self):
  manifest=m.readj(P/'fixed_manifest.json')
  self.assertEqual(manifest['focus_v1_original_prompt_sha256']['copy'],'c0373dfa0bb16f634d7944fe756ef692a64e404ceaaab1f7b0ee2f9fd8a63193')
if __name__=='__main__':unittest.main()
