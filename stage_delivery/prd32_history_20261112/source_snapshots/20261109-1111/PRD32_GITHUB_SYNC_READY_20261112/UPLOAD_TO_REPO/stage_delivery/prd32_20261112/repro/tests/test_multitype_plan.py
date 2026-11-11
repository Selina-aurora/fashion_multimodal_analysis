import unittest,sys,json,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_vqa_multitype_pilot_v1 as m
class Tests(unittest.TestCase):
 def test_plan(self):
  x=m.read_plan();self.assertEqual(len(x['targets']),16);self.assertEqual(len(x['questions']),32)
 def test_types(self):
  x=m.read_plan();self.assertEqual(set(q['type'] for q in x['questions']),m.TYPES)
 def test_two_question_each_target(self):
  c=collections.Counter((q['case_id'],q['instance_id']) for q in m.read_plan()['questions']);self.assertEqual(sorted(c.values()),[2]*16)
 def test_never_publish(self):
  x=m.read_plan();self.assertFalse(x['production_release_allowed']);self.assertFalse(x['independent_gt']);self.assertFalse(x['holdout_used'])
 def test_immutability_sha(self):
  self.assertEqual(m.digest_file(m.PLAN),m.PLAN_SHA.read_text().strip())
 def test_confirmed34(self):
  import csv
  with (m.HERE/'V2_34_USER_CONFIRMED_DEVELOPMENT_REVIEW.csv').open(encoding='utf-8-sig',newline='') as f:
   r=list(csv.DictReader(f))
  self.assertEqual(len(r),34);self.assertTrue(all(x['independent_blind_gt']=='NO' for x in r))
if __name__=='__main__':unittest.main()
