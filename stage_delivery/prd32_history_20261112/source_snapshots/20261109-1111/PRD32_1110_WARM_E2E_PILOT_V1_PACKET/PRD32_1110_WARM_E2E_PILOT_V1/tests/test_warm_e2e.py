import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from warm_e2e_pilot_v1 import percentile,stats
from producer_contract_v7 import QWEN_SMOKE_TARGETS, TARGETS, refuse_restricted

class TestSafetyAndTimings(unittest.TestCase):
 def test_p95_interpolation(self):
  self.assertEqual(percentile([10,20,30,40],95),38.5)
 def test_empty(self):
  self.assertIsNone(percentile([],95))
  self.assertIsNone(stats([]))
 def test_constant(self):
  self.assertEqual(percentile([10,10,10,10],95),10)
 def test_stats(self):
  s=stats([100,200,300,400]);self.assertEqual(s['mean_ms'],250)
  self.assertEqual(s['median_ms'],250)
  self.assertEqual(s['p95_ms'],385)
 def test_target_subset_and_unique(self):
  self.assertEqual(len(QWEN_SMOKE_TARGETS),4)
  self.assertEqual(len(set(QWEN_SMOKE_TARGETS)),4)
  self.assertTrue(set(QWEN_SMOKE_TARGETS).issubset(TARGETS))
 def test_holdout_refusal(self):
  with self.assertRaises(ValueError):refuse_restricted(Path('/workspace/holdout/images/test.png'))
 def test_warmup_count(self):
  keys=[QWEN_SMOKE_TARGETS[0],*QWEN_SMOKE_TARGETS]
  self.assertEqual(len(keys),5)
  self.assertEqual(len(keys)*3,15)
  self.assertEqual((len(keys)-1)*3,12)
 def test_refuse_uncontrolled_percentile(self):
  with self.assertRaises(ValueError):percentile([1,2,3],101)

if __name__=='__main__':unittest.main()
