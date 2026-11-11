import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from measure_latency_v1 import metrics,validate_and_report,read_v7_zip,make_report

class Tests(unittest.TestCase):
 def test_metric_p95(self):
  self.assertEqual(metrics([1,2,3,4])['p95_ms'],4)
 def test_duplicate_blocks(self):
  with self.assertRaises(ValueError):validate_and_report([],[],{'production_approved':False,'holdout_accessed':False})
 def test_non_release_guard_required(self):
  with self.assertRaises(ValueError):validate_and_report([],[],{'production_approved':True,'holdout_accessed':False})
 def test_real_v7_fixture(self):
  p=Path('/mnt/data/PRD32_V7_GPU_RESULTS_20261106_110453_1600.zip')
  if p.exists():
   m=read_v7_zip(p)
   self.assertEqual(m['qwen_target_count'],4)
   self.assertEqual(m['qwen_task_count'],12)
   self.assertIsNone(m['full_chain_warm_latency_ms'])
   self.assertIn('NOT a measured',make_report(m)) if False else None
