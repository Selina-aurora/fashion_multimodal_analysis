import csv,json,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from joint_eval_stage2_dev60 import manifest_verified
class ManifestTests(unittest.TestCase):
 def test_full_manifest(self):
  m,keys=manifest_verified()
  self.assertEqual(len(keys),171)
  self.assertEqual(len({c for c,i in keys}),60)
  self.assertEqual(len(set(keys)),171)
 def test_chunk_counts(self):
  m,_=manifest_verified()
  self.assertEqual([sum(r['chunk_id']==n for r in m['targets']) for n in range(1,9)],[24]*7+[3])
 def test_coverage_priority(self):
  m,_=manifest_verified()
  self.assertTrue(all(r['coverage_group']=='NEW_NEVER_LIVE_MEASURED' for r in m['targets'][:135]))
 def test_reference_scope(self):
  with (Path(__file__).resolve().parents[1]/'DEV60_171_REVIEW_REFERENCE_NOT_MODEL_INPUT.csv').open(encoding='utf-8-sig') as f:
   rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),171)
  self.assertTrue(all(r['reference_scope']=='USER_CHAT_CONFIRMED_DEV_REFERENCE_NOT_INDEPENDENT_GT' for r in rows))
 def test_bad_manifest_rejected(self):
  from tempfile import TemporaryDirectory
  with TemporaryDirectory() as d:
   p=Path(d)/'bad.json';p.write_text('{}')
   with self.assertRaises(ValueError):manifest_verified(p)
if __name__=='__main__':unittest.main()
