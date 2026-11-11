import csv,hashlib,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joint_eval_stage1_pilot12 as j
class Stage1Test(unittest.TestCase):
    def test_manifest_fixed(self):
        m,keys=j.load_manifest(j.DEFAULT_MANIFEST)
        self.assertEqual(len(keys),12)
        self.assertGreaterEqual(len(set(c for c,i in keys)),10)
        self.assertEqual(len({x['stratum'] for x in m['targets']}),8)
    def test_reference_join_and_no_prior(self):
        m,keys=j.load_manifest(j.DEFAULT_MANIFEST)
        rows=list(csv.DictReader((Path(__file__).resolve().parents[1]/'PILOT12_REVIEW_REFERENCE_NOT_MODEL_INPUT.csv').open(encoding='utf-8-sig')))
        self.assertEqual(set(keys),{(r['case_id'],r['instance_id']) for r in rows})
        self.assertTrue(all(int(r['warm_live_repeat_count'])==0 for r in rows))
        self.assertTrue(all(r['review_confirmed']=='YES' for r in rows))
    def test_guard_no_release_config(self):
        m,keys=j.load_manifest(j.DEFAULT_MANIFEST)
        self.assertIs(m['allow_release'],False)
        self.assertIs(m['independent_holdout_used'],False)
    def test_mutated_manifest_is_rejected(self):
        # Check digest gate without altering any source file.
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as d:
            p=Path(d)/'manifest.json'; p.write_text('{}')
            with self.assertRaises(ValueError):j.load_manifest(p)
if __name__=='__main__':unittest.main()
