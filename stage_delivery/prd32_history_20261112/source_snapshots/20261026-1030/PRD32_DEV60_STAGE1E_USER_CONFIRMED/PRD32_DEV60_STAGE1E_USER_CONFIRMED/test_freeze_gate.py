import unittest, tempfile, json
from pathlib import Path
from check_dev60_freeze_gate import parse_registry, cohort_checks, review_checks

class TestDev60Stage1e(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.d=Path(self.temp.name)
    def test_registry_missing(self):self.assertIsNone(parse_registry(None))
    def test_registry_valid(self):
        p=self.d/'approved_sha256_only.json';p.write_text(json.dumps({'sha256':['%064x'%i for i in range(8)]}))
        self.assertEqual(8,len(parse_registry(p)))
    def test_registry_not_8(self):
        p=self.d/'reg.txt';p.write_text('\n'.join('%064x'%i for i in range(7)))
        with self.assertRaises(ValueError):parse_registry(p)
    def test_registry_rejects_extra_keys(self):
        p=self.d/'reg.json';p.write_text(json.dumps({'sha256':['%064x'%i for i in range(8)],'paths':[]}))
        with self.assertRaises(ValueError):parse_registry(p)
    def test_registry_rejects_path_names(self):
        p=self.d/'reg.txt';p.write_text('path/to/sealed.jpg')
        with self.assertRaises(ValueError):parse_registry(p)
    def test_cohort_keeps_exact_sequence(self):
        rows=[dict(case_id=f'fresh_dev_{i:03}',image_sha256='%064x'%i) for i in range(1,61)]
        ref={'images':rows}
        self.assertTrue(cohort_checks(rows,ref))
    def test_cohort_detects_changed_sha(self):
        rows=[dict(case_id=f'fresh_dev_{i:03}',image_sha256='%064x'%i) for i in range(1,61)]
        ref={'images':[dict(x) for x in rows]}
        rows[22]['image_sha256']='0'*64
        with self.assertRaises(ValueError):cohort_checks(rows,ref)
    def test_cohort_detects_labels_leak(self):
        rows=[dict(case_id=f'fresh_dev_{i:03}',image_sha256='%064x'%i) for i in range(1,61)]
        ref={'images':[dict(x) for x in rows]};rows[0]['category']='shoe'
        with self.assertRaises(ValueError):cohort_checks(rows,ref)
    def test_review_all_confirmed(self):
        rows=[dict(case_id=f'fresh_dev_{i:03}',image_sha256='%064x'%i) for i in range(1,61)]
        rv={'schema':'prd32_dev60_new48_human_image_review_v1','review_completed':True,'ai_prefill_disclosure':True,'official_prd_acceptance':False,'rows':[dict(case_id=f'fresh_dev_{i:03}',image_sha256='%064x'%i,decision='ACCEPT',confirmed=True,visible_categories_human=['top']) for i in range(13,61)]}
        self.assertEqual(48,review_checks(rv,rows)['top'])
    def test_review_not_all_confirmed(self):
        rows=[dict(case_id=f'fresh_dev_{i:03}',image_sha256='%064x'%i) for i in range(1,61)]
        rv={'schema':'prd32_dev60_new48_human_image_review_v1','review_completed':True,'ai_prefill_disclosure':True,'official_prd_acceptance':False,'rows':[dict(case_id=f'fresh_dev_{i:03}',image_sha256='%064x'%i,decision='ACCEPT',confirmed=True,visible_categories_human=['top']) for i in range(13,61)]}
        rv['rows'][3]['confirmed']=False
        with self.assertRaises(ValueError):review_checks(rv,rows)
if __name__=='__main__':unittest.main()
