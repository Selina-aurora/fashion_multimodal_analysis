import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_baseline as r

class MetricsBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen=r.load_frozen()
        cls.review=r.load_confirmed()
        cls.records=r.read_jsonl(r.BASE/'fixtures/targeted32_192.jsonl')
        cls.b=[x for x in cls.records if x['model_variant']=='baseline']
        cls.f=[x for x in cls.records if x['model_variant']=='focus_v1']
    def test_scope(self):
        self.assertEqual(len(self.frozen),171)
        self.assertEqual(len(set(k[0] for k in self.frozen)),60)
    def test_review_scope(self):
        self.assertEqual(len(self.review),12)
        self.assertEqual(sum(x['category_gt_is_determinate']=='YES' for x in self.review.values()),9)
    def test_category_one(self):
        x=next(x for x in self.b if x['case_id']=='fresh_dev_004' and x['target_instance_id']=='pred_0002' and x['task']=='tags')
        self.assertEqual(r.parsed_tag(x).get('category'),'pants')
    def test_validate_targeted_scope(self):
        result=r.validate_records(self.b,'baseline',self.frozen,False)
        self.assertEqual((result['record_count'],result['unique_instances']),(96,32))
    def test_full_rejects_partial(self):
        with self.assertRaises(ValueError):r.validate_records(self.b,'baseline',self.frozen,True)
    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            r.validate_records(self.b+[self.b[0]],'baseline',self.frozen,False)
    def test_bad_id_rejected(self):
        x=dict(self.b[0]);x['request_id']='wrong'
        with self.assertRaisesRegex(ValueError,'Request ID'):
            r.validate_records([x],'baseline',self.frozen,False)
    def test_bad_source_image_rejected(self):
        x=dict(self.b[0]);x['image_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Image SHA'):
            r.validate_records([x],'baseline',self.frozen,False)
    def test_bad_publish_rejected(self):
        x=dict(self.b[0]);x['publish_allowed']=True
        with self.assertRaisesRegex(ValueError,'Publish flag'):
            r.validate_records([x],'baseline',self.frozen,False)
    def test_forbidden_path(self):
        self.assertTrue(r.is_forbidden(Path('/workspace/holdout/test.jsonl')))
        self.assertFalse(r.is_forbidden(Path('/workspace/PRD32_DEV60_QWEN_BASELINE_20261030_run01/qwen_results.jsonl')))
    def test_percentile(self):
        self.assertEqual(r.percentile([1,2,3,4,5],.95),4.8)
    def test_category_accuracy(self):
        b=r.validate_records(self.b,'baseline',self.frozen,False)['index']
        f=r.validate_records(self.f,'focus_v1',self.frozen,False)['index']
        self.assertEqual((r.category_eval(b,self.review)['correct'],r.category_eval(b,self.review)['denominator']),(6,9))
        self.assertEqual((r.category_eval(f,self.review)['correct'],r.category_eval(f,self.review)['denominator']),(4,9))
    def test_metadata_not_gt(self):
        b=r.validate_records(self.b,'baseline',self.frozen,False)
        f=r.validate_records(self.f,'focus_v1',self.frozen,False)
        s,comparison,review=r.evaluate({'baseline':b,'focus_v1':f},self.frozen,self.review,'TARGETED_FIXTURE',{})
        self.assertEqual(len(comparison),171)
        self.assertEqual(len(review),171)
        self.assertEqual(sum(x['category_GT_status']=='CONFIRMED_DETERMINATE' for x in review),9)
        self.assertFalse(s['scope']['human_vqa_gt_available'])
    def test_vqa_timing(self):
        b=r.validate_records(self.b,'baseline',self.frozen,False)
        f=r.validate_records(self.f,'focus_v1',self.frozen,False)
        s,_,_=r.evaluate({'baseline':b,'focus_v1':f},self.frozen,self.review,'TARGETED_FIXTURE',{})
        x=s['metrics']['baseline']['task_timings']['vqa']['generation_ms']
        self.assertEqual((x['n'],x['count_le_threshold']),(32,20))
    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'review'
            s=r.run(mode='TARGETED_FIXTURE',baseline=None,focus=None,fixture=r.BASE/'fixtures/targeted32_192.jsonl',out=out)
            self.assertTrue((out/'baseline_report.md').is_file())
            self.assertFalse(s['holdout_opened'])
            with self.assertRaises(FileExistsError):r.run(mode='TARGETED_FIXTURE',baseline=None,focus=None,fixture=r.BASE/'fixtures/targeted32_192.jsonl',out=out)

if __name__=='__main__':unittest.main()
