import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/'prd32_fixed_cohort_ab.py'
spec=importlib.util.spec_from_file_location('stage2_fixed_dev',SCRIPT)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class FixedDevTests(unittest.TestCase):
    def test_request_id_unique(self):
        self.assertEqual(len({f'c_{i}__pred_0000__{t}' for i in range(31) for t in m.TASKS}),93)
    def test_question_hash_consistent(self):
        self.assertEqual(m.text_sha(m.QUESTION), 'cf3731bf947065512007bfd63d5f4a8a19de07ca747ec0cdb51dd1c66e24a99e')
    def test_debug_filter(self):
        self.assertTrue(m.DEBUG.search('红色框内是一件上衣'))
        self.assertFalse(m.DEBUG.search('白色短袖，圆领'))
    def test_abstention_filter(self):
        self.assertTrue(m.ABSTAIN.search('无法确定。'))
        self.assertFalse(m.ABSTAIN.search('白色'))
    def make_fixture(self,base):
        known={}
        for i in range(31):
            cid=f'fresh_dev_{i:03d}';iid='pred_0000'
            known[(cid,iid)]={'case_id':cid,'target_instance_id':iid,
                'target_bbox_xyxy':[1,2,30,40],'predicted_category_v5_unverified':'top',
                'predicted_score_v5_uncalibrated':0.7,'image_sha256':'a'*64}
        f={'frozen_instance_manifest_sha256':'b'*64,'request_manifest_sha256':'c'*64}
        prompt_hashes={t:'d'*64 for t in m.TASKS}
        rows=[]
        for (cid,iid),rec in known.items():
            for t in m.TASKS:
                rows.append({'case_id':cid,'target_instance_id':iid,'task':t,'request_id':f'{cid}__{iid}__{t}',
                   'status':'GENERATED_REVIEW_REQUIRED','v5_predicted_bbox_xyxy':rec['target_bbox_xyxy'],
                   'v5_category_unverified':'top','v5_predicted_score_uncalibrated':0.7,
                   'image_sha256':'a'*64,'human_label_or_gt_bbox_used':False,'publish_allowed':False,
                   'prompt_sha256':prompt_hashes[t],'marked_image_sha256':'e'*64,
                   'question_text':m.QUESTION if t=='vqa' else None,
                   'question_sha256':m.text_sha(m.QUESTION) if t=='vqa' else None,
                   'raw_text':'白色' if t!='tags' else '{"category":"top"}',
                   'tag_parse':{'canonical_service_json':{'category':'top'}} if t=='tags' else None})
        p=Path(base);p.mkdir(exist_ok=True)
        m.jwrite(p/'provenance.json',{'variant':'baseline','instances_sha256':'b'*64,
            'requests_sha256':'c'*64,'prompt_hashes':prompt_hashes})
        (p/'qwen_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        return p,f,known,rows
    def test_valid_93_requests(self):
        with tempfile.TemporaryDirectory() as d:
            path,f,known,_=self.make_fixture(Path(d)/'run')
            rows=m.validate_run(path,f,known,'baseline')
            self.assertEqual(len(rows),93)
            self.assertEqual(m.stats(rows)['instances'],31)
    def test_reject_missing_request(self):
        with tempfile.TemporaryDirectory() as d:
            path,f,known,rows=self.make_fixture(Path(d)/'run')
            (path/'qwen_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows[:-1]))
            with self.assertRaisesRegex(ValueError,'INCOMPLETE_REQUESTS'):
                m.validate_run(path,f,known,'baseline')
    def test_reject_wrong_variant(self):
        with tempfile.TemporaryDirectory() as d:
            path,f,known,_=self.make_fixture(Path(d)/'run')
            with self.assertRaisesRegex(ValueError,'COHORT_PROVENANCE_MISMATCH'):
                m.validate_run(path,f,known,'focus_v1')
    def test_reject_changed_box(self):
        with tempfile.TemporaryDirectory() as d:
            path,f,known,rows=self.make_fixture(Path(d)/'run')
            rows[0]['v5_predicted_bbox_xyxy']=[0,0,50,50]
            (path/'qwen_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'CHANGED_FROZEN_INPUT'):
                m.validate_run(path,f,known,'baseline')
    def test_reject_wrong_vqa_question(self):
        with tempfile.TemporaryDirectory() as d:
            path,f,known,rows=self.make_fixture(Path(d)/'run')
            for r in rows:
                if r['task']=='vqa':r['question_text']='别的问题';break
            (path/'qwen_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'QUESTION_MISMATCH'):
                m.validate_run(path,f,known,'baseline')
    def test_reject_non_success(self):
        with tempfile.TemporaryDirectory() as d:
            path,f,known,rows=self.make_fixture(Path(d)/'run')
            rows[4]['status']='CASE_FAILED'
            (path/'qwen_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'INVALID_REQUEST_OR_STATUS'):
                m.validate_run(path,f,known,'baseline')

if __name__=='__main__':unittest.main()
