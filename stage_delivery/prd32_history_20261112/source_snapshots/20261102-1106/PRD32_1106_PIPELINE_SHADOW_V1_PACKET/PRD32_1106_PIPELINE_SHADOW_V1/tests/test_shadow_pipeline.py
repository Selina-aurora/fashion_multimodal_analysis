from __future__ import annotations
import copy, unittest
from pathlib import Path
from run_shadow_pipeline import load_policy,jsonlines,validate_one,replay,validate_path
from runtime_identity import InstanceKey,make_request,attach_host_response
ROOT=Path(__file__).resolve().parents[1]

class TestHistoricalReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen=load_policy(ROOT/'inputs/v5_dev60_policy_171.csv')
        cls.all=list(r for _,r in jsonlines(ROOT/'inputs/qwen_prior32_192.jsonl'))
        cls.records={v:[r for r in cls.all if r['model_variant']==v] for v in ('baseline','focus_v1')}

    def test_frozen_scope(self):self.assertEqual(len(self.frozen),171)
    def test_sample_source_count(self):self.assertEqual((len(self.all),len(set((r['case_id'],r['target_instance_id']) for r in self.all))),(192,32))
    def test_fixture_metadata_consistent(self):
        rows,s=replay(self.frozen,self.records)
        self.assertEqual(len(rows),171);self.assertEqual(s['instances_with_qwen_records'],32)
        self.assertEqual(s['metadata_violation_counts'],{})
        self.assertEqual(s['qwen_records_seen'],192)
        self.assertFalse(s['production_approved'])
    def test_request_tamper(self):
        r=copy.deepcopy(self.all[0]);r['request_id']='fresh_dev_001__pred_0001__tags'
        self.assertIn('REQUEST_IDENTITY_OR_TASK_MISMATCH',validate_one(r,'baseline',self.frozen))
    def test_image_hash_tamper(self):
        r=copy.deepcopy(self.all[0]);r['image_sha256']='0'*64
        self.assertIn('SOURCE_IMAGE_SHA_MISMATCH',validate_one(r,'baseline',self.frozen))
    def test_v5_label_tamper(self):
        r=copy.deepcopy(self.all[0]);r['v5_category_unverified']='shoe'
        self.assertIn('V5_CATEGORY_SOURCE_MISMATCH',validate_one(r,'baseline',self.frozen))
    def test_bbox_tamper(self):
        r=copy.deepcopy(self.all[0]);r['v5_predicted_bbox_xyxy']=[1,5,1,6]
        self.assertIn('BAD_TARGET_BBOX',validate_one(r,'baseline',self.frozen))
    def test_unapproved_publication_tamper(self):
        r=copy.deepcopy(self.all[0]);r['publish_allowed']=True
        self.assertIn('SOURCE_UNAUTHORIZED_PUBLISH_FLAG',validate_one(r,'baseline',self.frozen))
    def test_unknown_instance(self):
        r=copy.deepcopy(self.all[0]);r['target_instance_id']='pred_9999'
        self.assertIn('UNKNOWN_OR_UNFROZEN_INSTANCE_KEY',validate_one(r,'baseline',self.frozen))
    def test_variant_tamper(self):
        r=copy.deepcopy(self.all[0]);r['model_variant']='focus_v1'
        self.assertIn('VARIANT_MISMATCH',validate_one(r,'baseline',self.frozen))
    def test_tags_parse_fail(self):
        r=copy.deepcopy(self.all[0]);r['tag_parse']['schema_valid']=False
        self.assertIn('TAGS_PARSE_INVALID',validate_one(r,'baseline',self.frozen))
    def test_duplicate_detection(self):
        a={v:list(rows) for v,rows in self.records.items()}
        a['baseline'].append(copy.deepcopy(self.all[0]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):replay(self.frozen,a)
    def test_missing_task_detected(self):
        a={v:[r for r in rows if not (r['case_id']=='fresh_dev_001' and r['target_instance_id']=='pred_0000' and r['task']=='copy')] for v,rows in self.records.items()}
        rows,_=replay(self.frozen,a)
        row=next(x for x in rows if x['case_id']=='fresh_dev_001' and x['instance_id']=='pred_0000')
        self.assertIn('INCOMPLETE_TASK_VARIANT_GRID',row['metadata_violations'])
    def test_cross_task_marked_mismatch_detected(self):
        a={v:[copy.deepcopy(r) for r in rows] for v,rows in self.records.items()}
        for r in a['baseline']:
            if r['case_id']=='fresh_dev_001' and r['target_instance_id']=='pred_0000' and r['task']=='copy':
                r['marked_image_sha256']='f'*64;break
        rows,_=replay(self.frozen,a)
        row=next(x for x in rows if x['case_id']=='fresh_dev_001' and x['instance_id']=='pred_0000')
        self.assertIn('CROSS_TASK_MARKED_IMAGE_MISMATCH',row['metadata_violations'])
    def test_holdout_paths_rejected(self):
        with self.assertRaises(ValueError):validate_path(Path('/workspace/HOLDOUT/qwen_results.jsonl'))
        with self.assertRaises(ValueError):validate_path(Path('/workspace/SEALEd/results.jsonl'),True)

class TestRuntimeEnvelope(unittest.TestCase):
    def setUp(self):
        self.key=InstanceKey('fresh_dev_001','pred_0000','a'*64)
        self.env=make_request(self.key,'tags','baseline','pred_0000','pred_0000',b'marked-image',(1,2,10,20))
    def test_creates_host_correlatable_envelope(self):
        self.assertEqual(self.env.request_id,'fresh_dev_001__pred_0000__tags')
    def test_cross_instance_roi_rejected(self):
        with self.assertRaisesRegex(ValueError,'Cross-instance'):make_request(self.key,'tags','baseline','pred_0001','pred_0000',b'marked-image',(1,2,10,20))
    def test_cross_instance_mask_rejected(self):
        with self.assertRaisesRegex(ValueError,'Cross-instance'):make_request(self.key,'tags','baseline','pred_0000','pred_0001',b'marked-image',(1,2,10,20))
    def test_request_id_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'request ID mismatch'):
            attach_host_response(self.env,completed_request_id='wrong',raw_text='hello',response_origin_instance_id='pred_0000')
    def test_host_origin_must_be_known(self):
        with self.assertRaisesRegex(ValueError,'origin is not evidenced'):
            attach_host_response(self.env,completed_request_id=self.env.request_id,raw_text='hello')
    def test_cross_instance_response_rejected(self):
        with self.assertRaisesRegex(ValueError,'Cross-instance'):
            attach_host_response(self.env,completed_request_id=self.env.request_id,raw_text='hello',response_origin_instance_id='pred_0001')
    def test_host_bound_still_not_publishable(self):
        x=attach_host_response(self.env,completed_request_id=self.env.request_id,raw_text='hello',response_origin_instance_id='pred_0000')
        self.assertFalse(x['publish_allowed'])

if __name__=='__main__':unittest.main()
