"""Deterministic standard-library tests; all synthetic inputs, no model/GT."""
from __future__ import annotations
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from fashion_multimodal_analysis.integration.prd32_output_guard import guard_outputs


def sample(case='one',inst='pred_0000'):
    return dict(case_id=case,instance_id=inst,v5_category='top',bbox_xyxy=[0,0,100,100],
                tags=dict(case_id=case,target_instance_id=inst,request_id='req-1',payload={'category':'top','pattern':'solid'}),
                vqa=dict(case_id=case,target_instance_id=inst,request_id='req-1',payload='白色'),
                copy=dict(case_id=case,target_instance_id=inst,request_id='req-1',payload='白色短袖上衣。'),
                request_id='req-1',vqa_question='衣服是什么颜色？')

class GuardTests(unittest.TestCase):
    def test_clean_record_still_requires_review(self):
        x=guard_outputs([sample()]);self.assertEqual(x['instances'][0]['flags'],[])
        self.assertFalse(x['copy_publish_allowed']);self.assertFalse(x['official_prd_acceptance'])
        self.assertEqual(x['instances'][0]['service_action'],'REVIEW_REQUIRED')
    def test_does_not_mutate_inputs(self):
        a=sample();b=copy.deepcopy(a);guard_outputs([a]);self.assertEqual(a,b)
    def test_category_disagreement_never_auto_corrects(self):
        a=sample();a['tags']['payload']['category']='skirt'
        out=guard_outputs([a]); self.assertIn('V5_QWEN_CATEGORY_DISAGREEMENT',out['instances'][0]['flags'])
        self.assertEqual(a['tags']['payload']['category'],'skirt')
        self.assertFalse(out['automatic_category_override_performed'])
    def test_fenced_tags_parsing(self):
        a=sample();a['tags']['payload']='```json\n{"category":"top"}\n```'
        self.assertEqual(guard_outputs([a])['instances'][0]['flags'],[])
    def test_malformed_tags_fail_closed(self):
        a=sample();a['tags']['payload']='hi';self.assertIn('TAGS_NOT_JSON_OBJECT',guard_outputs([a])['instances'][0]['flags'])
    def test_routing_mismatch(self):
        a=sample();a['copy']['target_instance_id']='other'
        self.assertIn('COPY_TARGET_INSTANCE_ID_MISMATCH',guard_outputs([a])['instances'][0]['flags'])
    def test_missing_routing_is_unverified(self):
        a=sample();a['copy']='白色上衣'
        self.assertIn('COPY_ROUTING_UNVERIFIED',guard_outputs([a])['instances'][0]['flags'])
    def test_request_id_mismatch(self):
        a=sample();a['vqa']['request_id']='wrong'
        self.assertIn('VQA_REQUEST_ID_MISMATCH',guard_outputs([a])['instances'][0]['flags'])
    def test_question_missing_not_auto_graded(self):
        a=sample();del a['vqa_question'];out=guard_outputs([a])
        self.assertEqual(out['instances'][0]['vqa_verification'],'question_missing')
    def test_abstention_is_not_assumed_wrong(self):
        a=sample();a['vqa']['payload']='无法确定';out=guard_outputs([a])
        self.assertIn('VQA_ABSTENTION_REQUIRES_QUESTION_REVIEW',out['instances'][0]['flags'])
    def test_frame_meta_blocks_copy(self):
        a=sample();a['copy']['payload']='红色边框内是白色上衣'
        self.assertIn('COPY_CONTAINS_FRAME_DEBUG_TEXT',guard_outputs([a])['instances'][0]['flags'])
    def test_no_auto_dedup_on_high_overlap(self):
        a=sample();b=sample(inst='pred_0001')
        for k in ['tags','vqa','copy']:b[k]['target_instance_id']='pred_0001'
        b['bbox_xyxy']=[1,1,99,99]
        out=guard_outputs([a,b]);self.assertEqual(len(out['instances']),2)
        self.assertEqual(len(out['geometric_overlap_pairs']),1)
        self.assertFalse(out['automatic_deduplication_performed'])
        self.assertFalse(out['geometric_overlap_pairs'][0]['confirmed_same_garment'])
    def test_different_case_never_geometry_pair(self):
        a=sample();b=sample(case='other')
        self.assertFalse(guard_outputs([a,b])['geometric_overlap_pairs'])
    def test_upstream_duplicate_does_not_delete_instance(self):
        a=sample();a['duplicate_candidate']=True
        out=guard_outputs([a]);self.assertEqual(len(out['instances']),1)
        self.assertIn('UPSTREAM_DUPLICATE_CANDIDATE',out['instances'][0]['flags'])
    def test_duplicate_record_identity_rejected(self):
        a=sample()
        with self.assertRaises(ValueError):guard_outputs([a,a])
    def test_invalid_bbox_flagged(self):
        a=sample();a['bbox_xyxy']=[0,0,-2,3]
        self.assertIn('INVALID_PREDICTED_BBOX',guard_outputs([a])['instances'][0]['flags'])
    def test_unsupported_category(self):
        a=sample();a['v5_category']='unknown'
        self.assertIn('INVALID_V5_CATEGORY',guard_outputs([a])['instances'][0]['flags'])
    def test_invalid_overlap_threshold(self):
        with self.assertRaises(ValueError):guard_outputs([sample()],duplicate_overlap=0)

if __name__=='__main__':unittest.main()
