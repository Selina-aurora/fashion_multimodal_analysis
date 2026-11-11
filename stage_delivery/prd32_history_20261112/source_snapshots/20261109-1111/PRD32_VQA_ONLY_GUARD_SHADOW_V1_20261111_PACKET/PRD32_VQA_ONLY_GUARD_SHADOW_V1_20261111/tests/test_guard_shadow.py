import unittest,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from policy_vqa_only_v1 import evaluate,sha256_text

class GuardShadowTests(unittest.TestCase):
    def setUp(self):
        self.r={'case_id':'fresh_dev_001','instance_id':'pred_0000','route':'VQA_ONLY_SHADOW',
            'vqa_answer':'黑色','vqa_response_sha256':sha256_text('黑色'),
            'vqa_matches_frozen_baseline':True,'frozen_mask_iou':1.0,
            'production_release':False,'original_guard_invoked':False,'holdout_used':False}
        self.rev={'case_id':'fresh_dev_001','instance_id':'pred_0000',
            'ai_prefill_identity':'匹配','ai_prefill_mask':'正确','ai_prefill_vqa':'正确',
            'category_reference_status':'DETERMINATE'}
        self.base={'vqa_response_sha256':sha256_text('黑色')}
    def eval(self):return evaluate(self.r,self.rev,self.base)
    def test_clean_is_shadow_not_release(self):
        v=self.eval();self.assertEqual(v['primary_status'],'SHADOW_REVIEW_CANDIDATE');self.assertFalse(v['publish_allowed'])
    def test_abstain(self):
        self.r['vqa_answer']='无法确定';self.r['vqa_response_sha256']=sha256_text('无法确定');self.base['vqa_response_sha256']=sha256_text('无法确定')
        self.assertIn('MODEL_ABSTAINED',self.eval()['risk_reasons'])
    def test_long_debug(self):
        self.r['vqa_answer']='红框内目标服饰的主要可见颜色是黑白条纹。';self.r['vqa_response_sha256']=sha256_text(self.r['vqa_answer']);self.base['vqa_response_sha256']=sha256_text(self.r['vqa_answer'])
        self.assertIn('DEBUG_OR_TARGET_MARKER_TEXT',self.eval()['risk_reasons'])
    def test_partial_mask_blocks(self):
        self.rev['ai_prefill_mask']='部分正确';self.assertEqual(self.eval()['primary_status'],'BLOCKED_TARGET_EVIDENCE')
    def test_uncertain_identity_blocks(self):
        self.rev['ai_prefill_identity']='难判定';self.assertIn('TARGET_IDENTITY_NOT_FULLY_MATCHED',self.eval()['risk_reasons'])
    def test_missing_category_reference(self):
        self.rev['category_reference_status']='UNRESOLVED';self.assertIn('CATEGORY_REFERENCE_UNRESOLVED',self.eval()['risk_reasons'])
    def test_wrong_answer_review(self):
        self.rev['ai_prefill_vqa']='错误';self.assertIn('DEV_COLOR_QA_NOT_CONFIRMED_CORRECT',self.eval()['risk_reasons'])
    def test_hash_tamper(self):
        self.r['vqa_response_sha256']='0'*64;self.assertEqual(self.eval()['primary_status'],'BLOCKED_INTEGRITY')
    def test_frozen_drift(self):
        self.base['vqa_response_sha256']='0'*64;self.assertIn('FROZEN_BASELINE_OUTPUT_DRIFT',self.eval()['risk_reasons'])
    def test_original_guard_claim(self):
        self.r['original_guard_invoked']=True;self.assertEqual(self.eval()['primary_status'],'BLOCKED_INTEGRITY')
    def test_release_claim(self):
        self.r['production_release']=True;self.assertEqual(self.eval()['primary_status'],'BLOCKED_INTEGRITY')
    def test_holdout_claim(self):
        self.r['holdout_used']=True;self.assertEqual(self.eval()['primary_status'],'BLOCKED_INTEGRITY')
    def test_unsupported_question(self):
        self.assertIn('QUESTION_TYPE_NOT_VALIDATED',evaluate(self.r,self.rev,self.base,question_type='fabric')['risk_reasons'])
    def test_cross_target_join_refused(self):
        self.rev['instance_id']='pred_9999';self.assertRaises(ValueError,self.eval)
    def test_empty_answer(self):
        self.r['vqa_answer']='';self.r['vqa_response_sha256']=sha256_text('');self.base['vqa_response_sha256']=sha256_text('');self.assertIn('MISSING_VQA_OUTPUT',self.eval()['risk_reasons'])
    def test_non_source_lineage(self):
        self.r['frozen_mask_iou']=.5;self.assertIn('FROZEN_MASK_LINEAGE_MISMATCH',self.eval()['risk_reasons'])

if __name__=='__main__':unittest.main()
