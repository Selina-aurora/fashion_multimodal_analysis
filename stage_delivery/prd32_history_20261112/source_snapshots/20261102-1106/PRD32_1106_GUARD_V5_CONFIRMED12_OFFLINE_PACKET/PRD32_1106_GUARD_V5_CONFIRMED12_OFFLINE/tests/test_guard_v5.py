"""Offline and source-integrity tests. Does not access holdout or GPUs."""
from __future__ import annotations
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from guard_v5_contract import evaluate_v5_candidate
from run_guard_v5 import build, validated_inputs, EXPECTED_SHA256

ROOT=Path(__file__).resolve().parents[1]


def load_table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def source():
    return {'case_id':'fresh_dev_004','instance_id':'pred_0002','image_sha256':'a'*64}


def candidate():
    s=source()
    return {**s,'query_instance_id':s['instance_id'],
            'roi_instance_id':s['instance_id'], 'mask_instance_id':s['instance_id'],
            'response_instance_id':s['instance_id'],
            'candidate_category':'pants', 'category_published':False,
            'attributes':[{'source_instance_id':s['instance_id']}]}


def confirmed_review():
    s=source()
    return {**s, 'review_status':'USER_CONFIRMED_IN_CHAT',
            'category_gt':'pants', 'instance_relation':'独立对象',
            'target_box':'正确对象','qwen_semantic':'两版本均一致'}


class TestV5Contract(unittest.TestCase):
    def test_metadata_perfect_still_shadow(self):
        r=evaluate_v5_candidate(source(),candidate())
        self.assertEqual(r['decision'],'SHADOW_ONLY_NOT_PRODUCTION_APPROVED')
        self.assertFalse(r['production_approved'])
    def test_any_category_publication_denied(self):
        r=evaluate_v5_candidate(source(),candidate())
        self.assertFalse(r['category_publication_approved'])
    def test_any_alias_fusion_denied(self):
        self.assertFalse(evaluate_v5_candidate(source(),candidate())['alias_merge_approved'])
    def test_query_mismatch_fails(self):
        c=candidate();c['query_instance_id']='pred_0001'
        self.assertEqual(evaluate_v5_candidate(source(),c)['decision'],'BLOCK_IDENTITY_OR_INTEGRITY')
    def test_response_mismatch_fails(self):
        c=candidate();c['response_instance_id']='pred_0001'
        self.assertIn('MISMATCH_RESPONSE_INSTANCE_ID',evaluate_v5_candidate(source(),c)['identity_violations'])
    def test_roi_mismatch_fails(self):
        c=candidate();c['roi_instance_id']='pred_0001'
        self.assertIn('MISMATCH_ROI_INSTANCE_ID',evaluate_v5_candidate(source(),c)['identity_violations'])
    def test_mask_mismatch_fails(self):
        c=candidate();c['mask_instance_id']='pred_0001'
        self.assertIn('MISMATCH_MASK_INSTANCE_ID',evaluate_v5_candidate(source(),c)['identity_violations'])
    def test_attributes_source_mismatch_fails(self):
        c=candidate();c['attributes'][0]['source_instance_id']='pred_0001'
        self.assertIn('ATTR_INSTANCE_CROSSOVER_0',evaluate_v5_candidate(source(),c)['identity_violations'])
    def test_missing_identity_fails(self):
        c=candidate();del c['query_instance_id']
        self.assertEqual(evaluate_v5_candidate(source(),c)['decision'],'BLOCK_IDENTITY_OR_INTEGRITY')
    def test_unapproved_alias_is_blocked(self):
        c=candidate();c['alias_target_instance_id']='pred_0000'
        self.assertIn('ALIAS_NOT_AUTHORIZED',evaluate_v5_candidate(source(),c)['identity_violations'])
    def test_shadow_alias_risk(self):
        c=candidate();c['shadow_alias_candidate']='YES'
        self.assertEqual(evaluate_v5_candidate(source(),c)['decision'],'BLOCK_REVIEW_OR_ALIAS_RISK')
    def test_incorrect_review_identity_fails(self):
        x=confirmed_review();x['instance_id']='pred_0005'
        self.assertIn('REVIEW_IDENTITY_MISMATCH',evaluate_v5_candidate(source(),candidate(),x)['identity_violations'])
    def test_review_sha_mismatch_fails(self):
        x=confirmed_review();x['image_sha256']='0'*64
        self.assertIn('REVIEW_IMAGE_SHA_MISMATCH',evaluate_v5_candidate(source(),candidate(),x)['identity_violations'])
    def test_pending_review_fails(self):
        x=confirmed_review();x['review_status']='PENDING'
        self.assertIn('REVIEW_NOT_USER_CONFIRMED',evaluate_v5_candidate(source(),candidate(),x)['identity_violations'])
    def test_duplicate_relation_blocks(self):
        x=confirmed_review();x['instance_relation']='同物重复'
        self.assertEqual(evaluate_v5_candidate(source(),candidate(),x)['decision'],'BLOCK_REVIEW_OR_ALIAS_RISK')
    def test_parent_part_relation_blocks(self):
        x=confirmed_review();x['instance_relation']='整体与局部'
        self.assertEqual(evaluate_v5_candidate(source(),candidate(),x)['decision'],'BLOCK_REVIEW_OR_ALIAS_RISK')
    def test_multiobject_blocks(self):
        x=confirmed_review();x['instance_relation']='不同对象混淆'
        self.assertEqual(evaluate_v5_candidate(source(),candidate(),x)['decision'],'BLOCK_REVIEW_OR_ALIAS_RISK')
    def test_partial_target_blocks(self):
        x=confirmed_review();x['target_box']='部分对象/混入多物'
        self.assertEqual(evaluate_v5_candidate(source(),candidate(),x)['decision'],'BLOCK_REVIEW_OR_ALIAS_RISK')
    def test_category_mismatch_flag_not_override(self):
        c=candidate();c['candidate_category']='skirt'
        r=evaluate_v5_candidate(source(),c,confirmed_review())
        self.assertIn('CANDIDATE_CATEGORY_DISAGREES_WITH_REVIEW',r['review_risk_flags'])
        self.assertFalse(r['category_publication_approved'])
    def test_unknown_gt_remains_unknown(self):
        x=confirmed_review();x['category_gt']='无法判断/不适用'
        r=evaluate_v5_candidate(source(),candidate(),x)
        self.assertIn('GT_CATEGORY_UNRESOLVED',r['review_risk_flags'])
    def test_semantic_mismatch_blocks(self):
        x=confirmed_review();x['qwen_semantic']='混合/无法判断'
        r=evaluate_v5_candidate(source(),candidate(),x)
        self.assertIn('CROSS_TASK_SEMANTIC_REVIEW_FLAG',r['review_risk_flags'])
    def test_unknown_relation_fails(self):
        x=confirmed_review();x['instance_relation']='any'
        self.assertIn('REVIEW_UNKNOWN_RELATION',evaluate_v5_candidate(source(),candidate(),x)['identity_violations'])


class TestV5Data(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            cls.summary=build(ROOT,Path(tmp)/'results')
        cls.confirmed=load_table(ROOT/'results'/'v5_user_confirmed_review_12.csv')
        cls.policy=load_table(ROOT/'results'/'v5_dev60_policy_171.csv')
        cls.comparison=load_table(ROOT/'results'/'v5_reviewed12_comparison.csv')
    def test_scope(self):
        self.assertEqual(len(self.policy),171)
    def test_confirmed12(self):
        self.assertEqual(len(self.confirmed),12)
        self.assertTrue(all(x['review_status']=='USER_CONFIRMED_IN_CHAT' for x in self.confirmed))
    def test_known9_unknown3(self):
        self.assertEqual(sum(x['category_gt_is_determinate']=='YES' for x in self.confirmed),9)
        self.assertEqual(sum(x['category_gt_is_determinate']=='NO' for x in self.confirmed),3)
    def test_alias22_unchanged(self):
        self.assertEqual(sum(x['shadow_alias_candidate']=='YES' for x in self.policy),22)
    def test_conflict93_unchanged(self):
        self.assertEqual(sum(x['category_agreement']=='NO' for x in self.policy),93)
    def test_same_or_parent_part10(self):
        self.assertEqual(sum(x['instance_relation'] in ('同物重复','整体与局部') for x in self.confirmed),10)
    def test_semantic9(self):
        self.assertEqual(sum(x['qwen_semantic']=='混合/无法判断' for x in self.confirmed),9)
    def test_user_attested_not_signed_export(self):
        self.assertTrue(all('NO_SEPARATE_EXPORTED' in x['review_provenance'] for x in self.confirmed))
    def test_no_mainline_release(self):
        self.assertTrue(all(x['v5_mainline_approved']=='NO' for x in self.policy))
        self.assertFalse(self.summary['mainline_deployment_approved'])
    def test_no_auto_category_override(self):
        self.assertTrue(all(x['v5_category_auto_override']=='NO' for x in self.policy))
    def test_no_auto_merge(self):
        self.assertTrue(all(x['v5_duplicate_auto_merge']=='NO' for x in self.policy))
    def test_known9_comparison_matches(self):
        self.assertEqual((self.summary['v5_match_of_confirmed9'],self.summary['qwen_baseline_match_of_confirmed9'],self.summary['qwen_focus_match_of_confirmed9']),(2,6,4))
    def test_all_confirmed12_blocked(self):
        self.assertEqual(sum(x['v5_policy']=='BLOCK_CONFIRMED_REVIEW_RISK' for x in self.policy),12)
    def test_v4_source_shas(self):
        self.assertEqual(set(self.summary['source_sha256'].keys()),set(EXPECTED_SHA256))
        self.assertEqual(self.summary['source_sha256'],EXPECTED_SHA256)
    def test_no_holdout_read(self):
        self.assertFalse(self.summary['independent_holdout_opened'])
    def test_inputs_mismatch_sha_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            t=Path(temp)/'inputs';t.mkdir()
            for p in (ROOT/'inputs').iterdir(): shutil.copy2(p,t/p.name)
            with (t/'v4_policy_decisions_171.csv').open('a',encoding='utf8') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'Frozen input SHA256 changed'):
                validated_inputs(Path(temp))
    def test_missing_source_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            t=Path(temp)/'inputs';t.mkdir()
            for p in (ROOT/'inputs').iterdir():
                if p.name!='v4_summary.json':shutil.copy2(p,t/p.name)
            with self.assertRaises(FileNotFoundError):
                validated_inputs(Path(temp))


if __name__=='__main__':unittest.main()
