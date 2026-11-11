#!/usr/bin/env python3
import unittest,sys,copy,json,csv,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from shadow_full43 import run,analyze_pairs,csvread,DUP,MIXED,UNCERTAIN
from binding_guard import check_binding,choose_category_state,make_report

class Full43ShadowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shadow=run()
        cls.bind=make_report()
        out=ROOT/'output'
        cls.group=csvread(out/'conflict_groups_full43.csv')
        cls.alias=csvread(out/'shadow_alias_proposals_strict.csv')
        cls.review=csvread(ROOT/'input/43_pairs_review_prefill_chat_confirmed.csv')
        cls.cases=[json.loads(s) for s in (ROOT/'input/v5_raw_predictions.jsonl').read_text(encoding='utf-8').splitlines() if s]
        cls.map={x['case_id']:x for x in cls.cases}
    def test_cardinality_and_no_mutation(self):
        self.assertEqual((self.shadow['source_images'],self.shadow['source_instances'],self.shadow['source_qwen_outputs_untouched']),(60,171,513))
        self.assertEqual(self.shadow['original_instances_changed'],0)
        self.assertEqual(self.shadow['qwen_answers_migrated'],0)
        self.assertFalse(self.shadow['holdout_opened'])
    def test_pair_partition(self):
        n=collections.Counter(x['pair_relationship'] for x in self.review)
        self.assertEqual(sum(n[x] for x in DUP),29)
        self.assertEqual(sum(n[x] for x in MIXED),11)
        self.assertEqual(sum(n[x] for x in UNCERTAIN),3)
    def test_no_alias_on_mixed_and_uncertain(self):
        self.assertEqual(len(self.group),25)
        self.assertEqual(len(self.alias),22)
        self.assertTrue(all(x['group_status']=='SHADOW_PROPOSAL_ONLY' and x['qwen_output_migrated']=='NO' for x in self.alias))
        for g in self.group:
            if g['cross_relation_count']!='0':self.assertTrue(g['status'].startswith('BLOCK_'))
    def test_triple_same_garment_group(self):
        g=[x for x in self.group if x['case_id']=='fresh_dev_003'];self.assertEqual(len(g),1)
        self.assertEqual(g[0]['member_count'],'3');self.assertEqual(g[0]['duplicate_pair_count'],'3')
        self.assertEqual(g[0]['proposed_canonical_id'],'pred_0002')
        dropped={x['suppressed_instance_id'] for x in self.alias if x['case_id']=='fresh_dev_003'}
        self.assertEqual(dropped,{'pred_0000','pred_0001'})
    def test_ambiguous_groups_still_blocked(self):
        blocked={g['case_id'] for g in self.group if g['status'].startswith('BLOCK_')}
        for case in ('fresh_dev_012','fresh_dev_042','fresh_dev_053','fresh_dev_060'):
            self.assertIn(case,blocked)
    def test_weak_relation_not_automatically_merged(self):
        self.assertFalse(any(x['case_id']=='fresh_dev_042' and x['proposed_canonical_instance_id']=='pred_0002' for x in self.alias))
    def test_identity_metadata_audit(self):
        self.assertEqual(self.bind['integrity_error_tasks'],0)
        self.assertEqual(self.bind['v5_tags_class_disagreement'],93)
        self.assertEqual(self.bind['v5_historically_correct_but_tags_disagree'],30)
    def test_swapped_instance_or_bad_bbox_fail(self):
        im=self.map['fresh_dev_001'];p=im['instances'][0]
        q=[json.loads(s) for s in (ROOT/'input/focus_v1_qwen_results_513.jsonl').read_text(encoding='utf-8').splitlines() if s]
        task=next(x for x in q if x['case_id']=='fresh_dev_001' and x['target_instance_id']==p['instance_id'])
        self.assertEqual(check_binding(p,im,task),[])
        bad=copy.deepcopy(task);bad['target_instance_id']='pred_9999'
        self.assertIn('INSTANCE_ID_MISMATCH',check_binding(p,im,bad))
        bad=copy.deepcopy(task);bad['image_sha256']='fakehash'
        self.assertIn('IMAGE_HASH_MISMATCH',check_binding(p,im,bad))
        bad=copy.deepcopy(task);bad['v5_predicted_bbox_xyxy']=[0,0,1,1]
        self.assertIn('BBOX_METADATA_MISMATCH',check_binding(p,im,bad))
    def test_class_conflict_not_auto_overwritten(self):
        self.assertEqual(choose_category_state('top','dress'),'CONFLICT_REVIEW_REQUIRED_KEEP_V5_UNVERIFIED')
        self.assertEqual(choose_category_state('top','top'),'AGREE_NOT_PROOF_OF_TRUTH')
    def test_stable_hash_and_provenance(self):
        self.assertEqual(len(self.shadow['input_sha256']),4)
        self.assertIn('EXPORTED_EDITED_CSV_NOT_PROVIDED',self.shadow['review_provenance'])

if __name__=='__main__':unittest.main()
