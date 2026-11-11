import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validate_new48_human_review import summarize,expected_from_audit

class ReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected=expected_from_audit(Path(__file__).resolve().parents[1]/'new48_selection_audit.csv')
    def draft(self, all_confirmed=True):
        return {'schema':'prd32_dev60_new48_human_image_review_v1','review_completed':all_confirmed,'rows':[
            {'case_id':key,'image_sha256':sha,'decision':'ACCEPT','visible_categories_human':['top'],'scene_flags_human':[], 'confirmed':all_confirmed,'note':''}
            for key,sha in sorted(self.expected.items())]}
    def test_48_sha_preserved(self):
        self.assertEqual(len(self.expected),48)
    def test_complete_accept(self):
        r=summarize(self.draft(),self.expected)
        self.assertEqual(r['accepted_images'],48)
        self.assertTrue(r['review_completion_verified_structurally'])
        self.assertFalse(r['cohort_frozen'])
    def test_unconfirmed_draft_not_approved(self):
        r=summarize(self.draft(False),self.expected)
        self.assertEqual(r['reviewed_confirmed_images'],0)
        self.assertEqual(r['accepted_images'],0)
    def test_reject_decreases_final_population(self):
        rev=self.draft();rev['rows'][0]['decision']='REJECT'
        r=summarize(rev,self.expected)
        self.assertEqual(r['remaining_to_target_if_only_accepted_retained'],1)
    def test_duplicate_case_rejected(self):
        rev=self.draft();rev['rows'][0]['case_id']=rev['rows'][1]['case_id']
        with self.assertRaises(ValueError):summarize(rev,self.expected)
    def test_mismatched_image_hash_rejected(self):
        rev=self.draft();rev['rows'][2]['image_sha256']='0'*64
        with self.assertRaises(ValueError):summarize(rev,self.expected)
    def test_fake_complete_flag_rejected(self):
        rev=self.draft();rev['rows'][0]['confirmed']=False
        with self.assertRaises(ValueError):summarize(rev,self.expected)
    def test_invalid_class_rejected(self):
        rev=self.draft();rev['rows'][0]['visible_categories_human']=['shoe','fake']
        with self.assertRaises(ValueError):summarize(rev,self.expected)
    def test_confirmed_but_unreviewed_rejected(self):
        rev=self.draft();rev['rows'][0]['decision']='UNREVIEWED'
        with self.assertRaises(ValueError):summarize(rev,self.expected)
    def test_partial_confirm_is_not_full(self):
        rev=self.draft();rev['rows'][0]['confirmed']=False;rev['review_completed']=False
        r=summarize(rev,self.expected)
        self.assertEqual(r['reviewed_confirmed_images'],47)
    def test_no_gt_or_model_approval(self):
        r=summarize(self.draft(),self.expected)
        self.assertFalse(r['ready_for_main_ab_regression'])
        self.assertFalse(r['holdout_sha_exclusion_verified'])

if __name__=='__main__':unittest.main()
