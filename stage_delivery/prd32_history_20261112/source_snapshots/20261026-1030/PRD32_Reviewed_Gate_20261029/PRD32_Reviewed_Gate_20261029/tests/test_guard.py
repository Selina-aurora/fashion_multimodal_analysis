import copy
import json
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prd32_guard import read_jsonl, runtime_gate, offline_backtest, validate_review, iou

BASE=Path(__file__).resolve().parents[1]
ROWS=read_jsonl(BASE/'evidence/qwen_results.jsonl')
REVIEW=json.loads((BASE/'evidence/human_review_confirmed.json').read_text(encoding='utf-8'))

class GateTests(unittest.TestCase):
    def test_exact_confirmed_fixture(self):
        result=runtime_gate(ROWS)
        self.assertEqual((result['instances'],result['requests']), (5,15))
        self.assertEqual(result['counts']['copy_blocked'],2)
        self.assertEqual(result['counts']['duplicate_candidate_instances'],4)
        self.assertEqual(result['counts']['vqa_abstentions'],1)
        self.assertEqual(len(result['duplicate_candidate_pairs']),2)
        self.assertFalse(result['auto_merge_performed'])
        self.assertFalse(result['publish_allowed'])
        by={(r['case_id'],r['instance_id']):r for r in result['rows']}
        self.assertIn('COPY_CONTAINS_DEBUG_MARKER',by[('fresh_dev_004','pred_0000')]['copy_block_reasons'])
        self.assertIn('COPY_CATEGORY_CONFLICT_WITH_TAGS',by[('fresh_dev_004','pred_0001')]['copy_block_reasons'])
        self.assertIn('COPY_COLOR_CONFLICT_WITH_TAGS',by[('fresh_dev_004','pred_0001')]['copy_block_reasons'])
        self.assertEqual(by[('fresh_dev_004','pred_0002')]['copy_disposition'],'REVIEW_REQUIRED')

    def test_review_is_only_posthoc(self):
        gate=runtime_gate(ROWS)
        back=offline_backtest(gate, REVIEW)
        self.assertTrue(back['review_validation']['valid'])
        self.assertEqual(back['copy_human_errors_caught_n'],2)
        self.assertEqual(back['copy_false_positive_in_reviewed_5_n'],0)
        self.assertEqual(back['vqa_human_errors_flagged_n'],1)
        self.assertEqual(len([p for p in back['human_confirmed_duplicate_pairs'] if p['confirmation']=='BOTH_DUPLICATE']),2)
        draft=copy.deepcopy(REVIEW);draft['reviewed_by_human']=False
        self.assertFalse(validate_review(draft,{(r['case_id'],r['instance_id']) for r in gate['rows']})['valid'])
        # Modifying the human data doesn't change any runtime gate result.
        draft['rows'][0]['tags_quality']='CORRECT'
        self.assertEqual(gate, runtime_gate(ROWS))

    def test_provenance_fails_closed(self):
        x=copy.deepcopy(ROWS);x[0]['human_label_or_gt_bbox_used']=True
        with self.assertRaisesRegex(ValueError,'GT_CONTAMINATION'): runtime_gate(x)
        x=copy.deepcopy(ROWS);x[0]['target_bbox_origin']='human_gt'
        with self.assertRaisesRegex(ValueError,'GT_OR_UNKNOWN_BBOX_ORIGIN'):runtime_gate(x)
        x=copy.deepcopy(ROWS);x[0]['publish_allowed']=True
        with self.assertRaisesRegex(ValueError,'UNSAFE_PUBLISH'):runtime_gate(x)
        x=copy.deepcopy(ROWS);x.pop()
        with self.assertRaisesRegex(ValueError,'INCOMPLETE'): runtime_gate(x)
        x=copy.deepcopy(ROWS);x.append(copy.deepcopy(ROWS[0]))
        with self.assertRaisesRegex(ValueError,'DUPLICATE_TASK_REQUEST'):runtime_gate(x)

    def test_overlap_not_auto_merge(self):
        self.assertAlmostEqual(iou([0,0,10,10],[0,0,10,10]),1)
        self.assertEqual(iou([0,0,10,10],[20,20,30,30]),0)
        gate=runtime_gate(ROWS,threshold=.99)
        self.assertEqual(len(gate['duplicate_candidate_pairs']),0)
        self.assertEqual(gate['instances'],5)
        self.assertFalse(gate['auto_merge_performed'])

if __name__=='__main__': unittest.main(verbosity=2)
