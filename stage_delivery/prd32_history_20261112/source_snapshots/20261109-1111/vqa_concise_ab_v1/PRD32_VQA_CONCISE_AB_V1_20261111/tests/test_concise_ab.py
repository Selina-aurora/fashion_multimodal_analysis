import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_vqa_concise_ab_v1 import (load_selection, local_check, canonical_color_text,
                                   inspection_issues, summarize, sha_text, MAX_NEW_TOKENS)

class ConciseABTests(unittest.TestCase):
    def test_complete_fixed_cohorts(self):
        rows,manifest=local_check()
        self.assertEqual(len(rows),34)
        self.assertEqual(sum(r['cohort']=='SLOW17' for r in rows),17)
        self.assertEqual(sum(r['cohort']=='CONTROL17' for r in rows),17)
    def test_selection_no_holdout_or_gt(self):
        for r in load_selection():
            self.assertFalse(r['is_independent_gt'])
            self.assertTrue(r['case_id'].startswith('fresh_dev_'))
            self.assertEqual(r['frozen_vqa_sha256'],sha_text(r['frozen_vqa_answer']))
    def test_normalize_single_and_multi_colors(self):
        self.assertEqual(canonical_color_text('红框内目标服饰的主要可见颜色是黑白条纹。'),'黑白条纹')
        self.assertEqual(canonical_color_text('红框内目标服饰的主要可见颜色是橙色和白色。'),'橙色和白色')
        self.assertEqual(canonical_color_text('蓝色'),'蓝色')
        self.assertNotEqual(canonical_color_text('蓝色和白色'),canonical_color_text('蓝色'))
    def test_markers_rejected(self):
        self.assertIn('DEBUG_MARKER_TEXT',inspection_issues('红框内蓝色'))
        self.assertEqual(inspection_issues('黑白条纹'),[])
        self.assertEqual(inspection_issues('无法确定'),[])
    def test_cap_unmodified(self):
        self.assertEqual(MAX_NEW_TOKENS,128)
    def test_aggregate_fail_closed(self):
        rows=[]
        for i in range(34):
            ms=500 if i<17 else 240
            base={'v5_plus_mark_plus_qwen_ms':ms,'host_qwen_ms':ms-50,'generated_tokens':14 if i<17 else 2}
            cand={'v5_plus_mark_plus_qwen_ms':ms-100,'host_qwen_ms':ms-150,'generated_tokens':5}
            rows.append({'cohort':'SLOW17' if i<17 else 'CONTROL17','baseline':base,'concise':cand,
                'normalized_answer_matches':True,'candidate_flags':[]})
        report=summarize(rows)
        self.assertFalse(report['publish_allowed'])
        self.assertFalse(report['original_guard_invoked'])
        self.assertEqual(report['cohorts']['ALL34']['n'],34)
        self.assertEqual(report['cohorts']['SLOW17']['n'],17)
        self.assertEqual(report['cohorts']['CONTROL17']['n'],17)

if __name__=='__main__': unittest.main()
