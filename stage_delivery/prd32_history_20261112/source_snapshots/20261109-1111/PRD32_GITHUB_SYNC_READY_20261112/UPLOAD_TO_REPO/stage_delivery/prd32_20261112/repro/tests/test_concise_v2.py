import sys, unittest, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_vqa_concise_ab_v2 as m

class V2UnitTests(unittest.TestCase):
    def test_fixed_34_identity_and_v1_reference_hashes(self):
        rows,manifest=m.local_check()
        self.assertEqual(len(rows),34)
        self.assertEqual(sum(x['cohort']=='SLOW17' for x in rows),17)
        self.assertEqual(sum(x['cohort']=='CONTROL17' for x in rows),17)
    def test_prompt_v1_preserved_and_v2_different(self):
        self.assertNotEqual(m.sha_text(m.PROMPT_SUFFIX),m.sha_text(m.PROMPT_SUFFIX_V2))
        for token in ['无法确定','浅蓝色','黑色和白色','不要猜颜色']:
            self.assertIn(token,m.PROMPT_SUFFIX_V2)
        self.assertEqual(m.MAX_NEW_TOKENS,128)
    def test_review_reference_no_independent_gt(self):
        d=json.loads(m.V1_GPU_REFERENCE.read_text(encoding='utf8'))
        self.assertEqual(len(d),34)
        self.assertTrue(all(x['v1_gpu_result_is_independent_gt'] is False for x in d.values()))
    def test_marker_flag(self):
        self.assertIn('DEBUG_MARKER_TEXT',m.inspection_issues('红框的颜色是红色'))
    def test_summary_shadow_only(self):
        rows=[]
        for i in range(34):
            b={'v5_plus_mark_plus_qwen_ms':450,'generated_tokens':15}
            v1={'v5_plus_mark_plus_qwen_ms':290,'generated_tokens':4}
            v2={'v5_plus_mark_plus_qwen_ms':280,'generated_tokens':4}
            rows.append({'cohort':'SLOW17' if i<17 else 'CONTROL17','baseline':b,'concise':v1,'guarded_v2':v2,
                         'normalized_answer_matches':True,'v2_matches_v1':True,'candidate_flags':[]})
        out=m.summarize(rows)
        self.assertFalse(out['publish_allowed'])
        self.assertFalse(out['original_guard_invoked'])
        self.assertEqual(out['cohorts']['ALL34']['n'],34)
        self.assertEqual(out['cohorts']['ALL34']['v2_under_400ms'],34)

if __name__=='__main__':unittest.main()
