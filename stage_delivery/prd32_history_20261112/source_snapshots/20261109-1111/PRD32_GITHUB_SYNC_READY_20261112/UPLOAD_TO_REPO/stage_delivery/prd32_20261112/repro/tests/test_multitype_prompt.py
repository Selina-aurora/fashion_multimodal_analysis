import unittest, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_vqa_multitype_pilot_v1 as m

class TestQuestionPassing(unittest.TestCase):
    def test_all_32_questions_present(self):
        qs=m.read_plan()['questions']
        prompts=m.validate_question_prompts(qs,'冻结的旧颜色问题占位')
        self.assertEqual(len(prompts),32)
        self.assertEqual(len({q['question'] for q in qs}),len(set(prompts)))
        for q,p in zip(qs,prompts):
            self.assertEqual(p.count(q['question']),1)
            self.assertIn('用户当前问题：',p)

    def test_types_not_hardcoded_to_color(self):
        qs=m.read_plan()['questions']
        for q in qs:
            p=m.build_multitype_prompt(q['question'],q['type'])
            self.assertIn(q['question'],p)
            if q['type']=='styling_advice':self.assertIn('搭配建议',p)
            if q['type']=='material_process_boundary':self.assertIn('纤维成分',p)

    def test_stale_question_detected(self):
        qs=m.read_plan()['questions']
        with self.assertRaisesRegex(ValueError,'ORIGINAL_COLOR_QUESTION_REMAINS'):
            m.validate_question_prompts(qs,'用户当前问题：')

    def test_no_unsupported_type(self):
        with self.assertRaises(ValueError):m.build_multitype_prompt('问题','other')

    def test_no_empty_question(self):
        with self.assertRaises(ValueError):m.build_multitype_prompt(' ','attribute_visible')

if __name__=='__main__':unittest.main()
