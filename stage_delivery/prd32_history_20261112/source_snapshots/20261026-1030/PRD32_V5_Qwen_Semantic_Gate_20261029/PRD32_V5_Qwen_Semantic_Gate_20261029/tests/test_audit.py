import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_3task import box_iou,copy_category,color_set, evaluate

class TestGate(unittest.TestCase):
    def test_box_iou(self):
        self.assertAlmostEqual(box_iou([0,0,10,10],[0,0,10,10]),1.)
        self.assertAlmostEqual(box_iou([0,0,10,10],[12,0,20,10]),0.)
    def test_copy_and_color(self):
        self.assertEqual(copy_category('红色方框内展示的是浅蓝色牛仔短裤'),'pants')
        self.assertEqual(color_set('红色方框内展示的是浅蓝色牛仔短裤',copy=True),{'蓝'})
        self.assertEqual(color_set('黄色'),{'黄'})
    def test_real_fixture(self):
        import json
        base=Path(__file__).resolve().parents[1]
        summary=json.loads((base/'inputs/gt_free_qwen_smoke_summary.json').read_text(encoding='utf-8'))
        rows=[json.loads(x) for x in (base/'inputs/qwen_results.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
        stats,instances,dupes,_=evaluate(summary,rows,'known')
        self.assertEqual(stats['requests'],15)
        self.assertEqual(stats['v5_qwen_category_conflicts'],2)
        self.assertEqual(stats['copy_tag_category_conflicts'],1)
        self.assertEqual(stats['copy_tag_color_conflicts'],1)
        self.assertEqual(stats['copy_debug_mentions'],2)
        self.assertEqual(stats['original_debug_guard_misses'],1)
        self.assertEqual(stats['vqa_abstentions'],1)
        self.assertEqual(stats['possible_duplicate_pairs'],2)
        self.assertEqual(stats['copy_display_blocked_instances'],2)
        self.assertFalse(stats['warnings'])

if __name__=='__main__':unittest.main()
