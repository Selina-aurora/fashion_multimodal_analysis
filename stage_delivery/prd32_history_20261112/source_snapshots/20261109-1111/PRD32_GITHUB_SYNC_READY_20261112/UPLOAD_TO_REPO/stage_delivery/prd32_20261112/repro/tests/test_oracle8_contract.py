import sys,unittest,tempfile
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_oracle8_ab_v1 as ab

class Oracle8ContractTests(unittest.TestCase):
 def test_manifest_groups(self):
  rows,questions,prior=ab.load_selection()
  self.assertEqual(len(rows),8)
  self.assertEqual(sum(x['eligible_for_oracle_comparison'] for x in rows),2)
  self.assertEqual(sum(x['ai_review_decision']=='needs_mask_or_target_review' for x in rows),1)
 def test_corrected_bbox_differs(self):
  rows,_,_=ab.load_selection()
  for x in rows:
   if x['eligible_for_oracle_comparison']:
    self.assertNotEqual(x['v5_bbox_xyxy'],x['oracle_bbox_xyxy'])
 def test_confirmations_not_blind_gt(self):
  rows,_,_=ab.load_selection()
  self.assertTrue(all(x['user_chat_confirmed_ai_assisted_review'] and not x['independent_blind_gt'] for x in rows))
 def test_prompt_is_truly_question_bearing(self):
  rows,qq,prior=ab.load_selection()
  for x in rows:
   q=qq[x['question_id']];prompt=ab.build_multitype_prompt(q['question'],q['type'])
   self.assertEqual(ab.sha_text(prompt),prior[x['question_id']]['prompt_sha256'])
   if x['ai_review_decision'] not in ('needs_mask_or_target_review','bbox_correction_candidate'):
    p=ab.neutralize_prompt(q,'marker')
    self.assertIn('标记框',p)
    self.assertNotIn('红框',p)
    self.assertEqual(p.count(q['question'].replace('红框','标记框')),1)
 def test_two_colors_only_differ_on_border(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);src=root/'x.png';Image.new('RGB',(100,90),(100,100,100)).save(src)
   a=root/'a.png';b=root/'b.png'
   ab.draw_color_border(src,[20,20,70,70],'red',a)
   ab.draw_color_border(src,[20,20,70,70],'cyan',b)
   with Image.open(a) as ia,Image.open(b) as ib:
    self.assertEqual(ia.getpixel((0,0)),ib.getpixel((0,0)))
    self.assertNotEqual(ia.getpixel((20,20)),ib.getpixel((20,20)))
 def test_reject_out_of_bounds_box(self):
  with self.assertRaises(ValueError):ab.check_bbox([0,0,999,40],100,100)
 def test_no_production_release_literal(self):
  rows,_,_=ab.load_selection()
  self.assertTrue(all(x['production_release'] is False and x['holdout_used'] is False for x in rows))

if __name__=='__main__': unittest.main()
