import importlib.util
import hashlib
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('copy_gate',ROOT/'prd32_copy_gate_v2.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


def make_triplet():
    base=dict(case_id='fresh_dev_001',target_instance_id='pred_0000',
       status='GENERATED_REVIEW_REQUIRED',target_bbox_origin='prd31_v5_live_prediction',
       image_sha256='a'*64,marked_image_sha256='b'*64,
       v5_category_unverified='top',v5_predicted_bbox_xyxy=[1,2,10,11],
       human_label_or_gt_bbox_used=False,publish_allowed=False)
    rows=[]
    for task in ('tags','vqa','copy'):
        x=dict(base,task=task,request_id=f'fresh_dev_001__pred_0000__{task}',raw_text='测试')
        if task=='vqa':
            x['question_text']='红框内衣物颜色？'
            x['question_sha256']=hashlib.sha256(x['question_text'].encode()).hexdigest()
        rows.append(x)
    return rows

class Tests(unittest.TestCase):
    def test_red_rectangle_target_block(self):
        self.assertTrue(gate.detect_copy('红色矩形指向的服饰为白色绷带。')['hard_block_reasons'])
    def test_red_frame_inside_block(self):
        self.assertTrue(gate.detect_copy('红色框内展示的是上衣。')['hard_block_reasons'])
    def test_red_border_block(self):
        self.assertTrue(gate.detect_copy('红色边框内有一件衬衫。')['hard_block_reasons'])
    def test_red_rectangle_pattern_is_not_box(self):
        self.assertFalse(gate.detect_copy('裙子上有红色矩形图案。')['hard_block_reasons'])
    def test_red_color_does_not_block(self):
        self.assertFalse(gate.detect_copy('红色短裙，腰间有黑色扣子。')['hard_block_reasons'])
    def test_unsupported_structure_manual_not_hard(self):
        o=gate.detect_copy('黑色无裤腿分叉独立半身裙。')
        self.assertFalse(o['hard_block_reasons'])
        self.assertIn('UNVERIFIED_STRUCTURE_CLAIM',o['manual_review_flags'])
    def test_tactile_manual_not_hard(self):
        self.assertIn('SUBJECTIVE_OR_NONVISUAL_CLAIM',gate.detect_copy('柔软舒适的T恤。')['manual_review_flags'])
    def test_clean_is_not_approved(self):
        o=gate.detect_copy('白色上衣，圆领，纯色。')
        self.assertEqual(o['copy_disposition'],'REVIEW_REQUIRED');self.assertFalse(o['publish_allowed'])
    def test_complete_triplet(self):
        r=gate.run_gate(make_triplet())
        self.assertEqual(r['instances'],1);self.assertFalse(r['human_labels_used'])
    def test_request_id_mismatch_rejected(self):
        rows=make_triplet();rows[2]['request_id']='wrong'
        with self.assertRaises(ValueError):gate.run_gate(rows)
    def test_gt_contamination_rejected(self):
        rows=make_triplet();rows[2]['human_label_or_gt_bbox_used']=True
        with self.assertRaises(ValueError):gate.run_gate(rows)
    def test_mismatched_marked_image_rejected(self):
        rows=make_triplet();rows[2]['marked_image_sha256']='c'*64
        with self.assertRaises(ValueError):gate.run_gate(rows)
    def test_bad_vqa_hash_rejected(self):
        rows=make_triplet();rows[1]['question_sha256']='a'*64
        with self.assertRaises(ValueError):gate.run_gate(rows)
    def test_missing_task_rejected(self):
        with self.assertRaises(ValueError):gate.run_gate(make_triplet()[:2])
    def test_invalid_bbox_rejected(self):
        rows=make_triplet();rows[0]['v5_predicted_bbox_xyxy']=[1,2,1,10]
        with self.assertRaises(ValueError):gate.run_gate(rows)
    def test_empty_copy_is_blocked(self):
        self.assertIn('EMPTY_COPY_TEXT',gate.detect_copy('  ')['hard_block_reasons'])
    def test_bbox_literal_block(self):
        self.assertTrue(gate.detect_copy('bounding box shows a skirt')['hard_block_reasons'])

if __name__=='__main__':unittest.main()
