from __future__ import annotations
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('focus_ab',ROOT/'prd32_focus_ab.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixtures():
    rows=[]
    recs={
        'fresh_dev_003':[
            ('pred_0000','dress',[10,10,110,110],.65),
            ('pred_0001','outerwear',[11,11,109,109],.61),
            ('pred_0002','top',[12,12,108,108],.73),
        ],
        'fresh_dev_005':[
            ('pred_0000','top',[0,0,40,40],.6),
            ('pred_0001','pants',[100,100,200,200],.7),
            ('pred_0002','skirt',[102,102,198,198],.9),
        ]
    }
    for cid,instances in recs.items():
        for inst,cat,box,score in instances:
            for task in m.TASKS:
                record={'request_id':f'{cid}__{inst}__{task}','case_id':cid,
                    'target_instance_id':inst,'task':task,'status':'GENERATED_REVIEW_REQUIRED',
                    'target_bbox_origin':'prd31_v5_live_prediction',
                    'human_label_or_gt_bbox_used':False,'publish_allowed':False,
                    'v5_category_unverified':cat,'v5_predicted_bbox_xyxy':box,
                    'v5_predicted_score_uncalibrated':score,
                    'marked_image':'/nonexistent/marked.jpg',
                    'raw_text':('{"category":"top","primary_color":"白色"}' if task=='tags' else '白色')}
                if task=='tags':record['tag_parse']={'canonical_service_json':{'category':'top','primary_color':'白色'}}
                rows.append(record)
    return rows


class Tests(unittest.TestCase):
    def test_complete_triplets(self):
        info=m.strict_verify(fixtures())
        self.assertEqual((info['instances'],info['requests']),(6,18))
        self.assertTrue(info['missing_provenance_evidence'])

    def test_strict_request_id(self):
        rows=fixtures();rows[0]['request_id']='fresh_dev_005__pred_0000__tags'
        with self.assertRaisesRegex(ValueError,'REQUEST_ID_MISMATCH'):m.strict_verify(rows)

    def test_missing_task(self):
        rows=fixtures()[:-1]
        with self.assertRaisesRegex(ValueError,'MISSING_TASKS'):m.strict_verify(rows)

    def test_duplicate_task(self):
        rows=fixtures();rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError,'DUPLICATE_REQUEST'):m.strict_verify(rows)

    def test_bbox_cross_task_tamper(self):
        rows=fixtures();rows[1]['v5_predicted_bbox_xyxy']=[0,0,1,1]
        with self.assertRaisesRegex(ValueError,'CROSS_TASK_MISMATCH'):m.strict_verify(rows)

    def test_marked_image_cross_task_tamper(self):
        rows=fixtures();rows[1]['marked_image']='/nonexistent/different.jpg'
        with self.assertRaisesRegex(ValueError,'CROSS_TASK_MISMATCH'):m.strict_verify(rows)

    def test_gt_and_publish_block(self):
        rows=fixtures();rows[0]['human_label_or_gt_bbox_used']=True
        with self.assertRaisesRegex(ValueError,'GT_LEAKAGE'):m.strict_verify(rows)
        rows=fixtures();rows[0]['publish_allowed']=True
        with self.assertRaisesRegex(ValueError,'UNSAFE_PUBLISH'):m.strict_verify(rows)

    def test_question_hash(self):
        rows=fixtures();r=next(r for r in rows if r['task']=='vqa')
        r['question_text']='颜色？';r['question_sha256']='invalid'
        with self.assertRaisesRegex(ValueError,'QUESTION_HASH_MISMATCH'):m.strict_verify(rows)

    def test_identical_marked_hash_must_match_bytes_if_present(self):
        rows=fixtures()
        with tempfile.TemporaryDirectory() as td:
            fp=Path(td)/'fake.jpg';fp.write_bytes(b'marked-image')
            for r in rows:
                r['marked_image']=str(fp);r['marked_image_sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'MARKED_IMAGE_FILE_HASH_MISMATCH'):m.strict_verify(rows)

    def test_duplicate_components_no_merging(self):
        gate={'rows':[],'duplicate_candidate_pairs':[
            {'case_id':'fresh_dev_003','instance_a':'pred_0000','instance_b':'pred_0001'},
            {'case_id':'fresh_dev_003','instance_a':'pred_0001','instance_b':'pred_0002'},
            {'case_id':'fresh_dev_005','instance_a':'pred_0001','instance_b':'pred_0002'},
        ]}
        for r in fixtures():
            if r['task']=='tags':gate['rows'].append({'case_id':r['case_id'],'instance_id':r['target_instance_id'],'v5_category':r['v5_category_unverified']})
        c=m.build_duplicate_groups(gate)
        self.assertEqual(sorted(x['size'] for x in c),[2,3])
        self.assertFalse(any(x['auto_deduplicated'] for x in c))
        self.assertFalse(any(x['primary_instance_selected'] for x in c))

    def test_prompt_no_ground_truth_and_copy_no_debug_words(self):
        class Helper:
            @staticmethod
            def make_prompt(rec, input_mode, pass_unverified):
                return 'old '+rec['task']
        for t in m.TASKS:
            p=m.build_prompt(Helper,{},t)
            self.assertIn('独立服饰实例',p)
            self.assertNotIn('human_gt',p)
            if t=='copy':self.assertIn('不要出现红框',p)

    def test_freeze_manifest_is_present(self):
        manifest=m.readj(ROOT/'frozen_source_hashes.json')
        self.assertEqual(len(manifest),4)
        self.assertTrue(all(len(s)==64 for s in manifest.values()))

if __name__=='__main__':unittest.main()
