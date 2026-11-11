from __future__ import annotations
import json,sys, tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import next_batch as nxt

ROOT=Path(__file__).resolve().parents[1]
class TestNextBatch(unittest.TestCase):
    def test_known_five_regression_unchanged(self):
        e=ROOT/'evidence/known5'
        rows=nxt.bridge.readjl(e/'qwen_results.jsonl')
        result=nxt.guard.runtime_gate(rows,threshold=.8)
        prior=nxt.readj(e/'runtime_gate.json')
        self.assertEqual(result,prior)
        self.assertEqual(result['counts']['copy_blocked'],2)
        self.assertEqual(len(result['duplicate_candidate_pairs']),2)
        self.assertTrue(nxt.readj(e/'human_review_backtest.json')['in_sample_only'])
    def test_selection_by_id_no_human_context(self):
        rows=[{'case_id':x} for x in ['fresh_dev_001']*2+['fresh_dev_002']*2+['fresh_dev_003']*3+['fresh_dev_004']*3+['fresh_dev_005']*3+['fresh_dev_006']*1]
        self.assertEqual(nxt.choose_new_cases(rows),(['fresh_dev_002','fresh_dev_003','fresh_dev_005'],8))
        self.assertEqual(nxt.choose_new_cases(rows,max_images=2,max_instances=3),(['fresh_dev_002','fresh_dev_006'],3))
    def test_mask_iou(self):
        from PIL import Image
        import numpy as np
        with tempfile.TemporaryDirectory() as td:
            a=np.zeros((12,12),dtype='uint8');a[2:7,2:7]=255
            b=np.zeros((12,12),dtype='uint8');b[2:7,2:7]=255
            x=Path(td)/'a.png';y=Path(td)/'b.png'
            Image.fromarray(a).save(x);Image.fromarray(b).save(y)
            data=nxt.mask_overlap(x,y,[12,12])
            self.assertEqual(data['mask_iou'],1)
            b[8:11,8:11]=255;Image.fromarray(b).save(y)
            data=nxt.mask_overlap(x,y,[12,12]);self.assertLess(data['mask_iou'],1)
    def test_new_review_and_service_gate_synthetic(self):
        # Reuse 5 known records with synthetic NEW case IDs. This tests plumbing,
        # NOT actual predictions or out-of-sample performance.
        original=nxt.bridge.readjl(ROOT/'evidence/known5/qwen_results.jsonl')
        mapping={'fresh_dev_001':'fresh_dev_002','fresh_dev_004':'fresh_dev_003'}
        for r in original:
            r['case_id']=mapping[r['case_id']]
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);plan=base/'plan';gpu=base/'gpu';score=base/'score';plan.mkdir();gpu.mkdir()
            helper=ROOT/'frozen_helpers/prd32_integrated_pipeline_20261027.py'
            p={'schema':'prd32_nextbatch_gtfree_plan_v1','case_ids':['fresh_dev_002','fresh_dev_003'],
              'n_new_images':2,'n_new_v5_instances':5,'n_planned_qwen_requests':15,
              'v5_raw_sha256':'fake_for_synthetic_test','frozen_prompt_helper_sha256':nxt.digest(helper)}
            nxt.writej(plan/'next_batch_plan.json',p)
            nxt.writej(gpu/'gt_free_qwen_smoke_summary.json',{'status':'COMPLETED_FOR_REVIEW','failed_outputs':0,
              'n_requests':15,'n_images':2,'n_predicted_instances_processed':5,'v5_raw_sha256':'fake_for_synthetic_test'})
            nxt.writej(gpu/'smoke_provenance.json',{'helper_sha256':nxt.digest(helper),'GT_was_not_loaded':True,
                'case_ids':['fresh_dev_002','fresh_dev_003'],'instance_policy':'ALL_INSTANCES_IN_SELECTED_IMAGES'})
            nxt.bridge.writejl(gpu/'qwen_results.jsonl',original)
            nxt.score(SimpleNamespace(score_out=score,plan=plan,gpu_out=gpu))
            summary=nxt.readj(score/'service_gate_summary.json')
            self.assertEqual(summary['instances'],5)
            self.assertFalse(summary['human_review_completed'])
            self.assertEqual(summary['runtime_counts']['copy_blocked'],2)
            html=(score/'Review_NewInstances.html').read_text(encoding='utf-8')
            self.assertIn('我已亲自逐条核对',html)
            self.assertIn('UNREVIEWED',html)
            envelopes=nxt.bridge.readjl(score/'service_candidate_envelopes.jsonl')
            self.assertTrue(all(not r['publish_allowed'] for r in envelopes))
            self.assertTrue(all(r['case_id'] not in nxt.PREVIOUS_CASES for r in envelopes))
    def test_marked_image_embeds_only_from_smoke_dir(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'gpu';marks=root/'marked_inputs';marks.mkdir(parents=True)
            path=marks/'example.jpg';Image.new('RGB',(12,10),(250,250,250)).save(path)
            encoded=nxt.base64_img(str(path),marks)
            self.assertTrue(encoded.startswith('data:image/jpeg;base64,'))
            other=Path(td)/'other.jpg';Image.new('RGB',(10,10)).save(other)
            with self.assertRaisesRegex(ValueError,'OUTSIDE_APPROVED'):
                nxt.base64_img(str(other),marks)

    def test_fail_closed_malformed(self):
        rows=nxt.bridge.readjl(ROOT/'evidence/known5/qwen_results.jsonl')
        rows[0]['human_label_or_gt_bbox_used']=True
        with self.assertRaisesRegex(ValueError,'GT_CONTAMINATION'):
            nxt.guard.runtime_gate(rows)

if __name__=='__main__':unittest.main()
