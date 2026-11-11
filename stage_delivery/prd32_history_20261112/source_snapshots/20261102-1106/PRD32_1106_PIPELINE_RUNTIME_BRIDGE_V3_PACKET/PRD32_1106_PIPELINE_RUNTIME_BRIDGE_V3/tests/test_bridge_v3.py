import json
import unittest
from hashlib import sha256
from pathlib import Path
from runtime_bridge_v3 import RuntimeGuardBridge, RuntimeBindingError
from verified_guard_loader import load_guard

BASE=Path(__file__).resolve().parents[1]
IMAGE=b'synthetic-original-for-smoke'
MASK=b'synthetic-mask-binary'
MARKED=b'synthetic-marked-with-target'
H=lambda b:sha256(b).hexdigest()

class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.b=RuntimeGuardBridge(load_guard(BASE/'reference_guard/prd32_output_guard.py'))
        self.kw={
            'case_id':'fresh_dev_001', 'instance_id':'pred_0000',
            'source_image_bytes':IMAGE,'expected_image_sha256':H(IMAGE),
            'bbox_xyxy':(1.0,2.0,100.0,150.0),'v5_category':'top',
            'roi_source_instance_id':'pred_0000','mask_source_instance_id':'pred_0000',
            'roi_crop_bytes':b'actual-crop','expected_roi_sha256':H(b'actual-crop'),
            'mask_bytes':MASK,'expected_mask_sha256':H(MASK),
            'marked_image_bytes':MARKED,'task':'tags','variant':'baseline',
            'job_id':'host_1','attempt_id':'run1','vqa_question':None,
            'duplicate_candidate':False}

    def dispatch(self, **change):
        a={**self.kw,**change}
        return self.b.dispatch(**a)

    def complete(self,job,rid,text='{"category":"top"}',**change):
        return self.b.complete(host_job_id=job,response_host_job_id=change.get('response_host_job_id',job),
                               completed_request_id=change.get('completed_request_id',rid),raw_text=text)

    def triplet(self,variant='baseline',inst='pred_0000',bbox=(1,2,100,150),v5='top',mask=MASK,marked=MARKED):
        req=[]
        for task in ('tags','vqa','copy'):
            job=f'host_{variant}_{inst}_{task}'
            rid=self.dispatch(task=task,variant=variant,instance_id=inst,
                              roi_source_instance_id=inst,mask_source_instance_id=inst,
                              bbox_xyxy=bbox,v5_category=v5,mask_bytes=mask,
                              expected_mask_sha256=H(mask),marked_image_bytes=marked,
                              job_id=job,vqa_question='What color?' if task=='vqa' else None)
            raw={'tags':json.dumps({'category':'top'}),'vqa':'blue','copy':'A blue top.'}[task]
            self.complete(job,rid,raw)
            req.append(rid)
        return req

    def test_synthetic_triplet_is_only_shadow(self):
        ids=self.triplet(); out=self.b.finalize_variant('baseline')
        self.assertEqual(out['instance_count'],1)
        self.assertEqual(len(set(ids)),3)
        self.assertFalse(out['production_approved'])
        self.assertEqual(out['guard_service_status'],'REVIEW_REQUIRED_NOT_RELEASED')
        self.assertNotIn('TAGS_REQUEST_ID_MISMATCH',out['guard_flags'])
    def test_two_instance_batch_keeps_overlap_cues(self):
        self.triplet()
        self.triplet(inst='pred_0001',bbox=(2,3,99,149),v5='pants',mask=b'other-mask')
        result=self.b.finalize_variant('baseline')
        self.assertEqual(result['instance_count'],2)
        self.assertEqual(len(result['guard_geometric_overlap_pairs']),1)
        self.assertIn('V5_QWEN_CATEGORY_DISAGREEMENT',result['guard_flags'])
    def test_baseline_focus_isolated(self):
        a=self.triplet(); b=self.triplet(variant='focus_v1')
        self.assertFalse(set(a)&set(b))
        self.assertEqual(self.b.finalize_variant('focus_v1')['instance_count'],1)
    def test_wrong_image_sha(self):
        with self.assertRaisesRegex(RuntimeBindingError,'image SHA256'):
            self.dispatch(expected_image_sha256='0'*64)
    def test_wrong_roi_sha(self):
        with self.assertRaisesRegex(RuntimeBindingError,'ROI crop SHA256'):
            self.dispatch(expected_roi_sha256='0'*64)
    def test_wrong_mask_sha(self):
        with self.assertRaisesRegex(RuntimeBindingError,'mask SHA256'):
            self.dispatch(expected_mask_sha256='0'*64)
    def test_wrong_roi_origin(self):
        with self.assertRaises(RuntimeBindingError):self.dispatch(roi_source_instance_id='pred_0001')
    def test_wrong_mask_origin(self):
        with self.assertRaises(RuntimeBindingError):self.dispatch(mask_source_instance_id='pred_0001')
    def test_wrong_job_id_completion(self):
        rid=self.dispatch()
        with self.assertRaises(RuntimeBindingError):self.complete('host_1',rid,response_host_job_id='other')
    def test_wrong_request_id_completion(self):
        rid=self.dispatch()
        with self.assertRaises(RuntimeBindingError):self.complete('host_1',rid,completed_request_id='wrong')
    def test_duplicate_response_blocked(self):
        rid=self.dispatch();self.complete('host_1',rid)
        with self.assertRaises(RuntimeBindingError):self.complete('host_1',rid)
    def test_missing_vqa_question(self):
        with self.assertRaisesRegex(RuntimeBindingError,'VQA question'):
            self.dispatch(task='vqa')
    def test_non_dev_case_blocked(self):
        with self.assertRaisesRegex(RuntimeBindingError,'DEV-only'):
            self.dispatch(case_id='holdout_001')
    def test_empty_mask_rejected(self):
        with self.assertRaises(RuntimeBindingError):self.dispatch(mask_bytes=b'')
    def test_empty_marked_rejected(self):
        with self.assertRaises(RuntimeBindingError):self.dispatch(marked_image_bytes=b'')
    def test_duplicate_request_rejected(self):
        self.dispatch()
        with self.assertRaises(RuntimeBindingError):self.dispatch(job_id='host_2')
    def test_incomplete_triplet_cannot_finalize(self):
        rid=self.dispatch();self.complete('host_1',rid)
        with self.assertRaisesRegex(RuntimeBindingError,'incomplete task triplet'):
            self.b.finalize_variant('baseline')
    def test_unfinished_job_cannot_finalize(self):
        self.dispatch()
        with self.assertRaisesRegex(RuntimeBindingError,'unfinished'):
            self.b.finalize_variant('baseline')
    def test_changed_bbox_fails(self):
        self.dispatch()
        with self.assertRaisesRegex(RuntimeBindingError,'facts changed'):
            self.dispatch(task='vqa',job_id='host_2',bbox_xyxy=(1,3,100,151),vqa_question='Color?')
    def test_changed_mask_fails(self):
        self.dispatch()
        with self.assertRaisesRegex(RuntimeBindingError,'facts changed'):
            self.dispatch(task='vqa',job_id='host_2',mask_bytes=b'other',expected_mask_sha256=H(b'other'),vqa_question='Color?')
    def test_changed_category_fails(self):
        self.dispatch()
        with self.assertRaisesRegex(RuntimeBindingError,'facts changed'):
            self.dispatch(task='vqa',job_id='host_2',v5_category='pants',vqa_question='Color?')
    def test_changed_marked_image_between_tasks_fails(self):
        self.triplet()
        # triplet already finalized not immutable; start new instance with mismatched marks
        for task,marked in [('tags',b'a'),('vqa',b'b'),('copy',b'a')]:
            job='job2_'+task
            rid=self.dispatch(instance_id='pred_0001',roi_source_instance_id='pred_0001',
                              mask_source_instance_id='pred_0001',job_id=job,task=task,
                              marked_image_bytes=marked,vqa_question='Color?' if task=='vqa' else None)
            self.complete(job,rid,'{"category":"top"}' if task=='tags' else 'blue')
        with self.assertRaisesRegex(RuntimeBindingError,'marked image changed'):
            self.b.finalize_variant('baseline')
    def test_guards_not_given_global_request_id(self):
        called=[]
        def wrap(rows):
            called.extend(rows)
            return load_guard(BASE/'reference_guard/prd32_output_guard.py')(rows)
        self.b=RuntimeGuardBridge(wrap)
        self.triplet()
        self.b.finalize_variant('baseline')
        self.assertNotIn('request_id',called[0])
        self.assertEqual(len(set(called[0][t]['request_id'] for t in ('tags','vqa','copy'))),3)
    def test_invalid_variant_blocks(self):
        with self.assertRaises(RuntimeBindingError):self.b.finalize_variant('unknown')
    def test_no_complete_results_blocks(self):
        with self.assertRaises(RuntimeBindingError):self.b.finalize_variant('baseline')
    def test_invalid_raw_response_rejected(self):
        rid=self.dispatch()
        with self.assertRaises(RuntimeBindingError):self.complete('host_1',rid,None)

if __name__=='__main__':unittest.main()
