import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import verify_frozen_v5_live_lineage as g


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False) + '\n', encoding='utf-8')


def jsonl(p, rows):
    p.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows), encoding='utf-8')


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        self.images=root/'fashion_data'/'raw'/'train'/'train'/'image'
        self.images.mkdir(parents=True)
        self.raw=root/'DEV60_V5_REAL'
        self.raw.mkdir()
        self.live=root/'LIVE_DEV_BASELINE'
        (self.live/'marked_inputs').mkdir(parents=True)
        raws=[]
        for idx in range(1,61):
            case=f'fresh_dev_{idx:03d}'
            img=self.images/f'{idx}.jpg'
            Image.new('RGB',(20,20),color=(idx,idx,idx)).save(img)
            n=3 if idx<=51 else 2
            ins=[]
            for j in range(n):
                ins.append({'instance_id':f'pred_{j:04d}',
                            'predicted_category':'top',
                            'bbox_xyxy':[1.,1.,18.,18.],
                            'bbox_coordinate_space':'original_image',
                            'prediction_origin':'V5_CHECKPOINT_LIVE_MODEL_OUTPUT_NOT_GT',
                            'score':0.8})
            raws.append({'case_id':case,'image_sha256':g.sha256_file(img),
                         'source_image_path':str(img),'image_size':[20,20],
                         'instances':ins,'instance_count':len(ins)})
        jsonl(self.raw/'v5_raw_predictions.jsonl',raws)
        dump(self.raw/'v5_real_summary.json',{'n_images':60,'n_predicted_instances':171,'checkpoint_sha256':g.V5_CKPT_SHA})
        dump(self.raw/'run_manifest.json',{'checkpoint_sha256':g.V5_CKPT_SHA,'human_gt_used_for_inference':False,'score_threshold':0.4,'mask_threshold':0.5})
        self.fixture_sha=g.sha256_file(self.raw/'v5_raw_predictions.jsonl')
        self.target=('fresh_dev_004','pred_0002')
        mask=Image.new('L',(20,20),0)
        for x in range(2,18):
            for y in range(2,18): mask.putpixel((x,y),255)
        mask.save(self.raw/'fresh_dev_004_pred_0002_mask.png')
        marked=self.live/'marked_inputs'/'marked.jpg'
        Image.new('RGB',(20,20),(250,0,0)).save(marked)
        self.marked_sha=g.sha256_file(marked)
        self.original=raws[3]
        result=[];events=[]
        for t in ('tags','vqa','copy'):
            request=f'fresh_dev_004__pred_0002__{t}__baseline__attempt_01'
            txt=f'synthetic answer for {t}'
            prompt_sha=hashlib.sha256(t.encode()).hexdigest()
            result.append({'case_id':'fresh_dev_004','target_instance_id':'pred_0002','task':t,'model_variant':'baseline',
                          'host_request_id':request,'image_sha256':self.original['image_sha256'],
                          'marked_image_sha256':self.marked_sha,'v5_bbox_xyxy':[1.,1.,18.,18.],
                          'prompt_sha256':prompt_sha,'raw_text':txt,'v5_category_unverified':'top',
                          'publish_allowed':False,'source_roi_identity_verified':False,'source_mask_identity_verified':False})
            events.append({'case_id':'fresh_dev_004','instance_id':'pred_0002','task':t,
                           'variant':'baseline','host_request_id':request,'job_id':request,
                           'image_sha256':self.original['image_sha256'],'marked_sha256':self.marked_sha,
                           'bbox_xyxy':[1.,1.,18.,18.],'prompt_sha256':prompt_sha,
                           'response_sha256':hashlib.sha256(txt.encode()).hexdigest(),
                           'status':'HOST_SYNCHRONOUS_CALL_CORRELATED_SHADOW_ONLY','release_allowed':False})
        self.results=result;self.events=events
        self.write_live()

    def write_live(self):
        jsonl(self.live/'qwen_shadow_results.jsonl',self.results)
        jsonl(self.live/'host_call_events.jsonl',self.events)
        dump(self.live/'run_summary.json',{'status':'DEV_ONLY_LIVE_SHADOW_COMPLETE_NOT_APPROVED','new_model_calls':3,
                                          'model_variant':'baseline','holdout_used':False,'publish_allowed':False})
        dump(self.live/'guard_shadow_report.json',{'status':'LIVE_HOST_CALLS_CORRELATED_BUT_ROI_MASK_VISUAL_UNVERIFIED',
                                                   'instances':1,'tasks_correlated':3,'holdout_accessed':False,
                                                   'release_allowed':False,'guard_service_status':'REVIEW_REQUIRED_NOT_RELEASED'})

    def audit(self):
        with patch.object(g,'V5_RAW_SHA',self.fixture_sha):
            return g.audit(self.raw,self.live,self.images.parents[3])

    def test_frozen_snapshot_to_live_valid(self):
        report=self.audit()
        self.assertEqual(report['selected_dev_instances'],1)
        self.assertEqual(report['host_tasks_checked'],3)
        self.assertFalse(report['roi_mask_producer_time_verified'])
        self.assertEqual(report['instances'][0]['mask_pixel_coverage_in_v5_bbox_margin2'],1.0)

    def test_wrong_v5_sha_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'V5_RAW_SHA_MISMATCH'):
            g.check_v5_raw(self.raw,self.images.parents[3])

    def test_mutated_response_rejected(self):
        self.results[0]['raw_text']='tampered'
        self.write_live()
        with self.assertRaisesRegex(ValueError,'RESPONSE_BYTES_MISMATCH'):self.audit()

    def test_cross_instance_id_rejected(self):
        self.events[0]['instance_id']='pred_0001'
        self.write_live()
        with self.assertRaisesRegex(ValueError,'HOST_IDENTITY_MISMATCH'):self.audit()

    def test_missing_mask_rejected(self):
        (self.raw/'fresh_dev_004_pred_0002_mask.png').unlink()
        with self.assertRaises(FileNotFoundError):self.audit()

    def test_wrong_mask_shape_rejected(self):
        Image.new('L',(9,9),255).save(self.raw/'fresh_dev_004_pred_0002_mask.png')
        with self.assertRaisesRegex(ValueError,'MASK_IMAGE_SHAPE_MISMATCH'):self.audit()

    def test_marked_file_tamper_rejected(self):
        Image.new('RGB',(20,20),(0,0,0)).save(self.live/'marked_inputs'/'marked.jpg')
        with self.assertRaisesRegex(ValueError,'MARKED_FILE_SHA_NOT_UNIQUE_OR_MISSING'):self.audit()

    def test_publish_true_rejected(self):
        self.results[0]['publish_allowed']=True
        self.write_live()
        with self.assertRaisesRegex(ValueError,'CLAIMED_UNPROVEN_ROI_MASK_OR_PUBLISH'):self.audit()

    def test_restricted_paths_rejected(self):
        self.assertTrue(g.restricted(Path('/workspace/my_holdout/images')))
        self.assertFalse(g.restricted(Path('/workspace/fashion_data/raw/train/train/image/a.jpg')))

    def test_invalid_bbox(self):
        with self.assertRaises(ValueError):g.checked_bbox([1,2,1,4])
        with self.assertRaises(ValueError):g.checked_bbox([1,2,float('nan'),4])


if __name__=='__main__':unittest.main()
