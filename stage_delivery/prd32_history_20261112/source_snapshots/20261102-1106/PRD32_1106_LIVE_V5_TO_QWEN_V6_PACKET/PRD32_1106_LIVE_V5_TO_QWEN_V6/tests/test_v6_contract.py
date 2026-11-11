import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from producer_contract_v6 import (
    TARGETS, V5_CKPT_SHA256, bbox_iou, canonical_mask_digest, canonical_sha,
    check_producer_match, digest_file, mask_iou, refuse_restricted, verified_event,
    verified_train_image,
)
from run_qwen_from_v5_producer_v6 import prepare_qwen_rows, evidence_join


class V6ContractTests(unittest.TestCase):
    def test_bbox_identity(self):
        self.assertAlmostEqual(bbox_iou([1,2,5,7],[1,2,5,7]),1)

    def test_bbox_disjoint(self):
        self.assertEqual(bbox_iou([1,2,5,7],[10,10,12,12]),0)

    def test_bbox_invalid(self):
        with self.assertRaises(ValueError):bbox_iou([3,3,1,5],[3,3,1,5])

    def test_mask_identity(self):
        a=np.array([[0,1],[1,0]],dtype=np.uint8)
        self.assertEqual(mask_iou(a,a),1)

    def test_mask_disjoint(self):
        self.assertEqual(mask_iou(np.array([[1,0]],dtype=np.uint8),np.array([[0,1]],dtype=np.uint8)),0)

    def test_mask_wrong_shape(self):
        with self.assertRaises(ValueError):mask_iou(np.ones((2,2)),np.ones((2,3)))

    def test_canonical_mask_does_not_depend_on_0_255(self):
        a=np.array([[0,1],[1,0]],dtype=np.uint8)
        self.assertEqual(canonical_mask_digest(a),canonical_mask_digest(a*255))

    def test_canonical_mask_rejects_empty(self):
        with self.assertRaises(ValueError):canonical_mask_digest(np.zeros((3,3),dtype=np.uint8))

    def test_producer_matches_frozen(self):
        ref={'instance_id':'pred_0002','expected_category':'bag','expected_bbox':[0,0,5,5],'expected_score':.8}
        ins={'instance_id':'pred_0002','category':'bag','bbox':[0,0,5,5],'score':.8}
        output=check_producer_match(target=ref,ins=ins,reference_mask=np.ones((5,5)),produced_mask=np.ones((5,5)))
        self.assertEqual(output['mask_iou'],1)

    def test_producer_changed_category(self):
        ref={'instance_id':'pred_0002','expected_category':'bag','expected_bbox':[0,0,5,5],'expected_score':.8}
        ins={'instance_id':'pred_0002','category':'top','bbox':[0,0,5,5],'score':.8}
        with self.assertRaises(ValueError):check_producer_match(target=ref,ins=ins,reference_mask=np.ones((5,5)),produced_mask=np.ones((5,5)))

    def test_producer_changed_mask(self):
        ref={'instance_id':'pred_0002','expected_category':'bag','expected_bbox':[0,0,5,5],'expected_score':.8}
        ins={'instance_id':'pred_0002','category':'bag','bbox':[0,0,5,5],'score':.8}
        with self.assertRaises(ValueError):check_producer_match(target=ref,ins=ins,reference_mask=np.eye(5),produced_mask=np.ones((5,5)))

    def test_restricted_paths(self):
        for p in ['/workspace/holdout_images/1.jpg','/workspace/sealed/test.png','/workspace/independent_test/x']:
            with self.assertRaises(ValueError):refuse_restricted(Path(p))

    def test_verified_train_image_reject_outside(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'train').mkdir();(root/'test').mkdir()
            x=root/'test'/'x.jpg';x.write_bytes(b'anything')
            with self.assertRaises(ValueError):verified_train_image(x,root,digest_file(x))

    def make_event(self, folder, tamper=None):
        mask=np.array([[0,255],[255,0]],dtype=np.uint8)
        path=folder/'mask.png';Image.fromarray(mask).save(path)
        e={'schema':'prd32_live_v5_mask_producer_event_v6',
           'status':'V5_RUNTIME_PRODUCER_CAPTURED_SHADOW_NOT_RELEASED',
           'case_id':'fresh_dev_004','instance_id':'pred_0002',
           'source_image_sha256':'abc',
           'producer_mask_sha256':canonical_mask_digest(mask),
           'mask_png_sha256':digest_file(path),
           'producer_bbox_xyxy':[0,0,2,2],
           'producer_category':'skirt',
           'v5_checkpoint_sha256':V5_CKPT_SHA256,
           'publish_allowed':False}
        e['producer_event_sha256']=canonical_sha(e)
        if tamper:e.update(tamper)
        ref={('fresh_dev_004','pred_0002'):{'image_sha256':'abc','expected_category':'skirt'}}
        return e,path,ref

    def test_event_verified(self):
        with tempfile.TemporaryDirectory() as d:
            e,path,ref=self.make_event(Path(d));self.assertTrue(verified_event(e,path,ref))

    def test_event_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            e,path,ref=self.make_event(Path(d),{'producer_category':'dress'})
            with self.assertRaises(ValueError):verified_event(e,path,ref)

    def test_event_mask_pixel_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            e,path,ref=self.make_event(Path(d))
            Image.fromarray(np.array([[255,0],[0,255]],dtype=np.uint8)).save(path)
            with self.assertRaises(ValueError):verified_event(e,path,ref)

    def test_prepare_rows_from_live_not_frozen_boxes(self):
        produced={};frozen={}
        for case,iid in TARGETS:
            produced[case,iid]={'producer_bbox_xyxy':[1,1,11,11],
                'producer_category':'bag','producer_score':.7,'producer_event_sha256':'h'}
            frozen[case,iid]={'image_path':'/example.jpg','image_sha256':'img','expected_bbox':[2,2,10,10]}
        got=prepare_qwen_rows(produced,frozen)
        self.assertEqual(len(got),2)
        self.assertEqual(got[0]['target_bbox_xyxy'],[1,1,11,11])

    def test_evidence_join_six_task_records(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            produced={};frozen={};rows=[];ev=[]
            from producer_contract_v6 import digest_bytes
            for case,iid in TARGETS:
                produced[case,iid]={'producer_bbox_xyxy':[1.,2.,11.,22.],
                    'producer_category':'bag','producer_event_sha256':case,
                    'producer_mask_sha256':'mask','source_image_sha256':'image'}
                frozen[case,iid]={'image_sha256':'image'}
                for task in ('tags','vqa','copy'):
                    req=f'{case}__{iid}__{task}__baseline__attempt_01';response='ok'
                    rows.append({'case_id':case,'target_instance_id':iid,'task':task,
                        'host_request_id':req,'image_sha256':'image','v5_category_unverified':'bag',
                        'marked_image_sha256':'marked','raw_text':response})
                    ev.append({'host_request_id':req,'bbox_xyxy':[1.,2.,11.,22.],
                               'response_sha256':digest_bytes(response.encode())})
            (out/'qwen_shadow_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (out/'host_call_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ev))
            report=evidence_join(out,produced,frozen,{'guard_service_status':'REVIEW_REQUIRED_NOT_RELEASED'})
            self.assertEqual(report['qwen_tasks'],6)
            self.assertFalse(report['mask_actually_fed_to_qwen'])
            self.assertEqual(len(json.loads((out/'producer_to_qwen_identity_join.json').read_text())),6)

    def test_evidence_join_rejects_wrong_guard(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises((ValueError,FileNotFoundError)):
                evidence_join(Path(d),{}, {},{'guard_service_status':'APPROVED'})

if __name__=='__main__':unittest.main()
