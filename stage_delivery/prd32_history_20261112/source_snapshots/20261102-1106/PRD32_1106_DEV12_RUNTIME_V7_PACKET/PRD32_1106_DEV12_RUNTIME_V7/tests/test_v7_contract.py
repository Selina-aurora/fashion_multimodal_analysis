"""Deterministic CPU-only contract tests. No model forward is claimed."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from producer_contract_v7 import (
    TARGETS, QWEN_SMOKE_TARGETS, V5_CKPT_SHA256,
    canonical_sha, canonical_mask_digest, bbox_iou, mask_iou,
    check_producer_match, digest_file, refuse_restricted,
    verified_event, verified_train_image,
)
from run_qwen_from_v5_producer_v7 import prepare_qwen_rows, evidence_join, load_and_verify_producer


class CohortTests(unittest.TestCase):
    def test_twelve_unique_targets(self):
        self.assertEqual(len(TARGETS), 12)
        self.assertEqual(len(set(TARGETS)), 12)

    def test_ten_unique_dev_images(self):
        self.assertEqual(len({case for case, _ in TARGETS}), 10)

    def test_four_distinct_smoke_targets(self):
        self.assertEqual(len(QWEN_SMOKE_TARGETS), 4)
        self.assertEqual(len(set(QWEN_SMOKE_TARGETS)), 4)
        self.assertTrue(set(QWEN_SMOKE_TARGETS).issubset(set(TARGETS)))

    def test_no_holdout_key(self):
        self.assertTrue(all(case.startswith('fresh_dev_') and iid.startswith('pred_') for case,iid in TARGETS))


class GeometryTests(unittest.TestCase):
    def test_bbox_identity(self):
        self.assertEqual(bbox_iou([1,2,3,4],[1,2,3,4]), 1)
    def test_bbox_disjoint(self):
        self.assertEqual(bbox_iou([1,2,3,4],[5,6,7,8]), 0)
    def test_bbox_degenerate_rejected(self):
        with self.assertRaises(ValueError):bbox_iou([1,2,1,4],[1,2,1,4])
    def test_nan_rejected(self):
        with self.assertRaises(ValueError):bbox_iou([0,0,float('nan'),2],[0,0,1,2])
    def test_mask_identity(self):
        a=np.eye(3, dtype=np.uint8)
        self.assertEqual(mask_iou(a,a),1)
    def test_mask_disjoint(self):
        self.assertEqual(mask_iou([[1,0]],[[0,1]]),0)
    def test_mask_shape_rejected(self):
        with self.assertRaises(ValueError):mask_iou(np.ones((2,2)),np.ones((2,3)))
    def test_mask_empty_rejected(self):
        with self.assertRaises(ValueError):mask_iou([[0,0]],[[0,0]])
    def test_mask_digest_encoding_invariant(self):
        a=np.eye(3,dtype=np.uint8)
        self.assertEqual(canonical_mask_digest(a),canonical_mask_digest(a*255))
    def test_mask_digest_rejects_empty(self):
        with self.assertRaises(ValueError):canonical_mask_digest(np.zeros((3,3),dtype=np.uint8))
    def test_producer_matches(self):
        t={'instance_id':'pred_0000','expected_category':'bag','expected_bbox':[0,0,3,3],'expected_score':.65}
        p={'instance_id':'pred_0000','category':'bag','bbox':[0,0,3,3],'score':.65}
        self.assertEqual(check_producer_match(target=t,ins=p,reference_mask=np.ones((3,3)),produced_mask=np.ones((3,3)))['mask_iou'],1)
    def test_producer_changed_class_blocked(self):
        t={'instance_id':'pred_0000','expected_category':'bag','expected_bbox':[0,0,3,3],'expected_score':.65}
        p={'instance_id':'pred_0000','category':'top','bbox':[0,0,3,3],'score':.65}
        with self.assertRaises(ValueError):check_producer_match(target=t,ins=p,reference_mask=np.ones((3,3)),produced_mask=np.ones((3,3)))
    def test_producer_changed_mask_blocked(self):
        t={'instance_id':'pred_0000','expected_category':'bag','expected_bbox':[0,0,3,3],'expected_score':.65}
        p={'instance_id':'pred_0000','category':'bag','bbox':[0,0,3,3],'score':.65}
        with self.assertRaises(ValueError):check_producer_match(target=t,ins=p,reference_mask=np.eye(3),produced_mask=np.ones((3,3)))
    def test_restricted_paths_rejected(self):
        for path in ('/workspace/holdout/x.jpg','/workspace/sealed/input','/workspace/independent_test/a','/workspace/blind/a'):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):refuse_restricted(Path(path))
    def test_outside_train_image_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'train').mkdir();(root/'test').mkdir()
            p=root/'test'/'x.jpg';p.write_bytes(b'x')
            with self.assertRaises(ValueError):verified_train_image(p,root,digest_file(p))


class EventAndJoinTests(unittest.TestCase):
    def make_event(self, d):
        mask=np.array([[0,255],[255,0]],dtype=np.uint8)
        p=d/'mask.png';Image.fromarray(mask).save(p)
        e={'schema':'prd32_live_v5_mask_producer_event_v6',
           'status':'V5_RUNTIME_PRODUCER_CAPTURED_SHADOW_NOT_RELEASED',
           'case_id':'fresh_dev_004', 'instance_id':'pred_0002',
           'source_image_sha256':'abc','producer_mask_sha256':canonical_mask_digest(mask),
           'mask_png_sha256':digest_file(p),'producer_bbox_xyxy':[0,0,2,2],
           'producer_category':'skirt','v5_checkpoint_sha256':V5_CKPT_SHA256,
           'publish_allowed':False}
        e['producer_event_sha256']=canonical_sha(e)
        ref={('fresh_dev_004','pred_0002'):{'image_sha256':'abc','expected_category':'skirt'}}
        return e,p,ref

    def test_event_sha_and_pixels(self):
        with tempfile.TemporaryDirectory() as d:
            e,p,ref=self.make_event(Path(d))
            self.assertTrue(verified_event(e,p,ref))
    def test_tampered_event_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            e,p,ref=self.make_event(Path(d));e['producer_category']='dress'
            with self.assertRaises(ValueError):verified_event(e,p,ref)
    def test_tampered_mask_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            e,p,ref=self.make_event(Path(d));Image.fromarray(np.array([[255,0],[0,255]],dtype=np.uint8)).save(p)
            with self.assertRaises(ValueError):verified_event(e,p,ref)

    def fixture_join(self, d):
        out=Path(d);(out/'marked_inputs').mkdir()
        produced={};frozen={};qwen=[];events=[]
        marked=b'fakejpgbytes';mark_sha=hashlib.sha256(marked).hexdigest()
        (out/'marked_inputs'/'dummy_marked_full_image.jpg').write_bytes(marked)
        for case,iid in TARGETS:
            produced[case,iid]={'producer_bbox_xyxy':[1.,2.,11.,22.],
                'producer_category':'bag','producer_event_sha256':f'{case}-{iid}',
                'producer_mask_sha256':'mask','source_image_sha256':'image', 'producer_score':.8}
            frozen[case,iid]={'image_path':'/example.jpg','image_sha256':'image','expected_bbox':[2,2,10,10]}
        for case,iid in QWEN_SMOKE_TARGETS:
            for task in ('tags','vqa','copy'):
                rid=f'{case}__{iid}__{task}__baseline__attempt_01'
                qwen.append({'case_id':case,'target_instance_id':iid,'task':task,
                    'host_request_id':rid,'image_sha256':'image','v5_category_unverified':'bag',
                    'v5_bbox_xyxy':[1.,2.,11.,22.], 'marked_image_sha256':mark_sha,'raw_text':'ok'})
                events.append({'host_request_id':rid,'case_id':case,'instance_id':iid,'task':task,
                    'image_sha256':'image','marked_sha256':mark_sha,
                    'bbox_xyxy':[1.,2.,11.,22.],
                    'response_sha256':hashlib.sha256(b'ok').hexdigest()})
        def save():
            (out/'qwen_shadow_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in qwen),encoding='utf8')
            (out/'host_call_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events),encoding='utf8')
        save()
        return out,produced,frozen,qwen,events,save

    def test_prepare_rows_uses_producer_bbox(self):
        with tempfile.TemporaryDirectory() as d:
            _,produced,frozen,*_=self.fixture_join(d)
            got=prepare_qwen_rows(produced,frozen)
            self.assertEqual(len(got),4)
            self.assertEqual(got[0]['target_bbox_xyxy'],[1.,2.,11.,22.])

    def test_four_target_twelve_task_join(self):
        with tempfile.TemporaryDirectory() as d:
            out,produced,frozen,*_=self.fixture_join(d)
            report=evidence_join(out,produced,frozen,{'guard_service_status':'REVIEW_REQUIRED_NOT_RELEASED'})
            self.assertEqual(report['qwen_tasks'],12)
            self.assertEqual(report['producer_instances'],12)
            self.assertFalse(report['mask_actually_fed_to_qwen'])
            self.assertFalse(report['production_approved'])
            self.assertEqual(len(json.loads((out/'producer_to_qwen_identity_join.json').read_text())),12)

    def test_missing_task_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            out,produced,frozen,q,events,save=self.fixture_join(d)
            q.pop();save()
            with self.assertRaises(ValueError):evidence_join(out,produced,frozen,{'guard_service_status':'REVIEW_REQUIRED_NOT_RELEASED'})

    def test_wrong_target_bbox_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            out,produced,frozen,q,events,save=self.fixture_join(d)
            events[0]['bbox_xyxy']=[2.,2.,11.,22.];save()
            with self.assertRaises(ValueError):evidence_join(out,produced,frozen,{'guard_service_status':'REVIEW_REQUIRED_NOT_RELEASED'})

    def test_wrong_response_sha_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            out,produced,frozen,q,events,save=self.fixture_join(d)
            q[0]['raw_text']='tampered';save()
            with self.assertRaises(ValueError):evidence_join(out,produced,frozen,{'guard_service_status':'REVIEW_REQUIRED_NOT_RELEASED'})

    def test_bad_marked_image_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            out,produced,frozen,q,events,save=self.fixture_join(d)
            (out/'marked_inputs'/'dummy_marked_full_image.jpg').unlink()
            with self.assertRaises(ValueError):evidence_join(out,produced,frozen,{'guard_service_status':'REVIEW_REQUIRED_NOT_RELEASED'})

    def test_wrong_guard_status_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            out,produced,frozen,*_=self.fixture_join(d)
            with self.assertRaises(ValueError):evidence_join(out,produced,frozen,{'guard_service_status':'APPROVED'})

    def test_duplicate_host_request_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            out,produced,frozen,q,events,save=self.fixture_join(d)
            events[1]['host_request_id']=events[0]['host_request_id'];save()
            with self.assertRaises(ValueError):evidence_join(out,produced,frozen,{'guard_service_status':'REVIEW_REQUIRED_NOT_RELEASED'})

    def test_cross_case_identity_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            out,produced,frozen,q,events,save=self.fixture_join(d)
            events[0]['case_id']='fresh_dev_999';save()
            with self.assertRaises(ValueError):evidence_join(out,produced,frozen,{'guard_service_status':'REVIEW_REQUIRED_NOT_RELEASED'})


if __name__=='__main__':unittest.main()
