"""模型之外的回归测试：几何保护、坐标和实例保留，不验证模型精度。"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import sys
import types

import numpy as np
from PIL import Image
from fashion_multimodal_analysis.integration.quality_guards import (
    constrain_local_mask, overlap_review_flags,
)
from fashion_multimodal_analysis.integration.pipeline import GarmentROI, FashionPipeline


class QualityGuardTests(unittest.TestCase):
    def test_mask_does_not_expand_past_proposal(self):
        mask = constrain_local_mask(np.ones((8, 10)), np.ones((8, 10)), [2, 1, 5, 4])
        self.assertEqual(int(mask.sum()), 9)
        self.assertFalse(mask[:, 5:].any())

    def test_mask_stays_inside_garment(self):
        garment = np.zeros((8, 10)); garment[2:3, 3:4] = 1
        mask = constrain_local_mask(np.ones((8, 10)), garment, [2, 1, 5, 4])
        self.assertEqual(int(mask.sum()), 1)

    def test_mask_shape_mismatch_is_error(self):
        with self.assertRaises(ValueError):
            constrain_local_mask(np.ones((5, 5)), np.ones((8, 10)), [2, 1, 5, 4])

    def test_overlap_marks_both_without_deleting(self):
        rois = [SimpleNamespace(instance_id='a', category='dress', bbox=[0, 0, 10, 10]),
                SimpleNamespace(instance_id='b', category='top', bbox=[1, 1, 9, 9])]
        flags = overlap_review_flags(rois)
        self.assertEqual(len(rois), 2)
        self.assertEqual(flags['a'][0]['other_instance_id'], 'b')
        self.assertEqual(flags['b'][0]['automatic_action'], 'none')

    def test_disjoint_instances_not_flagged(self):
        rois = [SimpleNamespace(instance_id='a', category='shoe', bbox=[0, 0, 4, 4]),
                SimpleNamespace(instance_id='b', category='shoe', bbox=[5, 5, 9, 9])]
        self.assertEqual(overlap_review_flags(rois), {'a': [], 'b': []})

    def test_pipeline_preserves_predictions_and_flags_sleeveless(self):
        image = Image.new('RGB', (10, 10))
        rois = [GarmentROI('a', 'dress', .8, [0, 0, 10, 10], np.ones((10, 10))),
                GarmentROI('b', 'top', .7, [1, 1, 9, 9], np.ones((10, 10)))]
        backend = SimpleNamespace(
            segment=lambda image: (rois, 1),
            ground=lambda image, roi, region: ({'status': 'missed'}, 1),
            attributes=lambda *args, **kwargs: ({'sleeve_length':
                {'status': 'predicted', 'label': 'sleeveless'}}, 1),
        )
        result = FashionPipeline(backend).run(image, ['cuff'], ['sleeve_length'])
        self.assertEqual(len(result['instances']), 2)
        self.assertEqual(result['instances'][0]['regions']['cuff']['status'], 'missed')
        self.assertEqual(result['instances'][0]['attributes']['sleeve_length']['label'], 'sleeveless')
        self.assertEqual(len(result['instances'][0]['review_flags']), 2)

    def check_ground(self, detections, region='collar'):
        # 明确隔离模型依赖；下面只测试后处理，不把 mock 当成模型验证。
        semantic = types.ModuleType('fashion_multimodal_analysis.integration.region_semantics')
        semantic.REGION_PROMPTS = {'collar': 'collar', 'pattern': 'pattern'}
        semantic.phrase_matches = lambda region, label: region == label
        semantic.supports_region = lambda category, region: True
        ground = types.ModuleType('fashion_multimodal_analysis.grounding.refinement.run_shoulder_waist_pattern_groundingdino_refinement_v1')
        ground.run_grounding_dino = lambda *args: (detections, 0)
        sam = types.ModuleType('fashion_multimodal_analysis.grounding.refinement.evaluate_sam_refinement_9case')
        sam.run_sam = lambda crop, *args: (np.ones((crop.height, crop.width)), .9, 0)
        modules = {module.__name__: module for module in (semantic, ground, sam)}
        with patch.dict(sys.modules, modules):
            # 已安装项目可能已导入 backend；仅替换本测试需要的依赖。
            from fashion_multimodal_analysis.integration.model_backends import ModelBackends
            with patch('fashion_multimodal_analysis.integration.model_backends.phrase_matches', semantic.phrase_matches), patch('fashion_multimodal_analysis.integration.model_backends.supports_region', semantic.supports_region):
                obj = ModelBackends.__new__(ModelBackends)
                obj.config = {'strict_region_grounding': True, 'grounding_box_threshold': .3,
                              'grounding_text_threshold': .25}
                obj.measure = lambda action: (action(), 0)
                obj.ground_processor = obj.ground_model = obj.sam_processor = obj.sam_model = obj.device = None
                roi = GarmentROI('a', 'top', .8, [10, 20, 90, 100], np.ones((120, 120)))
                return obj.ground(Image.new('RGB', (120, 120)), roi, region)[0]

    def test_oversized_local_box_is_rejected(self):
        result = self.check_ground([{'box': [0, 0, 80, 80], 'score': .9, 'label': 'collar'}])
        self.assertEqual(result['status'], 'rejected_region_geometry')
        self.assertIsNone(result['bbox'])

    def test_rejected_candidate_does_not_hide_valid_candidate(self):
        result = self.check_ground([
            {'box': [0, 0, 80, 80], 'score': .9, 'label': 'collar'},
            {'box': [20, 5, 40, 15], 'score': .8, 'label': 'collar'},
        ])
        self.assertEqual(result['bbox'], [30, 25, 50, 35])
        self.assertEqual(int(result['mask'].sum()), 200)
        self.assertEqual(len(result['geometry_rejections']), 1)

    def test_candidates_sorted_by_confidence(self):
        result = self.check_ground([
            {'box': [1, 1, 5, 5], 'score': .5, 'label': 'collar'},
            {'box': [20, 5, 40, 15], 'score': .8, 'label': 'collar'},
        ])
        self.assertEqual(result['score'], .8)
        self.assertEqual(result['bbox'], [30, 25, 50, 35])

    def test_pattern_is_not_subject_to_local_area_limit(self):
        result = self.check_ground([{'box': [0, 0, 80, 80], 'score': .9, 'label': 'pattern'}], 'pattern')
        self.assertEqual(result['status'], 'detected')
        self.assertFalse(result['local_mask_constrained_to_proposal'])


if __name__ == '__main__':
    unittest.main()
