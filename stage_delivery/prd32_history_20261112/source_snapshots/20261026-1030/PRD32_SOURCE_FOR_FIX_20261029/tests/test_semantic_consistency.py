import unittest
from types import SimpleNamespace
import numpy as np
from PIL import Image
from fashion_multimodal_analysis.integration.semantic_consistency import apply_consistency_review,pattern_phrase_label


class ConsistencyTests(unittest.TestCase):
    def test_same_cuff_and_hem_keep_geometry_but_require_review(self):
        regions={k:{'status':'detected','bbox':[0,0,10,10],'mask_path':'x.png'} for k in ('cuff','hem')}
        flags=apply_consistency_review(regions,{})
        self.assertEqual(len(flags),1)
        self.assertEqual(regions['cuff']['status'],'requires_review')
        self.assertEqual(regions['hem']['bbox'],[0,0,10,10])
        self.assertEqual(regions['hem']['model_status_before_review'],'detected')

    def test_separate_regions_not_changed(self):
        regions={'cuff':{'status':'detected','bbox':[0,0,2,2]},'hem':{'status':'detected','bbox':[5,5,9,9]}}
        self.assertEqual(apply_consistency_review(regions,{}),[])

    def test_generic_pattern_phrase_does_not_claim_floral(self):
        self.assertIsNone(pattern_phrase_label('printed pattern'))
        self.assertIsNone(pattern_phrase_label('floral striped pattern'))

    def test_pattern_conflict_preserves_both_candidates(self):
        regions={'pattern':{'status':'detected','model_phrase':'floral pattern','bbox':[0,0,10,10]}}
        attrs={'pattern':{'status':'predicted','label':'solid','confidence':.8}}
        apply_consistency_review(regions,attrs)
        self.assertEqual(attrs['pattern']['label'],'')
        self.assertEqual(attrs['pattern']['model_candidate_before_review']['label'],'solid')
        self.assertEqual(regions['pattern']['model_phrase'],'floral pattern')

    def test_agreeing_pattern_is_retained(self):
        regions={'pattern':{'status':'detected','model_phrase':'printed floral pattern'}}
        attrs={'pattern':{'status':'predicted','label':'floral'}}
        self.assertEqual(apply_consistency_review(regions,attrs),[])

    def test_other_pattern_is_not_forced(self):
        regions={'pattern':{'status':'detected','model_phrase':'floral pattern'}}
        attrs={'pattern':{'status':'predicted','label':'other_pattern'}}
        self.assertEqual(apply_consistency_review(regions,attrs),[])

    def test_conflicted_collar_propagates_to_neckline(self):
        regions={k:{'status':'detected','bbox':[0,0,10,10]} for k in ('collar','pocket')}
        attrs={'neckline':{'status':'predicted','label':'crew','confidence':.8}}
        apply_consistency_review(regions,attrs)
        self.assertEqual(attrs['neckline']['status'],'requires_review')
        self.assertEqual(attrs['neckline']['model_candidate_before_review']['label'],'crew')

    def test_missed_is_not_absence_or_review(self):
        r={'cuff':{'status':'missed','bbox':None},'hem':{'status':'detected','bbox':[0,0,10,10]}}
        self.assertEqual(apply_consistency_review(r,{}),[])
        self.assertEqual(r['cuff']['status'],'missed')

    def test_pipeline_explicit_option_checks_conflict_without_class_override(self):
        from fashion_multimodal_analysis.integration.pipeline import FashionPipeline,GarmentROI
        roi=GarmentROI('a','top',.8,[0,0,10,10],np.ones((10,10)))
        backend=SimpleNamespace(config={'semantic_consistency_review':True},
            segment=lambda image:([roi],0),
            ground=lambda *args:({'status':'detected','bbox':[0,0,10,10]},0),
            attributes=lambda *args,**kwargs:({},0))
        r=FashionPipeline(backend).run(Image.new('RGB',(10,10)),['cuff','hem'],[])['instances'][0]
        self.assertEqual(r['roi'].category,'top')
        self.assertEqual(r['regions']['cuff']['status'],'requires_review')
        self.assertEqual(r['review_flags'][0]['reason'],'different_regions_share_near_identical_box')


if __name__=='__main__':unittest.main()
