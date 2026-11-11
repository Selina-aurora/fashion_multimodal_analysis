import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from guard_v4_contract import validate_candidate_binding
from run_guard_v4 import build

SOURCE={'case_id':'fresh_dev_001','instance_id':'pred_0000','image_sha256':'a'*64}
BASE={
 'case_id':'fresh_dev_001', 'instance_id':'pred_0000','image_sha256':'a'*64,
 'query_instance_id':'pred_0000','roi_instance_id':'pred_0000',
 'mask_instance_id':'pred_0000','response_instance_id':'pred_0000',
 'candidate_category':'top', 'category_published':False,
 'attributes':[{'key':'color','value':'blue','source_instance_id':'pred_0000'}]
}
class TestContract(unittest.TestCase):
 def check_bad(self,updates, expected):
  d=dict(BASE);d.update(updates)
  res=validate_candidate_binding(SOURCE,d)
  self.assertEqual(res['decision'],'BLOCK_BINDING_VIOLATION')
  self.assertIn(expected,res['violations'])
  self.assertFalse(res['mainline_approved'])
 def test_valid_only_shadow(self):
  r=validate_candidate_binding(SOURCE,dict(BASE))
  self.assertEqual(r['decision'],'SHADOW_ONLY_NEEDS_INDEPENDENT_VISUAL_GT')
  self.assertFalse(r['category_gt_verified'])
  self.assertFalse(r['mainline_approved'])
 def test_wrong_case(self): self.check_bad({'case_id':'fresh_dev_002'},'MISMATCH_CASE_ID')
 def test_wrong_instance(self): self.check_bad({'instance_id':'pred_0001'},'MISMATCH_INSTANCE_ID')
 def test_wrong_image_hash(self): self.check_bad({'image_sha256':'b'*64},'MISMATCH_IMAGE_SHA256')
 def test_wrong_query_binding(self): self.check_bad({'query_instance_id':'pred_0001'},'MISMATCH_QUERY_INSTANCE_ID')
 def test_wrong_roi_binding(self): self.check_bad({'roi_instance_id':'pred_0001'},'MISMATCH_ROI_INSTANCE_ID')
 def test_wrong_mask_binding(self): self.check_bad({'mask_instance_id':'pred_0001'},'MISMATCH_MASK_INSTANCE_ID')
 def test_wrong_response_binding(self): self.check_bad({'response_instance_id':'pred_0001'},'MISMATCH_RESPONSE_INSTANCE_ID')
 def test_alias_cross_instance(self): self.check_bad({'alias_target_instance_id':'pred_0001'},'ALIAS_NOT_AUTHORIZED')
 def test_alias_even_same_instance(self): self.check_bad({'alias_target_instance_id':'pred_0000'},'ALIAS_NOT_AUTHORIZED')
 def test_invalid_category(self): self.check_bad({'candidate_category':'gloves'},'CATEGORY_MISSING_OR_OUT_OF_TAXONOMY')
 def test_missing_category(self): self.check_bad({'candidate_category':''},'CATEGORY_MISSING_OR_OUT_OF_TAXONOMY')
 def test_published_blocked(self): self.check_bad({'category_published':True},'UNAUTHORIZED_CATEGORY_PUBLICATION')
 def test_attr_crossover(self): self.check_bad({'attributes':[{'source_instance_id':'pred_0001'}]},'ATTR_INSTANCE_CROSSOVER_0')
 def test_attrs_not_list(self): self.check_bad({'attributes':{}},'ATTRIBUTES_NOT_LIST')
 def test_missing_id(self):
  d=dict(BASE);d.pop('roi_instance_id')
  r=validate_candidate_binding(SOURCE,d)
  self.assertIn('MISSING_ROI_INSTANCE_ID',r['violations'])
 def test_source_missing_key(self):
  d=dict(SOURCE);d.pop('image_sha256')
  r=validate_candidate_binding(d,dict(BASE))
  self.assertIn('BAD_SOURCE_IDENTITY',r['violations'])
 def test_builder_replays_included_dev60(self):
  import tempfile,json
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(__file__).resolve().parents[1]
   v=build(root/'PRD32_1106_GUARD_V3_RESULTS_INPUT.zip',Path(tmp))
   self.assertEqual(v['source_instances'],171)
   self.assertEqual(v['category_conflicts'],93)
   self.assertEqual(v['unreviewed_conflicts'],63)
   self.assertEqual(v['mask_bbox_geometry_compatible'],171)
   self.assertEqual(v['chat_attested_ai_prefill_without_exported_gt'],30)
   self.assertEqual(v['status'],'OFFLINE_CONTRACT_AUDIT_PASS_NOT_MODEL_EVALUATION')
   self.assertFalse(v['mainline_deployment_approved'])
if __name__=='__main__': unittest.main()
