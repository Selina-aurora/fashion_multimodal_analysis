import sys,unittest,hashlib,tempfile
from pathlib import Path
from PIL import Image
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mask_contour_preflight_v1 as mod

class BasicTests(unittest.TestCase):
 def test_keys_all_45(self):
  self.assertEqual(len(mod.load_keys(mod.ROOT/'KEYS_ONLY_INPUT_45.csv')),45)
 def test_all_references_sha(self):
  for r in mod.load_keys(mod.ROOT/'KEYS_ONLY_INPUT_45.csv'):
   self.assertEqual(mod.sha(mod.ROOT/'reference_marked'/r['reference_marked_name']),r['reference_marked_sha256'])
 def test_forbidden(self):
  with self.assertRaises(ValueError):mod.ensure_approved('/workspace/holdout/foo.jpg')
 def test_iou(self):
  self.assertAlmostEqual(mod.bbox_iou([0,0,4,4],[1,1,5,5]),9/23)
 def test_mask_metrics(self):
  m=np.zeros((40,60),dtype=bool);m[10:20,5:25]=1
  t=mod.assess_mask(m,[5,10,25,20]);self.assertEqual(t['mask_pixel_count'],200)
  self.assertAlmostEqual(t['bbox_overlap_iou'],1.)
  self.assertEqual(t['geometry_flags'],[])
 def test_empty_mask_rejected(self):
  with self.assertRaises(ValueError):mod.assess_mask(np.zeros((20,20)),[0,0,10,10])
 def test_dimension_mismatch_rejected(self):
  with self.assertRaises(ValueError):mod.draw_contour(Image.new('RGB',(20,21)),np.zeros((20,20),dtype=bool))
 def test_only_contour_changed(self):
  im=Image.new('RGB',(30,30),(90,90,90));mask=np.zeros((30,30),bool);mask[5:20,6:21]=1
  out=np.asarray(mod.draw_contour(im,mask));self.assertTrue(np.all(out[0,0]==[90,90,90]));self.assertTrue(np.any(out[:,:,:]!=[90,90,90]))
 def test_html_has_review_and_export(self):
  h=mod.page_html([{'review_key':'fresh_dev_004/pred_0002','filename':'fresh_dev_004__pred_0002.jpg','status':'READY_FOR_MANUAL_VISUAL_CHECK','bbox_quality_triage':'CLEAR','geometry_flags':[]}]);self.assertIn('exportCSV()',h);self.assertIn('UNREVIEWED',h)
 def test_output_does_not_claim_model(self):
  self.assertIn('NO model inference',mod.__doc__)

if __name__=='__main__':unittest.main()
