import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from prepare_pocket_review import digest, source_path, preflight, triage, prepare

class PocketReviewStage2Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
  self.base=Path(self.tmp.name)
  self.project=self.base/'fashion_multimodal_analysis';self.project.mkdir()
  self.imgdir=self.base/'fashion_data/raw/fashionpedia/images/test';self.imgdir.mkdir(parents=True)
 def make_case(self,i,file_id=None):
  if file_id is None:file_id=i
  path=self.imgdir/f'img_{file_id}.jpg'
  if not path.exists(): Image.new('RGB',(250,300),(100+i,125,140)).save(path)
  parts=[]
  n=2 if i<16 else (3 if i==16 else 1) # total 42
  for j in range(n):
   parts.append({'part_annotation_id':i*3+j,'part_bbox_xywh':[40.0+j*20,60.0,30.0,12.0],
    'part_attributes':[{'name':'patch (pocket)'}]})
  return {'record_id':f'v5_train_{i:05d}','split':'train' if i<19 else 'val',
      'source_image':f'../fashion_data/raw/fashionpedia/images/test/{path.name}',
      'source_image_sha256':digest(path),'garment_bbox':[20,20,210,240],
      'native_garment_annotation_id':i+100,'category':'top',
      'image_id':file_id,'parts':parts}
 def test_safe_path(self):
  with self.assertRaises(ValueError):source_path(self.project,'../secret.txt')
 def test_preflight_positive(self):
  r=self.make_case(0);a,b=preflight([r],self.project)
  self.assertEqual(len(a),1);self.assertEqual(b,[])
 def test_sha_mismatch_fail(self):
  r=self.make_case(0);r['source_image_sha256']='0'*64
  a,b=preflight([r],self.project)
  self.assertEqual(len(a),0);self.assertEqual(len(b),1)
 def test_missing_image_fail(self):
  r=self.make_case(0);(self.imgdir/'img_0.jpg').unlink()
  _,b=preflight([r],self.project);self.assertEqual(len(b),1)
 def test_triage_pending(self):
  q=triage([self.make_case(0)]);self.assertEqual(q[0]['review_status'],'pending')
 def test_full_dry_run(self):
  cases=[self.make_case(i, 0 if i in (0,1) else 1 if i in (19,20) else i) for i in range(24)]
  p=self.base/'cases.json';p.write_text(json.dumps(cases),'utf8')
  o=self.base/'out';self.assertEqual(prepare(p,self.project,o),0)
  h=(o/'prd312_pocket24_review_OFFLINE.html').read_text('utf8')
  self.assertIn('v5_train_00000',h);self.assertIn('导出人工审核JSON',h)
  summary=json.loads((o/'run_summary.json').read_text())
  self.assertEqual(summary['parts'],42);self.assertEqual(summary['reviewed_parts'],0)
  self.assertEqual(summary['train_cases'],19);self.assertEqual(summary['val_cases'],5)
  self.assertTrue((self.base/'out_UPLOAD_THIS.zip').is_file())
 def test_abort_on_missing_not_partial(self):
  cases=[self.make_case(i) for i in range(24)]
  cases[4]['source_image_sha256']='X'*64
  p=self.base/'cases.json';p.write_text(json.dumps(cases),'utf8')
  o=self.base/'bad';self.assertEqual(prepare(p,self.project,o),2)
  self.assertFalse((o/'prd312_pocket24_review_OFFLINE.html').exists())

if __name__=='__main__':unittest.main()
