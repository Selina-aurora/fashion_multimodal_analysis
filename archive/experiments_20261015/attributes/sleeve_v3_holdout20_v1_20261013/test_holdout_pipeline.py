import json,hashlib,subprocess,tempfile,unittest
from pathlib import Path
from PIL import Image
import prepare_holdout as ph
import run_frozen_inference as inf

def H(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class HoldoutTests(unittest.TestCase):
 def test_deterministic_exclusion_and_freeze(self):
  with tempfile.TemporaryDirectory() as td:
   td=Path(td);root=td/'repo';src=td/'photos';src.mkdir();
   (root/'configs').mkdir(parents=True);(root/'benchmark/prd_3_1_v1/manifests').mkdir(parents=True)
   imgs=[]
   for j in range(6):
    p=src/f'image_{j}.jpg';Image.new('RGB',(420,500),(j*27,20,90)).save(p);imgs.append(p)
   train=root/ph.TRAIN;train.write_text('record_id,image_path\n0,'+str(imgs[0])+'\n',encoding='utf-8')
   val=root/ph.VAL;val.write_text('record_id,image_path\n0,'+str(imgs[1])+'\n',encoding='utf-8')
   core=root/ph.CORE;core.write_text('stub',encoding='utf-8')
   old_expected=ph.EXPECTED
   ph.EXPECTED={ph.TRAIN:H(train),ph.VAL:H(val),ph.CORE:H(core)}
   try:
    # Patch CLI within the test only: no real project is required.
    import sys
    previous=sys.argv
    out=td/'out'
    sys.argv=['prepare_holdout.py','--root',str(root),'--candidate-dir',str(src),'--n','3','--out-dir',str(out)]
    try:ph.main()
    finally:sys.argv=previous
    m=json.loads((out/'holdout_images_FROZEN.json').read_text())
    self.assertEqual(len(m),3)
    self.assertTrue(all(Path(x['path']).name not in {imgs[0].name,imgs[1].name} for x in m))
    self.assertEqual(len({x['sha256'] for x in m}),3)
    inf.check_cohort(out)
    # Refuse modifications after freeze.
    m[0]['source_name']='tampered.jpg'
    (out/'holdout_images_FROZEN.json').write_text(json.dumps(m))
    with self.assertRaises(ValueError):inf.check_cohort(out)
   finally:ph.EXPECTED=old_expected
 def test_unrecognized_training_paths_fail_closed(self):
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/'train.csv'
   path.write_text('record_id,foo\na,test.jpg\n')
   with self.assertRaises(ValueError):ph.exclusion_from_csv(path)
 def test_mocked_inference_preserves_missing_endpoints(self):
  with tempfile.TemporaryDirectory() as td:
   td=Path(td)
   mask=Image.new('L',(120,150),255);mask.save(td/'mask.png')
   im=[{'image_id':'x','size':[120,150],'instances':[{'instance_id':1,'category':'top','bbox':[0,0,119,149],
        'mask_file':'mask.png','predicted_person_id':None,'arms':[{'anatomical_side':'left',
        'image_side':'image_left','auto_cuff':[50,70],'ratio_pred_or_null':0.3}]}]}]
   pose=[{'image_id':'x','people':[]}]
   out=inf.build_v3(im,pose,td)
   p=out[0]['instances'][0]['arms'][0]['v3_proposal']
   self.assertEqual(p['status'],'abstain');self.assertIsNone(p['B_inner_px']);self.assertIsNone(p['midpoint_px'])

if __name__=='__main__':unittest.main()
