import unittest,copy,json
from pathlib import Path
import numpy as np
from geometry import associate,arm_geometry,locate_cuff,infer_instances

def person(pid=0):
    xy=[[50.,50.] for _ in range(17)]
    for i,p in {5:[60.,20.],7:[60.,60.],9:[60.,100.],6:[20.,20.],8:[20.,60.],10:[20.,100.]}.items(): xy[i]=p
    return {'person_id':pid,'bbox':[0,0,100,115],'xy':xy,'conf':[.95]*17,'score':.99}

class GeometryChecks(unittest.TestCase):
    def test_side_mapping_uses_image_geometry(self):
        p=person();self.assertEqual(arm_geometry(p,'left',(120,120))['image_side'],'image_right')
        self.assertEqual(arm_geometry(p,'right',(120,120))['image_side'],'image_left')
    def test_ambiguous_people_rejected(self):
        pid,reason,_=associate([10,10,90,70],[person(),person(1)])
        self.assertIsNone(pid);self.assertEqual(reason,'ambiguous_person')
    def test_matching_empty_and_far_person(self):
        self.assertIsNone(associate([10,10,90,70],[])[0])
        self.assertIsNone(associate([200,200,300,300],[person()])[0])
    def test_synthetic_cuff_cut(self):
        m=np.zeros((120,120),bool);m[10:60,10:90]=1
        r=locate_cuff(m,arm_geometry(person(),'left',(120,120)))
        self.assertIsNone(r['failure_reason']);self.assertLess(abs(r['ratio_pred_or_null']-.5),.05)
        self.assertLess(np.linalg.norm(np.asarray(r['auto_cuff'])-[60,60]),4)
    def test_no_boundary_no_fake_long(self):
        r=locate_cuff(np.ones((120,120),bool),arm_geometry(person(),'left',(120,120)))
        self.assertIsNone(r['ratio_pred_or_null']);self.assertEqual(r['failure_reason'],'no_stable_cuff_boundary')
    def test_low_confidence_and_outside(self):
        p=person();p['conf'][7]=.1
        self.assertEqual(arm_geometry(p,'left',(120,120))['failure_reason'],'low_joint_confidence')
        p=person();p['xy'][9]=[121,100]
        self.assertEqual(arm_geometry(p,'left',(120,120))['failure_reason'],'joint_outside_image')
    def test_gt_values_cannot_choose_arm(self):
        inst={'instance_id':1,'category':'top','bbox':[10,10,90,70],'score':.9,'mask_file':'test'}
        mask=np.zeros((120,120),bool);mask[10:60,10:90]=1
        one=infer_instances([inst],[person()],(120,120),lambda _:mask)
        changed=copy.deepcopy(inst);changed.update(points={'shoulder':[999,999]},side='image_left',ratio=0,current_class='long')
        two=infer_instances([changed],[person()],(120,120),lambda _:mask)
        self.assertEqual(one[0]['arms'],two[0]['arms']);self.assertEqual(len(one[0]['arms']),2)
        self.assertEqual(one[0]['predicted_person_id'],two[0]['predicted_person_id'])
    def test_inference_manifest_contains_no_ground_truth(self):
        records=json.loads((Path(__file__).parent/'images.json').read_text())
        self.assertEqual(len(records),8)
        for r in records:self.assertEqual(set(r),{'image_id','path','sha256'})
    def test_reentry_is_not_sleeve_cuff(self):
        m=np.zeros((120,120),bool);m[10:50,10:90]=1;m[70:115,10:90]=1
        r=locate_cuff(m,arm_geometry(person(),'left',(120,120)))
        self.assertIsNone(r['ratio_pred_or_null'])
if __name__=='__main__': unittest.main()
