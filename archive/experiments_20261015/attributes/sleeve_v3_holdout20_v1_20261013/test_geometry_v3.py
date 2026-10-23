import unittest
import numpy as np
from geometry_v3_proposals import propose_endpoints, bilinear

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.arm={'failure_reason':None,'auto_cuff':[50,40],
                  'ratio_pred_or_null':1/6,
                  'points':{'shoulder':[50,20], 'elbow':[50,80], 'wrist':[50,140]}}

    def test_plain_opening(self):
        m=np.zeros((160,120),bool);m[:41,35:66]=True
        r=propose_endpoints(m,self.arm,55)
        self.assertEqual('proposal_requires_review',r['status'],r)
        self.assertAlmostEqual(50,r['midpoint_px'][0],delta=4)
        self.assertAlmostEqual(41,r['midpoint_px'][1],delta=3)

    def test_diagonal_opening(self):
        yy,xx=np.indices((160,120))
        m=(xx>=32)&(xx<=72)&(yy<=40+.30*(xx-50))
        r=propose_endpoints(m,self.arm,50)
        self.assertEqual('proposal_requires_review',r['status'],r)
        self.assertGreater(r['metrics']['relative_tilt'],0.1)

    def test_unbounded_torso_does_not_invent_B(self):
        m=np.zeros((160,140),bool)
        m[:41,35:]=True # left cuff edge, but continuing into torso on right
        r=propose_endpoints(m,self.arm,95)
        self.assertEqual('partial_requires_review',r['status'],r)
        self.assertIsNotNone(r['A_outer_px'])
        self.assertIsNone(r['B_inner_px'])
        self.assertIsNone(r['midpoint_px'])

    def test_all_mask_rejects(self):
        r=propose_endpoints(np.ones((160,120),bool),self.arm,55)
        self.assertEqual('abstain',r['status'])
        self.assertIsNone(r['midpoint_px'])

    def test_no_mask_rejects(self):
        r=propose_endpoints(np.zeros((160,120),bool),self.arm,55)
        self.assertEqual('abstain',r['status'])

    def test_missing_seed_rejects(self):
        arm={**self.arm,'auto_cuff':None}
        self.assertEqual('no_valid_v2_arm_seed',propose_endpoints(np.ones((160,120),bool),arm,55)['reason'])

    def test_v2_failure_not_overridden(self):
        arm={**self.arm,'failure_reason':'folded_arm'}
        self.assertEqual('abstain',propose_endpoints(np.ones((160,120),bool),arm,55)['status'])

    def test_bad_torso_rejects(self):
        self.assertEqual('no_torso_center',propose_endpoints(np.ones((160,120),bool),self.arm,float('nan'))['reason'])

    def test_bilinear_invalid_coordinates(self):
        vals=bilinear(np.ones((10,10),float),np.array([[4,5],[-1,8],[20,2]]))
        self.assertEqual(vals[0],1)
        self.assertTrue(np.isnan(vals[1:]).all())

    def test_gt_blind_interface(self):
        import inspect
        pars=list(inspect.signature(propose_endpoints).parameters)
        self.assertEqual(pars,['mask','arm','torso_x'])

if __name__=='__main__':unittest.main()
