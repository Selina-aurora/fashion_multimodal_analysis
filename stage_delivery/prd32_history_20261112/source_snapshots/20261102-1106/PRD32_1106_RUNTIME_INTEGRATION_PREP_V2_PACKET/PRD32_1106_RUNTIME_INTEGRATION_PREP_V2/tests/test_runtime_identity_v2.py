import unittest
from hashlib import sha256
from runtime_identity_v2 import SegmentedTarget, ObservedRegion, HostRequestLedger

H='a'*64

class RuntimeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.t=SegmentedTarget('fresh_dev_001','pred_0000',H,(1.,2.,30.,40.))
        self.region=ObservedRegion('pred_0000','pred_0000',sha256(b'mask').hexdigest())
        self.ledger=HostRequestLedger()

    def emit(self,**kw):
        a={'target':self.t,'region':self.region,'task':'tags','variant':'baseline',
           'job_id':'host1','attempt_id':'run1','marked_image_bytes':b'marked'}
        a.update(kw);return self.ledger.dispatch(**a)

    def test_valid_correlated_shadow_response(self):
        e=self.emit()
        r=self.ledger.complete(job_id=e.job_id,completed_request_id=e.request_id,
                               response_job_id=e.job_id,raw_text='top')
        self.assertFalse(r['production_approved'])
        self.assertEqual(r['mask_sha256'],sha256(b'mask').hexdigest())

    def test_cross_roi_blocked(self):
        with self.assertRaises(ValueError):self.emit(region=ObservedRegion('pred_0001','pred_0000',self.region.mask_sha256))

    def test_cross_mask_blocked(self):
        with self.assertRaises(ValueError):self.emit(region=ObservedRegion('pred_0000','pred_0001',self.region.mask_sha256))

    def test_wrong_request_blocked(self):
        e=self.emit()
        with self.assertRaises(ValueError):self.ledger.complete(job_id=e.job_id,completed_request_id='wrong',response_job_id=e.job_id,raw_text='top')

    def test_host_correlation_blocked(self):
        e=self.emit()
        with self.assertRaises(ValueError):self.ledger.complete(job_id=e.job_id,completed_request_id=e.request_id,response_job_id='other',raw_text='top')

    def test_response_replay_blocked(self):
        e=self.emit();a=dict(job_id=e.job_id,completed_request_id=e.request_id,response_job_id=e.job_id,raw_text='top')
        self.ledger.complete(**a)
        with self.assertRaises(ValueError):self.ledger.complete(**a)

    def test_variant_disjoint_request_ids(self):
        a=self.emit()
        b=self.emit(job_id='host2',variant='focus_v1')
        self.assertNotEqual(a.request_id,b.request_id)

    def test_attempt_disjoint_request_ids(self):
        a=self.emit();b=self.emit(job_id='host2',attempt_id='run2')
        self.assertNotEqual(a.request_id,b.request_id)

    def test_same_request_id_disallowed(self):
        self.emit()
        with self.assertRaises(ValueError):self.emit(job_id='host2')

    def test_same_job_disallowed(self):
        self.emit()
        with self.assertRaises(ValueError):self.emit(variant='focus_v1')

    def test_unknown_task_or_variant(self):
        with self.assertRaises(ValueError):self.emit(task='wrong')
        with self.assertRaises(ValueError):self.emit(variant='wrong')

    def test_missing_image_bytes_rejected(self):
        with self.assertRaises(ValueError):self.emit(marked_image_bytes=b'')

    def test_invalid_bbox_rejected(self):
        with self.assertRaises(ValueError):SegmentedTarget('fresh_dev_001','pred_0000',H,(2,2,1,4))

    def test_invalid_sha_rejected(self):
        with self.assertRaises(ValueError):SegmentedTarget('fresh_dev_001','pred_0000','a',(1,2,3,4))

if __name__=='__main__': unittest.main()
