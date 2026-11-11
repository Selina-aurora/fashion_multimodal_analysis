import hashlib
import tempfile
import unittest
from pathlib import Path

from live_observer_v4 import LiveShadowObserver


class FailClosedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.img = self.root / 'original.jpg'
        self.img.write_bytes(b'fresh dev image original')
        self.marked = self.root / 'marked.jpg'
        self.marked.write_bytes(b'actual marked image')
        self.record = {'case_id': 'fresh_dev_002', 'target_instance_id': 'pred_0001',
                       'image_path': str(self.img), 'image_sha256': hashlib.sha256(self.img.read_bytes()).hexdigest(),
                       'target_bbox_xyxy':[0,1,10,20], 'predicted_category_v5_unverified':'pants'}
        self.qrec = {'image_path':str(self.img),'image_sha256':self.record['image_sha256'],
                     'target_instance_id':'pred_0001','target_bbox_xyxy':[0,1,10,20],
                     'task':'tags','query':'color?'}
        self.prompt = 'prompt-v1'
        self.prompt_hash = hashlib.sha256(self.prompt.encode()).hexdigest()
    def begin(self, observer, **overrides):
        kwargs=dict(rec=self.record,qrec=self.qrec,task='tags',marked=self.marked,
                    prompt=self.prompt,prompt_sha=self.prompt_hash,job_id='job1')
        kwargs.update(overrides)
        return observer.before_model(**kwargs)
    def test_source_instance_mismatch_blocked(self):
        bad={**self.qrec,'target_instance_id':'pred_9999'}
        with self.assertRaisesRegex(ValueError,'target/task'):
            self.begin(LiveShadowObserver('baseline'),qrec=bad)
    def test_source_path_mismatch_blocked(self):
        bad={**self.qrec,'image_path':str(self.marked)}
        with self.assertRaisesRegex(ValueError,'path mismatch'):
            self.begin(LiveShadowObserver('baseline'),qrec=bad)
    def test_marked_bytes_required(self):
        self.marked.unlink()
        with self.assertRaisesRegex(ValueError,'marked image'):
            self.begin(LiveShadowObserver('baseline'))
    def test_prompt_sha_blocked(self):
        with self.assertRaisesRegex(ValueError,'prompt hash'):
            self.begin(LiveShadowObserver('baseline'),prompt_sha='0'*64)
    def test_invalid_variant(self):
        with self.assertRaisesRegex(ValueError,'unsupported variant'):
            LiveShadowObserver('v2')
    def test_holdout_path_blocked(self):
        folder=self.root/'holdout';folder.mkdir()
        target=folder/'a.jpg';target.write_bytes(b'x')
        bad={**self.record, 'image_path':str(target), 'image_sha256':hashlib.sha256(b'x').hexdigest()}
        q={**self.qrec, 'image_path':str(target),'image_sha256':bad['image_sha256']}
        with self.assertRaisesRegex(ValueError,'restricted'):
            self.begin(LiveShadowObserver('baseline'),rec=bad,qrec=q)
    def test_missing_vqa_question_blocked(self):
        bad={**self.qrec,'task':'vqa','query':''}
        with self.assertRaisesRegex(ValueError,'question'):
            self.begin(LiveShadowObserver('baseline'),task='vqa',qrec=bad)
    def test_reused_job_id_blocked(self):
        o=LiveShadowObserver('baseline')
        rid=self.begin(o)
        o.after_model(job_id='job1',response_job_id='job1',completed_host_request_id=rid,raw_text='{"category":"pants"}')
        bad={**self.qrec,'task':'copy'}
        with self.assertRaisesRegex(ValueError,'duplicate host'):
            self.begin(o,task='copy',qrec=bad)
    def test_duplicate_target_task_blocked(self):
        o=LiveShadowObserver('baseline')
        rid=self.begin(o)
        o.after_model(job_id='job1',response_job_id='job1',completed_host_request_id=rid,raw_text='{"category":"pants"}')
        with self.assertRaisesRegex(ValueError,'duplicate host'):
            self.begin(o,job_id='job2')
    def test_partial_triplet_blocked(self):
        o=LiveShadowObserver('baseline')
        rid=self.begin(o)
        o.after_model(job_id='job1',response_job_id='job1',completed_host_request_id=rid,raw_text='{"category":"pants"}')
        with self.assertRaisesRegex(ValueError,'incomplete target'):
            o.finalize(lambda x:{'instance_count':len(x)})
    def test_empty_response_blocked(self):
        o=LiveShadowObserver('baseline');rid=self.begin(o)
        with self.assertRaisesRegex(ValueError,'missing model'):
            o.after_model(job_id='job1',response_job_id='job1',completed_host_request_id=rid,raw_text='')
    def test_mismatched_request_id_blocked(self):
        o=LiveShadowObserver('baseline');self.begin(o)
        with self.assertRaisesRegex(ValueError,'did not match'):
            o.after_model(job_id='job1',response_job_id='job1',completed_host_request_id='wrong',raw_text='x')

if __name__ == '__main__':unittest.main()
