import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from live_observer_v4 import LiveShadowObserver
from run_live_probe_v4 import run_triplets
from verified_guard_loader import load_guard

REFERENCE = Path(__file__).resolve().parents[1] / 'reference_guard' / 'prd32_output_guard.py'

class FakeHelper:
    def make_marked_image(self, qrec, folder, input_mode, padding):
        path = folder / (qrec['target_instance_id'] + '.jpg')
        path.write_bytes(b'marked-for-test')
        return path
    def run_model_case(self, model, proc, qrec, marked, prompt, task_limits):
        return ({'tags':'{"category":"top"}','vqa':'white','copy':'a white top'}[qrec['task']], {'end_to_end_ms':1.0})

class ObserverTests(unittest.TestCase):
    def setup_record(self, root):
        img = root / 'sample.jpg';img.write_bytes(b'actual-source-image')
        digest = hashlib.sha256(img.read_bytes()).hexdigest()
        rec = {'case_id':'fresh_dev_001','target_instance_id':'pred_0000','image_path':str(img),
               'image_sha256':digest,'target_bbox_xyxy':[1,2,30,40], 'predicted_category_v5_unverified':'top'}
        return rec
    def test_begin_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r=self.setup_record(root); m=root/'marked.jpg';m.write_bytes(b'marked')
            o=LiveShadowObserver('baseline');q={'image_path':r['image_path'], 'image_sha256':r['image_sha256'],
                                  'target_bbox_xyxy':r['target_bbox_xyxy'], 'target_instance_id':'pred_0000',
                                  'task':'tags','query':'color?'}
            p='test-prompt';h=hashlib.sha256(p.encode()).hexdigest()
            rid=o.before_model(rec=r,qrec=q,task='tags',marked=m,prompt=p,prompt_sha=h,job_id='host1')
            self.assertIn('baseline',rid)
            with self.assertRaisesRegex(ValueError,'did not match'):
                o.after_model(job_id='host1',response_job_id='wrong',completed_host_request_id=rid,raw_text='x')
            event=o.after_model(job_id='host1',response_job_id='host1',completed_host_request_id=rid,raw_text='x')
            self.assertFalse(event['mask_producer_identity_verified'])
            with self.assertRaisesRegex(ValueError,'no matching'):
                o.after_model(job_id='host1',response_job_id='host1',completed_host_request_id=rid,raw_text='x')
    def test_mismatched_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r=self.setup_record(root);m=root/'marked.jpg';m.write_bytes(b'x')
            q={'image_path':r['image_path'], 'image_sha256':r['image_sha256'],
               'target_bbox_xyxy':[2,2,30,40], 'target_instance_id':'pred_0000', 'task':'tags'}
            with self.assertRaisesRegex(ValueError,'bbox'):
                LiveShadowObserver('baseline').before_model(rec=r,qrec=q,task='tags',marked=m,
                   prompt='p',prompt_sha=hashlib.sha256(b'p').hexdigest(),job_id='j')
    def test_bad_img_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r=self.setup_record(root);m=root/'marked.jpg';m.write_bytes(b'x')
            Path(r['image_path']).write_bytes(b'changed-after-record')
            q={'image_path':r['image_path'], 'image_sha256':r['image_sha256'],
               'target_bbox_xyxy':r['target_bbox_xyxy'], 'target_instance_id':'pred_0000','task':'tags'}
            with self.assertRaisesRegex(ValueError,'SHA-256'):
                LiveShadowObserver('baseline').before_model(rec=r,qrec=q,task='tags',marked=m,
                   prompt='p',prompt_sha=hashlib.sha256(b'p').hexdigest(),job_id='j')
    def test_full_fake_model_and_existing_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r=self.setup_record(root)
            prompts={t:f'original-{t}-baseline' for t in ('tags','vqa','copy')}
            runner=SimpleNamespace(QUESTION='color?', PROMPT_SHA={'baseline':{t:hashlib.sha256(p.encode()).hexdigest() for t,p in prompts.items()}},
                                   tsha=lambda x:hashlib.sha256(x.encode()).hexdigest())
            core=SimpleNamespace(build_prompt=lambda helper,focus,qrec,task,v:prompts[task])
            # Use the reviewed guard implementation as a test-only source.
            guard = load_guard(REFERENCE)
            out=root/'probe';out.mkdir()
            report=run_triplets(rows=[r],runner=runner,core=core,helper=FakeHelper(),focus=None,
                               variant='baseline',model=None,processor=None,output=out,
                               guard=guard,max_instances=1)
            self.assertEqual(report['tasks_correlated'],3)
            self.assertFalse(report['roi_producer_identity_verified'])
            self.assertTrue((out/'guard_shadow_report.json').exists())
            self.assertEqual(len((out/'host_call_events.jsonl').read_text().splitlines()),3)
            self.assertEqual(len((out/'qwen_shadow_results.jsonl').read_text().splitlines()),3)

if __name__ == '__main__':unittest.main()
