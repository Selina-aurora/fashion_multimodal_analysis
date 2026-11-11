"""Synthetic only: validate controller ZIP selection, do not run GPU or modify project."""
import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_eval_stage2_dev60 import manifest_verified,MANIFEST_SHA256
from run_all_stage2_dev60 import validate_chunk_zip,scan_existing


def synthetic_chunk(path, manifest, ch, wrong_task=False):
    expected=[r for r in manifest['targets'] if r['chunk_id']==ch]
    buf=io.StringIO()
    cols=['case_id','instance_id','guard_status','publish_allowed','holdout_accessed','frozen_mask_iou']
    wr=csv.DictWriter(buf,fieldnames=cols)
    wr.writeheader()
    for r in expected:
        wr.writerow({'case_id':r['case_id'],'instance_id':r['instance_id'],
            'guard_status':'REVIEW_REQUIRED_NOT_RELEASED','publish_allowed':False,
            'holdout_accessed':False,'frozen_mask_iou':1.0})
    calls=[]
    for i,task in enumerate(['tags','vqa','copy']):
        calls.append({'cohort':'warmup_excluded','invocation_uid':'warmup_'+task,
                      'task':task,'publish_allowed':False})
    for ix,r in enumerate(expected,1):
        for task in ['tags','vqa','copy']:
            t = 'tags' if wrong_task and ix==1 and task=='copy' else task
            calls.append({'cohort':f'measured_t{ix:03d}','invocation_uid':f'{ix}_{task}',
                          'case_id':r['case_id'],'instance_id':r['instance_id'],
                          'task':t,'publish_allowed':False})
    summary={'status':'PRD31_PRD32_STAGE2_CHUNK_COMPLETE_SHADOW_ONLY','chunk':ch,
        'manifest_sha256':MANIFEST_SHA256,'target_count':len(expected),
        'all_guard_blocked':True,'publish_allowed':False,'holdout_accessed':False,
        'all_frozen_mask_comparisons_passed':True,'quality_gt':False,
        'measured_qwen_calls':3*len(expected)}
    with zipfile.ZipFile(path,'w') as z:
        for name, content in [
            ('summary.json',json.dumps(summary)),('per_request_timing.csv',buf.getvalue()),
            ('qwen_calls.jsonl','\n'.join(json.dumps(r) for r in calls)),
            ('preflight.json','{}'),('co_residency.json','{}')]:
            z.writestr('SYNTHETIC/'+name,content)

class ControllerContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest,_=manifest_verified()

    def test_valid_fixed_chunk(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'PRD31_PRD32_STAGE2_CHUNK02_SYNTHETIC.zip'
            synthetic_chunk(p,self.manifest,2)
            r=validate_chunk_zip(p,2,self.manifest)
            self.assertEqual(r['targets'],24)
            self.assertEqual(r['qwen_calls'],75)

    def test_wrong_task_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'PRD31_PRD32_STAGE2_CHUNK02_SYNTHETIC.zip'
            synthetic_chunk(p,self.manifest,2,wrong_task=True)
            with self.assertRaises(ValueError):validate_chunk_zip(p,2,self.manifest)

    def test_partial_or_missing_summary_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'PRD31_PRD32_STAGE2_CHUNK03_FAKE.zip'
            with zipfile.ZipFile(p,'w') as z:
                z.writestr('INCOMPLETE/failure.json','{}')
            found,invalid=scan_existing(Path(td),Path(td)/'none',self.manifest)
            self.assertEqual(found,{})
            self.assertEqual(len(invalid),1)

if __name__ == '__main__':unittest.main()
