"""Structural tests with SYNTHETIC per-chunk ZIPs; NOT real model results."""
import csv,io,json,sys,unittest,zipfile
from tempfile import TemporaryDirectory
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from joint_eval_stage2_dev60 import manifest_verified,MANIFEST_SHA256
from aggregate_stage2_8_chunks import audit_zips

class SyntheticMergeContractTests(unittest.TestCase):
    def _synthetic(self,folder,omit=None):
        manifest,_=manifest_verified()
        paths=[]
        for chunk in range(1,9):
            if chunk==omit:continue
            targets=[r for r in manifest['targets'] if r['chunk_id']==chunk]
            csv_io=io.StringIO()
            fields=['case_id','instance_id','total_host_wall_ms','v5_host_wall_ms',
                'tags_task_host_wall_ms','vqa_task_host_wall_ms','copy_task_host_wall_ms','guard_host_wall_ms',
                'guard_status','publish_allowed','holdout_accessed','frozen_mask_iou']
            w=csv.DictWriter(csv_io,fieldnames=fields);w.writeheader()
            for r in targets:
                w.writerow({'case_id':r['case_id'],'instance_id':r['instance_id'],
                    'total_host_wall_ms':2000,'v5_host_wall_ms':50,'tags_task_host_wall_ms':1000,
                    'vqa_task_host_wall_ms':200,'copy_task_host_wall_ms':700,'guard_host_wall_ms':0.1,
                    'guard_status':'REVIEW_REQUIRED_NOT_RELEASED','publish_allowed':False,
                    'holdout_accessed':False,'frozen_mask_iou':1.0})
            calls=[{'invocation_uid':f'synthetic_{chunk}_{i}'} for i in range(3*(len(targets)+1))]
            summary={'status':'PRD31_PRD32_STAGE2_CHUNK_COMPLETE_SHADOW_ONLY',
                'manifest_sha256':MANIFEST_SHA256,'chunk':chunk,'target_count':len(targets),
                'measured_qwen_calls':3*len(targets)}
            p=Path(folder)/f'FAKE_CHUNK_{chunk}.zip'
            with zipfile.ZipFile(p,'w') as z:
                z.writestr(f'FAKE_CHUNK_{chunk}/summary.json',json.dumps(summary))
                z.writestr(f'FAKE_CHUNK_{chunk}/per_request_timing.csv',csv_io.getvalue())
                z.writestr(f'FAKE_CHUNK_{chunk}/qwen_calls.jsonl','\n'.join(json.dumps(v) for v in calls))
            paths.append(p)
        return paths
    def test_full_contract_simulated_only(self):
        with TemporaryDirectory() as d:
            _,seen,chunks,calls=audit_zips(self._synthetic(d))
            self.assertEqual(len(seen),171)
            self.assertEqual(len(chunks),8)
            self.assertEqual(calls,537)
    def test_missing_chunk_rejected(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(ValueError):audit_zips(self._synthetic(d,omit=8))
    def test_duplicate_chunk_rejected(self):
        with TemporaryDirectory() as d:
            paths=self._synthetic(d)
            with self.assertRaises(ValueError):audit_zips(paths[:-1]+paths[:1])
if __name__=='__main__':unittest.main()
