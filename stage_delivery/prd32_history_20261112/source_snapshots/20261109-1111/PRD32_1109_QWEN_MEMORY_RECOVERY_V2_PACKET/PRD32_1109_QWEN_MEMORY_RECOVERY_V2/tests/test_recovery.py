import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SRC=Path(__file__).resolve().parents[1]/'resume_qwen_v2.py'
spec=importlib.util.spec_from_file_location('resume',SRC)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_no_holdout(self):
        with self.assertRaises(ValueError):m.check_dev_path(Path('/workspace/HOLDOUT_SEALED/something'))
    def test_normal_path(self):self.assertEqual(m.check_dev_path(Path('/workspace/PRD32_DEV60')),Path('/workspace/PRD32_DEV60'))
    def test_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_bytes(b'abc')
            self.assertEqual(m.sha256_path(p),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
    def test_reject_release_and_complete_case(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            rows=[]
            for case in range(4):
                for task in ('tags','vqa','copy'):
                    rows.append({'case_id':str(case),'target_instance_id':'pred_0000','task':task,'status':'GENERATED_REVIEW_REQUIRED','publish_allowed':False,'timing':{'end_to_end_ms':10}})
            (p/'qwen_shadow_results.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            (p/'guard_shadow_report.json').write_text(json.dumps({'guard_service_status':'REVIEW_REQUIRED_NOT_RELEASED','production_approved':False}))
            r=m.summarize_qwen_output(p)
            self.assertEqual(r['qwen_task_count'],12)
            self.assertIsNone(r['full_chain_warm_e2e_ms'])
            rows[0]['publish_allowed']=True
            (p/'qwen_shadow_results.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaises(ValueError):m.summarize_qwen_output(p)
    def test_historical_failed_zip_structure(self):
        import zipfile
        original=Path('/mnt/data/PRD32_1109_TIMED_V7_20261109_093701_294993.zip')
        if original.is_file():
            with tempfile.TemporaryDirectory() as d:
                with zipfile.ZipFile(original) as z:z.extractall(d)
                folder=next(Path(d).iterdir())
                result=m.audit_producer(folder)
                self.assertEqual(result['mask_sha256_verified'],12)
                self.assertIsNone(result['true_full_chain_latency_ms'])
    def test_bad_source_dir(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):m.audit_producer(Path(d))

if __name__=='__main__':unittest.main()
