from __future__ import annotations
import hashlib,json,sys,unittest,csv,zipfile,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from joint_eval_stage2_dev60 import manifest_verified
from run_vqa_only_dev60_v2 import read_snap,choose,approx_p95
from vqa_route_contract import validate_shadow_vqa,STATUS
from run_all_vqa_only_dev60_v2 import scan,audit_zip

ROOT=Path(__file__).resolve().parents[1]
class SafetyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest,_=manifest_verified()
        cls.baseline=read_snap(cls.manifest)
    def test_171_targets_and_8_chunks(self):
        self.assertEqual(len(self.baseline['targets']),171)
        self.assertEqual(sum(len(choose(self.manifest,i)) for i in range(1,9)),171)
        keys=[k for i in range(1,9) for k in choose(self.manifest,i)]
        self.assertEqual(len(set(keys)),171)
        self.assertEqual(len({x[0] for x in keys}),60)
    def test_source_snapshot_pin(self):
        source=(ROOT/'VQA_FROZEN_BASELINE_171.json').read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),(ROOT/'SOURCE_SNAPSHOT_SHA256.txt').read_text().strip())
    def test_reference_contains_only_dev_vqa_data(self):
        self.assertEqual(self.baseline['schema'],'PRD32_VQA_ONLY_DEV60_V2_FROZEN_BASELINE_REFERENCE')
        self.assertFalse(self.baseline['independent_gt'])
        for x in self.baseline['targets'].values():
            self.assertEqual(len(x['vqa_response_sha256']),64)
            self.assertEqual(x['baseline_guard_status'],'REVIEW_REQUIRED_NOT_RELEASED')
    def test_review_confirmed_all_16(self):
        with (ROOT/'AB16_RULE_COPY_REVIEW_USER_CONFIRMED.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),16)
        self.assertTrue(all('用户已确认' in r['confirmation_state'] for r in rows))
        self.assertTrue(all(r['candidate_can_replace_original'].startswith('不可以') for r in rows))
    def test_shadow_guard_accepts_complete_but_blocks_release(self):
        x={'case_id':'fresh_dev_001','instance_id':'pred_0000','vqa_answer':'蓝色','v5_mask_iou':1.0,
            'source_image_sha256':'a'*64,'publish_allowed':False}
        r=validate_shadow_vqa(x)
        self.assertEqual(r['status'],STATUS)
        self.assertFalse(r['publish_allowed'])
        self.assertFalse(r['original_three_task_guard_invoked'])
        self.assertFalse(r['quality_validated'])
    def test_shadow_guard_refuses_release(self):
        x={'case_id':'fresh_dev_001','instance_id':'pred_0000','vqa_answer':'蓝色','v5_mask_iou':1.0,
            'source_image_sha256':'a'*64,'publish_allowed':True}
        with self.assertRaises(ValueError):validate_shadow_vqa(x)
    def test_shadow_guard_refuses_empty_answer(self):
        x={'case_id':'fresh_dev_001','instance_id':'pred_0000','vqa_answer':'','v5_mask_iou':1.0,
            'source_image_sha256':'a'*64,'publish_allowed':False}
        with self.assertRaises(ValueError):validate_shadow_vqa(x)
    def test_shadow_guard_refuses_mask_drift(self):
        x={'case_id':'fresh_dev_001','instance_id':'pred_0000','vqa_answer':'蓝色','v5_mask_iou':0.5,
            'source_image_sha256':'a'*64,'publish_allowed':False}
        with self.assertRaises(ValueError):validate_shadow_vqa(x)
    def test_fake_incomplete_zip_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'PRD32_VQA_ONLY_V2_CHUNK01_bad.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('VQA_ONLY_CHUNK_SUMMARY.json',json.dumps({'status':'NOT_DONE'}))
            with self.assertRaises(ValueError):audit_zip(p,1,self.manifest,self.baseline)
    def test_scan_does_not_accept_empty_workspace(self):
        with tempfile.TemporaryDirectory() as d:
            selected,bad=scan(Path(d),self.manifest,self.baseline)
            self.assertEqual(selected,{})
    def test_p95_math(self):
        self.assertEqual(approx_p95([1,2,3,4,5]),4.8)

if __name__=='__main__':unittest.main()
