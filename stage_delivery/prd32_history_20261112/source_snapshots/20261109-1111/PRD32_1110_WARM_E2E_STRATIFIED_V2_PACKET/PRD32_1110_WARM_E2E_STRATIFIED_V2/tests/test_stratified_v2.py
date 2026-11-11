import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
import warm_e2e_stratified_v2 as m

class SafeExpandedTimingTests(unittest.TestCase):
    def test_manifest_24_targets_8_classes(self):
        info,keys=m.load_manifest(m.DEFAULT_MANIFEST)
        self.assertEqual(len(keys),24)
        self.assertEqual(len(set(keys)),24)
        self.assertEqual(len(set(k[0] for k in keys)),21)
        self.assertEqual({x['v5_frozen_category'] for x in info['targets']},m.CATEGORIES)
        self.assertNotIn('confirmed_category',str(info))

    def test_manifest_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'changed_manifest.json'
            data=json.loads(m.DEFAULT_MANIFEST.read_text())
            data['targets'][0]['case_id']='fresh_dev_999'
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'manifest changed'):
                m.load_manifest(p)

    def test_percentile_small_sample_interpolation(self):
        self.assertEqual(m.percentile([100,300],95),290)
        self.assertEqual(m.percentile([2,4,10],50),4)
        self.assertIsNone(m.percentile([],95))

    def test_duplicate_target_rejected_even_if_rehashed_externally(self):
        info,keys=m.load_manifest(m.DEFAULT_MANIFEST)
        self.assertEqual(len(keys),len(set(keys)))
        self.assertEqual(len([x for x in info['targets'] if x['stratum']=='shoe']),3)

    def test_restricted_paths_rejected(self):
        from producer_contract_v7 import refuse_restricted
        with self.assertRaises(ValueError):refuse_restricted('/workspace/holdout_SEALED_DO_NOT_OPEN')
        refuse_restricted('/workspace/PRD32_DEV60_V5_REAL_20261030_run01')

    def test_report_two_repeats_and_144_unique_calls(self):
        _,keys=m.load_manifest(m.DEFAULT_MANIFEST)
        d={(z['case_id'],z['instance_id']):z for z in json.loads(m.DEFAULT_MANIFEST.read_text())['targets']}
        rows=[];calls=[]
        for rep in range(2):
            for j,key in enumerate(keys):
                label=f'measured_r{rep+1:02d}_t{j+1:02d}'
                row={'cohort':label,'case_id':key[0],'instance_id':key[1],
                     'producer_category':d[key]['v5_frozen_category'],
                     'total_host_wall_ms':2000+j+rep*10,
                     'v5_host_wall_ms':100,'tags_task_host_wall_ms':800,
                     'vqa_task_host_wall_ms':300,'copy_task_host_wall_ms':790,
                     'guard_host_wall_ms':.15}
                rows.append(row)
                for task in m.TASKS:
                    calls.append({'cohort':label,'task':task,'model_runner_timing':{
                         'generated_tokens':40,'generation_ms':200.,'preprocess_ms':20.}})
        result=m.calculate_report(rows,calls,keys,2)
        self.assertEqual(result['request_level']['total_host_wall_ms']['n'],48)
        self.assertEqual(result['unique_target_level']['total_host_wall_ms']['n'],24)
        self.assertEqual(len(result['strata_target_means']),8)
        self.assertEqual(result['strata_target_means']['bag']['total_host_wall_ms']['n'],3)
        self.assertEqual(result['qwen_token_and_generation_cost']['tags']['generated_tokens']['mean_tokens'],40)

    def test_incomplete_repeats_rejected(self):
        _,keys=m.load_manifest(m.DEFAULT_MANIFEST)
        with self.assertRaisesRegex(ValueError,'incomplete target/repeat'):
            m.calculate_report([],[],keys,2)

    def test_pinned_vendor_files_equal_to_prior_success(self):
        import hashlib,zipfile
        prior=Path('/mnt/data/PRD32_1110_WARM_E2E_PILOT_V1_PACKET.zip')
        if not prior.exists():self.skipTest('prior success source only available in assistant runtime')
        with zipfile.ZipFile(prior) as z:
            root='PRD32_1110_WARM_E2E_PILOT_V1/'
            for name in ['producer_contract_v7.py','run_v5_runtime_producer_v7.py','run_live_probe_v4.py','live_observer_v4.py','verified_guard_loader.py']:
                self.assertEqual((BASE/name).read_bytes(),z.read(root+name))

if __name__=='__main__':unittest.main()
