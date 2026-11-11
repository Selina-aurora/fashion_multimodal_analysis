import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
import inference_mode_ab_v3 as m

class TestInferenceModeAB(unittest.TestCase):
    def test_locked_original_vendor_hashes(self):
        self.assertEqual(len(m.check_lock()),7)
    def test_one_per_frozen_V5_category(self):
        manifest,allkeys=m.base.load_manifest(m.base.DEFAULT_MANIFEST)
        keys=m.choose_eight(manifest)
        self.assertEqual(len(keys),8)
        self.assertEqual(len(set(keys)),8)
        cat={(r['case_id'],r['instance_id']):r['v5_frozen_category'] for r in manifest['targets']}
        self.assertEqual({cat[k] for k in keys},m.base.CATEGORIES)
    def test_order_counterbalanced_and_warmup(self):
        manifest,_=m.base.load_manifest(m.base.DEFAULT_MANIFEST)
        keys=m.choose_eight(manifest)
        seq=m.sequence(keys)
        self.assertEqual(len(seq),34)
        self.assertEqual(sum(p is None for _,_,_,p in seq),2)
        arms=[v for _,_,v,p in seq if p]
        self.assertEqual(arms.count('baseline'),16)
        self.assertEqual(arms.count('candidate'),16)
        for i in range(1,9):
            l1=[arm for label,key,arm,pair in seq if pair==(1,i)]
            l2=[arm for label,key,arm,pair in seq if pair==(2,i)]
            self.assertEqual(l2,list(reversed(l1)))
    def make_data(self,change=False):
        manifest,_=m.base.load_manifest(m.base.DEFAULT_MANIFEST)
        keys=m.choose_eight(manifest)
        cat={(r['case_id'],r['instance_id']):r['v5_frozen_category'] for r in manifest['targets']}
        seq=m.sequence(keys)
        rows=[];calls=[]
        for label,key,arm,pair in seq:
            if pair is None:continue
            ix=keys.index(key)
            row={'cohort':label,'case_id':key[0],'instance_id':key[1],
                'producer_category':cat[key], 'source_image_sha256':'a'*64,
                'fresh_mask_sha256':'b'*64,'guard_status':'REVIEW_REQUIRED_NOT_RELEASED',
                'total_host_wall_ms':2100+ix*10 if arm=='baseline' else 1800+ix*10}
            rows.append(row)
            for task in m.base.TASKS:
                text='same model answer'
                if change and arm=='candidate' and key==keys[0] and task=='tags':
                    text='changed model answer'
                calls.append({'cohort':label,'task':task,'case_id':key[0],
                    'instance_id':key[1],'image_sha256':'a'*64,'marked_sha256':'b'*64,
                    'prompt_sha256':'c'*64,'response':text,
                    'model_runner_timing':{'generated_tokens':40,'generation_ms':250,
                        'preprocess_ms':31},'qwen_task_host_wall_ms':300})
        return rows,calls,seq,keys
    def test_pair_score_good(self):
        pairs,tasks,report=m.compare(*self.make_data())
        self.assertEqual(len(pairs),16)
        self.assertEqual(len(tasks),48)
        self.assertEqual(report['exact_output_matches'],48)
        self.assertEqual(report['unique_targets_faster'],8)
        self.assertEqual(report['candidate_decision'],'CONSIDER_FURTHER_DEV60_TIMING_ONLY')
        self.assertGreater(report['mean_speedup_pct_positive_means_candidate_faster'],5)
    def test_response_drift_rejects(self):
        p,t,r=m.compare(*self.make_data(change=True))
        self.assertEqual(r['exact_output_matches'],46) # one target, two repeats
        self.assertEqual(r['candidate_decision'],'KEEP_BASELINE_NOT_PROVEN_FASTER_OR_OUTPUT_DRIFT')
    def test_truncated_pair_fails_closed(self):
        data=self.make_data()
        with self.assertRaisesRegex(ValueError,'expected exactly 32'):
            m.compare(data[0][:-1],data[1],data[2],data[3])
    def test_restricted_paths(self):
        with self.assertRaises(ValueError):m.base.refuse_restricted('/workspace/holdout_SEALED_1')
    def test_only_allowed_single_mode_change(self):
        import inspect
        s=inspect.getsource(m.InferenceModeHelper)
        self.assertIn('with self._torch.inference_mode()',s)
        self.assertIn('self._inner.run_model_case',s)

if __name__=='__main__':unittest.main()
