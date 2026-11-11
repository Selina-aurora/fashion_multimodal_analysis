import copy
import importlib.util
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('shadow',ROOT/'shadow_collision_audit.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class Dev60ShadowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=ROOT/'input/v5_raw_predictions.jsonl'
        cls.pairs=ROOT/'input/43_pairs_mask_audit.csv'
        cls.review=ROOT/'input/16_pairs_chat_accepted.csv'
        cls.records,cls.ids,cls.pair_rows,cls.review_rows=mod.load_and_validate(cls.raw,cls.review,cls.pairs)
        cls.groups,cls.alias,cls.reg,cls.other=mod.analyze(cls.records,cls.ids,cls.pair_rows,cls.review_rows)

    def test_dev60_scope(self):
        self.assertEqual((len(self.records),sum(map(len,self.ids.values()))),(60,171))
        self.assertEqual((len(self.pair_rows),len(self.review_rows),len(self.other)),(43,16,27))

    def test_group_dedup_not_pair_dedup(self):
        triple=[g for g in self.groups if g['case_id']=='fresh_dev_003']
        self.assertEqual(len(triple),1)
        self.assertEqual(triple[0]['pair_records'],3)
        self.assertEqual(triple[0]['num_members'],3)
        self.assertEqual(triple[0]['alias_candidate_count'],2)
        self.assertEqual(self.alias[('fresh_dev_003','pred_0000')],'pred_0002')
        self.assertEqual(self.alias[('fresh_dev_003','pred_0001')],'pred_0002')

    def test_uncertain_and_nonlocal_blocked(self):
        by_case={g['case_id']:g for g in self.groups}
        self.assertEqual(by_case['fresh_dev_022']['shadow_group_status'],'BLOCKED_CATEGORY_AMBIGUOUS')
        self.assertEqual(by_case['fresh_dev_033']['shadow_group_status'],'BLOCKED_CATEGORY_AMBIGUOUS')
        self.assertEqual(by_case['fresh_dev_053']['shadow_group_status'],'BLOCKED_CANONICAL_OUTSIDE_HIGH_IOU_GROUP')
        self.assertFalse(any(k[0] in {'fresh_dev_022','fresh_dev_033','fresh_dev_053'} for k in self.alias))

    def test_alias_no_inplace_mutation_and_no_qwen_rebind(self):
        before=copy.deepcopy(self.records)
        mod.analyze(self.records,self.ids,self.pair_rows,self.review_rows)
        self.assertEqual(before,self.records)
        self.assertEqual(len(self.alias),12)
        self.assertEqual(len(self.reg),171)
        self.assertTrue(all(r['qwen_output_reassigned']=='NO' for r in self.reg))
        self.assertTrue(all(r['original_prediction_preserved']=='YES' for r in self.reg))

    def test_scope_gate_rejects_sealed_holdout(self):
        payload=self.raw.read_text(encoding='utf-8')
        modified=payload.replace('TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY','SEALED_HOLDOUT',1)
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'raw.jsonl';path.write_text(modified,encoding='utf-8')
            with self.assertRaises(AssertionError):mod.load_and_validate(path,self.review,self.pairs)

    def test_wrong_user_attestation_rejected(self):
        with __import__('tempfile').TemporaryDirectory() as td:
            path=Path(td)/'16.csv'
            txt=self.review.read_text(encoding='utf-8-sig').replace('USER_CHAT_ACCEPTED_2026-11-06','NO',1)
            path.write_text(txt,encoding='utf-8-sig')
            with self.assertRaises(AssertionError):mod.load_and_validate(self.raw,path,self.pairs)

if __name__=='__main__':unittest.main(verbosity=2)
