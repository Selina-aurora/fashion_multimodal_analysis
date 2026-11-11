import importlib.util
import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

P=Path(__file__).resolve().parents[1]/'run_dev60_v5.py'
spec=importlib.util.spec_from_file_location('dev60_v5',P)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Dev60Tests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root=Path(self.t.name)
        self.data=self.root/'fashion_data'
        self.train=self.data/'raw'/'train'/'train'/'image'
        self.train.mkdir(parents=True)
        self.records=[]
        for i in range(1,61):
            p=self.train/f'{i:06}.jpg'
            p.write_bytes(b'testdata'+str(i).encode('ascii'))
            self.records.append({'case_id':f'fresh_dev_{i:03d}','image_path':str(p),
                  'image_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                  'human_reference_withheld':True,'do_not_use_historical_gt_bbox':True})
        self.manifest=self.root/'dev60.jsonl'
        self.write_manifest()
    def write_manifest(self):
        self.manifest.write_text(''.join(json.dumps(r)+'\n' for r in self.records))
    def checked(self,check_hash=True):
        with mock.patch.object(m,'EXPECTED_MANIFEST_SHA',m.sha(self.manifest)):
            return m.check_60_manifest(self.manifest,self.data,check_hash)
    def test_manifest_60_good(self):
        r=self.checked();self.assertEqual(len(r),60)
        self.assertEqual(set(r[0]),m.LEGACY_FIELDS)
    def test_reject_wrong_row_count(self):
        self.records.pop();self.write_manifest()
        with self.assertRaisesRegex(ValueError,'EXPECTED_60'):self.checked()
    def test_reject_unexpected_human_gt(self):
        self.records[0]['target_category']='top';self.write_manifest()
        with self.assertRaisesRegex(ValueError,'NOT_GT_FREE'):self.checked()
    def test_reject_unapproved_review_flag(self):
        self.records[0]['human_reference_withheld']=False;self.write_manifest()
        with self.assertRaisesRegex(ValueError,'REFERENCE_NOT_WITHHELD'):self.checked()
    def test_reject_missing_img(self):
        (self.train/'000005.jpg').unlink()
        with self.assertRaisesRegex(ValueError,'NOT_FOUND'):self.checked()
    def test_reject_sha_changed(self):
        (self.train/'000005.jpg').write_bytes(b'bad replacement')
        with self.assertRaisesRegex(ValueError,'IMAGE_BYTES_CHANGED'):self.checked()
    def test_reject_duplicate_content(self):
        self.records[2]['image_sha256']=self.records[1]['image_sha256'];self.write_manifest()
        with self.assertRaisesRegex(ValueError,'DUPLICATE_CONTENT_OR_PATH'):self.checked(False)
    def test_reject_holdout(self):
        p=self.data/'raw'/'holdout'/'train'/'image.jpg'
        with self.assertRaisesRegex(ValueError,'RESTRICTED_SPLIT'):m.approve_train_path(str(p),self.data,False)
    def test_reject_val(self):
        p=self.data/'raw'/'val'/'train'/'image.jpg'
        with self.assertRaisesRegex(ValueError,'RESTRICTED_SPLIT'):m.approve_train_path(str(p),self.data,False)
    def test_reject_outside_train(self):
        with self.assertRaisesRegex(ValueError,'NOT_APPROVED_TRAIN'):m.approve_train_path(str(self.root/'other.jpg'),self.data,False)
    def test_reject_bad_id_order(self):
        self.records[0]['case_id']='fresh_dev_002';self.write_manifest()
        with self.assertRaisesRegex(ValueError,'CASE_ORDER'):self.checked()
    def test_frozen_real_source_exact(self):
        real=Path('/mnt/data/prd32_v5_contract_inspect/v5_real_inference.py')
        if real.exists():
            legacy=m.load_unchanged_runner(real)
            self.assertTrue(hasattr(legacy,'run_gpu'))
            self.assertEqual(legacy.CKPT_SHA,m.EXPECTED_CHECKPOINT_SHA)
    def test_reject_wrong_original_runner(self):
        p=self.root/'fake.py';p.write_text('pass\n')
        with self.assertRaisesRegex(ValueError,'RUNNER_CHANGED'):m.load_unchanged_runner(p)
    def test_preflight_fields(self):
        x=m.preflight_record(None,self.records,None,{'schema':'original'})
        self.assertFalse(x['official_v2_freeze'])
        self.assertFalse(x['main_ab_inference_ran'])
        self.assertEqual(x['input_count'],60)
    def test_preflight_no_original_overwrite(self):
        args=types.SimpleNamespace(gate=str(self.root),source=str(self.manifest),data_root=str(self.data))
        with self.assertRaisesRegex(ValueError,'GATE_DIR_ALREADY_EXISTS'):m.build_preflight(args)
    def test_verify_synthetic_output(self):
        gate=self.root/'gate';gate.mkdir()
        run=self.root/'run';run.mkdir()
        staged=gate/'v5_image_only_inputs.jsonl';staged.write_text('original-only-prepared-data\n')
        legacy=types.SimpleNamespace(CLASS_TO_ID={'top':1})
        expected=[{'case_id':r['case_id'],'image_path':r['image_path'],'image_sha256':r['image_sha256']} for r in self.records]
        args=types.SimpleNamespace(gate=str(gate),run=str(run))
        rows=[]
        for r in expected:
            rows.append({'schema':'prd32_v5_live_raw_image_result_v1','case_id':r['case_id'],'image_sha256':r['image_sha256'],
                         'source_image_path':r['image_path'],'image_size':[400,600],'segmentation_ms':1.2,
                         'instances':[],'instance_count':0,'human_target_instance_selected':False,
                         'review_image':r['case_id']+'_v5_review.jpg','scope':m.SOURCE_SCOPE})
            (run/(r['case_id']+'_v5_review.jpg')).write_bytes(b'fake-jpg')
        with (run/'v5_raw_predictions.jsonl').open('w') as f:
            for r in rows:f.write(json.dumps(r)+'\n')
        m.dump(run/'run_manifest.json',{'schema':'prd32_v5_real_predictions_run_v1',
               'input_manifest_sha256':m.sha(staged),'checkpoint_sha256':m.EXPECTED_CHECKPOINT_SHA,
               'score_threshold':0.4,'mask_threshold':0.5,'human_gt_used_for_inference':False})
        m.dump(run/'v5_real_summary.json',{'status':'V5_RAW_INFERENCE_COMPLETED','n_images':60,
               'checkpoint_sha256':m.EXPECTED_CHECKPOINT_SHA,'v5_predictions_sha256':m.sha(run/'v5_raw_predictions.jsonl'),
               'n_predicted_instances':0,'n_images_with_no_instances':60,'categories_from_v5_predictions':{}})
        with mock.patch.object(m,'validate_gpu_contract',return_value=(legacy,expected,None,{})):
            result=m.verify_output(args)
        self.assertEqual(result['n_images'],60)
        self.assertEqual(result['n_empty_images'],60)
        self.assertFalse(result['official_prd_acceptance'])

if __name__=='__main__':unittest.main()
