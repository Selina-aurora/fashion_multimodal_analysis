import csv,json,tempfile,unittest
from pathlib import Path
from verify_gpu_shadow import audit

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        p=Path(self.temp.name);self.p=p
        row=dict(case_id='fresh_dev_001',instance_id='pred_0000',image_sha256='a'*64,
                 qwen_task_records='6',expected_task_records='6',risk_flags='ALIAS_QUARANTINE',
                 replay_status='SHADOW_ONLY_RUNTIME_BINDING_UNVERIFIED',
                 frozen_guard_v5_policy='BLOCK_SHADOW_ALIAS',qwen_baseline_tags='top',qwen_focus_tags='top',
                 publish_allowed='NO',auto_alias_merge='NO',auto_category_override='NO')
        self.rows=[row]
        self.summary={'frozen_dev60_instances':1,'qwen_records_seen':6,'metadata_violation_counts':{},
                      'marked_image_file_bytes_verified':True,'holdout_opened':False,'model_called':False,
                      'source_predictions_modified':False}
        self.save()

    def save(self):
        (self.p/'shadow_summary.json').write_text(json.dumps(self.summary))
        for name in ['shadow_decisions_171.csv','shadow_risk_queue.csv']:
            with (self.p/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=self.rows[0].keys());w.writeheader();w.writerows(self.rows)

    def test_valid_data(self):self.assertTrue(audit(self.p)['audit_pass'])
    def test_publication_never_ok(self):
        self.rows[0]['publish_allowed']='YES';self.save()
        self.assertFalse(audit(self.p)['audit_pass'])
    def test_missing_qwen_records(self):
        self.rows[0]['qwen_task_records']='5';self.save()
        self.assertFalse(audit(self.p)['audit_pass'])
    def test_holdout_touched(self):
        self.summary['holdout_opened']=True;self.save()
        self.assertFalse(audit(self.p)['audit_pass'])

if __name__=='__main__': unittest.main()
