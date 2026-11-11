import unittest
from run_live_probe_v4 import select_dev_rows

ROWS = [{'case_id':'fresh_dev_001','target_instance_id':'pred_0000'},
        {'case_id':'fresh_dev_004','target_instance_id':'pred_0002'},
        {'case_id':'fresh_dev_008','target_instance_id':'pred_0002'}]

class SelectionTests(unittest.TestCase):
    def test_targeted(self):
        rows=select_dev_rows(ROWS,['fresh_dev_008/pred_0002','fresh_dev_004/pred_0002'],2)
        self.assertEqual([r['case_id'] for r in rows],['fresh_dev_004','fresh_dev_008'])
    def test_default(self):
        self.assertEqual(len(select_dev_rows(ROWS,None,2)),2)
    def test_unknown_target(self):
        with self.assertRaisesRegex(ValueError,'not found'):
            select_dev_rows(ROWS,['fresh_dev_999/pred_0000'],2)
    def test_duplicates(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            select_dev_rows(ROWS,['fresh_dev_004/pred_0002']*2,2)

if __name__=='__main__':unittest.main()
