import importlib.util
from pathlib import Path
import unittest
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from audit_focus_binding import audit

class FocusBindingAuditTest(unittest.TestCase):
    def test_trace_integrity_and_category_disagreement(self):
        s=audit()
        self.assertEqual(s['total_task_outputs'],513)
        self.assertEqual(s['unique_instances'],171)
        self.assertEqual(s['metadata_issue_count_instances'],0)
        self.assertEqual(s['tags_v5_category_agreement_count']+s['tags_v5_category_disagreement_count'],171)
        self.assertFalse(s['independent_holdout_evaluated'])

if __name__=='__main__':unittest.main()
