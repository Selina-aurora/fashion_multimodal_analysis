import copy,json,pathlib,tempfile,unittest
from prd32_nextbatch_diagnostic import analyze
BASE=pathlib.Path(__file__).resolve().parent

def get(name):return json.loads((BASE/name).read_text(encoding='utf8'))
class TestDiagnostic(unittest.TestCase):
 def setUp(self):
  self.review=get('PRD32_NextBatch_HumanReview.json')
  self.summary=get('service_gate_summary_original.json')
  self.extracted=get('prd32_html_extracted.json')
 def run_audit(self,review=None,summary=None,extracted=None):
  return analyze(review if review is not None else self.review,
                 summary if summary is not None else self.summary,
                 extracted if extracted is not None else self.extracted,'TEST_HASH')
 def test_complete_review_passes_but_stays_blocked(self):
  report,rows=self.run_audit()
  self.assertEqual(len(rows),8)
  self.assertFalse(report['publish_allowed'])
  self.assertFalse(report['official_PRD_acceptance'])
  self.assertEqual(report['actual_deduplicated_instances'],0)
  self.assertEqual(report['hypothetical_distinct_candidate_groups_if_duplicates_resolved'],5)
 def test_unconfirmed_is_not_accepted(self):
  d=copy.deepcopy(self.review);d['rows'][0]['instance_review_confirmed']=False
  with self.assertRaises(ValueError):self.run_audit(review=d)
 def test_not_human_is_not_accepted(self):
  d=copy.deepcopy(self.review);d['reviewed_by_human']=False
  with self.assertRaises(ValueError):self.run_audit(review=d)
 def test_missing_instance_is_not_accepted(self):
  d=copy.deepcopy(self.review);d['rows'].pop()
  with self.assertRaises(ValueError):self.run_audit(review=d)
 def test_service_counts_must_reconcile(self):
  d=copy.deepcopy(self.summary);d['runtime_counts']['copy_blocked']=6
  with self.assertRaises(ValueError):self.run_audit(summary=d)
 def test_original_vqa_questions_not_given(self):
  d=copy.deepcopy(self.review);d['rows'][0]['vqa_quality']='CORRECT'
  with self.assertRaises(ValueError):self.run_audit(review=d)
 def test_debug_phrases_detected(self):
  _,rows=self.run_audit()
  self.assertEqual(sum(a['copy_debug_reference_detected'] for a in rows),6)
 def test_extracted_output_missing_is_not_accepted(self):
  d=copy.deepcopy(self.extracted);d.pop()
  with self.assertRaises(ValueError):self.run_audit(extracted=d)
if __name__=='__main__':unittest.main()
