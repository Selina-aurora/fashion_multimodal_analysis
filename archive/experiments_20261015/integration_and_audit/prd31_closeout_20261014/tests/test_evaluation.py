"""Synthetic fixtures only: not actual GPU outputs or human annotation results."""
import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import BASE, SCHEMA, ATTRS, REGIONS, load, sha, pixels
from evaluate import evaluate

class EvaluationSafety(unittest.TestCase):
    def setUp(self):
        self.cohort=load(BASE/'reference/cohort.json');self.tax=load(BASE/'reference/taxonomy.json')
        self.review={'schema_version':SCHEMA,'manifest_sha256':sha(BASE/'reference/cohort.json'),
            'reviewer':'SYNTHETIC_TEST_ONLY','cases':{}}
        self.pred={'schema_version':'prd31_same12_predictions_v1','manifest_sha256':self.review['manifest_sha256'],'cases':{}}
        for c in self.cohort:
            cid=c['case_id'];self.review['cases'][cid]={'source_sha256':c['source_sha256'],'confirmed':True,
                'reviewed_at':'2026-10-14T00:00:00Z','category':'top','target_status':'correct_target','target_box':[.1,.1,.9,.9],
                'labels':{a:self.tax[a][0] for a in ATTRS},
                'regions':{k:{'presence':'yes','grade':'','boxes':[[.2,.2,.4,.4]],'boxes_complete':True} for k in REGIONS}}
            self.pred['cases'][cid]={'source_sha256':c['source_sha256'],'image_size':[100,200],
                'pipeline_status':'completed','instances':[{'category':'top','bbox':[10,20,90,180],
                'regions':{k:{'status':'detected','bbox':[20,40,40,80]} for k in REGIONS},
                'attributes':{a:{'status':'predicted','label':self.tax[a][0]} for a in ATTRS}}]}

    def run_eval(self):return evaluate(self.review,self.pred,self.cohort,self.tax)
    def metric(self,s,k='primary_color'):return next(x for x in s['attribute_metrics'] if x['attribute']==k)
    def test_known_geometry_and_denominators(self):
        records,s=self.run_eval();self.assertEqual(len(records),144)
        self.assertEqual(self.metric(s)['accuracy'],1)
        self.assertEqual(pixels([.1,.2,.5,.6],[200,100]),[20,20,100,60])
    def test_unknown_is_not_negative(self):
        self.review['cases']['case_01']['labels']['primary_color']='uncertain'
        _,s=self.run_eval();self.assertEqual(self.metric(s)['eligible_run_tasks'],11)
    def test_no_model_output_counts_as_failure_not_exclusion(self):
        self.pred['cases']['case_01']['instances'][0]['attributes'].pop('primary_color')
        _,s=self.run_eval();self.assertEqual(self.metric(s)['eligible_run_tasks'],12)
        self.assertEqual(self.metric(s)['correct'],11)
    def test_wrong_class_counts_as_upstream_failure(self):
        self.pred['cases']['case_01']['instances'][0]['category']='dress'
        _,s=self.run_eval();self.assertEqual(s['status_counts']['attribute']['blocked_by_upstream'],6)
        self.assertEqual(self.metric(s)['eligible_run_tasks'],12)
    def test_not_run_is_pending_not_model_failure(self):
        self.pred['cases'].pop('case_01');_,s=self.run_eval()
        self.assertEqual(self.metric(s)['eligible_run_tasks'],11)
        self.assertEqual(s['status_counts']['garment']['prediction_not_run'],1)
    def test_wrong_source_rejected(self):
        self.pred['cases']['case_01']['source_sha256']='0'*64
        with self.assertRaises(ValueError):self.run_eval()
    def test_original_or_unconfirmed_answers_are_not_new_gt(self):
        self.review['cases']['case_01']['confirmed']=False
        _,s=self.run_eval();self.assertEqual(s['human_pending_cases'],['case_01'])
        self.assertEqual(self.metric(s)['eligible_run_tasks'],11)
    def test_missing_boxes_cannot_pass_iou(self):
        self.review['cases']['case_01']['regions']['cuff']['boxes_complete']=False
        _,s=self.run_eval();self.assertEqual(s['status_counts']['region']['human_part_boxes_incomplete'],1)
    def test_out_of_bounds_annotation_rejected(self):
        self.review['cases']['case_01']['target_box']=[0,0,2,1]
        with self.assertRaises(ValueError):self.run_eval()
    def test_absence_false_positive_stays_separate(self):
        r=self.review['cases']['case_01']['regions']['pocket'];r.update(presence='no',boxes_complete=False,boxes=[])
        _,s=self.run_eval();self.assertEqual(s['status_counts']['region']['absent_false_positive'],1)

if __name__=='__main__':unittest.main()
