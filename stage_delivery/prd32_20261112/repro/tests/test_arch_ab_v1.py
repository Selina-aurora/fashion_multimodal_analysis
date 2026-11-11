import csv,json,unittest
from pathlib import Path
from arch_ab_common import parse_tags,make_copy,stats,DEBUG_TERMS
from run_architecture_ab_v1 import select_keys,reconcile,ARMS,TASKS_BY_ARM

ROOT=Path(__file__).resolve().parents[1]

class ArchitectureTests(unittest.TestCase):
    def test_selected16(self):
        rows,keys,manifest=select_keys(16)
        self.assertEqual(len(keys),16)
        self.assertEqual(len(set(keys)),16)
        self.assertEqual(len({r['v5_frozen_category'] for r in rows}),8)
    def test_selected8(self):
        rows,keys,_=select_keys(8)
        self.assertEqual(len(keys),8)
        self.assertEqual(len({r['v5_frozen_category'] for r in rows}),8)
    def test_routes(self):
        self.assertEqual(len(ARMS),4)
        self.assertEqual(sum(map(len,TASKS_BY_ARM.values())),4)
    def test_json_markdown(self):
        self.assertEqual(parse_tags('```json\n{"category":"top","primary_color":"蓝色"}\n```')['category'],'top')
    def test_template_no_hallucinated_fabric(self):
        r=make_copy('```json\n{"category":"top","primary_color":"蓝色","pattern":"纯色","visible_design":"无设计"}\n```')
        self.assertEqual(r['status'],'TEMPLATE_DRAFT_NEEDS_HUMAN_REVIEW')
        self.assertTrue(20<=len(r['text'])<=80)
        for tok in ['纯棉','羊毛','天然','防水','100%','红框']:
            self.assertNotIn(tok,r['text'])
    def test_template_fails_closed(self):
        self.assertEqual(make_copy('not a json')['status'],'BLOCKED_TAGS_UNPARSABLE')
        self.assertEqual(make_copy('{"category":"top"}')['status'],'BLOCKED_INSUFFICIENT_TAGS')
    def test_offline_171(self):
        with (ROOT/'OFFLINE_171_ARCHITECTURE_COUNTERFACTUAL_NOT_LIVE.csv').open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
        self.assertEqual(len(rr),171)
        self.assertEqual(len({(r['case_id'],r['instance_id']) for r in rr}),171)
        self.assertTrue(all(float(r['baseline_measured_ms'])>float(r['vqa_only_offline_estimate_ms']) for r in rr))
        self.assertTrue(all(r['template_status']=='TEMPLATE_DRAFT_NEEDS_HUMAN_REVIEW' for r in rr))
    def test_comparison_synthetic(self):
        base=[{'cohort':'ab_r01_t01_baseline_three','case_id':'fresh_dev_001','instance_id':'pred_0001',
                'guard_status':'REVIEW_REQUIRED_NOT_RELEASED','total_host_wall_ms':2500}]
        calls=[{'cohort':base[0]['cohort'],'case_id':'fresh_dev_001','instance_id':'pred_0001','task':t,'response':t} for t in ('tags','vqa','copy')]
        alts=[]
        for arm in ('vqa_only','merchant_tags_copy','merchant_tags_template'):
            a={'arm':arm,'cohort':'ab_r01_t01_'+arm,'case_id':'fresh_dev_001','instance_id':'pred_0001',
              'total_host_wall_ms':1000,'qwen_calls':len(TASKS_BY_ARM[arm]),
              'outputs':{t:t for t in TASKS_BY_ARM[arm]},'template_status':'TEMPLATE_DRAFT_NEEDS_HUMAN_REVIEW'}
            if arm=='merchant_tags_template':a['outputs']['copy_template']='template text'
            alts.append(a)
        result=reconcile(base,alts,calls)
        self.assertEqual(len(result),1)
        self.assertTrue(result[0]['merchant_tags_copy_tags_equals_baseline'])
        self.assertEqual(result[0]['template_copy'],'template text')

if __name__=='__main__':unittest.main()
