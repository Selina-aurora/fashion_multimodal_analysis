import pathlib,sys,tempfile,unittest,csv,json
S=pathlib.Path(__file__).parent
sys.path.insert(0,str(S))
import audit_regions as audit
import gpu_preflight_readonly as preflight
class TestAudit(unittest.TestCase):
 def test_raw_28_cohorts(self):
  root=S.parent/'inputs';d=audit.read_json(root/'other_region_review_confirmed_20261009.json')
  self.assertEqual(len(d['cases']),28)
  from collections import Counter
  self.assertEqual(dict(Counter(x['target_region'] for x in d['cases'])),{'pocket':9,'waist':8,'hem':11})
 def test_pocket42(self):
  root=S.parent/'inputs';d=audit.read_json(root/'prd312_pocket24_HUMAN_REVIEW.json')
  self.assertEqual(len(d['records']),24)
  self.assertEqual(sum(len(x['parts']) for x in d['records']),42)
 def test_core_sha_format(self):self.assertEqual(len(preflight.CORE_SHA),64)
 def test_csv_headers(self):
  with tempfile.TemporaryDirectory() as t:
   p=pathlib.Path(t)/'a.csv';p.write_text('image_name,value\nA.jpg,1\n',encoding='utf-8')
   self.assertEqual(preflight.inspect_csv(p)['rows'],1)
 def test_region_csv_build(self):
  with tempfile.TemporaryDirectory() as t:
   s=S.parent/'inputs'
   audit.build(s/'other_region_review_confirmed_20261009.json',t)
   with open(pathlib.Path(t)/'hem_waist_19_candidate_register.csv',encoding='utf-8-sig',newline='') as f:
    lines=list(csv.DictReader(f))
   self.assertEqual(len(lines),19)
if __name__=='__main__':unittest.main(verbosity=2)
