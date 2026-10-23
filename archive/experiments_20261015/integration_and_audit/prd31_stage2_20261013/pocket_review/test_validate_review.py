import hashlib,json,unittest
from pathlib import Path
from validate_review import validate

class ReviewValidationTests(unittest.TestCase):
 def setUp(self):
  self.p=Path(__file__).parent/'inputs/review_cases.json'
  self.cases=json.loads(self.p.read_text('utf8'))
  self.sha=hashlib.sha256(self.p.read_bytes()).hexdigest()
  self.review={'schema_version':'prd312_pocket_human_review_v1','source_cases_sha256':self.sha,
               'records':[{'record_id':c['record_id'],'split':c['split'],'category':c['category'],
                'source_image_sha256':c['source_image_sha256'],'garment_annotation_id':c['native_garment_annotation_id'],
                'parts':[{'part_annotation_id':p['part_annotation_id'],
                'review':{'association':'pending','bbox':'pending','type':'pending','note':''}} for p in c['parts']]} for c in self.cases]}
 def test_valid_pending(self):
  r=validate(self.review,self.cases,self.sha)
  self.assertTrue(r['structural_valid']);self.assertEqual(len(r['pending_parts']),42)
 def test_missing_record(self):
  self.review['records'].pop()
  self.assertFalse(validate(self.review,self.cases,self.sha)['structural_valid'])
 def test_source_changed(self):
  self.review['records'][0]['source_image_sha256']='0'*64
  self.assertFalse(validate(self.review,self.cases,self.sha)['structural_valid'])
 def test_wrong_part_id(self):
  self.review['records'][0]['parts'][0]['part_annotation_id']=-1
  self.assertFalse(validate(self.review,self.cases,self.sha)['structural_valid'])
 def test_completed_record(self):
  for r in self.review['records']:
   for p in r['parts']:p['review'].update({'association':'yes','bbox':'yes','type':'uncertain'})
  q=validate(self.review,self.cases,self.sha)
  self.assertTrue(q['structural_valid']);self.assertEqual(q['completed_parts'],42)
 def test_incorrect_manifest_hash(self):
  self.review['source_cases_sha256']='0'*64
  self.assertFalse(validate(self.review,self.cases,self.sha)['structural_valid'])

if __name__=='__main__':unittest.main()
