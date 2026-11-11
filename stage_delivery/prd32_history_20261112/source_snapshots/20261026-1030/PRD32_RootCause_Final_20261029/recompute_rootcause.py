import json,collections,pathlib
p=pathlib.Path(__file__).parent/'RootCause_18_HumanReview_Final.json'
r=json.loads(p.read_text('utf-8'))['rows']
assert len(r)==18 and all(x['review_status']=='REVIEWED' for x in r)
for v in ['v2','v3']:
 c=collections.Counter(x[v+'_root_cause'] for x in r)
 print(v,dict(c));assert sum(c.values())==18
assert '黑色裙装' in next(x for x in r if x['case_id']=='train_dev_022')['reviewer_note']
