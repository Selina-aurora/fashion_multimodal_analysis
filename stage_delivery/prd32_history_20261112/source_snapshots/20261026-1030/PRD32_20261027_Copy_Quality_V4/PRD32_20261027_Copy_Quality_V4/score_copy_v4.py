#!/usr/bin/env python3
"""Score human review of V4 against original 3-item DEV score. NO official acceptance."""
import argparse,collections,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--review',required=True);p.add_argument('--old-review',default='');p.add_argument('--output',default='');a=p.parse_args()
 run=Path(a.run_dir); rows=[json.loads(x) for x in (run/'candidate_drafts.jsonl').read_text(encoding='utf8').splitlines() if x.strip()]
 review=json.loads(Path(a.review).read_text(encoding='utf8'))
 if review.get('schema')!='prd32_copy_v4_review_dev_v1':raise ValueError('Wrong review schema')
 rev={x['case_id']:x for x in review['results']}
 if len(rev)!=len(review['results']):raise ValueError('Duplicate case ID in review')
 vals=[]
 for r in rows:
  cid=r['case_id'];label=rev.get(cid,{}).get('verdict','PENDING')
  if label not in ('PENDING','SATISFIED','NOT_SATISFIED','BLOCKED_REVIEW'):raise ValueError('Unknown verdict '+str(label))
  if not r.get('draft') and label=='SATISFIED':raise ValueError('Blocked/no-draft case cannot be called satisfied: '+cid)
  vals.append({'case_id':cid,'label':label,'candidate_exists':r.get('draft') is not None,'comment':rev.get(cid,{}).get('notes','')})
 reviewed=sum(r['label']!='PENDING' for r in vals)
 positive=sum(r['label']=='SATISFIED' for r in vals)
 result={'schema':'prd32_copy_dev_comparison_v4','v4_satisfied_count':positive,'reviewed_count':reviewed,'total_cases':len(vals),
    'satisfaction_over_all_cases':round(positive/len(vals),4) if vals else None,
    'all_cases_reviewed':reviewed==len(vals),'records':vals,'prd_acceptance':False,
    'note':'Development overlap only. Holds count as unsatisfied, NEVER drop from denominator.'}
 if a.old_review:
  o=json.loads(Path(a.old_review).read_text(encoding='utf8'))
  old={x['case_id']:x['verdict'] for x in o['results'] if x.get('task')=='copy'}
  common=[x for x in vals if x['case_id'] in old]
  result['old_development_comparison']={'matched_cases':len(common),
     'old_satisfied':sum(old[x['case_id']]=='SATISFIED' for x in common),
     'v4_satisfied':sum(x['label']=='SATISFIED' for x in common),
     'not_independent':True}
 path=Path(a.output) if a.output else run/'copy_v4_review_metrics.json'
 path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
