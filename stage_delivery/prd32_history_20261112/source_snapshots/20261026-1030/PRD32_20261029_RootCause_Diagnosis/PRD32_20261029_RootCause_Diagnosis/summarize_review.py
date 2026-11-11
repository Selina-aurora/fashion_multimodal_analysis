#!/usr/bin/env python3
"""Summarize human-adjudicated cause labels; refuses unreviewed as verified."""
import argparse,json,pathlib,collections,csv
HERE=pathlib.Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--review-json',required=True)
p.add_argument('--output',default='')
args=p.parse_args();args.output=args.output or str(HERE/'RootCause_Final_Results.json')
base=json.loads((HERE/'RootCause_18_ReviewSeed.json').read_text(encoding='utf-8'))['rows']
base_idx={x['case_id']:x for x in base}
actual=json.loads(pathlib.Path(args.review_json).read_text(encoding='utf-8'))
idx={x['case_id']:x for x in actual['rows']}
if len(idx)!=18 or set(idx)!=set(base_idx):raise RuntimeError('18-case ID set mismatch; do not merge unrelated samples')
allowed={'NOT_ERROR','BINDING','CLASS_BOUNDARY','BOTH','INSUFFICIENT'}
result={'schema':'prd32_rootcause_adjudication_summary_v1','scope':'TRAIN_SOURCE_DEV_NOT_INDEPENDENT','total_cases':18,'completed_cases':0,'model_case_errors':{'v2':14,'v3':11},'classified_by_version':{},'warnings':[]}
for key in ('v2','v3'):
 cnt=collections.Counter();checked=0
 for id,row in idx.items():
  a=base_idx[id]
  should_error=a[key+'_category']!=a['human_category'];val=row.get(key+'_root_cause','UNREVIEWED')
  if row.get('review_status')!='REVIEWED': continue
  if val not in allowed:result['warnings'].append(f'{id} {key}: root cause unreviewed/invalid');continue
  if not should_error and val!='NOT_ERROR':result['warnings'].append(f'{id} {key}: correct category but erroneous cause {val}');continue
  if should_error and val=='NOT_ERROR':result['warnings'].append(f'{id} {key}: wrong category but NOT_ERROR');continue
  cnt[val]+=1;checked+=1
 result['classified_by_version'][key]={'reviewed_and_valid':checked,'counts':dict(cnt)}
result['completed_cases']=sum(x.get('review_status')=='REVIEWED' for x in idx.values())
result['final_distribution_confirmed']=result['completed_cases']==18 and not result['warnings']
if not result['final_distribution_confirmed']:
 result['note']='INCOMPLETE. Do not cite category-specific root cause counts as final.'
path=pathlib.Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'SUMMARY_WRITTEN','cases_reviewed':result['completed_cases'],'final_confirmed':result['final_distribution_confirmed'],'warnings':result['warnings'][:10],'output':str(path)},ensure_ascii=False,indent=2))
