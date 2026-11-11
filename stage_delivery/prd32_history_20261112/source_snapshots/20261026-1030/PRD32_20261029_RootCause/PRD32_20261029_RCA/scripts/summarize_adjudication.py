#!/usr/bin/env python3
"""Summarize reviewed root causes without fabricating decisions."""
import argparse,csv,json,collections,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
ALLOWED={'CORRECT','UNREVIEWED','TARGET_BINDING','CLASS_CONFUSION','MIXED','INSUFFICIENT'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--review-json',required=True);p.add_argument('--output',default=str(ROOT/'rootcause_final_summary.json'))
 a=p.parse_args();j=json.load(open(a.review_json,encoding='utf8'));assert j['schema']=='prd32_rootcause_human_adjudication_v1'
 baseline={x['case']['case_id']:x for x in json.load(open(ROOT/'RootCause_18Case_Detail.json',encoding='utf8'))}
 results=[];seen=set()
 for row in j['cases']:
  cid=row['case_id'];assert cid in baseline and cid not in seen,cid;seen.add(cid)
  for v in ('V2','V3'):
   info=row['version_results'][v];b=baseline[cid]['case'];actual_correct=b[f'{v.lower()}_final_root_cause']=='CORRECT'
   assert info['correct']==actual_correct,(cid,v,'correct flag changed')
   cause=info['cause'];assert cause in ALLOWED,(cid,v,cause)
   if actual_correct:assert cause=='CORRECT',(cid,v)
   else:assert cause!='CORRECT',(cid,v)
   results.append({'case_id':cid,'version':v,'human_category':row['human_category'],
    'correct':actual_correct,'cause':cause,'confidence':info.get('confidence',''),
    'evidence':info.get('evidence','')})
 assert len(seen)==18
 errors=[x for x in results if not x['correct']];assert len(errors)==25
 counts=collections.Counter(x['cause'] for x in errors)
 per_version={v:dict(collections.Counter(x['cause'] for x in errors if x['version']==v))for v in ('V2','V3')}
 summary={'schema':'prd32_rootcause_final_aggregation_v1','scope':'train_source_dev_not_independent',
 'cases':18,'error_predictions':25,'reviewed_predictions':25-counts.get('UNREVIEWED',0),
 'cause_counts_errors':dict(counts),'per_version_errors':per_version,
 'confirmed_binding':counts.get('TARGET_BINDING',0),'confirmed_class_confusion':counts.get('CLASS_CONFUSION',0),
 'mixed':counts.get('MIXED',0),'insufficient':counts.get('INSUFFICIENT',0),
 'unreviewed':counts.get('UNREVIEWED',0),
 'notes':['Counts are per wrong prediction, not per image.','Unreviewed and insufficient are never included in confirmed root-cause percentages.','Human diagnosis is a visual audit, not a model attention trace.']}
 pathlib.Path(a.output).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
 path=pathlib.Path(a.output).with_name('RootCause25_Adjudicated.csv')
 with open(path,'w',encoding='utf-8-sig',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(errors[0]));w.writeheader();w.writerows(errors)
 print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
