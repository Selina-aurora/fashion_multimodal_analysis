#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired-category sensitivity: leave the disputed skirt_diag_002 OUT, without modifying official frozen labels."""
import argparse,csv,json
from pathlib import Path

def yes(v):
 v=str(v).strip().lower()
 if v in {'true','1'}:return True
 if v in {'false','0'}:return False
 raise ValueError(f'Invalid scored boolean: {v}')

def summarize(rows):
  n=len(rows);v2=sum(yes(x['v2_correct']) for x in rows);v3=sum(yes(x['v3_correct']) for x in rows)
  return {'n':n,'v2_correct':v2,'v3_correct':v3,'v2_rate':round(v2/n,4) if n else None,'v3_rate':round(v3/n,4) if n else None,'v3_minus_v2_correct':v3-v2}

def main():
  ap=argparse.ArgumentParser();ap.add_argument('--comparison',required=True);ap.add_argument('--summary',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
  with open(a.comparison,encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
  assert len(rows)==5 and len({r['case_id'] for r in rows})==5, 'Unexpected sample counts/duplicates'
  scored=[r for r in rows if str(r['scoreable']).strip().lower()=='true']
  assert len(scored)==4 and any(r['case_id']=='skirt_diag_002' for r in scored),'Expected four scoreable including 002'
  assert [r for r in rows if r['case_id']=='skirt_diag_005'][0]['pair_group']=='EXCLUDED', 'Expected excluded 005'
  base=summarize(scored);reduced=summarize([r for r in scored if r['case_id']!='skirt_diag_002'])
  declared=json.load(open(a.summary,encoding='utf8'))
  assert declared['scoreable']==4 and declared['v2_match']==base['v2_correct'] and declared['v3_match']==base['v3_correct'],'Aggregate score mismatch'
  report={'schema':'prd32_skirt5_case002_label_sensitivity_v1','scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY_NOT_INDEPENDENT',
    'main_approved_review_including_002':base,'sensitivity_excluding_002':reduced,
    'excluded_from_main_scoring':['skirt_diag_005'],'not_a_relabel':True,
    'why':'skirt_diag_002 was changed from preliminary uncertain to user-reviewed skirt; report robustness without altering frozen GT',
    'caution':'Small, training-source, human/AI-assisted development subset. Not statistical significance or PRD acceptance.'}
  dst=Path(a.output);dst.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  print('SKIRT5_SENSITIVITY_OK',json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
