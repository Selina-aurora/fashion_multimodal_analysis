#!/usr/bin/env python3
"""Apply user-checked 13-case Crop/ID reviews to a 171-case cross-version CSV, preserving all originals.
Usage: python 11.05_merge_checked_scores.py <exported_review.json> [directory]
"""
import sys,csv,json,collections,pathlib
if len(sys.argv)<2: raise SystemExit('Usage: python 11.05_merge_checked_scores.py <exported_review.json> [output_folder]')
root=pathlib.Path(__file__).resolve().parent; review=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf8'))
assert review.get('schema')=='prd32_1105_copy_cross_variant_review_v1'
lookup={(x['case_id'],x['instance_id']):x for x in review['records']};assert len(lookup)==13
with (root/'171_Copy_四版本跨实验对照_旧分与建议分.csv').open(encoding='utf-8-sig',newline='') as f:
 r=csv.DictReader(f); cols=r.fieldnames;rows=list(r)
newcols=['crop_same_rubric_score','idv1_same_rubric_score','crop_checked','idv1_checked','recheck_note']
for x in rows:
 k=(x['case_id'],x['instance_id']); y=lookup.get(k)
 x['crop_same_rubric_score']=y['crop_reviewer_score'] if y and y['crop_user_checked'] else x['crop_original']
 x['idv1_same_rubric_score']=y['idv1_reviewer_score'] if y and y['idv1_user_checked'] else x['idv1_original']
 x['crop_checked']=str(bool(y and y['crop_user_checked']))
 x['idv1_checked']=str(bool(y and y['idv1_user_checked']))
 x['recheck_note']=y.get('reviewer_note','') if y else ''
unc=[x for x in rows if x['review_required']=='YES' and not (x['crop_checked']=='True' and x['idv1_checked']=='True')]
if unc: print('WARNING: 13-case rubric sync incomplete:',len(unc),'pending. No cross-version final claims.')
out=pathlib.Path(sys.argv[2]) if len(sys.argv)>2 else root;out.mkdir(exist_ok=True,parents=True)
with (out/'11.05_用户复核后_四版本Copy评分.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=cols+newcols);w.writeheader();w.writerows(rows)
summary={'review_coverage':{'required':13,'fully_checked':13-len(unc),'pending':len(unc)},'correct_counts':{k:sum(x[col]=='CORRECT' for x in rows) for k,col in [('focus','focus_revised'),('cv1','cv1_revised'),('crop','crop_same_rubric_score'),('idv1','idv1_same_rubric_score')]},'caution':'Even full 13-case review only handles 11/5 targeted changes; remaining 149 not blinded/rubric re-reviewed; comparison exploratory only.'}
(out/'11.05_用户复核后_谨慎汇总.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
