#!/usr/bin/env python3
# coding: utf-8
"""Offline POST-HOC V5 target matching; never called by V5 inference.

Historical pilot bbox + AI-assisted human class used solely for development diagnosis.
"""
import argparse,csv,hashlib,json,math,sys,statistics
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(1048576),b''):h.update(v)
 return h.hexdigest()
def j(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def jl(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def box_iou(a,b):
 if any(not math.isfinite(float(x)) for x in a+b):raise ValueError('Non-finite box')
 x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3]); intersect=max(0,x2-x1)*max(0,y2-y1)
 area1=max(0,a[2]-a[0])*max(0,a[3]-a[1]);area2=max(0,b[2]-b[0])*max(0,b[3]-b[1]);union=area1+area2-intersect
 return intersect/union if union>0 else 0.
def analyze(frozen,raw,output):
 fr=Path(frozen);pv=Path(raw);out=Path(output)
 if out.exists():raise ValueError('Refusing to overwrite evaluation directory')
 meta=j(fr/'freeze.json');pf=fr/'frozen_inference_plan.jsonl';hf=fr/'frozen_human_reference.jsonl'
 if meta.get('status')!='FROZEN_DEV_NOT_INDEPENDENT' or sha(pf)!=meta.get('plan_sha256') or sha(hf)!=meta.get('human_reference_sha256'):
  raise ValueError('Frozen identity mismatch')
 pred=j(pv/'v5_real_summary.json')
 if pred.get('status')!='V5_RAW_INFERENCE_COMPLETED' or sha(pv/'v5_raw_predictions.jsonl')!=pred.get('v5_predictions_sha256'):
  raise ValueError('V5 result incomplete or changed')
 plans={r['case_id']:r for r in jl(pf)};humans={r['case_id']:r for r in jl(hf)}; raws={r['case_id']:r for r in jl(pv/'v5_raw_predictions.jsonl')}
 if not(len(plans)==len(humans)==len(raws)==12 and plans.keys()==humans.keys()==raws.keys()):raise ValueError('Case IDs mismatch')
 rows=[]
 for cid,plan in sorted(plans.items()):
  h=humans[cid];p=raws[cid]
  if plan['image_sha256']!=h['image_sha256'] or plan['image_sha256']!=p['image_sha256']:raise ValueError('Image SHA mismatch '+cid)
  if plan.get('bbox_origin')!='historical_inventory_gt_pilot_NOT_V5_prediction':raise ValueError('Unexpected historical bbox origin')
  gt=list(map(float,plan['bbox']))
  if len(gt)!=4 or not(0<=gt[0]<gt[2] and 0<=gt[1]<gt[3]):raise ValueError('Invalid pilot bbox')
  cand=[(box_iou(gt,list(map(float,x['bbox_xyxy']))),x) for x in p['instances']]
  cand.sort(key=lambda x:(-x[0],-x[1]['score'],x[1]['instance_id']))
  best_iou,best=cand[0] if cand else (0,None)
  hit=bool(best and best_iou>=.5);correct=bool(hit and best['predicted_category']==h['category'])
  rows.append({'case_id':cid,'human_category':h['category'],'scoreable':bool(h['scoreable']),
    'v5_instances':len(cand),'max_bbox_iou':round(best_iou,4),'selected_pred_instance_id_FOR_DIAGNOSIS_ONLY':best['instance_id'] if best else '',
    'selected_pred_class_FOR_DIAGNOSIS_ONLY':best['predicted_category'] if best else '',
    'target_bbox_overlap_at_0_5':hit,'target_bbox_and_class_match_at_0_5':correct,'confidence':best['score'] if best else '',
    'overlap_other_garments':h['overlap_other_garments'],'note':'Pilot bbox posthoc evaluation; this match did not influence V5 inference or any Qwen input'})
 scored=[r for r in rows if r['scoreable']]; hits=sum(r['target_bbox_overlap_at_0_5'] for r in scored); cls=sum(r['target_bbox_and_class_match_at_0_5'] for r in scored)
 summary={'schema':'prd32_v5_raw_detection_posthoc_pilot_summary_v1','scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY_NOT_INDEPENDENT',
 'images':len(rows),'scorable_human_cases':len(scored),'bbox_overlap_at_0_5':hits,'bbox_and_class_match_at_0_5':cls,
 'bbox_overlap_rate':round(hits/len(scored),4) if scored else None,'joint_match_rate':round(cls/len(scored),4) if scored else None,
 'no_v5_instances_images':sum(r['v5_instances']==0 for r in rows),
 'human_labels_used_only_posthoc':True,'selection_is_oracle_pilot_bbox_FOR_DIAGNOSIS_NOT_DEPLOYMENT':True,
 'no_mask_iou_evaluated':True,'official_PRD_acceptance':False,
 'caution':'Historical pilot boxes do not guarantee a complete, independent GT. Best-IoU association is not an operational target selector.'}
 out.mkdir(parents=True)
 with (out/'v5_posthoc_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 save(out/'v5_posthoc_summary.json',summary)
 print('V5_POSTHOC_DONE',json.dumps({'scored':len(scored),'bbox_hits':hits,'joint_hits':cls},ensure_ascii=False))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--frozen',default='/workspace/PRD32_NEWCohort_FROZEN_20261029_run01');ap.add_argument('--raw',default='/workspace/PRD32_V5_REAL_20261029_run01');ap.add_argument('--output',default='/workspace/PRD32_V5_POSTHOC_20261029_run01');a=ap.parse_args()
 try:analyze(a.frozen,a.raw,a.output)
 except Exception as e:print('V5_POSTHOC_BLOCKED',type(e).__name__+': '+str(e),file=sys.stderr);sys.exit(2)
