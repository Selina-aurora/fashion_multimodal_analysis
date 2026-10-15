#!/usr/bin/env python3
"""Read-only audit of manually reviewed emitted candidates, without claim of recall."""
from __future__ import annotations
import argparse, json, csv, collections, hashlib, pathlib, math

TYPES={
 'case_01_pred_0001_waist': 'hand/pocket confusion',
 'case_01_pred_0002_waist': 'hem/waist semantic ambiguity',
 'case_02_pred_0001_hem': 'knee fold vs trouser hem',
 'case_02_pred_0001_waist': 'knee opening vs waistband; waistband outside frame',
 'case_06_pred_0002_hem': 'shoe vs dress hem',
 'case_07_pred_0001_hem': 'handbag occlusion',
 'case_07_pred_0003_waist': 'background gap vs waistband',
 'case_07_pred_0007_hem': 'bag hardware vs garment hem',
 'case_07_pred_0007_waist': 'bag hardware vs garment waist; note copied from hem',
 'case_08_pred_0004_hem': 'oversized garment-body proposal',
 'case_08_pred_0006_hem': 'cross-garment skirt confusion',
 'case_04_pred_0001_hem': 'false dress garment category',
 'case_06_pred_0003_hem': 'false top: dress split into top',
 'case_06_pred_0003_waist': 'false top: dress split into top',
 'case_08_pred_0005_hem': 'false dress: top + skirt merged',
 'case_08_pred_0005_waist': 'false dress: top + skirt merged',
}

def read_json(path): return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
def write_csv(path,rows,fields):
 with pathlib.Path(path).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def build(source, out):
 cases=read_json(source)['cases']
 assert len(cases)==28 and all(x['review_status']=='confirmed' for x in cases), 'Expected 28 confirmed diagnostic cases'
 rows=[]; stats=collections.defaultdict(collections.Counter)
 for c in cases:
  x1,y1,x2,y2=map(float,c['garment_bbox']); a,b,d,e=map(float,c['region_bbox'])
  h=max(1e-8,y2-y1);w=max(1e-8,x2-x1)
  feature={'normalized_region_center_y':round(((b+e)/2-y1)/h,4),
     'normalized_region_width':round((d-a)/w,4),
     'normalized_region_height':round((e-b)/h,4),
     'region_height_px':round(e-b,1),
     'region_width_px':round(d-a,1)}
  reason=TYPES.get(c['review_id'], '' if c['verdict']=='correct' else 'read reviewer comment')
  rows.append({**{k:c[k] for k in ('review_id','case_id','instance_id','source_image_sha256','predicted_category','target_region','verdict','model_region_status')},
     'garment_bbox':json.dumps(c['garment_bbox'],ensure_ascii=False),'region_bbox':json.dumps(c['region_bbox'],ensure_ascii=False),
     **feature,'error_mechanism_hypothesis':reason,'human_comment':c['comment'],
     'supervised_bbox_gt_available':'NO','independent_validation':'NO'})
  stats[c['target_region']][c['verdict']]+=1
 assert {k:sum(v.values()) for k,v in stats.items()}=={'pocket':9,'hem':11,'waist':8}
 assert {k:sum(v['correct'] for v in stats.values()) for k in ['n']}['n']==3
 fields=list(rows[0]);out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
 write_csv(out/'historical_28_candidate_error_register.csv',rows,fields)
 write_csv(out/'hem_waist_19_candidate_register.csv',[r for r in rows if r['target_region'] in ('hem','waist')],fields)
 summary=[{'region':k,'total':sum(v.values()),'correct':v['correct'],'wrong_region':v['wrong_region'],'upstream_instance_error':v['upstream_instance_error'],'comment':'Emitted candidates only; NOT recall / IoU / mAP'} for k,v in stats.items()]
 summary=sorted(summary,key=lambda x:['hem','waist','pocket'].index(x['region']))
 write_csv(out/'historical_28_summary_by_region.csv',summary,list(summary[0]))
 (out/'summary.json').write_text(json.dumps({'source_sha256':hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest(),'n_candidates':28,'n_images':len({c['source_image_sha256'] for c in cases}),'by_region':summary,'interpretation':'The review covers 28 emitted candidates only, lacks per-region exhaustively annotated positives and GT bbox; cannot compute recall/IoU/mAP; mixed batch not comparable with pocket24 audit.'},ensure_ascii=False,indent=2),encoding='utf-8')
 return rows

def pocket_audit(source,out):
 original=read_json(source)
 parts=[]
 for c in original['records']:
  for p in c['parts']:
   r=p['review'];txt=r.get('note',''); flags=[]
   if r['bbox']=='yes' and ('定位不准确' in txt or '候选框偏大' in txt or '框偏大' in txt): flags.append('BBOX_NOTE_CONTRADICTION')
   if r['type']=='yes' and ('类型待复核' in txt or '类型依据不足' in txt or '无法确认' in txt):flags.append('TYPE_NOTE_CONTRADICTION')
   parts.append({'record_id':c['record_id'],'split':c['split'],'category':c['category'],'part_annotation_id':p['part_annotation_id'],'part_type_proposal':p['part_type_proposal'],'association':r['association'],'bbox':r['bbox'],'type':r['type'],'flags':';'.join(flags),'note':txt})
 assert len(parts)==42
 out=pathlib.Path(out);write_csv(out/'pocket24_42_human_review_flat.csv',parts,list(parts[0]))
 bysplit=[]
 for s in ['train','val']:
  p=[x for x in parts if x['split']==s]
  bysplit.append({'split':s,'reviewed_parts':len(p),'association_yes':sum(x['association']=='yes' for x in p),'bbox_yes':sum(x['bbox']=='yes' for x in p),'type_yes':sum(x['type']=='yes' for x in p),'conflict_flags':sum(bool(x['flags']) for x in p)})
 write_csv(out/'pocket24_train_val_separate_summary.csv',bysplit,list(bysplit[0]))
 return parts,bysplit

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--historical',required=True);p.add_argument('--pocket',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 rows=build(a.historical,a.out);pp,ss=pocket_audit(a.pocket,a.out)
 print('HISTORICAL_28_OK',len(rows),'HEM_WAIST',sum(r['target_region'] in ['hem','waist'] for r in rows))
 print('POCKET24_PARTS_OK',len(pp),'split_summary',ss)
