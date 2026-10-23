"""Validate persisted full-chain outputs; does not grade semantic accuracy."""
import hashlib,json,math
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def verify(report):
 report=Path(report);cases=json.loads((report/'cases.json').read_text());config=json.loads((report/'run_config.json').read_text())
 issues=[];counts=Counter();regions=Counter();attrs=Counter();evidence=[]
 def check(ok,case,instance,name):
  counts['checks']+=1
  if not ok:issues.append({'case':case,'instance_id':instance,'check':name})
 def box_ok(b,w,h):
  return isinstance(b,(list,tuple)) and len(b)==4 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in b) and 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h
 for c in cases:
  case=c['case_dir'];folder=report/case;payload=json.loads((folder/'predictions.json').read_text())
  path=report/'source_images'/(case+'.jpg')
  if not path.is_file():path=Path(c['source_image'])
  with Image.open(path) as im:w,h=im.size
  if c.get('source_sha256'):check(sha(path)==c['source_sha256'],case,'','original_source_sha256')
  check(payload['status']=='completed' and c['pipeline_status']=='completed',case,'','chain_completed')
  check(not payload['failures'] and not c['failures'],case,'','no_failed_downstream_calls')
  check(not payload.get('interface_issues'),case,'','exporter_interface_checks')
  check(len(payload['instances'])==c['instance_count'],case,'','instance_count')
  ids=[i['instance_id'] for i in payload['instances']]
  check(len(ids)==len(set(ids)),case,'','unique_instance_ids')
  for t in payload['timings']:
   check(isinstance(t.get('ms'),(int,float)) and math.isfinite(t['ms']) and t['ms']>=0,case,t.get('instance_id',''),'finite_timings')
   check(t.get('module') in {'segmentation','grounding','attributes'},case,t.get('instance_id',''),'module_name')
   check(t.get('instance_id') in ids if t.get('module')!='segmentation' else True,case,t.get('instance_id',''),'timing_instance_link')
  for i in payload['instances']:
   iid=i['instance_id'];counts['instances']+=1
   check(i['category'] in config['required_prd_classes'],case,iid,'category_taxonomy')
   check(isinstance(i['score'],(int,float)) and math.isfinite(i['score']) and 0<=i['score']<=1,case,iid,'segmentation_score')
   check(i['provenance']=='segmentation_prediction',case,iid,'predicted_roi_provenance')
   check(box_ok(i['bbox'],w,h),case,iid,'garment_original_bbox')
   gm=np.array(Image.open(folder/i['mask_path']))>0
   check(gm.shape==(h,w),case,iid,'garment_mask_original_dimensions')
   check(set(config['required_regions'])<=set(i['regions']),case,iid,'required_region_fields')
   check(set(config['required_attributes'])<=set(i['attributes']),case,iid,'required_attribute_fields')
   for region,p in i['regions'].items():
    counts['region_records']+=1;regions[p['status']]+=1
    check(p.get('target_region',region)==region,case,iid,region+':target_name')
    check(p['status'] in {'detected','requires_review','not_applicable','missed','empty_mask','rejected_region_geometry'},case,iid,region+':status')
    if p.get('bbox') is not None:
     check(box_ok(p['bbox'],w,h),case,iid,region+':original_bbox')
     check(p.get('coordinate_space')=='original_image',case,iid,region+':coordinate_space')
     b=i['bbox'];expected=[max(0,math.floor(b[0])),max(0,math.floor(b[1])),min(w,math.ceil(b[2])),min(h,math.ceil(b[3]))]
     check(p.get('crop_origin')==expected,case,iid,region+':crop_origin_floor_ceil')
     if p.get('proposal_bbox'):
      pb=p['proposal_bbox'];ox,oy,x2,y2=expected
      check(box_ok(pb,w,h) and ox<=pb[0]<pb[2]<=x2 and oy<=pb[1]<pb[3]<=y2,case,iid,region+':proposal_inside_crop')
    if p.get('mask_path'):
     pm=np.array(Image.open(folder/p['mask_path']))>0;counts['region_masks']+=1
     check(pm.shape==(h,w),case,iid,region+':mask_original_dimensions')
     if pm.shape==gm.shape:check(not np.any(pm & ~gm),case,iid,region+':mask_inside_instance')
     if p.get('bbox') and pm.shape==(h,w) and pm.any():
      yy,xx=np.nonzero(pm);tight=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]
      check(np.allclose(p['bbox'],tight,atol=1e-5),case,iid,region+':bbox_matches_full_mask')
    if p['status']=='detected':check(bool(p.get('bbox')) and bool(p.get('mask_path')),case,iid,region+':detected_has_geometry')
   for name,a in i['attributes'].items():
    counts['attribute_records']+=1;attrs[a['status']]+=1
    check(a['status'] in {'predicted','requires_review','not_applicable','localization_missing'},case,iid,name+':status')
    if a['status']=='predicted':
     check(a.get('label') in config['attribute_taxonomy'][name],case,iid,name+':label_taxonomy')
    conf=a.get('confidence')
    if conf is not None:check(isinstance(conf,(int,float)) and math.isfinite(conf) and 0<=conf<=1,case,iid,name+':confidence')
   modules=Counter(t['module'] for t in payload['timings'] if t.get('instance_id')==iid)
   check(modules['grounding']==len(config['required_regions']) and modules['attributes']==1,case,iid,'all_downstream_stage_calls')
  evidence.append({'case':case,'image_size':[w,h],'prediction_sha256':sha(folder/'predictions.json'),'instances':len(ids)})
 return {'status':'PASS' if not issues else 'FAIL','images':len(cases),'counts':dict(counts),'region_status_counts':dict(regions),'attribute_status_counts':dict(attrs),'issues':issues,'evidence':evidence,'model_inference_run':False,'semantic_accuracy_evaluated':False,'formal_prd_acceptance':False}

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('report');p.add_argument('--output');a=p.parse_args()
 result=verify(a.report)
 if a.output:Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False,indent=2))
 raise SystemExit(0 if result['status']=='PASS' else 1)
