"""Deterministic interface tests on installed project code; no model inference."""
import numpy as np
from PIL import Image
from fashion_multimodal_analysis.integration.pipeline import GarmentROI, FashionPipeline, crop_bounds, roi_images
from fashion_multimodal_analysis.evaluation.protocol import original_box


def run_checks():
 results=[]
 def ok(name,condition):
  if not condition:raise AssertionError(name)
  results.append({'test':name,'status':'PASS'})
 image=Image.new('RGB',(100,80),'white');mask=np.zeros((80,100),dtype=bool);mask[20:50,10:40]=True
 roi=GarmentROI('pred_test','top',.9,[10.2,20.7,39.3,49.4],mask)
 ok('fractional_roi_floor_ceil',crop_bounds(roi.bbox,image.size)==[10,20,40,50])
 crop,cm,origin=roi_images(image,roi)
 ok('crop_mask_size_alignment',crop.size==cm.size==(30,30) and origin==[10,20,40,50])
 ok('offset_xyxy_mapping',original_box([2,3,8,9],origin)==[12.,23.,18.,29.])
 ok('anisotropic_scale_mapping',original_box([2,3,8,9],[10,20,40,50],2,.5)==[14.,21.5,26.,24.5])
 ok('boundary_clipping',crop_bounds([-2.5,-3.2,101.1,82],image.size)==[0,0,100,80])
 for name,box in [('zero_area',[1,1,1,2]),('nan',[float('nan'),0,5,5]),('outside',[110,0,120,4])]:
  try:crop_bounds(box,image.size)
  except ValueError:ok('reject_'+name,True)
  else:raise AssertionError('accepted invalid box '+name)
 try:roi_images(image,GarmentROI('bad','top',.9,[1,1,5,5],np.ones((5,5))))
 except ValueError:ok('reject_wrong_mask_dimensions',True)
 else:raise AssertionError('accepted invalid mask dimensions')
 class Backend:
  config={}
  def segment(self,im):return [roi],1.
  def ground(self,im,r,region):
   if region=='pocket':raise RuntimeError('injected grounding failure')
   return {'status':'missed','target_region':region,'bbox':None},1.
  def attributes(self,im,r,regions,attributes,controlled=False):
   assert r.instance_id=='pred_test' and controlled is False
   assert set(regions)=={'collar','pocket'}
   return {a:{'status':'predicted','label':'black','confidence':.9} for a in attributes},1.
 result=FashionPipeline(Backend()).run(image,['collar','pocket'],['primary_color'])
 ok('all_predicted_instances_transferred',len(result['instances'])==1 and result['instances'][0]['roi'] is roi)
 ok('grounding_failure_recorded',result['failures']==[{'instance_id':'pred_test','stage':'grounding','target':'pocket'}])
 ok('remaining_stages_continue',result['instances'][0]['attributes']['primary_color']['label']=='black')
 class AttributeFailure(Backend):
  def attributes(self,*a,**k):raise RuntimeError('injected attributes failure')
 result=FashionPipeline(AttributeFailure()).run(image,['collar'],['primary_color','pattern'])
 ok('attribute_failure_fields_present',set(result['instances'][0]['attributes'])=={'primary_color','pattern'} and all(a['status']=='attribute_failed' for a in result['instances'][0]['attributes'].values()))
 class SegFailure(Backend):
  def segment(self,im):raise RuntimeError('injected segmentation failure')
 result=FashionPipeline(SegFailure()).run(image,['collar'],['primary_color'])
 ok('segmentation_failure_stops_chain',result['status']=='segmentation_failed' and not result['instances'])
 return {'status':'PASS','model_inference_run':False,'tests':results}

if __name__=='__main__':
 import json
 print(json.dumps(run_checks(),ensure_ascii=False,indent=2))
