"""Full frozen pipeline execution; no inference result cache or GT inputs."""
import copy
from dataclasses import is_dataclass
import hashlib
import math
import numbers
import time
import numpy as np
from PIL import Image
from white_reuse import WhiteReuse
from run_attributes import SIX,write


def compare(a,b,atol=1e-4):
    errors=[];deltas=[]
    def visit(x,y,path):
        if isinstance(x,dict) and isinstance(y,dict):
            if set(x)!=set(y):errors.append({'path':path,'reason':'keys_changed'})
            for k in sorted(set(x)&set(y)):visit(x[k],y[k],path+'.'+str(k))
        elif isinstance(x,list) and isinstance(y,list):
            if len(x)!=len(y):errors.append({'path':path,'reason':'length_changed'})
            for i,(xx,yy) in enumerate(zip(x,y)):visit(xx,yy,path+f'[{i}]')
        elif isinstance(x,bool) or isinstance(y,bool):
            if type(x)!=type(y) or x!=y:errors.append({'path':path,'reason':'boolean_changed'})
        elif isinstance(x,numbers.Real) and isinstance(y,numbers.Real):
            delta=abs(float(x)-float(y));deltas.append(delta)
            if not math.isfinite(float(x)) or not math.isfinite(float(y)) or delta>atol:
                errors.append({'path':path,'reason':'numeric_drift','absolute_difference':delta})
        elif type(x)!=type(y) or x!=y:errors.append({'path':path,'reason':'value_changed'})
    visit(a,b,'output')
    return {'passed':not errors,'numeric_atol':atol,'max_numeric_absolute_difference':max(deltas,default=0.),'differences':errors}


def mask_summary(mask,size):
    array=np.asarray(mask)
    if array.shape!=(size[1],size[0]):raise ValueError('MASK_NOT_FULL_ORIGINAL_IMAGE')
    if not np.isin(array,[0,1,255]).all():raise ValueError('NON_BINARY_MASK')
    binary=np.ascontiguousarray(array>0,dtype=np.uint8)
    return {'shape':list(array.shape),'pixels':int(binary.sum()),'binary_sha256':hashlib.sha256(binary.tobytes()).hexdigest()}


def canonical(value,size,key=None):
    if value is None:return None
    if key=='mask':return mask_summary(value,size)
    # Avoid asdict() copying full image masks just to hash them.
    if is_dataclass(value):value=vars(value)
    if isinstance(value,dict):return {str(k):canonical(v,size,k) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [canonical(v,size) for v in value]
    if isinstance(value,np.ndarray):return canonical(value.tolist(),size)
    if isinstance(value,np.generic):return value.item()
    if hasattr(value,'detach'):return canonical(value.detach().cpu().tolist(),size)
    if isinstance(value,(str,int,float,bool)):return value
    raise TypeError('UNSUPPORTED_OUTPUT_TYPE: '+type(value).__name__)


class FullBackend:
    def __init__(self,backend,candidate,capture_raw=False):
        self.backend=backend;self.config=backend.config;self.candidate=candidate
        self.capture_raw=capture_raw;self.raw={};self.cache={}

    def segment(self,image):return self.backend.segment(image)
    def ground(self,image,roi,region):return self.backend.ground(image,roi,region)

    def attributes(self,image,roi,regions,names,controlled=False):
        if controlled or names!=SIX:raise RuntimeError('UNEXPECTED_ATTRIBUTE_INPUT')
        if self.candidate:
            state=WhiteReuse(self.backend)
            with state.activate():result,ms=self.backend.attributes(image,roi,regions,names,controlled=False)
            if self.capture_raw:
                self.cache[roi.instance_id]={'robust_calls':state.robust_calls,'white_reuses':state.white_reuses,'fallback_encodes':state.fallback_encodes}
        else:result,ms=self.backend.attributes(image,roi,regions,names,controlled=False)
        if self.capture_raw:self.raw[roi.instance_id]=copy.deepcopy(result)
        return result,ms


def full_call(backend,Pipeline,path,queries,candidate,sync,capture_raw=False,mask_export=None):
    adapter=FullBackend(backend,candidate,capture_raw);pipeline=Pipeline(adapter)
    # Model setup is outside. Actual open/decode, segmentation, all grounding,
    # attributes, reviews and final CUDA synchronization are inside this timer.
    sync();start=time.perf_counter()
    with Image.open(path) as source:image=source.convert('RGB')
    result=pipeline.run(image,list(queries),list(SIX))
    sync();elapsed=(time.perf_counter()-start)*1000
    # Audit hashing/JSON conversion/copying below is never measured as inference.
    semantic=canonical({k:v for k,v in result.items() if k!='timings'},image.size)
    audit={'raw_attributes':canonical(adapter.raw,image.size),'candidate_cache':adapter.cache} if capture_raw else None
    if mask_export is not None:
        mask_export.mkdir(parents=True,exist_ok=True);inventory=[]
        for n,item in enumerate(result.get('instances',[])):
            values=[('garment',item['roi'].mask)]+[(q,r.get('mask')) for q,r in item['regions'].items()]
            for k,(name,mask) in enumerate(values):
                if mask is None:continue
                filename=f'{n:04d}_{k:02d}.png'
                Image.fromarray((np.asarray(mask)>0).astype('uint8')*255).save(mask_export/filename)
                inventory.append({'instance_id':item['roi'].instance_id,'region':name,'file':filename})
        write(mask_export/'index.json',inventory)
    return semantic,elapsed,result.get('timings',[]),audit


def health(snapshot,queries):
    errors=[]
    if snapshot.get('status')!='completed':errors.append('PIPELINE_STATUS')
    if snapshot.get('failures'):errors.append('PIPELINE_FAILURES')
    ids=[]
    for item in snapshot.get('instances',[]):
        ids.append(item['roi']['instance_id'])
        if item['roi'].get('provenance')!='segmentation_prediction':errors.append('NON_PREDICTED_ROI')
        if set(item['regions'])!=set(queries):errors.append('REGION_KEYS')
        if set(item['attributes'])!=set(SIX):errors.append('ATTRIBUTE_KEYS')
        if any(v.get('status')=='attribute_failed' for v in item['attributes'].values()):errors.append('ATTRIBUTE_FAILED')
    if len(ids)!=len(set(ids)):errors.append('DUPLICATE_INSTANCE_ID')
    return {'passed':not errors,'errors':errors}


def reference_check(snapshot,reference):
    current=[item['roi'] for item in snapshot['instances']]
    errors=[]
    if len(current)!=len(reference):errors.append({'reason':'instance_count'})
    for n,(a,b) in enumerate(zip(current,reference)):
        exact={k:a[k] for k in ['instance_id','category','mask']}
        expected={k:b[k] for k in exact}
        if not compare(exact,expected,atol=0)['passed']:errors.append({'index':n,'reason':'id_category_or_mask'})
        if not compare(a['bbox'],b['bbox'],atol=1e-3)['passed']:errors.append({'index':n,'reason':'bbox'})
        if not compare(a['score'],b['score'],atol=1e-5)['passed']:errors.append({'index':n,'reason':'score'})
    return {'passed':not errors,'bbox_atol':1e-3,'score_atol':1e-5,'mask_exact':True,'differences':errors}
