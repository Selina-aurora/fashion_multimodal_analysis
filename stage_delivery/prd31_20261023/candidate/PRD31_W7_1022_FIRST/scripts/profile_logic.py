"""Nested wall-time accounting; wrappers exist only inside this process."""
from collections import defaultdict
from contextlib import contextmanager
import functools
import time
import numpy as np


class SpanProfiler:
    def __init__(self, synchronize):
        self.synchronize=synchronize
        self.events=[]
        self.stack=[]
        self.patches=[]

    def call(self,name,fn,*args,**kwargs):
        self.synchronize()
        start=time.perf_counter()
        frame={'name':name,'children_ms':0.0,'depth':len(self.stack)}
        self.stack.append(frame)
        failed=False
        try:
            return fn(*args,**kwargs)
        except BaseException:
            failed=True
            raise
        finally:
            self.synchronize()
            elapsed=(time.perf_counter()-start)*1000
            self.stack.pop()
            if self.stack:self.stack[-1]['children_ms']+=elapsed
            self.events.append({'name':name,'inclusive_ms':elapsed,
                                'exclusive_ms':elapsed-frame['children_ms'],
                                'depth':frame['depth'],'failed':failed})

    def patch(self,obj,name,label):
        original=getattr(obj,name)
        owned=name in vars(obj)
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            return self.call(label,original,*args,**kwargs)
        setattr(obj,name,wrapped)
        self.patches.append((obj,name,original,owned))

    @contextmanager
    def instrument(self,backend,module):
        try:
            self.patch(module,'roi_images','roi_preparation')
            self.patch(module.Image,'composite','image_composite')
            self.patch(backend.color,'robust_garment_embedding','color_embedding_total')
            self.patch(backend.color,'classify_color','color_classification')
            self.patch(backend.pattern,'encode_image','image_encoding_helper')
            self.patch(backend.pattern,'classify_with_prototypes','prototype_classification')
            self.patch(backend.clip_model,'get_image_features','clip_visual_forward')
            yield self
        finally:
            for obj,name,original,owned in reversed(self.patches):
                if owned:setattr(obj,name,original)
                else:delattr(obj,name)
            self.patches.clear()


def summarize_spans(events):
    by=defaultdict(list)
    for e in events:by[e['name']].append(e)
    roots=[e for e in events if e['name']=='attribute_bundle']
    total=sum(e['inclusive_ms'] for e in roots)
    result={}
    for name,values in sorted(by.items()):
        result[name]={'calls':len(values),
                      'inclusive_total_ms':sum(v['inclusive_ms'] for v in values),
                      'exclusive_total_ms':sum(v['exclusive_ms'] for v in values),
                      'mean_inclusive_ms_per_call':float(np.mean([v['inclusive_ms'] for v in values])),
                      'exclusive_fraction_of_profiled_total':sum(v['exclusive_ms'] for v in values)/total if total else None}
    exclusive=sum(e['exclusive_ms'] for e in events)
    return {'spans':result,'profiled_root_calls':len(roots),'profiled_total_ms':total,
            'sum_all_exclusive_ms':exclusive,'accounting_residual_ms':total-exclusive,
            'inclusive_spans_overlap_do_not_sum':True,
            'instrumented_times_include_synchronization_overhead':True,
            'profiled_times_are_not_acceptance_latency':True}


def check_collar_geometry(collar,size):
    if collar.get('status')!='detected':
        return {'detected':False,'geometry_valid':None,'reason':collar.get('status','missing')}
    width,height=size
    box=np.asarray(collar.get('bbox'),dtype=float)
    mask=np.asarray(collar.get('mask'))
    valid=(box.shape==(4,) and np.isfinite(box).all() and
           0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height and
           mask.shape==(height,width) and np.isin(mask,[0,1,255]).all() and bool((mask>0).any()) and
           collar.get('coordinate_space')=='original_image')
    if valid:
        ys,xs=np.nonzero(mask)
        valid=(xs.min()>=box[0]-1e-3 and xs.max()+1<=box[2]+1e-3 and
               ys.min()>=box[1]-1e-3 and ys.max()+1<=box[3]+1e-3)
    return {'detected':True,'geometry_valid':bool(valid),'mask_shape':list(mask.shape)}
