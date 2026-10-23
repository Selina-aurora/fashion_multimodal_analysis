"""One fixed candidate: reuse the SAME batch-one white embedding.

Scopes are limited to one attribute invocation. No cross-instance cache,
checkpoint edits, dtype changes, alternative views, or global installation.
"""
from contextlib import contextmanager
import functools
import numpy as np


@contextmanager
def patch(obj,name,replacement):
    original=getattr(obj,name);owned=name in vars(obj)
    setattr(obj,name,replacement)
    try:yield original
    finally:
        if owned:setattr(obj,name,original)
        else:delattr(obj,name)


def vector_array(value):
    return value.detach().cpu().float().numpy().reshape(-1)


class ForwardCounter:
    def __init__(self,vision):self.vision=vision;self.batch_sizes=[];self.input_sha256=[]
    def __enter__(self):
        original=self.vision.forward
        @functools.wraps(original)
        def forward(*args,**kwargs):
            values=kwargs.get('pixel_values',args[0] if args else None)
            self.batch_sizes.append(int(values.shape[0]))
            import hashlib
            self.input_sha256.append(hashlib.sha256(values.detach().cpu().numpy().tobytes()).hexdigest())
            return original(*args,**kwargs)
        self.scope=patch(self.vision,'forward',forward);self.scope.__enter__();return self
    def __exit__(self,*exc):return self.scope.__exit__(*exc)


class WhiteReuse:
    """Per-call candidate state; the original attribute method stays unchanged."""
    def __init__(self,backend):
        self.backend=backend;self.backgrounds=None;self.merged=None
        self.white_key=None;self.white_feature=None
        self.robust_calls=0;self.white_reuses=0;self.fallback_encodes=0

    @staticmethod
    def image_key(image):return image.mode,image.size,image.tobytes()

    @contextmanager
    def activate(self):
        backend=self.backend;color=backend.color;pattern=backend.pattern
        original_encode=pattern.encode_image
        original_robust=color.robust_garment_embedding
        original_color_encode=color.encode_image
        def robust(crop,mask,model,processor,device):
            self.robust_calls+=1
            if self.robust_calls!=1:raise RuntimeError('Candidate scope must cover exactly one attribute invocation')
            if list(color.BACKGROUND_RGB)!=[(255,255,255),(127,127,127),(0,0,0)]:
                raise RuntimeError('FROZEN_BACKGROUND_ORDER_CHANGED')
            captured=[]
            def observe(view,model,processor,device):
                embedding=original_color_encode(view,model,processor,device)
                captured.append(embedding)
                if len(captured)==1:
                    self.white_key=self.image_key(view)
                    self.white_feature=embedding
                return embedding
            # Execute the frozen color function itself, in its original order.
            # The observer retains a reference; it changes no tensor values.
            with patch(color,'encode_image',observe):
                merged=original_robust(crop,mask,model,processor,device)
            if len(captured)!=3 or any(tuple(x.shape)!=(1,512) for x in captured) or tuple(merged.shape)!=(1,512):
                raise RuntimeError('FROZEN_COLOR_FEATURE_LAYOUT_CHANGED')
            self.backgrounds=captured;self.merged=merged
            return merged
        def encode(image,model,processor,device):
            # Pixel equality is checked inside the candidate call, not assumed
            # from image size. A different local view always uses the original.
            if self.white_feature is not None and self.image_key(image)==self.white_key:
                self.white_reuses+=1
                return self.white_feature
            self.fallback_encodes+=1
            return original_encode(image,model,processor,device)
        with patch(color,'robust_garment_embedding',robust),patch(pattern,'encode_image',encode):
            yield self


def compare_outputs(left,right,atol=1e-4):
    """All keys/strings exact; all numeric prediction values within declared atol."""
    mismatches=[];numeric_differences=[]
    def visit(a,b,path):
        if isinstance(a,dict) and isinstance(b,dict):
            if set(a)!=set(b):mismatches.append({'path':path,'reason':'keys_changed'})
            for key in set(a)&set(b):visit(a[key],b[key],path+'.'+key)
        elif isinstance(a,(float,int,np.number)) and not isinstance(a,bool) and isinstance(b,(float,int,np.number)) and not isinstance(b,bool):
            delta=abs(float(a)-float(b));numeric_differences.append(delta)
            if not np.isfinite([a,b]).all() or delta>atol:
                mismatches.append({'path':path,'reason':'numeric_tolerance_exceeded','absolute_difference':delta})
        elif a!=b:mismatches.append({'path':path,'reason':'value_changed','baseline':a,'candidate':b})
    visit(left,right,'attributes')
    return {'matches_within_tolerance':not mismatches,'max_numeric_absolute_difference':max(numeric_differences,default=0.0),
            'numeric_atol':atol,'all_keys_and_text_values_exact':not any(x['reason'] in ['keys_changed','value_changed'] for x in mismatches),
            'differences':mismatches}
