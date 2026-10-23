"""Single-query active wall clock with explicit, nested exclusions.

This is not CUDA kernel time or a continuous end-to-end wall interval.
"""
from collections import Counter,defaultdict
from contextlib import contextmanager
import ast,functools,hashlib,inspect,textwrap,time,types


class CoreClock:
    def __init__(self,sync,now=time.perf_counter):
        self.sync=sync;self.now=now;self.started=False;self.finished=False;self.depth=0
        self.active_ms=0.;self.excluded_ms=0.;self.exclusions=Counter();self.exclusion_ms=defaultdict(float)
        self.intervals=[];self.forward=Counter();self.status=None;self.local_bbox=None;self.local_mask=None

    def begin(self):
        if self.started:return
        if self.finished or self.depth:raise RuntimeError('INVALID_CORE_START')
        self.sync();self.started=True;self.t0=self.now();self.wall0=self.t0

    def stop_interval(self):
        self.sync();end=self.now();elapsed=(end-self.t0)*1000
        self.active_ms+=elapsed;self.intervals.append({'kind':'core','ms':elapsed})

    @contextmanager
    def exclude(self,reason):
        tracked=self.started and not self.finished and self.depth==0
        self.depth+=1
        if tracked:self.stop_interval();start=self.now()
        try:yield
        finally:
            self.depth-=1
            if tracked:
                self.sync();end=self.now();elapsed=(end-start)*1000
                self.excluded_ms+=elapsed;self.exclusion_ms[reason]+=elapsed;self.exclusions[reason]+=1
                self.intervals.append({'kind':'excluded','reason':reason,'ms':elapsed})
                if not self.finished:self.t0=self.now()

    def finish(self,status,bbox=None,mask=None):
        if self.finished:return
        if self.depth:raise RuntimeError('CORE_FINISH_INSIDE_EXCLUSION')
        if self.started:self.stop_interval();self.wall1=self.now()
        self.finished=True;self.status=status;self.local_bbox=bbox;self.local_mask=mask

    def record(self):
        if not self.finished:raise RuntimeError('CORE_ENDPOINT_NOT_REACHED')
        return {'core_active_wall_ms':self.active_ms if self.started else None,
            'eligible_core_sample':self.started,'status':self.status,'excluded_wall_ms':self.excluded_ms,
            'exclusions':dict(self.exclusions),'exclusion_ms':dict(self.exclusion_ms),'intervals':self.intervals,
            'forward_calls':dict(self.forward),'start_gpu_inputs_ready':True if self.started else None,
            'endpoint':'ROI-local refined bbox and binary clipped mask, before full-image canvas/metadata formatting',
            'clock_type':'sum of active synchronized wall intervals within ONE query; excludes explicit specified operations',
            'not_pure_gpu_kernel_time':True,'not_continuous_end_to_end_wall_time':True}


def compile_ground(original,clock,source=None):
    """Insert timing boundaries only, preserving every original expression.

    Source is hash-pinned by the entry point; production functions are untouched.
    """
    source=textwrap.dedent(source if source is not None else inspect.getsource(original));tree=ast.parse(source)
    outer=tree.body[0];infer=next(x for x in outer.body if isinstance(x,ast.FunctionDef) and x.name=='infer')
    modified=0
    class Boundary(ast.NodeTransformer):
        def visit_Assign(self,node):
            nonlocal modified
            if any(isinstance(t,ast.Name) and t.id=='refined' for t in node.targets):
                modified+=1
                end=ast.parse("_core_clock.finish('detected', refined, mask)").body[0]
                return [node,end]
            return node
        def visit_Return(self,node):
            if not isinstance(node.value,ast.Dict):return node
            status=None
            for key,value in zip(node.value.keys,node.value.values):
                if isinstance(key,ast.Constant) and key.value=='status' and isinstance(value,ast.Constant):status=value.value
            if status is None:raise RuntimeError('UNRECOGNIZED_GROUND_RETURN')
            # Return paths without SAM legitimately end with no emitted mask.
            text=f"_core_clock.finish({status!r}, None, mask)" if status=='empty_mask' else f"_core_clock.finish({status!r})"
            return [ast.parse(text).body[0],node]
    Boundary().visit(infer)
    if modified!=1:raise RuntimeError('PINNED_MASK_ENDPOINT_CHANGED')
    tree.body.insert(0,ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0))
    ast.fix_missing_locations(tree);clock.original_source=source;clock.timed_source=ast.unparse(tree)
    clock.boundary_sha256=hashlib.sha256(clock.timed_source.encode()).hexdigest()
    namespace=dict(original.__func__.__globals__);namespace['_core_clock']=clock
    exec(compile(tree,'<pinned_ground_timing_boundaries>','exec'),namespace)
    return types.MethodType(namespace[outer.name],original.__self__)


def compile_sam_setup(original,clock,source=None):
    """Exclude the exact original SAM input setup, including size clones.

    The DINO-derived box is still live; no saved proposal is substituted.
    """
    source=textwrap.dedent(source if source is not None else inspect.getsource(original));tree=ast.parse(source);fn=tree.body[0]
    start=1 if isinstance(fn.body[0],ast.Expr) and isinstance(fn.body[0].value,ast.Constant) else 0
    names=['inputs','original_sizes','reshaped_sizes','tensor_inputs']
    for node,name in zip(fn.body[start:start+4],names):
        if not isinstance(node,ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name) or node.targets[0].id!=name:
            raise RuntimeError('PINNED_SAM_INPUT_SETUP_CHANGED')
    if len(fn.body[start:start+4])!=4:raise RuntimeError('PINNED_SAM_INPUT_SETUP_INCOMPLETE')
    with_node=ast.parse("with _core_clock.exclude('sam_dynamic_prompt_preparation'):\n    pass").body[0]
    with_node.body=fn.body[start:start+4];fn.body[start:start+4]=[with_node]
    tree.body.insert(0,ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0));ast.fix_missing_locations(tree)
    clock.sam_original_source=source;clock.sam_timed_source=ast.unparse(tree)
    namespace=dict(original.__globals__);namespace['_core_clock']=clock
    exec(compile(tree,'<pinned_sam_input_timing_exclusion>','exec'),namespace)
    return namespace[fn.name]


class Restorer:
    def __init__(self):self.undo=[]
    def set(self,obj,name,value):
        original=getattr(obj,name);owned=name in vars(obj);setattr(obj,name,value)
        self.undo.append((obj,name,original,owned))
    def restore(self):
        for obj,name,original,owned in reversed(self.undo):
            if owned:setattr(obj,name,original)
            else:delattr(obj,name)
        self.undo.clear()


def target_device(torch,tensor,args,kwargs,method):
    if method=='cpu':return 'cpu'
    if method=='cuda':return 'cuda'
    if method=='type_as':return args[0].device.type
    value=kwargs.get('device',args[0] if args else None)
    if isinstance(value,torch.Tensor):return value.device.type
    if isinstance(value,str):
        if value.split(':')[0] in ['cpu','cuda']:return value.split(':')[0]
    if hasattr(value,'type') and str(value.type) in ['cpu','cuda']:return value.type
    return None


@contextmanager
def exclude_tensor_materialization(torch,clock,restore):
    """Explicit CPU/GPU transfers and scalar/list/NumPy materialization only.

    Same-device dtype arithmetic stays in the core clock. Wrapping is temporary
    and single-process; it is not installed in the product pipeline.
    """
    for name in ['to','cpu','cuda','type_as','item','tolist','numpy']:
        original=getattr(torch.Tensor,name)
        def make(original,name):
            @functools.wraps(original)
            def wrapped(tensor,*args,**kwargs):
                target=target_device(torch,tensor,args,kwargs,name)
                transfer=target is not None and target!=tensor.device.type
                materialize=name in ['item','tolist','numpy']
                if transfer or materialize:
                    reason='cpu_gpu_transfer' if transfer else 'tensor_materialization_or_scalar_read'
                    with clock.exclude(reason):return original(tensor,*args,**kwargs)
                return original(tensor,*args,**kwargs)
            return wrapped
        restore.set(torch.Tensor,name,make(original,name))
    # The frozen DINO helper constructs target_sizes with torch.tensor(...,
    # device='cuda') after forward. It is a host-to-device literal staging
    # operation, rather than a Tensor.to call, and must also be excluded.
    for name in ['tensor','as_tensor']:
        if not hasattr(torch,name):continue
        original=getattr(torch,name)
        def make_factory(original):
            @functools.wraps(original)
            def wrapped(data,*args,**kwargs):
                device=kwargs.get('device');target=device.split(':')[0] if isinstance(device,str) else getattr(device,'type',None)
                source=data.device.type if isinstance(data,torch.Tensor) else 'cpu'
                if target is not None and source!=target:
                    with clock.exclude('cpu_gpu_tensor_factory'):return original(data,*args,**kwargs)
                return original(data,*args,**kwargs)
            return wrapped
        restore.set(torch,name,make_factory(original))
    yield


class PreparedProcessor:
    def __init__(self,original,kind,prepared,clock,torch,key):
        self.original=original;self.kind=kind;self.prepared=prepared;self.clock=clock;self.torch=torch;self.key=key
    @staticmethod
    def image_key(image):return image.mode,image.size,image.tobytes()
    def __getattr__(self,name):return getattr(self.original,name)
    def __call__(self,*args,**kwargs):
        with self.clock.exclude('processor_dynamic_prompt_and_input_staging'):
            image=kwargs.get('images',args[0] if args else None)
            if self.image_key(image)!=self.key:raise RuntimeError('ROI_PIXELS_CHANGED_AFTER_INPUT_PREPARATION')
            if self.kind=='dino':
                if kwargs.get('text')!=self.prepared['text']:raise RuntimeError('QUERY_CHANGED_AFTER_INPUT_PREPARATION')
                return dict(self.prepared['gpu'])
            # DINO-derived box coordinates are not known at initial input
            # preparation. Preserve the original SAM processor's exact box
            # scaling. Its preprocessing/copies are explicitly excluded.
            values=dict(self.original(*args,**kwargs))
            if not self.torch.equal(values['pixel_values'],self.prepared['pixel_cpu']):raise RuntimeError('SAM_PREPROCESSED_PIXELS_CHANGED')
            values['pixel_values']=self.prepared['pixel_gpu']
            return values


def prepare_inputs(backend,module,image,roi,region,torch):
    if not backend.config.get('strict_region_grounding'):raise RuntimeError('FROZEN_STRICT_CONFIG_REQUIRED')
    if not module.supports_region(roi.category,region):return None
    crop,mask,origin=module.roi_images(image,roi);text=module.REGION_PROMPTS[region].strip()
    if not text.endswith('.'):text+='.'
    raw=dict(backend.ground_processor(images=crop,text=text,return_tensors='pt'))
    gpu={k:v.to(backend.device) if isinstance(v,torch.Tensor) else v for k,v in raw.items()}
    sam=dict(backend.sam_processor(images=crop,return_tensors='pt'))
    if 'pixel_values' not in sam:raise RuntimeError('SAM_IMAGE_INPUT_LAYOUT_CHANGED')
    pixel_cpu=sam['pixel_values'];pixel_gpu=pixel_cpu.to(backend.device)
    if any(v.device.type!='cuda' for v in gpu.values() if isinstance(v,torch.Tensor)) or pixel_gpu.device.type!='cuda':
        raise RuntimeError('INPUTS_NOT_GPU_RESIDENT')
    torch.cuda.synchronize()
    return {'key':PreparedProcessor.image_key(crop),'crop_size':crop.size,'origin':origin,'dino':{'gpu':gpu,'text':text},
            'sam':{'pixel_cpu':pixel_cpu,'pixel_gpu':pixel_gpu}}


@contextmanager
def core_scope(backend,prepared,clock,torch,sam_module=None):
    restore=Restorer()
    try:
        if sam_module is not None:restore.set(sam_module,'run_sam',compile_sam_setup(sam_module.run_sam,clock))
        if prepared is not None:
            restore.set(backend,'ground_processor',PreparedProcessor(backend.ground_processor,'dino',prepared['dino'],clock,torch,prepared['key']))
            restore.set(backend,'sam_processor',PreparedProcessor(backend.sam_processor,'sam',prepared['sam'],clock,torch,prepared['key']))
        original=backend.ground_model.forward
        def dino(*args,**kwargs):
            clock.begin();clock.forward['dino']+=1
            return original(*args,**kwargs)
        restore.set(backend.ground_model,'forward',dino)
        for name,obj in [('sam',backend.sam_model),('sam_vision',backend.sam_model.vision_encoder)]:
            fn=obj.forward
            def make(fn,name):
                def forwarded(*a,**kw):clock.forward[name]+=1;return fn(*a,**kw)
                return forwarded
            restore.set(obj,'forward',make(fn,name))
        with exclude_tensor_materialization(torch,clock,restore):yield
    finally:restore.restore()
