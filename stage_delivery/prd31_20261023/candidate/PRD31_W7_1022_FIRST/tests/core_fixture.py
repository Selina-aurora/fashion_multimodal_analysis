"""Small CPU stand-ins for the new reusable clock scope; no CUDA execution."""
import contextlib
import types
import numpy as np


class Tensor:
    def __init__(self, values=1, device='cuda'):
        self.values=np.array(values)
        self.device=types.SimpleNamespace(type=device)
    def to(self, device=None, **kwargs):return Tensor(self.values,device or self.device.type)
    def cpu(self):return Tensor(self.values,'cpu')
    def cuda(self):return Tensor(self.values,'cuda')
    def type_as(self, other):return Tensor(self.values,other.device.type)
    def item(self):return self.values.item()
    def tolist(self):return self.values.tolist()
    def numpy(self):return self.values
    def clone(self):return Tensor(self.values.copy(),self.device.type)
    def detach(self):return self


TORCH=types.SimpleNamespace(Tensor=Tensor,equal=lambda a,b:np.array_equal(a.values,b.values),
    cuda=types.SimpleNamespace(synchronize=lambda:None),inference_mode=contextlib.nullcontext)


def run_sam(image,box,processor,model,device):
    inputs=processor(images=image,input_boxes=[[box]],return_tensors='pt')
    original_sizes=inputs['original_sizes'].clone()
    reshaped_sizes=inputs['reshaped_input_sizes'].clone()
    tensor_inputs={key:value.to(device) for key,value in inputs.items() if isinstance(value,Tensor)}
    return model(**tensor_inputs)


class Layer:
    def __init__(self):self.calls=0;self.fail=False
    def forward(self, **kwargs):
        self.calls+=1
        if self.fail:raise RuntimeError('injected model error')
        return kwargs


class Sam(Layer):
    def __init__(self):super().__init__();self.vision_encoder=Layer()
    def forward(self,**kwargs):
        self.calls+=1
        self.vision_encoder.forward(**kwargs)
        return np.ones((4,4),dtype=np.uint8)


class Processor:
    def __call__(self,images,**kwargs):
        return {'pixel_values':Tensor(np.asarray(images),'cpu')}


class Backend:
    config={'strict_region_grounding':True}
    device='cuda'
    def __init__(self):
        self.ground_model=Layer();self.sam_model=Sam()
        self.ground_processor=Processor();self.sam_processor=Processor()
    def ground(self,image,roi,region):
        def infer():
            inputs=self.ground_processor(images=image,text=region+'.',return_tensors='pt')
            self.ground_model.forward(**inputs)
            if region=='pocket':
                return {'status':'missed','mask':None,'bbox':None}
            sam_inputs=self.sam_processor(images=image,input_boxes=[[[0,0,4,4]]],return_tensors='pt')
            mask=self.sam_model.forward(**sam_inputs)
            refined=[0,0,4,4]
            return {'status':'detected','bbox':refined,'mask':mask}
        return infer(),0.


MODULE=types.SimpleNamespace(supports_region=lambda cat,region:True,
    roi_images=lambda image,roi:(image,np.ones((4,4),dtype=np.uint8),(0,0,4,4)),
    REGION_PROMPTS={'collar':'collar','pocket':'pocket'})


class Compiler:
    def __init__(self,torch,original,dynamic):
        self.original=original;self.executions=0;self.compiled_graphs=0
    def call(self,*args,**kwargs):
        self.executions+=1;self.compiled_graphs=1
        return self.original(*args,**kwargs)
    def record(self):return {'compiled_graphs':self.compiled_graphs,'compiled_graph_executions':self.executions}
