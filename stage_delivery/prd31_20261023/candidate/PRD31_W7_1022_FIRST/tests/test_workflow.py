import contextlib
import copy
from dataclasses import dataclass
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import core_compile_probe as probe
from collect_new_regression import choose_new, dhash, manifest_paths
from run_candidate import infer_image
from run_new_regression import run as new_regression
from run_attributes import SIX, sha
from run_1022 import child
import core_fixture


class PixelTensor:
    def __init__(self,a,device='cuda:0'):self.a=np.array(a,copy=True);self.device=device
    @property
    def shape(self):return self.a.shape
    @property
    def dtype(self):return self.a.dtype
    def stride(self):return tuple(x//self.a.itemsize for x in self.a.strides)
    def detach(self):return self
    def clone(self):return PixelTensor(self.a,self.device)
    def cpu(self):return PixelTensor(self.a,'cpu')
    def contiguous(self):return self
    def numpy(self):return self.a


TORCH=types.SimpleNamespace(is_tensor=lambda x:isinstance(x,PixelTensor),
    equal=lambda a,b:np.array_equal(a.a,b.a),cuda=types.SimpleNamespace(synchronize=lambda:None),
    inference_mode=contextlib.nullcontext)


@dataclass
class ROI:
    instance_id:str
    category:str='top'
    score:float=1.
    bbox:tuple=(0.,0.,4.,4.)
    mask:object=None
    provenance:str='segmentation_prediction'


class Vision:
    training=False
    def __init__(self):self.calls=0
    def forward(self,pixel_values):
        self.calls+=1
        return {'last_hidden_state':PixelTensor(pixel_values.a*2)}


class Model:
    def __init__(self):self.vision_encoder=Vision()
    def forward(self,pixels=None):
        return self.vision_encoder.forward(pixels) if pixels is not None else None


class ImageBackend:
    config={'required_regions':['collar','cuff']}
    def __init__(self):self.ground_model=Model();self.sam_model=Model()
    def ground(self,image,roi,region):
        self.ground_model.forward()
        self.sam_model.forward(PixelTensor(np.asarray(image,dtype=np.float32)))
        return {'status':'detected','bbox':[0.,0.,4.,4.],'mask':np.ones((4,4),dtype=np.uint8)},0.


class Pipeline:
    def __init__(self,backend):self.backend=backend
    def run(self,image,queries,attrs):
        instances=[]
        for i in range(2):
            roi=ROI('pred_'+str(i),mask=np.ones((4,4),dtype=np.uint8))
            regions={q:self.backend.ground(image,roi,q)[0] for q in queries}
            instances.append({'roi':roi,'regions':regions,'attributes':{a:{'status':'predicted'} for a in attrs}})
        return {'status':'completed','instances':instances,'timings':[]}


class WorkflowTests(unittest.TestCase):
    def test_entry_uses_verified_sam_path_and_exports_linked_masks(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'img.png';Image.new('RGB',(4,4),'red').save(source)
            outputs=[]
            for mode in ['white','white-sam']:
                out=root/mode;out.mkdir()
                result=infer_image(ImageBackend(),Pipeline,TORCH,source,out,mode)
                outputs.append(result['output'])
                self.assertEqual(len(result['mask_files']),6)
                stats=json.loads((out/'runtime_observation.json').read_text())['sam_cache']
                self.assertEqual(stats['actual_vision_forwards'],4 if mode=='white' else 2)
                self.assertEqual(stats['cache_hits'],0 if mode=='white' else 2)
                self.assertEqual(set(result['output']['instances'][0]['attributes']),set(SIX))
            self.assertEqual(outputs[0],outputs[1])

    def test_sampler_never_opens_sealed_names_and_filters_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pool=root/'pool';pool.mkdir();out=root/'out';(out/'previews').mkdir(parents=True)
            random=np.random.default_rng(17)
            Image.fromarray(random.integers(0,255,(32,32,3),dtype=np.uint8)).save(pool/'known.png')
            # Invalid image bytes prove sealed images must be skipped before decode.
            (pool/'sealed.png').write_bytes(b'never read this image')
            (pool/'duplicate.png').write_bytes((pool/'known.png').read_bytes())
            for i in range(12):Image.fromarray(random.integers(0,255,(32,32,3),dtype=np.uint8)).save(pool/f'new_{i}.png')
            forbidden={sha(pool/'known.png')}
            with Image.open(pool/'known.png') as im:known=[dhash(im)]
            original=Image.open
            def guarded(path,*a,**kw):
                self.assertNotEqual(Path(path).name,'sealed.png')
                return original(path,*a,**kw)
            with patch('PIL.Image.open',guarded):
                selected,audit=choose_new([pool],{'sealed.png','known.png'},set(),forbidden,known,6,out)
            self.assertEqual(len(selected),6)
            self.assertFalse(any(Path(r['path']).name in {'known.png','sealed.png','duplicate.png'} for r in selected))
            self.assertTrue(all(r['scene_review']=='pending' for r in selected))
            self.assertEqual(len({r['sha256'] for r in selected}),6)

    def test_manifest_path_resolution_and_unknown_schema_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);file=root/'train.csv';file.write_text('source_image,record_id\nimgs/a.jpg,1\n')
            self.assertEqual(manifest_paths(root,file),[(root/'imgs/a.jpg').resolve()])
            file.write_text('record_id,category\n1,top\n')
            with self.assertRaisesRegex(RuntimeError,'UNRESOLVED'):manifest_paths(root,file)

    def test_query_selection_reproducible_stratified_and_excludes_not_applicable(self):
        jobs=[];ref={}
        for region in ['a','b','c','d']:
            for status in ['detected','empty_mask','missed','rejected_region_geometry']:
                for i in range(2):
                    key=(f'{status}{i}','roi',region)
                    jobs.append({'key':key,'region':region})
                    ref[key]={'status':status}
        key=('not_applicable','roi','a');jobs.append({'key':key,'region':'a'});ref[key]={'status':'not_applicable'}
        selected=probe.select_queries(jobs,ref,16)
        self.assertEqual(selected,probe.select_queries(list(reversed(jobs)),ref,16))
        self.assertEqual(len({(j['region'],ref[j['key']]['status']) for j in selected}),16)

    def core_scope(self):
        backend=core_fixture.Backend()
        originals=(backend.ground_processor,backend.sam_processor,backend.ground_model.forward,
            backend.sam_model.forward,backend.sam_model.vision_encoder.forward,core_fixture.Tensor.to)
        module=types.SimpleNamespace(run_sam=core_fixture.run_sam)
        runtime=patch.object(probe.importlib,'import_module',return_value=module)
        return backend,originals,runtime

    def test_paired_core_wrappers_stable_inputs_reprepared_and_models_recomputed(self):
        backend,originals,context=self.core_scope()
        image=Image.new('RGB',(4,4),'green')
        roi=types.SimpleNamespace(category='top')
        with context:
            with probe.PairedCore(backend,core_fixture.MODULE,core_fixture.TORCH,core_fixture.Compiler) as runner:
                stable=id(core_fixture.Tensor.to)
                for region in ['collar','pocket']:
                    results=[]
                    for mode in (False,True,False,True):
                        pred,row=runner.invoke({'roi':roi,'region':region},image,mode)
                        results.append(pred)
                        self.assertEqual(id(core_fixture.Tensor.to),stable)
                        self.assertEqual(row['forward_calls']['dino'],1)
                        self.assertEqual(row['compiled_dino_graph_executions'],int(mode))
                    self.assertTrue(all(p==results[0] for p in results))
                self.assertEqual(backend.ground_model.calls,8)
                self.assertEqual(backend.sam_model.vision_encoder.calls,4)
        restored=(backend.ground_processor,backend.sam_processor,backend.ground_model.forward,
            backend.sam_model.forward,backend.sam_model.vision_encoder.forward,core_fixture.Tensor.to)
        self.assertEqual(restored,originals)

    def test_paired_core_failure_restores_every_method(self):
        backend,originals,context=self.core_scope();backend.ground_model.fail=True
        with context,self.assertRaisesRegex(RuntimeError,'injected'):
            with probe.PairedCore(backend,core_fixture.MODULE,core_fixture.TORCH,core_fixture.Compiler) as runner:
                runner.invoke({'roi':types.SimpleNamespace(category='top'),'region':'collar'},Image.new('RGB',(4,4)),True)
        restored=(backend.ground_processor,backend.sam_processor,backend.ground_model.forward,
            backend.sam_model.forward,backend.sam_model.vision_encoder.forward,core_fixture.Tensor.to)
        self.assertEqual(restored,originals)

    def test_compiler_fallback_without_actual_graph_execution_is_rejected(self):
        class NoGraph(core_fixture.Compiler):
            def call(self,*args,**kwargs):return self.original(*args,**kwargs)
        backend,originals,context=self.core_scope()
        with context,self.assertRaisesRegex(RuntimeError,'COMPILED_DINO_WORK_NOT_OBSERVED'):
            with probe.PairedCore(backend,core_fixture.MODULE,core_fixture.TORCH,NoGraph) as runner:
                runner.invoke({'roi':types.SimpleNamespace(category='top'),'region':'collar'},Image.new('RGB',(4,4)),True)

    def test_compile_benchmark_counts_both_modes_and_stops_on_output_drift(self):
        jobs=[{'key':('img','roi',region),'image_id':'img','instance_id':'roi','region':region} for region in ['collar','pocket']]
        reference={j['key']:{'status':'missed','score':.5} for j in jobs}
        class Runner:
            def __init__(self,drift=False):
                self.dino=self.vision=types.SimpleNamespace(record=lambda:{'compiled_graphs':1})
                self.calls=0;self.drift=drift
            def invoke(self,job,image,candidate):
                self.calls+=1
                pred=copy.deepcopy(reference[job['key']])
                if self.drift and self.calls>24:pred['score']=.7
                return pred,{'core_active_wall_ms':1. if candidate else 2.,'new_compiled_graphs_during_call':0}
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d)
            result=probe.benchmark(Runner(),jobs,{'img':None},reference,root,passes=2)
            self.assertTrue(result['all_output_checks_pass'])
            self.assertEqual(result['stages']['compiled']['n'],4)
            self.assertEqual(result['paired_queries'],4)
            self.assertFalse(result['full_527_query_coverage'])
            self.assertTrue(result['do_not_compare_screen_mean_directly_to_historical_139_63ms'])
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'TIMED_OUTPUT_DRIFT'):
                probe.benchmark(Runner(True),jobs,{'img':None},reference,Path(d),passes=2)
            self.assertFalse(json.loads((Path(d)/'timed_checks.json').read_text())[-1]['passed'])

    def test_tracked_backend_records_execution_and_preserves_boxed_call_marker(self):
        captured={}
        def fake_inductor(graph,inputs,**kwargs):
            graph._boxed_call=True
            return graph
        def compile_fn(original,**kwargs):
            captured.update(kwargs)
            return kwargs['backend'](original,[])
        fake_torch=types.SimpleNamespace(compile=compile_fn)
        registry=types.SimpleNamespace(lookup_backend=lambda name:fake_inductor)
        with patch.dict(sys.modules,{'torch._dynamo.backends.registry':registry}):
            tracked=probe.TrackedCompile(fake_torch,lambda x:x*2,True)
            self.assertEqual(tracked.executions,0)
            self.assertEqual(tracked.call(3),6)
            self.assertEqual(tracked.executions,1)
            self.assertTrue(tracked.call._boxed_call)
            self.assertTrue(captured['dynamic']);self.assertFalse(captured['fullgraph'])

    def test_new_image_regression_is_one_pass_not_accuracy(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);out=root/'out';out.mkdir();source=root/'new.png';Image.new('RGB',(4,4),'blue').save(source)
            manifest=root/'images.json';manifest.write_text(json.dumps({'images':[{'image_id':'new','path':str(source),'sha256':sha(source)}]}))
            result=new_regression(ImageBackend(),Pipeline,TORCH,manifest,out)
            self.assertTrue(result['all_output_checks_pass'])
            self.assertFalse(result['accuracy_claim']);self.assertFalse(result['formal_performance_benchmark'])
            self.assertEqual(result['paired_passes'],1)

    def test_child_timeout_kills_process_and_preserves_log(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            folder=Path(d)/'child'
            result=child([sys.executable,'-u','-c','import time; print("started",flush=True); time.sleep(30)'],folder,.15)
            self.assertEqual(result['status'],'STAGE_TIMEOUT_OR_INTERRUPTION')
            self.assertEqual(result['process_exit_code'],124)
            self.assertIn('started',folder.with_suffix('.log').read_text())


if __name__=='__main__':unittest.main()
