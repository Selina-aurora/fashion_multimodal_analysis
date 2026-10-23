"""Candidate single-image entry, independent of experiment/reference folders.

Only a sequential process adapter enables verified white embedding reuse.
No production file installation or modification is performed.
"""
import argparse,inspect,os,platform,sys,time,traceback,zipfile
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
from run_attributes import read,write,verify,sha,HERE,SIX,CHECKPOINT_SHA
from e2e_logic import FullBackend,canonical


def load_backend(project,out):
    verify(HERE,read(HERE/'PACKAGE_SHA256.json'))
    for name in ['source_hashes.json','FROZEN_ATTRIBUTE_SOURCES.json','GROUND_HELPER_SHA256.json']:verify(project,read(HERE/'inputs'/name))
    cfg=read(HERE/'inputs/config.json')
    if sha(project/cfg['segmentation_checkpoint'])!=CHECKPOINT_SHA:raise RuntimeError('FROZEN_CHECKPOINT_CHANGED')
    os.environ.pop('PRD31_TIMING_LOG',None)
    os.environ.update(HF_HOME=str(project/'hf_cache'),HF_HUB_CACHE=str(project/'hf_cache/hub'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    os.chdir(project);sys.path.insert(0,str(project/'src'))
    from cache_probe import discover,verify_processors
    resolved,audit=discover(cfg);write(out/'model_cache_audit.json',audit)
    if not audit['all_model_assets_found']:raise RuntimeError('OFFLINE_MODEL_ASSET_MISSING')
    processors=verify_processors(resolved);write(out/'processor_audit.json',processors)
    if any(v['processor_load']!='ok' for v in processors.values()):raise RuntimeError('PROCESSOR_LOAD_FAILED')
    cfg.update(resolved)
    import torch
    from fashion_multimodal_analysis.integration.model_backends import ModelBackends
    from fashion_multimodal_analysis.integration.pipeline import FashionPipeline
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
    torch.manual_seed(20261021);backend=ModelBackends(cfg,device='cuda')
    for name,expected in read(HERE/'inputs/TRANSFORMERS_SOURCE_SHA256.json').items():
        source=Path(inspect.getsourcefile(type(getattr(backend,name))))
        if sha(source)!=expected:raise RuntimeError('TRANSFORMERS_IMPLEMENTATION_CHANGED: '+name)
    write(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(),'config':cfg,'candidate_version':read(HERE/'CANDIDATE_VERSION.json'),
        'production_modified':False,'temporary_white_reuse_adapter_enabled':True})
    return backend,FashionPipeline,torch


def run_image(backend,Pipeline,torch,image_path,out):
    source_sha=sha(image_path);torch.cuda.synchronize();start=time.perf_counter()
    with Image.open(image_path) as source:image=source.convert('RGB')
    with torch.inference_mode():result=Pipeline(FullBackend(backend,True)).run(image,backend.config['required_regions'],SIX)
    torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
    semantic=canonical({k:v for k,v in result.items() if k!='timings'},image.size)
    masks=out/'masks';masks.mkdir();inventory=[]
    for n,item in enumerate(result.get('instances',[])):
        values=[('garment',item['roi'].mask)]+[(q,p.get('mask')) for q,p in item['regions'].items()]
        for k,(name,mask) in enumerate(values):
            if mask is None:continue
            filename=f'{n:04d}_{k:02d}.png';Image.fromarray((np.asarray(mask)>0).astype('uint8')*255).save(masks/filename)
            inventory.append({'instance_id':item['roi'].instance_id,'region':name,'file':'masks/'+filename})
    write(out/'prediction.json',{'source_image':str(image_path),'source_image_sha256':source_sha,'source_size':list(image.size),
        'output':semantic,'mask_files':inventory})
    write(out/'runtime_observation.json',{'full_image_operational_wall_ms':elapsed,'model_initialization_excluded':True,
        'decode_segmentation_grounding_attributes_review_included':True,'mask_hashes_and_export_excluded':True,
        'formal_core_latency':False,'accuracy_claim':False,'module_diagnostic_timings':result.get('timings',[])})
    return result.get('status')=='completed' and not result.get('failures')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',default='/workspace/fashion_multimodal_analysis')
    ap.add_argument('--image',required=True);ap.add_argument('--output-root',default='/workspace/prd31_candidate_outputs');a=ap.parse_args()
    project=Path(a.project).resolve();image=Path(a.image).resolve();root=Path(a.output_root).resolve();root.mkdir(parents=True,exist_ok=True)
    out=root/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir();code=1
    try:
        backend,Pipeline,torch=load_backend(project,out);passed=run_image(backend,Pipeline,torch,image,out)
        for name in ['source_hashes.json','FROZEN_ATTRIBUTE_SOURCES.json','GROUND_HELPER_SHA256.json']:verify(project,read(HERE/'inputs'/name))
        write(out/'summary.json',{'status':'CANDIDATE_IMAGE_COMPLETE' if passed else 'PIPELINE_FAILURES','candidate_version':read(HERE/'CANDIDATE_VERSION.json'),
            'production_sources_unchanged':True,'formal_core_latency_measured':False,'accuracy_claim':False})
        code=0 if passed else 1
    except BaseException:
        error=traceback.format_exc();print(error,flush=True);write(out/'fatal_error.json',{'error':error})
        write(out/'summary.json',{'status':'BLOCKED_OR_PARTIAL','production_promoted':False})
    finally:
        archive=out.with_name(out.name+'_UPLOAD_THIS.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(out))
        print('UPLOAD_THIS: '+str(archive),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
