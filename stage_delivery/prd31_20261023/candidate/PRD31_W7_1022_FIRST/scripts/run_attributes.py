"""Fixed24 cost diagnosis and collar -> six-attribute interface smoke run.

No training, manual labels, alternative ROI inputs or production file edits.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import zipfile
import numpy as np
from PIL import Image
from bench_logic import attribute_signature,aggregate
from profile_logic import SpanProfiler,summarize_spans,check_collar_geometry

HERE=Path(__file__).resolve().parents[1]
FIVE=['primary_color','pattern','sleeve_length','silhouette_fit','fashion_style']
SIX=['primary_color','pattern','sleeve_length','neckline','silhouette_fit','fashion_style']
CHECKPOINT_SHA='c5ba795225349b1c521aac0517ac7c493d9818dcc17d558ac4a3f6f8669452f0'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    def scalar(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return x.tolist()
        if hasattr(x,'detach'):return x.detach().cpu().tolist()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=scalar),encoding='utf-8')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify(root,expected):
    for rel,digest in expected.items():
        if sha(root/rel)!=digest:raise RuntimeError('FROZEN_FILE_CHANGED: '+str(root/rel))


def timed(sync,fn):
    sync();start=time.perf_counter();result=fn();sync()
    return result,(time.perf_counter()-start)*1000


def profile_stage(backend,module,jobs,images,reference,out,sync):
    """Pairs ordinary and instrumented calls; alternating order across passes."""
    timings=[];comparisons=[];events=[];errors=[];outputs=[]
    for n in range(10):
        rid,roi,index=jobs[n%len(jobs)]
        backend.attributes(images[rid],roi,{},FIVE,controlled=False)
    sync()
    for p in range(5):
        for rid,roi,index in jobs:
            meta={'pass':p,'image_id':rid,'instance_id':roi.instance_id,'feature_index':index}
            results={}
            for instrument in ([False,True] if p%2==0 else [True,False]):
                mode='instrumented' if instrument else 'ordinary'
                try:
                    def call():return backend.attributes(images[rid],roi,{},FIVE,controlled=False)
                    if instrument:
                        profiler=SpanProfiler(sync)
                        try:
                            with profiler.instrument(backend,module):
                                (pred,internal),elapsed=timed(sync,lambda:profiler.call('attribute_bundle',call))
                        finally:
                            events.extend({**meta,**e} for e in profiler.events)
                    else:(pred,internal),elapsed=timed(sync,call)
                    if set(pred)!=set(FIVE):raise RuntimeError('FIVE_ATTRIBUTE_KEYS_CHANGED')
                    results[mode]=pred
                    outputs.append({**meta,'mode':mode,'attributes':pred})
                    timings.append({**meta,'stage':mode+'_five_attribute_bundle','ms':elapsed})
                except Exception:
                    errors.append({**meta,'mode':mode,'error':traceback.format_exc()})
            ordinary=results.get('ordinary');instrumented=results.get('instrumented')
            comparisons.append({**meta,
                'ordinary_matches_reference':ordinary is not None and attribute_signature(ordinary)==attribute_signature(reference[(rid,roi.instance_id)]),
                'instrumented_matches_ordinary':ordinary is not None and instrumented is not None and attribute_signature(ordinary)==attribute_signature(instrumented)})
        write(out/'profile_events.json',events);write(out/'profile_timings.json',timings)
        write(out/'profile_comparisons.json',comparisons);write(out/'profile_predictions.json',outputs)
        write(out/'profile_errors.json',errors)
        print(f'ATTRIBUTE PROFILE {p+1}/5: ordinary + instrumented paired',flush=True)
    summary=summarize_spans(events)
    summary.update({'latency':aggregate(timings),'expected_ordinary_calls':185,'expected_instrumented_calls':185,
        'actual_ordinary_calls':sum(r['stage'].startswith('ordinary_') for r in timings),
        'actual_instrumented_calls':sum(r['stage'].startswith('instrumented_') for r in timings),
        'output_comparisons':len(comparisons),
        'all_status_and_labels_match':all(r['ordinary_matches_reference'] and r['instrumented_matches_ordinary'] for r in comparisons),
        'error_count':len(errors),'warmup_calls':10,'timed_passes':5,
        'paired_order_alternates_by_pass':True,'no_speed_optimization_applied':True,
        'note':'Inclusive spans overlap. Exclusive costs include instrumentation overhead; use for diagnosis, not PRD latency or speedup.'})
    summary['complete']=not errors and summary['actual_ordinary_calls']==185 and summary['actual_instrumented_calls']==185 and len(comparisons)==185
    write(out/'profile_summary.json',summary)
    return summary


def interface_stage(backend,Pipeline,jobs,images,reference,out,sync):
    """One downstream smoke pass; no new segmentation, no formal timing claim."""
    grouped={rid:[] for rid in images}
    for rid,roi,index in jobs:grouped[rid].append((roi,index))
    records=[];errors=[];pipeline_failures=[];image_timings=[];raw_by_key={}
    original=backend.attributes;owned='attributes' in vars(backend);active={'image_id':None}
    def capture(image,roi,regions,names,controlled=False):
        if controlled:raise RuntimeError('CONTROLLED_GT_PATH_FORBIDDEN')
        result,ms=original(image,roi,regions,names,controlled=False)
        raw_by_key[(active['image_id'],roi.instance_id)]=copy.deepcopy(result)
        return result,ms
    backend.attributes=capture
    masks=out/'collar_masks';masks.mkdir(exist_ok=True)
    try:
        pipe=Pipeline(backend)
        for n,(rid,pairs) in enumerate(grouped.items(),1):
            active['image_id']=rid;indices={roi.instance_id:index for roi,index in pairs}
            try:
                result,ms=timed(sync,lambda:pipe.run_downstream(images[rid],[roi for roi,index in pairs],['collar'],SIX))
                image_timings.append({'image_id':rid,'ms':ms,'scope':'single downstream smoke call, no warmup; not formal latency'})
                pipeline_failures.extend({'image_id':rid,**f} for f in result.get('failures',[]))
                observed=set()
                for item in result['instances']:
                    roi=item['roi'];iid=roi.instance_id
                    if iid not in indices or iid in observed:raise RuntimeError('PIPELINE_INSTANCE_LINKAGE_CHANGED')
                    observed.add(iid)
                    collar=item['regions'].get('collar',{})
                    geometry=check_collar_geometry(collar,images[rid].size)
                    raw=raw_by_key.get((rid,iid),{})
                    five={name:raw[name] for name in FIVE if name in raw}
                    neck=raw.get('neckline',{})
                    invariant=(neck.get('status')=='localization_missing') if collar.get('status')!='detected' else neck.get('status') not in ('localization_missing','attribute_failed',None)
                    exported=copy.copy(collar);mask=exported.pop('mask',None)
                    if mask is not None:
                        array=np.asarray(mask)
                        name=f'collar_masks/{rid}_{iid}.png'
                        if array.shape!=(images[rid].height,images[rid].width):raise RuntimeError('COLLAR_MASK_DIMENSIONS_CHANGED')
                        Image.fromarray((array>0).astype('uint8')*255).save(out/name)
                        exported['mask_file']=name;exported['mask_shape']=list(array.shape)
                    records.append({'image_id':rid,'instance_id':iid,'feature_index':indices[iid],
                        'predicted_category':roi.category,'garment_bbox':roi.bbox,'collar':exported,
                        'collar_geometry_check':geometry,'raw_attributes_before_semantic_review':raw,
                        'delivered_attributes':item['attributes'],'review_flags':item['review_flags'],
                        'five_raw_status_and_labels_match_prior':len(five)==5 and attribute_signature(five)==attribute_signature(reference[(rid,iid)]),
                        'neckline_localization_invariant':invariant,
                        'six_attribute_keys_valid':set(raw)==set(SIX) and set(item['attributes'])==set(SIX)})
                if observed!=set(indices):raise RuntimeError('PIPELINE_MISSING_INSTANCE')
            except Exception:
                errors.append({'image_id':rid,'error':traceback.format_exc()})
            write(out/'interface_predictions.json',records);write(out/'interface_errors.json',errors)
            write(out/'interface_pipeline_failures.json',pipeline_failures)
            write(out/'interface_image_diagnostic_timings.json',image_timings)
            print(f'COLLAR -> SIX ATTRIBUTES {n}/24 {rid}',flush=True)
    finally:
        if owned:backend.attributes=original
        else:delattr(backend,'attributes')
    summary={'expected_instances':37,'observed_instances':len(records),'expected_images':24,'completed_images':len(image_timings),
        'collar_statuses':dict(Counter(r['collar'].get('status','missing') for r in records)),
        'raw_neckline_statuses':dict(Counter(r['raw_attributes_before_semantic_review'].get('neckline',{}).get('status','missing') for r in records)),
        'delivered_neckline_statuses':dict(Counter(r['delivered_attributes'].get('neckline',{}).get('status','missing') for r in records)),
        'detected_collar_geometry_failures':sum(r['collar_geometry_check']['detected'] and not r['collar_geometry_check']['geometry_valid'] for r in records),
        'neckline_localization_invariant_failures':sum(not r['neckline_localization_invariant'] for r in records),
        'five_raw_attribute_drift_count':sum(not r['five_raw_status_and_labels_match_prior'] for r in records),
        'six_attribute_key_failures':sum(not r['six_attribute_keys_valid'] for r in records),
        'pipeline_failure_count':len(pipeline_failures),'error_count':len(errors),
        'scope':'37 cached predicted upper-body garment instances on 24 developer images; collar query only; six attributes; one smoke pass',
        'formal_latency':False,'accuracy':None,'grounding_IoU':None,'end_to_end_full_chain':False,
        'manual_fallback':False,'new_input_variants':False,'semantic_review_can_change_delivered_status':True}
    summary['complete']=len(records)==37 and len(image_timings)==24 and not errors and not pipeline_failures
    summary['interface_checks_pass']=summary['complete'] and not any(summary[k] for k in ['detected_collar_geometry_failures','neckline_localization_invariant_failures','five_raw_attribute_drift_count','six_attribute_key_failures'])
    write(out/'interface_summary.json',summary)
    return summary


def prepare(args,out):
    project=Path(args.project).resolve();prior=Path(args.prior).resolve();bench=Path(args.benchmark).resolve()
    verify(HERE,read(HERE/'PACKAGE_SHA256.json'))
    verify(prior,read(HERE/'inputs/PRIOR_FILES_SHA256.json'))
    verify(bench,read(HERE/'inputs/REFERENCE_BENCH_SHA256.json'))
    verify(project,read(HERE/'inputs/source_hashes.json'))
    cfg=read(HERE/'inputs/config.json')
    if sha(project/cfg['segmentation_checkpoint'])!=CHECKPOINT_SHA:raise RuntimeError('CHECKPOINT_CHANGED')
    rows=read(prior/'images_inference_only.json');segs=read(prior/'segmentation_all_predictions.json')
    if len(rows)!=24 or len({r['sha256'] for r in rows})!=24:raise RuntimeError('FIXED24_CHANGED')
    images={}
    for r in rows:
        if sha(Path(r['path']))!=r['sha256']:raise RuntimeError('SOURCE_IMAGE_CHANGED')
        with Image.open(r['path']) as im:images[r['image_id']]=im.convert('RGB')
    index=read(prior/'feature_index.json')
    ref_rows=[r for r in read(bench/'attribute_predictions.json') if r['pass']==0]
    reference={(r['image_id'],r['instance_id']):r['attributes'] for r in ref_rows}
    if len(reference)!=37:raise RuntimeError('REFERENCE37_CHANGED')
    os.environ.pop('PRD31_TIMING_LOG',None)
    os.environ.update(HF_HOME=str(project/'hf_cache'),HF_HUB_CACHE=str(project/'hf_cache/hub'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    os.chdir(project);sys.path.insert(0,str(project/'src'))
    from cache_probe import discover,verify_processors
    resolved,audit=discover(cfg);write(out/'model_cache_audit.json',audit)
    if not audit['all_model_assets_found']:raise RuntimeError('OFFLINE_ASSET_MISSING')
    processors=verify_processors(resolved);write(out/'processor_audit.json',processors)
    if any(r['processor_load']!='ok' for r in processors.values()):raise RuntimeError('PROCESSOR_PREFLIGHT_FAILED')
    cfg.update(resolved)
    import torch
    import fashion_multimodal_analysis.integration.model_backends as module
    from fashion_multimodal_analysis.integration.pipeline import GarmentROI,FashionPipeline
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
    torch.manual_seed(20261021);backend=module.ModelBackends(cfg,device='cuda')
    if sha(Path(backend.pattern.__file__))!=read(bench/'environment.json')['clip_feature_module_sha256']:raise RuntimeError('FEATURE_MODULE_CHANGED')
    jobs=[]
    for record in segs:
        rid=record['image_id']
        for inst in record['instances']:
            if inst.get('feature_index') is None:continue
            k=inst['feature_index'];mapping=index[k]
            if (mapping['record_id'],mapping['predicted_instance_id'],mapping['feature_index'])!=(rid,inst['instance_id'],k):raise RuntimeError('FEATURE_LINKAGE_CHANGED')
            with Image.open(prior/inst['mask_file']) as im:mask=(np.asarray(im)>0).astype('uint8')
            if mask.shape!=(images[rid].height,images[rid].width):raise RuntimeError('MASK_DIMENSIONS_CHANGED')
            jobs.append((rid,GarmentROI(instance_id=inst['instance_id'],category=inst['category'],score=inst['score'],bbox=inst['bbox'],mask=mask),k))
    if len(jobs)!=37 or {(rid,roi.instance_id) for rid,roi,k in jobs}!=set(reference):raise RuntimeError('JOB37_CHANGED')
    dependencies={}
    for value in [module,backend.color,backend.pattern,backend.design]:
        path=Path(value.__file__);dependencies[str(path.relative_to(project))]=sha(path)
    for rel in ['src/fashion_multimodal_analysis/integration/attribute_context.py','src/fashion_multimodal_analysis/integration/semantic_consistency.py']:
        if (project/rel).is_file():dependencies[rel]=sha(project/rel)
    write(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(),'config':cfg,'source_files':dependencies,'checkpoint_sha256':CHECKPOINT_SHA,
        'profiling_changes':'temporary in-process wrappers restored after each call; no source file changes',
        'production_promotion':False,'image_count':24,'cached_upper_garment_instances':37})
    return project,prior,bench,backend,module,FashionPipeline,jobs,images,reference,torch


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',default='/workspace/fashion_multimodal_analysis')
    ap.add_argument('--prior',default='/workspace/prd31_week7_1021_batch/run_20261021_033050_887414')
    ap.add_argument('--benchmark',default='/workspace/prd31_week7_1021_benchmark/run_20261021_040120_362409')
    ap.add_argument('--output-root',default='/workspace/prd31_week7_1021_attributes')
    args=ap.parse_args();root=Path(args.output_root).resolve();root.mkdir(parents=True,exist_ok=True)
    out=root/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    code=1;stages={};stage_errors={}
    try:
        project,prior,bench,backend,module,Pipeline,jobs,images,reference,torch=prepare(args,out)
        with torch.inference_mode():
            for name,fn in [('profile',lambda:profile_stage(backend,module,jobs,images,reference,out,torch.cuda.synchronize)),
                            ('interface',lambda:interface_stage(backend,Pipeline,jobs,images,reference,out,torch.cuda.synchronize))]:
                try:stages[name]=fn()
                except Exception:
                    stage_errors[name]=traceback.format_exc();print(stage_errors[name],flush=True)
                write(out/'stage_errors.json',stage_errors)
        verify(prior,read(HERE/'inputs/PRIOR_FILES_SHA256.json'))
        verify(bench,read(HERE/'inputs/REFERENCE_BENCH_SHA256.json'))
        verify(project,read(HERE/'inputs/source_hashes.json'))
        verify(project,read(out/'environment.json')['source_files'])
        complete=not stage_errors and all(stages.get(s,{}).get('complete',False) for s in ['profile','interface'])
        checks=complete and stages['profile']['all_status_and_labels_match'] and stages['interface']['interface_checks_pass']
        write(out/'summary.json',{'status':'AUTOMATIC_CHECKS_COMPLETE' if complete else 'PARTIAL_OR_BLOCKED',
            'all_interface_and_consistency_checks_pass':checks,'stages':stages,'stage_errors':stage_errors,
            'prior_outputs_unchanged':True,'production_sources_unchanged':True,'training_done':False,
            'new_sleeve_head_integrated':False,'human_labels_created':0,'accuracy_claim':False,
            'segmentation_pose_or_geometry_rerun':False,'end_to_end_full_chain':False})
        code=0 if checks else 1
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
        write(out/'fatal_error.json',{'error':error})
        write(out/'summary.json',{'status':'PARTIAL_OR_BLOCKED','stages':stages,'error':error,'accuracy_claim':False})
    finally:
        archive=out.with_name(out.name+'_UPLOAD_THIS.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(out))
        print('UPLOAD_THIS: '+str(archive),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
