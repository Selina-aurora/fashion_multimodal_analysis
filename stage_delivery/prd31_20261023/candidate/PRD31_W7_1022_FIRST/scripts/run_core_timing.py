"""Frozen ground code, explicit per-query core clock, fixed GPU-ready inputs."""
import argparse,inspect,json,importlib
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
import traceback,zipfile
import numpy as np
from PIL import Image
from core_clock import CoreClock,compile_ground,prepare_inputs,core_scope
from run_attributes import prepare,read,write,verify,sha,HERE
from e2e_logic import canonical,compare,mask_summary


def core_call(backend,module,image,roi,region,torch,audit_dir=None):
    prepared=prepare_inputs(backend,module,image,roi,region,torch)
    clock=CoreClock(torch.cuda.synchronize);ground=compile_ground(backend.ground,clock)
    if audit_dir is not None:
        (audit_dir/'original_ground_method.py').write_text(clock.original_source)
        (audit_dir/'timed_ground_method.py').write_text(clock.timed_source+'\n')
    sam_module=importlib.import_module('fashion_multimodal_analysis.grounding.refinement.evaluate_sam_refinement_9case')
    with core_scope(backend,prepared,clock,torch,sam_module):prediction,unused=ground(image,roi,region)
    if audit_dir is not None:
        (audit_dir/'original_sam_helper.py').write_text(clock.sam_original_source)
        (audit_dir/'timed_sam_helper.py').write_text(clock.sam_timed_source+'\n')
    row=clock.record();status=prediction['status'];sam=int(status in ['detected','empty_mask'])
    if status=='not_applicable':
        if clock.started or clock.forward:raise RuntimeError('NOT_APPLICABLE_MUST_NOT_ENTER_MODEL_CLOCK')
    else:
        if prepared is None or not clock.started or clock.active_ms<=0:raise RuntimeError('APPLICABLE_QUERY_WITHOUT_GPU_CORE_CLOCK')
        if dict(clock.forward)!={**{'dino':1},**({'sam':1,'sam_vision':1} if sam else {})}:raise RuntimeError('CORE_MODEL_FORWARD_COVERAGE_CHANGED')
    if sam:
        if clock.local_mask is None:raise RuntimeError('LOCAL_BINARY_MASK_ENDPOINT_MISSING')
        row['endpoint_mask']=mask_summary(clock.local_mask,prepared['crop_size']);row['endpoint_bbox_roi_local']=clock.local_bbox
        if status=='detected':
            x,y=prepared['origin'][:2];b=clock.local_bbox
            if list(prediction['bbox'])!=[b[0]+x,b[1]+y,b[2]+x,b[3]+y]:raise RuntimeError('CORE_BBOX_ENDPOINT_CHANGED')
        elif row['endpoint_mask']['pixels']!=0:raise RuntimeError('EMPTY_ENDPOINT_NOT_EMPTY')
    row['timing_boundary_source_sha256']=clock.boundary_sha256
    return canonical(prediction,image.size),row


def stats(rows):
    values=[r['core_active_wall_ms'] for r in rows if r['eligible_core_sample']]
    if not values:return {'n':0,'mean_ms':None,'p50_ms':None,'p95_ms':None}
    return {'n':len(values),'mean_ms':float(np.mean(values)),'p50_ms':float(np.percentile(values,50)),'p95_ms':float(np.percentile(values,95))}


def benchmark(jobs,reference,invoke,out,passes=5,warmups=10):
    applicable=[j for j in jobs if reference[j['key']]['status']!='not_applicable']
    # This list is checked against the structural router in core_call/preflight;
    # absence/missed/geometry-rejected queries remain timed, without selection by success.
    for n in range(warmups):
        job=applicable[n%len(applicable)];print(f'CORE WARMUP {n+1}/{warmups} {job["key"]}',flush=True);invoke(job)
    checks=[];preflight=[];records=[];output=out/'preflight_outputs';output.mkdir()
    image_jobs=defaultdict(list)
    for job in jobs:image_jobs[job['image_id']].append(job)
    for n,(rid,group) in enumerate(image_jobs.items()):
        per_image=[]
        print(f'CORE PREFLIGHT {n+1}/{len(image_jobs)} {rid}',flush=True)
        for job in group:
            pred,row=invoke(job);match=compare(reference[job['key']],pred)
            if (reference[job['key']]['status']!='not_applicable')!=row['eligible_core_sample']:raise RuntimeError('STRUCTURAL_ROUTING_CHANGED')
            item={**{k:job[k] for k in ['image_id','instance_id','region']},'comparison':match,'passed':match['passed']}
            checks.append(item);per_image.append({'meta':{k:job[k] for k in ['image_id','instance_id','region']},'prediction':pred,'clock':row})
            preflight.append({**item,'clock':row})
            if not match['passed']:
                write(output/f'{rid}_partial.json',per_image);write(out/'preflight_checks.json',checks)
                return blocked(out,'PREFLIGHT_OUTPUT_DRIFT',item)
        write(output/f'{rid}.json',per_image);write(out/'preflight_checks.json',checks)
    write(out/'preflight_clock_records.json',preflight)
    timed_checks=[]
    for p in range(passes):
        for n,(rid,group) in enumerate(image_jobs.items()):
            print(f'CORE TIMED PASS {p+1}/{passes} IMAGE {n+1}/{len(image_jobs)} {rid}',flush=True)
            for job in group:
                if reference[job['key']]['status']=='not_applicable':continue
                pred,row=invoke(job);match=compare(reference[job['key']],pred)
                meta={k:job[k] for k in ['image_id','instance_id','region']}
                records.append({'pass':p,**meta,**row});timed_checks.append({'pass':p,**meta,**match})
                if not match['passed']:
                    write(out/'timed_records.json',records);write(out/'timed_checks.json',timed_checks)
                    write(out/'drift_output.json',{'pass':p,**meta,'actual':pred,'reference':reference[job['key']]})
                    return blocked(out,'TIMED_OUTPUT_DRIFT',timed_checks[-1])
            write(out/'timed_records.json',records);write(out/'timed_checks.json',timed_checks)
    complete=len(records)==passes*len(applicable) and all(x['passed'] for x in timed_checks)
    by_region={q:stats([r for r in records if r['region']==q]) for q in sorted({r['region'] for r in records})}
    by_status={status:stats([r for r in records if r['status']==status]) for status in sorted({r['status'] for r in records})}
    summary={'status':'CORE_ACTIVE_WALL_MEASUREMENT_COMPLETE' if complete else 'INCOMPLETE','complete':complete,
        'all_output_checks_pass':complete,'unique_images':len(image_jobs),'unique_queries':len(jobs),'applicable_queries':len(applicable),
        'not_applicable_excluded_queries':len(jobs)-len(applicable),'warmup_calls':warmups,'timed_passes':passes,
        'all_applicable_query_core_active_wall':stats(records),'by_region':by_region,'by_return_status':by_status,
        'mask_generated_paths':stats([r for r in records if r['status'] in ['detected','empty_mask']]),
        'no_mask_candidate_paths':stats([r for r in records if r['status'] not in ['detected','empty_mask']]),
        'per_pass':{str(p):stats([r for r in records if r['pass']==p]) for p in range(passes)},
        'timing_keys_unique':len({(r['pass'],r['image_id'],r['instance_id'],r['region']) for r in records})==len(records),
        'exclusion_counts':dict(sum((Counter(r['exclusions']) for r in records),Counter())),
        'clock_type':'active synchronized wall intervals within one ROI + one query; explicitly excludes preprocessing/transfers/materialization and final full-image formatting',
        'continuous_gpu_resident_end_to_end_interval':False,'pure_gpu_kernel_time':False,
        'dynamic_sam_box_staging_excluded_and_explicitly_recorded':True,
        'text_image_model_embeddings_recomputed_each_call':True,'output_mask_device_at_endpoint':'CPU, original SAM CPU post_process_masks path; D2H transfer excluded',
        'source_images_training_origin_not_blind':True,'accuracy_claim':False,'new_optimization_applied':False,'production_modified':False,
        'scope_review_note':'Compare to teacher target only as the declared active-core wall metric. If a continuous all-GPU implementation is required, mark that alternative metric unassessed; do not relabel this result.'}
    if complete:summary['mean_vs_configured_30ms_core_target']='at_or_below' if summary['all_applicable_query_core_active_wall']['mean_ms']<=30 else 'above'
    write(out/'summary.json',summary);return summary


def blocked(out,status,detail):
    result={'status':status,'complete':False,'all_output_checks_pass':False,'detail':detail,'production_modified':False}
    write(out/'summary.json',result);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',default='/workspace/fashion_multimodal_analysis')
    ap.add_argument('--prior',default='/workspace/prd31_week7_1021_batch/run_20261021_033050_887414')
    ap.add_argument('--benchmark',default='/workspace/prd31_week7_1021_benchmark/run_20261021_040120_362409')
    ap.add_argument('--profile',default='/workspace/prd31_week7_1021_ground_profile/run_20261021_080233_916752')
    ap.add_argument('--output-root',default='/workspace/prd31_week7_1021_core_timing');args=ap.parse_args()
    root=Path(args.output_root).resolve();root.mkdir(parents=True,exist_ok=True)
    out=root/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir();code=1
    try:
        reference_dir=Path(args.profile).resolve();verify(reference_dir,read(HERE/'inputs/PROFILE_REFERENCE_SHA256.json'))
        project=Path(args.project).resolve();verify(project,read(HERE/'inputs/GROUND_HELPER_SHA256.json'))
        project,prior,bench,backend,module,Pipeline,old_jobs,images,old,torch=prepare(args,out)
        verify(project,read(HERE/'inputs/FROZEN_ATTRIBUTE_SOURCES.json'))
        for name in ['ground_model','sam_model']:
            source=Path(inspect.getsourcefile(type(getattr(backend,name))))
            expected=read(reference_dir/'hook_inventory.json')['dino_model' if name=='ground_model' else 'sam_model']['source_sha256']
            if sha(source)!=expected:raise RuntimeError('TRANSFORMERS_IMPLEMENTATION_CHANGED: '+name)
        from fashion_multimodal_analysis.integration.pipeline import GarmentROI
        jobs=[];reference={};queries=read(HERE/'inputs/config.json')['required_regions']
        for record in read(prior/'segmentation_all_predictions.json'):
            rid=record['image_id'];ref=read(reference_dir/'outputs'/f'{rid}_ordinary.json')['raw_ground_queries']
            for row in ref:
                key=(rid,row['instance_id'],row['region']);reference[key]=row['prediction']
            for item in record['instances']:
                with Image.open(prior/item['mask_file']) as im:mask=(np.asarray(im)>0).astype('uint8')
                roi=GarmentROI(instance_id=item['instance_id'],category=item['category'],score=item['score'],bbox=item['bbox'],mask=mask)
                for region in queries:jobs.append({'image_id':rid,'instance_id':roi.instance_id,'region':region,'roi':roi,'key':(rid,roi.instance_id,region)})
        if len(jobs)!=944 or len(reference)!=944 or {j['key'] for j in jobs}!=set(reference):raise RuntimeError('FIXED_QUERY_SET_CHANGED')
        structural_applicable=sum(module.supports_region(j['roi'].category,j['region']) for j in jobs)
        if structural_applicable!=527:raise RuntimeError('FIXED_STRUCTURAL_APPLICABILITY_CHANGED')
        write(out/'measurement_definition.json',{'unit':'one frozen predicted garment ROI + one fixed natural-language query',
            'initial_ready_inputs':'DINO image/text tensors and SAM image pixel tensor on GPU; no model embeddings cached',
            'core_start':'immediately before DINO forward; GPU inputs synchronized and ready',
            'core_end':'ROI-local refined bbox + binary clipped mask before full-image canvas/metadata',
            'method':'active synchronized wall intervals; explicit pauses for original dynamic SAM input processor, CPU/GPU transfers and tensor materialization',
            'sam_dynamic_box':'derived from this DINO call; never prefilled from saved predictions; original processor scaling unchanged',
            'mask_generation':'original SAM decoder and original CPU mask interpolation/thresholding retained; D2H copy excluded',
            'no_pure_gpu_or_continuous_all_gpu_latency_claim':True,'model_io_wrappers_have_some_overhead':True,
            'not_applicable_417_not_in_denominator':True,'candidate_unchanged':True,'new_roi_boundary_inputs':False})
        audit=out/'timing_source';audit.mkdir()
        wrote=False
        def invoke(job):
            nonlocal wrote
            pred,row=core_call(backend,module,images[job['image_id']],job['roi'],job['region'],torch,audit if not wrote else None)
            wrote=True;return pred,row
        del old_jobs,old
        with torch.inference_mode():result=benchmark(jobs,reference,invoke,out)
        if result.get('all_output_checks_pass'):
            from run_candidate import run_image
            rid=jobs[0]['image_id'];source=next(Path(x['path']) for x in read(prior/'images_inference_only.json') if x['image_id']==rid)
            smoke=out/'candidate_entry_smoke';smoke.mkdir()
            print('CANDIDATE ENTRY INFERENCE SMOKE '+rid,flush=True)
            smoke_ok=run_image(backend,Pipeline,torch,source,smoke)
            comparison=compare(read(reference_dir/'outputs'/f'{rid}_ordinary.json')['delivered_output'],read(smoke/'prediction.json')['output'])
            result['candidate_entry_inference_smoke']={'passed':bool(smoke_ok and comparison['passed']),'image_id':rid,
                'comparison':comparison,'existing_backend_used':True,'standalone_model_initialization_not_separately_smoked':True}
            write(out/'candidate_entry_smoke_check.json',result['candidate_entry_inference_smoke'])
        for folder,name in [(prior,'PRIOR_FILES_SHA256.json'),(bench,'REFERENCE_BENCH_SHA256.json'),(reference_dir,'PROFILE_REFERENCE_SHA256.json')]:verify(folder,read(HERE/'inputs'/name))
        for name in ['source_hashes.json','FROZEN_ATTRIBUTE_SOURCES.json','GROUND_HELPER_SHA256.json']:verify(project,read(HERE/'inputs'/name))
        for row in read(prior/'images_inference_only.json'):
            if sha(Path(row['path']))!=row['sha256']:raise RuntimeError('SOURCE_IMAGE_CHANGED')
        result.update(source_images_unchanged=True,production_sources_unchanged=True,reference_files_unchanged=True)
        write(out/'summary.json',result)
        code=0 if result.get('all_output_checks_pass') and result.get('candidate_entry_inference_smoke',{}).get('passed',False) else 1
    except BaseException:
        error=traceback.format_exc();print(error,flush=True);write(out/'fatal_error.json',{'error':error})
        blocked(out,'BLOCKED_OR_INTERRUPTED',error)
    finally:
        archive=out.with_name(out.name+'_UPLOAD_THIS.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(out))
        print('UPLOAD_THIS: '+str(archive),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
