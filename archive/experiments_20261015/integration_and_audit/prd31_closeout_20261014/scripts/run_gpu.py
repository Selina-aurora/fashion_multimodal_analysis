#!/usr/bin/env python3
"""Capture a read-only run snapshot, same-image predictions and warmed chain timing.

This runner never supplies human boxes/labels to inference, changes a model,
installs a batch optimization, trains, commits Git, or modifies the repository.
The twelve-image cohort is development data, not a clean blind test.
"""
import argparse, hashlib, io, json, os, platform, statistics, subprocess, sys, time, traceback, zipfile
from datetime import datetime, timezone
from pathlib import Path
from common import BASE, sha, load, dump, csvout

EXPECTED = {
 'configs/prd_8class_train_v5_balanced150_20261008.csv':'e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f',
 'configs/prd_8class_val_v2_full_targets.csv':'39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93',
 'benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv':'c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89',
 'outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth':'c5ba795225349b1c521aac0517ac7c493d9818dcc17d558ac4a3f6f8669452f0',
 'src/fashion_multimodal_analysis/integration/pipeline.py':'8c6e8b3278cc94fdbc934defbd7f51aa5081dabb1d767a03c9160854dc22da68',
 'src/fashion_multimodal_analysis/integration/model_backends.py':'0efd47f3a29e6c25d7396717cb4fc1b1540ed30c16df171a9ad7b93fade7f7ec',
 'src/fashion_multimodal_analysis/integration/semantic_consistency.py':'4aaa4c17604d272a557c1aa14a9acd6d1138a5129ec1a94ded622046acc8a49c',
}

def fingerprint(root):
    """Record actual working-tree content even when Git has uncommitted edits."""
    paths = set(root/p for p in EXPECTED)
    for folder, suffixes in [('src', {'.py'}), ('configs', {'.json','.csv','.yaml','.yml'})]:
        paths.update(p for p in (root/folder).rglob('*') if p.is_file() and p.suffix in suffixes)
    return {str(p.relative_to(root)):sha(p) for p in sorted(paths)}

def git_info(root):
    # GIT_EXECUTABLE_OPTIONAL_GUARD
    import shutil
    if shutil.which("git") is None:
        return {"status": "GIT_METADATA_UNAVAILABLE",
                "commit": "UNAVAILABLE",
                "working_tree": "UNAVAILABLE",
                "reason": "git executable not installed"}
    result = {}
    for key,args in [('commit',['rev-parse','HEAD']),('working_tree',['status','--porcelain'])]:
        p = subprocess.run(['git','-C',str(root),*args], capture_output=True, text=True)
        result[key] = p.stdout.strip() if p.returncode == 0 else 'UNAVAILABLE'
    return result

def summary(values):
    import numpy as np
    return {'n':len(values),'mean_ms':statistics.mean(values),'median_ms':statistics.median(values),
            'p95_ms':float(np.percentile(values,95)), 'max_ms':max(values)}

def hf_snapshot(backend, cfg):
    """Hash the cache revision actually loaded; do not download or guess revision IDs."""
    from transformers.utils.hub import cached_file
    out = {}
    for key, obj in [('grounding_model',backend.ground_model),('sam_model',backend.sam_model),('clip_model',backend.clip_model)]:
        revision = getattr(obj.config,'_commit_hash',None)
        if not revision:
            out[key]={'status':'LOADED_REVISION_UNAVAILABLE','model_id':cfg[key]}; continue
        p = Path(cached_file(cfg[key], 'config.json', revision=revision, local_files_only=True))
        files = {str(f.relative_to(p.parent)):sha(f) for f in sorted(p.parent.rglob('*')) if f.is_file()}
        out[key]={'status':'LOADED_CACHE_SNAPSHOT_HASHED','model_id':cfg[key],'revision':revision,'files':files}
    return out

def stage15_crop_hash(image, roi):
    from fashion_multimodal_analysis.integration.pipeline import roi_images
    crop,_,origin = roi_images(image,roi)
    preview = crop.convert('RGB').copy(); preview.thumbnail((1000,1000))
    b = io.BytesIO(); preview.save(b,'JPEG',quality=82,optimize=True)
    return hashlib.sha256(b.getvalue()).hexdigest(), list(crop.size), origin

def preferred(roi):
    rank={'top':0,'outerwear':1,'dress':2,'pants':3,'skirt':4,'bag':5,'shoe':6,'accessory':7}
    return (rank.get(roi.category,10),-roi.score)

def pack(out):
    path = out.with_suffix('.zip')
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(out))
    return path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/workspace/fashion_multimodal_analysis'))
    p.add_argument('--out-root',type=Path,default=Path('/workspace/prd31_closeout_runs_20261014'))
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--warmup-passes',type=int,default=1,help='Complete unmeasured passes over all twelve images')
    p.add_argument('--timed-passes',type=int,default=3)
    args=p.parse_args()
    if args.warmup_passes<1 or args.timed_passes<1: p.error('At least one warmup and one timed pass are required')
    root=args.root.resolve(); parent=args.out_root.resolve()
    if parent==root or root in parent.parents: p.error('Output must be outside the project')
    parent.mkdir(parents=True,exist_ok=True)
    out=parent/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir()
    audit={'status':'STARTED','formal_prd_acceptance':False,'production_modified':False,
           'gt_used_for_inference':False,'cohort':'Stage15_DEV12_not_blind',
           'manifest_sha256':sha(BASE/'reference/cohort.json'),'date_utc':datetime.now(timezone.utc).isoformat()}
    rc=0
    try:
        if not root.is_dir(): raise FileNotFoundError(root)
        os.chdir(root)
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
        os.environ.setdefault('HF_HOME','/workspace/hf_cache'); os.environ.setdefault('HF_HUB_CACHE','/workspace/hf_cache/hub')
        os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
        sys.path.insert(0,str(root/'src'))
        before=fingerprint(root); audit['git']=git_info(root)
        dump(out/'working_tree_hashes_before.json',before)
        bad=[k for k,v in EXPECTED.items() if before.get(k)!=v]
        if bad: raise ValueError('Baseline differs from Oct14 archive; do not compare blindly: '+', '.join(bad))
        cohort=load(BASE/'reference/cohort.json'); cfg=load(BASE/'reference/run_config.json')
        for c in cohort:
            if sha(c['source_image']) != c['source_sha256']: raise ValueError('Image changed: '+c['case_id'])
        dump(out/'cohort.json',cohort);dump(out/'run_config.json',cfg)
        audit['inputs_verified']=True
        if args.preflight_only:
            audit['status']='PREFLIGHT_ONLY_NO_INFERENCE'; return 0
        from PIL import Image
        import torch
        from fashion_multimodal_analysis.integration.model_backends import ModelBackends
        from fashion_multimodal_analysis.integration.pipeline import FashionPipeline
        from smoke_export import save_prediction
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        audit['environment']={'python':platform.python_version(),'torch':torch.__version__,
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}
        audit['installed_packages']=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout.splitlines()
        print('Loading existing baseline; models stay unchanged.',flush=True)
        backend=ModelBackends(cfg,device='cuda'); pipeline=FashionPipeline(backend)
        audit['hf_snapshots']=hf_snapshot(backend,cfg)
        images=[]
        for c in cohort:
            with Image.open(c['source_image']) as im: images.append(im.convert('RGB'))
        timings=[]; case_outputs={}; replay=[]
        for phase, count in [('warmup',args.warmup_passes),('measured',args.timed_passes)]:
            for rep in range(count):
                for c,im in zip(cohort,images):
                    torch.cuda.synchronize();start=time.perf_counter()
                    with torch.inference_mode(): result=pipeline.run(im,cfg['required_regions'],cfg['required_attributes'])
                    torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                    # The entire production call, every predicted garment, every query and
                    # semantic guard is inside the timer. File decoding/export are outside.
                    failure=len(result.get('failures',[]))>0 or result.get('status')!='completed'
                    if phase=='measured':
                        timings.append({'case_id':c['case_id'],'pass':rep+1,'chain_wall_ms':elapsed,
                            'instance_count':len(result.get('instances',[])),'pipeline_status':result.get('status'),
                            'failed_calls':len(result.get('failures',[])),'has_failure':failure})
                        csvout(out/'chain_timing.csv',timings)
                        if rep==0:
                            payload=save_prediction(im,result,out/c['case_id'])
                            item={**c,'image_size':list(im.size),'pipeline_status':result.get('status'),
                                  'failures':result.get('failures',[]),'instances':payload['instances'],
                                  'interface_issues':payload['interface_issues']}
                            case_outputs[c['case_id']]=item
                            # Check selected-crop bytes in addition to class and size. A mismatch
                            # blocks automatic use of the historical Stage15 qualitative grades.
                            rois=[x['roi'] for x in result.get('instances',[])]
                            selected=sorted(rois,key=preferred)[0] if rois else None
                            got,size,origin=stage15_crop_hash(im,selected) if selected else ('',[],[])
                            replay.append({'case_id':c['case_id'],'selected_instance_id':selected.instance_id if selected else '',
                                'crop_sha256':got,'crop_size':json.dumps(size),'crop_origin':json.dumps(origin),
                                'exact_stage15_crop_match':got==c['stage15_crop_sha256']})
                            dump(out/'predictions_same12.json',{'schema_version':'prd31_same12_predictions_v1',
                                'manifest_sha256':audit['manifest_sha256'],'cases':case_outputs})
                    print(phase,rep+1,c['case_id'],f'{elapsed:.1f} ms', 'FAILED' if failure else 'done',flush=True)
        csvout(out/'stage15_crop_replay.csv',replay)
        dump(out/'latency_summary.json',{'scope':'preloaded_RGB_image_to_complete_pipeline_result_all_predicted_garments',
            'includes':['segmentation','all_configured_region_queries','all_configured_attributes','consistency_checks','CPU_GPU_synchronization'],
            'excludes':['model_loading','disk_image_decode','metrics','image_JSON_export'],
            'warmup_passes':args.warmup_passes,'timed_passes':args.timed_passes,
            'all_calls':summary([r['chain_wall_ms'] for r in timings]),
            'failure_calls':sum(r['has_failure'] for r in timings),
            'empty_prediction_calls':sum(r['instance_count']==0 for r in timings),
            'prd_end_to_end_target_ms':None,'acceptance':'MEASURED_NO_AGREED_CHAIN_THRESHOLD',
            'note':'Do not compare image-level chain time to per-region 30 ms or attribute 20 ms.'})
        after=fingerprint(root); dump(out/'working_tree_hashes_after.json',after)
        if before!=after:raise RuntimeError('Working tree or protected assets changed during run')
        audit['unchanged_during_run']=True
        audit['stage15_exact_crop_replays']=sum(x['exact_stage15_crop_match'] for x in replay)
        audit['interface_issues']=sum(len(x['interface_issues']) for x in case_outputs.values())
        issues=sum(r['has_failure'] for r in timings)+audit['interface_issues']
        audit['status']='PREDICTIONS_AND_TIMING_RECORDED' if not issues else 'COMPLETED_WITH_FAILURES_REVIEW_REQUIRED'
        audit['freeze_status']='RUN_SNAPSHOT_RECORDED_HUMAN_ACCEPTANCE_PENDING'
        audit['selection_decision']='RETAIN_EXISTING_BASELINE_BATCH_OPTIMIZATION_NOT_DEPLOYED'
        if issues:rc=2
    except Exception as exc:
        rc=2;audit.update(status='FAILED',error=str(exc))
        (out/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('ERROR:',exc,flush=True)
    finally:
        dump(out/'run_audit.json',audit)
        inventory={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
        dump(out/'FILES_SHA256.json',inventory)
        print('STATUS:',audit['status'],flush=True);print('UPLOAD_THIS_ZIP:',pack(out),flush=True)
    return rc

if __name__=='__main__':raise SystemExit(main())
