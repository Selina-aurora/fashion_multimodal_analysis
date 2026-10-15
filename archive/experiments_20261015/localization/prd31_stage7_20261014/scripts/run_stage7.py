#!/usr/bin/env python3
"""Stage7: read-only, CUDA-synchronized profile of DINO and SAM phase runtimes.

This measures phase wall time with extra sync; NOT PRD acceptance, model quality,
or an optimization. No datasets, project code, model weights, or cache are modified.
"""
from __future__ import annotations
import argparse
import collections
import importlib
import json
import os
import statistics
import sys
import time
import traceback
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT_DEFAULT = Path('/workspace/fashion_multimodal_analysis')
OUT_DEFAULT = Path('/workspace/prd31_stage7_runs_20261014')
sys.path.insert(0,str(BASE))
import stage6_helpers as helper

DINO_MODULE = 'fashion_multimodal_analysis.grounding.refinement.run_shoulder_waist_pattern_groundingdino_refinement_v1'
SAM_MODULE = 'fashion_multimodal_analysis.grounding.refinement.evaluate_sam_refinement_9case'


def phase_wrapper(original, phase, phase_log, sync):
    """Preserve exact call arguments and return value; collect elapsed wall time."""
    def wrapper(*args, **kwargs):
        sync()
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            sync()
            phase_log.append({'stage':phase,'ms':(time.perf_counter()-start)*1000})
    return wrapper


def summary(vals):
    import numpy as np
    if not vals:return {'n':0,'mean_ms':None,'median_ms':None,'p95_ms':None}
    return {'n':len(vals),'mean_ms':round(statistics.mean(vals),4),
            'median_ms':round(statistics.median(vals),4),
            'p95_ms':round(float(np.percentile(vals,95)),4)}


def choose_rois(backend, selection):
    from PIL import Image
    candidates=collections.defaultdict(list)
    for k,s in enumerate(selection,1):
        with Image.open(s['source_image']) as raw:
            im=raw.convert('RGB')
        rois,_=backend.segment(im)
        if not rois:continue
        roi=sorted(rois,key=helper.preferred)[0]
        for r in helper.allowed_regions(roi.category):
            candidates[r].append((f'case_{k:02d}', im, roi,r))
    return candidates


def test_profile(backend, selection, out, warmups, passes, pairs_per_region):
    import torch
    from fashion_multimodal_analysis.integration.model_backends import ModelBackends
    if not isinstance(backend, ModelBackends): raise TypeError('expected real ModelBackends')
    candidates=choose_rois(backend,selection)
    samples=[]
    for region in helper.ALL_REGIONS:
        samples.extend(sorted(candidates[region],key=lambda x:x[0])[:pairs_per_region])
    if not samples:raise ValueError('No eligible predicted garment region inputs')
    write_sample=[{'case_id':c,'instance_id':roi.instance_id,'category':roi.category,
                   'region':r,'image_width':img.width,'image_height':img.height}
                   for c,img,roi,r in samples]
    helper.write_csv(out/'fixed_profile_inputs.csv',write_sample)
    dino_mod=importlib.import_module(DINO_MODULE)
    sam_mod=importlib.import_module(SAM_MODULE)
    old_dino=dino_mod.run_grounding_dino
    old_sam=sam_mod.run_sam
    def sync():torch.cuda.synchronize()
    recorded=[]
    try:
        for j in range(warmups):
            backend.ground(samples[0][1],samples[0][2],samples[0][3])
        print('WARMUP_COMPLETE',warmups,flush=True)
        for rep in range(1,passes+1):
            for case,img,roi,region in samples:
                phase_log=[]
                dino_mod.run_grounding_dino=phase_wrapper(old_dino,'dino',phase_log,sync)
                sam_mod.run_sam=phase_wrapper(old_sam,'sam',phase_log,sync)
                try:
                    result,total=backend.ground(img,roi,region)
                    error=''
                except Exception as exc:
                    result={'status':'ERROR'}
                    total=None
                    error=f'{type(exc).__name__}: {exc}'
                finally:
                    dino_mod.run_grounding_dino=old_dino
                    sam_mod.run_sam=old_sam
                grouped=collections.defaultdict(list)
                for ph in phase_log:grouped[ph['stage']].append(ph['ms'])
                dino_ms=sum(grouped['dino'])
                sam_ms=sum(grouped['sam'])
                overhead=(total-dino_ms-sam_ms) if total is not None else None
                recorded.append({'pass':rep,'case_id':case,'category':roi.category,'region':region,
                    'status':result.get('status','unknown'),'total_ms':total,
                    'dino_ms':round(dino_ms,5),'sam_ms':round(sam_ms,5),
                    'other_overhead_ms':round(overhead,5) if overhead is not None else '',
                    'dino_calls':len(grouped['dino']),'sam_calls':len(grouped['sam']),
                    'error':error,'scope':'CUDA_SYNC_PHASE_WALLTIME_DIAGNOSTIC'})
            print('TIMED_PASS_DONE',rep,'/',passes,flush=True)
    finally:
        dino_mod.run_grounding_dino=old_dino
        sam_mod.run_sam=old_sam
    helper.write_csv(out/'phase_timing_raw.csv',recorded)
    valid=[r for r in recorded if not r['error'] and r['total_ms'] is not None]
    by_status=[]
    for stat in sorted({r['status'] for r in valid}):
        subset=[r for r in valid if r['status']==stat]
        by_status.append({'status':stat,'n':len(subset),
                          'mean_total_ms':summary([r['total_ms'] for r in subset])['mean_ms'],
                          'mean_dino_ms':summary([r['dino_ms'] for r in subset])['mean_ms'],
                          'mean_sam_ms':summary([r['sam_ms'] for r in subset])['mean_ms'],
                          'sam_invoked_count':sum(r['sam_calls']>0 for r in subset)})
    helper.write_csv(out/'phase_timing_by_status.csv',by_status,
                     fields=['status','n','mean_total_ms','mean_dino_ms','mean_sam_ms','sam_invoked_count'])
    out_summary={'status':'STAGE7_PROFILE_COMPLETED' if len(valid)==len(recorded) else 'STAGE7_PROFILE_PARTIAL',
                 'profiled_queries':len(samples), 'timed_passes':passes,'warmup_runs':warmups,
                 'total_calls':len(recorded),'valid_calls':len(valid),'error_calls':len(recorded)-len(valid),
                 'all':{k:summary([float(r[k]) for r in valid]) for k in
                        ('total_ms','dino_ms','sam_ms','other_overhead_ms')},
                 'sam_invocations':sum(r['sam_calls'] for r in valid),
                 'by_status':by_status,
                 'notes':['CUDA synchronizations added inside per-phase wrappers; relative phase diagnostics only.',
                         'SAM is skipped for missed DINO proposals; zero indicates no call, not a zero-cost SAM inference.',
                         'Other overhead includes crop/mask geometry, Python bookkeeping, and synchronization overhead.',
                         'This is 8-image repeated-call development profiling; NOT independent latency acceptance.',
                         'No threshold sweeps, no model/data changes, no human-reviewed GT labels.']}
    helper.write_json(out/'phase_summary.json',out_summary)
    return out_summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(ROOT_DEFAULT))
    p.add_argument('--out-root',default=str(OUT_DEFAULT))
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--warmup-runs',type=int,default=10)
    p.add_argument('--timed-passes',type=int,default=5)
    p.add_argument('--pairs-per-region',type=int,default=2)
    args=p.parse_args()
    if args.warmup_runs<10 or args.timed_passes<5 or args.pairs_per_region<1:
        p.error('must use at least 10 warmups, 5 timed passes, 1 pair per region')
    root=Path(args.root).resolve()
    out_root=Path(args.out_root).resolve()
    if root==out_root or root in out_root.parents or out_root==root.parent:
        p.error('refusing to write within project repository')
    out=helper.new_out(out_root)
    log={'status':'STARTED','dry_run':args.dry_run,'project_files_modified':False,
         'source':'Stage6 same V5 Balanced150 eight-image diagnostic'}
    exit_code=0
    try:
        os.environ.setdefault('HF_HOME','/workspace/hf_cache')
        os.environ.setdefault('HF_HUB_CACHE','/workspace/hf_cache/hub')
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        sys.path.insert(0,str(root/'src'))
        config,selection=helper.check_inputs(root,out)
        if args.dry_run:
            log['status']='DRY_RUN_PASS'
            print('DRY_RUN_PASS',flush=True)
        else:
            log['checkpoint']=helper.inspect_checkpoint(root)
            from fashion_multimodal_analysis.integration.model_backends import ModelBackends
            print('LOADING_EXISTING_MODELS',flush=True)
            backend=ModelBackends(config,device='cuda')
            print('START_PHASE_PROFILING',flush=True)
            result=test_profile(backend,selection,out,args.warmup_runs,args.timed_passes,args.pairs_per_region)
            log['profile_summary']=result
            log['status']=result['status']
    except Exception as e:
        log['status']='STAGE7_FAILED'
        log['error']=f'{type(e).__name__}: {e}'
        (out/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('ERROR:',log['error'],flush=True)
        exit_code=2
    finally:
        helper.write_json(out/'run_audit.json',log)
        try:
            z=out.parent / f'prd31_stage7_{out.name}_UPLOAD_THIS.zip'
            with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
                for file in sorted(out.rglob('*')):
                    if file.is_file():archive.write(file,file.relative_to(out))
            with zipfile.ZipFile(z) as archive:
                if archive.testzip() is not None:raise ValueError('Output ZIP CRC failed')
            print('STATUS:',log['status'],flush=True)
            print('UPLOAD_THIS_ZIP:',z,flush=True)
            print('PROJECT_FILES_MODIFIED: False',flush=True)
        except Exception as e:
            print('ZIP_PACK_FAILED:',e,flush=True)
            exit_code=3
    return exit_code

if __name__=='__main__':sys.exit(main())
