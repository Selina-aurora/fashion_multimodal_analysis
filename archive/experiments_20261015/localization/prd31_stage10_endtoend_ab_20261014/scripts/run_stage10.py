#!/usr/bin/env python3
"""Read-only production-call vs joint DINO/SAM batching, same image and query set.

This diagnoses speed and output parity. No ground-truth part-localization accuracy,
no production edits, and no assertion of per-query PRD latency compliance.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import zipfile

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import stage6_helpers as h
import stage9_dino_impl as dino
import stage8_sam_impl as sam

ROOT = Path('/workspace/fashion_multimodal_analysis')
OUTROOT = Path('/workspace/prd31_stage10_runs_20261014')


def find_cases(backend, images, limit=4, query_count=4):
    from PIL import Image
    from fashion_multimodal_analysis.integration.pipeline import roi_images
    from fashion_multimodal_analysis.benchmarking.prepare_prd31_recovery import PROMPTS
    cases, skipped = [], []
    for i, record in enumerate(images, 1):
        if len(cases) >= limit: break
        with Image.open(record['source_image']) as f: image = f.convert('RGB')
        rois, _ = backend.segment(image)
        if not rois:
            skipped.append({'image_index':i, 'reason':'segmentation_returned_no_instances'})
            continue
        roi = sorted(rois, key=h.preferred)[0]
        regions = h.allowed_regions(roi.category)[:query_count]
        if len(regions)<2:
            skipped.append({'image_index':i,'reason':'fewer_than_two_queries'})
            continue
        crop, garment_mask, origin = roi_images(image, roi)
        isolated = Image.composite(crop, Image.new('RGB', crop.size, 'white'), garment_mask)
        cases.append({'case':f'case_{i:02d}', 'category':roi.category,'image':image,
                      'roi':roi,'crop':crop,'mask':garment_mask,'origin':origin,
                      'isolated':isolated,'regions':regions,'prompts':[PROMPTS[x] for x in regions]})
    return cases, skipped


def old_ground(backend, c):
    """The current application's public ground() method, once per region."""
    return [backend.ground(c['image'], c['roi'], region)[0] for region in c['regions']]


def new_ground(backend, c):
    """Keep the same DINO top candidate, SAM multimask argmax, and mask intersection.

    DINO requests use the identical isolated crop with different prompts in one batch.
    SAM is called once on the unisolated garment crop with multiple box prompts.
    """
    import numpy as np
    from fashion_multimodal_analysis.evaluation.protocol import valid_box, original_box
    from fashion_multimodal_analysis.integration.pipeline import crop_bounds

    detections = dino.predict_dino(backend, c['isolated'], c['prompts'], batch=True)
    if len(detections) != len(c['regions']):
        raise RuntimeError('DINO returned wrong query count')
    results=[None]*len(c['regions']); boxes=[]; index=[]
    for j, (name, proposals) in enumerate(zip(c['regions'],detections)):
        usable=[d for d in proposals if valid_box(d['box']) and math.isfinite(float(d['score']))]
        if not usable:
            results[j]={'status':'missed','target_region':name,'bbox':None,'mask':None}
            continue
        best=usable[0]
        local_box=crop_bounds(best['box'],c['crop'].size)
        boxes.append(local_box); index.append((j, name, best))
    if boxes:
        refined=sam.batch_sam(backend, c['crop'], boxes)
        if len(refined)!=len(boxes):raise ValueError('SAM result length mismatch')
        garment=np.asarray(c['mask'])>0
        for (j,name, proposal),(mask,score),box in zip(index,refined,boxes):
            m=np.asarray(mask,dtype=bool)
            if m.shape!=garment.shape:
                raise ValueError(f'SAM/crop-mask shape mismatch: {m.shape} vs {garment.shape}')
            m=m & garment
            ys,xs=np.nonzero(m)
            if len(xs)==0:
                results[j]={'status':'empty_mask','target_region':name,'bbox':None,'mask':None}
                continue
            local=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
            full=np.zeros((c['image'].height,c['image'].width),dtype=np.uint8)
            x1,y1,x2,y2=c['origin']
            if m.shape!=(y2-y1,x2-x1):raise ValueError('Invalid crop dimensions')
            full[y1:y2,x1:x2]=m
            results[j]={'status':'detected','target_region':name,
                        'model_phrase':proposal['label'],'score':proposal['score'],
                        'bbox':original_box(local,c['origin']),
                        'mask':full,'proposal_bbox':original_box(box,c['origin']),
                        'sam_score':float(score),'coordinate_space':'original_image',
                        'crop_origin':c['origin']}
    if any(x is None for x in results):raise RuntimeError('Unassigned region output')
    return results


def timed(backend, fn, case):
    backend.synchronize()
    st=time.perf_counter()
    val=fn(backend, case)
    backend.synchronize()
    return val,(time.perf_counter()-st)*1000.0


def parity_row(case_id, region, old, new):
    import numpy as np
    status_a=old.get('status');status_b=new.get('status')
    row={'case':case_id,'region':region,'old_status':status_a,'new_status':status_b,
         'status_equal':status_a==status_b,'mask_iou':None,'mask_pixel_equal':None,
         'bbox_max_abs_px':None,'proposal_bbox_max_abs_px':None,
         'dino_score_abs':None,'sam_score_abs':None,'model_phrase_equal':None}
    if status_a=='detected' and status_b=='detected':
        ma=np.asarray(old['mask'])>0; mb=np.asarray(new['mask'])>0
        if ma.shape==mb.shape:
            union=int(np.count_nonzero(ma|mb))
            row['mask_iou']=(float(np.count_nonzero(ma&mb))/union if union else 1.0)
            row['mask_pixel_equal']=bool(np.array_equal(ma,mb))
        for field,outfield in [('bbox','bbox_max_abs_px'),('proposal_bbox','proposal_bbox_max_abs_px')]:
            x=old.get(field); y=new.get(field)
            if x and y:row[outfield]=max(abs(float(a)-float(b)) for a,b in zip(x,y))
        for field,outfield in [('score','dino_score_abs'),('sam_score','sam_score_abs')]:
            if field in old and field in new:
                row[outfield]=abs(float(old[field])-float(new[field]))
        row['model_phrase_equal']=old.get('model_phrase')==new.get('model_phrase')
    row['parity_pass']=bool(row['status_equal'])
    if status_a=='detected' and status_b=='detected':
        row['parity_pass']=all([
            row['mask_iou'] is not None and row['mask_iou']>=0.999,
            row['bbox_max_abs_px'] is not None and row['bbox_max_abs_px']<=1.0,
            row['proposal_bbox_max_abs_px'] is not None and row['proposal_bbox_max_abs_px']<=1.0,
            row['dino_score_abs'] is not None and row['dino_score_abs']<=0.005,
            row['sam_score_abs'] is not None and row['sam_score_abs']<=0.005,
            row['model_phrase_equal'],
        ])
    return row


def run(backend, selection, out, max_cases, queries, warmups, passes):
    cases,skip=find_cases(backend,selection,max_cases,queries)
    h.write_json(out/'selection.json',{'cases':[{'case':c['case'],'category':c['category'],
            'regions':c['regions'],'crop_size':list(c['crop'].size),
            'roi_provenance':c['roi'].provenance} for c in cases],'skipped':skip})
    if not cases:raise ValueError('No eligible cases to compare')
    timers=[];parity=[]
    for c in cases:
        print('RUNNING',c['case'],c['category'],len(c['regions']),'queries',flush=True)
        for _ in range(warmups):
            old_ground(backend,c);new_ground(backend,c)
        for rep in range(1,passes+1):
            if rep%2:
                a,ta=timed(backend,old_ground,c); b,tb=timed(backend,new_ground,c)
            else:
                b,tb=timed(backend,new_ground,c); a,ta=timed(backend,old_ground,c)
            n=len(c['regions'])
            timers.append({'case':c['case'],'pass':rep,'query_count':n,
                'baseline_total_ms':round(ta,4),'joint_batch_total_ms':round(tb,4),
                'baseline_amortized_ms':round(ta/n,4),'joint_batch_amortized_ms':round(tb/n,4),
                'speedup':round(ta/tb,4) if tb else None})
            if rep==1:
                for region,x,y in zip(c['regions'],a,b):
                    parity.append(parity_row(c['case'],region,x,y))
    h.write_csv(out/'full_localization_timing.csv',timers)
    h.write_csv(out/'final_region_parity.csv',parity)
    base=statistics.mean(x['baseline_total_ms'] for x in timers)
    bat=statistics.mean(x['joint_batch_total_ms'] for x in timers)
    all_equal=bool(parity) and all(x['parity_pass'] for x in parity)
    summ={'status':'STAGE10_END_TO_END_AB_COMPLETED', 'cases':len(cases), 'paired_queries':len(parity),
       'timed_rows':len(timers),'baseline_mean_group_ms':round(base,4),
       'joint_batch_mean_group_ms':round(bat,4),
       'baseline_mean_amortized_ms':round(statistics.mean(x['baseline_amortized_ms'] for x in timers),4),
       'joint_batch_mean_amortized_ms':round(statistics.mean(x['joint_batch_amortized_ms'] for x in timers),4),
       'overall_speedup':round(base/bat,4) if bat else None,
       'mask_parity_within_tolerance':all_equal,
       'parity_passes':sum(bool(x['parity_pass']) for x in parity),
       'parity_failures':sum(not x['parity_pass'] for x in parity),
       'integration_decision':'DO_NOT_INTEGRATE' if not all_equal else 'EXPLORATORY_ONLY_MORE_TESTS_REQUIRED',
       'limitations':['Same frozen diagnostic images, no reviewed local-part GT.',
         'Measured group throughput for multi-query same-instance; amortized ms/query is NOT standalone per-query latency.',
         'No guarantee of 30ms PRD target, no official accuracy claim.',
         'DINO image is duplicated across batch; GPU memory may increase.',
         'Only warmed calls; model loading, disk IO, and ROI selection excluded.',
         'Not a production patch.']}
    h.write_json(out/'full_localization_summary.json',summ)
    return summ


def package(out):
    dest=out.parent/f'prd31_stage10_{out.name}_UPLOAD_THIS.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(out))
    with zipfile.ZipFile(dest) as z:
        bad=z.testzip()
        if bad:raise ValueError('Bad zip entry '+bad)
    return dest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--max-cases',type=int,default=4)
    ap.add_argument('--query-count',type=int,default=4)
    ap.add_argument('--warmup-runs',type=int,default=3)
    ap.add_argument('--timed-passes',type=int,default=3)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--out-root',type=Path,default=OUTROOT)
    a=ap.parse_args()
    if not(1<=a.max_cases<=8 and 2<=a.query_count<=8 and a.warmup_runs>=2 and a.timed_passes>=1):
        ap.error('Invalid cases/queries/warmups/passes')
    root=a.root.resolve();outroot=a.out_root.resolve()
    if root==outroot or root in outroot.parents or outroot==root.parent:
        ap.error('Output must be outside repository')
    out=h.new_out(outroot)
    audit={'status':'STARTED','project_files_modified':False}
    code=0
    try:
        os.environ.setdefault('HF_HOME','/workspace/hf_cache')
        os.environ.setdefault('HF_HUB_CACHE','/workspace/hf_cache/hub')
        os.environ['HF_HUB_OFFLINE']='1'
        os.environ['TRANSFORMERS_OFFLINE']='1'
        sys.path.insert(0,str(root/'src'))
        config,selection=h.check_inputs(root,out)
        if a.dry_run:
            audit['status']='DRY_RUN_PASS'
        else:
            audit['checkpoint']=h.inspect_checkpoint(root)
            from fashion_multimodal_analysis.integration.model_backends import ModelBackends
            backend=ModelBackends(config,device='cuda')
            audit['result']=run(backend,selection,out,a.max_cases,a.query_count,a.warmup_runs,a.timed_passes)
            audit['status']=audit['result']['status']
    except Exception as e:
        audit['status']='STAGE10_FAILED'
        audit['error']=type(e).__name__+': '+str(e)
        (out/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('ERROR:',audit['error'],flush=True)
        code=2
    finally:
        h.write_json(out/'run_audit.json',audit)
        try:
            z=package(out)
            print('STATUS:',audit['status'])
            if a.dry_run:print('DRY_RUN_PASS' if audit['status']=='DRY_RUN_PASS' else 'DRY_RUN_FAILED')
            print('UPLOAD_THIS_ZIP:',z)
            print('PROJECT_FILES_MODIFIED: False')
        except Exception as e:
            print('PACK_FAILED:',e)
            code=3
    return code

if __name__=='__main__':sys.exit(main())
