#!/usr/bin/env python3
"""Read-only paired SAM serial vs multi-box microbenchmark on actual DINO proposals.

This is a feasibility diagnostic and DOES NOT modify the application pipeline.
It does not prove per-query latency compliance or semantic localization accuracy.
"""
from __future__ import annotations
import argparse
import collections
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

ROOT = Path('/workspace/fashion_multimodal_analysis')
OUT_BASE = Path('/workspace/prd31_stage8_runs_20261014')


def sanity_box(box, size):
    """Check a box before forwarding it to SAM; do not alter its coordinates."""
    from fashion_multimodal_analysis.evaluation.protocol import valid_box
    return valid_box(box) and all(math.isfinite(float(v)) for v in box) and (
        0 <= float(box[0]) < float(box[2]) <= size[0]
        and 0 <= float(box[1]) < float(box[3]) <= size[1]
    )


def safe_mask_iou(a, b):
    import numpy as np
    a=np.asarray(a, dtype=bool)
    b=np.asarray(b, dtype=bool)
    if a.shape != b.shape: return None
    union=int(np.count_nonzero(a | b))
    if union==0:return 1.0
    return float(np.count_nonzero(a & b)/union)


def make_cases(backend, selection, max_cases=2, max_boxes=4):
    """Select same-image predicted garment ROIs and real DINO proposals."""
    from PIL import Image
    from fashion_multimodal_analysis.integration.pipeline import roi_images, crop_bounds
    from fashion_multimodal_analysis.benchmarking.prepare_prd31_recovery import PROMPTS
    from fashion_multimodal_analysis.grounding.refinement.run_shoulder_waist_pattern_groundingdino_refinement_v1 import run_grounding_dino
    cases=[]; rejected=[]
    for i, entry in enumerate(selection):
        if len(cases)>=max_cases: break
        with Image.open(entry['source_image']) as opened:
            image=opened.convert('RGB')
        rois,_=backend.segment(image)
        if not rois:
            rejected.append({'image_index':i+1,'reason':'no_predicted_roi'});continue
        roi=sorted(rois,key=h.preferred)[0]
        crop,garment_mask,origin=roi_images(image,roi)
        isolated=Image.composite(crop,Image.new('RGB',crop.size,'white'),garment_mask)
        proposals=[]; seen=set()
        for region in h.allowed_regions(roi.category):
            det,_=run_grounding_dino(isolated,PROMPTS[region],
                backend.config['grounding_box_threshold'],
                backend.config['grounding_text_threshold'],
                backend.ground_processor,backend.ground_model,backend.device)
            usable=[d for d in det if sanity_box(d['box'],crop.size)]
            if not usable:continue
            # Existing ModelBackends.ground uses detections[0], then integer clipping.
            local_box=crop_bounds(usable[0]['box'],crop.size)
            if not sanity_box(local_box,crop.size):continue
            key=tuple(local_box)
            # Duplicate candidate boxes would inflate equivalence/speed statistics.
            if key in seen: continue
            seen.add(key)
            proposals.append({'region':region,'box':local_box})
            if len(proposals)>=max_boxes:break
        if len(proposals)<2:
            rejected.append({'image_index':i+1,'category':roi.category,'reason':'fewer_than_two_distinct_dino_proposals',
                'found':len(proposals)})
            continue
        cases.append({'case_id':f'case_{i+1:02d}','roi_category':roi.category,
            'source_image':entry['source_image'],'crop':crop,'proposals':proposals})
    return cases,rejected


def serial_sam(backend,crop,boxes):
    from fashion_multimodal_analysis.grounding.refinement.evaluate_sam_refinement_9case import run_sam
    results=[]
    for box in boxes:
        mask,score,_=run_sam(crop,box,backend.sam_processor,backend.sam_model,backend.device)
        results.append((mask,score))
    return results


def batch_sam(backend,crop,boxes):
    """One image with N box prompts. Verify runtime tensor shapes explicitly."""
    import numpy as np
    import torch
    proc=backend.sam_processor
    encoded=proc(images=crop,
        input_boxes=[[[float(v) for v in box] for box in boxes]],
        return_tensors='pt')
    original_sizes=encoded['original_sizes'].clone()
    reshaped_sizes=encoded['reshaped_input_sizes'].clone()
    tensor_inputs={k:v.to(backend.device) for k,v in encoded.items() if isinstance(v,torch.Tensor)}
    with torch.inference_mode():
        outputs=backend.sam_model(**tensor_inputs,multimask_output=True)
    masks=proc.image_processor.post_process_masks(
        outputs.pred_masks.detach().cpu(), original_sizes, reshaped_sizes)
    scores=outputs.iou_scores.detach().cpu()
    if len(masks)!=1 or len(masks[0])!=len(boxes) or scores.ndim!=3 or scores.shape[0]!=1 or scores.shape[1]!=len(boxes):
        raise ValueError('SAM_multi_prompt_shape_incompatible: '
            +str({'boxes':len(boxes),'mask_shape':tuple(masks[0].shape),
                  'score_shape':tuple(scores.shape)}))
    result=[]
    for i in range(len(boxes)):
        c=masks[0][i];s=scores[0][i]
        if c.shape[0]!=s.shape[0]:raise ValueError('SAM candidate count mismatch')
        idx=int(torch.argmax(s).item())
        result.append((c[idx].numpy().astype(bool),float(s[idx].item())))
    return result


def timed_call(backend, fn, *args):
    backend.synchronize()
    start=time.perf_counter()
    val=fn(backend,*args)
    backend.synchronize()
    return val, (time.perf_counter()-start)*1000


def run_diag(backend, selection, out, warmups, passes, max_cases,max_boxes):
    cases, skipped=make_cases(backend,selection,max_cases,max_boxes)
    h.write_json(out/'proposal_selection.json',
        {'eligible_cases':len(cases),'skipped':skipped,
         'cases':[{'case_id':c['case_id'],'roi_category':c['roi_category'],
                   'proposals':c['proposals']} for c in cases],
         'description':'Real DINO proposals, generated outside SAM timing; not part-region GT.'})
    if not cases:
        raise RuntimeError('NO_MULTI_BOX_SAME_ROI_PROPOSALS: inspect proposal_selection.json')
    row=[]; parity=[]
    for c in cases:
        boxes=[x['box'] for x in c['proposals']]
        for i in range(warmups):
            # Separate model initialization/caching effects from timed passes.
            serial_sam(backend,c['crop'],boxes)
            batch_sam(backend,c['crop'],boxes)
        for rep in range(1,passes+1):
            if rep%2:
                s,ts=timed_call(backend,serial_sam,c['crop'],boxes)
                b,tb=timed_call(backend,batch_sam,c['crop'],boxes)
            else:
                b,tb=timed_call(backend,batch_sam,c['crop'],boxes)
                s,ts=timed_call(backend,serial_sam,c['crop'],boxes)
            row.append({'case_id':c['case_id'],'pass':rep,'num_boxes':len(boxes),
                       'serial_total_ms':round(ts,4),'batch_total_ms':round(tb,4),
                       'serial_per_box_ms':round(ts/len(boxes),4),
                       'batch_amortized_per_box_ms':round(tb/len(boxes),4),
                       'batch_speedup':round(ts/tb,4) if tb>0 else None})
            if rep==1:
                for i,(sr,br) in enumerate(zip(s,b)):
                    parity.append({'case_id':c['case_id'],'region':c['proposals'][i]['region'],
                                   'sam_box':json.dumps(boxes[i]),
                                   'mask_iou_serial_vs_batch':safe_mask_iou(sr[0],br[0]),
                                   'score_abs_difference':abs(sr[1]-br[1]),
                                   'serial_score':sr[1],'batch_score':br[1],
                                   'mask_shape_serial':str(sr[0].shape),
                                   'mask_shape_batch':str(br[0].shape)})
    h.write_csv(out/'sam_serial_vs_batch_timing.csv',row)
    h.write_csv(out/'sam_serial_vs_batch_parity.csv',parity)
    means=lambda k:round(statistics.mean(float(t[k]) for t in row),4)
    parity_iou=[p['mask_iou_serial_vs_batch'] for p in parity]
    usable=bool(parity) and all(v is not None and v>=0.995 for v in parity_iou)
    result={'status':'STAGE8_SAM_BATCH_FEASIBILITY_COMPLETED','cases':len(cases),
        'box_prompts_compared':len(parity),'timed_rows':len(row),
        'mean_serial_total_ms':means('serial_total_ms'),
        'mean_batch_total_ms':means('batch_total_ms'),
        'mean_serial_per_box_ms':means('serial_per_box_ms'),
        'mean_batch_amortized_per_box_ms':means('batch_amortized_per_box_ms'),
        'mean_speedup_ratio':means('batch_speedup'),
        'min_mask_iou':min(parity_iou),
        'max_score_abs_diff':max(p['score_abs_difference'] for p in parity),
        'parity_screen_0_995':('PASS_DIAGNOSTIC_ONLY' if usable else 'FAIL_DO_NOT_INTEGRATE'),
        'scope':'same-image multiple-box SAM throughput, DINO not included; no full per-query latency claim',
        'limitations':['No changes to app pipeline or model weights.','No comparison with human local-part GT.',
         'Batch amortized latency only applies when multiple distinct proposals for same ROI are available.',
         'Image input is reprocessed within each method; runs alternate order to mitigate ordering effects.',
         'Full pipeline still invokes DINO for each query; 30ms per-query target not achieved by this test.']}
    h.write_json(out/'sam_batch_summary.json',result)
    return result


def pack(out):
    dest=out.parent/f'prd31_stage8_{out.name}_UPLOAD_THIS.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file(): z.write(f,f.relative_to(out))
    with zipfile.ZipFile(dest) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('ZIP_BAD:'+str(bad))
    return dest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--max-cases',type=int,default=2)
    p.add_argument('--max-boxes',type=int,default=4)
    p.add_argument('--warmup-runs',type=int,default=10)
    p.add_argument('--timed-passes',type=int,default=5)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--out-root',type=Path,default=OUT_BASE)
    a=p.parse_args()
    if a.max_cases<1 or a.max_boxes<2 or a.warmup_runs<2 or a.timed_passes<1:
        p.error('invalid caps/warmups/passes')
    root=a.root.resolve(); outroot=a.out_root.resolve()
    if root==outroot or root in outroot.parents or outroot==root.parent:
        p.error('output directory must not reside in project tree')
    out=h.new_out(outroot)
    audit={'status':'STARTED','project_files_modified':False,'scope':'GPU diagnostic ONLY; no production changes'}
    exitcode=0
    try:
        os.environ.setdefault('HF_HOME','/workspace/hf_cache')
        os.environ.setdefault('HF_HUB_CACHE','/workspace/hf_cache/hub')
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        sys.path.insert(0,str(root/'src'))
        config,selection=h.check_inputs(root,out)
        if a.dry_run:
            audit['status']='DRY_RUN_PASS'
        else:
            audit['checkpoint']=h.inspect_checkpoint(root)
            from fashion_multimodal_analysis.integration.model_backends import ModelBackends
            backend=ModelBackends(config,device='cuda')
            audit['result']=run_diag(backend,selection,out,a.warmup_runs,a.timed_passes,a.max_cases,a.max_boxes)
            audit['status']=audit['result']['status']
    except Exception as e:
        audit['status']='STAGE8_FAILED'
        audit['error']=f'{type(e).__name__}: {e}'
        (out/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('ERROR:',audit['error'],flush=True)
        exitcode=2
    finally:
        h.write_json(out/'run_audit.json',audit)
        try:
            zp=pack(out)
            print('STATUS:',audit['status'])
            if a.dry_run:print('DRY_RUN_PASS' if audit['status']=='DRY_RUN_PASS' else 'DRY_RUN_FAILED')
            print('UPLOAD_THIS_ZIP:',zp)
            print('PROJECT_FILES_MODIFIED: False')
        except Exception as e:
            exitcode=3
            print('PACK_FAILED:',e)
    return exitcode

if __name__=='__main__':sys.exit(main())
