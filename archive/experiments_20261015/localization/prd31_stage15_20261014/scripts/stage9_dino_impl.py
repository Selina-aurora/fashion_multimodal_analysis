#!/usr/bin/env python3
"""Stage9 read-only DINO: same predicted garment ROI, serial queries vs query batches.

This is a feasibility check, not an application patch, local-part GT, or latency acceptance.
All output paths are external to the project. Never change the frozen data.
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

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import stage6_helpers as h
ROOT=Path('/workspace/fashion_multimodal_analysis')
OUTBASE=Path('/workspace/prd31_stage9_runs_20261014')


def serialize_dino(processor, outputs, input_ids, size, box_t, text_t, count, device):
    """Normalize one batch's model outputs, retaining original query identity."""
    import torch
    sizes=torch.tensor([[size[1],size[0]] for _ in range(count)],device=device)
    try:
        results=processor.post_process_grounded_object_detection(
            outputs,input_ids,box_threshold=box_t,text_threshold=text_t,target_sizes=sizes)
    except TypeError:
        results=processor.post_process_grounded_object_detection(
            outputs,input_ids,threshold=box_t,text_threshold=text_t,target_sizes=sizes)
    if len(results)!=count:
        raise RuntimeError('Postprocessed result count mismatch')
    formatted=[]
    for item in results:
        def convert(v):
            if torch.is_tensor(v):return v.detach().cpu().tolist()
            return v
        boxes=convert(item.get('boxes',[]));scores=convert(item.get('scores',[]))
        labels=convert(item.get('text_labels',item.get('labels',[])))
        if len(boxes)!=len(scores):raise ValueError('DINO box/score lengths mismatch')
        det=[]
        for j,b in enumerate(boxes):
            coords=list(map(float,b));score=float(scores[j]);
            if len(coords)!=4 or not all(math.isfinite(v) for v in coords) or not math.isfinite(score):
                raise ValueError('Invalid DINO candidate')
            det.append({'box':coords,'score':score,'label':str(labels[j]) if j<len(labels) else ''})
        det.sort(key=lambda d:d['score'],reverse=True)
        formatted.append(det)
    return formatted


def predict_dino(backend, crop, phrases, batch):
    """Measure identical API preprocessing, forward and postprocess for each variant."""
    import torch
    proc=backend.ground_processor
    texts=[p.strip().rstrip('.')+'.' for p in phrases]
    images=[crop]*len(texts)
    if batch:
        inp=proc(images=images,text=texts,return_tensors='pt',padding=True)
        inp={k:v.to(backend.device) if torch.is_tensor(v) else v for k,v in inp.items()}
        with torch.inference_mode():outs=backend.ground_model(**inp)
        return serialize_dino(proc,outs,inp['input_ids'],crop.size,
                backend.config['grounding_box_threshold'],backend.config['grounding_text_threshold'],
                len(texts),backend.device)
    output=[]
    for text in texts:
        inp=proc(images=crop,text=text,return_tensors='pt')
        inp={k:v.to(backend.device) if torch.is_tensor(v) else v for k,v in inp.items()}
        with torch.inference_mode():outs=backend.ground_model(**inp)
        output.extend(serialize_dino(proc,outs,inp['input_ids'],crop.size,
                backend.config['grounding_box_threshold'],backend.config['grounding_text_threshold'],
                1,backend.device))
    return output


def elapsed_call(backend, fn, *args):
    backend.synchronize();t=time.perf_counter();res=fn(*args);backend.synchronize()
    return res,(time.perf_counter()-t)*1000


def match_outputs(serial,batched):
    """Strictly compare per-query candidates; do not alter the proposal ordering."""
    rows=[]
    if len(serial)!=len(batched):raise ValueError('Query length mismatch')
    for a,b in zip(serial,batched):
        count_a=len(a);count_b=len(b);n=min(count_a,count_b)
        box_diff=max((max(abs(x-y) for x,y in zip(a[i]['box'],b[i]['box'])) for i in range(n)),default=0.0)
        score_diff=max((abs(a[i]['score']-b[i]['score']) for i in range(n)),default=0.0)
        labels_equal=all(a[i]['label']==b[i]['label'] for i in range(n))
        # Even when neither produced candidates, count-equivalence is recorded explicitly.
        row={'serial_candidates':count_a,'batch_candidates':count_b,'matching_candidate_count':count_a==count_b,
             'max_box_abs_px':box_diff,'max_score_abs':score_diff,'labels_equal':labels_equal,
             'exact_zero_candidate_both':count_a==0 and count_b==0}
        row['strict_parity']=row['matching_candidate_count'] and row['labels_equal'] and box_diff<=0.5 and score_diff<=0.005
        rows.append(row)
    return rows


def pick_cases(backend,selection,max_cases,queries):
    from PIL import Image
    from fashion_multimodal_analysis.integration.pipeline import roi_images
    from fashion_multimodal_analysis.benchmarking.prepare_prd31_recovery import PROMPTS
    out=[];skip=[]
    for index,entry in enumerate(selection,1):
        if len(out)>=max_cases:break
        with Image.open(entry['source_image']) as f:image=f.convert('RGB')
        rois,_=backend.segment(image)
        if not rois:
            skip.append({'index':index,'reason':'no_predicted_garment'});continue
        roi=sorted(rois,key=h.preferred)[0]
        crop,mask,_=roi_images(image,roi)
        isolated=Image.composite(crop,Image.new('RGB',crop.size,'white'),mask)
        regions=h.allowed_regions(roi.category)[:queries]
        if len(regions)<2:
            skip.append({'index':index,'reason':'fewer_than_two_applicable_regions'});continue
        out.append({'id':f'case_{index:02d}','category':roi.category,'image':isolated,
                    'regions':regions,'prompts':[PROMPTS[r] for r in regions],
                    'size':list(isolated.size)})
    return out,skip


def run_benchmark(backend,selection,out,max_cases,queries,warmups,passes):
    cases,skipped=pick_cases(backend,selection,max_cases,queries)
    h.write_json(out/'case_manifest.json',{
      'cases':[{'case':c['id'],'category':c['category'],'regions':c['regions'],'crop_size':c['size']} for c in cases],
      'skipped':skipped,'notes':'Predicted garment ROIs; identical isolated crop per query within each case; not human local-part GT.'})
    if not cases:raise RuntimeError('No eligible cases with >=2 queries')
    timings=[];parity=[]
    for case in cases:
        for w in range(warmups):
            predict_dino(backend,case['image'],case['prompts'],batch=False)
            predict_dino(backend,case['image'],case['prompts'],batch=True)
        for k in range(1,passes+1):
            if k%2:
                serial,ts=elapsed_call(backend,predict_dino,backend,case['image'],case['prompts'],False)
                batch,tb=elapsed_call(backend,predict_dino,backend,case['image'],case['prompts'],True)
            else:
                batch,tb=elapsed_call(backend,predict_dino,backend,case['image'],case['prompts'],True)
                serial,ts=elapsed_call(backend,predict_dino,backend,case['image'],case['prompts'],False)
            n=len(case['regions'])
            timings.append({'case':case['id'],'pass':k,'queries':n,'serial_total_ms':round(ts,4),
                'batch_total_ms':round(tb,4),'serial_per_query_ms':round(ts/n,4),
                'batch_amortized_per_query_ms':round(tb/n,4),'serial_batch_speedup':round(ts/tb,4) if tb>0 else None})
            if k==1:
                rows=match_outputs(serial,batch)
                for region,s in zip(case['regions'],rows):
                    parity.append({'case':case['id'],'region':region,**s})
    h.write_csv(out/'dino_query_serial_vs_batch_timing.csv',timings)
    h.write_csv(out/'dino_query_serial_vs_batch_parity.csv',parity)
    means=lambda field:round(statistics.mean(float(row[field]) for row in timings),4)
    all_parity=all(bool(r['strict_parity']) for r in parity)
    summary={'status':'STAGE9_DINO_BATCH_DIAGNOSTIC_COMPLETED','cases':len(cases),
       'query_pairs_compared':len(parity),'timed_rows':len(timings),
       'mean_serial_total_ms':means('serial_total_ms'),
       'mean_batch_total_ms':means('batch_total_ms'),
       'mean_serial_per_query_ms':means('serial_per_query_ms'),
       'mean_batch_amortized_per_query_ms':means('batch_amortized_per_query_ms'),
       'mean_speedup_ratio':means('serial_batch_speedup'),
       'strict_candidate_parity':all_parity,
       'parity_failures':sum(not r['strict_parity'] for r in parity),
       'limitations':['No SAM calls included in timed DINO variants.',
            'Batch duplicates the same image, which need not reuse the image encoder.',
            'Results on diagnostic images are not held-out semantic localization accuracy.',
            'Batch GPU memory and scheduling may differ from production.','No PRD 30ms acceptance claim.'],
       'integration_status':'NOT_AUTHORIZED_EXPLORATORY' if all_parity else 'DO_NOT_INTEGRATE_PARITY_FAILED'}
    h.write_json(out/'dino_batch_summary.json',summary)
    return summary


def pack(out):
    dest=out.parent/f'prd31_stage9_{out.name}_UPLOAD_THIS.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out))
    with zipfile.ZipFile(dest) as z:
        bad=z.testzip()
        if bad:raise RuntimeError(f'Corrupt ZIP {bad}')
    return dest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--max-cases',type=int,default=4)
    p.add_argument('--query-count',type=int,default=4)
    p.add_argument('--warmup-runs',type=int,default=5)
    p.add_argument('--timed-passes',type=int,default=5)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--out-root',type=Path,default=OUTBASE)
    a=p.parse_args()
    if not(1<=a.max_cases<=8 and 2<=a.query_count<=8 and a.warmup_runs>=2 and a.timed_passes>=1):
        p.error('Invalid case/query/warmup/pass bounds')
    root=a.root.resolve();outroot=a.out_root.resolve()
    if root==outroot or root in outroot.parents or outroot==root.parent:
        p.error('Output root must be outside project')
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
            audit['result']=run_benchmark(backend,selection,out,a.max_cases,a.query_count,a.warmup_runs,a.timed_passes)
            audit['status']=audit['result']['status']
    except Exception as exc:
        audit['status']='STAGE9_FAILED'
        audit['error']=type(exc).__name__+': '+str(exc)
        (out/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('ERROR:',audit['error'],flush=True)
        code=2
    finally:
        h.write_json(out/'run_audit.json',audit)
        try:
            z=pack(out)
            print('STATUS:',audit['status'])
            if a.dry_run: print('DRY_RUN_PASS' if audit['status']=='DRY_RUN_PASS' else 'DRY_RUN_FAILED')
            print('UPLOAD_THIS_ZIP:',z)
            print('PROJECT_FILES_MODIFIED: False')
        except Exception as e:
            print('PACK_FAILED:',e)
            code=3
    return code

if __name__=='__main__':sys.exit(main())
