#!/usr/bin/env python3
"""Stage13: production-protocol matched A/B, timed without extra native DINO calls.

Reproduce strict ModelBackends.ground semantics: raw image, REGION_PROMPTS,
phrase gate, geometric rejection, local-mask constraint. Never patch project.
Invoking script supplies selected images; NOT formal precision acceptance.
"""
from __future__ import annotations
import argparse, json, math, os, sys, statistics, time, traceback, zipfile
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import stage6_helpers as h
import stage9_dino_impl as dino
import stage8_sam_impl as sam
ROOT=Path('/workspace/fashion_multimodal_analysis')
OUTROOT=Path('/workspace/prd31_stage13_runs_20261014')

def det_parity(a,b):
    """Compare ranked DINO candidates; use matched list indices and tolerances."""
    n=min(len(a),len(b))
    db=max((max(abs(float(v)-float(w)) for v,w in zip(a[i]['box'],b[i]['box'])) for i in range(n)),default=0.)
    ds=max((abs(float(a[i]['score'])-float(b[i]['score'])) for i in range(n)),default=0.)
    labels=all(a[i]['label']==b[i]['label'] for i in range(n))
    return dict(serial_n=len(a),batch_n=len(b),max_box_difference_px=db,
                max_score_difference=ds,labels_equal=labels,
                pass_screen=(len(a)==len(b) and db<=0.5 and ds<=0.005 and labels))

def gate_proposals(backend,mb,roi,region,detections,crop_size):
    """Mirror GPU runtime ground() strict routing and DINO candidate gates."""
    from fashion_multimodal_analysis.integration.pipeline import crop_bounds
    from fashion_multimodal_analysis.evaluation.protocol import valid_box
    strict=backend.config.get('strict_region_grounding',False)
    if strict and not mb.supports_region(roi.category,region):
        return {'status':'not_applicable','proposal':None,'box':None,'rejections':[]}
    filtered=[x for x in detections if valid_box(x['box']) and math.isfinite(float(x['score']))]
    if strict:
        rejected=[x for x in filtered if not mb.phrase_matches(region,x['label'])]
        filtered=[x for x in filtered if mb.phrase_matches(region,x['label'])]
        if not filtered and rejected:
            return {'status':'rejected_wrong_phrase','proposal':None,'box':None,'rejections':rejected}
    if not filtered:
        return {'status':'missed','proposal':None,'box':None,'rejections':[]}
    filtered.sort(key=lambda x:-float(x['score']))
    limit=float(backend.config.get('max_local_region_crop_area_ratio',0.65))
    if not 0<limit<=1:raise ValueError('Invalid production area-ratio gate')
    rejected=[]
    for x in filtered:
        try:box=crop_bounds(x['box'],crop_size)
        except ValueError:
            rejected.append({'reason':'outside_crop','proposal':x});continue
        ratio=mb.proposal_area_ratio(box,crop_size)
        if strict and region in mb.LOCAL_REGIONS and ratio>limit:
            rejected.append({'reason':'local_region_covers_whole_garment','ratio':float(ratio),'proposal':x});continue
        return {'status':'ready_for_sam','proposal':x,'box':box,'rejections':rejected}
    return {'status':'rejected_region_geometry','proposal':None,'box':None,'rejections':rejected}

def choose_cases(backend,selection,max_cases,query_count):
    from PIL import Image
    from fashion_multimodal_analysis.integration.pipeline import roi_images
    from fashion_multimodal_analysis import integration
    cases=[];skipped=[]
    for j,record in enumerate(selection,1):
        if len(cases)>=max_cases:break
        with Image.open(record['source_image']) as f:image=f.convert('RGB')
        rois,_=backend.segment(image)
        if not rois:skipped.append({'case':j,'why':'no_predicted_garment'});continue
        roi=sorted(rois,key=h.preferred)[0]
        regions=h.allowed_regions(roi.category)[:query_count]
        if len(regions)<2:skipped.append({'case':j,'why':'fewer_than_two_regions'});continue
        crop,mask,origin=roi_images(image,roi)
        cases.append({'case':f'case_{j:02d}','image':image,'roi':roi,'regions':regions,
                      'crop':crop,'garment_mask':mask,'origin':origin})
    return cases,skipped

def read_only_batch(backend,mb,c,verify_native=False):
    """Same strict production gates; optionally validate native DINO OUTSIDE timed region."""
    import numpy as np
    from PIL import Image
    from fashion_multimodal_analysis.integration.pipeline import crop_bounds
    from fashion_multimodal_analysis.evaluation.protocol import original_box
    strict=backend.config.get('strict_region_grounding',False)
    crop=c['crop'];mask=c['garment_mask'];origin=c['origin']
    dino_input=crop if strict else Image.composite(crop,Image.new('RGB',crop.size,'white'),mask)
    regions=c['regions'];prompts=[mb.REGION_PROMPTS[r] if strict else mb.PROMPTS[r] for r in regions]
    batch=dino.predict_dino(backend,dino_input,prompts,batch=True)
    if len(batch)!=len(regions):raise ValueError('Invalid batched DINO outputs')
    native_rows=[];answers=[None]*len(regions);needs_sam=[];boxes=[];checked=[]
    # Independently verify batch candidate parity on same production image/text.
    from fashion_multimodal_analysis.grounding.refinement.run_shoulder_waist_pattern_groundingdino_refinement_v1 import run_grounding_dino
    for i,(r,txt,bat) in enumerate(zip(regions,prompts,batch)):
        if verify_native:
            native,_=run_grounding_dino(dino_input,txt,
                backend.config['grounding_box_threshold'],backend.config['grounding_text_threshold'],
                backend.ground_processor,backend.ground_model,backend.device)
            par=det_parity(native,bat)
            native_rows.append({'case':c['case'],'region':r,**par,
                               'production_image_view':'raw_crop' if strict else 'masked_crop',
                               'production_prompt':txt,
                               'native_top':json.dumps(native[0] if native else None),
                               'batch_top':json.dumps(bat[0] if bat else None)})
            if not par['pass_screen']:
                raise RuntimeError(f"PRODUCTION_DINO_BATCH_MISMATCH: {c['case']} {r} - do not integrate")
        selected=gate_proposals(backend,mb,c['roi'],r,bat,crop.size)
        if selected['status']!='ready_for_sam':
            answers[i]={'status':selected['status'],'target_region':r,'bbox':None,'mask':None}
        else:
            boxes.append(selected['box']);needs_sam.append((i,r,selected))
    if boxes:
        masks=sam.batch_sam(backend,crop,boxes)
        if len(masks)!=len(boxes):raise ValueError('SAM batch count incorrect')
        garment_array=np.asarray(mask)>0
        for (i,r,selected),(raw_mask,sam_score) in zip(needs_sam,masks):
            raw=np.asarray(raw_mask,dtype=bool)
            if raw.shape!=garment_array.shape:raise ValueError('SAM mask dimensions mismatch')
            current=raw & garment_array
            if strict and r in mb.LOCAL_REGIONS:
                current=mb.constrain_local_mask(raw,mask,selected['box'])
            ys,xs=np.nonzero(current)
            if not len(xs):
                answers[i]={'status':'empty_mask','target_region':r,'bbox':None,'mask':None};continue
            refined=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
            full=np.zeros((c['image'].height,c['image'].width),dtype=np.uint8)
            x1,y1,x2,y2=origin
            if current.shape!=(y2-y1,x2-x1):raise ValueError('Mask/crop shape mismatch')
            full[y1:y2,x1:x2]=current
            answers[i]={'status':'detected','target_region':r,
                        'model_phrase':selected['proposal']['label'],
                        'score':float(selected['proposal']['score']),
                        'bbox':original_box(refined,origin),'mask':full,
                        'proposal_bbox':original_box(selected['box'],origin),
                        'sam_score':float(sam_score),
                        'coordinate_space':'original_image','crop_origin':origin}
    if any(a is None for a in answers):raise RuntimeError('Unassigned batch result')
    return answers,native_rows

def output_compare(case,r,a,b):
    import numpy as np
    stat_a=a.get('status'); stat_b=b.get('status')
    row={'case':case,'region':r,'production_status':stat_a,'batch_status':stat_b,
         'status_equal':stat_a==stat_b,'mask_iou':None,'mask_pixel_equal':None,
         'bbox_difference_px':None,'proposal_box_difference_px':None,
         'score_difference':None,'sam_score_difference':None,'phrase_equal':None}
    passed=stat_a==stat_b
    if stat_a=='detected' and stat_b=='detected':
        am=np.asarray(a['mask'])>0;bm=np.asarray(b['mask'])>0
        if am.shape==bm.shape:
            union=int(np.count_nonzero(am|bm));row['mask_iou']=float(np.count_nonzero(am&bm)/union) if union else 1.
            row['mask_pixel_equal']=bool(np.array_equal(am,bm))
        for field,target in [('bbox','bbox_difference_px'),('proposal_bbox','proposal_box_difference_px')]:
            x=a.get(field);y=b.get(field)
            if x and y:row[target]=max(abs(float(m)-float(n)) for m,n in zip(x,y))
        for field,target in [('score','score_difference'),('sam_score','sam_score_difference')]:
            if field in a and field in b:row[target]=abs(float(a[field])-float(b[field]))
        row['phrase_equal']=a.get('model_phrase')==b.get('model_phrase')
        passed=passed and all([row['mask_iou'] is not None and row['mask_iou']>=0.999,
                               row['bbox_difference_px'] is not None and row['bbox_difference_px']<=1.0,
                               row['proposal_box_difference_px'] is not None and row['proposal_box_difference_px']<=1.0,
                               row['score_difference'] is not None and row['score_difference']<=0.005,
                               row['sam_score_difference'] is not None and row['sam_score_difference']<=0.005,
                               row['phrase_equal']])
    row['parity_screen_pass']=bool(passed)
    return row

def measure(backend,call):
    backend.synchronize();t=time.perf_counter();ret=call();backend.synchronize()
    return ret,(time.perf_counter()-t)*1000.

def run(backend,cfg,sel,out,max_cases,query_count,warmups,passes):
    from fashion_multimodal_analysis.integration import model_backends as mb
    required=['supports_region','phrase_matches','proposal_area_ratio','LOCAL_REGIONS','constrain_local_mask','REGION_PROMPTS','PROMPTS']
    missing=[name for name in required if not hasattr(mb,name)]
    if missing:raise RuntimeError('RUNTIME_PRODUCTION_HELPERS_MISSING: '+','.join(missing))
    if not bool(backend.config.get('strict_region_grounding')):raise ValueError('Strict production mode not enabled; abort')
    cases,skips=choose_cases(backend,sel,max_cases,query_count)
    h.write_json(out/'selection.json',{'cases':[{'id':x['case'],'class':x['roi'].category,
                       'crop_size':x['crop'].size,'regions':x['regions']} for x in cases],'skipped':skips})
    if not cases:raise RuntimeError('No diagnostic garment regions available')
    dino_rows=[];rows=[];times=[];repeat_parity=[]
    for c in cases:
        print('CHECK',c['case'],c['roi'].category,c['regions'],flush=True)
        for _ in range(warmups):
            for r in c['regions']:
                backend.ground(c['image'],c['roi'],r)
            read_only_batch(backend,mb,c,verify_native=False)
        # Verify native single-query DINO and full production parity OFF the stopwatch.
        reference=[backend.ground(c['image'],c['roi'],r)[0] for r in c['regions']]
        batched, native_verification=read_only_batch(backend,mb,c,verify_native=True)
        dino_rows.extend(native_verification)
        rows.extend(output_compare(c['case'],r,a,b) for r,a,b in zip(c['regions'],reference,batched))
        for rep in range(1,passes+1):
            def prod():return [backend.ground(c['image'],c['roi'],r)[0] for r in c['regions']]
            def opt():return read_only_batch(backend,mb,c,verify_native=False)[0]
            if rep%2:
                a,ta=measure(backend,prod); b,tb=measure(backend,opt)
            else:
                b,tb=measure(backend,opt); a,ta=measure(backend,prod)
            times.append({'case':c['case'],'pass':rep,'query_count':len(c['regions']),
                          'production_group_ms':round(ta,4),'batch_group_ms':round(tb,4),
                          'production_amortized_ms_per_query':round(ta/len(c['regions']),4),
                          'batch_amortized_ms_per_query':round(tb/len(c['regions']),4),
                          'ratio_production_over_batch':round(ta/tb,4) if tb>0 else None,
                          'timing_note':'Both arms include preprocessing, DINO, SAM and postprocessing; native parity check outside timer'})
            repeat_parity.extend({'pass':rep,**output_compare(c['case'],r,x,y)} for r,x,y in zip(c['regions'],a,b))
    h.write_csv(out/'strict_dino_parity.csv',dino_rows)
    h.write_csv(out/'final_region_parity.csv',rows)
    h.write_csv(out/'repeat_region_parity.csv',repeat_parity)
    h.write_csv(out/'optimized_group_timing.csv',times)
    import statistics
    sm=lambda k: round(statistics.mean(float(row[k]) for row in times),4)
    detected=[r for r in rows if r['production_status']=='detected' and r['batch_status']=='detected']
    all_screen=all(bool(x['parity_screen_pass']) for x in rows+repeat_parity)
    summary={'status':'STAGE13_STRICT_BATCH_TIMING_COMPLETED',
             'cases':len(cases),'queries':len(rows),'warmups':warmups,'timed_passes':passes,
             'timed_group_count':len(times),'strict_grounding':True,
             'native_dino_batch_parity_passes':sum(bool(x['pass_screen']) for x in dino_rows),
             'final_region_parity_passes':sum(bool(x['parity_screen_pass']) for x in rows),
             'repeat_region_parity_passes':sum(bool(x['parity_screen_pass']) for x in repeat_parity),
             'repeat_region_parity_total':len(repeat_parity),
             'detected_production':sum(x['production_status']=='detected' for x in rows),
             'detected_batch':sum(x['batch_status']=='detected' for x in rows),
             'detected_mask_pixel_equal':sum(x['mask_pixel_equal'] is True for x in detected),
             'detected_both_count':len(detected),
             'min_detected_mask_iou':min((float(x['mask_iou']) for x in detected),default=None),
             'mean_production_group_ms':sm('production_group_ms'),
             'mean_batch_group_ms':sm('batch_group_ms'),
             'mean_production_amortized_ms_per_query':sm('production_amortized_ms_per_query'),
             'mean_batch_amortized_ms_per_query':sm('batch_amortized_ms_per_query'),
             'mean_group_speedup_ratio':sm('ratio_production_over_batch'),
             'ratio_of_mean_group_times':round(sm('production_group_ms')/sm('batch_group_ms'),4),
             'timing_valid_for_speedup':True,
             'integration_decision':('DIAGNOSTIC_ONLY_MORE_VALIDATION_REQUIRED' if all_screen else 'DO_NOT_INTEGRATE_PARITY_MISMATCH'),
             'notes':['A/B uses production strict raw crop, REGION_PROMPTS and all gates.',
                     'Native DINO candidate parity is checked outside both timed regions.',
                     'Both measured branches include complete region postprocessing; model and project untouched.',
                     'Batch speedup applies to grouped requests, not single-query PRD 30ms criterion.',
                     'Development image selection is recorded separately by the invoking runner; no human semantic/part GT.',
                     'Pass thresholds permit minor mask differences; pixel-identical count reported separately.',
                     'No formal 3.1.2 localization accuracy or 30ms acceptance.']}
    h.write_json(out/'stage13_summary.json',summary)
    return summary

def pack(out):
    zpath=out.parent/f'prd31_stage13_{out.name}_UPLOAD_THIS.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(out))
    with zipfile.ZipFile(zpath) as z:
        if z.testzip():raise ValueError('Corrupt result ZIP')
    return zpath

def main():
    p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true')
    p.add_argument('--max-cases',type=int,default=8);p.add_argument('--query-count',type=int,default=4)
    p.add_argument('--warmups',type=int,default=3);p.add_argument('--passes',type=int,default=5)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--out-root',type=Path,default=OUTROOT)
    a=p.parse_args()
    if not(1<=a.max_cases<=8 and 2<=a.query_count<=8 and 0<=a.warmups<=10 and 1<=a.passes<=10):p.error('Bad sample settings')
    root=a.root.resolve();out_root=a.out_root.resolve()
    if root==out_root or root in out_root.parents or out_root==root.parent:p.error('Output must stay outside repo')
    out=h.new_out(out_root);audit={'status':'STARTED','project_files_modified':False}
    rc=0
    try:
        os.environ.setdefault('HF_HOME','/workspace/hf_cache')
        os.environ.setdefault('HF_HUB_CACHE','/workspace/hf_cache/hub')
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        sys.path.insert(0,str(root/'src'))
        cfg,sel=h.check_inputs(root,out)
        if a.dry_run:audit['status']='DRY_RUN_PASS'
        else:
            audit['checkpoint']=h.inspect_checkpoint(root)
            from fashion_multimodal_analysis.integration.model_backends import ModelBackends
            backend=ModelBackends(cfg,device='cuda')
            audit['result']=run(backend,cfg,sel,out,a.max_cases,a.query_count,a.warmups,a.passes)
            audit['status']=audit['result']['status']
    except Exception as exc:
        audit['status']='STAGE13_FAILED';audit['error']=type(exc).__name__+': '+str(exc)
        (out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf8')
        print('ERROR:',audit['error'],flush=True);rc=2
    finally:
        h.write_json(out/'run_audit.json',audit)
        try:
            dest=pack(out)
            print('STATUS:',audit['status'],flush=True)
            if a.dry_run:print('DRY_RUN_PASS' if audit['status']=='DRY_RUN_PASS' else 'DRY_RUN_FAILED')
            print('UPLOAD_THIS_ZIP:',dest,flush=True)
            print('PROJECT_FILES_MODIFIED: False')
        except Exception as exc:
            print('ZIP_PACK_FAILED:',repr(exc));rc=3
    return rc
if __name__=='__main__':sys.exit(main())
