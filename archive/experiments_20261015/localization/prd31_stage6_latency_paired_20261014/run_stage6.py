#!/usr/bin/env python3
"""Read-only warmed module timing + GT/pred garment-ROI perturbation diagnostic.

This is NOT PRD acceptance, regional ground truth evaluation or attribute accuracy.
New outputs are written outside the source repository only.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT_DEFAULT = Path('/workspace/fashion_multimodal_analysis')
OUT_DEFAULT = Path('/workspace/prd31_stage6_20261014_runs')
CKPT = 'outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth'
TRAIN = 'configs/prd_8class_train_v5_balanced150_20261008.csv'
VAL = 'configs/prd_8class_val_v2_full_targets.csv'
CORE = 'benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv'
EXPECTED_SHA = {
    TRAIN:'e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f',
    VAL:'39141f545ea5512d16ed074788b6f98474b33faff111d3b663ec0ffcc4c40b93',
    CORE:'c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89',
}
CLASSES = {'top':1,'pants':2,'skirt':3,'outerwear':4,'dress':5,'shoe':6,'bag':7,'accessory':8}
UPPER={'top','outerwear','dress'}
CLOTHING=UPPER|{'pants','skirt'}
ALL_REGIONS=['collar','cuff','hem','pocket','shoulder','waist','pattern','decoration']


def filehash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def write_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')


def write_csv(p,rows,fields=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(rows[0]) if rows else fields
    if not fields: fields=['status']
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def read_csv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def new_out(root):
    root.mkdir(parents=True,exist_ok=True)
    stamp=time.strftime('%Y%m%d_%H%M%S')
    for i in range(100):
        p=root/(f'run_{stamp}' if i==0 else f'run_{stamp}_{i:02d}')
        try:p.mkdir();return p
        except FileExistsError:pass
    raise RuntimeError('no unique output directory')


def pack(out):
    dest=out.parent/f'prd31_stage6_{out.name}_UPLOAD_THIS.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(out))
    with zipfile.ZipFile(dest) as z:
        if z.testzip():raise ValueError('Output ZIP CRC failed')
    return dest


def check_inputs(root,out):
    verified={}
    for rel, expected in EXPECTED_SHA.items():
        path=root/rel
        if not path.is_file():raise FileNotFoundError(f'Missing protected input: {path}')
        actual=filehash(path)
        verified[rel]={'sha256':actual,'expected_sha256':expected,'match':actual==expected}
        if actual!=expected:raise ValueError(f'Protected input changed: {rel}')
    if not (root/CKPT).is_file():raise FileNotFoundError(str(root/CKPT))
    config=json.loads((BASE/'run_config.json').read_text(encoding='utf-8'))
    config.update(segmentation_checkpoint=CKPT,train_manifest=TRAIN,val_manifest=VAL)
    config['checkpoint_settings']={
        'batch_size':1,'workers':2,'min_size':640,'max_size':1024,
        'seed':20260921,'lr':0.0025,'momentum':0.9,'weight_decay':0.0005,
        'step_size':5,'gamma':0.1,'sampler_alpha':0.5,
        'no_hflip':False,'no_balanced_sampler':False,
    }
    if config.get('required_regions')!=ALL_REGIONS:raise ValueError('Unexpected required regions')
    if len(read_csv(root/CORE))!=400:raise ValueError('Core400 manifest changed in row count')
    selection=json.loads((BASE/'selected_images.json').read_text(encoding='utf-8'))
    if len(selection)!=8:raise ValueError('Expected fixed 8-image diagnostic manifest')
    for s in selection:
        p=Path(s['source_image'])
        if not p.is_file():raise FileNotFoundError(f'Diagnostic source missing: {p}')
        if filehash(p)!=s['source_sha256']:
            raise ValueError('Diagnostic image hash mismatch: '+str(p))
    write_json(out/'input_audit.json',{'status':'INPUTS_VERIFIED','hashes':verified,
               'diagnostic_image_count':len(selection), 'val_rows':len(read_csv(root/VAL)),
               'checkpoint_path':str(root/CKPT),
               'accuracy_acceptance':False,'latency_acceptance':False})
    return config, selection


def inspect_checkpoint(root):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA not available')
    cp=torch.load(root/CKPT,map_location='cpu',weights_only=False)
    if not isinstance(cp,dict) or cp.get('epoch')!=15 or cp.get('class_to_id')!=CLASSES:
        raise ValueError('Checkpoint not valid V5 Balanced150 epoch15/class mapping')
    if cp.get('train_manifest_sha256')!=EXPECTED_SHA[TRAIN] or cp.get('val_manifest_sha256')!=EXPECTED_SHA[VAL]:
        raise ValueError('Checkpoint provenance SHA mismatch')
    if not isinstance(cp.get('model_state_dict'),dict):raise ValueError('Checkpoint missing model weights')
    return {'device':torch.cuda.get_device_name(0),'torch':torch.__version__,
            'checkpoint_sha256':filehash(root/CKPT),
            'min_size':cp.get('args',{}).get('min_size'),
            'max_size':cp.get('args',{}).get('max_size')}


def allowed_regions(category):
    if category in UPPER:
        return ALL_REGIONS
    if category in {'pants','skirt'}:
        return ['hem','pocket','waist','pattern','decoration']
    return ['pattern','decoration']


def preferred(roi):
    priority={'top':0,'outerwear':1,'dress':2,'pants':3,'skirt':4,'bag':5,'shoe':6,'accessory':7}
    return (priority.get(roi.category,10),-roi.score)


def summary_ms(values,target=None):
    v=sorted(float(x) for x in values)
    if not v:return {'n':0,'mean_ms':None,'median_ms':None,'p95_ms':None,'max_ms':None,'exceed_target_count':None}
    import numpy as np
    return {'n':len(v),'mean_ms':round(statistics.mean(v),4),
            'median_ms':round(statistics.median(v),4),
            'p95_ms':round(float(np.percentile(v,95)),4),
            'max_ms':round(max(v),4),'threshold_ms':target,
            'exceed_target_count':sum(x>target for x in v) if target is not None else None}


def timing_phase(backend,selection,config,out, warmups,passes,per_region):
    from PIL import Image
    import torch
    source=[]
    for k,s in enumerate(selection,1):
        with Image.open(s['source_image']) as im:image=im.convert('RGB')
        source.append((f'case_{k:02d}',image,s))
    # Select predicted garment ROIs once (selection itself is not timed).
    selected=[]
    for case,image,s in source:
        rois,_=backend.segment(image)
        if rois:
            roi=sorted(rois,key=preferred)[0]
            selected.append((case,image,roi))
    if not selected:raise ValueError('No predicted ROIs available for latency sampling')
    records=[]; warm_records=[]
    # Use the SAME API and full preprocessing as the deployed integration backend.
    for j in range(warmups):
        backend.segment(source[0][1]);warm_records.append({'module':'segmentation','iteration':j+1})
    for i in range(passes):
        for case,image,s in source:
            rois,ms=backend.segment(image)
            records.append({'module':'segmentation','sample_id':case,'region':'',
                            'unit':'image','pass':i+1,'ms':ms,'status':'ran',
                            'predicted_count':len(rois),'attribute_predicted_count':''})
    candidates=defaultdict(list)
    for case,image,roi in selected:
        for region in allowed_regions(roi.category):
            candidates[region].append((case,image,roi,region))
    ground_pairs=[]
    for r in ALL_REGIONS:
        items=sorted(candidates[r],key=lambda c:c[0])
        # Avoid cherry-picking only high-confidence successful localizers.
        ground_pairs.extend(items[:per_region])
    if ground_pairs:
        case,image,roi,r=ground_pairs[0]
        for j in range(warmups):
            backend.ground(image,roi,r);warm_records.append({'module':'grounding','iteration':j+1})
        for i in range(passes):
            for case,image,roi,region in ground_pairs:
                result,ms=backend.ground(image,roi,region)
                records.append({'module':'grounding','sample_id':case+':'+roi.instance_id,
                                'region':region,'unit':'query_instance','pass':i+1,
                                'ms':ms,'status':result.get('status','unknown'),
                                'predicted_count':'','attribute_predicted_count':''})
    # Upstream collar localizations are precomputed and excluded from attribute timers.
    attribute_inputs=[]
    for case,image,roi in selected:
        region_dict={}
        if roi.category in UPPER:
            region_dict['collar'],_=backend.ground(image,roi,'collar')
        attribute_inputs.append((case,image,roi,region_dict))
    if attribute_inputs:
        case,image,roi,rd=attribute_inputs[0]
        for j in range(warmups):
            backend.attributes(image,roi,rd,config['required_attributes'],controlled=False)
            warm_records.append({'module':'attributes','iteration':j+1})
        for i in range(passes):
            for case,image,roi,rd in attribute_inputs:
                result,ms=backend.attributes(image,roi,rd,config['required_attributes'],controlled=False)
                statuses=Counter(v.get('status','unknown') for v in result.values())
                records.append({'module':'attributes','sample_id':case+':'+roi.instance_id,
                                'region':'','unit':'garment_attribute_set','pass':i+1,
                                'ms':ms,'status':';'.join(f'{k}:{v}' for k,v in sorted(statuses.items())),
                                'predicted_count':'','attribute_predicted_count':statuses.get('predicted',0)})
    write_csv(out/'warmed_timing_raw.csv',records)
    write_csv(out/'warmup_records.csv',warm_records)
    module_agg={}
    for key,target in [('segmentation',50),('grounding',30),('attributes',20)]:
        v=[x['ms'] for x in records if x['module']==key]
        module_agg[key]={**summary_ms(v,target),
             'unit':{'segmentation':'image','grounding':'query_instance','attributes':'garment_attribute_set'}[key],
             'unique_sample_ids':len({x['sample_id'] for x in records if x['module']==key}),
             'warmup_runs':warmups,'timed_passes':passes,
             'observed_statuses':dict(Counter(x['status'] for x in records if x['module']==key)),
             'scope':'WARMED_DIAGNOSTIC_NOT_FORMAL_PRD_ACCEPTANCE'}
    group=[]
    for region in ALL_REGIONS:
        v=[x['ms'] for x in records if x['module']=='grounding' and x['region']==region]
        if v:group.append({'region':region,**summary_ms(v,30)})
    write_csv(out/'grounding_latency_by_region.csv',group)
    write_json(out/'warmed_latency_summary.json',{
        'status':'WARMED_TIMING_FINISHED','notes':['Matched previous 8 image hashes, but new timed calls.',
        'Each grounding record is one (garment instance, text query), potentially DINO plus SAM.',
        'Attribute time covers the full category-applicable attribute set per garment, not one isolated attribute.',
        'Some attribute labels depend on previously precomputed collar localization; collar call is excluded from attribute timer.',
        '8-image small diagnostic set, repeated measurements are not independent examples.',
        'No product image IO/model load/warmups inside module timers; CUDA sync is in existing backend.',
        'An attribute-set timing >20ms is NOT identical to single-attribute timing; PRD timing granularity must be confirmed.',
        'No frozen human-reviewed V2, no PRD acceptance claim.'],
        'modules':module_agg,'per_region':group,
        'selected_predicted_roi_count':len(selected), 'ground_pair_count':len(ground_pairs)})
    return {'modules':module_agg,'selected_roi_count':len(selected),'ground_pair_count':len(ground_pairs)}


def val_pair_phase(backend,root,config,out,max_pairs,max_val_images,regions_per_pair):
    import numpy as np
    from PIL import Image
    from fashion_multimodal_analysis.common.paths import resolve_path
    from fashion_multimodal_analysis.integration.pipeline import GarmentROI
    from fashion_multimodal_analysis.image_processing.masks import reconstruct_full_mask
    from fashion_multimodal_analysis.evaluation.metrics import box_iou
    from fashion_multimodal_analysis.evaluation.protocol import confidence_first_match
    from fashion_multimodal_analysis.benchmarking.prepare_prd31_recovery import applies
    source_rows=read_csv(root/VAL)
    by_source=defaultdict(list)
    for r in source_rows:by_source[r['source_image']].append(r)
    candidate=[]; failures=[]
    scanned=0
    # Deterministic rotation ensures results are not selected based on agreement outcome.
    keys=sorted(by_source)
    random.Random(20261014).shuffle(keys)
    for src in keys[:max_val_images]:
        scanned+=1
        try:
            path=resolve_path(src)
            with Image.open(path) as im:image=im.convert('RGB')
            gt_rows=by_source[src]
            gb=[]
            for row in gt_rows:
                box=[float(row['gt_bbox_'+k]) for k in ('x1','y1','x2','y2')]
                gb.append(box)
            predictions,_=backend.segment(image)
            if not predictions:continue
            matched=confidence_first_match(gb,[r.bbox for r in predictions],0.50,[r.score for r in predictions])
            for gi,pi,bio in matched:
                row=gt_rows[gi]; pred=predictions[pi]
                if row['garment_category'].strip()!=pred.category:continue
                try:
                    mask_path=resolve_path(row['gt_mask_path'])
                    x1,y1,x2,y2=map(lambda v:int(float(v)),gb[gi])
                    mask=np.asarray(reconstruct_full_mask(mask_path,image.size,(x1,y1,x2,y2)))>0
                    if not mask.any():raise ValueError('Empty GT mask')
                    gt=GarmentROI('controlled_gt_'+str(row.get('garment_id','unknown')),
                                  pred.category,1.0,list(gb[gi]),mask,'controlled_gt_garment_roi')
                    mi=float(np.logical_and(mask,np.asarray(pred.mask)>0).sum())/max(1,float(np.logical_or(mask,np.asarray(pred.mask)>0).sum()))
                    candidate.append((path,image,row,gt,pred,bio,mi))
                except Exception as e:
                    failures.append({'source_image':src,'stage':'gt_mask','error':f'{type(e).__name__}: {e}'})
        except Exception as e:
            failures.append({'source_image':src,'stage':'source_or_segmentation','error':f'{type(e).__name__}: {e}'})
    # Prefer diversity of garment categories, without looking at labels or agreement.
    candidate.sort(key=lambda t:(CLASSES.get(t[4].category,99),str(t[0]),str(t[2].get('garment_id',''))))
    selected=[]; seen_cats=set()
    for x in candidate:
        if x[4].category not in seen_cats and len(selected)<max_pairs:
            selected.append(x);seen_cats.add(x[4].category)
    seen_ids={id(x) for x in selected}
    for x in candidate:
        if len(selected)>=max_pairs:break
        if id(x) not in seen_ids:
            selected.append(x);seen_ids.add(id(x))
    # Do not count the selection step as performance or semantic quality.
    attr_rows=[];part_rows=[]; pair_rows=[]
    for j,(path,image,row,gt,pred,bi,mi) in enumerate(selected,1):
        pair_id=f'valpair_{j:02d}'
        category=pred.category
        names=[n for n in config['required_attributes'] if applies(n,category)]
        gt_attrs,_=backend.attributes(image,gt,{},names,controlled=True)
        pred_attrs,_=backend.attributes(image,pred,{},names,controlled=True)
        agree_count=0;valid_attr=0
        for name in names:
            g=gt_attrs.get(name,{});p=pred_attrs.get(name,{})
            ok=g.get('status')=='predicted' and p.get('status')=='predicted' and bool(g.get('label')) and bool(p.get('label'))
            same=bool(ok and g.get('label')==p.get('label'))
            if ok:valid_attr+=1;agree_count+=int(same)
            attr_rows.append({'pair_id':pair_id,'category':category,'attribute':name,
                     'gt_garment_roi_label':g.get('label',''),'pred_garment_roi_label':p.get('label',''),
                     'gt_status':g.get('status',''),'pred_status':p.get('status',''),
                     'both_predicted':int(bool(ok)),'label_agreement':int(same) if ok else '',
                     'meaning':'ROI_PERTURBATION_AGREEMENT_NOT_ATTRIBUTE_ACCURACY'})
        allowed=allowed_regions(category)
        picked=[allowed[(j-1+k)%len(allowed)] for k in range(min(regions_per_pair,len(allowed)))]
        for region in picked:
            g,_=backend.ground(image,gt,region)
            p,_=backend.ground(image,pred,region)
            comparable=(g.get('status')=='detected' and p.get('status')=='detected' and g.get('bbox') and p.get('bbox'))
            score=float(box_iou(g['bbox'],p['bbox'])) if comparable else None
            part_rows.append({'pair_id':pair_id,'category':category,'region':region,
                   'gt_garment_roi_status':g.get('status',''),
                   'pred_garment_roi_status':p.get('status',''),
                   'both_detected':int(bool(comparable)),'pred_box_vs_gt_input_box_iou':score,
                   'meaning':'CROSS_INPUT_BOX_STABILITY_NOT_REGION_GT_ACCURACY'})
        pair_rows.append({'pair_id':pair_id,'source_image':str(path),
              'garment_id':row.get('garment_id',''),'category':category,
              'bbox_iou_matched':round(float(bi),6),'garment_mask_iou':round(mi,6),
              'attributes_comparable':valid_attr,'attributes_agree':agree_count,
              'evaluated_regions':';'.join(picked)})
        print(f'PAIRED_GT_VS_PRED: {j}/{len(selected)}',flush=True)
    write_csv(out/'paired_instances.csv',pair_rows)
    write_csv(out/'paired_attribute_agreement.csv',attr_rows)
    write_csv(out/'paired_region_stability.csv',part_rows)
    write_csv(out/'paired_skipped_sources.csv',failures)
    comparable=[r for r in attr_rows if r['both_predicted']]
    region_ok=[r for r in part_rows if r['both_detected']]
    summary={'status':'PAIRED_DIAGNOSTIC_FINISHED' if selected else 'NO_CLASS_CORRECT_MATCHED_PAIRS',
      'candidate_pairs_before_limit':len(candidate),
      'selected_pairs':len(selected),'validation_images_scanned':scanned,
      'source':'configs/prd_8class_val_v2_full_targets.csv',
      'source_scope':'Validation development diagnostic, not Core400 or blinded test.',
      'pair_rule':'bbox_IoU>=0.5 and exact 8class match, deterministic selection',
      'attributes_comparable':len(comparable),
      'attribute_label_agreement_count':sum(int(x['label_agreement']) for x in comparable),
      'attribute_label_agreement_rate':(sum(int(x['label_agreement']) for x in comparable)/len(comparable)) if comparable else None,
      'regional_queries_run':len(part_rows),'both_region_boxes_detected':len(region_ok),
      'mean_cross_input_region_box_iou':statistics.mean(float(x['pred_box_vs_gt_input_box_iou']) for x in region_ok) if region_ok else None,
      'skipped_sources_count':len(failures),
      'not_measured':['True attribute accuracy (no regional/attribute GT labels).',
          'True region localization accuracy (GT garment ROI is NOT GT local part ROI).',
          'Uncontrolled full-chain accuracy.',
          'Independent blind holdout acceptance.'],
      'critical_notes':['controlled=True is used on BOTH garment ROIs. Neckline uses the same top-center PROXY rule, not GT neckline and not predicted collar.',
          'Reported agreement is consistency across garment ROI changes, not correctness.',
          'GT garment ROI masks exist in validation annotations; no regional or attribute ground-truth labels are fabricated.',
          'Pair membership depends on the model correctly localizing and classifying the garment, thus conditional and selection-biased.']}
    write_json(out/'paired_summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(ROOT_DEFAULT));p.add_argument('--out-root',default=str(OUT_DEFAULT))
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--mode',choices=['all','latency','paired'],default='all')
    p.add_argument('--warmup-runs',type=int,default=10);p.add_argument('--timed-passes',type=int,default=5)
    p.add_argument('--pairs-per-region',type=int,default=2)
    p.add_argument('--max-val-pairs',type=int,default=8)
    p.add_argument('--max-val-images',type=int,default=32)
    p.add_argument('--regions-per-pair',type=int,default=2)
    args=p.parse_args()
    if args.warmup_runs<10 or args.timed_passes<5 or args.pairs_per_region<1 or args.max_val_pairs<1 or args.regions_per_pair<1:
        p.error('Expected >=10 warmups, >=5 timed passes, positive sample caps')
    root=Path(args.root).resolve();out_root=Path(args.out_root).resolve()
    if root==out_root or root in out_root.parents or out_root==root.parent:
        p.error('Output root must be outside project repository')
    out=new_out(out_root)
    audit={'status':'STARTED','mode':args.mode,'dry_run':args.dry_run,
           'project_files_modified':False,'scope':'warmed latency and/or controlled GT/pred garment ROI development diagnostic',
           'generated_by':'run_stage6.py'}
    exit_code=0
    try:
        os.environ.setdefault('HF_HOME','/workspace/hf_cache')
        os.environ.setdefault('HF_HUB_CACHE','/workspace/hf_cache/hub')
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        sys.path.insert(0,str(root/'src'))
        config,selection=check_inputs(root,out)
        if args.dry_run:
            audit['status']='DRY_RUN_PASS'
            print('DRY_RUN_PASS',flush=True)
        else:
            audit['checkpoint']=inspect_checkpoint(root)
            print('LOADING_EXISTING_MODELS...',flush=True)
            from fashion_multimodal_analysis.integration.model_backends import ModelBackends
            backend=ModelBackends(config,device='cuda')
            if args.mode in ('all','latency'):
                print('START_WARMED_LATENCY',flush=True)
                audit['latency']=timing_phase(backend,selection,config,out,args.warmup_runs,args.timed_passes,args.pairs_per_region)
            if args.mode in ('all','paired'):
                print('START_VAL_GT_VS_PRED_GARMENT_ROI',flush=True)
                audit['paired']=val_pair_phase(backend,root,config,out,args.max_val_pairs,args.max_val_images,args.regions_per_pair)
            audit['status']='STAGE6_DIAGNOSTIC_COMPLETED'
    except Exception as e:
        audit['status']='STAGE6_FAILED';audit['error']=f'{type(e).__name__}: {e}'
        (out/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('ERROR:',audit['error'],flush=True);exit_code=2
    finally:
        write_json(out/'run_audit.json',audit)
        try:
            zip_path=pack(out)
            print('STATUS:',audit['status'],flush=True)
            print('UPLOAD_THIS_ZIP:',zip_path,flush=True)
            print('PROJECT_FILES_MODIFIED: False',flush=True)
        except Exception as e:
            print('PACK_FAILED:',e,flush=True);exit_code=3
    return exit_code

if __name__=='__main__':sys.exit(main())
