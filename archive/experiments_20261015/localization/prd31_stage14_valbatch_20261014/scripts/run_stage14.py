#!/usr/bin/env python3
"""Stage14: validation-manifest image A/B for strict DINO+SAM scheduler.

A distinct-source extension of Stage13. Uses only files already present on GPU.
No training, no changes to source repo, no PRD semantic/latency acceptance.
"""
from __future__ import annotations
import argparse, json, sys, traceback, os, zipfile
from pathlib import Path
from collections import Counter, defaultdict
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import stage6_helpers as h
import strict_batch_engine as engine
ROOT=Path('/workspace/fashion_multimodal_analysis')
OUTROOT=Path('/workspace/prd31_stage14_runs_20261014')
CLASSES=('top','pants','skirt','outerwear','dress','shoe','bag','accessory')

def canonical(raw):
    from fashion_multimodal_analysis.common.paths import resolve_path
    return str(Path(resolve_path(raw)).resolve(strict=False))

def build_selection(root,previous,max_cases=12):
    """Choose validation images excluding Stage13 and known train/Core paths.

    We intentionally do NOT claim image-content deduplication/blind independence.
    """
    val=h.read_csv(root/h.VAL)
    train=h.read_csv(root/h.TRAIN)
    core=h.read_csv(root/h.CORE)
    old={str(Path(x['source_image']).resolve(strict=False)) for x in previous}
    protected_train={canonical(x['source_image']) for x in train}
    protected_core={canonical(x['source_image']) for x in core}
    # Count unique resolved images; preserve all GT classes for transparency.
    pool={}
    for x in val:
        image=canonical(x['source_image'])
        if image not in pool:
            pool[image]={'source_image':image,'gt_classes':set(),'gt_rows':0}
        pool[image]['gt_classes'].add(x['garment_category'])
        pool[image]['gt_rows']+=1
    skipped=[];eligible=[]
    for image,info in sorted(pool.items()):
        why=[]
        if image in old:why.append('USED_IN_STAGE13')
        if image in protected_train:why.append('TRAIN_PATH_OVERLAP')
        if image in protected_core:why.append('CORE400_PATH_OVERLAP')
        if not Path(image).is_file():why.append('IMAGE_NOT_FOUND')
        if why:
            skipped.append({'source_image':image,'reason':','.join(why)})
        else:
            eligible.append(info)
    if len(eligible)<max(4,min(6,max_cases)):
        raise ValueError(f'Insufficient nonoverlapping validation images: eligible={len(eligible)}; consider using a different clean manifest')
    # Deterministic diverse first-pass, then include remaining available images
    # so engine can skip images without a usable predicted ROI and still get cases.
    chosen=[];seen=set()
    for c in CLASSES:
        for info in eligible:
            if c in info['gt_classes'] and info['source_image'] not in seen:
                chosen.append(info);seen.add(info['source_image']);break
    for info in eligible:
        if info['source_image'] not in seen:
            chosen.append(info);seen.add(info['source_image'])
    candidates=[]
    for info in chosen:
        p=Path(info['source_image'])
        candidates.append({'source_image':str(p),'source_sha256':h.filehash(p),
                           'validation_gt_classes':sorted(info['gt_classes']),
                           'validation_gt_rows':info['gt_rows']})
    return candidates,{
        'selection_protocol':'val_manifest_distinct_source_image_path_exclude_stage13_train_core_known_paths',
        'origin_manifest':str(h.VAL),'origin_manifest_sha256':h.EXPECTED_SHA[h.VAL],
        'total_val_rows':len(val),'unique_val_images':len(pool),
        'eligible_val_images':len(eligible),'source_candidates':len(candidates),
        'requested_max_cases':max_cases,'skipped_paths':skipped,
        'candidate_image_sha256':[{'source_image':x['source_image'],'source_sha256':x['source_sha256'],
                                   'validation_gt_classes':x['validation_gt_classes']}
                                  for x in candidates],
        'exclusion_is_path_based_only':True,'not_blind_or_semantic_acceptance':True,
        'diagnostic_purpose':'validate speed and output parity on additional validation-manifest images',
    }

def pack(out):
    dest=out.parent/f'prd31_stage14_{out.name}_UPLOAD_THIS.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(out))
    with zipfile.ZipFile(dest) as z:
        if z.testzip():raise ValueError('Bad output ZIP CRC')
    return dest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--max-cases',type=int,default=12)
    p.add_argument('--query-count',type=int,default=4)
    p.add_argument('--warmups',type=int,default=3)
    p.add_argument('--passes',type=int,default=5)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--out-root',type=Path,default=OUTROOT)
    a=p.parse_args()
    if not(4<=a.max_cases<=20 and 2<=a.query_count<=8 and 0<=a.warmups<=10 and 1<=a.passes<=10):
        p.error('invalid experiment settings')
    root=a.root.resolve();out_root=a.out_root.resolve()
    if root==out_root or root in out_root.parents or out_root==root.parent:
        p.error('Output must not be created inside source repo')
    out=h.new_out(out_root)
    audit={'status':'STARTED','project_files_modified':False,'accuracy_acceptance':False,'latency_acceptance':False}
    rc=0
    try:
        os.environ.setdefault('HF_HOME','/workspace/hf_cache')
        os.environ.setdefault('HF_HUB_CACHE','/workspace/hf_cache/hub')
        os.environ['HF_HUB_OFFLINE']='1'
        os.environ['TRANSFORMERS_OFFLINE']='1'
        sys.path.insert(0,str(root/'src'))
        cfg,previous=h.check_inputs(root,out)
        selection,sel_audit=build_selection(root,previous,a.max_cases)
        h.write_json(out/'stage14_selection_audit.json',sel_audit)
        # Re-hash images at the point of use, before model initialization.
        for row in selection:
            if h.filehash(row['source_image'])!=row['source_sha256']:
                raise ValueError('Stage14 selected image changed')
        if not cfg.get('strict_region_grounding'):
            raise ValueError('Strict protocol disabled')
        if a.dry_run:
            audit['status']='DRY_RUN_PASS'
            audit['selection']={'eligible_val_images':sel_audit['eligible_val_images'],
                                'candidate_count':len(selection)}
        else:
            audit['checkpoint']=h.inspect_checkpoint(root)
            from fashion_multimodal_analysis.integration.model_backends import ModelBackends
            backend=ModelBackends(cfg,device='cuda')
            base_summary=engine.run(backend,cfg,selection,out,a.max_cases,a.query_count,a.warmups,a.passes)
            actual=h.read_csv(out/'optimized_group_timing.csv')
            final=h.read_csv(out/'final_region_parity.csv')
            repeat=h.read_csv(out/'repeat_region_parity.csv')
            min_mask=base_summary.get('min_detected_mask_iou')
            every_parity=all(x['parity_screen_pass'].lower()=='true' for x in final+repeat)
            decision=('DO_NOT_INTEGRATE' if not every_parity or min_mask is not None and min_mask<0.999
                      else 'DIAGNOSTIC_PASS_DO_NOT_DEPLOY_YET')
            summary={**base_summary,
                'status':'STAGE14_VALIDATION_MANIFEST_AB_COMPLETED',
                'selection_type':'different validation-manifest images; path-disjoint from Stage13/train/Core400',
                'selected_case_count':base_summary['cases'],
                'selected_query_count':base_summary['queries'],
                'all_region_parity_passed':every_parity,
                'deployment_decision':decision,
                'validation_images_blind':False,
                'has_human_part_gt':False,
                'semantic_accuracy_computed':False,
                'per_query_latency_prd_passed':False,
                'validation_selection_details':'stage14_selection_audit.json',
            }
            summary['notes']=[x for x in summary.get('notes',[]) if not x.startswith('Only 8')]+[
                'Additional validation-manifest images; separate from 8 Stage13 diagnostic images by canonical source path.',
                'Validation-manifest images may have been used during model development; this is NOT a clean blind test.',
                'All parity checks are tolerance-based; exact pixel equality separately recorded.',
                'Data split overlap is checked by resolved paths, not content-level/perceptual duplicates.',
                'No production code modification or official 3.1.2 human semantic evaluation.',
            ]
            h.write_json(out/'stage14_summary.json',summary)
            audit['status']=summary['status'];audit['result']=summary
    except Exception as exc:
        audit['status']='STAGE14_FAILED'
        audit['error']=f'{type(exc).__name__}: {exc}'
        (out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf8')
        print('ERROR:',audit['error'],flush=True);rc=2
    finally:
        h.write_json(out/'run_audit.json',audit)
        try:
            dest=pack(out)
            print('STATUS:',audit['status'],flush=True)
            if a.dry_run:
                print('DRY_RUN_PASS' if audit['status']=='DRY_RUN_PASS' else 'DRY_RUN_FAILED')
            print('UPLOAD_THIS_ZIP:',dest,flush=True)
            print('PROJECT_FILES_MODIFIED: False',flush=True)
        except Exception as exc:
            print('PACK_FAILED:',repr(exc),flush=True);rc=3
    return rc
if __name__=='__main__':sys.exit(main())
