#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recover skirt *target instances* from source-approved multi-instance TRAIN pictures.

NO inference; NO use of holdout/fullval/test/val image bytes. Source labels are NOT GT.
Integrates with the already installed PRD32_20261029_Validation_Gate/prepare_validation.py
freeze command, preserving its inference/GT separation.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, sys
from collections import defaultdict
from pathlib import Path

DEFAULT_VALIDATION='/workspace/PRD32_20261029_Validation_Gate'
DEFAULT_AUDIT='/workspace/PRD32_SOURCE_GAP_AUDIT_20261029_run01'
DEFAULT_SOURCES='/workspace/PRD32_20261029_SourceGap_NextSteps'

def load_helper(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing original script: {path}. Extract the earlier Validation_Gate package first.')
    spec=importlib.util.spec_from_file_location('prd32_existing_prepare_for_skirt_recovery',str(path))
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

def load_12_hashes(file):
    obj=json.loads(file.read_text(encoding='utf8'))
    rows=obj.get('rows',[])
    digests={x['image_sha256'].strip().lower() for x in rows}
    if len(rows)!=12 or len(digests)!=12 or any(len(x)!=64 for x in digests):
        raise ValueError('The prior 12-image review registry is invalid.')
    return digests

def run(project,data,inventory,validation,sources,audit,out):
    if out.exists():raise FileExistsError(f'Output already exists: {out}; use a new output directory')
    prep=load_helper(validation/'prepare_validation.py')
    old=prep.old40_hashes()
    new=load_12_hashes(sources/'evidence'/'new12_blind_review.json')
    if old & new:raise ValueError('Historical 40/12 registries overlap; stop')
    expected_summary=audit/'summary.json'
    if not expected_summary.is_file():raise FileNotFoundError(f'Missing existing source audit summary: {expected_summary}')
    prior=json.loads(expected_summary.read_text(encoding='utf8'))
    if prior.get('status')!='METADATA_AND_ALLOWED_TRAIN_SHA_AUDIT_COMPLETE':
        raise ValueError('The source audit did not complete')
    if not inventory.is_file():raise FileNotFoundError(f'Inventory missing: {inventory}')

    permitted=[]; rejected=[]; path_sha={}
    for i,r in enumerate(prep.read_csv(inventory),1):
        hint=(r.get('garment_category') or '').strip().lower()
        source=(r.get('source_image') or '').strip()
        p,why=prep.safe_image_path(source,project,data)
        box=prep.bbox_from(r)
        if why or box is None or hint not in prep.CATS:
            rejected.append({'inventory_row':i,'reason':why or ('INVALID_BBOX' if box is None else 'INVALID_HINT')})
            continue
        try:
            # The path guard passed before any image bytes are accessed.
            digest=path_sha.get(str(p))
            if digest is None:
                digest=prep.sha(p);path_sha[str(p)]=digest
        except OSError:
            rejected.append({'inventory_row':i,'reason':'ALLOWED_TRAIN_READ_ERROR'});continue
        if digest in old or digest in new:
            rejected.append({'inventory_row':i,'reason':'PREVIOUS_40_OR_12_SHA'});continue
        permitted.append({'inventory_row':i,'hint':hint,'digest':digest,'path':p,'source_relative':source,
          'box':box,'source_record_id':r.get('record_id',''), 'source_garment_id':r.get('garment_id','')})
    byhash=defaultdict(list)
    for rec in permitted:byhash[rec['digest']].append(rec)
    skirt_hashes=sorted(h for h,records in byhash.items() if any(x['hint']=='skirt' for x in records))
    valid=[];failures=[];all_targets=[]
    for digest in skirt_hashes:
        allrows=byhash[digest]
        # One picture only once. Prefer a skirt-hint *instance bbox* over other classes.
        skirtrows=sorted((r for r in allrows if r['hint']=='skirt'),key=lambda x:x['inventory_row'])
        selected=skirtrows[0]
        case=f'skirt_diag_{len(valid)+1:03d}'
        try:
            # Recheck image identity just before extracting the blind preview.
            if prep.sha(selected['path'])!=digest:raise ValueError('SHA_CHANGED_AFTER_METADATA_SCAN')
            full,crop,(w,h)=prep.preview(selected['path'],selected['box'])
        except Exception as e:
            failures.append({'digest_prefix':digest[:12],'inventory_row':selected['inventory_row'],
                             'reason':type(e).__name__+':'+str(e)[:160]})
            continue
        chosen={'case_id':case,'sha256':digest,'source_path':str(selected['path']),
          'source_relative':selected['source_relative'],'bbox':selected['box'],
          'source_record_id':selected['source_record_id'],'source_garment_id':selected['source_garment_id'],
          'inventory_hint_NOT_GT':'skirt','row_index':selected['inventory_row'],
          'width':w,'height':h,'marked_preview':full,'crop_preview':crop}
        valid.append(chosen)
        for r in allrows:
            all_targets.append({'case_id':case,'image_sha256':digest,
                'inventory_row':r['inventory_row'],'source_hint_NOT_GT':r['hint'],
                'selected_target':int(r is selected),'bbox_xyxy':json.dumps(r['box']),
                'selected_sibling_image_sha_prefix':digest[:12]})
    out.mkdir(parents=True,exist_ok=False)
    if valid:prep.build_review_html(valid,out/'Blind_Review_Skirt_Targets.html')
    manifest=[{k:v for k,v in r.items() if not k.endswith('_preview')} for r in valid]
    with (out/'candidate_manifest.jsonl').open('w',encoding='utf8') as f:
        for m in manifest:f.write(json.dumps(m,ensure_ascii=False)+'\n')
    fields=['case_id','image_sha256','inventory_row','source_hint_NOT_GT','selected_target','bbox_xyxy','selected_sibling_image_sha_prefix']
    prep.write_csv(out/'candidate_all_instances_audit.csv',all_targets,fields)
    report={'schema':'prd32_skirt_multi_instance_target_recovery_v1','status':'BLIND_REVIEW_READY' if valid else 'NO_USABLE_SKIRT_TARGETS',
        'source':'EXPLICIT_TRAIN_PATH_ONLY','scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY_NOT_INDEPENDENT',
        'old40_excluded':len(old),'new12_excluded':len(new),'skirt_hint_image_hashes_found':len(skirt_hashes),
        'selected_count':len(valid),'source_hint_NOT_GT':'skirt',
        'eligible_multi_instance_images':sum(len(byhash[h])>1 for h in skirt_hashes),
        'coexisting_instance_hints_by_case':{r['case_id']:sorted({x['source_hint_NOT_GT'] for x in all_targets if x['case_id']==r['case_id']}) for r in valid},
        'preview_failures':failures,'model_calls':0,'holdout_opened':False,'human_GT_generated':False,
        'official_accuracy':None,'candidate_manifest_sha256':prep.sha(out/'candidate_manifest.jsonl'),
        'source_inventory_hash':prep.sha(inventory),'review_html_shows_source_hint':False,
        'limitations':['Only exact SHA duplication excluded.','Multi-instance box/hint might be wrong; blind review required.',
                       'TRAIN source does not imply independent evaluation data.']}
    prep.write_json(out/'summary.json',report)
    (out/'README_FIRST.md').write_text(
        '# Skirt target recovery (train-source diagnostics only)\n\n'
        f'Found {len(skirt_hashes)} new-to-prior40-and12 skirt-hint picture hashes.\n'
        f'Successfully previewed {len(valid)} unique-picture skirt-target bboxes.\n\n'
        'This reselects target instances in multi-garment images and does **not** change any earlier V2/V3 predictions.\n'
        'The blind reviewer must independently verify the bbox, target and category.\n'
        'Freeze with the original prepare_validation.py freeze command after review.\n'
        'These are training-source development samples, NOT independent PRD acceptance.\n',encoding='utf8')
    print('SKIRT_TARGET_RECOVERY_OK',json.dumps({'skirt_hint_images':len(skirt_hashes),'selected':len(valid),
          'multi_instance_images':report['eligible_multi_instance_images'],'output':str(out)},ensure_ascii=False))
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',default='/workspace/fashion_multimodal_analysis')
    ap.add_argument('--data-root',default='/workspace/fashion_data')
    ap.add_argument('--validation-dir',default=DEFAULT_VALIDATION)
    ap.add_argument('--source-package-dir',default=DEFAULT_SOURCES)
    ap.add_argument('--audit-dir',default=DEFAULT_AUDIT)
    ap.add_argument('--inventory')
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    project=Path(a.project_root).resolve();data=Path(a.data_root).resolve()
    inv=Path(a.inventory).resolve() if a.inventory else project/'configs'/'prd_8class_unified_manifest_v2.csv'
    run(project,data,inv,Path(a.validation_dir).resolve(),Path(a.source_package_dir).resolve(),Path(a.audit_dir).resolve(),Path(a.output).resolve())

if __name__=='__main__':main()
