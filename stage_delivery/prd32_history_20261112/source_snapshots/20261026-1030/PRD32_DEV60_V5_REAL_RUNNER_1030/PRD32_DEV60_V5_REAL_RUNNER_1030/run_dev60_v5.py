#!/usr/bin/env python3
# coding: utf-8
"""PRD 3.2 DEV60 read-only-cohort V5 inference using untouched 2026-10-29 runner.

Two explicit modes: preflight only (no model); gpu (same 10/29 original run_gpu).
The ONLY monkeypatch is an audited input validator accepting 60 GT-free images.
Never reads a sealed/holdout/test/val image; never claims official PRD acceptance.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path

SCHEMA = 'prd32_dev60_v5_real_runner_v1'
EXPECTED_RUNNER_SHA = 'bdb955ac88f24b35bcacde1eed790f8207bbc9bcfb304e85b1fee76771d9a91b'
EXPECTED_MANIFEST_SHA = 'c629ff0ca557a0c57c7b97c93cf389502f9660de63a934463cb1f5e8e722752f'
EXPECTED_OLD12_MANIFEST_SHA = 'c4be7e85bd2db2ae1ea9704dba584c8952d1227f7a1d02cbfba24be4b3ecb01d'
EXPECTED_CHECKPOINT_SHA = 'c5ba795225349b1c521aac0517ac7c493d9818dcc17d558ac4a3f6f8669452f0'
EXPECTED_SOURCE_FIELDS = {'case_id', 'image_path', 'image_sha256', 'do_not_use_historical_gt_bbox', 'human_reference_withheld'}
LEGACY_FIELDS = EXPECTED_SOURCE_FIELDS | {'requested_output', 'scope'}
SOURCE_SCOPE = 'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY'
DEFAULT_PROJECT = '/workspace/fashion_multimodal_analysis'
DEFAULT_DATA = '/workspace/fashion_data'
DEFAULT_SOURCE = '/workspace/PRD32_FIXED_DEV_V2_60IMG_CANDIDATES_20261030/v5_image_only_inputs_60.jsonl'
DEFAULT_LEGACY = '/workspace/PRD32_V5_RealInference_20261029/v5_real_inference.py'
DEFAULT_OLD_GATE = '/workspace/PRD32_V5_LIVE_PREFLIGHT_20261029_run01'
DEFAULT_GATE = '/workspace/PRD32_DEV60_V5_PREFLIGHT_20261030'
DEFAULT_RUN = '/workspace/PRD32_DEV60_V5_REAL_20261030_run01'
DEFAULT_EXPORT = '/workspace/PRD32_DEV60_V5_REAL_20261030_UPLOAD_THIS.zip'
VALID_CLASSES = {'top','pants','skirt','outerwear','dress','shoe','bag','accessory'}
RESTRICTED = {'holdout','sealed','fullval','val','valid','validation','test','testing','testset','eval','evaluation','devtest'}
HEX64 = re.compile(r'[0-9a-f]{64}\Z')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):h.update(block)
    return h.hexdigest()

def json_load(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def jsonl_load(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def dump(p, obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def is_restricted(part):
    s=part.lower()
    return any(s==x or s.startswith(x+'_') or s.endswith('_'+x) for x in RESTRICTED)

def approve_train_path(v, data_root, check_exists=True):
    if not isinstance(v,str) or not v.startswith('/'):
        raise ValueError('ABSOLUTE_IMAGE_PATH_REQUIRED')
    p=Path(v)
    if any(is_restricted(part) for part in p.parts):
        raise ValueError('RESTRICTED_SPLIT_INPUT_PATH')
    resolved=p.resolve(strict=False); base=Path(data_root).resolve(strict=False)
    if any(is_restricted(part) for part in resolved.parts):
        raise ValueError('RESTRICTED_SPLIT_RESOLVED_PATH')
    if not resolved.is_relative_to(base) or 'train' not in [x.lower() for x in resolved.parts]:
        raise ValueError('IMAGE_NOT_APPROVED_TRAIN_SOURCE')
    if resolved.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'}:
        raise ValueError('IMAGE_EXTENSION_NOT_APPROVED')
    if check_exists and not resolved.is_file():
        raise ValueError('TRAIN_IMAGE_NOT_FOUND')
    return resolved

def check_60_manifest(p, data_root, check_hash=True):
    p=Path(p)
    if not p.is_file() or sha(p)!=EXPECTED_MANIFEST_SHA:
        raise ValueError('DEV60_SOURCE_MANIFEST_SHA_MISMATCH')
    rows=jsonl_load(p)
    if len(rows)!=60:
        raise ValueError('EXPECTED_60_DEV_IMAGE_RECORDS')
    seen_sha=set(); seen_paths=set(); result=[]
    for n, r in enumerate(rows, 1):
        cid=f'fresh_dev_{n:03d}'
        if not isinstance(r,dict) or set(r)!=EXPECTED_SOURCE_FIELDS:
            raise ValueError(f'{cid} NOT_GT_FREE_5_FIELD_SCHEMA')
        if r['case_id']!=cid:
            raise ValueError(f'{cid} CASE_ORDER_OR_ID_CHANGED')
        if r['human_reference_withheld'] is not True or r['do_not_use_historical_gt_bbox'] is not True:
            raise ValueError(f'{cid} HUMAN_REFERENCE_NOT_WITHHELD')
        digest=r['image_sha256']
        if not isinstance(digest,str) or not HEX64.fullmatch(digest):
            raise ValueError(f'{cid} INVALID_DIGEST')
        rp=approve_train_path(r['image_path'], data_root, check_exists=check_hash)
        if digest in seen_sha or str(rp) in seen_paths:
            raise ValueError(f'{cid} DUPLICATE_CONTENT_OR_PATH')
        if check_hash and sha(rp)!=digest:
            raise ValueError(f'{cid} IMAGE_BYTES_CHANGED')
        seen_sha.add(digest);seen_paths.add(str(rp))
        result.append({'case_id':cid,'image_path':str(rp),'image_sha256':digest,
                       'requested_output':'PRD31_V5_RAW_PREDICTIONS_ONLY',
                       'scope':SOURCE_SCOPE,'do_not_use_historical_gt_bbox':True,
                       'human_reference_withheld':True})
    return result

def check_image_headers(rows):
    # This reads ONLY train-source image headers. No GT/holdout images are touched.
    from PIL import Image
    for r in rows:
        cid=r['case_id']
        with Image.open(r['image_path']) as im:
            if im.getexif().get(274,1) not in (1,None):
                raise ValueError(f'{cid} EXIF_ROTATION_UNSUPPORTED_BY_LEGACY_RUNNER')
            if im.width*im.height>45_000_000:
                raise ValueError(f'{cid} IMAGE_TOO_LARGE_FOR_LEGACY_RUNNER')
            if im.width<1 or im.height<1:
                raise ValueError(f'{cid} BAD_IMAGE_SIZE')

def load_unchanged_runner(path):
    p=Path(path)
    if not p.is_file() or sha(p)!=EXPECTED_RUNNER_SHA:
        raise ValueError('ORIGINAL_20261029_V5_RUNNER_CHANGED_OR_MISSING')
    spec=importlib.util.spec_from_file_location('prd32_original_v5_runner_locked_20261029',p)
    if spec is None or spec.loader is None:raise ValueError('IMPORT_SPEC_FAILED')
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if getattr(mod,'CKPT_SHA',None)!=EXPECTED_CHECKPOINT_SHA or mod.CLASS_TO_ID!={
        'top':1,'pants':2,'skirt':3,'outerwear':4,'dress':5,'shoe':6,'bag':7,'accessory':8}:
        raise ValueError('LEGACY_V5_CONTRACT_MISMATCH')
    return mod

def check_previous_12(mod,args,rows):
    checked,ckpt,summary=mod.validate_inputs(args.old_gate,args.data_root,args.project,check_hash=True)
    old_manifest=Path(args.old_gate)/'v5_image_only_inputs.jsonl'
    if sha(old_manifest)!=EXPECTED_OLD12_MANIFEST_SHA or len(checked)!=12:
        raise ValueError('ORIGINAL_12_RUN_INPUTS_CHANGED')
    for old, new in zip(checked,rows[:12]):
        if any(old[k]!=new[k] for k in ('case_id','image_path','image_sha256')):
            raise ValueError('ORIGINAL_12_IMAGE_IDENTITY_OR_ORDER_CHANGED')
    if str(ckpt.resolve())!=str((Path(args.project)/mod.CHECKPOINT_REL).resolve()):
        raise ValueError('ORIGINAL_CHECKPOINT_PATH_MISMATCH')
    return ckpt,summary

def preflight_record(args,rows,ckpt,legacy_gate_summary):
    return {
        'schema':'prd32_dev60_v5_gpu_preflight_v1',
        'status':'DEV60_PREFLIGHT_PASS_DIAGNOSTIC_ONLY',
        'scope':SOURCE_SCOPE,
        'input_manifest_sha256':EXPECTED_MANIFEST_SHA,
        'input_count':len(rows),
        'legacy_12_preserved':True,
        'legacy_runner_sha256':EXPECTED_RUNNER_SHA,
        'checkpoint_sha256':EXPECTED_CHECKPOINT_SHA,
        'checkpoint_bytes_verified':True,
        'legacy_preflight_schema':legacy_gate_summary.get('schema'),
        'score_threshold':0.4,
        'mask_threshold':0.5,
        'human_gt_used_for_inference':False,
        'sealed_test_val_opened':False,
        'legacy_v5_model_logic_unchanged':True,
        'original_12_v5_results_not_modified':True,
        'official_prd_acceptance':False,
        'official_v2_freeze':False,
        'main_ab_inference_ran':False,
        'ready_for_dev60_diagnostic_gpu':True,
        'note':'Preflight checks only. No model inference or reserved-split access.'}

def build_preflight(args):
    gate=Path(args.gate)
    if gate.exists():raise ValueError('GATE_DIR_ALREADY_EXISTS_NO_OVERWRITE: '+str(gate))
    rows=check_60_manifest(args.source,args.data_root,check_hash=True)
    check_image_headers(rows)
    mod=load_unchanged_runner(args.legacy)
    ckpt,old_summary=check_previous_12(mod,args,rows)
    if sha(ckpt)!=EXPECTED_CHECKPOINT_SHA:raise ValueError('CHECKPOINT_SHA_MISMATCH')
    record=preflight_record(args,rows,ckpt,old_summary)
    gate.mkdir(parents=True,exist_ok=False)
    with (gate/'v5_image_only_inputs.jsonl').open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    record['legacy_7field_manifest_sha256']=sha(gate/'v5_image_only_inputs.jsonl')
    dump(gate/'dev60_preflight.json',record)
    print('DEV60_PREFLIGHT_PASS',json.dumps({'image_count':60,'rehashed':60,'checkpoint_sha256':EXPECTED_CHECKPOINT_SHA,
                                            'original_12_preserved':True,'model_loaded':False,'gate':str(gate)},ensure_ascii=False),flush=True)

def validate_gpu_contract(args):
    gate=Path(args.gate)
    preflight=json_load(gate/'dev60_preflight.json')
    if preflight.get('schema')!='prd32_dev60_v5_gpu_preflight_v1' or preflight.get('status')!='DEV60_PREFLIGHT_PASS_DIAGNOSTIC_ONLY':
        raise ValueError('INVALID_PREPARED_GATE')
    rows=check_60_manifest(args.source,args.data_root,check_hash=True)
    mod=load_unchanged_runner(args.legacy)
    cp,old_summary=check_previous_12(mod,args,rows)
    if sha(cp)!=EXPECTED_CHECKPOINT_SHA:raise ValueError('CHECKPOINT_SHA_MISMATCH')
    if preflight.get('input_manifest_sha256')!=EXPECTED_MANIFEST_SHA or preflight.get('legacy_runner_sha256')!=EXPECTED_RUNNER_SHA:
        raise ValueError('DEV60_GATE_INPUT_OR_RUNNER_CHANGED')
    prepared=gate/'v5_image_only_inputs.jsonl'
    if sha(prepared)!=preflight.get('legacy_7field_manifest_sha256'):
        raise ValueError('STAGED_INPUTS_CHANGED')
    staged=jsonl_load(prepared)
    if staged!=rows or any(set(x)!=LEGACY_FIELDS for x in staged):
        raise ValueError('STAGED_INPUT_SCHEMA_OR_ROWS_DIFFER')
    if not all(x.get('human_reference_withheld') is True for x in staged):
        raise ValueError('HUMAN_GT_LEAKAGE')
    return mod,rows,cp,old_summary

def run_gpu(args):
    if Path(args.run).exists():raise ValueError('RUN_OUTPUT_EXISTS_NO_OVERWRITE: '+args.run)
    mod,rows,cp,old=validate_gpu_contract(args)
    if args.score_threshold!=0.4 or args.mask_threshold!=0.5 or args.max_masks!=8:
        raise ValueError('LEGACY_V5_SCORE_MASK_AND_SAVE_POLICY_MUST_REMAIN_UNCHANGED')
    print('DEV60_V5_GPU_START original_legacy_run_gpu_unchanged input_images=60',flush=True)
    old_validator=mod.validate_inputs
    def verified_60_validator(gate_dir, data_root, project, check_hash=True):
        if not check_hash or str(Path(gate_dir).resolve())!=str(Path(args.gate).resolve()):
            raise ValueError('ONLY_VERIFIED_DEV60_GATE_ALLOWED')
        if Path(project).resolve()!=Path(args.project).resolve() or Path(data_root).resolve()!=Path(args.data_root).resolve():
            raise ValueError('APPROVED_ROOTS_CHANGED')
        return rows,cp,old
    # Preserve the original 2026-10-29 Mask R-CNN run_gpu byte-for-byte;
    # replace ONLY its 12-image validator, after a separate 60-image full audit.
    mod.validate_inputs=verified_60_validator
    try:
        a=argparse.Namespace(gate=args.gate,project=args.project,data_root=args.data_root,output=args.run,
                             score_threshold=args.score_threshold,mask_threshold=args.mask_threshold,max_masks=args.max_masks)
        mod.run_gpu(a)
    finally:
        mod.validate_inputs=old_validator
    summary=verify_output(args)
    print('DEV60_V5_GPU_AND_VERIFY_PASS',json.dumps({'images':summary['n_images'],'instances':summary['n_instances'],
                'no_predictions_images':summary['n_empty_images'],'run_dir':args.run},ensure_ascii=False),flush=True)

def verify_output(args):
    mod,expected,cp,old=validate_gpu_contract(args)
    out=Path(args.run)
    raw=out/'v5_raw_predictions.jsonl'
    manifest=out/'run_manifest.json'
    summary=out/'v5_real_summary.json'
    if not all(x.is_file() for x in (raw,manifest,summary)):
        raise ValueError('RUN_OUTPUT_INCOMPLETE')
    runmeta=json_load(manifest);runsummary=json_load(summary)
    if runmeta.get('schema')!='prd32_v5_real_predictions_run_v1' or runmeta.get('input_manifest_sha256')!=sha(Path(args.gate)/'v5_image_only_inputs.jsonl'):
        raise ValueError('RUN_INPUT_MANIFEST_MISMATCH')
    if runmeta.get('checkpoint_sha256')!=EXPECTED_CHECKPOINT_SHA or runmeta.get('score_threshold')!=0.4 or runmeta.get('mask_threshold')!=0.5 or runmeta.get('human_gt_used_for_inference') is not False:
        raise ValueError('RUN_MODEL_CONFIG_MISMATCH')
    if runsummary.get('status')!='V5_RAW_INFERENCE_COMPLETED' or runsummary.get('n_images')!=60 or runsummary.get('checkpoint_sha256')!=EXPECTED_CHECKPOINT_SHA:
        raise ValueError('V5_60_IMAGE_RUN_NOT_COMPLETE')
    if runsummary.get('v5_predictions_sha256')!=sha(raw):
        raise ValueError('RAW_PREDICTION_OUTPUT_SHA_MISMATCH')
    rows=jsonl_load(raw)
    if len(rows)!=60:raise ValueError('PREDICTION_OUTPUT_NOT_60_IMAGES')
    total=0; blank=0; bycat=Counter(); mask_saved=0; missing_previews=0
    for record,src in zip(rows,expected):
        if record.get('case_id')!=src['case_id'] or record.get('image_sha256')!=src['image_sha256'] or Path(record.get('source_image_path','')).resolve()!=Path(src['image_path']).resolve():
            raise ValueError('CASE_ID_PATH_OR_SHA_PAIRING_BROKEN')
        if record.get('schema')!='prd32_v5_live_raw_image_result_v1' or record.get('scope')!=SOURCE_SCOPE or record.get('human_target_instance_selected') is not False:
            raise ValueError('PREDICTION_RECORD_SCOPE_OR_SCHEMA_CHANGED')
        size=record.get('image_size');entries=record.get('instances')
        if not (isinstance(size,list) and len(size)==2 and all(type(q) is int and q>0 for q in size)):
            raise ValueError('INVALID_IMAGE_SIZE')
        if not isinstance(entries,list) or record.get('instance_count')!=len(entries):
            raise ValueError('INVALID_INSTANCE_COUNT')
        if not (out/record.get('review_image','INVALID')).is_file(): missing_previews+=1
        if not entries:blank+=1
        ids=set()
        for inst in entries:
            total+=1
            iid=inst.get('instance_id')
            if not isinstance(iid,str) or not re.fullmatch(r'pred_\d{4,}',iid) or iid in ids:raise ValueError('INSTANCE_ID_DUPLICATED_OR_INVALID')
            ids.add(iid)
            cat=inst.get('predicted_category')
            if cat not in VALID_CLASSES:raise ValueError('PRED_CATEGORY_INVALID')
            if inst.get('predicted_class_id')!=mod.CLASS_TO_ID[cat] or inst.get('prediction_origin')!='V5_CHECKPOINT_LIVE_MODEL_OUTPUT_NOT_GT':
                raise ValueError('PRED_ORIGIN_OR_CATEGORY_CONTRACT_MISMATCH')
            box=inst.get('bbox_xyxy')
            if not (isinstance(box,list) and len(box)==4 and all(isinstance(q,(int,float)) and math.isfinite(q) for q in box)
                    and 0<=box[0]<box[2]<=size[0] and 0<=box[1]<box[3]<=size[1]):
                raise ValueError('PRED_BOX_INVALID')
            mask=inst.get('mask_path')
            if mask:
                if Path(mask).name!=mask or not (out/mask).is_file():raise ValueError('PRED_MASK_MISSING')
                mask_saved+=1
            bycat[cat]+=1
    if missing_previews: raise ValueError(f'REVIEW_PREVIEW_MISSING_COUNT={missing_previews}')
    if total!=runsummary.get('n_predicted_instances') or blank!=runsummary.get('n_images_with_no_instances'):
        raise ValueError('SUMMARY_PRED_COUNT_MISMATCH')
    if dict(bycat)!=runsummary.get('categories_from_v5_predictions'):
        raise ValueError('PRED_CATEGORY_SUMMARY_MISMATCH')
    result={
        'schema':'prd32_dev60_v5_real_inference_verification_v1',
        'status':'DEV60_V5_REAL_INFERENCE_VERIFIED_DEV_DIAGNOSTIC_ONLY',
        'scope':SOURCE_SCOPE,
        'n_images':60,'n_instances':total,'n_empty_images':blank,
        'n_saved_masks':mask_saved,'n_review_previews':60,
        'per_predicted_category_NOT_GT':dict(bycat),
        'input_image_manifest_sha256':EXPECTED_MANIFEST_SHA,
        'legacy_runner_sha256':EXPECTED_RUNNER_SHA,
        'checkpoint_sha256':EXPECTED_CHECKPOINT_SHA,
        'raw_prediction_jsonl_sha256':sha(raw),
        'old12_preserved':True,
        'score_threshold':0.4,'mask_threshold':0.5,'max_masks_per_image':8,
        'qwen_calls':0,'official_v2_freeze':False,
        'official_prd_acceptance':False,'publish_allowed':False,
        'holdout_test_val_images_read':False,
        'model_annotations_are_not_human_gt':True,
        'note':'Predicted instances may include duplicates/misses; no official QA claims or human GT recall.'
    }
    dump(out/'dev60_verification.json',result)
    print('DEV60_V5_VERIFICATION_PASS',json.dumps({'images':60,'predicted_instances':total,'zero_instance_images':blank,
                     'v5_categories':dict(bycat),'output':str(out/'dev60_verification.json')},ensure_ascii=False),flush=True)
    return result

def export_run(args):
    result=verify_output(args)
    target=Path(args.export)
    if target.exists():raise ValueError('EXPORT_EXISTS_REFUSE_OVERWRITE')
    out=Path(args.run);gate=Path(args.gate)
    files=[gate/'dev60_preflight.json',gate/'v5_image_only_inputs.jsonl',out/'run_manifest.json',
           out/'v5_real_summary.json',out/'v5_raw_predictions.jsonl',out/'dev60_verification.json']
    files+=sorted(out.glob('fresh_dev_*_v5_review.jpg'))
    if len(files)!=66:raise ValueError('UNEXPECTED_EXPORT_FILE_COUNT')
    if any(not p.is_file() for p in files):raise ValueError('EXPORT_FILE_MISSING')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for f in files:
            z.write(f,('preflight/' if f.parent==gate else 'run/')+f.name)
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:raise ValueError('ZIP_CRC_FAILED')
    print('DEV60_V5_UPLOAD_THIS_ZIP',str(target),'files',len(files),'images',result['n_images'],
          'instances',result['n_instances'],flush=True)

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['preflight','gpu','verify','export'])
    ap.add_argument('--source',default=DEFAULT_SOURCE)
    ap.add_argument('--legacy',default=DEFAULT_LEGACY)
    ap.add_argument('--old-gate',default=DEFAULT_OLD_GATE)
    ap.add_argument('--gate',default=DEFAULT_GATE)
    ap.add_argument('--run',default=DEFAULT_RUN)
    ap.add_argument('--export',default=DEFAULT_EXPORT)
    ap.add_argument('--project',default=DEFAULT_PROJECT)
    ap.add_argument('--data-root',default=DEFAULT_DATA)
    ap.add_argument('--score-threshold',type=float,default=0.4)
    ap.add_argument('--mask-threshold',type=float,default=0.5)
    ap.add_argument('--max-masks',type=int,default=8)
    args=ap.parse_args(argv)
    if args.mode=='preflight':build_preflight(args)
    elif args.mode=='gpu':run_gpu(args)
    elif args.mode=='verify':verify_output(args)
    elif args.mode=='export':export_run(args)

if __name__=='__main__':
    try: main()
    except Exception as e:
        print('DEV60_V5_BLOCKED',type(e).__name__+': '+str(e),file=sys.stderr,flush=True)
        sys.exit(2)
