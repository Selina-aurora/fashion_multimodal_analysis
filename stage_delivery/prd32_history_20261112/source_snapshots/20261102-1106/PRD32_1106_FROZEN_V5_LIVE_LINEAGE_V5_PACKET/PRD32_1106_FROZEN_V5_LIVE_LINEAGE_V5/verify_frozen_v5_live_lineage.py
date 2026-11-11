#!/usr/bin/env python3
"""DEV60-only frozen V5 artifact -> live Qwen host-call evidence gate.

Does not invoke models, modify original outputs or infer live ROI/mask producer identity.
Requires pinned 2026-10-30 V5 raw bytes, actual source images, existing V5 mask files
and actual Live Shadow V4 output (the selected DEV-only run).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

V5_RAW_SHA = "688cb801daebd0484e8740c877c021eca86c83f362857d4fc982d02fb50f2710"
V5_CKPT_SHA = "c5ba795225349b1c521aac0517ac7c493d9818dcc17d558ac4a3f6f8669452f0"
TASKS = {"tags", "vqa", "copy"}
CATEGORIES = {"top", "pants", "skirt", "outerwear", "dress", "shoe", "bag", "accessory"}
HEX64 = re.compile(r"^[a-f0-9]{64}$")
CASE = re.compile(r"^fresh_dev_\d{3}$")
INST = re.compile(r"^pred_\d{4}$")
FORBIDDEN = {"holdout", "sealed", "blind", "validation", "testset", "independent_test"}


def sha256_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''):h.update(b)
    return h.hexdigest()


def restricted(p: Path) -> bool:
    return any(any(s in part.lower() for s in FORBIDDEN) for part in p.parts)


def checked_path(path: Path, root: Path, *, name: str) -> Path:
    p=path.resolve(strict=True)
    if restricted(p) or restricted(path): raise ValueError(f"restricted {name} path")
    if not p.is_relative_to(root.resolve(strict=True)):
        raise ValueError(f"{name} path escapes approved development root")
    if not p.is_file(): raise ValueError(f"{name} is not a file")
    return p


def read_json(p: Path):return json.loads(p.read_text(encoding='utf-8-sig'))


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def checked_bbox(box):
    if not isinstance(box,list) or len(box)!=4 or any(type(x) not in (int,float) or not math.isfinite(x) for x in box):
        raise ValueError("invalid bbox")
    x1,y1,x2,y2=map(float,box)
    if not(0<=x1<x2 and 0<=y1<y2):raise ValueError("nonpositive bbox")
    return (x1,y1,x2,y2)


def check_v5_raw(raw_dir: Path, image_root: Path):
    raw=raw_dir/'v5_raw_predictions.jsonl'
    if sha256_file(raw)!=V5_RAW_SHA:raise ValueError('V5_RAW_SHA_MISMATCH')
    rows=read_jsonl(raw)
    summary=read_json(raw_dir/'v5_real_summary.json')
    manifest=read_json(raw_dir/'run_manifest.json')
    if len(rows)!=60 or (summary.get('n_images'),summary.get('n_predicted_instances'))!=(60,171):
        raise ValueError('V5_DEV60_COUNTS_CHANGED')
    if summary.get('checkpoint_sha256')!=V5_CKPT_SHA or manifest.get('checkpoint_sha256')!=V5_CKPT_SHA:
        raise ValueError('V5_CHECKPOINT_CHANGED')
    if manifest.get('human_gt_used_for_inference') is not False:
        raise ValueError('HUMAN_GT_INFERENCE_NOT_EXCLUDED')
    if (manifest.get('score_threshold'),manifest.get('mask_threshold'))!=(0.4,0.5):
        raise ValueError('V5_THRESHOLDS_CHANGED')
    by_key={}
    for i,pred in enumerate(rows,1):
        case=f'fresh_dev_{i:03d}'
        if pred.get('case_id')!=case:raise ValueError('CASE_ORDER_CHANGED')
        sha=pred.get('image_sha256')
        if not isinstance(sha,str) or not HEX64.fullmatch(sha):raise ValueError('BAD_IMAGE_SHA')
        img=checked_path(Path(pred['source_image_path']), image_root, name='original image')
        if img.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'}:raise ValueError('BAD_IMAGE_TYPE')
        if 'train' not in [v.lower() for v in img.parts]:raise ValueError('IMAGE_NOT_TRAIN_SOURCE')
        if sha256_file(img)!=sha:raise ValueError(f'ORIGINAL_IMAGE_SHA_MISMATCH:{case}')
        w,h=pred['image_size']
        if not isinstance(w,int) or not isinstance(h,int) or min(w,h)<1:
            raise ValueError('INVALID_IMAGE_SIZE')
        instances=pred.get('instances',[])
        if len(instances)!=pred.get('instance_count'):raise ValueError('INSTANCE_COUNT_CHANGED')
        for j,ins in enumerate(instances):
            iid=f'pred_{j:04d}'
            if ins.get('instance_id')!=iid:raise ValueError('INSTANCE_ID_CHANGED')
            cat=ins.get('predicted_category')
            if cat not in CATEGORIES:raise ValueError('INVALID_V5_CATEGORY')
            bbox=checked_bbox(ins.get('bbox_xyxy'))
            if bbox[2]>w or bbox[3]>h:raise ValueError('V5_BBOX_OUTSIDE_IMAGE')
            if ins.get('bbox_coordinate_space')!='original_image' or ins.get('prediction_origin')!='V5_CHECKPOINT_LIVE_MODEL_OUTPUT_NOT_GT':
                raise ValueError('V5_SOURCE_ORIGIN_CHANGED')
            by_key[(case,iid)]={'image_path':str(img),'image_sha256':sha,'size':(w,h),'category':cat,
                                'bbox':list(bbox),'mask_path_raw':ins.get('mask_path'),
                                'score':ins.get('score')}
    if len(by_key)!=171:raise ValueError('V5_INSTANCE_COUNT_NOT_171')
    return by_key


def check_live_records(live_dir: Path):
    results=read_jsonl(live_dir/'qwen_shadow_results.jsonl')
    events=read_jsonl(live_dir/'host_call_events.jsonl')
    summary=read_json(live_dir/'run_summary.json')
    report=read_json(live_dir/'guard_shadow_report.json')
    if summary.get('status')!='DEV_ONLY_LIVE_SHADOW_COMPLETE_NOT_APPROVED' or summary.get('model_variant')!='baseline':
        raise ValueError('NOT_COMPLETED_BASELINE_DEV_SHADOW')
    if report.get('status')!='LIVE_HOST_CALLS_CORRELATED_BUT_ROI_MASK_VISUAL_UNVERIFIED':
        raise ValueError('UNEXPECTED_HOST_REPORT')
    if summary.get('holdout_used') is not False or report.get('holdout_accessed') is not False:
        raise ValueError('HOLDOUT_NOT_EXCLUDED')
    if report.get('release_allowed') is not False or summary.get('publish_allowed') is not False:
        raise ValueError('UNSAFE_RELEASE_STATE')
    if len(results)!=len(events) or len(results)!=summary.get('new_model_calls') or len(results)!=report.get('tasks_correlated'):
        raise ValueError('TASK_COUNT_MISMATCH')
    byid={}
    for e in events:
        request=e.get('host_request_id')
        if request in byid or not request:raise ValueError('DUPLICATE_HOST_REQUEST')
        byid[request]=e
    groups=defaultdict(set)
    for row in results:
        case=row.get('case_id');iid=row.get('target_instance_id');task=row.get('task');request=row.get('host_request_id')
        if not CASE.fullmatch(str(case)) or not INST.fullmatch(str(iid)) or task not in TASKS:
            raise ValueError('INVALID_DEV_INSTANCE_TASK')
        if row.get('model_variant')!='baseline':raise ValueError('VARIANT_MISMATCH')
        if request != f'{case}__{iid}__{task}__baseline__attempt_01':raise ValueError('REQUEST_FORMAT_MISMATCH')
        e=byid.get(request)
        if not e:raise ValueError('MISSING_HOST_EVENT')
        pairs=[('case_id','case_id'),('target_instance_id','instance_id'),('task','task'),('model_variant','variant'),
               ('image_sha256','image_sha256'),('marked_image_sha256','marked_sha256'),('prompt_sha256','prompt_sha256')]
        for a,b in pairs:
            if row.get(a)!=e.get(b):raise ValueError(f'HOST_IDENTITY_MISMATCH:{a}')
        if e.get('job_id')!=request or e.get('status')!='HOST_SYNCHRONOUS_CALL_CORRELATED_SHADOW_ONLY':
            raise ValueError('HOST_JOB_STATUS_MISMATCH')
        raw=row.get('raw_text')
        if not isinstance(raw,str) or hashlib.sha256(raw.encode('utf-8')).hexdigest()!=e.get('response_sha256'):
            raise ValueError('RESPONSE_BYTES_MISMATCH')
        if row.get('publish_allowed') is not False or row.get('source_roi_identity_verified') is not False or row.get('source_mask_identity_verified') is not False:
            raise ValueError('CLAIMED_UNPROVEN_ROI_MASK_OR_PUBLISH')
        if e.get('release_allowed') is not False:raise ValueError('EVENT_MUST_BLOCK')
        groups[(case,iid)].add(task)
        row['_validated_event']=e
    if not groups or any(tasks!=TASKS for tasks in groups.values()):raise ValueError('INCOMPLETE_TASK_GROUP')
    if set(byid)!={x['host_request_id'] for x in results}:raise ValueError('ORPHAN_HOST_EVENT')
    if report.get('instances')!=len(groups):raise ValueError('GUARD_INSTANCE_COUNT_MISMATCH')
    if report.get('guard_service_status')!='REVIEW_REQUIRED_NOT_RELEASED':raise ValueError('GUARD_NOT_BLOCKED')
    return results,report


def check_mask(mask_path: Path, size:tuple[int,int], bbox:list[float]):
    from PIL import Image
    with Image.open(mask_path) as im:
        im.load()
        if im.mode not in {'L','1'}:raise ValueError('MASK_NOT_SINGLE_CHANNEL')
        if im.size!=size:raise ValueError('MASK_IMAGE_SHAPE_MISMATCH')
        binary=im.convert('L').point(lambda v:255 if v>=128 else 0)
        mask_bounds=binary.getbbox()
        if mask_bounds is None:raise ValueError('EMPTY_MASK')
        total=binary.histogram()[255]
        left,top,right,bottom=map(float,bbox)
        # 2 px margin accommodates rasterization of floating-point V5 boxes.
        ix1=max(0,int(math.floor(left))-2);iy1=max(0,int(math.floor(top))-2)
        ix2=min(size[0],int(math.ceil(right))+2);iy2=min(size[1],int(math.ceil(bottom))+2)
        roi=binary.crop((ix1,iy1,ix2,iy2))
        inside=roi.histogram()[255]
        coverage=inside/total
    return {'foreground_pixels':total,'mask_tight_bbox_xyxy':list(mask_bounds),
            'mask_pixel_coverage_in_v5_bbox_margin2':round(coverage,6),
            'geometry_review_required':coverage<0.95}


def audit(raw_dir:Path,live_dir:Path,image_root:Path):
    # Only two upstream-approved roots. Check all originals against frozen raw record.
    v5=check_v5_raw(raw_dir,image_root)
    rows,guard=check_live_records(live_dir)
    per=[]
    for key in sorted({(r['case_id'],r['target_instance_id']) for r in rows}):
        rec=v5.get(key)
        if rec is None:raise ValueError(f'LIVE_TARGET_NOT_IN_V5_RAW:{key}')
        samples=[r for r in rows if (r['case_id'],r['target_instance_id'])==key]
        case,iid=key
        marked=set()
        for row in samples:
            if row['image_sha256']!=rec['image_sha256'] or checked_bbox(row['v5_bbox_xyxy'])!=tuple(rec['bbox']):
                raise ValueError('LIVE_ORIGINAL_SHA_OR_BBOX_MISMATCH')
            if row.get('v5_category_unverified')!=rec['category']:
                raise ValueError('LIVE_V5_CATEGORY_CHANGED')
            if row['_validated_event']['bbox_xyxy']!=rec['bbox']:
                raise ValueError('EVENT_BBOX_CHANGED')
            marked.add(row['marked_image_sha256'])
        if len(marked)!=1:raise ValueError('TASKS_USING_DIFFERENT_MARKED_IMAGE')
        expected_marked=next(iter(marked))
        marked_dir=live_dir/'marked_inputs'
        if not marked_dir.is_dir():raise ValueError('MARKED_INPUTS_MISSING')
        found=[p for p in marked_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'} and sha256_file(p)==expected_marked]
        if len(found)!=1:raise ValueError('MARKED_FILE_SHA_NOT_UNIQUE_OR_MISSING')
        mask=checked_path(raw_dir/f'{case}_{iid}_mask.png',raw_dir,name='V5 development mask')
        geometry=check_mask(mask,rec['size'],rec['bbox'])
        per.append({'case_id':case,'instance_id':iid,'v5_category_not_gt':rec['category'],
                    'source_image_sha256':rec['image_sha256'], 'v5_bbox_xyxy':rec['bbox'],
                    'source_mask_file_sha256':sha256_file(mask),
                    'source_mask_path':str(mask),'marked_image_sha256':expected_marked,
                    'marked_image_path':str(found[0]),'host_tasks_verified':len(samples),
                    'frozen_v5_mask_file_link_verified':True,
                    'producer_time_roi_mask_identity_verified':False,
                    'model_visual_attention_verified':False,'release_allowed':False,**geometry})
    return {'schema':'prd32_dev60_frozen_v5_to_qwen_live_lineage_v5',
            'status':'FROZEN_V5_ARTIFACT_TO_LIVE_QWEN_BYTES_VERIFIED_PRODUCER_UNVERIFIED',
            'selected_dev_instances':len(per),'total_dev60_frozen_instances':len(v5),
            'host_tasks_checked':len(rows),'geometry_review_instances':sum(int(x['geometry_review_required']) for x in per),
            'frozen_v5_raw_sha256':V5_RAW_SHA,'frozen_v5_checkpoint_sha256':V5_CKPT_SHA,
            'guard_status':guard['guard_service_status'],
            'roi_mask_producer_time_verified':False,'independent_holdout_used':False,
            'original_prediction_overwritten':False,'release_allowed':False,
            'instances':per}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir',type=Path,default=Path('/workspace/PRD32_DEV60_V5_REAL_20261030_run01'))
    p.add_argument('--image-root',type=Path,default=Path('/workspace/fashion_data'))
    p.add_argument('--live-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists():raise SystemExit('OUTPUT_EXISTS_REFUSING_OVERWRITE')
    if restricted(a.out) or restricted(a.raw_dir) or restricted(a.live_dir) or restricted(a.image_root):
        raise SystemExit('RESTRICTED_PATH')
    if not a.live_dir.is_dir():raise SystemExit('LIVE_DIR_MISSING')
    report=audit(a.raw_dir,a.live_dir,a.image_root)
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'frozen_to_live_lineage_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    cols=('case_id','instance_id','v5_category_not_gt','source_image_sha256','source_mask_file_sha256',
          'marked_image_sha256','host_tasks_verified','foreground_pixels','mask_pixel_coverage_in_v5_bbox_margin2',
          'geometry_review_required','frozen_v5_mask_file_link_verified','producer_time_roi_mask_identity_verified','release_allowed')
    with (a.out/'lineage_instances.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(report['instances'])
    print('FROZEN_LINEAGE_EVIDENCE_COMPLETE',report['selected_dev_instances'],
          'instances',report['host_tasks_checked'],'host tasks',
          'PRODUCER_IDENTITY_UNVERIFIED', 'NO_RELEASE')


if __name__=='__main__':
    try:main()
    except Exception as e:
        print('FROZEN_LINEAGE_BLOCKED',type(e).__name__,str(e))
        raise SystemExit(2)
