"""Read-only diagnostic of YOLO pose keypoints against eight approved sleeve labels.

This script DOES NOT infer sleeve cuffs, sleeve coverage, or production arm selection.
Human annotation points are used *only* for matching/comparing pose detections.
No training manifests, labels, or frozen evaluation files are modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean, median
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, ImageFont

WORKSPACE = Path('/workspace')
REPO = WORKSPACE / 'fashion_multimodal_analysis'
LABELS = WORKSPACE / 'sleeve_pilot8_controlled_20261012' / 'pilot8_candidates_FINAL.json'
LABELS_EXPECTED_SHA256 = 'cb34eff7920e99c6803546afc4aa29c5ae3c0a1adacd313ecd0a2619cc8cef15'
OUT = WORKSPACE / 'sleeve_pose_sandbox' / 'pilot8_pose_batch_20261012'
WEIGHT_CANDIDATES = [
    WORKSPACE / 'sleeve_pose_sandbox' / 'yolo11n-pose.pt',
    WORKSPACE / 'yolo11n-pose.pt',
    REPO / 'yolo11n-pose.pt',
]
JOINT_INDEX = {
    'left': {'shoulder': 5, 'elbow': 7, 'wrist': 9},
    'right': {'shoulder': 6, 'elbow': 8, 'wrist': 10},
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def resolve_image(raw: str) -> Path:
    p = Path(raw)
    return (p if p.is_absolute() else REPO / p).resolve()


def d(a, b) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])


def norm_points(annotation: dict, width: int, height: int) -> dict:
    return {k: (float(v[0])*width, float(v[1])*height) for k, v in annotation['points'].items()}


def available_keypoints(keypoints, person_idx):
    xy = keypoints.xy[person_idx].cpu().tolist()
    cf = None if keypoints.conf is None else keypoints.conf[person_idx].cpu().tolist()
    return xy, cf


def choose_arm(keypoints, gt, diag):
    """Select the closest detected arm by human GT, for diagnostics ONLY."""
    best = None
    for pi in range(len(keypoints.xy)):
        xy, conf = available_keypoints(keypoints, pi)
        for side, indices in JOINT_INDEX.items():
            valid = True
            distances = {}
            joint_confs = {}
            for joint, idx in indices.items():
                coord = xy[idx]
                confidence = None if conf is None else float(conf[idx])
                joint_confs[joint] = confidence
                if (len(coord) < 2 or not all(math.isfinite(float(x)) for x in coord[:2])
                    or coord == [0.0,0.0] or (confidence is not None and confidence < 0.25)):
                    valid = False
                    break
                distances[joint] = d(coord, gt[joint]) / diag
            if valid:
                score = sum(distances.values()) / 3
                if best is None or score < best['mean_joint_distance']:
                    best = {
                        'person_index': pi,
                        'coco_anatomical_side': side,
                        'joint_coordinates': {k: [float(xy[v][0]),float(xy[v][1])] for k,v in indices.items()},
                        'joint_confidences': joint_confs,
                        'joint_error_normalized_diagonal': distances,
                        'mean_joint_distance': score,
                    }
    return best


def draw_pair(path, result, gt, matching, output):
    with Image.open(path) as im:
        im = im.convert('RGB')
        width, height = im.size
    # Detection render intentionally includes all people and limbs.
    yolo = Image.fromarray(result.plot()[:, :, ::-1]).convert('RGB')
    # Raw image + human reference and selected detected arm.
    annotated = im.copy()
    dd = ImageDraw.Draw(annotated)
    def draw_joint(pt,color,label,radius=6):
        x,y=pt
        dd.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color,outline='black',width=1)
        dd.text((x+8,y-13),label,fill=color,stroke_width=2,stroke_fill='black')
    for joint in ('shoulder','elbow','wrist','cuff'):
        if joint in gt:
            draw_joint(gt[joint], 'yellow', 'GT_'+joint[0].upper())
    if matching:
        for joint,pt in matching['joint_coordinates'].items():
            draw_joint(pt,'cyan','YOLO_'+joint[0].upper(),radius=5)
    # Resize both to legible per-image width, max 680px, preserve aspect ratio.
    scale=min(1.0, 680/max(width,1))
    scaled=(max(1,int(width*scale)),max(1,int(height*scale)))
    yolo=yolo.resize(scaled,Image.Resampling.LANCZOS)
    annotated=annotated.resize(scaled,Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(scaled[0]*2,scaled[1]+45),'white')
    dc=ImageDraw.Draw(canvas)
    dc.text((10,12),'YOLO pose (all people)',fill='black')
    dc.text((scaled[0]+10,12),'Yellow: manual GT; Cyan: YOLO (diagnostic match)',fill='black')
    canvas.paste(yolo,(0,45))
    canvas.paste(annotated,(scaled[0],45))
    output.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(output,quality=87,optimize=True)


def main():
    import torch
    from ultralytics import YOLO
    if not LABELS.is_file():
        sys.exit(f'MISSING_PILOT8_LABELS: {LABELS}')
    if sha256(LABELS) != LABELS_EXPECTED_SHA256:
        sys.exit(f'LABELS_SHA256_MISMATCH: expected={LABELS_EXPECTED_SHA256}, found={sha256(LABELS)}')
    payload=json.loads(LABELS.read_text(encoding='utf-8-sig'))
    anns=payload['annotations']
    if len(anns)!=8 or len({a['record_id'] for a in anns})!=8:
        sys.exit('PILOT8_COUNT_OR_ID_MISMATCH')
    weight=next((p for p in WEIGHT_CANDIDATES if p.is_file()), None)
    if weight is None:
        print('WEIGHT_NOT_FOUND. Please locate the existing yolo11n-pose.pt, do not download another model yet.')
        print('SUGGESTED_CHECK: find /workspace -maxdepth 5 -name "*pose*.pt" -type f')
        sys.exit(2)
    images=[]
    for a in anns:
        if a.get('state') != 'measurable' or not a.get('same_arm_confirmed'):
            sys.exit(f'NONMEASURABLE_LABEL: {a["record_id"]}')
        path=resolve_image(a['source_image'])
        if not path.is_file():
            sys.exit(f'MISSING_IMAGE: {path}')
        if sha256(path)!=a['source_image_sha256']:
            sys.exit(f'IMAGE_SHA256_MISMATCH: {a["record_id"]}')
        images.append(path)
    OUT.mkdir(parents=True,exist_ok=True)
    device=0 if torch.cuda.is_available() else 'cpu'
    model=YOLO(str(weight))
    records=[]
    for idx,(a,path) in enumerate(zip(anns,images),start=1):
        with Image.open(path) as im:
            width,height=im.size
        diagonal=math.hypot(width,height)
        gt=norm_points(a,width,height)
        result=model.predict(source=str(path),imgsz=640,device=device,conf=0.25,verbose=False)[0]
        n_people=0 if result.boxes is None else len(result.boxes)
        best=(choose_arm(result.keypoints,gt,diagonal)
              if result.keypoints is not None and len(result.keypoints.xy)>0 else None)
        output_path=OUT / 'visuals' / f'{idx:02d}_{a["record_id"]}.jpg'
        draw_pair(path,result,gt,best,output_path)
        rec={
            'record_id':a['record_id'],
            'current_class':a['current_class'],
            'manual_ratio':a['ratio'],
            'manual_side_in_image':a['side'],
            'image':str(path),
            'persons_detected':n_people,
            'matched_arm_available':best is not None,
            'matched_arm_diagnostic_only':best,
            'visual_path':str(output_path),
        }
        records.append(rec)
        print(f'POSE_BATCH8 {idx}/8 | {a["record_id"]} | people={n_people} | mean_keypoint_err=' +
              (f'{best["mean_joint_distance"]:.4f}' if best else 'NA'),flush=True)
    errors=[r['matched_arm_diagnostic_only']['mean_joint_distance'] for r in records if r['matched_arm_available']]
    summary={
        'status':'POSE_8_DIAGNOSTIC_COMPLETED',
        'labels_sha256': LABELS_EXPECTED_SHA256,
        'pose_weights_path':str(weight),
        'pose_weights_sha256':sha256(weight),
        'samples':len(records),
        'detections_with_matching_arm':len(errors),
        'mean_keypoint_error_normalized_image_diagonal': mean(errors) if errors else None,
        'median_keypoint_error_normalized_image_diagonal':median(errors) if errors else None,
        'gt_used_for_arm_matching_diagnostic_only':True,
        'no_cuff_detection':True,
        'no_automatic_sleeve_ratio_prediction':True,
        'no_training_or_data_change':True,
        'notes':'Keypoint error is not sleeve-length error. Manual GT used solely for side/person pairing; do not report as fully autonomous pipeline accuracy.',
    }
    (OUT/'pose_8_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'pose_8_details.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'pose_8_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=[
            'record_id','class','manual_side','persons','matched_side_coco','shoulder_error_norm','elbow_error_norm','wrist_error_norm','mean_error_norm','shoulder_conf','elbow_conf','wrist_conf',
        ])
        writer.writeheader()
        for r in records:
            arm=r['matched_arm_diagnostic_only'] or {}
            err=arm.get('joint_error_normalized_diagonal',{})
            conf=arm.get('joint_confidences',{})
            writer.writerow({
                'record_id':r['record_id'],'class':r['current_class'],'manual_side':r['manual_side_in_image'],
                'persons':r['persons_detected'],'matched_side_coco':arm.get('coco_anatomical_side',''),
                'shoulder_error_norm':err.get('shoulder',''),'elbow_error_norm':err.get('elbow',''),
                'wrist_error_norm':err.get('wrist',''),'mean_error_norm':arm.get('mean_joint_distance',''),
                'shoulder_conf':conf.get('shoulder',''),'elbow_conf':conf.get('elbow',''),
                'wrist_conf':conf.get('wrist',''),
            })
    zpath=WORKSPACE/'sleeve_pose_batch8_results_20261012.zip'
    with ZipFile(zpath,'w',ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file(): z.write(p,arcname=p.relative_to(OUT))
    print('POSE_BATCH8_DIAGNOSTIC_COMPLETED')
    print('SAMPLES:',len(records))
    print('MATCHED_ARMS:',len(errors))
    print('MEAN_JOINT_ERROR_NORM_DIAG:',summary['mean_keypoint_error_normalized_image_diagonal'])
    print('NO_TRAINING_OR_DATA_CHANGE: TRUE')
    print('UPLOAD_THIS_ZIP:',zpath)

if __name__=='__main__':
    main()
