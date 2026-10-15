"""Evaluate 18 reviewed garments with existing CLIP attribute helpers.

No training or production changes. GT ROI diagnoses attributes; predicted ROI
also counts segmentation association failures. This is a small validation
diagnostic, not blind PRD acceptance. Prompt scores are not calibrated certainty.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

NAMES = ('primary_color', 'pattern', 'sleeve_length')
BASE = 'outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth'
BASE_SHA = 'c5ba795225349b1c521aac0517ac7c493d9818dcc17d558ac4a3f6f8669452f0'
VAL_SHA = '39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93'
ANNOTATION_SHA = '578631308a786d0f49166a80b067476a65ad3fc20976aaf7d916988fc760d8c2'
SELECTION_SHA = 'f36d08d710565703406c5fa48c29ccea78ea2ca30429ea2991b5f2bed60d47cd'
PATTERN_REVIEW_FLAG = 'v5_validation_00027'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def iou(a, b):
    area = lambda q: max(0, q[2]-q[0]) * max(0, q[3]-q[1])
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = area(a) + area(b) - intersection
    return intersection / union if union else 0.


def metric(records, name, omit_flagged=False):
    """Unknown GT excluded; missing predictions remain end-to-end errors."""
    usable = [r for r in records if r['truth'][name] != 'unknown'
              and not (omit_flagged and r['record_id'] == PATTERN_REVIEW_FLAG)]
    emitted = [r for r in usable if r['prediction'].get(name, {}).get('label')]
    correct = sum(r['prediction'].get(name, {}).get('label') == r['truth'][name] for r in usable)
    confusion = {}
    for r in usable:
        gt = r['truth'][name]
        pred = r['prediction'].get(name, {}).get('label') or '<missing>'
        confusion.setdefault(gt, {})[pred] = confusion.setdefault(gt, {}).get(pred, 0) + 1
    return {'eligible_targets': len(usable), 'correct': correct,
            'wrong_or_missing': len(usable)-correct, 'emitted': len(emitted),
            'missing': len(usable)-len(emitted),
            'accuracy_all_eligible': correct/len(usable) if usable else None,
            'accuracy_on_emitted': correct/len(emitted) if emitted else None,
            'unknown_gt_excluded': sum(r['truth'][name] == 'unknown' for r in records),
            'confusion': confusion}


def validate_annotations(data, selected):
    if data.get('schema_version') != 'attribute_review_v1' or data.get('selection_sha256') != SELECTION_SHA:
        raise ValueError('Wrong annotation version or selection')
    cases = data['cases']
    if len(cases) != 18 or len({c['record_id'] for c in cases}) != 18:
        raise ValueError('Expected exactly 18 unique targets')
    options = selected['options']
    native = {c['record_id']: c for c in selected['cases']}
    for c in cases:
        if c['review_status'] != 'confirmed':
            raise ValueError('Unconfirmed target: '+c['record_id'])
        for key in ('category', 'garment_id', 'garment_bbox', 'source_image_sha256', 'source_image_reference', 'source_dataset'):
            if c[key] != native[c['record_id']][key]:
                raise ValueError('Target metadata changed: '+c['record_id']+'/'+key)
        for name in NAMES:
            if c['attributes'][name] not in dict(options[name]):
                raise ValueError('Invalid label: '+name)
    return cases


def main():
    # Offline loading only; use the existing GPU cache, no downloads.
    os.environ.setdefault('HF_HOME', '/workspace/hf_cache')
    os.environ.setdefault('HF_HUB_CACHE', '/workspace/hf_cache/hub')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from fashion_multimodal_analysis.common.paths import project_root, resolve_path
    from fashion_multimodal_analysis.common.schema import CLASS_TO_ID
    from fashion_multimodal_analysis.segmentation.modeling import build_evaluation_model
    from fashion_multimodal_analysis.image_processing.masks import reconstruct_full_mask
    from fashion_multimodal_analysis.integration.pipeline import crop_bounds
    from fashion_multimodal_analysis.attributes.color import run_primary_color_baseline_v1 as color
    from fashion_multimodal_analysis.attributes.pattern import run_pattern_hierarchical_v3 as pattern
    from fashion_multimodal_analysis.attributes.design import run_design_attribute_labels_v1 as design

    here = Path(__file__).resolve().parent
    root = project_root()
    annotation = here / 'attribute_review_confirmed_20261009.json'
    if sha(annotation) != ANNOTATION_SHA:
        raise ValueError('Frozen confirmation file changed')
    selected = json.loads((here / 'selection_metadata.json').read_text())
    data = json.loads(annotation.read_text())
    cases = validate_annotations(data, selected)
    val = resolve_path('configs/prd_8class_val_v2_full_targets.csv')
    base = root / BASE
    if sha(val) != VAL_SHA or sha(base) != BASE_SHA:
        raise ValueError('Validation manifest or V5 checkpoint changed')
    with val.open(encoding='utf-8-sig', newline='') as stream:
        rows = {r['record_id']: r for r in csv.DictReader(stream)}
    # Check all images, target boxes and masks before expensive model loading.
    inputs = {}
    for c in cases:
        row = rows[c['record_id']]
        source = resolve_path(row['source_image'])
        box = [float(row['bbox_'+k]) for k in ('x1','y1','x2','y2')]
        if sha(source) != c['source_image_sha256'] or box != c['garment_bbox'] or row['garment_category'] != c['category']:
            raise ValueError('Original target changed: '+c['record_id'])
        picture = Image.open(source).convert('RGB')
        mask = np.asarray(reconstruct_full_mask(resolve_path(row['mask_path']), picture.size, tuple(int(v) for v in box))) > 0
        if mask.shape != (picture.height, picture.width) or not mask.any():
            raise ValueError('Invalid original mask: '+c['record_id'])
        inputs[c['record_id']] = (picture, mask)
    if not torch.cuda.is_available():
        raise RuntimeError('Run in GPU py312 environment')
    device = torch.device('cuda')
    model_id = 'openai/clip-vit-base-patch32'
    processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
    clip = CLIPModel.from_pretrained(model_id, local_files_only=True).to(device).eval()
    segmenter = build_evaluation_model(640, 1024)
    cp = torch.load(base, map_location='cpu', weights_only=False)
    segmenter.load_state_dict(cp['model_state_dict'], strict=True)
    del cp
    segmenter.to(device).eval()
    with torch.inference_mode():
        color_proto = color.build_color_prototypes(clip, processor, device)
        gate_proto = pattern.build_class_prototypes(pattern.STAGE1_PROMPTS, clip, processor, device)
        pattern_proto = pattern.build_class_prototypes(pattern.STAGE2_PROMPTS, clip, processor, device)
        sleeve_proto = pattern.build_class_prototypes({k:[v] for k,v in design.SLEEVE_PROMPTS.items()}, clip, processor, device)
    out = root / 'reports/reruns/recovery_20261009' / ('attribute_results_'+time.strftime('%Y%m%d_%H%M%S')+f'_{time.time_ns()%1000000:06d}')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'views').mkdir()
    records = {'gt_roi': [], 'predicted_roi': []}

    def attributes(image, mask, box, tag):
        bounds = crop_bounds(box, image.size)
        crop = image.crop(bounds)
        crop_mask = Image.fromarray(mask.astype('uint8')*255).crop(bounds)
        if not np.asarray(crop_mask).any():
            return {}, bounds
        whole = Image.composite(crop, Image.new('RGB', crop.size, 'white'), crop_mask)
        crop.save(out/'views'/(tag+'_raw.jpg'), quality=90)
        whole.save(out/'views'/(tag+'_masked.jpg'), quality=90)
        color_feature = color.robust_garment_embedding(crop, crop_mask, clip, processor, device)
        color_pred = color.classify_color(color_feature, *color_proto, clip)
        feature = pattern.encode_image(whole, clip, processor, device)
        gate = pattern.classify_with_prototypes(feature, *gate_proto, clip)
        if pattern.binary_route(gate['top1_label']) == 'visible_pattern':
            pred = pattern.classify_with_prototypes(feature, *pattern_proto, clip)
            pattern_label, pattern_score = pred['top1_label'], pred['top1_score']
        else:
            pattern_label, pattern_score = 'solid', gate['top1_score']
        sleeve = pattern.classify_with_prototypes(feature, *sleeve_proto, clip)
        return {'primary_color': {'label':color_pred['primary_color'], 'score':color_pred['top1_score']},
                'pattern': {'label':pattern_label, 'score':pattern_score, 'stage1_bucket':gate['top1_label']},
                'sleeve_length': {'label':sleeve['top1_label'], 'score':sleeve['top1_score']}}, bounds

    print('MODELS_READY; 18 targets, no training', flush=True)
    with torch.inference_mode():
        for number, c in enumerate(cases, 1):
            rid = c['record_id']
            image, native_mask = inputs[rid]
            common = {'record_id':rid, 'category':c['category'], 'source_dataset':c['source_dataset'],
                      'truth':c['attributes'], 'review_comment':c.get('comment',''),
                      'pattern_label_review_flag':rid == PATTERN_REVIEW_FLAG}
            prediction, bounds = attributes(image, native_mask, c['garment_bbox'], rid+'_gt')
            records['gt_roi'].append({**common, 'prediction':prediction, 'status':'GT_ROI' if prediction else 'EMPTY_GT_CROP_MASK', 'bounds':bounds})
            rgb = np.asarray(image, dtype=np.float32).copy()/255.
            pred = segmenter([torch.from_numpy(rgb).permute(2,0,1).contiguous().to(device)])[0]
            boxes = pred['boxes'].cpu().tolist()
            ids = [i for i,s in enumerate(pred['scores'].cpu().tolist()) if s >= .4]
            chosen = max(ids, key=lambda i:iou(boxes[i], c['garment_bbox'])) if ids else None
            prediction = {}; bounds = None; overlap = 0.; predicted_box = None; predicted_class = None
            status = 'NO_GARMENT_PREDICTION'
            if chosen is not None:
                predicted_box = boxes[chosen]
                predicted_class = int(pred['labels'][chosen])
                overlap = iou(predicted_box, c['garment_bbox'])
                status = 'GARMENT_IOU_BELOW_0.5' if overlap < .5 else 'GARMENT_CLASS_MISMATCH' if predicted_class != CLASS_TO_ID[c['category']] else 'MATCHED_PREDICTED_TARGET'
                if status == 'MATCHED_PREDICTED_TARGET':
                    mask = pred['masks'][chosen,0].cpu().numpy() >= .5
                    prediction, bounds = attributes(image, mask, predicted_box, rid+'_predicted')
                    if not prediction: status = 'EMPTY_PREDICTED_CROP_MASK'
            records['predicted_roi'].append({**common, 'prediction':prediction, 'status':status, 'bounds':bounds,
                                             'garment_iou':overlap, 'predicted_garment_box':predicted_box, 'predicted_class_id':predicted_class})
            dump(out/'per_target_results.json', records)
            print(f'TARGET_DONE {number}/18 {rid}: {status}', flush=True)
    summary = {'status':'ATTRIBUTE_DIAGNOSTIC_COMPLETED', 'images':18, 'formal_prd_acceptance':False,
               'annotations_sha256':ANNOTATION_SHA, 'checkpoint_sha256':BASE_SHA, 'val_manifest_sha256':VAL_SHA,
               'clip_model':model_id, 'segmentation_min_size':640, 'segmentation_max_size':1024,
               'score_threshold':.4, 'mask_threshold':.5, 'association_iou':.5,
               'association_policy':'Highest-IoU prediction above score threshold, then require IoU>=0.5 and correct class; one reviewed target per image',
               'metrics':{role:{name:metric(items,name) for name in NAMES} for role,items in records.items()},
               'pattern_sensitivity_excluding_case_2':{role:metric(items,'pattern',True) for role,items in records.items()},
               'matched_predicted_targets':sum(r['status']=='MATCHED_PREDICTED_TARGET' for r in records['predicted_roi']),
               'pattern_label_review_note':'Case 2 confirmed solid with comment sleeve stripes; preserve confirmed label and report sensitivity without this target.',
               'limitations':['Original segmentation validation images; small diagnostic, not blind end-to-end acceptance.',
                              'Only specified garments reviewed, not every instance in each image.',
                              'Unknown GT excluded per attribute; missing association counts wrong in all-eligible metrics.',
                              'GT target association used only for evaluation; GT attribute labels never used during inference.',
                              'Uses existing color/pattern/sleeve helpers and masked garment views; no region grounding/SAM calls.',
                              'Prompt-relative scores are not calibrated confidence; no accuracy inferred from them.']}
    dump(out/'summary.json', summary)
    dump(out/'confirmed_annotations.json', data)
    helper_files = [color.__file__, pattern.__file__, design.__file__]
    dump(out/'code_provenance.json', {'runner_sha256':sha(Path(__file__)), 'helpers':{str(p):sha(p) for p in helper_files}})
    table = []
    for role, items in records.items():
        for r in items:
            for name in NAMES:
                truth = r['truth'][name]; pred = r['prediction'].get(name,{})
                table.append({'role':role,'record_id':r['record_id'],'category':r['category'],'attribute':name,
                              'truth':truth,'predicted':pred.get('label',''),'score':pred.get('score',''),
                              'eligible':truth!='unknown','correct':pred.get('label')==truth if truth!='unknown' else '',
                              'target_status':r['status'],'review_comment':r['review_comment']})
    with (out/'attribute_results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(out.rglob('*')):
            if path.is_file():zipped.write(path,path.relative_to(out))
    print(json.dumps(summary['metrics'],ensure_ascii=False,indent=2),flush=True)
    print('UPLOAD_THIS_ZIP:',archive,flush=True)


if __name__ == '__main__':
    main()
