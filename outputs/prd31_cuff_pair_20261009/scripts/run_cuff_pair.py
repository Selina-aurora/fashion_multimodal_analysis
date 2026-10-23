"""Paired development experiment; never installs or modifies the product backend."""
import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = 'short sleeve cuff . long sleeve cuff .'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def iou(a, b):
    if not a or not b:
        return 0.0
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - x*y
    return x*y/area if area > 0 else 0.0


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items() if k != 'mask'}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, 'item'):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def paired_ground(backend, prompts, image, roi, baseline, candidate, order):
    # Same ROI, weights, thresholds, phrase gate, geometry guard and SAM in both arms.
    original = prompts['cuff']
    result = {}
    try:
        for arm in order:
            prompts['cuff'] = baseline if arm == 'baseline' else candidate
            prediction, elapsed = backend.ground(image, roi, 'cuff')
            result[arm] = {'prediction': json_safe(prediction), 'elapsed_ms': elapsed}
    finally:
        prompts['cuff'] = original
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    pack = json.loads((ROOT/'development_labels.json').read_text())
    from PIL import Image, ImageDraw
    for case in pack['cases']:
        source = ROOT/case['image']
        if digest(source) != case['image_sha256']:
            raise ValueError('Changed image: '+case['sample_id'])
        with Image.open(source) as im:
            assert list(im.size) == case['image_size']
            for box in case['draft_cuff_boxes']:
                assert 0 <= box[0] < box[2] <= im.width and 0 <= box[1] < box[3] <= im.height
    if args.validate_only:
        print('VALIDATED: 8 images, 5 draft positives, 3 draft negatives; no GPU/model call.')
        return
    from fashion_multimodal_analysis.integration import model_backends as module
    config = json.loads((ROOT/'run_config.json').read_text())
    cp = Path(config['segmentation_checkpoint'])
    if digest(cp) != pack['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from original V5; experiment stopped.')
    output = Path('reports/reruns/recovery_20261009')/('cuff_pair_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    baseline = module.REGION_PROMPTS['cuff']
    if baseline.strip() == CANDIDATE.strip():
        raise ValueError('Baseline equals candidate; no contrast.')
    backend = module.ModelBackends(config)
    records = []
    for number, case in enumerate(pack['cases']):
        image = Image.open(ROOT/case['image']).convert('RGB')
        rois, segment_ms = backend.segment(image)
        predictions = []
        # Ground all eligible predicted garments BEFORE consulting target labels.
        for ri, roi in enumerate(rois):
            if roi.category not in ('top', 'outerwear', 'dress'):
                continue
            order = ('baseline', 'candidate') if (number+ri)%2 == 0 else ('candidate', 'baseline')
            pair = paired_ground(backend, module.REGION_PROMPTS, image, roi, baseline, CANDIDATE, order)
            predictions.append({'instance_id': roi.instance_id, 'category': roi.category,
                                'bbox': roi.bbox, 'score': roi.score, 'call_order': order, **pair})
        # Evaluation only: native target box chooses the predicted garment by IoU.
        matches = [(iou(case['garment_bbox'], p['bbox']), i) for i, p in enumerate(predictions)]
        best = max(matches, default=(0.0, -1))
        matched = predictions[best[1]] if best[0] >= config['bbox_match_iou'] else None
        record = {'sample_id': case['sample_id'], 'draft_presence': case['draft_presence'],
                  'review_status': 'ASSISTANT_DRAFT_PENDING_HUMAN_REVIEW',
                  'garment_match_iou': best[0], 'segment_ms': segment_ms,
                  'matched_category_correct': bool(matched and matched['category'] == case['category']),
                  'all_predicted_garments': predictions, 'arms': {}}
        draw = ImageDraw.Draw(image)
        draw.rectangle(case['garment_bbox'], outline='yellow', width=2)
        for b in case['draft_cuff_boxes']:
            draw.rectangle(b, outline='orange', width=3)
        for arm, color in [('baseline', 'lime'), ('candidate', 'cyan')]:
            pred = matched[arm]['prediction'] if matched else {'status': 'segmentation_missed', 'bbox': None}
            box = pred.get('bbox')
            detected = pred.get('status') == 'detected' and bool(box)
            overlap = max((iou(box, b) for b in case['draft_cuff_boxes']), default=0.0) if detected else 0.0
            # Draft diagnostics: rejected proposals are NOT correct absence predictions.
            record['arms'][arm] = {'prediction': pred, 'draft_best_cuff_iou': overlap,
                                  'draft_positive_bbox50_hit': bool(case['draft_presence']=='present' and overlap>=0.5),
                                  'draft_negative_false_detection': bool(case['draft_presence']=='absent' and detected)}
            if detected:
                draw.rectangle(box, outline=color, width=3)
        image.save(output/(case['sample_id']+'_pair.jpg'))
        records.append(record)
        (output/'predictions.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
        print(f"{number+1}/8 {case['sample_id']} match={best[0]:.3f}", flush=True)
    summary = {'status': 'DEVELOPMENT_ONLY_PENDING_LABEL_REVIEW', 'formal_accuracy': None,
               'baseline_prompt': baseline, 'candidate_prompt': CANDIDATE,
               'changed_variable': 'cuff_prompt_only', 'cases': len(records),
               'segmentation_target_matches': sum(r['garment_match_iou']>=0.5 for r in records),
               'positive_denominator': 5, 'negative_denominator': 3,
               'labels_sha256': digest(ROOT/'development_labels.json'),
               'checkpoint_sha256': digest(cp), 'config': config,
               'backend_sha256': digest(module.__file__),
               'latency_note': 'Cold and sequential small-sample timings; not PRD latency acceptance.',
               'arms': {arm: {'draft_positive_hits_bbox50': sum(r['arms'][arm]['draft_positive_bbox50_hit'] for r in records),
                              'draft_negative_false_detections': sum(r['arms'][arm]['draft_negative_false_detection'] for r in records)}
                        for arm in ('baseline','candidate')}}
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    for f in ('development_labels.json', 'run_config.json'):
        (output/f).write_bytes((ROOT/f).read_bytes())
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in output.rglob('*'):
            if f.is_file(): z.write(f, f.relative_to(output))
    print('RESULT_ZIP='+str(archive.resolve()), flush=True)


if __name__ == '__main__':
    main()
