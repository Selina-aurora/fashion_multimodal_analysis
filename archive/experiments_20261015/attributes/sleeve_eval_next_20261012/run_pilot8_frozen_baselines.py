#!/usr/bin/env python3
"""Read-only 8-case sleeve inference: frozen historical CLIP prompt and linear head.
NOT an evaluation of automatic continuous-ratio prediction or a formal PRD test.
Never trains, tunes, or merges any dataset.
"""
import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

TRAIN_SHA = 'e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f'
VAL_SHA = '39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93'
HEAD_SHA = '826bf1cdd911981fa5be8b176c76e4240a08fb01e56c01d5dade83011beec17c'
LABELS = ['sleeveless', 'cap', 'short', 'elbow', 'three_quarter', 'long']
DEFAULT_HEAD = 'reports/reruns/recovery_20261009/sleeve_linear_train_20261009_073022_027703/sleeve_head_best.pth'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def metrics(items, key):
    correct = sum(row[key] == row['truth'] for row in items)
    by_class = {}
    for label in LABELS:
        group = [r for r in items if r['truth'] == label]
        if group:
            by_class[label] = {'n': len(group), 'correct': sum(r[key] == label for r in group)}
    return {'eligible': len(items), 'correct': correct, 'accuracy': correct / len(items) if items else None,
            'by_class': by_class}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='/workspace/fashion_multimodal_analysis')
    parser.add_argument('--pilot', default='/workspace/sleeve_pilot8_controlled_20261012/pilot8_candidates_FINAL.json')
    parser.add_argument('--head', default=DEFAULT_HEAD)
    parser.add_argument('--core400-hashes', default='/workspace/core400_source_sha256_20261012.txt')
    parser.add_argument('--preflight-report', default='/workspace/pilot8_preflight_core400_report.json')
    parser.add_argument('--output-root', default='/workspace/sleeve_pilot8_evaluation_20261012')
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    pilot_path = Path(args.pilot).resolve()
    head_path = Path(args.head)
    if not head_path.is_absolute(): head_path = repo / head_path
    if not head_path.is_file():
        raise FileNotFoundError(f'缺少10月9日训练的分类头权重: {head_path}; 请勿重新训练，先找回权重')
    if sha256(head_path) != HEAD_SHA:
        raise ValueError('历史分类头 SHA256 不匹配，禁止静默替换模型')
    train_file = repo / 'configs/prd_8class_train_v5_balanced150_20261008.csv'
    val_file = repo / 'configs/prd_8class_val_v2_full_targets.csv'
    if sha256(train_file) != TRAIN_SHA or sha256(val_file) != VAL_SHA:
        raise ValueError('历史训练/验证清单变动，停止评估')
    preflight = json.loads(Path(args.preflight_report).read_text(encoding='utf-8'))
    if preflight.get('status') != 'PREFLIGHT_PASSED_NO_OVERLAP' or not preflight.get('checks', {}).get('core400_overlap_free'):
        raise ValueError('缺少已通过的 Core400 重叠检查结果')
    payload = json.loads(pilot_path.read_text(encoding='utf-8'))
    cases = payload['annotations']
    if len(cases) != 8 or len({c['record_id'] for c in cases}) != 8:
        raise ValueError('Pilot8 样本数量或唯一性有误')
    hashes = Path(args.core400_hashes).read_text(encoding='utf-8').splitlines()
    if len(hashes) != 400 or len(set(hashes)) != 400 or any(len(h) != 64 for h in hashes):
        raise ValueError('Core400 图像 SHA256 清单无效')
    core_hashes = set(hashes)
    rows = {r['record_id']: r for r in read_csv(train_file)}
    # Read checkpoint safely; no arbitrary pickle execution is needed.
    import torch
    from PIL import Image
    os.environ.setdefault('HF_HOME', '/workspace/hf_cache')
    os.environ.setdefault('HF_HUB_CACHE', '/workspace/hf_cache/hub')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from transformers import CLIPModel, CLIPProcessor
    from fashion_multimodal_analysis.common.paths import resolve_path
    from fashion_multimodal_analysis.integration.pipeline import crop_bounds
    from fashion_multimodal_analysis.attributes.pattern import run_pattern_hierarchical_v3 as pattern
    from fashion_multimodal_analysis.attributes.design import run_design_attribute_labels_v1 as design
    ckpt = torch.load(head_path, map_location='cpu', weights_only=True)
    if ckpt['labels'] != LABELS or ckpt['clip_model'] != 'openai/clip-vit-base-patch32' or ckpt['input_view'] != 'raw_garment_bbox_no_expansion':
        raise ValueError('分类头配置与历史基线不一致')
    old_hashes = {c['source_image_sha256'] for split in ('train', 'dev') for c in ckpt['split'][split]}
    prepared = []
    for case in cases:
        rid = case['record_id']
        if rid not in rows: raise ValueError(f'训练清单中缺少 {rid}')
        row = rows[rid]
        gt = case['current_class']
        if gt not in LABELS or not case.get('class_confirmed') or case['review_status'] != 'confirmed' or case['state'] != 'measurable':
            raise ValueError(f'不是有效审核样本: {rid}')
        if not (0 <= float(case['ratio']) <= 1): raise ValueError(f'比例越界: {rid}')
        box = [float(row['bbox_' + k]) for k in ('x1', 'y1', 'x2', 'y2')]
        if row['garment_id'] != case['garment_id'] or row['source_image'] != case['source_image']:
            raise ValueError(f'身份不一致: {rid}')
        path = resolve_path(row['source_image'])
        digest = sha256(path)
        if digest != case['source_image_sha256'] or digest in old_hashes or digest in core_hashes:
            raise ValueError(f'图像哈希/数据隔离失败: {rid}')
        prepared.append((case, path, box))
    if not torch.cuda.is_available(): raise RuntimeError('需要GPU上的 py312 环境，未检测到 CUDA')
    device = torch.device('cuda')
    processor = CLIPProcessor.from_pretrained(ckpt['clip_model'], local_files_only=True)
    clip = CLIPModel.from_pretrained(ckpt['clip_model'], local_files_only=True).to(device).eval()
    clip.requires_grad_(False)
    feature_dim = int(ckpt['feature_dim'])
    head = torch.nn.Linear(feature_dim, len(LABELS)).to(device)
    head.load_state_dict(ckpt['head_state_dict'])
    head.eval()
    with torch.inference_mode():
        proto = pattern.build_class_prototypes(
            {k: [v] for k, v in design.SLEEVE_PROMPTS.items()}, clip, processor, device)
        results = []
        for i, (case, path, box) in enumerate(prepared, 1):
            image = Image.open(path).convert('RGB')
            crop = image.crop(crop_bounds(box, image.size))
            feature = pattern.encode_image(crop, clip, processor, device).detach().to(device)
            if feature.shape[-1] != feature_dim:
                raise ValueError('CLIP特征维度不匹配')
            scores = head(feature.clone().to(device)).softmax(1)[0].detach().cpu()
            head_ix = int(scores.argmax().item())
            prompt = pattern.classify_with_prototypes(feature, *proto, clip)
            results.append({
                'record_id': case['record_id'], 'truth': case['current_class'],
                'human_ratio_gt_not_model_prediction': float(case['ratio']),
                'supervised_head': LABELS[head_ix], 'supervised_score_uncalibrated': float(scores[head_ix]),
                'prompt_baseline': prompt['top1_label'], 'prompt_score_uncalibrated': float(prompt['top1_score']),
                'source_image_sha256': case['source_image_sha256']
            })
            print(f'PILOT8_INFERENCE {i}/8 {case["record_id"]}', flush=True)
    hm = metrics(results, 'supervised_head')
    pm = metrics(results, 'prompt_baseline')
    paired = Counter()
    for r in results:
        h = r['supervised_head'] == r['truth']
        p = r['prompt_baseline'] == r['truth']
        paired['both_correct' if h and p else 'head_only' if h else 'prompt_only' if p else 'both_wrong'] += 1
    summary = {
        'status': 'PILOT8_FROZEN_CLASSIFIER_INFERENCE_COMPLETED',
        'interpretation': 'Eight reviewed train-origin images were not part of the historical 44-label head training/dev split; diagnostic only, not formal PRD acceptance.',
        'n': 8, 'manual_class_counts': dict(Counter(x['truth'] for x in results)),
        'supervised_head': hm, 'same_raw_view_prompt_baseline': pm,
        'paired': dict(paired),
        'no_training': True, 'production_modified': False,
        'continuous_ratio_model_evaluated': False,
        'reason': 'Human ratios are reference annotations, not automated continuous-ratio predictions.',
        'trained_head_sha256': sha256(head_path),
        'pilot_sha256': sha256(pilot_path),
        'train_csv_sha256': TRAIN_SHA, 'val_csv_sha256': VAL_SHA,
        'limitations': ['Only cap/short/elbow represented; no sleeveless/three-quarter/long.',
                        'Source data were previously selected and reviewed; cannot claim independent blind test.',
                        'Scores are not calibrated, sample size is 8.',
                        'ROI uses ground-truth garment bbox, not end-to-end predicted ROI.']
    }
    output_root = Path(args.output_root)
    out = output_root / ('run_' + time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'predictions.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out / 'per_target_comparison.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (out / 'run_info.json').write_text(json.dumps({'script_sha256': sha256(Path(__file__)), 'checkpoint_path': str(head_path), 'input_view': 'raw_garment_bbox_no_expansion', 'clip_model': ckpt['clip_model']}, indent=2), encoding='utf-8')
    archive = out.with_suffix('.zip')
    with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
        for file in out.iterdir(): z.write(file, arcname=file.name)
    print('HEAD_ACCURACY:', hm['correct'], '/ 8')
    print('PROMPT_ACCURACY:', pm['correct'], '/ 8')
    print('NO_TRAINING_OR_DATA_CHANGE: TRUE')
    print('UPLOAD_THIS_ZIP:', archive, flush=True)


if __name__ == '__main__':
    main()
