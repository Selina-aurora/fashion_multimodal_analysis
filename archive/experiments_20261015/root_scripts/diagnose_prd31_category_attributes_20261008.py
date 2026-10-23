"""诊断类别/袖长/图案对裁剪视图的敏感性；不修改模型预测或权重。

Reuse project CLIP helpers, offline snapshots and the latest eight-case guard
regression report. Scores are relative prompt scores, not calibrated accuracy.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import zipfile

CATEGORY_PROMPTS = {
    'top': ['a photo of a shirt, blouse, T-shirt or sweater worn on the upper body',
            'a separate upper-body garment, not a full-length dress'],
    'pants': ['a photo of trousers, jeans or pants with two separate trouser legs',
              'a pair of denim jeans or long pants'],
    'skirt': ['a photo of a skirt worn below the waist, with a waistband and skirt hem',
              'a separate skirt, not an upper-body garment'],
    'outerwear': ['a photo of an outer coat, jacket or cardigan worn over other clothes',
                 'an outerwear garment with sleeves and an open or closed front'],
    'dress': ['a photo of a one-piece dress connecting an upper bodice and a skirt',
              'a dress worn on the body from shoulders down to a skirt hem'],
    'shoe': ['a photo of a shoe, boot, sandal or sneaker worn on the foot'],
    'bag': ['a photo of a handbag, shoulder bag, tote or purse'],
    'accessory': ['a photo of a clothing accessory such as eyeglasses, belt, scarf or jewelry'],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', help='原来的 guard regression 解压目录，默认选最近的有效报告')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    # 必须在导入 transformers/huggingface 之前设置缓存，避免找错位置。
    os.environ.setdefault('HF_HOME', '/workspace/hf_cache')
    os.environ.setdefault('HF_HUB_CACHE', '/workspace/hf_cache/hub')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    from PIL import Image
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from fashion_multimodal_analysis.common.paths import project_root, resolve_path
    from fashion_multimodal_analysis.integration.pipeline import crop_bounds
    from fashion_multimodal_analysis.attributes.pattern import run_pattern_hierarchical_v3 as pattern
    from fashion_multimodal_analysis.attributes.design import run_design_attribute_labels_v1 as design

    root = project_root()
    if args.report_dir:
        report = resolve_path(args.report_dir)
    else:
        candidates = []
        for path in (root / 'reports/reruns/recovery_20261008').glob('prd31_guard_regression_*'):
            if not path.is_dir() or not (path / 'summary.json').is_file():
                continue
            summary = json.loads((path / 'summary.json').read_text())
            if summary.get('regression_patch_version') == 'semantic_guards_20261008_v1' and summary.get('pipeline_status_counts') == {'completed': 8}:
                candidates.append(path)
        if not candidates:
            raise FileNotFoundError('找不到成功完成的 guard regression 报告，请用 --report-dir 指定目录')
        report = max(candidates, key=lambda p: p.name)
    cases = json.loads((report / 'cases.json').read_text())
    if len(cases) != 8:
        raise ValueError('此诊断要求原来的 8 个案例')
    # 先验证输入、mask 和报告完整性，再加载唯一需要的 CLIP 模型。
    for case in cases:
        source = report / 'source_images' / (case['case_dir'] + '.jpg')
        if sha(source) != case['source_sha256']:
            raise ValueError('原图哈希不一致: ' + case['case_dir'])
        payload = json.loads((report / case['case_dir'] / 'predictions.json').read_text())
        if len(payload['instances']) != case['instance_count']:
            raise ValueError('实例数量不一致: ' + case['case_dir'])
        for instance in payload['instances']:
            if not (report / case['case_dir'] / instance['mask_path']).is_file():
                raise FileNotFoundError(instance['mask_path'])
    config = json.loads((report / 'run_config.json').read_text())
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    print('INPUT_REPORT:', report, flush=True)
    model_id = config['clip_model']
    processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
    model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(device).eval()
    print('CLIP_READY', flush=True)
    with torch.inference_mode():
        prototypes = {
            'category': pattern.build_class_prototypes(CATEGORY_PROMPTS, model, processor, device),
            'pattern': pattern.build_class_prototypes(pattern.STAGE2_PROMPTS, model, processor, device),
            'pattern_gate': pattern.build_class_prototypes(pattern.STAGE1_PROMPTS, model, processor, device),
            'sleeve_length': pattern.build_class_prototypes({k: [v] for k, v in design.SLEEVE_PROMPTS.items()}, model, processor, device),
        }
    output = root / 'reports/reruns/recovery_20261008' / ('category_attribute_diagnostic_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    records, table = [], []
    with torch.inference_mode():
        for number, case in enumerate(cases, 1):
            cid = case['case_dir']
            image = Image.open(report / 'source_images' / (cid + '.jpg')).convert('RGB')
            payload = json.loads((report / cid / 'predictions.json').read_text())
            for instance in payload['instances']:
                iid = instance['instance_id']
                folder = output / cid / iid
                folder.mkdir(parents=True)
                bounds = crop_bounds(instance['bbox'], image.size)
                raw = image.crop(bounds)
                mask = Image.open(report / cid / instance['mask_path']).convert('L')
                if mask.size != image.size:
                    raise ValueError('Mask size differs from image: ' + cid + '/' + iid)
                mask = mask.point(lambda p: 255 if p > 0 else 0).crop(bounds)
                masked = Image.composite(raw, Image.new('RGB', raw.size, 'white'), mask)
                record = {'case_id': cid, 'instance_id': iid, 'original_category': instance['category'],
                          'original_attributes': instance['attributes'], 'bbox': instance['bbox'],
                          'original_segmentation_score': instance['score'], 'views': {}}
                views = {'raw_full': raw, 'masked_full': masked}
                # 上身视图仅做袖长对照，绝不用于直接修改类别。
                if instance['category'] in {'top', 'outerwear', 'dress'}:
                    views['raw_upper_60pct'] = raw.crop((0, 0, raw.width, max(1, round(raw.height * .6))))
                for name, view in views.items():
                    view.save(folder / (name + '.jpg'), quality=95)
                    feature = pattern.encode_image(view, model, processor, device)
                    tasks = ['sleeve_length'] if name == 'raw_upper_60pct' else ['category', 'pattern_gate', 'pattern']
                    if name != 'raw_upper_60pct' and instance['category'] in {'top', 'outerwear', 'dress'}:
                        tasks.append('sleeve_length')
                    result = {}
                    for task in tasks:
                        ranked = pattern.classify_with_prototypes(feature, *prototypes[task], model)
                        if task == 'category':
                            original_index = prototypes[task][0].index(instance['category'])
                            probs = (model.logit_scale.exp().detach() * feature @ prototypes[task][1].T)[0].softmax(dim=0)
                            ranked['original_category_score'] = float(probs[original_index].cpu())
                        if task == 'pattern_gate':
                            ranked['binary_route'] = pattern.binary_route(ranked['top1_label'])
                        if task == 'pattern':
                            ranked['conditional_note'] = 'Fine-pattern ranking only; use alongside pattern_gate, not as an unconditional final label.'
                        result[task] = ranked
                        table.append({'case_id': cid, 'instance_id': iid, 'original_category': instance['category'],
                                      'view': name, 'task': task, 'top1': ranked['top1_label'],
                                      'top1_score': ranked['top1_score'], 'top2': ranked['top2_label'],
                                      'top2_score': ranked['top2_score'], 'margin': ranked['margin'],
                                      'original_category_score': ranked.get('original_category_score', '')})
                    record['views'][name] = result
                records.append(record)
            print(f'CASE_DONE {number}/8: {cid}', flush=True)
            dump(output / 'diagnostic_results.json', records)
    with (output / 'diagnostic_scores.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    disagreements = {view: sum(r['views'][view]['category']['top1_label'] != r['original_category'] for r in records)
                     for view in ('raw_full', 'masked_full')}
    dump(output / 'summary.json', {'status': 'diagnostic_completed', 'images': len(cases), 'instances': len(records),
         'input_report': str(report), 'clip_model': model_id, 'category_disagreement_counts': disagreements,
         'predictions_modified': False, 'accuracy_evaluated': False,
         'note': 'CLIP disagreement is a review signal, not a confirmed correction. Scores are prompt-relative, not calibrated probabilities.'})
    (output / 'readme.md').write_text('类别/袖长/图案裁剪敏感性诊断。没有修改原始预测或权重。CLIP 分歧不等于 CLIP 正确；fine pattern 分数需结合 pattern_gate 判断。仅供开发案例分析，不能作为正式验收。', encoding='utf-8')
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(output.rglob('*')):
            if path.is_file(): zipped.write(path, path.relative_to(output))
    print('DIAGNOSTIC_COMPLETED', flush=True)
    print('UPLOAD_THIS_ZIP:', archive, flush=True)


if __name__ == '__main__':
    main()
