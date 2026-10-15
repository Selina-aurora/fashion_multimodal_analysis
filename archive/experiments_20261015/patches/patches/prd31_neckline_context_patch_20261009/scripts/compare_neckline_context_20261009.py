"""配对报告比较，按输入哈希选基线，不将变化数量解释为准确率。"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from PIL import Image


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def find_baseline(root, explicit, selected, config, sha, resolve_path):
    expected = {k: sha(resolve_path(config[k])) for k in
                ('segmentation_checkpoint', 'train_manifest', 'val_manifest')}
    candidates = [resolve_path(explicit)] if explicit else sorted(
        (root / 'reports/reruns/recovery_20261008').glob('prd31_guard_regression_*'),
        reverse=True,
    )
    wanted = [r['source_sha256'] for r in selected]
    for folder in candidates:
        if not folder.is_dir():
            continue
        try:
            summary = read(folder / 'summary.json')
            images = read(folder / 'selected_images.json')
            if summary.get('regression_patch_version') != 'semantic_guards_20261008_v1':
                continue
            if summary.get('pipeline_status_counts') != {'completed': 8}:
                continue
            if summary.get('failed_downstream_calls') != 0 or summary.get('interface_issues') != 0:
                continue
            if summary.get('input_sha256') != expected:
                continue
            if [r['source_sha256'] for r in images] != wanted:
                continue
            if all((folder / f'case_{i:02d}/predictions.json').is_file() for i in range(1, 9)):
                return folder
        except (OSError, ValueError, KeyError):
            continue
    raise FileNotFoundError('Cannot find matching completed yesterday baseline. Use --baseline-dir reports/.../prd31_guard_regression_...')


def compare_reports(baseline, current):
    result = {'images': 8, 'neckline_pairs': [], 'non_neckline_differences': [],
              'accuracy_evaluated': False,
              'note': 'Changes and agreement are diagnostic counts, not accuracy. Any non-neckline drift requires review.'}
    for i in range(1, 9):
        case = f'case_{i:02d}'
        a_dir, b_dir = baseline / case, current / case
        a_list, b_list = read(a_dir / 'predictions.json')['instances'], read(b_dir / 'predictions.json')['instances']
        a = {r['instance_id']: r for r in a_list}
        b = {r['instance_id']: r for r in b_list}
        if list(a) != list(b):
            result['non_neckline_differences'].append({'case': case, 'target': 'instance_ids'})
        for key in a.keys() & b.keys():
            old, new = a[key], b[key]
            for field in ('category', 'score', 'bbox', 'provenance', 'regions', 'review_flags'):
                if old.get(field) != new.get(field):
                    result['non_neckline_differences'].append({'case': case, 'instance_id': key, 'target': field})
            # Compare actual garment/region arrays, not only matching filenames.
            paths = [('garment_mask', old['mask_path'], new['mask_path'])]
            for region in old['regions'].keys() & new['regions'].keys():
                oa, nb = old['regions'][region], new['regions'][region]
                if oa.get('mask_path') and nb.get('mask_path'):
                    paths.append((region+'_mask', oa['mask_path'], nb['mask_path']))
            for target, op, np_ in paths:
                with Image.open(a_dir/op) as oa, Image.open(b_dir/np_) as nb:
                    equal = np.array_equal(np.asarray(oa), np.asarray(nb))
                if not equal:
                    result['non_neckline_differences'].append({'case': case, 'instance_id': key, 'target': target})
            for attr in old['attributes'].keys() | new['attributes'].keys():
                if attr != 'neckline' and old['attributes'].get(attr) != new['attributes'].get(attr):
                    result['non_neckline_differences'].append({'case': case, 'instance_id': key, 'target': attr})
            neck = new['attributes'].get('neckline', {})
            if 'context_candidate' in neck:
                result['neckline_pairs'].append({'case': case, 'instance_id': key,
                    'category': new['category'], 'previous_label': old['attributes'].get('neckline', {}).get('label'),
                    'legacy_candidate': neck['legacy_candidate'], 'context_candidate': neck['context_candidate'],
                    'input_view_agreement': neck['input_view_agreement'], 'status': neck['status'],
                    'legacy_view_path': case+'/'+neck['legacy_view_path'],
                    'context_view_path': case+'/'+neck['context_view_path'],
                    'semantic_review_status': 'pending'})
    result['paired_neckline_count'] = len(result['neckline_pairs'])
    result['view_disagreement_count'] = sum(not r['input_view_agreement'] for r in result['neckline_pairs'])
    result['legacy_prediction_drift_count'] = sum(r['previous_label'] != r['legacy_candidate']['label'] for r in result['neckline_pairs'])
    result['non_neckline_difference_count'] = len(result['non_neckline_differences'])
    return result
