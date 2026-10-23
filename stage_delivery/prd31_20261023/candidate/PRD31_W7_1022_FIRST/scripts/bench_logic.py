"""Comparison and aggregation rules for frozen, annotation-free replay."""
from collections import Counter
import numpy as np


def segmentation_difference(current, reference, bbox_atol=1e-3, score_atol=1e-5):
    differences = []
    if len(current) != len(reference):
        differences.append({'kind': 'instance_count', 'current': len(current), 'reference': len(reference)})
    # Existing backend instance IDs/order are part of the interface. Do not rematch
    # with annotation boxes or silently reorder to hide a drift.
    for i, (a, b) in enumerate(zip(current, reference)):
        fields = []
        for k in ('instance_id', 'category'):
            if a[k] != b[k]: fields.append(k)
        if not np.allclose(a['bbox'], b['bbox'], atol=bbox_atol, rtol=0): fields.append('bbox')
        if not np.isclose(a['score'], b['score'], atol=score_atol, rtol=0): fields.append('score')
        if not np.array_equal(np.asarray(a['mask']) > 0, np.asarray(b['mask']) > 0): fields.append('mask')
        if fields: differences.append({'index': i, 'kind': 'prediction_drift', 'fields': fields})
    return differences


def feature_comparison(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != (512,) or b.shape != (512,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected two finite 512D vectors')
    if min(np.linalg.norm(a), np.linalg.norm(b)) <= 1e-10:
        raise ValueError('Zero feature')
    max_abs = float(np.max(np.abs(a-b)))
    cosine = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    return {'max_abs': max_abs, 'cosine': cosine,
            'within_declared_tolerance': max_abs <= 1e-4 and cosine >= 0.99999}


def attribute_signature(result):
    return {key: (value.get('status'), value.get('label')) for key,value in sorted(result.items())}


def aggregate(timings):
    result = {}
    for stage in sorted({r['stage'] for r in timings}):
        rows = [r for r in timings if r['stage'] == stage]
        values = [r['ms'] for r in rows]
        result[stage] = {'n': len(values), 'mean_ms': float(np.mean(values)),
                        'p50_ms': float(np.percentile(values,50)),
                        'p95_ms': float(np.percentile(values,95)),
                        'unique_images': len({r['image_id'] for r in rows}),
                        'per_pass_mean_ms': {str(p): float(np.mean([r['ms'] for r in rows if r['pass'] == p]))
                                             for p in sorted({r['pass'] for r in rows})}}
    return result
