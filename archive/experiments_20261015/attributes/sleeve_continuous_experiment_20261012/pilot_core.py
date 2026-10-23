"""Small deterministic paired diagnostic; no torch dependency in this module."""
import numpy as np

CLASSES = ['sleeveless', 'cap', 'short', 'elbow', 'three_quarter', 'long']
ALPHA = 0.1  # Fixed before looking at new development predictions.

def fit_ridge(x, y, alpha=ALPHA):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x-xm, y-ym
    # Dual solve avoids a large feature-dimensional inversion on a tiny pilot.
    w = xc.T @ np.linalg.solve(xc @ xc.T + alpha*np.eye(len(x)), yc)
    return w, ym-xm @ w

def class_metrics(truth, pred):
    truth, pred = np.asarray(truth), np.asarray(pred)
    cm = np.zeros((6, 6), dtype=int)
    for a, b in zip(truth, pred): cm[int(a), int(b)] += 1
    f1 = []
    for i in range(6):
        denom = cm[i].sum()+cm[:, i].sum()
        f1.append(float(2*cm[i, i]/denom) if denom else 0.0)
    return {'correct': int((truth == pred).sum()), 'n': len(truth),
            'accuracy': float((truth == pred).mean()), 'macro_f1_fixed_six_classes': float(np.mean(f1)),
            'class_order': CLASSES, 'support': cm.sum(axis=1).tolist(),
            'confusion_matrix_rows_truth': cm.tolist()}

def ratio_metrics(truth, pred):
    d = np.asarray(pred)-np.asarray(truth)
    return {'n': len(d), 'mae': float(np.abs(d).mean()), 'rmse': float(np.sqrt((d*d).mean()))}

def evaluate_variant(xtrain, xdev, train, dev):
    yt = np.array([r['ratio'] for r in train], float)
    yd = np.array([r['ratio'] for r in dev], float)
    ct = np.array([CLASSES.index(r['current_class']) for r in train])
    cd = np.array([CLASSES.index(r['current_class']) for r in dev])
    if set(ct) != set(range(6)): raise ValueError('Training must cover all six classes')
    means = np.array([yt[ct == i].mean() for i in range(6)])
    medians = np.array([np.median(yt[ct == i]) for i in range(6)])
    if not np.all(np.diff(medians)>0):
        raise ValueError('Train class medians are not ordered; do not tune on dev to fix this')
    cuts = (medians[:-1]+medians[1:])/2
    wc, bc = fit_ridge(xtrain, np.eye(6)[ct])
    wr, br = fit_ridge(xtrain, yt)
    direct = (xdev @ wc+bc).argmax(axis=1)
    raw_ratio = xdev @ wr+br
    ratio = np.clip(raw_ratio, 0, 1)
    via_ratio = np.searchsorted(cuts, ratio, side='right')
    majority = int(np.bincount(ct, minlength=6).argmax())
    oracle_train = np.searchsorted(cuts, yt, side='right')
    oracle_dev = np.searchsorted(cuts, yd, side='right')
    metrics = {
        'n_train': len(train), 'n_dev': len(dev), 'alpha_fixed': ALPHA,
        'train_ids': [r['record_id'] for r in train], 'dev_ids': [r['record_id'] for r in dev],
        'train_class_support': np.bincount(ct, minlength=6).tolist(),
        'thresholds_train_median_midpoints': cuts.tolist(), 'train_class_mean_ratios': means.tolist(),
        'direct_six_class_ridge': class_metrics(cd, direct),
        'continuous_ridge_to_class': class_metrics(cd, via_ratio),
        'continuous_ridge': ratio_metrics(yd, ratio),
        'direct_class_to_train_mean_ratio': ratio_metrics(yd, means[direct]),
        'train_majority_class_baseline': class_metrics(cd, np.full(len(dev), majority)),
        'train_mean_ratio_baseline': ratio_metrics(yd, np.full(len(dev), yt.mean())),
        'manual_ratio_mapping_NOT_model_accuracy': {
            'train': class_metrics(ct, oracle_train), 'dev': class_metrics(cd, oracle_dev),
            'note': 'Uses human target ratios. Label-consistency diagnostic only; not model performance.'}}
    rows=[]
    for i, r in enumerate(dev):
        rows.append({'record_id': r['record_id'], 'side': r['side'],
                     'truth_class': r['current_class'], 'truth_ratio': float(yd[i]),
                     'direct_class': CLASSES[direct[i]], 'regression_ratio_raw': float(raw_ratio[i]),
                     'regression_ratio_clipped': float(ratio[i]), 'regression_class': CLASSES[via_ratio[i]],
                     'regression_abs_error': float(abs(ratio[i]-yd[i])),
                     'direct_class_mean_ratio': float(means[direct[i]]),
                     'manual_ratio_mapping_NOT_prediction': CLASSES[oracle_dev[i]]})
    weights={'classifier_w':wc, 'classifier_b':bc, 'regressor_w':wr, 'regressor_b':br,
             'thresholds':cuts, 'class_mean_ratios':means}
    return metrics, rows, weights

def validate_manifest(data):
    train, dev = data['paired_train'], data['paired_dev']
    assert len(train)==14 and len(dev)==9, 'Unexpected pilot sizes'
    ids=[r['record_id'] for r in train+dev]
    assert len(ids)==len(set(ids)), 'Duplicate record IDs'
    assert not ({r['source_image_sha256'] for r in train} & {r['source_image_sha256'] for r in dev}), 'Image leakage'
    for split, rows in [('train', train), ('dev', dev)]:
        for r in rows:
            assert r['experiment_split']==split and r['valid_side_label']
            assert r['side'] in ('image_left', 'image_right')
            assert r['current_class'] in CLASSES and 0<=r['ratio']<=1
            assert np.all(np.isfinite(r['garment_bbox_original']))
    assert next(r for r in dev if r['record_id']=='v5_train_01075')['side']=='image_right'
    return train, dev
