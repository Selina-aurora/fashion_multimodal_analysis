"""Read-only native pocket ownership proposals across all five clothing classes.
No old audit dependency, no training, no automatically confirmed labels.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import random
import re
import time
import zipfile

MANIFESTS = {
    'train': ('configs/prd_8class_train_v5_balanced150_20261008.csv', 'e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f'),
    'val': ('configs/prd_8class_val_v2_full_targets.csv', '39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93'),
}
CLOTHING = ('top', 'pants', 'skirt', 'outerwear', 'dress')
RAW_SHA = 'f831f415edad12d52acde8b138fe7ebd1592b57bf0dcd6a3b63ba8cabd208223'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def basename(value):
    return str(value).replace('\\', '/').rsplit('/', 1)[-1]


def annotation_id(row):
    ref = row.get('annotation_ref', '').strip()
    if ref.isdigit():
        return int(ref)
    match = re.search(r'fashionpedia_ann_(\d+)', row['garment_id'])
    return int(match.group(1)) if match else None


def main():
    import numpy as np
    from PIL import Image, ImageDraw
    from pycocotools import mask as maskutils
    from fashion_multimodal_analysis.common.paths import project_root, resolve_path, data_root
    root = project_root()
    raw_path = data_root() / 'raw/fashionpedia/annotations/instances_attributes_val2020.json'
    if sha(raw_path) != RAW_SHA:
        raise ValueError('Frozen native annotation file changed')
    raw = json.loads(raw_path.read_text(encoding='utf-8-sig'))
    cats = {a['id']: a for a in raw['categories']}
    attrs = {a['id']: a for a in raw['attributes']}
    images = {a['id']: a for a in raw['images']}
    anns = {a['id']: a for a in raw['annotations']}
    garments = defaultdict(list)
    pockets = defaultdict(list)
    for a in raw['annotations']:
        cat = cats[a['category_id']]
        if cat['supercategory'] in {'upperbody', 'lowerbody', 'wholebody'}:
            garments[a['image_id']].append(a)
        if cat['name'] == 'pocket':
            pockets[a['image_id']].append(a)
    selected = defaultdict(list)
    unmatched = []
    all_rows = {}
    source_hashes = {}
    for split, (relative, expected) in MANIFESTS.items():
        path = root / relative
        if sha(path) != expected:
            raise ValueError('Frozen manifest changed: ' + relative)
        with path.open(encoding='utf-8-sig', newline='') as f:
            all_rows[split] = list(csv.DictReader(f))
        for row in all_rows[split]:
            if row['source_dataset'] != 'Fashionpedia' or row['garment_category'] not in CLOTHING:
                continue
            a = anns.get(annotation_id(row))
            if not a or basename(images[a['image_id']]['file_name']) != basename(row['source_image']) or cats[a['category_id']]['name'] != row['fine_category']:
                unmatched.append({'split': split, 'record_id': row['record_id']})
                continue
            item = {'split': split, 'record_id': row['record_id'], 'category': row['garment_category'],
                    'source_image': row['source_image'], 'garment_id': row['garment_id'],
                    'native_garment_annotation_id': a['id'], 'image_id': a['image_id'],
                    'garment_bbox': [float(row['bbox_' + k]) for k in ('x1', 'y1', 'x2', 'y2')]}
            selected[a['id']].append(item)
    def decode(a, h, w):
        seg = a['segmentation']
        if isinstance(seg, list):
            rle = maskutils.merge(maskutils.frPyObjects(seg, h, w))
        elif isinstance(seg.get('counts'), list):
            rle = maskutils.frPyObjects(seg, h, w)
        else:
            rle = seg
        result = maskutils.decode(rle)
        if result.ndim == 3:
            result = np.any(result, axis=2)
        if result.shape != (h, w):
            raise ValueError('Mask dimensions mismatch')
        return result > 0
    relevant_images = {r['image_id'] for rows in selected.values() for r in rows}
    evidence = []
    grouped = defaultdict(list)
    for image_id in sorted(relevant_images & set(pockets)):
        meta = images[image_id]
        h, w = int(meta['height']), int(meta['width'])
        gm = {a['id']: decode(a, h, w) for a in garments[image_id]}
        for part in pockets[image_id]:
            pm = decode(part, h, w)
            area = int(pm.sum())
            ranked = sorted([{'garment_annotation_id': aid,
                              'part_coverage': float(np.logical_and(pm, mask).sum() / area) if area else 0.0}
                             for aid, mask in gm.items()], key=lambda x: (-x['part_coverage'], x['garment_annotation_id']))
            best = ranked[0] if ranked else None
            second = ranked[1]['part_coverage'] if len(ranked) > 1 else 0.0
            proposed = best['garment_annotation_id'] if area and best and best['part_coverage'] >= 0.8 and second <= 0.2 else None
            status = 'PROPOSED_SELECTED_GARMENT' if proposed in selected else 'PROPOSED_NONSELECTED_GARMENT' if proposed is not None else 'AMBIGUOUS_OR_LOW_COVERAGE'
            item = {'image_id': image_id, 'image_file': meta['file_name'], 'part_annotation_id': part['id'],
                    'part_bbox_xywh': part['bbox'], 'part_attributes': [attrs[i] for i in part.get('attribute_ids', []) if i in attrs],
                    'ranked_garment_coverages': ranked, 'association_status': status,
                    'proposed_native_garment_annotation_id': proposed, 'association_confirmed': False}
            evidence.append(item)
            if proposed in selected:
                for target in selected[proposed]:
                    grouped[(target['split'], target['record_id'])].append(item)
    candidates = []
    for rows in selected.values():
        for target in rows:
            parts = grouped.get((target['split'], target['record_id']))
            if parts:
                candidates.append({**target, 'parts': parts, 'review_status': 'pending', 'confirmed_gt': False})
    # Exact source-byte checks include ALL validation images, not only pocket positives.
    def source_sha(value):
        key = str(resolve_path(value))
        if key not in source_hashes:
            source_hashes[key] = sha(key)
        return source_hashes[key]
    val_hashes = {source_sha(r['source_image']) for r in all_rows['val']}
    excluded = []
    eligible = []
    for c in candidates:
        c['source_image_sha256'] = source_sha(c['source_image'])
        if c['split'] == 'train' and c['source_image_sha256'] in val_hashes:
            excluded.append(c['record_id'])
        else:
            eligible.append(c)
    rng = random.Random(20261009)
    review = []
    for category in CLOTHING:
        pool = sorted([c for c in eligible if c['split'] == 'train' and c['category'] == category], key=lambda c: c['record_id'])
        rng.shuffle(pool)
        review.extend(pool[:4])
    review.extend(sorted([c for c in eligible if c['split'] == 'val'], key=lambda c: c['record_id']))
    out = root / 'reports/reruns/recovery_20261009' / ('pocket_candidates_v2_' + time.strftime('%Y%m%d_%H%M%S') + f'_{time.time_ns()%1000000:06d}')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'review_images').mkdir()
    for number, c in enumerate(review, 1):
        image = Image.open(resolve_path(c['source_image'])).convert('RGB')
        source_path = f'review_images/{number:02d}_source.jpg'
        marked_path = f'review_images/{number:02d}_marked.jpg'
        image.save(out / source_path, quality=95)
        pen = ImageDraw.Draw(image)
        width = max(2, image.width // 220)
        pen.rectangle(c['garment_bbox'], outline='yellow', width=width)
        for p in c['parts']:
            x, y, w, h = p['part_bbox_xywh']
            pen.rectangle([x, y, x+w, y+h], outline='red', width=width)
        image.thumbnail((1000, 1100))
        image.save(out / marked_path, quality=93)
        c['review_number'] = number
        c['source_image_export'] = source_path
        c['marked_image_export'] = marked_path
    summary = {'status': 'POCKET_ALL_CLOTHING_CANDIDATES_PREPARED', 'clothing_classes': list(CLOTHING),
               'native_file_sha256': RAW_SHA, 'manifest_provenance': MANIFESTS,
               'joined_garments_by_split': {s: dict(Counter(r['category'] for rs in selected.values() for r in rs if r['split'] == s)) for s in MANIFESTS},
               'unmatched_records': unmatched, 'pocket_parts_on_selected_clothing_images': len(evidence),
               'association_status_counts': dict(Counter(e['association_status'] for e in evidence)),
               'eligible_candidate_garments_by_split': {s: dict(Counter(c['category'] for c in eligible if c['split'] == s)) for s in MANIFESTS},
               'review_samples_by_split': dict(Counter(c['split'] for c in review)),
               'train_candidates_excluded_for_validation_image_overlap': excluded,
               'ownership_confirmed': False, 'training_started': False, 'formal_prd_acceptance': False,
               'limitations': ['Mask coverage only proposes ownership; all labels pending human review.',
                              'All native upper/lower/wholebody garments compete, including unselected garments.',
                              'Sample contains pocket-positive candidates only; no recall or negative-case accuracy measurement.',
                              'Existing validation remains separate; reviewed validation is diagnostic, not an unseen blind benchmark.']}
    for name, value in [('summary.json', summary), ('part_association_evidence.json', evidence),
                        ('all_native_candidates.json', eligible), ('review_cases.json', review)]:
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():
                z.write(p, p.relative_to(out))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('UPLOAD_THIS_ZIP:', archive, flush=True)


if __name__ == '__main__':
    main()
