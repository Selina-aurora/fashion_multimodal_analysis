#!/usr/bin/env python3
"""Read-only preflight for W7 Dev12 -> PRD 3.2. No sealed holdout access."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT = Path('/workspace')
OUT = ROOT / 'PRD32_W7DEV12_20261026'
PREVIOUS = ROOT / 'PRD32_DEV_20261026/dev_manifest.csv'
M3 = ROOT / 'PRD31_W7_M3/inputs/W7_dev12_SELECTED.json'
ASSIST = ROOT / 'PRD31_W7_ASSIST/inputs/W7_dev12_SELECTED.json'
HUMAN_SCREEN = ROOT / 'PRD31_W7_M2/inputs/W7_dev12_human_screen_2026-10-15.csv'
# Restored from the previously reviewed W7_dev12_GT_HUMAN_REVIEW.json.
# Reference category / box are development annotations, not PRD 3.2 attribute GT.
ROWS = [
 ('dev_01','train/train/image/105983.jpg','961282b5e1100a4fe0620e8fec947626b2eb1882c14389ab3632d5ada9e1d865','outerwear','158, 2, 565, 401','yes'),
 ('dev_02','validation/validation/image/008889.jpg','fde7bd42023d4ed1742cae3c9d735e9f90c0a68111e3ee522abd1757e93b4cdb','outerwear','5, 16, 468, 797','yes'),
 ('dev_03','train/train/image/007644.jpg','1d6b8a69c8f39c122b8deabf59d7b81d89140c865870f9780a3095151c9a5a02','top','276, 423, 595, 705','no'),
 ('dev_04','validation/validation/image/028673.jpg','0fe745bf8f6f3ed23580e6b0bc88c961e5aab6dc90e55593d10eb6262b9e250d','top','28, 161, 575, 820','yes'),
 ('dev_05','train/train/image/009252.jpg','b05624cb438de3d82ae8901ba39a9b1b95c02e109194abcd669e4d7e1e4c7042','top','110, 2, 537, 463','yes'),
 ('dev_06','validation/validation/image/029438.jpg','dbf57c8e2adf52b551740635c06ef01e14c714464e39ea77491132777a52a798','top','150, 5, 496, 499','yes'),
 ('dev_07','fashionpedia/images/test/759a7660c0f1268c90bad70cae7427ca.jpg','5c5bd3897baee2c180c13e34d3ba4b4a38c95a710378376a8eee3c02b63e9fdc','top','118, 380, 267, 514','yes'),
 ('dev_08','fashionpedia/images/test/cc066bcf334a4f0345764c594e85a929.jpg','35b2352e3606fb0459ddc4bf32d179ccd0b71f5cc66dac3b98522f432421de24','top','167, 230, 462, 676','yes'),
 ('dev_09','fashionpedia/images/test/9750434d2cd9ced43c09f15ae11ed6b2.jpg','d4529f8daa5a9e330267ec43c7a09c2a39e4fd38537e750b669abff2b0239a52','outerwear','169, 281, 581, 856','yes'),
 ('dev_10','fashionpedia/images/test/975bc03b803a8187fe7b005966f7cdd9.jpg','c178846c2d7981dc7fb11c532740f934177bbf275f7e3b4bd5eecdfaef9932fd','dress','54, 298, 682, 1024','yes'),
 ('dev_11','train/train/image/047522.jpg','8ca4635a05d63cca63cdd97e68450b426fcfad867cca61d6555497df5762ff9e','top','217, 270, 696, 583','yes'),
 ('dev_12','fashionpedia/images/test/09ff9e179fe9d63f48c1bf6d5565f659.jpg','ee374ecb334c2037c54a42e1e95fd83b69b794be4ead5b6d523e3aba01bcefd8','outerwear','243, 213, 519, 584','yes'),
]

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()

def main():
    if OUT.exists():
        raise SystemExit('Refusing to overwrite: ' + str(OUT))
    prior = set()
    if PREVIOUS.exists():
        with PREVIOUS.open(encoding='utf-8-sig', newline='') as f:
            prior = {r['image_sha256'] for r in csv.DictReader(f) if r.get('image_sha256')}
    source_checks = {}
    for name, path in [('selected_assist', ASSIST), ('selected_m3', M3), ('human_screen', HUMAN_SCREEN)]:
        source_checks[name] = {'exists': path.is_file(), 'sha256': sha256(path) if path.is_file() else None}
    result = []
    for image_id, rel, expected, cat, box, completed in ROWS:
        path = ROOT / 'fashion_data/raw' / rel
        if any(term in str(path).lower() for term in ['sealed', 'holdout']):
            raise ValueError('Forbidden sealed input: ' + str(path))
        exists = path.is_file()
        hash_ok = exists and sha256(path) == expected
        coords = [int(x.strip()) for x in box.split(',')]
        size, bounds_ok = None, False
        if hash_ok:
            try:
                with Image.open(path) as im:
                    size = [im.width, im.height]
                x1, y1, x2, y2 = coords
                bounds_ok = 0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]
            except Exception:
                pass
        result.append({
            'image_id': image_id, 'image_path': str(path), 'image_sha256': expected,
            'reference_category': cat, 'reference_bbox_xyxy': json.dumps(coords),
            'reference_provenance': 'historical_W7_dev12_human_assisted_review',
            'target_review_complete': completed, 'attribute_gt_status': 'NOT_REVIEWED',
            'image_exists': exists, 'sha256_verified': bool(hash_ok),
            'reference_bbox_in_bounds': bool(bounds_ok), 'image_size': json.dumps(size),
            'overlaps_previous_dev_by_sha256': expected in prior,
            'dev_inference_candidate': bool(hash_ok and bounds_ok),
            'notes': 'Development only; category/box are references, NOT a frozen 3.1 prediction.'
        })
    OUT.mkdir(parents=True, exist_ok=False)
    csv_path = OUT / 'w7_dev12_preflight.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    report = {
        'status': 'W7_DEV12_PREFLIGHT_ONLY_NOT_PRD_ACCEPTED',
        'source_checks': source_checks,
        'selected_json_same_sha256': source_checks['selected_assist']['sha256'] == source_checks['selected_m3']['sha256'] if (source_checks['selected_assist']['exists'] and source_checks['selected_m3']['exists']) else None,
        'total': len(result), 'categories': dict(Counter(x['reference_category'] for x in result)),
        'hash_verified': sum(x['sha256_verified'] for x in result),
        'bbox_in_bounds': sum(x['reference_bbox_in_bounds'] for x in result),
        'dev_inference_candidates': sum(x['dev_inference_candidate'] for x in result),
        'review_incomplete': [x['image_id'] for x in result if x['target_review_complete'] != 'yes'],
        'overlap_previous_dev': [x['image_id'] for x in result if x['overlaps_previous_dev_by_sha256']],
        'limitations': ['No PRD 3.2 color/pattern/design ground truth supplied', 'No model inference performed', 'No sealed holdout accessed', 'Source paths include historic datasets with test in their directory names; all are treated as exposed dev samples']
    }
    (OUT / 'preflight_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('MANIFEST:', csv_path)

if __name__ == '__main__':
    main()
