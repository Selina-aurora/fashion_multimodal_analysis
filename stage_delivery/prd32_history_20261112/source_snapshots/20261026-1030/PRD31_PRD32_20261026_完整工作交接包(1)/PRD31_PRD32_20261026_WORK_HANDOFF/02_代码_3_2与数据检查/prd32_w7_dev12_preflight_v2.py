#!/usr/bin/env python3
"""Audited W7 Dev12 -> PRD 3.2 development-only preflight.

Read-only inputs. Source/review identity disagreements are quarantined, never
silently fixed. Human attribute GT is NOT inferred. Sealed holdout is forbidden.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

SELECTED = Path('/workspace/PRD31_W7_ASSIST/inputs/W7_dev12_SELECTED.json')
REVIEW = Path('/workspace/PRD31_W7_M2/inputs/W7_dev12_human_screen_2026-10-15.csv')
PREVIOUS = Path('/workspace/PRD32_DEV_20261026/dev_manifest.csv')
OUTPUT = Path('/workspace/PRD32_W7_DEV12_PREFLIGHT_V2_20261026')
BLOCKED = ('sealed', 'holdout')


def no_sealed(path):
    if any(s in str(path).lower() for s in BLOCKED):
        raise ValueError(f'Sealed/holdout path prohibited: {path}')


def sha256_file(path):
    hasher = hashlib.sha256()
    with path.open('rb') as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def box_from_csv(s):
    if not str(s or '').strip():
        return None
    pieces = re.findall(r'-?\d+(?:\.\d+)?', str(s))
    if len(pieces) != 4:
        return 'INVALID_FORMAT'
    numbers = [float(x) for x in pieces]
    return numbers if all(math.isfinite(x) for x in numbers) else 'INVALID_FORMAT'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected', type=Path, default=SELECTED)
    parser.add_argument('--review', type=Path, default=REVIEW)
    parser.add_argument('--previous-manifest', type=Path, default=PREVIOUS)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    for path in (args.selected, args.review, args.previous_manifest, args.output):
        no_sealed(path)
    if args.output.exists():
        parser.error(f'Refusing to overwrite existing folder: {args.output}')
    if not args.selected.is_file() or not args.review.is_file():
        parser.error('Missing W7 selected JSON or review CSV')

    selected = json.loads(args.selected.read_text(encoding='utf-8'))
    with args.review.open(encoding='utf-8-sig', newline='') as fp:
        reviews = list(csv.DictReader(fp))
    if not isinstance(selected, list) or len(selected) != 12 or len(reviews) != 12:
        raise ValueError('Expected 12 selected and 12 review records')
    if any(not isinstance(x, dict) for x in selected):
        raise ValueError('Selected records must be objects')
    rmap = {r.get('image_id'): r for r in reviews}
    ids = [r.get('image_id') for r in selected]
    if len(set(ids)) != 12 or len(rmap) != 12 or set(ids) != set(rmap):
        raise ValueError('Duplicate or unmatched image_id: stop for manual review')
    previous = set()
    if args.previous_manifest.is_file():
        with args.previous_manifest.open(encoding='utf-8-sig', newline='') as fp:
            previous = {r.get('image_sha256', '') for r in csv.DictReader(fp)}

    rows, previews = [], []
    for record in selected:
        id_ = record['image_id']
        rev = rmap[id_]
        source = Path(str(record.get('source_image') or ''))
        no_sealed(source)
        rsource = str(rev.get('source_image') or '')
        no_sealed(rsource)
        s_hash = str(record.get('source_sha256') or '')
        r_hash = str(rev.get('source_sha256') or '')
        actual = sha256_file(source) if source.is_file() else ''
        issues = []
        if not actual:
            issues.append('SOURCE_IMAGE_MISSING')
        elif actual != s_hash:
            issues.append('SOURCE_HASH_MISMATCH')
        if r_hash != s_hash:
            issues.append('REVIEW_HASH_MISMATCH')
        if rsource != str(source):
            issues.append('REVIEW_PATH_MISMATCH')

        picture = None
        bounds = None
        if actual == s_hash and actual:
            with Image.open(source) as pic:
                picture = pic.convert('RGB')
            w, h = picture.size
            expect_w, expect_h = record.get('width'), record.get('height')
            if expect_w is not None and int(expect_w) != w:
                issues.append('SOURCE_WIDTH_MISMATCH')
            if expect_h is not None and int(expect_h) != h:
                issues.append('SOURCE_HEIGHT_MISMATCH')
            bounds = (w, h)

        valid_identity = not issues
        raw_box = rev.get('target_garment_bbox_xyxy')
        box = box_from_csv(raw_box)
        if box is None:
            box_status = 'MISSING'
        elif isinstance(box, str):
            box_status = 'INVALID_FORMAT'
        elif bounds and not (0 <= box[0] < box[2] <= bounds[0] and
                             0 <= box[1] < box[3] <= bounds[1]):
            box_status = 'INVALID_COORDINATES'
        elif not bounds:
            box_status = 'UNVERIFIED_IMAGE'
        else:
            box_status = 'GEOMETRY_VALID_NOT_GT'

        status = 'QUARANTINED_IDENTITY' if not valid_identity else (
            'REVIEW_BBOX_PENDING' if box_status != 'GEOMETRY_VALID_NOT_GT' else 'DEV_REFERENCE_CANDIDATE'
        )
        can_use = valid_identity and box_status == 'GEOMETRY_VALID_NOT_GT'
        rows.append({
            'image_id': id_, 'selected_image_path': str(source),
            'review_image_path': rsource,
            'selected_sha256': s_hash, 'review_sha256': r_hash,
            'actual_sha256': actual, 'identity_errors': ';'.join(issues),
            'preflight_status': status, 'bbox_status': box_status,
            'reference_bbox_xyxy': json.dumps(box if isinstance(box, list) else None),
            'reference_category_unverified_for_prd32': rev.get('target_category', ''),
            'review_action': rev.get('human_review_status_accept_replace', ''),
            'previously_seen_in_prd32': str(s_hash in previous),
            'eligible_reference_roi_diagnostic': str(can_use),
            'human_primary_color_gt': '', 'human_pattern_gt': '',
            'human_visible_design_gt': '', 'attribute_gt_status': 'NOT_ANNOTATED',
            'scoring_eligible': 'FALSE',
            'source_manifest': str(args.selected), 'review_manifest': str(args.review),
        })
        if picture is not None and valid_identity:
            if can_use:
                marked = picture.copy()
                ImageDraw.Draw(marked).rectangle(box, outline='red', width=5)
                roi = picture.crop(tuple(round(x) for x in box))
            else:
                marked, roi = picture, None
        else:
            marked, roi = None, None
        previews.append((id_, status, marked, roi))

    # Only generate output after all records have been inspected.
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.output / 'w7_dev12_prd32_manifest.csv'
    with manifest.open('w', encoding='utf-8-sig', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    width, height = 1200, 6 * 330
    contact = Image.new('RGB', (width, height), 'white')
    painter = ImageDraw.Draw(contact)
    for i, (id_, status, marked, roi) in enumerate(previews):
        left, top = (i % 2) * 600, (i // 2) * 330
        painter.text((left + 10, top + 8), f'{id_}: {status}', fill='black')
        if marked:
            contact.paste(ImageOps.contain(marked, (280, 280)), (left + 10, top + 35))
        else:
            painter.text((left + 40, top + 150), 'IMAGE IDENTITY UNVERIFIED', fill='red')
        if roi:
            contact.paste(ImageOps.contain(roi, (280, 280)), (left + 310, top + 35))
        else:
            painter.text((left + 330, top + 150), 'NO VERIFIED REVIEW ROI', fill='red')
    contact_file = args.output / 'w7_dev12_contact_sheet.jpg'
    contact.save(contact_file, quality=90)
    summary = {
        'status': 'W7_DEV12_PREFLIGHT_PARTIAL' if any(r['preflight_status'] != 'DEV_REFERENCE_CANDIDATE' for r in rows) else 'W7_DEV12_PREFLIGHT_CANDIDATES_READY',
        'total_selected': len(rows),
        'preflight_status_counts': dict(Counter(r['preflight_status'] for r in rows)),
        'bbox_status_counts': dict(Counter(r['bbox_status'] for r in rows)),
        'identity_issues': {r['image_id']: r['identity_errors'] for r in rows if r['identity_errors']},
        'reference_roi_candidates_not_gt': sum(r['eligible_reference_roi_diagnostic'] == 'True' for r in rows),
        'previously_seen_in_prd32': sum(r['previously_seen_in_prd32'] == 'True' for r in rows),
        'human_attribute_gt_verified': 0, 'accuracy_scoring_allowed': False,
        'sealed_holdout_accessed': False,
        'original_inputs_modified': False,
        'manifest': str(manifest), 'contact_sheet': str(contact_file),
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('W7_V2_PREFLIGHT_COMPLETED_WITH_QUARANTINE')

if __name__ == '__main__':
    main()
