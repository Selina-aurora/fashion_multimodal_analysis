#!/usr/bin/env python3
"""Prepare a read-only PRD 3.2 preflight from W7 Dev12 development images.

Validates hashes, joins the selected JSON and human review CSV by image_id,
flags missing/invalid reference boxes, and builds a visual contact sheet.
This does NOT infer attributes, declare GT accuracy, or access the sealed holdout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

DEFAULT_SELECTED = Path(
    '/workspace/PRD31_W7_ASSIST/inputs/W7_dev12_SELECTED.json'
)
DEFAULT_REVIEW = Path(
    '/workspace/PRD31_W7_M2/inputs/W7_dev12_human_screen_2026-10-15.csv'
)
DEFAULT_PREVIOUS = Path('/workspace/PRD32_DEV_20261026/dev_manifest.csv')
DEFAULT_OUTPUT = Path('/workspace/PRD32_W7_DEV12_PREFLIGHT_20261026')
BLOCKED_TERMS = ('sealed', 'holdout')


def is_blocked(path: Path | str) -> bool:
    return any(term in str(path).lower() for term in BLOCKED_TERMS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_box(value: str | None) -> list[float] | None:
    if value is None or not str(value).strip():
        return None
    nums = re.findall(r'-?\d+(?:\.\d+)?', str(value))
    if len(nums) != 4:
        raise ValueError(f'Expected 4 coordinates; got {value!r}')
    return [float(x) for x in nums]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected', type=Path, default=DEFAULT_SELECTED)
    parser.add_argument('--review', type=Path, default=DEFAULT_REVIEW)
    parser.add_argument('--previous-manifest', type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    for path in (args.selected, args.review, args.output):
        if is_blocked(path):
            parser.error('Refusing to access sealed/holdout paths')
    for path in (args.selected, args.review):
        if not path.is_file():
            parser.error(f'Missing input: {path}')
    if args.output.exists():
        parser.error(f'Output already exists (will not overwrite): {args.output}')

    selected = json.loads(args.selected.read_text(encoding='utf-8'))
    if not isinstance(selected, list) or len(selected) != 12:
        raise ValueError('Expected 12 records in W7 selected development manifest')
    with args.review.open(encoding='utf-8-sig', newline='') as stream:
        reviews = list(csv.DictReader(stream))
    if len(reviews) != 12:
        raise ValueError('Expected 12 W7 human review CSV rows')
    review_map = {r['image_id']: r for r in reviews}
    if len(review_map) != 12:
        raise ValueError('Duplicate image_id in review CSV')

    seen_ids = set()
    if args.previous_manifest.is_file():
        if is_blocked(args.previous_manifest):
            parser.error('Refusing a holdout reference manifest')
        with args.previous_manifest.open(encoding='utf-8-sig', newline='') as stream:
            previous_hashes = {
                r.get('image_sha256', '') for r in csv.DictReader(stream)
            }
    else:
        previous_hashes = set()

    rows: list[dict] = []
    visual_items = []
    for record in selected:
        image_id = record['image_id']
        if image_id in seen_ids or image_id not in review_map:
            raise ValueError(f'Duplicate or unmatched selected image: {image_id}')
        seen_ids.add(image_id)
        review = review_map[image_id]
        image_path = Path(record['source_image'])
        if is_blocked(image_path):
            raise ValueError(f'Sealed/holdout image not allowed: {image_path}')
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        expected_hash = record['source_sha256']
        if sha256_file(image_path) != expected_hash:
            raise ValueError(f'Source image SHA256 mismatch: {image_id}')
        if review.get('source_sha256') != expected_hash:
            raise ValueError(f'Review image SHA256 mismatch: {image_id}')
        if review.get('source_image') != str(image_path):
            raise ValueError(f'Source path differs between manifests: {image_id}')

        with Image.open(image_path) as im:
            image = im.convert('RGB')
        w, h = image.size
        bbox = parse_box(review.get('target_garment_bbox_xyxy'))
        if bbox is None:
            bbox_status = 'MISSING'
        elif not (0 <= bbox[0] < bbox[2] <= w and 0 <= bbox[1] < bbox[3] <= h):
            bbox_status = 'INVALID'
        else:
            bbox_status = 'GEOMETRY_VALID_NOT_GT_VERIFIED'

        rows.append({
            'image_id': image_id,
            'image_path': str(image_path),
            'image_sha256': expected_hash,
            'dataset': record.get('dataset', ''),
            'review_category': review.get('target_category', ''),
            'review_decision': review.get('human_review_status_accept_replace', ''),
            'bbox_status': bbox_status,
            'bbox_xyxy': json.dumps(bbox, ensure_ascii=False),
            'previously_seen_in_prd32': str(expected_hash in previous_hashes),
            'human_color_gt': '',
            'human_pattern_gt': '',
            'human_design_gt': '',
            'attribute_gt_status': 'NOT_ANNOTATED',
            'review_notes': review.get('review_notes', ''),
        })
        if bbox_status == 'GEOMETRY_VALID_NOT_GT_VERIFIED':
            marked = image.copy()
            ImageDraw.Draw(marked).rectangle(bbox, outline='red', width=5)
            crop = image.crop(tuple(map(int, bbox)))
        else:
            marked = image
            crop = None
        visual_items.append((image_id, marked, crop, bbox_status))

    if {r['image_id'] for r in reviews} != seen_ids:
        raise ValueError('Selected/review image IDs do not match')

    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.output / 'w7_dev12_prd32_manifest.csv'
    with manifest.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tile_w, tile_h = 600, 330
    sheet = Image.new('RGB', (tile_w * 2, tile_h * 6), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (image_id, marked, crop, status) in enumerate(visual_items):
        col, row = i % 2, i // 2
        x, y = col * tile_w, row * tile_h
        draw.text((x + 10, y + 8), f'{image_id} | {status}', fill='black')
        sheet.paste(ImageOps.contain(marked, (280, 280)), (x + 10, y + 35))
        if crop is None:
            draw.text((x + 330, y + 145), 'NO REVIEWED BBOX', fill='red')
        else:
            sheet.paste(ImageOps.contain(crop, (280, 280)), (x + 310, y + 35))

    contact = args.output / 'w7_dev12_contact_sheet.jpg'
    sheet.save(contact, quality=90)
    summary = {
        'status': 'W7_DEV12_PREFLIGHT_ONLY',
        'n_images': len(rows),
        'categories': dict(Counter(r['review_category'] for r in rows)),
        'bbox_status': dict(Counter(r['bbox_status'] for r in rows)),
        'previously_seen': sum(r['previously_seen_in_prd32'] == 'True' for r in rows),
        'attribute_gt_verified': 0,
        'sealed_holdout_used': False,
        'image_hashes_verified': True,
        'full_prd_acceptance': False,
    }
    summary_path = args.output / 'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('MANIFEST:', manifest)
    print('CONTACT_SHEET:', contact)
    print('SUMMARY:', summary_path)
    print('W7_PREFLIGHT_COMPLETED')


if __name__ == '__main__':
    main()
