#!/usr/bin/env python3
# coding: utf-8
"""Local-only, digest-based comparison between a sealed 8-image manifest and 60-image DEV cohort.

Sealed metadata is read only when the user explicitly supplies --allow-sealed-metadata-read.
No held-out pixels are ever loaded; no held-out paths/IDs/labels/digests are ever emitted.
This is a diagnostic, NOT clearance for official cohort freeze.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path

HEX64 = re.compile(r"[0-9a-fA-F]{64}\Z")
HEX16 = re.compile(r"(?:0[xX])?[0-9a-fA-F]{1,16}\Z")
RESTRICTED = {'holdout', 'sealed', 'test', 'testing', 'eval', 'evaluation', 'val', 'valid', 'validation', 'fullval', 'devtest'}

def sha_file(path: Path):
    obj = hashlib.sha256()
    with path.open('rb') as fin:
        for block in iter(lambda: fin.read(1024 * 1024), b''):
            obj.update(block)
    return obj.hexdigest()

def image_dhash(path: Path) -> int:
    # Common 9x8 grayscale horizontal dHash. ONLY a heuristic until the sealed dHash implementation is verified.
    from PIL import Image
    with Image.open(path) as image:
        im = image.convert('L').resize((9, 8), resample=Image.Resampling.LANCZOS)
        pix = im.load()
        bits = 0
        for y in range(8):
            for x in range(8):
                bits = (bits << 1) | int(pix[x + 1, y] > pix[x, y])
        return bits

def parse_dhash(value):
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < (1 << 64):
        return value
    if not isinstance(value, str) or not HEX16.fullmatch(value):
        raise ValueError('INVALID_SEALED_DHASH_FORMAT')
    return int(value, 16)

def parse_sealed_metadata(path: Path):
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, list) or len(data) != 8:
        raise ValueError('SEALED_MANIFEST_NOT_8_ROWS')
    digests, dhashes = [], []
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError('SEALED_ROW_NOT_OBJECT')
        s = entry.get('source_sha256')
        if not isinstance(s, str) or not HEX64.fullmatch(s):
            raise ValueError('INVALID_SEALED_SOURCE_SHA256')
        digests.append(s.lower())
        dhashes.append(parse_dhash(entry.get('dhash64')))
    if len(set(digests)) != 8:
        raise ValueError('SEALED_SOURCE_SHA256_NOT_UNIQUE')
    # Return digests in memory only. Never serialize them or their associated metadata.
    return set(digests), dhashes

def read_cohort(manifest: Path, reference: Path, root: Path):
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    ref = json.loads(reference.read_text(encoding='utf-8-sig'))
    if len(rows) != 60 or len(ref.get('images', [])) != 60:
        raise ValueError('DEV60_NOT_60')
    expected = [(item['case_id'],item['image_sha256']) for item in ref['images']]
    found = [(item.get('case_id'),item.get('image_sha256')) for item in rows]
    if expected != found or [x[0] for x in found] != [f'fresh_dev_{i:03}' for i in range(1,61)]:
        raise ValueError('DEV60_MANIFEST_NOT_FROZEN_REFERENCE')
    if sha_file(manifest) != ref.get('original_stage1_model_manifest_sha256'):
        raise ValueError('DEV60_MANIFEST_BYTES_CHANGED')
    if len({item[1] for item in found}) != 60 or not all(isinstance(x[1], str) and HEX64.fullmatch(x[1]) for x in found):
        raise ValueError('DEV60_IMAGE_SHA_INVALID_OR_REPEATED')
    root = root.resolve()
    paths = []
    for i, row in enumerate(rows):
        p = Path(row['image_path']).resolve()
        if not p.is_relative_to(root) or 'train' not in (segment.lower() for segment in p.parts) or any(segment.lower() in RESTRICTED for segment in p.parts):
            raise ValueError(f'DEV60_UNSAFE_PATH_ROW_{i+1}')
        if sha_file(p) != row['image_sha256']:
            raise ValueError(f'DEV60_CHANGED_IMAGE_ROW_{i+1}')
        paths.append(p)
    return rows, paths

def audit(manifest, reference, sealed, train_root, skip_dhash=False):
    rows, candidate_files = read_cohort(manifest, reference, train_root)
    sealed_sha, sealed_dh = parse_sealed_metadata(sealed)
    overlaps = sealed_sha.intersection({r['image_sha256'] for r in rows})
    result = {
        'schema': 'prd32_dev60_stage1f_sealed_digest_isolation_v1',
        'scope': 'LOCAL_ONLY_SEALED_METADATA_DIGEST_SCREENING',
        'dev_image_count': 60,
        'dev_original_12_and_new_48_identity_checked': True,
        'dev_image_bytes_rehashed': True,
        'sealed_manifest_records': 8,
        'sealed_source_sha256_unique_count': len(sealed_sha),
        'exact_source_sha_overlap_count': len(overlaps),
        'sealed_images_opened': False,
        'sealed_manifest_metadata_read_locally': True,
        'sealed_image_paths_ids_labels_or_hashes_exported': False,
        'sealed_source_provenance_independently_confirmed': False,
        'official_freeze_approved': False,
        'main_ab_inference_ran': False,
        'publish_allowed': False,
    }
    if skip_dhash:
        result.update({'dhash_heuristic_performed': False,
                       'near_duplicate_cross_split_exclusion_verified': False})
    else:
        from PIL import Image  # sanity-check dependency before long processing
        values = [image_dhash(p) for p in candidate_files]
        min_distances = [min((v ^ h).bit_count() for h in sealed_dh) for v in values]
        result.update({
            'dhash_heuristic_performed': True,
            'dhash_algorithm_matches_sealed_source_verified': False,
            'dhash_screen_threshold_bits': 6,
            'possible_near_duplicate_dev_images_le6': sum(d <= 6 for d in min_distances),
            'minimum_cross_split_dhash_distance_bits': min(min_distances),
            'near_duplicate_cross_split_exclusion_verified': False,
        })
    if overlaps:
        result['status'] = 'BLOCKED_EXACT_SOURCE_SHA_OVERLAP'
    elif result.get('possible_near_duplicate_dev_images_le6', 0):
        result['status'] = 'REVIEW_POSSIBLE_DHASH_NEAR_DUPLICATES'
    else:
        result['status'] = 'NO_EXACT_SHA_OVERLAP_PROVENANCE_AND_NEAR_DUPLICATE_REVIEW_PENDING'
    result['notes'] = [
        'This compares source_sha256 in the sealed manifest with current dev image bytes, without reading held-out images.',
        'Six identical copies of a PRD31 W7 manifest do not independently establish that it is the intended PRD32 holdout cohort.',
        'dHash is heuristic. The hash generation algorithm and thumbnail/encoding provenance have not been verified.',
        'Do not call this an official freeze, release permission, or holdout performance evaluation.',
    ]
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('/workspace/PRD32_FIXED_DEV_V2_60IMG_CANDIDATES_20261030/v5_image_only_inputs_60.jsonl'))
    parser.add_argument('--reference', type=Path, default=Path('/workspace/PRD32_DEV60_STAGE1E_USER_CONFIRMED/dev60_image_sha_reference.json'))
    parser.add_argument('--sealed', type=Path, default=Path('/workspace/PRD31_W7_M2/inputs/holdout_SEALED_DO_NOT_OPEN.json'))
    parser.add_argument('--train-root', type=Path, default=Path('/workspace/fashion_data'))
    parser.add_argument('--output', type=Path, default=Path('/workspace/PRD32_DEV60_STAGE1F_LOCAL_ISOLATION_RESULT.json'))
    parser.add_argument('--skip-dhash', action='store_true', help='Check SHA only; skips perceptual-hash heuristic')
    parser.add_argument('--allow-sealed-metadata-read', action='store_true', help='Explicitly authorize LOCAL-only inspection of digest fields in sealed JSON; images are not opened')
    args = parser.parse_args()
    if not args.allow_sealed_metadata_read:
        parser.error('Explicit --allow-sealed-metadata-read required; otherwise do not access sealed metadata')
    if args.output.exists():
        parser.error('Output already exists; will not overwrite previous evidence')
    try:
        result = audit(args.manifest, args.reference, args.sealed, args.train_root, args.skip_dhash)
    except Exception as exc:
        # Catch errors without leaking sealed content, hashes, or paths.
        result = {'schema': 'prd32_dev60_stage1f_sealed_digest_isolation_v1',
                  'status': 'BLOCKED_LOCAL_VALIDATION_FAILURE',
                  'error_code': str(exc).split(':')[0][:96],
                  'official_freeze_approved': False,
                  'publish_allowed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('STAGE1F_STATUS:', result['status'])
    print('REPORT:', args.output)
    print('NO_HOLDOUT_IMAGE_OPENED_NO_MAIN_AB_NO_OFFICIAL_FREEZE')

if __name__ == '__main__':
    main()
