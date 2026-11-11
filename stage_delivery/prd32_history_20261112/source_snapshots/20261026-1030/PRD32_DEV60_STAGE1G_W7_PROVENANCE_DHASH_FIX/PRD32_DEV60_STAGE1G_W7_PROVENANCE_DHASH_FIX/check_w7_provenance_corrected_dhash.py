#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD32 DEV60 vs original W7 sealed 8: local SHA and ORIGINAL dhash cross-check.

Safety: reads only metadata from sealed JSON with explicit opt-in. Does NOT open
holdout pixels. Output contains only aggregate metrics, not holdout IDs, paths,
source hashes, dhashes, labels, or individual-case nearest neighbors.
NOT an official PRD32 cohort approval.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

EXPECTED_W7_SEALED_FILE_SHA = '750b7a06e49c6db573e5fe642c4387600759993e197a5abe23c6a4ce6bdd37d4'
HEX64 = re.compile(r'^[0-9a-fA-F]{64}$')
HEX16 = re.compile(r'^[0-9a-fA-F]{1,16}$')
RESTRICTED = {'holdout','sealed','test','testing','val','valid','validation','fullval','devtest','evaluation','eval'}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def original_w7_dhash(path: Path) -> int:
    """Exact algorithm in W7 select_week7.py: EXIF, L, 9x8 LANCZOS, left>right."""
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        grayscale = ImageOps.exif_transpose(image).convert('L').resize((9, 8), Image.Resampling.LANCZOS)
        pix = list(grayscale.getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(pix[y * 9 + x] > pix[y * 9 + x + 1])
    return bits


def verify_w7_algorithm_source(script: Path) -> bool:
    compact = ''.join(script.read_text(encoding='utf-8').split())
    mandatory = (
        "ImageOps.exif_transpose(image).convert('L').resize((9,8),Image.Resampling.LANCZOS)",
        'int(pix[y*9+x]>pix[y*9+x+1])',
    )
    return all(token in compact for token in mandatory)


def valid_sealed_8(path: Path):
    if sha_file(path) != EXPECTED_W7_SEALED_FILE_SHA:
        raise ValueError('W7_SEALED_FILE_SHA_NOT_ORIGINAL')
    obj = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(obj, list) or len(obj) != 8:
        raise ValueError('W7_SEALED_RECORDS_NOT_8')
    shas, dhashes = [], []
    for item in obj:
        if not isinstance(item, dict) or not isinstance(item.get('source_sha256'), str) or not HEX64.fullmatch(item['source_sha256']):
            raise ValueError('W7_SEALED_INVALID_SOURCE_SHA')
        d = item.get('dhash64')
        if not isinstance(d, str) or not HEX16.fullmatch(d):
            raise ValueError('W7_SEALED_INVALID_DHASH')
        shas.append(item['source_sha256'].lower())
        dhashes.append(int(d, 16))
    if len(set(shas)) != 8:
        raise ValueError('W7_SEALED_SOURCE_SHA_NOT_UNIQUE')
    return shas, dhashes


def verify_w7_original_split(w7_m3_dir: Path, sealed_file: Path, shas: list[str]) -> dict:
    # Nothing from held-out records is disclosed, only booleans.
    other = w7_m3_dir / 'holdout_SEALED_DO_NOT_OPEN.json'
    selection = w7_m3_dir / 'W7_dev12_SELECTED.json'
    audit = w7_m3_dir / 'selection_audit.json'
    human_choice = w7_m3_dir / 'W7_DEV11_HUMAN_CHOICE_ORIGINAL.json'
    if any(not p.is_file() for p in (other, selection, audit, human_choice)):
        raise ValueError('W7_M3_ORIGINAL_EVIDENCE_FILES_MISSING')
    if sha_file(other) != EXPECTED_W7_SEALED_FILE_SHA or other.read_bytes() != sealed_file.read_bytes():
        raise ValueError('W7_M3_SEALED_COPY_MISMATCH')
    original_dev = json.loads(selection.read_text(encoding='utf-8'))
    m3audit = json.loads(audit.read_text(encoding='utf-8'))
    choice = json.loads(human_choice.read_text(encoding='utf-8'))
    if not isinstance(original_dev, list) or len(original_dev) != 12:
        raise ValueError('W7_M3_DEV12_WRONG_RECORDS')
    if [x.get('image_id') for x in original_dev] != [f'dev_{i:02d}' for i in range(1,13)]:
        raise ValueError('W7_M3_DEV_IDS_CHANGED')
    deva = [x.get('source_sha256') for x in original_dev]
    if not all(isinstance(h, str) and HEX64.fullmatch(h) for h in deva):
        raise ValueError('W7_M3_DEV_SOURCE_SHA_INVALID')
    if len(set(h.lower() for h in deva) | set(shas)) != 20:
        raise ValueError('W7_M3_ORIGINAL_SPLIT_OVERLAP')
    if m3audit.get('holdout_manifest_sha256') != EXPECTED_W7_SEALED_FILE_SHA:
        raise ValueError('W7_M3_SELECTION_AUDIT_SHA_MISMATCH')
    if choice.get('candidate_id') != 'cand_02' or choice.get('human_scene_confirmed') is not True:
        raise ValueError('W7_M3_DEV11_CHOICE_NOT_VERIFIED')
    return {'w7_m3_12_dev_8_holdout_original_split_verified': True,
            'w7_m3_sealed_copy_byte_identical': True,
            'w7_dev11_documented_replacement_verified': True}


def verify_dev60(manifest: Path, reference: Path, train_root: Path):
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    ref = json.loads(reference.read_text(encoding='utf-8-sig'))
    snap = ref.get('images', [])
    if len(rows) != 60 or len(snap) != 60:
        raise ValueError('DEV60_COUNT_INVALID')
    ordered = [(r.get('case_id'), r.get('image_sha256')) for r in rows]
    expected = [(r.get('case_id'), r.get('image_sha256')) for r in snap]
    if ordered != expected or [cid for cid,_ in ordered] != [f'fresh_dev_{i:03}' for i in range(1,61)]:
        raise ValueError('DEV60_REFERENCE_IDENTITY_MISMATCH')
    if sha_file(manifest) != ref.get('original_stage1_model_manifest_sha256'):
        raise ValueError('DEV60_MANIFEST_BYTES_CHANGED')
    shas = [h for _,h in ordered]
    if len(set(shas)) != 60 or not all(isinstance(h,str) and HEX64.fullmatch(h) for h in shas):
        raise ValueError('DEV60_SHA_MISSING_OR_REPEATED')
    root = train_root.resolve()
    images = []
    for i,r in enumerate(rows,1):
        p = Path(r['image_path']).resolve()
        if not p.is_relative_to(root) or 'train' not in [q.lower() for q in p.parts] or any(q.lower() in RESTRICTED for q in p.parts):
            raise ValueError('DEV60_PATH_UNSAFE')
        if not p.is_file() or sha_file(p) != r['image_sha256']:
            raise ValueError('DEV60_IMAGE_CHANGED')
        images.append(p)
    return shas, images


def audit(manifest: Path, reference: Path, w7sealed: Path, w7algorithm: Path,
          w7_m3: Path, train_root: Path) -> dict:
    if not verify_w7_algorithm_source(w7algorithm):
        raise ValueError('W7_SOURCE_DHASH_ALGORITHM_DIFFERENT')
    shas, images = verify_dev60(manifest, reference, train_root)
    held_shas, held_dh = valid_sealed_8(w7sealed)
    lineage = verify_w7_original_split(w7_m3, w7sealed, held_shas)
    overlaps = set(shas) & set(held_shas)
    corrected_dh = [original_w7_dhash(image) for image in images]
    distances = [min((v ^ h).bit_count() for h in held_dh) for v in corrected_dh]
    zero4 = sum(d <= 4 for d in distances)
    zero6 = sum(d <= 6 for d in distances)
    result = {
        'schema': 'prd32_dev60_stage1g_w7_provenance_corrected_dhash_v1',
        'scope': 'LOCAL_DEV60_VS_W7_8_DIGEST_AUDIT_ONLY',
        'w7_original_sealed_file_sha_verified_against_protocol': True,
        'w7_original_dhash_algorithm_source_matched': True,
        'w7_original_dhash_rule': 'exif_transpose_grayscale_9x8_lanczos_left_gt_right',
        **lineage,
        'dev60_image_count_and_identity_verified': len(images) == 60,
        'dev60_image_bytes_rehashed': True,
        'w7_sealed_record_count': 8,
        'exact_source_sha_overlap_count': len(overlaps),
        'possible_near_duplicate_dev_images_le4': zero4,
        'possible_near_duplicate_dev_images_le6': zero6,
        'minimum_cross_split_dhash_distance_bits_corrected': min(distances),
        'dhash_is_heuristic_not_semantic_duplicate_proof': True,
        'w7_is_prd32_intended_holdout_confirmed': False,
        'prd32_official_v2_freeze_approved': False,
        'holdout_images_opened': False,
        'heldout_ids_paths_hashes_or_labels_exported': False,
        'main_ab_inference_ran': False,
        'publish_allowed': False,
    }
    if overlaps:
        result['status'] = 'BLOCKED_W7_EXACT_SHA_OVERLAP'
    elif zero6:
        result['status'] = 'REVIEW_W7_CROSS_SPLIT_NEAR_DUPLICATES'
    else:
        result['status'] = 'W7_SOURCE_LINEAGE_AND_DIGEST_SCREEN_PASS_PRD32_COHORT_LINK_PENDING'
    result['notes'] = [
        'W7 original 12+8 split provenance is traced and matches the original sealed byte SHA.',
        'This does NOT prove W7 eight images are the intended PRD32 holdout cohort.',
        'Original W7 dHash has LEFT > RIGHT polarity and EXIF transpose; Stage1F had reversed polarity and no EXIF transpose.',
        'Near-duplicate screening is heuristic only; image similarity not guaranteed.',
        'No held-out image pixels were opened and no official PRD32 V2 freeze is granted.'
    ]
    return result


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--manifest', type=Path, default=Path('/workspace/PRD32_FIXED_DEV_V2_60IMG_CANDIDATES_20261030/v5_image_only_inputs_60.jsonl'))
    a.add_argument('--reference', type=Path, default=Path('/workspace/PRD32_DEV60_STAGE1E_USER_CONFIRMED/dev60_image_sha_reference.json'))
    a.add_argument('--sealed', type=Path, default=Path('/workspace/PRD31_W7_M2/inputs/holdout_SEALED_DO_NOT_OPEN.json'))
    a.add_argument('--w7-algorithm', type=Path, default=Path('/workspace/PRD31_W7_M2/scripts/select_week7.py'))
    a.add_argument('--w7-m3', type=Path, default=Path('/workspace/PRD31_W7_M3/inputs'))
    a.add_argument('--train-root', type=Path, default=Path('/workspace/fashion_data'))
    a.add_argument('--output', type=Path, default=Path('/workspace/PRD32_DEV60_STAGE1G_CORRECTED_W7_RESULT.json'))
    a.add_argument('--allow-sealed-metadata-read', action='store_true')
    args = a.parse_args()
    if not args.allow_sealed_metadata_read:
        a.error('Requires explicit permission for local sealed digest metadata only; never open sealed pictures')
    if args.output.exists():
        a.error('Refusing to overwrite prior evidence')
    try:
        result = audit(args.manifest,args.reference,args.sealed,args.w7_algorithm,args.w7_m3,args.train_root)
    except Exception as exc:
        # Do not leak any restricted file content in error messages.
        result = {'schema':'prd32_dev60_stage1g_w7_provenance_corrected_dhash_v1',
                  'status':'BLOCKED_VALIDATION_ERROR',
                  'error_code': (str(exc) if isinstance(exc, ValueError) and str(exc).startswith(('W7_','DEV60_')) else type(exc).__name__),
                  'prd32_official_v2_freeze_approved': False,
                  'publish_allowed':False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('STAGE1G_STATUS:',result['status'])
    print('RESULT_FILE:',args.output)
    print('NO_HOLDOUT_IMAGE_OPENED_NO_MAIN_AB_NO_PRD32_FREEZE')

if __name__ == '__main__':
    main()
