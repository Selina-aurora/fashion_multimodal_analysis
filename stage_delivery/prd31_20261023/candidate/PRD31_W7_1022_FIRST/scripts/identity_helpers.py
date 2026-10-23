"""W7 identity fields only; never decodes sealed images or exports labels."""
import hashlib,json,re
from pathlib import Path
TRAIN_REL="configs/prd_8class_train_v5_balanced150_20261008.csv"
VAL_REL="configs/prd_8class_val_v2_full_targets.csv"
TRAIN_SHA="e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f"
VAL_SHA="39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93"

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sealed_identities(project, supplied=None):
    """Retain identity fields only. Never resolve/open a sealed image here."""
    options = [supplied] if supplied else [
        project.parent / 'PRD31_W7_M3/inputs/holdout_SEALED_DO_NOT_OPEN.json',
        project.parent / 'PRD31_W7_M2/inputs/holdout_SEALED_DO_NOT_OPEN.json',
    ]
    manifest = next((Path(p) for p in options if p and Path(p).is_file()), None)
    if manifest is None:
        raise RuntimeError('W7_IDENTITY_MANIFEST_MISSING: no images exported; provide --w7-identities PATH')
    hashes, names, ids = set(), set(), set()
    hash_keys = {'source_image_sha256', 'image_sha256', 'source_sha256'}
    image_keys = {'source_image', 'image_path', 'image_file', 'file_name', 'filename', 'image'}
    def visit(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in hash_keys and isinstance(v, str) and re.fullmatch('[a-fA-F0-9]{64}', v):
                    hashes.add(v.lower())
                elif k in image_keys and isinstance(v, str) and Path(v).suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                    names.add(Path(v.replace('\\', '/')).name.lower())
                elif k == 'record_id' and isinstance(v, str):
                    ids.add(v)
                elif k in {'sha256', 'source_hashes', 'image_hashes'} and isinstance(v, list):
                    hashes.update(x.lower() for x in v if isinstance(x, str) and re.fullmatch('[a-fA-F0-9]{64}', x))
                elif isinstance(v, (dict, list)):
                    visit(v)
        elif isinstance(node, list):
            for item in node:
                visit(item)
    visit(read(manifest))
    # Image-level identity is mandatory; instance IDs alone cannot exclude other
    # garments from the same source image. Unexpected schemas fail closed.
    if max(len(hashes), len(names)) < 8:
        raise RuntimeError('W7_IDENTITY_SCHEMA_UNRESOLVED: fewer than eight image hashes/names; no images exported')
    return hashes, names, ids, {'manifest_path': str(manifest), 'manifest_sha256': digest(manifest),
        'retained_image_hashes': len(hashes), 'retained_image_names': len(names),
        'retained_record_ids': len(ids), 'sealed_images_decoded_or_exported': False,
        'labels_used_or_exported': False, 'method': 'image identity fields only; basename exclusion is conservative'}
