"""Sample model-free regression candidates outside known project image manifests.

The result remains an audited candidate pool; historical inventory and semantic
scene screening must be reviewed before calling it a fully independent test set.
"""
import argparse
import base64
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import html
import io
from pathlib import Path
import traceback
import numpy as np
from PIL import Image
from run_attributes import HERE, read, write, sha
from identity_helpers import sealed_identities, TRAIN_REL, VAL_REL, TRAIN_SHA, VAL_SHA


def dhash(image):
    a = np.asarray(image.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
    bits = (a[:, 1:] > a[:, :-1]).reshape(-1)
    return sum(int(v) << i for i, v in enumerate(bits))


def manifest_paths(project, file):
    with file.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    keys = ('source_image','image_path','image_file','image','path')
    if not rows or not any(k in rows[0] for k in keys):
        raise RuntimeError('UNRESOLVED_IMAGE_MANIFEST: '+str(file))
    result = []
    for row in rows:
        value = next((row[k] for k in keys if row.get(k)), None)
        if not value:
            raise RuntimeError('EMPTY_MANIFEST_IMAGE_PATH: '+str(file))
        result.append((project/Path(value.replace('\\','/'))).resolve())
    return result


def choose_new(pool_roots, forbidden_names, forbidden_paths, forbidden_hashes,
               training_dhashes, count, out, max_scan=5000):
    buckets = []
    for root in pool_roots:
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}]
        files.sort(key=lambda p: hashlib.sha256(('prd31-1022-new:'+str(p)).encode()).hexdigest())
        buckets.append(iter(files))
    selected = []
    selected_dhashes = []
    skipped = Counter()
    active = list(range(len(buckets)))
    examined = 0
    while active and len(selected) < count and examined < max_scan:
        for i in active.copy():
            if len(selected) >= count or examined >= max_scan:
                break
            try:
                path = next(buckets[i])
            except StopIteration:
                active.remove(i)
                continue
            examined += 1
            # Sealed names are in forbidden_names; skip before opening bytes.
            if path.name.lower() in forbidden_names or path.resolve() in forbidden_paths:
                skipped['known_path_or_basename'] += 1
                continue
            digest = sha(path)
            if digest in forbidden_hashes:
                skipped['known_byte_hash'] += 1
                continue
            try:
                with Image.open(path) as raw:
                    image = raw.convert('RGB')
                    phash = dhash(image)
                    distances = [(phash ^ v).bit_count() for v in training_dhashes+selected_dhashes]
                    nearest = min(distances) if distances else None
                    if nearest is not None and nearest <= 4:
                        skipped['near_duplicate_dhash_le_4'] += 1
                        continue
                    image_id = f'new1022_{len(selected)+1:04d}'
                    preview = image.copy()
                    preview.thumbnail((900,900))
                    preview_path = out/'previews'/f'{image_id}.jpg'
                    preview.save(preview_path, 'JPEG', quality=85)
                    selected.append({'image_id': image_id, 'path': str(path.resolve()),
                        'sha256': digest, 'size': list(image.size), 'pool_root': str(pool_roots[i]),
                        'dhash': f'{phash:016x}', 'nearest_available_known_or_selected_dhash_distance': nearest,
                        'scene_review': 'pending', 'human_ground_truth_available': False,
                        'independence_status': 'known_manifests_screened_history_confirmation_pending'})
            except (OSError, ValueError):
                skipped['unreadable_image'] += 1
                continue
            forbidden_hashes.add(digest)
            selected_dhashes.append(phash)
    return selected, {'examined_file_names': examined, 'skipped': dict(skipped)}


def collect(project, out, count=20, supplied_sealed=None, supplied_roots=None):
    if sha(project/TRAIN_REL) != TRAIN_SHA or sha(project/VAL_REL) != VAL_SHA:
        raise RuntimeError('CURRENT_V5_TRAIN_OR_VAL_MANIFEST_CHANGED')
    sealed_hashes, sealed_names, sealed_ids, sealed_audit = sealed_identities(project, supplied_sealed)
    history = read(HERE/'inputs/HISTORY_EXCLUSIONS.json')
    forbidden_hashes = set(history['sha256']) | sealed_hashes
    forbidden_names = set(history['basenames']) | sealed_names
    # Inspect current configs and archived copies with the project naming scheme.
    manifests = set(project.rglob('prd_8class_train*.csv')) | set(project.rglob('prd_8class_val*.csv'))
    manifests.update([project/TRAIN_REL, project/VAL_REL])
    forbidden_paths = set()
    current_train = manifest_paths(project, project/TRAIN_REL)
    inventory = []
    for file in sorted(manifests):
        paths = manifest_paths(project, file)
        forbidden_paths.update(paths)
        forbidden_names.update(p.name.lower() for p in paths)
        inventory.append({'path': str(file), 'sha256': sha(file), 'rows': len(paths),
                          'unique_source_paths': len(set(paths))})
    known_dhashes = []
    missing = []
    for n, path in enumerate(sorted(forbidden_paths)):
        if n % 100 == 0:
            print(f'IDENTITY AUDIT {n}/{len(forbidden_paths)} known paths', flush=True)
        if path.name.lower() in sealed_names:
            continue
        if not path.is_file():
            missing.append(str(path))
            continue
        forbidden_hashes.add(sha(path))
        with Image.open(path) as im:
            known_dhashes.append(dhash(im))
    pool_roots = [Path(p).resolve() for p in supplied_roots] if supplied_roots else sorted({p.parent for p in current_train if p.parent.is_dir()})
    if not pool_roots:
        raise RuntimeError('NO_LOCAL_POOL_ROOT; supply --pool-root containing source images')
    if any(not p.is_dir() for p in pool_roots):
        raise RuntimeError('POOL_ROOT_MISSING')
    (out/'previews').mkdir()
    images, screening = choose_new(pool_roots, forbidden_names, forbidden_paths,
                                   forbidden_hashes, known_dhashes, count, out)
    for item in inventory:
        if sha(Path(item['path'])) != item['sha256']:
            raise RuntimeError('IMAGE_MANIFEST_CHANGED_DURING_SELECTION')
    audit = {'status': 'NEW_REGRESSION_CANDIDATES_READY' if len(images)==count else 'INSUFFICIENT_NEW_CANDIDATES',
        'requested': count, 'selected': len(images), 'manifest_inventory': inventory,
        'missing_known_image_paths': missing, 'known_byte_and_dhash_audit_complete': not missing,
        'pool_roots': [str(p) for p in pool_roots], 'sealed_identity_audit': sealed_audit,
        'selection_before_inference': True, 'model_predictions_used_to_select': False,
        'near_duplicate_threshold_dhash': 4, 'new_human_labels_created': 0,
        'known_image_source_paths': len(forbidden_paths), 'available_known_dhashes': len(known_dhashes),
        'training_inventory_limit': 'current/archived prd_8class_train*.csv and val*.csv plus packaged history hashes; unknown historical manifests/pretraining not certified',
        'cross_history_near_duplicate_audit_complete': False,
        'certified_blind_or_accuracy_test': False, 'screening': screening}
    write(out/'selection_audit.json', audit)
    write(out/'images.json', {'schema': 'prd31_1022_new_regression_candidates_v1',
        'scope': 'known-project-manifest isolated regression candidates; scene/history confirmation pending',
        'images': images, 'accuracy_claim': False})
    with (out/'scene_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['image_id','scene_review','multi_instance','occluded','loose_garment','notes'])
        writer.writeheader()
        for r in images:writer.writerow({'image_id':r['image_id'],'scene_review':'pending'})
    cards = ''.join(f'<article><h2>{html.escape(r["image_id"])}</h2><img src="previews/{r["image_id"]}.jpg" alt="待复核样本"><p>{html.escape(Path(r["path"]).name)}</p></article>' for r in images)
    (out/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>新增回归候选</title><style>body{font:16px sans-serif;margin:24px}main{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}img{width:100%;height:360px;object-fit:contain}article{border:1px solid #ddd;padding:12px}</style><h1>新增回归候选</h1><p>尚未人工确认场景及完整历史隔离，不作为盲测精度报告。场景记录填写scene_review.csv。</p><main>'+cards+'</main>',encoding='utf-8')
    return audit


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='/workspace/fashion_multimodal_analysis')
    ap.add_argument('--output-root',default='/workspace/prd31_week7_1022_new_samples')
    ap.add_argument('--count',type=int,default=20)
    ap.add_argument('--w7-identities')
    ap.add_argument('--pool-root',action='append')
    args=ap.parse_args()
    out=Path(args.output_root).resolve()/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    try:
        result=collect(Path(args.project).resolve(),out,args.count,args.w7_identities,args.pool_root)
        print(result['status'],out,flush=True)
    except BaseException:
        write(out/'selection_audit.json',{'status':'BLOCKED_OR_INTERRUPTED','error':traceback.format_exc()})
        raise


if __name__=='__main__':main()
