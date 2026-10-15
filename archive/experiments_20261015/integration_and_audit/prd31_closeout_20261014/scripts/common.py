"""Shared, dependency-free checks. Unknown human labels never become negatives."""
from pathlib import Path
import csv, hashlib, json, math

BASE = Path(__file__).resolve().parent.parent
REGIONS = ['collar', 'cuff', 'hem', 'pocket', 'waist']
ATTRS = ['primary_color', 'pattern', 'sleeve_length', 'neckline', 'silhouette_fit', 'fashion_style']
SCHEMA = 'prd31_same_cohort_review_v1'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def csvout(path, data, fields=None):
    fields = fields or (list(data[0]) if data else ['status'])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(data)

def box_valid(b, normalized=False):
    return (isinstance(b, list) and len(b) == 4 and
            all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in b)
            and 0 <= b[0] < b[2] and 0 <= b[1] < b[3]
            and (not normalized or max(b) <= 1))

def pixels(b, size):
    if not box_valid(b, True): raise ValueError('Invalid normalized box')
    return [b[0]*size[0], b[1]*size[1], b[2]*size[0], b[3]*size[1]]

def iou(a, b):
    if not box_valid(a) or not box_valid(b): return 0.0
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return inter / ((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def validate_review(obj, manifest, taxonomy):
    """Reject changed identities and impossible geometry; preserve pending records."""
    if obj.get('schema_version') != SCHEMA: raise ValueError('Unsupported review schema')
    if obj.get('manifest_sha256') != sha(BASE/'reference/cohort.json'):
        raise ValueError('Review belongs to a different frozen cohort')
    expected = {c['case_id']: c for c in manifest}
    if not isinstance(obj.get('cases'), dict) or set(obj['cases']) != set(expected):
        raise ValueError('Review must retain all twelve unique case IDs')
    if not isinstance(obj.get('reviewer', ''), str): raise ValueError('Reviewer must be text')
    pending = []
    for cid, r in obj['cases'].items():
        if r.get('source_sha256') != expected[cid]['source_sha256']: raise ValueError('Source hash mismatch: '+cid)
        if r.get('target_box') is not None and not box_valid(r['target_box'], True): raise ValueError('Bad target box: '+cid)
        if r.get('category','') not in ['', 'uncertain','top','pants','skirt','outerwear','dress','shoe','bag','accessory']:
            raise ValueError('Unknown garment category')
        if r.get('target_status','') not in ['', 'correct_target','partial_usable','wrong_target','uncertain']:
            raise ValueError('Unknown target status')
        for attr in ATTRS:
            label = r.get('labels', {}).get(attr, '')
            if label not in ['', 'uncertain', 'not_applicable'] + taxonomy[attr]:
                raise ValueError('Unknown attribute label: '+cid+'/'+attr+'/'+str(label))
        for name in REGIONS:
            p = r.get('regions', {}).get(name, {})
            if p.get('presence','') not in ['', 'yes','no','uncertain']: raise ValueError('Invalid presence')
            if p.get('grade','') not in ['', 'correct','coarse','wrong','missed','uncertain','not_applicable']:
                raise ValueError('Invalid qualitative grade')
            if not isinstance(p.get('boxes', []), list) or not all(box_valid(b, True) for b in p.get('boxes', [])):
                raise ValueError('Invalid part boxes: '+cid+'/'+name)
            if p.get('boxes_complete') is True and (p.get('presence') != 'yes' or not p.get('boxes')):
                raise ValueError('Complete boxes require visible targets and nonempty boxes')
        if not (obj.get('reviewer','').strip() and r.get('confirmed') is True and r.get('reviewed_at')
                and r.get('target_status') and r.get('category')
                and all(r.get('labels',{}).get(k) for k in ATTRS)
                and all(r.get('regions',{}).get(k,{}).get('presence') for k in REGIONS)):
            pending.append(cid)
    return pending
