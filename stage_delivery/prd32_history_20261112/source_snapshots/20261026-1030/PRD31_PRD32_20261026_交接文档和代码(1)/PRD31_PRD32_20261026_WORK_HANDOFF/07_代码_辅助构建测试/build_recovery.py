import json, pathlib, textwrap, re
source=pathlib.Path('/mnt/data/source/W7_dev12_GT_HUMAN_REVIEW.json')
doc=json.loads(source.read_text())
import hashlib
assert doc['manifest_sha256']=='15d2c9c699279a85dc840a35dd32ec2f36fead4968ed6a7c5f40841a4d566965'
assert len(doc['records'])==12
reference=[{
    'image_id': r['image_id'],
    'source_image':r['source_image'],
    'source_sha256':r['source_sha256'],
    'target_category':r['target_category'],
    'reference_bbox_xyxy':[float(p.strip()) for p in r['target_bbox_xyxy'].split(',')],
    'prior_review_complete':r['gt_review_complete']=='yes',
} for r in doc['records']]
ref_serial=json.dumps(reference, ensure_ascii=False, separators=(',',':'))
program=r'''#!/usr/bin/env python3
"""PRD3.2 W7 Dev12: recover PRIOR reviewed bbox *candidates*, not certified GT.

The embedded prior-review records came from W7_dev12_GT_HUMAN_REVIEW.json.
No inference, no writes to original W7 assets, no sealed inputs. Changes
are only written to a new review workspace. Categories remain unverified
for the 3.2 scoring protocol; boxes require visual review.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw, ImageOps

EXPECTED_SELECTED_SHA256='15d2c9c699279a85dc840a35dd32ec2f36fead4968ed6a7c5f40841a4d566965'
PRIOR_SOURCE_BASENAME='W7_dev12_GT_HUMAN_REVIEW.json'
PRIOR_SOURCE_FILE_SHA256='SOURCE_SHA'
REFERENCE=REF_LITERAL
BAD_PATH_TOKENS=('holdout','sealed')

def safe(path):
    if any(v in str(path).lower() for v in BAD_PATH_TOKENS):
        raise ValueError('Sealed/holdout path forbidden: '+str(path))

def sha256_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def load_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selected',type=Path,default=Path('/workspace/PRD31_W7_ASSIST/inputs/W7_dev12_SELECTED.json'))
    ap.add_argument('--review',type=Path,default=Path('/workspace/PRD31_W7_M2/inputs/W7_dev12_human_screen_2026-10-15.csv'))
    ap.add_argument('--output',type=Path,default=Path('/workspace/PRD32_W7_REF_RECOVERY_V3_20261026'))
    a=ap.parse_args()
    for p in (a.selected,a.review,a.output): safe(p)
    if a.output.exists():ap.error('Output already exists, will not overwrite: '+str(a.output))
    if not a.selected.is_file() or not a.review.is_file():ap.error('Selected JSON / review CSV not found')
    manifest_sha=sha256_file(a.selected)
    if manifest_sha!=EXPECTED_SELECTED_SHA256:
        ap.error('Selected manifest identity changed; stop, do not apply prior boxes')
    selected=json.loads(a.selected.read_text(encoding='utf-8'))
    csv_rows=load_csv(a.review)
    if len(selected)!=12 or len(csv_rows)!=12:ap.error('Expected exactly 12 selected and 12 CSV records')
    bysel={r['image_id']:r for r in selected}
    bycsv={r['image_id']:r for r in csv_rows}
    byref={r['image_id']:r for r in REFERENCE}
    if len(bysel)!=12 or len(bycsv)!=12 or len(byref)!=12 or set(bysel)!=set(bycsv)!=set(byref):
        ap.error('Image IDs duplicated/mismatched; stop')

    out=[]; views=[]
    for image_id in sorted(bysel):
        rec=bysel[image_id]; rev=bycsv[image_id]; prior=byref[image_id]
        source=Path(str(rec.get('source_image','')));safe(source)
        safe(rev.get('source_image',''))
        actual=sha256_file(source) if source.is_file() else None
        errors=[]
        if not actual:errors.append('SOURCE_NOT_FOUND')
        elif actual!=rec.get('source_sha256'):errors.append('SOURCE_SHA_MISMATCH')
        if prior['source_sha256']!=rec.get('source_sha256') or prior['source_image']!=str(source):
            errors.append('PRIOR_REFERENCE_IDENTITY_MISMATCH')
        if rev.get('source_sha256')!=rec.get('source_sha256'):
            errors.append('REVIEW_CSV_SHA_MISMATCH')
        if rev.get('source_image')!=str(source):errors.append('REVIEW_CSV_PATH_MISMATCH')
        if (rev.get('target_category') or '')!=prior.get('target_category'):
            errors.append('REVIEW_CSV_CATEGORY_MISMATCH')
        picture=None; valid_box=False
        box=prior['reference_bbox_xyxy']
        if actual==rec.get('source_sha256'):
            with Image.open(source) as im:picture=im.convert('RGB')
            w,h=picture.size
            if int(rec.get('width',w))!=w or int(rec.get('height',h))!=h:
                errors.append('IMAGE_DIMENSIONS_MISMATCH')
            if len(box)==4 and all(math.isfinite(v) for v in box):
                valid_box=0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h
            if not valid_box:errors.append('REFERENCE_BOX_INVALID')
        if image_id=='dev_11':
            # The current W7 review CSV has demonstrably conflicting path/hash for dev_11.
            # Do not unblock automatically even if a later CSV is silently changed.
            errors.append('DEV11_REQUIRES_EXPLICIT_IDENTITY_REVIEW')
        if errors:
            status='QUARANTINED'
        elif not prior['prior_review_complete']:
            status='REFERENCE_PENDING_PRIOR_REVIEW'
        else:
            status='REFERENCE_CANDIDATE_VISUAL_REVIEW_REQUIRED'
        box_out=box if not errors else None
        out.append({
            'image_id':image_id,'image_path':str(source),
            'image_sha256':rec.get('source_sha256'),
            'reference_category_prior_review_not_scoring_gt':prior['target_category'],
            'reference_bbox_xyxy':json.dumps(box_out,ensure_ascii=False),
            'provenance':PRIOR_SOURCE_BASENAME,
            'prior_review_complete':str(prior['prior_review_complete']),
            'reference_bbox_provenance':'historical_human_review_candidate_not_segmentation_gt',
            'identity_errors':';'.join(sorted(set(errors))),
            'status':status,
            'geometry_valid':str(valid_box and not errors),
            'bbox_visually_confirmed_for_prd32':'FALSE',
            'attribute_gt_verified':'FALSE',
            'eligible_for_accuracy_scoring':'FALSE',
            'human_color_gt':'','human_pattern_gt':'','human_visible_design_gt':'',
            'notes':'',
        })
        if picture is None: views.append((image_id,status,None,None))
        else:
            marked=picture.copy()
            roi=None
            if box_out is not None:
                ImageDraw.Draw(marked).rectangle(box,outline='red',width=max(3,min(picture.size)//160))
                cropbox=(int(box[0]),int(box[1]),min(picture.width,math.ceil(box[2])),min(picture.height,math.ceil(box[3])))
                roi=picture.crop(cropbox)
            views.append((image_id,status,marked,roi))

    a.output.mkdir(parents=True,exist_ok=False)
    manifest=a.output/'recovered_reference_candidates.csv'
    with manifest.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(out[0]));writer.writeheader();writer.writerows(out)
    canvas=Image.new('RGB',(1200,6*330),'white');draw=ImageDraw.Draw(canvas)
    for i,(id_,status,marked,roi) in enumerate(views):
        x,y=(i%2)*600,(i//2)*330
        draw.text((x+8,y+10),id_+' '+status,fill='black')
        if marked is not None:
            canvas.paste(ImageOps.contain(marked,(280,280)),(x+10,y+35))
        else:draw.text((x+15,y+120),'SOURCE NOT VERIFIED',fill='red')
        if roi is not None:
            canvas.paste(ImageOps.contain(roi,(280,280)),(x+310,y+35))
        else:draw.text((x+315,y+120),'REFERENCE UNAVAILABLE',fill='red')
    sheet=a.output/'reference_candidates_contact_sheet.jpg';canvas.save(sheet,quality=90)
    summary={
        'status':'PRD32_REFERENCE_RECOVERY_DEV_NOT_ACCEPTED',
        'selected_manifest_sha256':manifest_sha,
        'prior_reference_source':PRIOR_SOURCE_BASENAME,
        'prior_reference_raw_sha256':PRIOR_SOURCE_FILE_SHA256,
        'source_images':len(selected),
        'statuses':dict(Counter(r['status'] for r in out)),
        'quarantined':{r['image_id']:r['identity_errors'] for r in out if r['identity_errors']},
        'reference_bbox_visual_confirmation_count':0,
        'human_attribute_gt_verified_count':0,
        'scoring_permitted':False,
        'frozen_3_1_modified':False,
        'sealed_holdout_accessed':False,
        'review_sheet':str(manifest),
        'contact_sheet':str(sheet),
    }
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
'''
raw_sha=hashlib.sha256(source.read_bytes()).hexdigest()
program=program.replace('SOURCE_SHA',raw_sha).replace('REF_LITERAL',repr(reference))
out=pathlib.Path('/mnt/data/prd32_w7_recover_refs_v3_20261026.py')
out.write_text(program,encoding='utf8')
import py_compile
py_compile.compile(str(out),doraise=True)
print(out,len(program),raw_sha)
