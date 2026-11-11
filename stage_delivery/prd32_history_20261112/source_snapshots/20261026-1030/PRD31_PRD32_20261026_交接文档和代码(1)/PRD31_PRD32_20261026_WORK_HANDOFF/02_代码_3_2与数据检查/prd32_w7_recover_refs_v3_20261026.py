#!/usr/bin/env python3
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
PRIOR_SOURCE_FILE_SHA256='9bfe4e98e630c3abdee8b8c8cd83616783fbcab084458ca85dac59aa12d6972f'
REFERENCE=[{'image_id': 'dev_01', 'source_image': '/workspace/fashion_data/raw/train/train/image/105983.jpg', 'source_sha256': '961282b5e1100a4fe0620e8fec947626b2eb1882c14389ab3632d5ada9e1d865', 'target_category': 'outerwear', 'reference_bbox_xyxy': [158.0, 2.0, 565.0, 401.0], 'prior_review_complete': True}, {'image_id': 'dev_02', 'source_image': '/workspace/fashion_data/raw/validation/validation/image/008889.jpg', 'source_sha256': 'fde7bd42023d4ed1742cae3c9d735e9f90c0a68111e3ee522abd1757e93b4cdb', 'target_category': 'outerwear', 'reference_bbox_xyxy': [5.0, 16.0, 468.0, 797.0], 'prior_review_complete': True}, {'image_id': 'dev_03', 'source_image': '/workspace/fashion_data/raw/train/train/image/007644.jpg', 'source_sha256': '1d6b8a69c8f39c122b8deabf59d7b81d89140c865870f9780a3095151c9a5a02', 'target_category': 'top', 'reference_bbox_xyxy': [276.0, 423.0, 595.0, 705.0], 'prior_review_complete': False}, {'image_id': 'dev_04', 'source_image': '/workspace/fashion_data/raw/validation/validation/image/028673.jpg', 'source_sha256': '0fe745bf8f6f3ed23580e6b0bc88c961e5aab6dc90e55593d10eb6262b9e250d', 'target_category': 'top', 'reference_bbox_xyxy': [28.0, 161.0, 575.0, 820.0], 'prior_review_complete': True}, {'image_id': 'dev_05', 'source_image': '/workspace/fashion_data/raw/train/train/image/009252.jpg', 'source_sha256': 'b05624cb438de3d82ae8901ba39a9b1b95c02e109194abcd669e4d7e1e4c7042', 'target_category': 'top', 'reference_bbox_xyxy': [110.0, 2.0, 537.0, 463.0], 'prior_review_complete': True}, {'image_id': 'dev_06', 'source_image': '/workspace/fashion_data/raw/validation/validation/image/029438.jpg', 'source_sha256': 'dbf57c8e2adf52b551740635c06ef01e14c714464e39ea77491132777a52a798', 'target_category': 'top', 'reference_bbox_xyxy': [150.0, 5.0, 496.0, 499.0], 'prior_review_complete': True}, {'image_id': 'dev_07', 'source_image': '/workspace/fashion_data/raw/fashionpedia/images/test/759a7660c0f1268c90bad70cae7427ca.jpg', 'source_sha256': '5c5bd3897baee2c180c13e34d3ba4b4a38c95a710378376a8eee3c02b63e9fdc', 'target_category': 'top', 'reference_bbox_xyxy': [118.0, 380.0, 267.0, 514.0], 'prior_review_complete': True}, {'image_id': 'dev_08', 'source_image': '/workspace/fashion_data/raw/fashionpedia/images/test/cc066bcf334a4f0345764c594e85a929.jpg', 'source_sha256': '35b2352e3606fb0459ddc4bf32d179ccd0b71f5cc66dac3b98522f432421de24', 'target_category': 'top', 'reference_bbox_xyxy': [167.0, 230.0, 462.0, 676.0], 'prior_review_complete': True}, {'image_id': 'dev_09', 'source_image': '/workspace/fashion_data/raw/fashionpedia/images/test/9750434d2cd9ced43c09f15ae11ed6b2.jpg', 'source_sha256': 'd4529f8daa5a9e330267ec43c7a09c2a39e4fd38537e750b669abff2b0239a52', 'target_category': 'outerwear', 'reference_bbox_xyxy': [169.0, 281.0, 581.0, 856.0], 'prior_review_complete': True}, {'image_id': 'dev_10', 'source_image': '/workspace/fashion_data/raw/fashionpedia/images/test/975bc03b803a8187fe7b005966f7cdd9.jpg', 'source_sha256': 'c178846c2d7981dc7fb11c532740f934177bbf275f7e3b4bd5eecdfaef9932fd', 'target_category': 'dress', 'reference_bbox_xyxy': [54.0, 298.0, 682.0, 1024.0], 'prior_review_complete': True}, {'image_id': 'dev_11', 'source_image': '/workspace/fashion_data/raw/train/train/image/047522.jpg', 'source_sha256': '8ca4635a05d63cca63cdd97e68450b426fcfad867cca61d6555497df5762ff9e', 'target_category': 'top', 'reference_bbox_xyxy': [217.0, 270.0, 696.0, 583.0], 'prior_review_complete': True}, {'image_id': 'dev_12', 'source_image': '/workspace/fashion_data/raw/fashionpedia/images/test/09ff9e179fe9d63f48c1bf6d5565f659.jpg', 'source_sha256': 'ee374ecb334c2037c54a42e1e95fd83b69b794be4ead5b6d523e3aba01bcefd8', 'target_category': 'outerwear', 'reference_bbox_xyxy': [243.0, 213.0, 519.0, 584.0], 'prior_review_complete': True}]
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
    if len(bysel)!=12 or len(bycsv)!=12 or len(byref)!=12 or not (set(bysel)==set(bycsv)==set(byref)):
        ap.error('Image IDs duplicated/mismatched; stop')

    out=[]; views=[]
    for image_id in sorted(bysel):
        rec=bysel[image_id]; rev=bycsv[image_id]; prior=byref[image_id]
        source=Path(str(rec.get('source_image','')));safe(source)
        safe(rev.get('source_image',''))
        actual=sha256_file(source) if source.is_file() else None
        errors=[]
        if not actual:errors.append('SOURCE_NOT_FOUND')
        elif actual!=rec.get('source_sha256'):errors.append('9bfe4e98e630c3abdee8b8c8cd83616783fbcab084458ca85dac59aa12d6972f_MISMATCH')
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
