#!/usr/bin/env python3
"""Stage non-blind W7 dev12 human-reviewed target reference candidates for 3.2.
This does not create ground truth for colors, patterns, designs or score a model.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from collections import Counter
import hashlib, csv, json

SOURCE = """\
dev_01|train/train/image/105983.jpg|outerwear|158,2,565,401|961282b5e1100a4fe0620e8fec947626b2eb1882c14389ab3632d5ada9e1d865|yes
dev_02|validation/validation/image/008889.jpg|outerwear|5,16,468,797|fde7bd42023d4ed1742cae3c9d735e9f90c0a68111e3ee522abd1757e93b4cdb|yes
dev_03|train/train/image/007644.jpg|top|276,423,595,705|1d6b8a69c8f39c122b8deabf59d7b81d89140c865870f9780a3095151c9a5a02|no
dev_04|validation/validation/image/028673.jpg|top|28,161,575,820|0fe745bf8f6f3ed23580e6b0bc88c961e5aab6dc90e55593d10eb6262b9e250d|yes
dev_05|train/train/image/009252.jpg|top|110,2,537,463|b05624cb438de3d82ae8901ba39a9b1b95c02e109194abcd669e4d7e1e4c7042|yes
dev_06|validation/validation/image/029438.jpg|top|150,5,496,499|dbf57c8e2adf52b551740635c06ef01e14c714464e39ea77491132777a52a798|yes
dev_07|fashionpedia/images/test/759a7660c0f1268c90bad70cae7427ca.jpg|top|118,380,267,514|5c5bd3897baee2c180c13e34d3ba4b4a38c95a710378376a8eee3c02b63e9fdc|yes
dev_08|fashionpedia/images/test/cc066bcf334a4f0345764c594e85a929.jpg|top|167,230,462,676|35b2352e3606fb0459ddc4bf32d179ccd0b71f5cc66dac3b98522f432421de24|yes
dev_09|fashionpedia/images/test/9750434d2cd9ced43c09f15ae11ed6b2.jpg|outerwear|169,281,581,856|d4529f8daa5a9e330267ec43c7a09c2a39e4fd38537e750b669abff2b0239a52|yes
dev_10|fashionpedia/images/test/975bc03b803a8187fe7b005966f7cdd9.jpg|dress|54,298,682,1024|c178846c2d7981dc7fb11c532740f934177bbf275f7e3b4bd5eecdfaef9932fd|yes
dev_11|train/train/image/047522.jpg|top|217,270,696,583|8ca4635a05d63cca63cdd97e68450b426fcfad867cca61d6555497df5762ff9e|yes
dev_12|fashionpedia/images/test/09ff9e179fe9d63f48c1bf6d5565f659.jpg|outerwear|243,213,519,584|ee374ecb334c2037c54a42e1e95fd83b69b794be4ead5b6d523e3aba01bcefd8|yes
"""
ROOT = Path('/workspace/fashion_data/raw')
OUT = Path('/workspace/PRD32_DEV_W7_20261026')
OUT.mkdir(parents=True, exist_ok=True)
CSV = OUT / 'w7_dev12_reference_manifest.csv'
JPG = OUT / 'w7_dev12_reference_sheet.jpg'
REPORT = OUT / 'preflight_summary.json'
if any(p.exists() for p in (CSV, JPG, REPORT)):
    raise SystemExit('Existing outputs found, refusing to overwrite.')

rows = []
sheet = Image.new('RGB', (860, 340 * 12), 'white')
draw = ImageDraw.Draw(sheet)
for i, line in enumerate(SOURCE.strip().splitlines()):
    iid, rel, cls, box_csv, expected, reviewer_done = line.split('|')
    if any(word in rel.lower() for word in ('sealed', 'holdout')):
        raise ValueError('Sealed path prohibited')
    path = ROOT / rel
    box = [int(q) for q in box_csv.split(',')]
    status = 'MISSING'
    if path.is_file():
        h = hashlib.sha256()
        with path.open('rb') as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != expected:
            status = 'HASH_MISMATCH'
        else:
            with Image.open(path) as im:
                image = im.convert('RGB')
            w, hh = image.size
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= hh):
                status = 'BBOX_OUT_OF_RANGE'
            else:
                status = 'IMAGE_VERIFIED'
                marked = image.copy()
                ImageDraw.Draw(marked).rectangle(box, outline='red', width=5)
                crop = image.crop(tuple(box))
                marked = ImageOps.contain(marked, (390, 292))
                crop = ImageOps.contain(crop, (390, 292))
                sheet.paste(marked, (15, i * 340 + 35))
                sheet.paste(crop, (440, i * 340 + 35))
    draw.text((15, i * 340 + 8), f'{iid} | {cls} | {reviewer_done} | {status}', fill='black')
    rows.append(dict(image_id=iid,image_path=str(path),source_sha256=expected,
                     human_reviewed_target_category=cls,reference_bbox_xyxy=json.dumps(box),
                     w7_original_target_review_complete=reviewer_done,
                     image_asset_check=status,
                     target_reference_origin='W7 human-assisted, NOT 3.1 frozen prediction',
                     gt_primary_color='',gt_pattern='',gt_visible_design='',
                     gt_3_2_review_status='PENDING',gt_scoring_eligible='FALSE'))

with CSV.open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
sheet.save(JPG, quality=90)
summary = {'rows':len(rows), 'categories':dict(Counter(r['human_reviewed_target_category'] for r in rows)),
 'asset_check': dict(Counter(r['image_asset_check'] for r in rows)),
 'w7_target_review_complete':dict(Counter(r['w7_original_target_review_complete'] for r in rows)),
 'no_new_3_2_gt':True,'no_model_inference':True,'sealed_holdout_used':False,
 'output_csv':str(CSV),'output_sheet':str(JPG)}
REPORT.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
