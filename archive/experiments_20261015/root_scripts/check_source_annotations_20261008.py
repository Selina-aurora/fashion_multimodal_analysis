"""导出 8 个开发案例的原始类别/框，核对 CSV 转换及人工意见冲突。

Read-only access to manifests and annotations. Does not edit labels or train.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='configs/prd_8class_val_v2_full_targets.csv')
    args = parser.parse_args()
    from fashion_multimodal_analysis.common.paths import project_root, data_root, resolve_path
    from PIL import Image, ImageDraw
    root = project_root()
    cases = json.loads((root / 'configs/prd31_review_regression_cases_20261008.json').read_text())
    with resolve_path(args.manifest).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    fp_path = data_root() / 'raw/fashionpedia/annotations/instances_attributes_val2020.json'
    if not fp_path.is_file():
        raise FileNotFoundError(fp_path)
    fp = json.loads(fp_path.read_text(encoding='utf-8'))
    categories = {int(c['id']): c['name'] for c in fp['categories']}
    images = {str(i['file_name']).split('/')[-1]: i for i in fp['images']}
    # 同一个图像名出现多个 image id 时停止，避免静默选错。
    if len(images) != len(fp['images']):
        raise ValueError('Fashionpedia has duplicate image basenames; cannot resolve safely')
    target_ids = {int(images[Path(c['source_image']).name]['id'])
                  for c in cases if 'fashionpedia' in c['source_image']}
    annotations = {iid: [] for iid in target_ids}
    for ann in fp['annotations']:
        iid = int(ann['image_id'])
        if iid in annotations:
            annotations[iid].append(ann)
    out = root / 'reports/reruns/recovery_20261008' / ('source_annotation_check_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for case in cases:
        cid = case['case_id']; image_path = resolve_path(case['source_image'])
        selected = [r for r in rows if resolve_path(r['source_image']) == image_path]
        if not selected:
            raise ValueError('No CSV annotations for ' + cid)
        folder = out / cid; folder.mkdir()
        source_objects = []
        if selected[0]['source_dataset'] == 'Fashionpedia':
            info = images[image_path.name]; raw_objects = annotations[int(info['id'])]
            lookup = {str(a['id']): a for a in raw_objects}
            for a in raw_objects:
                x, y, width, height = map(float, a['bbox'])
                source_objects.append({'annotation_ref': str(a['id']), 'category_id': a['category_id'],
                     'category_name': categories[int(a['category_id'])], 'bbox_xyxy': [x, y, x+width, y+height]})
            mismatches = []
            for row in selected:
                a = lookup.get(row['annotation_ref'])
                if a is None:
                    mismatches.append({'record_id': row['record_id'], 'reason': 'annotation_id_missing_from_raw_image'})
                    continue
                if str(a['category_id']) != row['source_category_id'] or categories[int(a['category_id'])] != row['fine_category']:
                    mismatches.append({'record_id': row['record_id'], 'reason': 'source_category_mismatch',
                                       'raw_category': categories[int(a['category_id'])], 'csv_category': row['fine_category']})
                source_box = next(o['bbox_xyxy'] for o in source_objects if o['annotation_ref'] == row['annotation_ref'])
                csv_box = [float(row['bbox_'+key]) for key in ('x1', 'y1', 'x2', 'y2')]
                if max(abs(x-y) for x,y in zip(source_box, csv_box)) > 2:
                    mismatches.append({'record_id': row['record_id'], 'reason': 'bbox_diff_over_2_pixels',
                                       'raw_bbox': source_box, 'csv_bbox': csv_box})
            source_file = str(fp_path)
        else:
            raw_path = image_path.parent.parent / 'annos' / (image_path.stem + '.json')
            if not raw_path.is_file():
                raise FileNotFoundError(raw_path)
            raw = json.loads(raw_path.read_text(encoding='utf-8'))
            for name, item in raw.items():
                if name.startswith('item') and isinstance(item, dict):
                    source_objects.append({'annotation_ref': name, 'category_id': item['category_id'],
                         'category_name': 'df2_category_' + str(item['category_id']), 'bbox_xyxy': item['bounding_box']})
            lookup = {o['annotation_ref']: o for o in source_objects}
            mismatches = []
            for row in selected:
                item = lookup.get(row['annotation_ref'])
                if item is None:
                    mismatches.append({'record_id': row['record_id'], 'reason': 'annotation_ref_missing'})
                elif str(item['category_id']) != row['source_category_id']:
                    mismatches.append({'record_id': row['record_id'], 'reason': 'source_category_mismatch'})
            source_file = str(raw_path)
        record = {'case_id': cid, 'source_image': str(image_path), 'source_annotation_file': source_file,
                  'csv_objects': selected, 'raw_source_objects': source_objects, 'conversion_mismatches': mismatches,
                  'review_status': 'requires_visual_review_not_frozen_gt'}
        records.append(record)
        # 各实例单独画框，避免多个重叠标签遮住细节。
        original = Image.open(image_path).convert('RGB')
        original.save(folder / 'source.jpg', quality=95)
        for index, obj in enumerate(source_objects):
            canvas = original.copy(); draw = ImageDraw.Draw(canvas)
            draw.rectangle(obj['bbox_xyxy'], outline='red', width=3)
            draw.text((5, 5), f"{obj['annotation_ref']} {obj['category_name']}", fill='red')
            canvas.save(folder / f'raw_object_{index:02d}.jpg', quality=95)
        print(cid, 'raw:', [(o['annotation_ref'], o['category_name']) for o in source_objects],
              'conversion_mismatches:', len(mismatches), flush=True)
    (out / 'source_annotation_comparison.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'summary.json').write_text(json.dumps({'status':'source_annotation_check_completed', 'images':len(records),
        'conversion_mismatch_count':sum(len(r['conversion_mismatches']) for r in records),
        'labels_modified':False, 'human_conflicts_resolved':False}, indent=2))
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(out.rglob('*')):
            if path.is_file(): zipped.write(path, path.relative_to(out))
    print('SOURCE_CHECK_COMPLETED', flush=True)
    print('UPLOAD_THIS_ZIP:', archive, flush=True)


if __name__ == '__main__':
    main()
