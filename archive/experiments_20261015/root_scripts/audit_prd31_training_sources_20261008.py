"""检查训练侧原始标注与八类清单；不改数据、不训练、不读取审核答案。"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import zipfile

# 按现有项目的八类映射检查源物体类别；部位标签如 sleeve 不作为衣物实例。
FP_MAP = {
 'shirt, blouse':'top', 'top, t-shirt, sweatshirt':'top', 'sweater':'top', 'vest':'top',
 'cardigan':'outerwear', 'jacket':'outerwear', 'coat':'outerwear', 'cape':'outerwear',
 'pants':'pants', 'shorts':'pants', 'skirt':'skirt', 'dress':'dress', 'shoe':'shoe',
 'bag, wallet':'bag', 'glasses':'accessory', 'hat':'accessory',
 'headband, head covering, hair accessory':'accessory', 'tie':'accessory',
 'glove':'accessory', 'watch':'accessory', 'belt':'accessory', 'leg warmer':'accessory',
 'tights, stockings':'accessory', 'sock':'accessory', 'scarf':'accessory',
}
DF_MAP = {1:'top',2:'top',3:'outerwear',4:'outerwear',5:'top',6:'top',7:'pants',
          8:'pants',9:'skirt',10:'dress',11:'dress',12:'dress',13:'dress'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-csv', default='configs/prd_8class_train_v5_balanced150_20261008.csv')
    args = parser.parse_args()
    from fashion_multimodal_analysis.common.paths import project_root, data_root, resolve_path
    root = project_root()
    with resolve_path(args.train_csv).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['source_dataset'], str(resolve_path(row['source_image'])))].append(row)
    fp_path = data_root() / 'raw/fashionpedia/annotations/instances_attributes_val2020.json'
    fp = json.loads(fp_path.read_text(encoding='utf-8'))
    categories = {int(c['id']): c['name'] for c in fp['categories']}
    fp_images = defaultdict(list)
    for image in fp['images']:
        fp_images[Path(image['file_name']).name].append(image)
    fp_annotations = defaultdict(list)
    for ann in fp['annotations']:
        fp_annotations[int(ann['image_id'])].append(ann)
    out = root / 'reports/reruns/recovery_20261008' / ('training_source_audit_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    audit = []; counts = Counter()
    for number, ((dataset, image_path), selected) in enumerate(sorted(grouped.items()), 1):
        issues = []; source_objects = []; row_lookup = {r['annotation_ref']: r for r in selected}
        if dataset == 'Fashionpedia':
            images = fp_images[Path(image_path).name]
            if len(images) != 1:
                issues.append({'reason':'source_image_resolution_failed','matching_images':len(images)})
            else:
                for ann in fp_annotations[int(images[0]['id'])]:
                    name = categories[int(ann['category_id'])]
                    if name not in FP_MAP and name != 'jumpsuit':
                        continue
                    source_objects.append({'annotation_ref':str(ann['id']), 'source_category':name,
                                           'mapped_category':FP_MAP.get(name), 'bbox_xywh':ann.get('bbox')})
                    if name == 'jumpsuit':
                        issues.append({'reason':'unmapped_foreground_jumpsuit', 'annotation_ref':str(ann['id'])})
                        continue
                    row = row_lookup.get(str(ann['id']))
                    if row is None:
                        issues.append({'reason':'supported_source_instance_missing_from_manifest',
                                       'annotation_ref':str(ann['id']), 'source_category':name,
                                       'has_segmentation':bool(ann.get('segmentation')), 'iscrowd':ann.get('iscrowd',0)})
                    elif row['garment_category'] != FP_MAP[name] or row['fine_category'] != name or str(ann['category_id']) != row['source_category_id']:
                        issues.append({'reason':'source_to_manifest_category_mismatch', 'record_id':row['record_id'],
                                       'source_category':name,'expected':FP_MAP[name],'actual':row['garment_category']})
                native_ids = {str(ann['id']) for ann in fp_annotations[int(images[0]['id'])]}
                for row in selected:
                    if row['annotation_ref'] not in native_ids:
                        issues.append({'reason':'manifest_annotation_id_not_in_source_image', 'record_id':row['record_id']})
        elif dataset == 'DeepFashion2':
            path = Path(image_path).parent.parent / 'annos' / (Path(image_path).stem + '.json')
            if not path.is_file():
                issues.append({'reason':'source_annotation_file_missing','path':str(path)})
            else:
                native = json.loads(path.read_text(encoding='utf-8'))
                native_ids = set()
                for key, ann in native.items():
                    if not key.startswith('item') or not isinstance(ann,dict):
                        continue
                    native_ids.add(key); cid = int(ann['category_id'])
                    expected = DF_MAP.get(cid)
                    source_objects.append({'annotation_ref':key,'source_category_id':cid,'mapped_category':expected,
                                           'bbox_xyxy':ann.get('bounding_box')})
                    row = row_lookup.get(key)
                    if expected is None:
                        issues.append({'reason':'unknown_deepfashion2_category','source_category_id':cid})
                    elif row is None:
                        issues.append({'reason':'supported_source_instance_missing_from_manifest',
                                       'annotation_ref':key,'source_category_id':cid,'has_segmentation':bool(ann.get('segmentation'))})
                    elif row['garment_category'] != expected or str(cid) != row['source_category_id']:
                        issues.append({'reason':'source_to_manifest_category_mismatch','record_id':row['record_id'],
                                       'expected':expected,'actual':row['garment_category']})
                for row in selected:
                    if row['annotation_ref'] not in native_ids:
                        issues.append({'reason':'manifest_annotation_id_not_in_source_image','record_id':row['record_id']})
        else:
            issues.append({'reason':'unknown_source_dataset','dataset':dataset})
        for issue in issues: counts[issue['reason']]+=1
        audit.append({'source_dataset':dataset,'source_image':image_path,'manifest_instances':len(selected),
                      'source_objects':source_objects,'issues':issues})
        if number % 100 == 0:print(f'AUDITED {number}/{len(grouped)}',flush=True)
    summary = {'status':'training_source_audit_completed','training_images':len(grouped),
               'training_rows':len(rows),'images_with_review_flags':sum(bool(r['issues']) for r in audit),
               'issue_counts':dict(counts),'labels_modified':False,'training_started':False,
               'note':'Missing-source/unsupported-object flags require review. They do not prove all source labels are correct. Only jumpsuit is screened as unsupported foreground; this is not an exhaustive open-set audit.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'training_source_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    # 保存完整清单便于后续追溯；不输出自动改标签或自动过滤的训练集。
    (out/'readme.md').write_text('训练侧源标注审核。八类映射已显式记录于脚本。未映射 jumpsuit、保留实例的类别错配、源标注中存在但清单缺失的八类实例分别列出；缺失可能受空分割/过滤规则影响，需复核后处理。未检查原图内容去重、mask 有效性、未知但未原标注的衣物。',encoding='utf-8')
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(out.glob('*')):zipped.write(path,path.relative_to(out))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('AUDIT_COMPLETED',flush=True)
    print('UPLOAD_THIS_ZIP:',archive,flush=True)


if __name__ == '__main__':
    main()
