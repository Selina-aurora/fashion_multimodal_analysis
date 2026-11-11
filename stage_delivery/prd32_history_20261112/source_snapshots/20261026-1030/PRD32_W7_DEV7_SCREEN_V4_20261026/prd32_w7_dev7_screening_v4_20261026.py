#!/usr/bin/env python3
"""Create a conservative 7-case PRD3.2 development shortlist from W7 V3.

Only screens prior reference candidates. Does NOT certify bounding boxes,
produce verified attribute GT, access sealed holdout or invoke a model.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

DEFAULT_SOURCE = Path('/workspace/PRD32_W7_REF_RECOVERY_V3_20261026')
DEFAULT_OUTPUT = Path('/workspace/PRD32_W7_DEV7_SCREEN_V4_20261026')

# Decisions derive from human visual precheck of the W7 reference contact sheet.
VISUAL_REVIEW = {
    'dev_01': ('TARGET_MIXED_LAYER_REVIEW', '外层深色服饰与印花内搭重叠；需要确认外套实例归属、参考框内容'),
    'dev_02': ('DEV_CANDIDATE', '粉红色外套；框大体覆盖目标，人工框边界尚未正式核验'),
    'dev_03': ('PRIOR_REVIEW_PENDING', '深蓝色上衣可见；原始W7审核未完成，暂不纳入本轮'),
    'dev_04': ('DEV_CANDIDATE', '深色格纹上衣；框大体覆盖目标，顶部和手臂遮挡需留意'),
    'dev_05': ('DEV_CANDIDATE', '蓝白条纹上衣；框大体覆盖目标，存在手臂遮挡'),
    'dev_06': ('DEV_CANDIDATE', '白色短袖上衣；背包肩带与目标重叠'),
    'dev_07': ('TARGET_BOUNDARY_REVIEW', '浅褐色上衣；裁剪包含局部衣料及身体，需确认边界和目标归属'),
    'dev_08': ('DEV_CANDIDATE', '浅粉色无袖上衣；框大体覆盖目标，避免混入下装'),
    'dev_09': ('DEV_CANDIDATE', '深浅混合纹理外套；框大体覆盖目标'),
    'dev_10': ('QUARANTINED_CATEGORY', '人工审核CSV与恢复参考记录类别不一致；不得自动解除隔离'),
    'dev_11': ('QUARANTINED_IDENTITY', '路径与SHA256冲突；不得自动解除隔离'),
    'dev_12': ('DEV_CANDIDATE', '黑色西装外套；框大体覆盖目标，内搭混入需注意'),
}
SAFE_IDS = ('dev_02', 'dev_04', 'dev_05', 'dev_06', 'dev_08', 'dev_09', 'dev_12')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE)
    ap.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    source = args.source_dir.resolve(strict=True)
    output = args.output_dir
    if output.exists():
        ap.error(f'Output already exists; will not overwrite: {output}')
    if any(x in str(source).lower() for x in ('sealed', 'holdout')):
        ap.error('Sealed/holdout source is prohibited')
    summary_path = source / 'summary.json'
    manifest_path = source / 'recovered_reference_candidates.csv'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('selected_manifest_sha256') != '15d2c9c699279a85dc840a35dd32ec2f36fead4968ed6a7c5f40841a4d566965':
        ap.error('W7 selected manifest identity is not the previously reviewed one')
    if summary.get('source_images') != 12 or summary.get('sealed_holdout_accessed') is not False:
        ap.error('Unexpected source count or holdout state')
    expected_quarantine = {'dev_10','dev_11'}
    if set(summary.get('quarantined', {})) != expected_quarantine:
        ap.error('Quarantine state changed; re-review before continuing')
    with manifest_path.open(encoding='utf-8-sig',newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 'image_id' not in reader.fieldnames:
            ap.error('Recovered manifest must have image_id field')
        source_columns = list(reader.fieldnames)
        data = list(reader)
    ids = [r['image_id'] for r in data]
    if len(data) != 12 or len(set(ids)) != 12 or set(ids) != set(VISUAL_REVIEW):
        ap.error('Recovered manifest image IDs do not match W7 Dev12')
    cols = source_columns + [x for x in ('visual_precheck_status','visual_precheck_note','prd32_dev7_selected','attribute_gt_status','bbox_gt_status','selection_usage') if x not in source_columns]
    prepared = []
    for row in data:
        case_id = row['image_id']
        status, note = VISUAL_REVIEW[case_id]
        row = dict(row)
        row['visual_precheck_status'] = status
        row['visual_precheck_note'] = note
        row['prd32_dev7_selected'] = 'TRUE' if case_id in SAFE_IDS else 'FALSE'
        row['attribute_gt_status'] = 'NOT_VERIFIED'
        row['bbox_gt_status'] = 'REFERENCE_CANDIDATE_NOT_GT' if case_id in SAFE_IDS else 'NOT_AVAILABLE_FOR_SCORING'
        row['selection_usage'] = 'DEV_DIAGNOSTIC_ONLY_NOT_ACCURACY_SCORING' if case_id in SAFE_IDS else 'REQUIRES_SEPARATE_REVIEW'
        prepared.append(row)
    selected = [r for r in prepared if r['image_id'] in SAFE_IDS]
    if len(selected) != 7:
        ap.error('Unexpected dev shortlist length')
    for row in selected:
        if (row.get('status') != 'REFERENCE_CANDIDATE_VISUAL_REVIEW_REQUIRED'
                or row.get('geometry_valid', '').strip().lower() != 'true'
                or row.get('identity_errors', '').strip()
                or not row.get('reference_bbox_xyxy', '').strip()):
            ap.error('Cannot shortlist an invalid/unverified reference candidate: ' + row['image_id'])
        try:
            box = json.loads(row['reference_bbox_xyxy'])
            if not isinstance(box, list) or len(box) != 4 or not all(isinstance(x, (int,float)) for x in box):
                raise ValueError('bad box')
        except (ValueError, TypeError) as e:
            ap.error('Invalid reference bbox for ' + row['image_id'] + ': ' + str(e))
    # Avoid writing any output until all checks have succeeded.
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('visual_screening_all12.csv',prepared), ('dev7_candidate_manifest.csv',selected)]:
        with (output / name).open('w',encoding='utf-8-sig',newline='') as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            writer.writeheader(); writer.writerows(rows)
    report = {
        'schema': 'prd32_w7_visual_precheck_v4',
        'status': 'DEV7_CANDIDATES_PREPARED_NOT_ACCEPTED',
        'source_manifest_sha256': summary['selected_manifest_sha256'],
        'n_source': len(prepared), 'n_selected': len(selected),
        'selected_ids': list(SAFE_IDS),
        'statuses': dict(Counter(r['visual_precheck_status'] for r in prepared)),
        'quarantined_unchanged': ['dev_10','dev_11'],
        'human_bbox_gt_confirmed': 0,
        'human_attribute_gt_confirmed': 0,
        'accuracy_scoring_allowed': False,
        'model_inference_performed': False,
        'frozen_prd31_modified': False,
        'sealed_holdout_accessed': False,
        'warning': 'Visual precheck of contact sheet; selected candidate boxes are NOT ground truth or automatically validated labels.',
        'output_dir': str(output),
    }
    (output / 'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('SCREENING_V4_READY')

if __name__ == '__main__':
    main()
