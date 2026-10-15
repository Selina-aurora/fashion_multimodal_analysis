"""从原 V5 Core400 回归导出优先错误案例，不训练、不推理、不改测试标签。

必须确认 summary 与逐目标表一致，避免历史目录中混入其他模型的报告。
固定测试样本已反复用于诊断，不作为独立盲测，也不直接加入补训。
"""
from pathlib import Path
from collections import Counter
from datetime import datetime
import argparse
import csv
import json
import shutil
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', default='reports/reruns/recovery_20261008/v5_balanced150_core400')
    args = parser.parse_args()
    root = Path.cwd()
    report = (root / args.report_dir).resolve()
    summary = (report / 'evaluation_summary.txt').read_text(encoding='utf-8-sig')
    values = dict(line.split('=', 1) for line in summary.splitlines() if '=' in line)
    if '/v5_balanced150/' not in values.get('checkpoint', '').replace('\\', '/'):
        raise SystemExit('此目录不是原 V5 balanced150。请提供正确 --report-dir，未导出。')
    if values.get('validation_images') != '400' or values.get('gt_instances') != '400':
        raise SystemExit('不是既有 Core400 回归报告，未导出。')
    if float(values.get('score_threshold', '-1')) != .4:
        raise SystemExit('报告分数阈值不是 0.40，未导出。')
    with (report / 'per_gt_results.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    counts = Counter(r['outcome'] for r in rows)
    if len(rows) != 400 or len({r['record_id'] for r in rows}) != 400:
        raise SystemExit('逐目标记录数量或唯一 ID 不符，未导出。')
    for outcome in ('correct', 'low_mask_iou', 'wrong_class', 'missed'):
        if counts[outcome] != int(values[outcome]):
            raise SystemExit(f'summary 与 per_gt 不一致：{outcome}，未导出。')
    correct_class = sum(r['class_correct'] == 'True' and r['localized_bbox50'] == 'True' for r in rows)
    if abs(correct_class / 400 - float(values['end_to_end_bbox50_class_recall'])) > .000051:
        raise SystemExit('类别召回与 summary 不一致，未导出。')

    # 每个主要混淆方向最多取 3 例；按现有 record_id 排序可复现。
    targets = [('skirt', 'pants'), ('outerwear', 'top'), ('dress', 'top'), ('top', 'dress')]
    selected, selected_ids = [], set()
    for gt, pred in targets:
        group = sorted((r for r in rows if r['outcome'] == 'wrong_class' and
                        r['gt_class'] == gt and r['pred_class'] == pred), key=lambda r: r['record_id'])[:3]
        for row in group:
            selected.append((f'confusion_{gt}_to_{pred}', row)); selected_ids.add(row['record_id'])
    for gt in ('accessory', 'bag'):
        group = sorted((r for r in rows if r['outcome'] == 'missed' and r['gt_class'] == gt),
                       key=lambda r: r['record_id'])[:6]
        for row in group:
            if row['record_id'] not in selected_ids:
                selected.append((f'missed_{gt}', row)); selected_ids.add(row['record_id'])
    if not selected:
        raise SystemExit('未找到优先错误，未创建导出目录。')

    def source_path(value):
        text = value.replace('\\', '/')
        if text.startswith('../fashion_data/'):
            return root.parent / 'fashion_data' / text[len('../fashion_data/'):]
        return Path(text) if Path(text).is_absolute() else root / text

    prepared = []
    for reason, row in selected:
        source = source_path(row['source_image'])
        # 当前指定目录的可视化是唯一来源，不复制历史绝对路径指向的其他目录。
        visual_name = Path(row.get('visualization_path', '')).name
        visual = report / 'visualizations' / visual_name if visual_name else None
        if not source.is_file():
            raise SystemExit(f'缺少原图：{row["record_id"]}，路径：{source}；未开始导出。')
        # 评估默认只导出前 32 张可视化；记录存在不代表有可视化文件。
        if visual is not None and not visual.is_file():
            visual = None
        prepared.append((reason, row, source, visual))
    output = root / 'reports/reruns/recovery_20261009' / ('priority_errors_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    exported = []
    for index, (reason, row, source, visual) in enumerate(prepared, 1):
        folder = output / f'case_{index:02d}'
        folder.mkdir()
        shutil.copy2(source, folder / ('source' + source.suffix))
        if visual is not None:
            shutil.copy2(visual, folder / ('evaluation' + visual.suffix))
        entry = dict(row, export_case=folder.name, selection_reason=reason,
                     source_copy=folder.name + '/source' + source.suffix,
                     evaluation_copy=(folder.name + '/evaluation' + visual.suffix) if visual else None,
                     evaluation_visualization_available=visual is not None,
                     visualization_note='Existing evaluation image only; absent images are not rerendered.')
        exported.append(entry)
        (folder / 'record.json').write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'cases.json').write_text(json.dumps(exported, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copy2(report / 'evaluation_summary.txt', output / 'source_evaluation_summary.txt')
    (output / 'README.txt').write_text('原 V5 优先错误审核。仅选择现有回归错误，未运行模型或修改标签。\n'
        '原始测试标签不等于完整图像标注；不得把额外可见衣物自动计为误检。\n'
        '审核类别/实例归属、遮挡、尺寸和预测分数，修复依据需来自训练侧或通用模型逻辑。\n'
        '这些测试图片不能直接加入训练，本次导出不表示准确率改善。\n', encoding='utf-8')
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(output.rglob('*')):
            if f.is_file(): z.write(f, f.relative_to(output))
    print('EXPORTED_CASES:', len(exported), dict(Counter(reason for reason, *_ in prepared)))
    print('WITHOUT_OLD_VISUALIZATION:', sum(not r['evaluation_visualization_available'] for r in exported))
    print('UPLOAD_THIS_ZIP:', archive)


if __name__ == '__main__':
    main()
