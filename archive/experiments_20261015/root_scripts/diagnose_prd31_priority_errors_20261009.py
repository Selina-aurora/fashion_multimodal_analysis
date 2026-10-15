"""对原 V5 的 24 个优先错误查看低分候选，导出 GT/预测框和匹配诊断。

不训练，不改 0.40 正式阈值，不把测试图用于训练。0.05 只为观察已有
ROI head/NMS/最多检测数量限制后的候选，不能证明模型内部没有其他候选。
诊断使用 GT 比较模型输出，GT 框不进入模型，不产生产品侧类别覆盖。
"""
from pathlib import Path
from datetime import datetime
from collections import Counter
import argparse
import csv
import hashlib
import json
import zipfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def iou(a, b):
    w = max(0., min(a[2], b[2])-max(a[0], b[0]))
    h = max(0., min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0., a[2]-a[0])*max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0])*max(0., b[3]-b[1])
    return w*h/max(1e-12, area_a+area_b-w*h)


def candidate_summary(candidates, gt_class, old_outcome):
    # 此清单每张图有一个被评估目标；按正式协议的置信度顺序取首个 IoU>=.5。
    high = [c for c in candidates if c['score'] >= .4 and c['bbox_iou'] >= .5]
    localized = [c for c in candidates if c['bbox_iou'] >= .5]
    correct = [c for c in localized if c['category'] == gt_class]
    if old_outcome != 'missed':
        cause = 'localized_category_confusion'
    elif high:
        cause = 'baseline_match_drift_requires_review'
    elif correct:
        cause = 'correct_class_candidate_below_0_40'
    elif localized:
        cause = 'localized_low_score_candidates_wrong_class'
    else:
        cause = 'no_box50_candidate_above_0_05'
    return {'diagnostic_group': cause,
            'baseline_0_40_match': high[0] if high else None,
            'best_score_localized_candidate': localized[0] if localized else None,
            'best_score_correct_class_candidate': correct[0] if correct else None,
            'max_bbox_iou': max((c['bbox_iou'] for c in candidates), default=0.)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-dir')
    args = parser.parse_args()
    root = Path.cwd()
    from fashion_multimodal_analysis.common.paths import resolve_path
    from fashion_multimodal_analysis.common.schema import ID_TO_CLASS
    from fashion_multimodal_analysis.segmentation.training.checkpoint_state import validate_checkpoint, CONTROLLED_SETTINGS
    manifest = resolve_path('benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv')
    checkpoint = resolve_path('outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth')
    if args.cases_dir:
        cases_dir = resolve_path(args.cases_dir)
    else:
        options = sorted((root/'reports/reruns/recovery_20261009').glob('priority_errors_*'), reverse=True)
        options = [x for x in options if x.is_dir() and (x/'cases.json').is_file()]
        if not options: raise SystemExit('未找到刚导出的 priority_errors 目录。')
        cases_dir = options[0]
    cases = json.loads((cases_dir/'cases.json').read_text(encoding='utf-8'))
    if not cases or len(cases)>24 or len({r['record_id'] for r in cases})!=len(cases):
        raise SystemExit('错误案例数量或唯一性不符。')
    expected_checkpoint = 'c5ba795225349b1c521aac0517ac7c493d9818dcc17d558ac4a3f6f8669452f0'
    if sha(checkpoint)!=expected_checkpoint:
        raise SystemExit('权重不是昨日原 V5，未加载模型。')
    with manifest.open(encoding='utf-8-sig', newline='') as f: records=list(csv.DictReader(f))
    by_id = {r.get('record_id') or r.get('final_sample_id'):r for r in records}
    if len(records)!=400 or len(by_id)!=400: raise SystemExit('Core400 manifest 数量不符。')
    prepared=[]
    for case in cases:
        record=by_id[case['record_id']]
        if record['source_image'].replace('\\','/')!=case['source_image'].replace('\\','/'):
            raise SystemExit('案例与 manifest 原图不一致。')
        keys=('gt_bbox_x1','gt_bbox_y1','gt_bbox_x2','gt_bbox_y2') if 'gt_bbox_x1' in record else ('bbox_x1','bbox_y1','bbox_x2','bbox_y2')
        gt_box=[int(float(record[k])) for k in keys]
        gt_class=record.get('gt_class') or record.get('garment_category')
        if gt_class!=case['gt_class']: raise SystemExit('案例与 manifest GT 类别不一致。')
        source=resolve_path(record['source_image'])
        if not source.is_file(): raise SystemExit(f'缺少原图 {source}')
        from PIL import Image
        with Image.open(source) as raw: iw, ih = raw.size
        # 与正式 ValDataset 相同的越界裁剪，避免诊断引入不同 GT 框。
        x1,y1,x2,y2=gt_box
        x1=max(0,min(iw-1,x1)); y1=max(0,min(ih-1,y1))
        x2=max(x1+1,min(iw,x2)); y2=max(y1+1,min(ih,y2))
        gt_box=[x1,y1,x2,y2]
        copy=cases_dir/case['source_copy']
        if sha(source)!=sha(copy): raise SystemExit('原图与导出图片内容不同。')
        prepared.append((case, record, gt_box, source))

    import torch
    from PIL import Image, ImageDraw
    from torchvision.transforms.functional import to_tensor
    from fashion_multimodal_analysis.segmentation.modeling import build_evaluation_model
    if not torch.cuda.is_available(): raise SystemExit('需要昨日 GPU 环境。')
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    train=resolve_path('configs/prd_8class_train_v5_balanced150_20261008.csv')
    val=resolve_path('configs/prd_8class_val_v2_full_targets.csv')
    validate_checkpoint(cp,sha(train),sha(val),settings={**CONTROLLED_SETTINGS,'workers':2})
    saved=cp.get('args',{})
    model=build_evaluation_model(int(saved.get('min_size',640)),int(saved.get('max_size',1024)))
    model.load_state_dict(cp['model_state_dict']);model.to('cuda').eval()
    # 保持原来的 NMS、输出数量上限及内部 score floor，读取低分结果再诊断。
    internal_floor=float(model.roi_heads.score_thresh)
    if abs(internal_floor-.05)>1e-8: raise SystemExit(f'内部阈值不是预期0.05：{internal_floor}')
    del cp
    output=root/'reports/reruns/recovery_20261009'/('priority_candidate_diagnostic_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=False)
    results=[]
    with torch.inference_mode():
        for index,(case,record,gt_box,source) in enumerate(prepared,1):
            with Image.open(source) as raw: image=raw.convert('RGB')
            pred=model([to_tensor(image).to('cuda')])[0]
            boxes=pred['boxes'].detach().cpu().tolist()
            labels=pred['labels'].detach().cpu().tolist()
            scores=pred['scores'].detach().cpu().tolist()
            candidates=[{'category':ID_TO_CLASS.get(int(label),str(label)),
                         'score':float(score),'bbox':box,'bbox_iou':iou(gt_box,box)}
                        for box,label,score in zip(boxes,labels,scores) if score>=.05]
            candidates.sort(key=lambda r:-r['score'])
            diag=candidate_summary(candidates,case['gt_class'],case['outcome'])
            match=diag['baseline_0_40_match']
            drift=(match is not None) if case['outcome']=='missed' else (
                match is None or match['category']!=case['pred_class'] or
                abs(match['score']-float(case['pred_score']))>.001 or
                abs(match['bbox_iou']-float(case['bbox_iou']))>.001)
            entry=dict(case,gt_bbox=gt_box,source_sha256=sha(source),
                       image_size=[image.width,image.height],
                       gt_area_ratio=((gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1]))/(image.width*image.height),
                       baseline_reproduction_drift=drift,diagnostic=diag,candidates=candidates)
            folder=output/case['export_case'];folder.mkdir()
            image.save(folder/'source.jpg',quality=95)
            # 蓝框是审核用 GT；橙框是与该 GT IoU>=.1 的模型候选，不输入模型。
            canvas=image.copy();draw=ImageDraw.Draw(canvas)
            draw.rectangle(gt_box,outline='cyan',width=3)
            draw.text((gt_box[0],max(0,gt_box[1]-12)),'GT '+case['gt_class'],fill='cyan')
            for c in [r for r in candidates if r['bbox_iou']>=.1][:12]:
                draw.rectangle(c['bbox'],outline='orange',width=2)
                draw.text((c['bbox'][0],c['bbox'][1]),f"{c['category']} {c['score']:.2f} IoU {c['bbox_iou']:.2f}",fill='orange')
            canvas.save(folder/'gt_and_candidates.jpg',quality=95)
            (folder/'diagnostic.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2),encoding='utf-8')
            results.append(entry)
            print(f'{index}/{len(prepared)} {case["record_id"]}: {diag["diagnostic_group"]}; baseline_drift={drift}',flush=True)
            del pred
    summary={'cases':len(results),'checkpoint_sha256':sha(checkpoint),'manifest_sha256':sha(manifest),
             'diagnostic_group_counts':dict(Counter(r['diagnostic']['diagnostic_group'] for r in results)),
             'baseline_reproduction_drift_count':sum(r['baseline_reproduction_drift'] for r in results),
             'internal_model_score_floor':internal_floor,'formal_score_threshold_unchanged':.4,
             'accuracy_evaluated':False,'prd_acceptance_evaluated':False,
             'note':'Selected failures are not representative accuracy; candidate absence applies only to post-NMS, capped model outputs. GT used for analysis only.'}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'cases.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(output.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(output))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('UPLOAD_THIS_ZIP:',archive)


if __name__=='__main__':main()
