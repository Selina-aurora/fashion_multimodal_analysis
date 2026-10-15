"""单变量推理分辨率对照：原 V5 权重，固定 val32，min800/max1333。

仅在此进程包装现有评估模型工厂，不编辑项目源文件或 checkpoint。
score=.40/mask=.50/bboxIoU=.50/warmup10/timed5 均保持原评估设置。
此验证集仅有32图/88目标，结果不能作为全量 PRD 验收。
"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import sys
import zipfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()


def main():
    from fashion_multimodal_analysis.common.paths import project_root,resolve_path
    root=project_root()
    checkpoint=resolve_path('outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth')
    val=resolve_path('configs/prd_8class_val_v2_full_targets.csv')
    if sha(checkpoint)!='c5ba795225349b1c521aac0517ac7c493d9818dcc17d558ac4a3f6f8669452f0':
        raise SystemExit('不是昨日原 V5 权重，未运行评估。')
    if sha(val)!='39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93':
        raise SystemExit('验证清单已改变，未运行评估。')
    report=root/'reports/reruns/recovery_20261009'/('v5_resolution800_val32_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    report.mkdir(parents=True,exist_ok=False)
    metadata={'experiment':'inference_resolution_only','checkpoint':str(checkpoint),
              'checkpoint_sha256':sha(checkpoint),'val_csv':str(val),'val_sha256':sha(val),
              'baseline_inference_min_size':640,'baseline_inference_max_size':1024,
              'candidate_inference_min_size':800,'candidate_inference_max_size':1333,
              'checkpoint_training_resolution_unchanged':True,'weights_modified':False,
              'score_threshold':.4,'mask_threshold':.5,'match_bbox_iou':.5,
              'warmup_runs':10,'timed_passes':5,'accuracy_acceptance_claimed':False}
    (report/'resolution_experiment.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    from fashion_multimodal_analysis.segmentation.evaluation import eval_prd_8class_maskrcnn_baseline_v2 as evaluation
    original_factory=evaluation.build_model

    def candidate_factory(saved_min_size,saved_max_size):
        # 只覆盖本次推理 transform；checkpoint 的训练元数据保持原值。
        if (saved_min_size,saved_max_size)!=(640,1024):
            raise ValueError('Expected original V5 checkpoint min640/max1024')
        model=original_factory(800,1333)
        if tuple(model.transform.min_size)!=(800,) or model.transform.max_size!=1333:
            raise ValueError('Candidate resolution not applied')
        print('CANDIDATE_INFERENCE_RESOLUTION: min_size=800 max_size=1333',flush=True)
        return model

    evaluation.build_model=candidate_factory
    argv=sys.argv[:]
    sys.argv=[str(Path(__file__)), '--checkpoint',str(checkpoint),'--val-csv',str(val),
              '--score-threshold','0.40','--mask-threshold','0.50','--match-bbox-iou','0.50',
              '--warmup-runs','10','--timed-passes','5','--report-dir',str(report)]
    try:
        evaluation.main()
    finally:
        evaluation.build_model=original_factory
        sys.argv=argv
    summary=report/'evaluation_summary.txt'
    with summary.open('a',encoding='utf-8') as f:
        f.write('\nCandidate inference override (weights and checkpoint training metadata unchanged):\n'
                'actual_inference_min_size=800\nactual_inference_max_size=1333\n'
                'experiment_type=inference_resolution_only\n')
    archive=report.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(report.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(report))
    print('UPLOAD_THIS_ZIP:',archive,flush=True)


if __name__=='__main__':main()
