"""Revalidate the existing inference-resolution candidate on frozen Core400.
Two sequential evaluations, same checkpoint, labels and thresholds; no training.
Core400 has prior diagnostic exposure and is a regression set, not a new blind test.
"""
import csv,gc,hashlib,json,sys,time,zipfile
from pathlib import Path

CKPT_SHA='c5ba795225349b1c521aac0517ac7c493d9818dcc17d558ac4a3f6f8669452f0'
CORE_SHA='c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def values(path):
 return dict(line.split('=',1) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if '=' in line)

def main():
 from fashion_multimodal_analysis.common.paths import project_root,resolve_path
 root=project_root();checkpoint=resolve_path('outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth')
 core=resolve_path('benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv')
 if sha(checkpoint)!=CKPT_SHA or sha(core)!=CORE_SHA:raise ValueError('Frozen V5 checkpoint or Core400 manifest changed')
 with core.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 if len(rows)!=400 or len({r['source_image'] for r in rows})!=400:raise ValueError('Expected frozen 400 images / 400 GT')
 if len({(r.get('record_id') or r.get('final_sample_id', '')) for r in rows})!=400:raise ValueError('Duplicate Core record IDs')
 out=root/'reports/reruns/recovery_20261009'/('core400_resolution_pair_'+time.strftime('%Y%m%d_%H%M%S')+f'_{time.time_ns()%1000000:06d}')
 out.mkdir(parents=True)
 from fashion_multimodal_analysis.segmentation.evaluation import eval_prd_8class_maskrcnn_baseline_v2 as evaluation
 import torch
 factory=evaluation.build_model;argv=sys.argv[:];reports={}
 meta={'experiment':'existing_resolution_candidate_revalidated_on_Core400','checkpoint_sha256':CKPT_SHA,
       'core_manifest_sha256':CORE_SHA,'checkpoint':str(checkpoint),'manifest':str(core),
       'score_threshold':.4,'mask_threshold':.5,'bbox_match_iou':.5,'warmup_runs':10,'timed_passes':5,
       'weights_modified':False,'training_started':False,'formal_prd_acceptance':False,
       'note':'Fixed Core regression, not an unseen blind holdout. No further resolution sweep.'}
 (out/'experiment.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
 for name,minimum,maximum in [('baseline640',640,1024),('candidate800',800,1333)]:
  def override(saved_min,saved_max,minimum=minimum,maximum=maximum):
   if (saved_min,saved_max)!=(640,1024):raise ValueError('Unexpected saved training resolution')
   model=factory(minimum,maximum)
   if tuple(model.transform.min_size)!=(minimum,) or model.transform.max_size!=maximum:raise ValueError('Inference resolution override failed')
   return model
  folder=out/name
  evaluation.build_model=override
  sys.argv=[str(Path(__file__)),'--checkpoint',str(checkpoint),'--val-csv',str(core),'--device','cuda',
            '--score-threshold','0.40','--mask-threshold','0.50','--match-bbox-iou','0.50',
            '--warmup-runs','10','--timed-passes','5','--max-visualizations','8','--report-dir',str(folder)]
  print('RUNNING_CORE400:',name,'min',minimum,'max',maximum,flush=True)
  try:evaluation.main()
  finally:evaluation.build_model=factory;sys.argv=argv
  with (folder/'per_gt_results.csv').open(encoding='utf-8-sig',newline='') as f:gt=list(csv.DictReader(f))
  if {(r.get('record_id') or r.get('final_sample_id', '')) for r in gt}!={(r.get('record_id') or r.get('final_sample_id', '')) for r in rows}:raise ValueError('Evaluation GT coverage mismatch')
  reports[name]=values(folder/'evaluation_summary.txt')
  if reports[name].get('validation_images')!='400' or reports[name].get('gt_instances')!='400':raise ValueError('Incomplete Core400 evaluation')
  (folder/'inference_resolution.json').write_text(json.dumps({'min_size':minimum,'max_size':maximum,'checkpoint_training_settings_unchanged':True},indent=2))
  gc.collect();torch.cuda.empty_cache()
 metrics=['bbox50_recall','end_to_end_bbox50_class_recall','mean_mask_iou_correct_class','mask_iou_ge_0_85_rate','mean_inference_ms_per_image']
 comparison={key:{'baseline640':float(reports['baseline640'][key]),'candidate800':float(reports['candidate800'][key]),
                  'delta_candidate_minus_baseline':float(reports['candidate800'][key])-float(reports['baseline640'][key])} for key in metrics}
 result={'status':'CORE400_PAIR_COMPLETED','images_per_run':400,'gt_per_run':400,'comparisons':comparison,
         'baseline_recall_drift_from_previous_0_505':comparison['end_to_end_bbox50_class_recall']['baseline640']-.505,
         'production_promoted':False,'formal_prd_acceptance':False,
         'decision':'Review global and per-class regression before adoption; candidate not automatically selected.'}
 (out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
  for p in out.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(out))
 if sha(checkpoint)!=CKPT_SHA or sha(core)!=CORE_SHA:raise RuntimeError('Frozen input unexpectedly changed')
 print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
 print('UPLOAD_THIS_ZIP:',out.with_suffix('.zip'),flush=True)

if __name__=='__main__':main()
