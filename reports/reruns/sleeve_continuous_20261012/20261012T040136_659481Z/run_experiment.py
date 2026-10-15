#!/usr/bin/env python3
"""Run only on the user's existing GPU project. No training data downloaded."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
from datetime import datetime, timezone
import numpy as np
from pilot_core import CLASSES, evaluate_variant, validate_manifest

HERE = Path(__file__).resolve().parent

def read_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write_json(p, obj): Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def crop_bounds(box, size):
    w,h=size
    a,b,c,d=box
    bounds=(max(0,math.floor(a)),max(0,math.floor(b)),min(w,math.ceil(c)),min(h,math.ceil(d)))
    if bounds[2]<=bounds[0] or bounds[3]<=bounds[1]: raise ValueError('Invalid garment crop')
    return bounds

def extract_features(rows, root, cache, device_name):
    os.environ['HF_HOME']=str(cache)
    os.environ['HF_HUB_CACHE']=str(cache/'hub')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    import transformers
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    if device_name=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Activate the existing py312 GPU environment.')
    # Validate all sources before paying model-loading cost. Never use annotated review images.
    paths=[]
    for row in rows:
        p=(root/row['source_image']).resolve()
        if not p.is_file(): raise FileNotFoundError(f"Missing original image for {row['record_id']}: {p}")
        if sha(p)!=row['source_image_sha256']:
            raise ValueError(f"Original-image SHA256 mismatch: {row['record_id']}")
        paths.append(p)
    device=torch.device(device_name)
    model_id='openai/clip-vit-base-patch32'
    processor=CLIPProcessor.from_pretrained(model_id, local_files_only=True)
    model=CLIPModel.from_pretrained(model_id, local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    features=[]
    for i,(row,p) in enumerate(zip(rows,paths),1):
        with Image.open(p) as original:
            crop=original.convert('RGB').crop(crop_bounds(row['garment_bbox_original'],original.size))
        pixels=processor(images=crop, return_tensors='pt')['pixel_values'].to(device)
        with torch.inference_mode():
            pooled=model.vision_model(pixel_values=pixels).pooler_output
            f=model.visual_projection(pooled)
            f=f/f.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        # Side metadata is provided to BOTH heads. No points, target ratio, or class in input.
        side=-0.25 if row['side']=='image_left' else 0.25
        features.append(np.r_[f[0].float().cpu().numpy(), side])
        print(f"feature {i:02d}/{len(rows)} {row['record_id']}",flush=True)
    versions={'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
              'transformers':transformers.__version__,'device':str(device),'clip_model':model_id,
              'clip_config_commit':getattr(model.config,'_commit_hash',None),
              'cuda_name':torch.cuda.get_device_name(0) if device.type=='cuda' else None}
    return np.asarray(features,dtype=np.float64), versions

def write_report(out, results):
    lines=['# 10.12 连续袖长配对试验结果','',
           '这是小样本开发诊断，不是独立测试集结果，也不是 3.1 验收结论。',
           '固定开发集 12 条中，9 条有可用连续标签，标签覆盖率为 75%；3 条不可测已保留排除记录。',
           '输入为原图的人工服饰框裁剪及所选侧别；尚未检验预测分割框下的全链路效果。','',
           '| 训练方案 | 训练/开发数 | 六类直接预测 | 回归后分档 | 回归 MAE | 均值基线 MAE |',
           '|---|---:|---:|---:|---:|---:|']
    for name,m in results.items():
        dc=m['direct_six_class_ridge']; rc=m['continuous_ridge_to_class']
        lines.append(f"| {name} | {m['n_train']}/{m['n_dev']} | {dc['correct']}/{dc['n']} | {rc['correct']}/{rc['n']} | {m['continuous_ridge']['mae']:.4f} | {m['train_mean_ratio_baseline']['mae']:.4f} |")
    lines += ['', '完整 macro-F1、RMSE、逐类支持数、混淆矩阵见 metrics.json；逐图预测见 predictions.csv。',
              '两种输出头采用相同冻结 CLIP 特征、相同岭回归强度 alpha=0.1。六类头拟合 one-hot 后取最大得分类别，不是概率校准结果。',
              '回归结果裁剪到 [0,1]；分类阈值仅取训练集中各类比例中位数的相邻中点；不以开发结果选参数。',
              'manual_ratio_mapping_NOT_model_accuracy 使用人工真值比例，只用于检查分档与类别标签的一致性，不能作为模型准确率。',
              '比例是单侧二维可见覆盖代理量，不是三维长度，也不是老师所说的类别概率软标签。',
              '本次只比较两种目标编码的线性试验；样本少且类别极不均衡，不能据此排除其他分类器或宣称总体提升。',
              '请同时看主试验和剔除两条端点截断训练样本的敏感性试验；不根据开发表现择优汇报。']
    (out/'结果说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,default=Path('/workspace/fashion_multimodal_analysis'))
    parser.add_argument('--hf-cache',type=Path,default=Path('/workspace/hf_cache'))
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    for p,digest in read_json(HERE/'package_hashes.json').items():
        if sha(HERE/p)!=digest: raise ValueError(f'Package changed: {p}; retain the original for reproducibility')
    data=read_json(HERE/'experiment_labels.json'); train,dev=validate_manifest(data)
    print('LABEL_CHECK_OK train=14 dev=9; dev coverage=9/12; two clipped train targets flagged',flush=True)
    if args.validate_only: return
    root=args.project_root.resolve()
    if not (root/'src/fashion_multimodal_analysis').is_dir(): raise ValueError(f'Wrong project root: {root}')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=root/'reports/reruns/sleeve_continuous_20261012'/stamp
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'run_status.json',{'status':'RUNNING','model_fit_performed':False})
    try:
        features,versions=extract_features(train+dev,root,args.hf_cache.resolve(),args.device)
        n=len(train); xt,xd=features[:n],features[n:]
        np.savez_compressed(out/'clip_features.npz',features=features,record_ids=np.array([r['record_id'] for r in train+dev]))
        results={}; all_rows=[]
        for variant in ['main_all14','sensitivity_without_clipped12']:
            keep=np.array([not r.get('endpoint_clipped_training',False) if variant.startswith('sensitivity') else True for r in train])
            tr=[r for r,k in zip(train,keep) if k]
            metrics,pred,weights=evaluate_variant(xt[keep],xd,tr,dev)
            results[variant]=metrics
            all_rows.extend(dict(variant=variant,**r) for r in pred)
            np.savez_compressed(out/(variant+'_weights.npz'),**weights)
        write_json(out/'metrics.json',results)
        with (out/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
        for f in ['experiment_labels.json','review_summary.json','annotation_changes.json','package_hashes.json','pilot_core.py','run_experiment.py','README.md']:
            shutil.copy2(HERE/f,out/f)
        shutil.copytree(HERE/'provenance',out/'provenance')
        write_json(out/'environment.json',versions)
        write_report(out,results)
        write_json(out/'run_status.json',{'status':'COMPLETE','model_fit_performed':True,'timestamp_utc':stamp,
                   'development_only':True,'dev_target_count':12,'dev_evaluated_count':9,
                   'no_dev_hyperparameter_selection':True})
        archive=shutil.make_archive(str(out)+'_results','zip',root_dir=out)
        print('\n'+(out/'结果说明.md').read_text(encoding='utf-8'),flush=True)
        print(f'UPLOAD_THIS_ZIP: {archive}',flush=True)
    except Exception as exc:
        write_json(out/'run_status.json',{'status':'FAILED','error':repr(exc),'complete_results_available':False})
        raise

if __name__=='__main__': main()
