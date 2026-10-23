#!/usr/bin/env python3
"""8-case, read-only diagnostic: frozen CLIP semantic sleeve scalar vs human 2D coverage.
DO NOT interpret the CLIP expectation value as a physical sleeve length estimator.
Uses precisely the same raw garment GT bbox image view as prior CLIP/head experiment.
Does not update any model, train/validation file or frozen Core400.
"""
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path('/workspace/fashion_multimodal_analysis')
PILOT = Path('/workspace/sleeve_pilot8_controlled_20261012/pilot8_candidates_FINAL.json')
PREFLIGHT = Path('/workspace/pilot8_preflight_core400_report.json')
TRAIN_CSV = ROOT / 'configs/prd_8class_train_v5_balanced150_20261008.csv'
VAL_CSV = ROOT / 'configs/prd_8class_val_v2_full_targets.csv'
TRAIN_SHA = 'e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f'
VAL_SHA = '39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93'
PILOT_SHA = 'cb34eff7920e99c6803546afc4aa29c5ae3c0a1adacd313ecd0a2619cc8cef15'
LABELS = {'cap', 'short', 'elbow'}
BUNDLE = Path(__file__).resolve().parent
BASELINE = BUNDLE / 'prior_8case_frozen_classifiers.json'
EXPECTED_ANCHORS = {'sleeveless':0.0, 'cap':0.18, 'short':0.35, 'elbow':0.55, 'three_quarter':0.75, 'long':1.0}


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for piece in iter(lambda:f.read(1024*1024),b''):h.update(piece)
    return h.hexdigest()


def rows(path):
    with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def pearson(a,b):
    if len(a)<2:return None
    av=sum(a)/len(a);bv=sum(b)/len(b)
    num=sum((x-av)*(y-bv) for x,y in zip(a,b))
    den=math.sqrt(sum((x-av)**2 for x in a)*sum((y-bv)**2 for y in b))
    return num/den if den>1e-12 else None


def ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i])
    rr=[0.0]*len(values)
    start=0
    while start<len(values):
        end=start+1
        while end<len(values) and values[order[end]]==values[order[start]]:end+=1
        mean_rank=(start+1+end)/2
        for i in order[start:end]:rr[i]=mean_rank
        start=end
    return rr


def summarize(items):
    gt=[x['manual_2d_ratio_gt'] for x in items]
    pred=[x['automatic_clip_anchor_expectation'] for x in items]
    ae=[abs(a-b) for a,b in zip(gt,pred)]
    n=len(ae)
    c=t=ties=0
    for i in range(n):
        for j in range(i+1,n):
            d0=gt[i]-gt[j];d1=pred[i]-pred[j]
            if abs(d0)<1e-12 or abs(d1)<1e-12:ties+=1;continue
            t+=1;c+=int(d0*d1>0)
    g=defaultdict(list)
    for r in items:g[r['truth_class']].append(r['absolute_error_unscaled'])
    return dict(n=n, mae=sum(ae)/n, rmse=math.sqrt(sum(v*v for v in ae)/n),
      pearson=pearson(gt,pred),spearman=pearson(ranks(gt),ranks(pred)),
      ordered_pairs_correct=c,ordered_pairs_evaluable=t,pairwise_order_accuracy=c/t if t else None,
      tied_pairs_omitted=ties,
      per_class={k:{'n':len(v),'mae':sum(v)/len(v)} for k,v in sorted(g.items())},
      previous_classifier_accuracy={'linear_head':sum(r['prior_head_correct'] for r in items)/n,
                                    'clip_prompt':sum(r['prior_prompt_correct'] for r in items)/n})


def main():
    assert sha256(PILOT)==PILOT_SHA, 'pilot8 JSON SHA changed; stop'
    assert sha256(TRAIN_CSV)==TRAIN_SHA, 'train manifest changed; stop'
    assert sha256(VAL_CSV)==VAL_SHA, 'val manifest changed; stop'
    pfl=json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    assert pfl['status']=='PREFLIGHT_PASSED_NO_OVERLAP' and pfl['checks']['core400_overlap_free'], 'Core400 isolation not verified'
    cases=json.loads(PILOT.read_text(encoding='utf-8'))['annotations']
    assert len(cases)==8 and len({r['record_id'] for r in cases})==8
    trained={r['record_id']:r for r in rows(TRAIN_CSV)}
    prior=json.loads(BASELINE.read_text(encoding='utf-8'))
    prior={r['record_id']:r for r in prior}
    assert set(prior)=={r['record_id'] for r in cases},'baseline IDs mismatch'
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from fashion_multimodal_analysis.common.paths import resolve_path
    from fashion_multimodal_analysis.integration.pipeline import crop_bounds
    from fashion_multimodal_analysis.attributes.geometry import run_sleeve_length_continuous_v1 as cont
    from fashion_multimodal_analysis.attributes.design import run_design_attribute_labels_v1 as design
    assert {k:v[0] for k,v in cont.SLEEVE_ANCHORS.items()}==EXPECTED_ANCHORS,'old CLIP anchor implementation changed'
    torch.set_num_threads(4)
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ.setdefault('HF_HOME','/workspace/hf_cache')
    os.environ.setdefault('HF_HUB_CACHE','/workspace/hf_cache/hub')
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_name='openai/clip-vit-base-patch32'
    proc=CLIPProcessor.from_pretrained(model_name,local_files_only=True)
    clip=CLIPModel.from_pretrained(model_name,local_files_only=True).to(device).eval()
    clip.requires_grad_(False)
    output=[]
    for i,c in enumerate(cases,1):
        rid=c['record_id'];row=trained[rid];old=prior[rid]
        assert c['review_status']=='confirmed' and c['class_confirmed'] and c['state']=='measurable'
        assert c['current_class'] in LABELS and c['same_arm_confirmed'] and not c['sleeveless_confirmed']
        assert c['source_image']==row['source_image'] and c['garment_id']==row['garment_id']
        assert old['truth']==c['current_class'] and old['source_image_sha256']==c['source_image_sha256']
        path=resolve_path(row['source_image'])
        assert sha256(path)==c['source_image_sha256'],f'input image changed: {rid}'
        box=[float(row['bbox_'+k]) for k in ('x1','y1','x2','y2')]
        with Image.open(path) as im:
            image=im.convert('RGB')
            crop=image.crop(crop_bounds(box,image.size))
            with torch.inference_mode():p=cont.infer_continuous_ratio(crop,clip,proc,device)
        gt=float(c['ratio']);pred=float(p['sleeve_length_ratio'])
        assert math.isfinite(pred) and 0<=pred<=1
        r={'record_id':rid,'truth_class':c['current_class'],'manual_2d_ratio_gt':gt,
            'automatic_clip_anchor_expectation':pred,
            'signed_error_unscaled':pred-gt,'absolute_error_unscaled':abs(pred-gt),
            'auto_top_anchor':p['top_anchor'],'auto_top_anchor_probability':p['top_anchor_probability'],
            'auto_normalized_entropy':p['normalized_entropy'],
            'prior_head_class':old['supervised_head'],'prior_prompt_class':old['prompt_baseline'],
            'prior_head_correct':int(old['supervised_head']==old['truth']),
            'prior_prompt_correct':int(old['prompt_baseline']==old['truth']),
            'source_image_sha256':c['source_image_sha256']}
        output.append(r)
        print(f'CONTINUOUS_CLIP {i}/8 {rid}: human={gt:.3f} auto={pred:.3f}',flush=True)
    summary=summarize(output)
    summary.update({'status':'PILOT8_CLIP_CONTINUOUS_DIAGNOSTIC_COMPLETED',
       'not_an_anatomical_or_physical_ratio_model':True,'no_training':True,'production_modified':False,
       'input_view':'raw_garment_gt_bbox_no_expansion (matched to prior 8-case experiment)',
       'automated_target':'CLIP 6-class anchor probability expectation, not geometric shoulder-elbow-wrist-cuff prediction',
       'gt_target':'human-labelled single-arm 2D sleeve coverage ratio',
       'source_pilot_json_sha256':sha256(PILOT),
       'source_code_sha256':sha256(Path(cont.__file__)),
       'prior_8case_result_sha256':sha256(BASELINE),
       'limitations':['8 train-origin annotated images: exploratory, not blind test',
          'CLIP anchor expected value is not calibrated to physical sleeve coverage',
          'Different six-prompt wordings from prior separate prompt classifier',
          'Ground-truth garment bbox supplied; no full end-to-end predicted ROI',
          'Same eight were previously inspected; no tuning from these observations'],
       'next_step':'If ranking fails, test independent cuff and pose estimator without using GT keypoints at inference; reserve separate attribute holdout.'})
    outroot=Path('/workspace/sleeve_continuous_pilot8_diagnostic_20261012')
    target=outroot / ('run_'+time.strftime('%Y%m%d_%H%M%S'));target.mkdir(parents=True,exist_ok=False)
    (target/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (target/'predictions.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    with (target/'per_case_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    archive=target.with_suffix('.zip')
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for file in target.iterdir():z.write(file,arcname=file.name)
    print('DIAGNOSTIC_COMPLETED')
    print('SPEARMAN:',summary['spearman'])
    print('MAE_UNCALIBRATED:',summary['mae'])
    print('NO_TRAINING_OR_DATA_CHANGE: TRUE')
    print('UPLOAD_THIS_ZIP:',archive,flush=True)

if __name__=='__main__':main()
