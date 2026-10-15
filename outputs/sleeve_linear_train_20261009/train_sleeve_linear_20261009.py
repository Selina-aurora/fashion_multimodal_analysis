"""Small supervised sleeve head on frozen CLIP raw-garment embeddings.
44 reviewed training sources only; grouped development split chooses the head.
The prior 18 validation targets are evaluated only after head selection and
remain development diagnostics, not a blind PRD acceptance benchmark.
"""
import csv
import hashlib
import json
from pathlib import Path
import random
import time
import zipfile
from collections import Counter,defaultdict

LABELS=['sleeveless','cap','short','elbow','three_quarter','long']
LABEL_SHA='dd5d4dda69f10621e4cb8671130677401f5be34e099f6bd7d2199718303b306b'
TRAIN_SHA='e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f'
VAL_SHA='39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93'
DIAGNOSTIC='reports/reruns/recovery_20261009/attribute_results_20261009_064127_332405'
DIAG_SHA='578631308a786d0f49166a80b067476a65ad3fc20976aaf7d916988fc760d8c2'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def grouped_split(cases):
    """Fixed search for a label-stratified IMAGE split; no feature/model scores."""
    groups=defaultdict(list)
    for c in cases:groups[c['source_image_sha256']].append(c)
    total=Counter(c['sleeve_length'] for c in cases);best=None
    for attempt in range(300):
        rng=random.Random(20261009+attempt);dev=set();counts=Counter()
        for label in LABELS:
            pool=sorted(g for g,items in groups.items() if any(c['sleeve_length']==label for c in items))
            rng.shuffle(pool)
            for group in pool:
                if counts[label]>=2:break
                if group in dev:continue
                dev.add(group);counts.update(c['sleeve_length'] for c in groups[group])
        if any(counts[l]<1 or total[l]-counts[l]<2 for l in LABELS):continue
        score=sum(abs(counts[l]-2) for l in LABELS)
        if best is None or score<best[0]:best=(score,dev)
    if best is None:raise ValueError('Cannot make a grouped six-class split with adequate support')
    dev=best[1]
    return ([c for c in cases if c['source_image_sha256'] not in dev],
            [c for c in cases if c['source_image_sha256'] in dev])


def metrics(items):
    eligible=[r for r in items if r['truth']!='unknown'];confusion={}
    correct=sum(r.get('predicted')==r['truth'] for r in eligible)
    for r in eligible:
        g=r['truth'];p=r.get('predicted') or '<missing>';confusion.setdefault(g,{})[p]=confusion.setdefault(g,{}).get(p,0)+1
    return {'eligible':len(eligible),'correct':correct,'accuracy':correct/len(eligible) if eligible else None,
            'missing':sum(not r.get('predicted') for r in eligible),'unknown_excluded':len(items)-len(eligible),'confusion':confusion}


def main():
    import os
    os.environ.setdefault('HF_HOME','/workspace/hf_cache');os.environ.setdefault('HF_HUB_CACHE','/workspace/hf_cache/hub')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel,CLIPProcessor
    from fashion_multimodal_analysis.common.paths import project_root,resolve_path
    from fashion_multimodal_analysis.integration.pipeline import crop_bounds
    from fashion_multimodal_analysis.attributes.pattern import run_pattern_hierarchical_v3 as pattern
    from fashion_multimodal_analysis.attributes.design import run_design_attribute_labels_v1 as design
    root=project_root();here=Path(__file__).resolve().parent;labels_path=here/'reviewed_training_labels.json'
    if sha(labels_path)!=LABEL_SHA:raise ValueError('Reviewed training labels changed')
    cases=json.loads(labels_path.read_text())['cases']
    if len(cases)!=44 or len({c['record_id'] for c in cases})!=44:raise ValueError('Expected 44 unique usable reviewed targets')
    if any(c['split']!='train' or c['review_status']!='confirmed' or not c['confirmed_training_label'] or c['sleeve_length'] not in LABELS for c in cases):raise ValueError('Unapproved training label')
    def read(path):
        with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
    train_manifest=resolve_path('configs/prd_8class_train_v5_balanced150_20261008.csv');val_manifest=resolve_path('configs/prd_8class_val_v2_full_targets.csv')
    if sha(train_manifest)!=TRAIN_SHA or sha(val_manifest)!=VAL_SHA:raise ValueError('Frozen manifests changed')
    train_rows={r['record_id']:r for r in read(train_manifest)};val_rows={r['record_id']:r for r in read(val_manifest)}
    val_hashes={sha(resolve_path(r['source_image'])) for r in val_rows.values()}
    for c in cases:
        row=train_rows[c['record_id']];box=[float(row['bbox_'+k]) for k in ('x1','y1','x2','y2')]
        source_hash=sha(resolve_path(row['source_image']))
        if source_hash!=c['source_image_sha256'] or source_hash in val_hashes or box!=c['garment_bbox'] or row['garment_category']!=c['category'] or row['garment_id']!=c['garment_id']:raise ValueError('Training target changed or overlaps validation: '+c['record_id'])
    train,dev=grouped_split(cases)
    assert not {c['source_image_sha256'] for c in train}&{c['source_image_sha256'] for c in dev}
    diagnostic=root/DIAGNOSTIC;diag_annotations=diagnostic/'confirmed_annotations.json'
    if sha(diag_annotations)!=DIAG_SHA:raise ValueError('Diagnostic annotations changed')
    diag_cases=json.loads(diag_annotations.read_text())['cases']
    cached=json.loads((diagnostic/'per_target_results.json').read_text())
    if len(diag_cases)!=18:raise ValueError('Expected 18 diagnostic targets')
    for c in diag_cases:
        if sha(resolve_path(val_rows[c['record_id']]['source_image']))!=c['source_image_sha256']:raise ValueError('Diagnostic image changed')
    if not torch.cuda.is_available():raise RuntimeError('Run in GPU py312')
    random.seed(20261009);np.random.seed(20261009);torch.manual_seed(20261009);torch.cuda.manual_seed_all(20261009)
    device=torch.device('cuda');model_id='openai/clip-vit-base-patch32'
    processor=CLIPProcessor.from_pretrained(model_id,local_files_only=True)
    clip=CLIPModel.from_pretrained(model_id,local_files_only=True).to(device).eval()
    clip.requires_grad_(False)
    out=root/'reports/reruns/recovery_20261009'/('sleeve_linear_train_'+time.strftime('%Y%m%d_%H%M%S')+f'_{time.time_ns()%1000000:06d}')
    out.mkdir(parents=True,exist_ok=False)
    split={'train':train,'dev':dev,'group_key':'source_image_sha256'};dump(out/'split.json',split)
    print('SPLIT:',len(train),'train;',len(dev),'dev; no shared source hashes',flush=True)
    features={};flips={}
    with torch.inference_mode():
        for number,c in enumerate(cases,1):
            image=Image.open(resolve_path(train_rows[c['record_id']]['source_image'])).convert('RGB')
            crop=image.crop(crop_bounds(c['garment_bbox'],image.size))
            features[c['record_id']]=pattern.encode_image(crop,clip,processor,device).detach().cpu()
            if c in train:
                flips[c['record_id']]=pattern.encode_image(crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT),clip,processor,device).detach().cpu()
            print(f'FEATURE_DONE {number}/44',flush=True)
        proto=pattern.build_class_prototypes({k:[v] for k,v in design.SLEEVE_PROMPTS.items()},clip,processor,device)
    # Tensor cloning outside inference_mode allows autograd through the head.
    train_x=torch.cat([features[c['record_id']] for c in train]+[flips[c['record_id']] for c in train]).clone().to(device)
    train_y=torch.tensor([LABELS.index(c['sleeve_length']) for c in train]*2,device=device)
    dev_x=torch.cat([features[c['record_id']] for c in dev]).clone().to(device)
    dev_y=torch.tensor([LABELS.index(c['sleeve_length']) for c in dev],device=device)
    head=torch.nn.Linear(train_x.shape[1],len(LABELS)).to(device)
    counts=torch.bincount(train_y,minlength=len(LABELS)).float();weights=counts.sum()/(len(LABELS)*counts)
    criterion=torch.nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.05)
    history=[];best_key=None;best_state=None;best_epoch=None
    for epoch in range(1,151):
        head.train();optimizer.zero_grad();loss=criterion(head(train_x),train_y);loss.backward();optimizer.step()
        head.eval()
        with torch.no_grad():
            logits=head(dev_x);dev_loss=float(torch.nn.functional.cross_entropy(logits,dev_y));dev_correct=int((logits.argmax(1)==dev_y).sum())
        key=(dev_correct,-dev_loss)
        if best_key is None or key>best_key:
            best_key=key;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};best_epoch=epoch
        history.append({'epoch':epoch,'train_loss':float(loss.detach()),'dev_loss':dev_loss,'dev_correct':dev_correct,'dev_count':len(dev)})
        if epoch%25==0:print(f'EPOCH {epoch}/150: dev {dev_correct}/{len(dev)}',flush=True)
    head.load_state_dict(best_state);head.eval()
    checkpoint={'head_state_dict':best_state,'labels':LABELS,'feature_dim':train_x.shape[1],'clip_model':model_id,
                'input_view':'raw_garment_bbox_no_expansion','best_epoch':best_epoch,'training_labels_sha256':LABEL_SHA,
                'split':split,'production_enabled':False}
    torch.save(checkpoint,out/'sleeve_head_best.pth');dump(out/'training_history.json',history)
    def predict(feature):
        with torch.no_grad():prob=head(feature.clone().to(device)).softmax(1)[0].cpu();idx=int(prob.argmax())
        return LABELS[idx],float(prob[idx])
    supervised_results={};prompt_results={}
    for role,items in [('train',train),('dev',dev)]:
        supervised_results[role]=[];prompt_results[role]=[]
        for c in items:
            feature=features[c['record_id']];label,score=predict(feature)
            supervised_results[role].append({'record_id':c['record_id'],'truth':c['sleeve_length'],'predicted':label,'score':score})
            with torch.inference_mode():p=pattern.classify_with_prototypes(feature.to(device),*proto,clip)
            prompt_results[role].append({'record_id':c['record_id'],'truth':c['sleeve_length'],'predicted':p['top1_label'],'score':p['top1_score']})
    # No diagnostic predictions or accuracy computed before best head selection.
    diag_truth={c['record_id']:c for c in diag_cases}
    for role,items in cached.items():
        name='diagnostic_'+role;supervised_results[name]=[];prompt_results[name]=[]
        with torch.inference_mode():
            for r in items:
                rid=r['record_id'];gt=diag_truth[rid]['attributes']['sleeve_length'];label=None;score=None;prompt_label=None
                if role=='gt_roi' or r['status']=='MATCHED_PREDICTED_TARGET':
                    image=Image.open(resolve_path(val_rows[rid]['source_image'])).convert('RGB')
                    box=diag_truth[rid]['garment_bbox'] if role=='gt_roi' else r['predicted_garment_box']
                    feature=pattern.encode_image(image.crop(crop_bounds(box,image.size)),clip,processor,device)
                    label,score=predict(feature)
                    p=pattern.classify_with_prototypes(feature,*proto,clip);prompt_label=p['top1_label']
                supervised_results[name].append({'record_id':rid,'truth':gt,'predicted':label,'score':score,'target_status':r['status']})
                prompt_results[name].append({'record_id':rid,'truth':gt,'predicted':prompt_label,'target_status':r['status']})
    dump(out/'per_target_results.json',{'supervised_head':supervised_results,'same_raw_view_prompt_baseline':prompt_results})
    summary={'status':'EXPERIMENTAL_SLEEVE_HEAD_TRAINING_COMPLETED','reviewed_usable_targets':44,
             'label_counts':dict(Counter(c['sleeve_length'] for c in cases)),'train_targets':len(train),'dev_targets':len(dev),
             'train_label_counts':dict(Counter(c['sleeve_length'] for c in train)),'dev_label_counts':dict(Counter(c['sleeve_length'] for c in dev)),
             'best_epoch':best_epoch,'selection_rule':'Max dev correct, tie lower dev CE; diagnostic targets not used',
             'epochs':150,'seed':20261009,'lr':.01,'weight_decay':.05,'augmentation':'Train-only horizontal flip',
             'clip_frozen':True,'input_view':'raw_garment_bbox_no_expansion','head_checkpoint_sha256':sha(out/'sleeve_head_best.pth'),
             'metrics':{role:metrics(items) for role,items in supervised_results.items()},
             'same_raw_view_prompt_baseline_metrics':{role:metrics(items) for role,items in prompt_results.items()},
             'production_changed':False,'formal_prd_acceptance':False,
             'limitations':['44 small manually reviewed sources, native class hints corrected by review; generalization remains uncertain.',
                            'Internal dev is small and used for checkpoint selection; its accuracy is optimistic.',
                            '18 previously inspected validation targets are development diagnostics, not a fresh blind test.',
                            'Frozen CLIP, linear head only; no segmentation or local-region model updates.',
                            'Input-view choice followed prior diagnostic analysis; independently reviewed new images still needed before promotion.',
                            'Softmax scores are uncalibrated; unmatched diagnostic targets remain missing errors.']}
    dump(out/'summary.json',summary);dump(out/'reviewed_training_labels.json',json.loads(labels_path.read_text()))
    dump(out/'provenance.json',{'runner_sha256':sha(Path(__file__)),'training_labels_sha256':LABEL_SHA,
                              'pattern_helper_sha256':sha(pattern.__file__),'design_helper_sha256':sha(design.__file__),
                              'diagnostic_results_sha256':sha(diagnostic/'per_target_results.json')})
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():z.write(p,p.relative_to(out))
    print(json.dumps(summary['metrics'],ensure_ascii=False,indent=2),flush=True)
    print('UPLOAD_THIS_ZIP:',archive,flush=True)


if __name__=='__main__':main()
