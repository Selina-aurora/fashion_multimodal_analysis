"""One bounded sleeve-context diagnostic: cached masked baseline vs raw views.
Does not train, change labels, tune prompts or install a production candidate.
18 reviewed validation images are development diagnostics, not blind acceptance.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import zipfile

REPORT = 'reports/reruns/recovery_20261009/attribute_results_20261009_064127_332405'
ANNOTATION_SHA = '578631308a786d0f49166a80b067476a65ad3fc20976aaf7d916988fc760d8c2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def context_bounds(box, size, expansion):
    """Fixed margin on each side, independent of sleeve GT labels."""
    x,y,xx,yy=box;w,h=size
    dx=(xx-x)*expansion;dy=(yy-y)*expansion
    return [max(0,math.floor(x-dx)),max(0,math.floor(y-dy)),min(w,math.ceil(xx+dx)),min(h,math.ceil(yy+dy))]


def metrics(rows, view):
    eligible=[r for r in rows if r['truth']!='unknown']
    emitted=[r for r in eligible if r['views'].get(view,{}).get('label')]
    correct=sum(r['views'].get(view,{}).get('label')==r['truth'] for r in eligible)
    return {'eligible':len(eligible),'emitted':len(emitted),'correct':correct,
            'missing':len(eligible)-len(emitted),
            'accuracy_all_eligible':correct/len(eligible) if eligible else None,
            'accuracy_on_emitted':correct/len(emitted) if emitted else None,
            'unknown_excluded':sum(r['truth']=='unknown' for r in rows)}


def main():
    os.environ.setdefault('HF_HOME','/workspace/hf_cache')
    os.environ.setdefault('HF_HUB_CACHE','/workspace/hf_cache/hub')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel,CLIPProcessor
    from fashion_multimodal_analysis.common.paths import project_root,resolve_path
    from fashion_multimodal_analysis.attributes.pattern import run_pattern_hierarchical_v3 as pattern
    from fashion_multimodal_analysis.attributes.design import run_design_attribute_labels_v1 as design
    root=project_root();report=root/REPORT
    annotation=report/'confirmed_annotations.json'
    if sha(annotation)!=ANNOTATION_SHA:raise ValueError('Confirmed labels changed')
    confirmed=json.loads(annotation.read_text())['cases']
    if len(confirmed)!=18 or not all(c['review_status']=='confirmed' for c in confirmed):raise ValueError('Invalid confirmations')
    labels={c['record_id']:c for c in confirmed}
    baseline=json.loads((report/'per_target_results.json').read_text())
    summary=json.loads((report/'summary.json').read_text())
    if summary['annotations_sha256']!=ANNOTATION_SHA or summary['images']!=18:raise ValueError('Unexpected baseline')
    with resolve_path('configs/prd_8class_val_v2_full_targets.csv').open(encoding='utf-8-sig',newline='') as stream:
        manifest={r['record_id']:r for r in csv.DictReader(stream)}
    images={}
    for rid,c in labels.items():
        path=resolve_path(manifest[rid]['source_image'])
        if sha(path)!=c['source_image_sha256']:raise ValueError('Source changed: '+rid)
        images[rid]=Image.open(path).convert('RGB')
    for role,items in baseline.items():
        if len(items)!=18 or {r['record_id'] for r in items}!=set(labels):raise ValueError('Baseline target set changed')
        if metrics([{'truth':r['truth']['sleeve_length'],'views':{'masked_baseline':r['prediction'].get('sleeve_length',{})}} for r in items],'masked_baseline')['correct']!=3:
            raise ValueError('Baseline sleeve result changed')
        for r in items:
            if r['truth']!=labels[r['record_id']]['attributes']:raise ValueError('Baseline truth changed')
    if not torch.cuda.is_available():raise RuntimeError('Run in GPU py312')
    device=torch.device('cuda');model_id='openai/clip-vit-base-patch32'
    processor=CLIPProcessor.from_pretrained(model_id,local_files_only=True)
    model=CLIPModel.from_pretrained(model_id,local_files_only=True).to(device).eval()
    with torch.inference_mode():
        proto=pattern.build_class_prototypes({k:[v] for k,v in design.SLEEVE_PROMPTS.items()},model,processor,device)
    out=root/'reports/reruns/recovery_20261009'/('sleeve_context_'+time.strftime('%Y%m%d_%H%M%S')+f'_{time.time_ns()%1000000:06d}')
    out.mkdir(parents=True,exist_ok=False);(out/'views').mkdir()
    results={}
    with torch.inference_mode():
        for role,items in baseline.items():
            results[role]=[]
            for r in items:
                rid=r['record_id'];image=images[rid]
                result={'record_id':rid,'category':r['category'],'truth':r['truth']['sleeve_length'],
                        'status':r['status'],'views':{'masked_baseline':r['prediction'].get('sleeve_length',{})}}
                # Preserve the exact baseline association failures; never recover a
                # wrong-class target with a GT crop in the predicted-ROI arm.
                box=labels[rid]['garment_bbox'] if role=='gt_roi' else r.get('predicted_garment_box')
                eligible_roi=role=='gt_roi' or r['status']=='MATCHED_PREDICTED_TARGET'
                if eligible_roi:
                    for view,expansion in [('raw_garment',0.),('raw_context_10pct',.10)]:
                        bounds=context_bounds(box,image.size,expansion)
                        crop=image.crop(bounds)
                        crop.save(out/'views'/f'{rid}_{role}_{view}.jpg',quality=90)
                        feature=pattern.encode_image(crop,model,processor,device)
                        pred=pattern.classify_with_prototypes(feature,*proto,model)
                        result['views'][view]={'label':pred['top1_label'],'score':pred['top1_score'],
                                               'top2_label':pred['top2_label'],'margin':pred['margin'],'bounds':bounds}
                results[role].append(result)
                print(f'{role} {len(results[role])}/18 {rid}',flush=True)
    views=['masked_baseline','raw_garment','raw_context_10pct']
    paired={}
    for role,items in results.items():
        paired[role]={}
        for view in views[1:]:
            same=[r for r in items if r['truth']!='unknown' and r['views'].get(view,{}).get('label') and r['views']['masked_baseline'].get('label')]
            changed=[r['record_id'] for r in same if r['views'][view]['label']!=r['views']['masked_baseline']['label']]
            fixed=[r['record_id'] for r in same if r['views']['masked_baseline']['label']!=r['truth'] and r['views'][view]['label']==r['truth']]
            broken=[r['record_id'] for r in same if r['views']['masked_baseline']['label']==r['truth'] and r['views'][view]['label']!=r['truth']]
            paired[role][view]={'paired_eligible':len(same),'changed':changed,'fixed':fixed,'broken':broken}
    report_summary={'status':'SLEEVE_CONTEXT_DIAGNOSTIC_COMPLETED','baseline_report':str(report),
                    'metrics':{role:{view:metrics(items,view) for view in views} for role,items in results.items()},
                    'paired_changes':paired,'clip_model':model_id,'context_expansion_per_side':.10,
                    'production_changed':False,'formal_prd_acceptance':False,
                    'limitations':['Same reviewed segmentation validation targets; this is development analysis, not independent validation.',
                                   'Two fixed raw views, same model and sleeve prompts, no training or threshold tuning.',
                                   'GT ROI and predicted ROI have different scope; association failures remain misses.',
                                   'Even improved raw accuracy does not establish a production improvement without fresh reviewed evaluation.']}
    dump(out/'summary.json',report_summary);dump(out/'per_target_results.json',results)
    dump(out/'provenance.json',{'baseline_results_sha256':sha(report/'per_target_results.json'),
                               'annotations_sha256':ANNOTATION_SHA,'runner_sha256':sha(Path(__file__)),
                               'sleeve_prompts':design.SLEEVE_PROMPTS,'design_helper_sha256':sha(design.__file__),
                               'pattern_helper_sha256':sha(pattern.__file__)})
    table=[]
    for role,items in results.items():
        for r in items:
            for view in views:
                p=r['views'].get(view,{})
                table.append({'role':role,'record_id':r['record_id'],'category':r['category'],'view':view,
                              'truth':r['truth'],'predicted':p.get('label',''),'score':p.get('score',''),
                              'correct':p.get('label')==r['truth'] if r['truth']!='unknown' else '', 'target_status':r['status']})
    with (out/'sleeve_context_results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zipped:
        for p in sorted(out.rglob('*')):
            if p.is_file():zipped.write(p,p.relative_to(out))
    print(json.dumps(report_summary['metrics'],ensure_ascii=False,indent=2),flush=True)
    print('UPLOAD_THIS_ZIP:',archive,flush=True)


if __name__=='__main__':main()
