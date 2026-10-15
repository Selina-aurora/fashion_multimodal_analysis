#!/usr/bin/env python3
"""V5 Balanced150, GT-free GPU chain integration check (NOT PRD accuracy acceptance).

Reuses the legacy 2026-10-09 exporter/geometry/contract checks.
Only writes outside the project root. Original checkpoints and manifests read-only.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path

ROOT_DEFAULT = Path('/workspace/fashion_multimodal_analysis')
OUT_DEFAULT = Path('/workspace/prd31_chain_v5balanced_20261014_runs')
LEGACY_BASELINE = Path('reports/reruns/recovery_20261009/prd31_neckline_context_20261009_012052')
CONFIG_REL = 'configs/prd_3_1_recovery_v2.json'
CKPT_REL = 'outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth'
TRAIN_REL = 'configs/prd_8class_train_v5_balanced150_20261008.csv'
VAL_REL = 'configs/prd_8class_val_v2_full_targets.csv'
CORE_REL = 'benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv'
EXPECTED_SHA = {
    TRAIN_REL: 'e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f',
    VAL_REL: '39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93',
    CORE_REL: 'c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89',
}
CLASS_TO_ID = {'top':1,'pants':2,'skirt':3,'outerwear':4,'dress':5,'shoe':6,'bag':7,'accessory':8}


def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def dump(p: Path, value) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str),encoding='utf-8')


def new_run_dir(out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    stamp=time.strftime('%Y%m%d_%H%M%S')
    for n in range(100):
        path=out_root / (f'run_{stamp}' if n==0 else f'run_{stamp}_{n:02d}')
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise FileExistsError('No unique run dir available; change --out-root')


def zip_dir(root: Path, path: Path) -> None:
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file():
                z.write(p,p.relative_to(root))
    with zipfile.ZipFile(path) as z:
        bad=z.testzip()
        if bad:
            raise IOError('ZIP verification failed: '+bad)


def check_hashes(root: Path, audit: dict) -> None:
    current={}
    for rel,expected in EXPECTED_SHA.items():
        p=root/rel
        if not p.is_file():raise FileNotFoundError(str(p))
        actual=sha(p)
        current[rel]={'sha256':actual,'matches_expected':actual==expected}
        if actual!=expected:
            audit['manifest_hashes']=current
            raise ValueError('Frozen input SHA mismatch: '+rel)
    audit['manifest_hashes']=current
    for rel in (CKPT_REL,CONFIG_REL):
        if not (root/rel).is_file():raise FileNotFoundError(str(root/rel))
    with (root/CORE_REL).open(newline='',encoding='utf-8-sig') as f:
        count=sum(1 for _ in csv.DictReader(f))
    if count!=400:raise ValueError('Core400 count changed: '+str(count))
    audit['core_gt_count']=count


def make_config(root: Path, audit: dict) -> tuple[dict, list[dict]]:
    from fashion_multimodal_analysis.common.paths import resolve_path
    prev=root/LEGACY_BASELINE
    if (prev/'run_config.json').is_file() and (prev/'cases.json').is_file():
        config=json.loads((prev/'run_config.json').read_text(encoding='utf-8'))
        selected=json.loads((prev/'cases.json').read_text(encoding='utf-8'))
        if not isinstance(selected,list) or len(selected)!=8:
            raise ValueError('Legacy baseline does not contain the expected 8 images')
        audit['source_selection']='LEGACY_DIAGNOSTIC_8_REUSED'
        audit['legacy_baseline_dir']=str(prev)
        if (prev/'summary.json').is_file():
            old=json.loads((prev/'summary.json').read_text(encoding='utf-8'))
            audit['legacy_input_sha256']=old.get('input_sha256',{})
    else:
        config=json.loads((root/CONFIG_REL).read_text(encoding='utf-8'))
        with (root/VAL_REL).open(newline='',encoding='utf-8-sig') as f:
            rows=list(csv.DictReader(f))
        selected=[]; used=set()
        for cat in CLASS_TO_ID:
            item=next((r for r in rows if r['garment_category']==cat and str(resolve_path(r['source_image'])) not in used),None)
            if item:
                source=resolve_path(item['source_image'])
                used.add(str(source))
                selected.append({'source_image':str(source),'selection_category':cat})
        if len(selected)<1:raise ValueError('No eligible validation diagnostic images')
        audit['source_selection']='VAL_SNAPSHOT_DIAGNOSTIC_FALLBACK'
        audit['warning']='Legacy 8-image selected cases unavailable; evaluation is not directly paired to the 2026-10-09 smoke sample.'
    config.update(
        segmentation_checkpoint=CKPT_REL,
        train_manifest=TRAIN_REL,
        val_manifest=VAL_REL,
        semantic_consistency_review=True,
        regression_patch_version='v5balanced_chain_readonly_20261014',
    )
    from fashion_multimodal_analysis.segmentation.training.checkpoint_state import CONTROLLED_SETTINGS
    config['checkpoint_settings']={**CONTROLLED_SETTINGS,'workers':2}
    if not isinstance(config.get('required_regions'),list) or not config['required_regions']:
        raise ValueError('Missing required_regions in existing configuration')
    if not isinstance(config.get('required_attributes'),list) or not config['required_attributes']:
        raise ValueError('Missing required_attributes in existing configuration')
    resolved=[]
    for item in selected:
        path=resolve_path(item['source_image'])
        if not path.is_file():raise FileNotFoundError('Required image missing: '+str(path))
        current_sha=sha(path)
        old_sha=item.get('source_sha256')
        if old_sha and old_sha!=current_sha:
            raise ValueError('Selected source image changed: '+str(path))
        resolved.append({'source_image':str(path),
                         'source_sha256':current_sha,
                         'selection_category':item.get('selection_category','unknown')})
    audit['selected_images']=[{'selection_category':s['selection_category'],
                                'source_sha256':s['source_sha256'],
                                'source_image':s['source_image']} for s in resolved]
    return config,resolved


def inspect_checkpoint(root: Path, audit: dict) -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; GPU chain run is not possible')
    audit['cuda_name']=torch.cuda.get_device_name(0)
    audit['torch_version']=torch.__version__
    path=root/CKPT_REL
    cp=torch.load(path,map_location='cpu',weights_only=False)
    if not isinstance(cp,dict):raise ValueError('Checkpoint is not a dictionary')
    if cp.get('epoch')!=15:raise ValueError('Checkpoint is not epoch 15')
    if cp.get('class_to_id')!=CLASS_TO_ID:raise ValueError('Class mapping differs')
    if cp.get('train_manifest_sha256')!=EXPECTED_SHA[TRAIN_REL]:
        raise ValueError('Checkpoint training manifest provenance mismatch')
    if cp.get('val_manifest_sha256')!=EXPECTED_SHA[VAL_REL]:
        raise ValueError('Checkpoint validation manifest provenance mismatch')
    if not isinstance(cp.get('model_state_dict'),dict):
        raise ValueError('Checkpoint missing model_state_dict')
    audit['checkpoint']={'path':str(path),'sha256':sha(path),'epoch':15,
                         'saved_min_size':cp.get('args',{}).get('min_size'),
                         'saved_max_size':cp.get('args',{}).get('max_size')}
    del cp;gc.collect()


def run_chain(root: Path, out: Path, config: dict, selected: list[dict], audit: dict) -> None:
    from PIL import Image
    from smoke_export import save_prediction
    from verify_chain_contracts import verify
    from check_geometry import run_checks
    from fashion_multimodal_analysis.integration.pipeline import FashionPipeline
    import fashion_multimodal_analysis.integration.pipeline as pipeline_module
    import fashion_multimodal_analysis.integration.model_backends as backend_module

    geometry=run_checks()
    if geometry.get('status')!='PASS':
        raise RuntimeError('Geometry tests failed')
    dump(out/'geometry_tests.json',geometry)
    print('GEOMETRY_CHECKS_PASSED:',len(geometry['tests']),flush=True)
    modules=[pipeline_module,backend_module]
    import importlib.util
    installed=importlib.util.find_spec('fashion_multimodal_analysis.integration.semantic_consistency')
    semantic_patch=('semantic_consistency_review' in Path(pipeline_module.__file__).read_text(encoding='utf-8'))
    if installed and installed.origin:
        modules.append(__import__('fashion_multimodal_analysis.integration.semantic_consistency',fromlist=['_']))
    audit['semantic_consistency_patch_present']=bool(semantic_patch and installed)
    config['semantic_consistency_review']=bool(semantic_patch and installed)
    audit['installed_code_sha256']={str(Path(m.__file__).resolve()):sha(Path(m.__file__)) for m in modules}
    if not config['semantic_consistency_review']:
        audit.setdefault('warnings',[]).append('Legacy semantic-consistency patch unavailable; pipeline still runs but this specific patch is NOT tested.')
    dump(out/'run_config.json',config)
    dump(out/'selected_images.json',selected)
    print('LOADING_EXISTING_BACKENDS (may take time)',flush=True)
    from fashion_multimodal_analysis.integration.model_backends import ModelBackends
    pipeline=FashionPipeline(ModelBackends(config,device='cuda'))
    cases=[]
    (out/'source_images').mkdir()
    for n,item in enumerate(selected,1):
        source=Path(item['source_image'])
        dest=out/'source_images'/f'case_{n:02d}.jpg'
        shutil.copyfile(source,dest)
        with Image.open(source) as raw:
            image=raw.convert('RGB')
        print(f'RUNNING_REAL_GPU_CHAIN: {n}/{len(selected)}',flush=True)
        t0=time.perf_counter()
        result=pipeline.run(image,config['required_regions'],config['required_attributes'])
        case=f'case_{n:02d}'
        payload=save_prediction(image,result,out/case)
        cases.append({'case_dir':case,
            'source_image':str(source), 'source_sha256':item['source_sha256'],
            'selection_category':item['selection_category'],
            'pipeline_status':result['status'],
            'instance_count':len(result['instances']),
            'failures':result['failures'],
            'interface_issues':payload['interface_issues'],
            'chain_wall_ms_single_pass':(time.perf_counter()-t0)*1000})
        dump(out/'cases.json',cases)
    check=verify(out)
    check['model_inference_run']=True
    dump(out/'contract_check.json',check)
    audit['contract_check']={'status':check['status'],
                             'checks':check.get('counts',{}).get('checks'),
                             'issues':len(check.get('issues',[]))}
    audit['summary']={
        'status':'FULL_CHAIN_INTERFACE_PASS' if check['status']=='PASS' else 'FULL_CHAIN_INTERFACE_FAIL',
        'model_inference_run':True,
        'images':len(cases),
        'predicted_instances':sum(c['instance_count'] for c in cases),
        'failed_downstream_calls':sum(len(c['failures']) for c in cases),
        'contract_checks':check.get('counts',{}).get('checks'),
        'semantic_consistency_review_enabled':config['semantic_consistency_review'],
        'accuracy_evaluated':False,
        'prd_latency_acceptance':False,
        'formal_prd_acceptance':False,
        'scope':'Existing V5 Balanced150 checkpoint on a fixed diagnostic image set; actual model inference and interface contracts.',
        'limitations':['Only a GT-free integration/smoke check, not PRD benchmark metrics.',
                       'Single un-warmed diagnostic wall times must not be reported as acceptance latency.',
                       'No human sign-off or formal v2 benchmark freeze is fabricated.']
    }
    dump(out/'summary.json',audit['summary'])
    if check['status']!='PASS':
        raise RuntimeError('Contract validation failed; inspect contract_check.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(ROOT_DEFAULT))
    p.add_argument('--out-root',default=str(OUT_DEFAULT))
    p.add_argument('--dry-run',action='store_true',help='Check manifests, 8 selected images and source configuration without loading GPU models')
    args=p.parse_args()
    root=Path(args.root).resolve()
    out_root=Path(args.out_root).resolve()
    out=new_run_dir(out_root)
    audit={'status':'STARTED','date':'2026-10-14','root':str(root),
           'mode':'DRY_RUN' if args.dry_run else 'REAL_GPU_INFERENCE',
           'baseline_checkpoint':CKPT_REL,'manifest_protected':True,
           'project_files_written':False}
    status=0
    try:
        sys.path.insert(0,str(root/'src'))
        os.environ['HF_HUB_OFFLINE']='1'
        os.environ['TRANSFORMERS_OFFLINE']='1'
        check_hashes(root,audit)
        config,selected=make_config(root,audit)
        if args.dry_run:
            audit['status']='DRY_RUN_PASS'
            print('DRY_RUN_PASS: source checks complete; no models loaded',flush=True)
        else:
            inspect_checkpoint(root,audit)
            run_chain(root,out,config,selected,audit)
            audit['status']='FULL_CHAIN_INTERFACE_PASS'
    except Exception as e:
        audit['status']='DRY_RUN_FAILED' if args.dry_run else 'CHAIN_FAILED'
        audit['error']=f'{type(e).__name__}: {e}'
        (out/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('RUN_FAILED:',audit['error'],flush=True)
        status=2
    finally:
        dump(out/'run_audit.json',audit)
        stamp=out.name
        zip_path=out_root/f'prd31_chain_v5balanced_{stamp}_UPLOAD_THIS.zip'
        try:
            zip_dir(out,zip_path)
            print('STATUS:',audit['status'],flush=True)
            print('UPLOAD_THIS_ZIP:',zip_path,flush=True)
            print('PROJECT_FILES_WRITTEN: False',flush=True)
        except Exception as e:
            print('PACKAGING_FAILED:',e,flush=True)
            status=3
    return status

if __name__=='__main__':sys.exit(main())
