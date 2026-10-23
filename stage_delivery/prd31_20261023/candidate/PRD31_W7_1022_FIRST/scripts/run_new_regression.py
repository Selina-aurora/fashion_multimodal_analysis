"""One paired pass on frozen new image candidates; no GT accuracy claim."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import traceback
from candidate_base import load_backend
from run_candidate import infer_image
from run_attributes import HERE, read, write, sha, verify
from e2e_logic import compare


def run(backend, Pipeline, torch, manifest, out):
    records = read(manifest)
    rows = records['images']
    if not rows or len({r['sha256'] for r in rows}) != len(rows):
        raise RuntimeError('NEW_IMAGE_MANIFEST_EMPTY_OR_DUPLICATED')
    for row in rows:
        if sha(Path(row['path'])) != row['sha256']:
            raise RuntimeError('FROZEN_NEW_SOURCE_IMAGE_CHANGED')
    # Two warmup requests per mode; these observations are not a formal speed test.
    for n in range(2):
        row = rows[n % len(rows)]
        for mode in ('white','white-sam'):
            destination=out/'warmups'/f'{n}_{mode}'
            destination.mkdir(parents=True)
            infer_image(backend,Pipeline,torch,Path(row['path']),destination,mode)
    checks = []
    observed = []
    for n, row in enumerate(rows):
        rid=row['image_id']
        outputs={}
        for mode in (('white','white-sam') if n%2==0 else ('white-sam','white')):
            print(f'NEW IMAGE {n+1}/{len(rows)} {mode} {rid}',flush=True)
            folder=out/'predictions'/rid/mode
            folder.mkdir(parents=True)
            result=infer_image(backend,Pipeline,torch,Path(row['path']),folder,mode)
            outputs[mode]=result['output']
            observation=read(folder/'runtime_observation.json')
            observed.append({'image_id':rid,'mode':mode,**observation})
        comparison=compare(outputs['white'],outputs['white-sam'])
        checks.append({'image_id':rid,**comparison,'instances':len(outputs['white']['instances'])})
        write(out/'output_checks.json',checks)
        write(out/'observations.json',observed)
        if not comparison['passed']:
            raise RuntimeError('NEW_IMAGE_PAIRED_OUTPUT_DRIFT: '+rid)
    for row in rows:
        if sha(Path(row['path']))!=row['sha256']:
            raise RuntimeError('NEW_SOURCE_IMAGE_CHANGED_DURING_RUN')
    return {'status':'NEW_IMAGE_SINGLE_PASS_REGRESSION_COMPLETE','complete':True,
        'images':len(rows),'all_output_checks_pass':True,'instances':sum(r['instances'] for r in checks),
        'max_numeric_difference':max(r['max_numeric_absolute_difference'] for r in checks),
        'source_images_unchanged':True,'warmups_per_mode':2,'paired_passes':1,
        'formal_performance_benchmark':False,'formal_core_latency':False,'accuracy_claim':False,
        'certified_blind_test':False,'scene_and_full_history_review_pending':True,
        'note':'One paired regression pass on additional candidates. Do not merge it with fixed24 formal timing or call output equality accuracy.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='/workspace/fashion_multimodal_analysis')
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--output-dir',required=True)
    args=ap.parse_args()
    out=Path(args.output_dir).resolve()
    out.mkdir(parents=True,exist_ok=False)
    project=Path(args.project).resolve()
    try:
        manifest=Path(args.manifest).resolve()
        manifest_digest=sha(manifest)
        backend,Pipeline,torch=load_backend(project,out)
        result=run(backend,Pipeline,torch,manifest,out)
        if sha(manifest)!=manifest_digest:raise RuntimeError('NEW_IMAGE_MANIFEST_CHANGED')
        for name in ['source_hashes.json','FROZEN_ATTRIBUTE_SOURCES.json','GROUND_HELPER_SHA256.json']:
            verify(project,read(HERE/'inputs'/name))
        result.update(production_sources_unchanged=True,source_manifest_sha256=manifest_digest)
        write(out/'summary.json',result)
        return 0
    except BaseException:
        error=traceback.format_exc()
        print(error,flush=True)
        write(out/'summary.json',{'status':'NEW_REGRESSION_BLOCKED_OR_INTERRUPTED','complete':False,'error':error})
        return 1


if __name__=='__main__':raise SystemExit(main())
