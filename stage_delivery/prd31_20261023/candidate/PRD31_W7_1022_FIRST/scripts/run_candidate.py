"""Offline delivery entry: validated white/SAM adapter, explicit white fallback."""
import argparse
from datetime import datetime
from pathlib import Path
import traceback
import zipfile
import numpy as np
from PIL import Image
from candidate_base import load_backend
from run_attributes import HERE, read, write, verify, sha
from e2e_logic import compare, mask_summary
from sam_reuse_logic import full_call


def infer_image(backend, Pipeline, torch, source, out, mode='white-sam'):
    source = Path(source).resolve()
    before = sha(source)
    holder = {}
    # Run the exact GPU-validated full_call. Retain its raw return only for export.
    def capture_factory(adapter):
        class Capture:
            def run(self, *args, **kwargs):
                result = Pipeline(adapter).run(*args, **kwargs)
                holder['raw'] = result
                return result
        return Capture()
    with torch.inference_mode():
        snapshot, ms, modules, audit = full_call(backend, capture_factory, source,
            backend.config['required_regions'], mode == 'white-sam', torch.cuda.synchronize, torch)
    with Image.open(source) as im:
        size = im.size
    mask_dir = out/'masks'
    mask_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for n, item in enumerate(holder['raw']['instances']):
        values = [('garment', item['roi'].mask)] + [(q, r.get('mask')) for q, r in item['regions'].items()]
        for k, (region, mask) in enumerate(values):
            if mask is None:
                continue
            relative = f'masks/{n:04d}_{k:02d}.png'
            Image.fromarray((np.asarray(mask) > 0).astype('uint8') * 255).save(out/relative)
            with Image.open(out/relative) as exported:
                assert mask_summary(np.asarray(exported), size) == mask_summary(mask, size)
            index.append({'instance_id': item['roi'].instance_id, 'region': region, 'file': relative})
    if sha(source) != before:
        raise RuntimeError('SOURCE_IMAGE_CHANGED')
    prediction = {'source_image': str(source), 'source_image_sha256': before,
                  'source_size': list(size), 'mode': mode, 'output': snapshot, 'mask_files': index}
    write(out/'prediction.json', prediction)
    write(out/'runtime_observation.json', {'full_request_observed_ms': ms,
        'formal_core_latency': False, 'formal_performance_benchmark': False,
        'sam_reuse_enabled': mode == 'white-sam', 'sam_cache': audit['sam_cache'],
        'first_encode_and_cache_management_included': True, 'mask_export_outside_timer': True,
        'module_diagnostic_timings': modules})
    return prediction


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', default='/workspace/fashion_multimodal_analysis')
    ap.add_argument('--image', required=True)
    ap.add_argument('--mode', choices=['white-sam', 'white'], default='white-sam')
    ap.add_argument('--check-smoke-reference', action='store_true')
    ap.add_argument('--output-root', default='/workspace/prd31_candidate_1022')
    ap.add_argument('--output-dir', help='Exact fresh output folder, for orchestration')
    args = ap.parse_args()
    project = Path(args.project).resolve()
    source = Path(args.image).resolve()
    out = Path(args.output_dir).resolve() if args.output_dir else Path(args.output_root).resolve()/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    code = 1
    try:
        backend, Pipeline, torch = load_backend(project, out)
        # These three adapters are byte-identical to the validated 092427 package.
        for name, digest in read(HERE/'inputs/VERIFIED_SAM_HELPERS.json').items():
            if sha(HERE/'scripts'/name) != digest:
                raise RuntimeError('VERIFIED_HELPER_CHANGED: '+name)
        result = infer_image(backend, Pipeline, torch, source, out, args.mode)
        comparison = None
        if args.check_smoke_reference:
            reference = read(HERE/'inputs/SMOKE_REFERENCE.json')
            if result['source_image_sha256'] != reference['source']['sha256']:
                raise RuntimeError('WRONG_SMOKE_SOURCE_IMAGE')
            comparison = compare(reference['output'], result['output'])
            if not comparison['passed']:
                write(out/'reference_comparison.json', comparison)
                raise RuntimeError('CANDIDATE_ENTRY_OUTPUT_DRIFT')
        for name in ['source_hashes.json','FROZEN_ATTRIBUTE_SOURCES.json','GROUND_HELPER_SHA256.json']:
            verify(project, read(HERE/'inputs'/name))
        env = read(out/'environment.json')
        env.update(sam_reuse_enabled=args.mode == 'white-sam', selected_mode=args.mode,
                   entry_model_initialization_executed=True)
        write(out/'environment.json', env)
        write(out/'summary.json', {'status': 'DELIVERY_ENTRY_CHECK_COMPLETE', 'complete': True,
            'mode': args.mode, 'model_initialization_executed': True, 'comparison': comparison,
            'exported_masks': len(result['mask_files']), 'production_installed': False,
            'production_sources_unchanged': True, 'accuracy_claim': False})
        code = 0
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        write(out/'fatal_error.json', {'error': error})
        write(out/'summary.json', {'status': 'ENTRY_BLOCKED_OR_INTERRUPTED', 'complete': False,
                                  'production_installed': False})
    finally:
        archive = out.with_name(out.name+'_UPLOAD_THIS.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.rglob('*')):
                if f.is_file():
                    z.write(f, f.relative_to(out))
        print('UPLOAD_THIS: '+str(archive), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
