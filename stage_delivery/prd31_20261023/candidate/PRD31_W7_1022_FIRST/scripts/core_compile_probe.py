"""Paired same-query compiler screen with no embedding reuse or precision change."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime
import hashlib
import functools
import importlib
import inspect
from pathlib import Path
import traceback
import zipfile
import numpy as np
from PIL import Image
from core_clock import CoreClock, PreparedProcessor, compile_ground, core_scope, prepare_inputs
from e2e_logic import canonical, compare, mask_summary
from run_attributes import HERE, prepare, read, write, verify, sha
from white_reuse import patch


def select_queries(jobs, reference, count):
    eligible = [j for j in jobs if reference[j['key']]['status'] != 'not_applicable']
    if count >= len(eligible):
        return eligible
    groups = defaultdict(list)
    for j in eligible:
        groups[(j['region'], reference[j['key']]['status'])].append(j)
    for rows in groups.values():
        rows.sort(key=lambda j: hashlib.sha256(('1022-compile:'+repr(j['key'])).encode()).hexdigest())
    selected = []
    while len(selected) < count:
        for key in sorted(groups):
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
        if not any(groups.values()):
            break
    return selected


class SwitchingProcessor:
    def __init__(self, original, kind, owner):
        self.original, self.kind, self.owner = original, kind, owner

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, *args, **kwargs):
        prepared = self.owner.prepared
        if prepared is None:
            return self.original(*args, **kwargs)
        return PreparedProcessor(self.original, self.kind, prepared[self.kind],
            self.owner.clock, self.owner.torch, prepared['key'])(*args, **kwargs)


class TrackedCompile:
    """Count actual compiled graph execution, not only compile() construction."""
    def __init__(self, torch, original, dynamic):
        from torch._dynamo.backends.registry import lookup_backend
        self.attempts = self.compiled_graphs = self.executions = 0
        inductor = lookup_backend('inductor')

        def tracked(gm, inputs, **options):
            self.attempts += 1
            graph = inductor(gm, inputs, **options)
            self.compiled_graphs += 1
            @functools.wraps(graph)
            def execute(*args, **kwargs):
                self.executions += 1
                return graph(*args, **kwargs)
            return execute
        # Fullgraph=False permits supported graph regions with eager boundaries.
        # Every timed candidate query must nevertheless execute compiled DINO work.
        self.call = torch.compile(original, backend=tracked, dynamic=dynamic, fullgraph=False)

    def record(self):
        return {'compile_attempts': self.attempts, 'compiled_graphs': self.compiled_graphs,
                'compiled_graph_executions': self.executions}


class PairedCore:
    """Keep wrapper identities stable for BOTH modes and every query.

    Replacing Tensor methods separately for each call can invalidate Dynamo
    guards. One outer scope avoids that artifact while resetting all clock and
    input state per query. No visual/text embedding is retained.
    """
    def __init__(self, backend, module, torch, compiler_factory=TrackedCompile):
        self.backend, self.module, self.torch = backend, module, torch
        self.clock = CoreClock(torch.cuda.synchronize)
        self.prepared = None
        self.candidate = False
        self.dino_original = backend.ground_model.forward
        self.vision_original = backend.sam_model.vision_encoder.forward
        self.dino = compiler_factory(torch, self.dino_original, True)
        self.vision = compiler_factory(torch, self.vision_original, False)

    def __enter__(self):
        self.stack = ExitStack()
        try:
            self.stack.enter_context(patch(self.backend, 'ground_processor',
                SwitchingProcessor(self.backend.ground_processor, 'dino', self)))
            self.stack.enter_context(patch(self.backend, 'sam_processor',
                SwitchingProcessor(self.backend.sam_processor, 'sam', self)))
            def dino(*args, **kwargs):
                return (self.dino.call if self.candidate else self.dino_original)(*args, **kwargs)
            def vision(*args, **kwargs):
                return (self.vision.call if self.candidate else self.vision_original)(*args, **kwargs)
            self.stack.enter_context(patch(self.backend.ground_model, 'forward', dino))
            self.stack.enter_context(patch(self.backend.sam_model.vision_encoder, 'forward', vision))
            sam_module = importlib.import_module('fashion_multimodal_analysis.grounding.refinement.evaluate_sam_refinement_9case')
            self.stack.enter_context(core_scope(self.backend, None, self.clock, self.torch, sam_module))
            self.ground = compile_ground(self.backend.ground, self.clock)
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        self.prepared = None
        return self.stack.__exit__(*exc)

    def invoke(self, job, image, candidate):
        self.candidate = candidate
        self.prepared = None
        ready = prepare_inputs(self.backend, self.module, image, job['roi'], job['region'], self.torch)
        if ready is None:
            raise RuntimeError('SCREEN_QUERY_MUST_BE_STRUCTURALLY_APPLICABLE')
        # __init__ resets timing data; generated function source/audit attrs remain.
        self.clock.__init__(self.torch.cuda.synchronize)
        self.prepared = ready
        before = (self.dino.executions, self.vision.executions,
                  self.dino.compiled_graphs, self.vision.compiled_graphs)
        prediction, unused = self.ground(image, job['roi'], job['region'])
        row = self.clock.record()
        needs_sam = prediction['status'] in ('detected', 'empty_mask')
        required = {'dino': 1, **({'sam': 1, 'sam_vision': 1} if needs_sam else {})}
        if not row['eligible_core_sample'] or row['forward_calls'] != required:
            raise RuntimeError('PER_QUERY_VISUAL_TEXT_RECOMPUTATION_COVERAGE_CHANGED')
        row['compiled_dino_graph_executions'] = self.dino.executions-before[0]
        row['compiled_sam_vision_graph_executions'] = self.vision.executions-before[1]
        row['new_compiled_graphs_during_call'] = self.dino.compiled_graphs+self.vision.compiled_graphs-before[2]-before[3]
        if candidate:
            if row['compiled_dino_graph_executions'] < 1:
                raise RuntimeError('COMPILED_DINO_WORK_NOT_OBSERVED; eager fallback is not accepted as compiled')
            if needs_sam and row['compiled_sam_vision_graph_executions'] < 1:
                raise RuntimeError('COMPILED_SAM_VISION_WORK_NOT_OBSERVED')
        elif row['compiled_dino_graph_executions'] or row['compiled_sam_vision_graph_executions']:
            raise RuntimeError('BASELINE_MUST_USE_ORIGINAL_EAGER_FORWARD')
        if needs_sam:
            row['endpoint_mask'] = mask_summary(self.clock.local_mask, ready['crop_size'])
            row['endpoint_bbox_roi_local'] = self.clock.local_bbox
        row['timing_boundary_source_sha256'] = self.clock.boundary_sha256
        return canonical(prediction, image.size), row


def summarize(records, name):
    rows = [r for r in records if r['mode'] == name]
    values = [r['core_active_wall_ms'] for r in rows]
    return {'n': len(values), 'mean_ms': float(np.mean(values)),
            'p50_ms': float(np.percentile(values, 50)), 'p95_ms': float(np.percentile(values, 95))}


def benchmark(runner, jobs, images, reference, out, passes=5):
    print('COMPILE WARMUP may compile kernels; progress resumes after compilation', flush=True)
    for n in range(10):
        job = jobs[n % len(jobs)]
        for mode in (False, True):
            print(f'CORE WARMUP {n+1}/10 {"compiled" if mode else "eager"}', flush=True)
            runner.invoke(job, images[job['image_id']], mode)
    checks, outputs = [], []
    for n, job in enumerate(jobs):
        meta = {k: job[k] for k in ('image_id', 'instance_id', 'region')}
        versions = {}
        for candidate in ([False, True] if n % 2 == 0 else [True, False]):
            mode = 'compiled' if candidate else 'eager'
            print(f'CORE PREFLIGHT {n+1}/{len(jobs)} {mode}', flush=True)
            pred, clock = runner.invoke(job, images[job['image_id']], candidate)
            match = compare(reference[job['key']], pred)
            outputs.append({**meta, 'mode': mode, 'prediction': pred, 'clock': clock})
            versions[mode] = pred
            checks.append({**meta, 'mode': mode, **match})
            write(out/'preflight_outputs.json', outputs)
            write(out/'preflight_checks.json', checks)
            if not match['passed']:
                raise RuntimeError('COMPILE_PREFLIGHT_OUTPUT_DRIFT')
        if not compare(versions['eager'], versions['compiled'])['passed']:
            raise RuntimeError('PAIRED_PREFLIGHT_DRIFT')
    records, timed_checks = [], []
    for p in range(passes):
        for n, job in enumerate(jobs):
            meta = {'pass': p, **{k: job[k] for k in ('image_id', 'instance_id', 'region')}}
            versions = {}
            for candidate in ([False, True] if (p+n) % 2 == 0 else [True, False]):
                mode = 'compiled' if candidate else 'eager'
                print(f'CORE PASS {p+1}/{passes} QUERY {n+1}/{len(jobs)} {mode}', flush=True)
                pred, row = runner.invoke(job, images[job['image_id']], candidate)
                match = compare(reference[job['key']], pred)
                records.append({**meta, 'mode': mode, **row})
                timed_checks.append({**meta, 'mode': mode, **match})
                versions[mode] = pred
                write(out/'timed_records.json', records)
                write(out/'timed_checks.json', timed_checks)
                if not match['passed']:
                    write(out/'drift_output.json', {'reference': reference[job['key']], 'actual': pred, **meta, 'mode': mode})
                    raise RuntimeError('COMPILED_TIMED_OUTPUT_DRIFT')
            if not compare(versions['eager'], versions['compiled'])['passed']:
                raise RuntimeError('PAIRED_TIMED_DRIFT')
    stages = {mode: summarize(records, mode) for mode in ('eager', 'compiled')}
    pairs = defaultdict(dict)
    for row in records:
        pairs[(row['pass'], row['image_id'], row['instance_id'], row['region'])][row['mode']] = row['core_active_wall_ms']
    faster = sum(r['compiled'] < r['eager'] for r in pairs.values())
    reduction = (1-stages['compiled']['mean_ms']/stages['eager']['mean_ms'])*100
    recompiles = sum(r['new_compiled_graphs_during_call'] for r in records)
    return {'status': 'COMPILE_PAIRED_SCREEN_COMPLETE', 'complete': True,
        'all_output_checks_pass': True, 'unique_queries': len(jobs), 'passes': passes,
        'stages': stages, 'mean_reduction_percent': reduction, 'candidate_faster_pairs': faster,
        'paired_queries': len(pairs), 'model_embeddings_recomputed_every_query': True,
        'recompilation_events_in_timed_scope': recompiles,
        'timed_scope_contains_recompilation_if_any': True,
        'compiled_graph_tracking': {'dino': runner.dino.record(), 'sam_vision': runner.vision.record()},
        'full_527_query_coverage': len(jobs) == 527,
        'configured_30ms_target_status': ('at_or_below' if stages['compiled']['mean_ms'] <= 30 else 'above') if len(jobs) == 527 else f'not_assessed_on_{len(jobs)}_query_screen',
        'do_not_compare_screen_mean_directly_to_historical_139_63ms': len(jobs) != 527,
        'compiled_candidate_promoted': False, 'new_accuracy_claim': False,
        'production_modified': False, 'precision_settings_changed': False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', default='/workspace/fashion_multimodal_analysis')
    ap.add_argument('--prior', default='/workspace/prd31_week7_1021_batch/run_20261021_033050_887414')
    ap.add_argument('--benchmark', default='/workspace/prd31_week7_1021_benchmark/run_20261021_040120_362409')
    ap.add_argument('--profile', default='/workspace/prd31_week7_1021_ground_profile/run_20261021_080233_916752')
    ap.add_argument('--count', type=int, choices=[32,64,527], default=64)
    ap.add_argument('--output-root', default='/workspace/prd31_week7_1022_compile')
    ap.add_argument('--output-dir')
    args = ap.parse_args()
    out = Path(args.output_dir).resolve() if args.output_dir else Path(args.output_root).resolve()/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    code = 1
    try:
        profile = Path(args.profile).resolve()
        verify(profile, read(HERE/'inputs/PROFILE_REFERENCE_SHA256.json'))
        project, prior, bench, backend, module, Pipeline, old_jobs, images, old, torch = prepare(args, out)
        verify(project, read(HERE/'inputs/GROUND_HELPER_SHA256.json'))
        for name, expected in read(HERE/'inputs/TRANSFORMERS_SOURCE_SHA256.json').items():
            if sha(Path(inspect.getsourcefile(type(getattr(backend,name))))) != expected:
                raise RuntimeError('TRANSFORMERS_IMPLEMENTATION_CHANGED')
        precision_before={'float32_matmul_precision':torch.get_float32_matmul_precision(),
            'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}
        write(out/'precision_settings.json',precision_before)
        from fashion_multimodal_analysis.integration.pipeline import GarmentROI
        jobs, reference = [], {}
        for record in read(prior/'segmentation_all_predictions.json'):
            rid = record['image_id']
            for row in read(profile/'outputs'/f'{rid}_ordinary.json')['raw_ground_queries']:
                reference[(rid,row['instance_id'],row['region'])] = row['prediction']
            for item in record['instances']:
                with Image.open(prior/item['mask_file']) as image:
                    mask = (np.asarray(image)>0).astype('uint8')
                roi = GarmentROI(instance_id=item['instance_id'],category=item['category'],score=item['score'],bbox=item['bbox'],mask=mask)
                for region in backend.config['required_regions']:
                    jobs.append({'image_id':rid,'instance_id':roi.instance_id,'region':region,'roi':roi,'key':(rid,roi.instance_id,region)})
        if len(jobs) != 944 or sum(reference[j['key']]['status']!='not_applicable' for j in jobs) != 527:
            raise RuntimeError('FROZEN_QUERY_SET_CHANGED')
        selected = select_queries(jobs,reference,args.count)
        write(out/'selected_queries.json', [{k:j[k] for k in ('image_id','instance_id','region')} for j in selected])
        write(out/'measurement_definition.json', {
            'selection':'fixed seed region x original return-status strata; no selection using compiled results',
            'selected':len(selected),'original_applicable_queries':527,'excluded_not_applicable':417,
            'scope':'same active synchronized wall protocol as 085326; one GPU-ready ROI and query to local bbox/binary mask',
            'initial_precomputed_model_embeddings':False,'cross_query_embedding_cache':False,
            'compile_targets':['DINO forward','SAM vision encoder'],'dtype_batch_weights_inputs_unchanged':True,
            'partially_compiled_regions_possible':True,'actual_graph_executions_checked_per_query':True,
            'warmups_per_mode':10,'passes':5,'numeric_atol':0.0001,'binary_masks_exact':True,
            'formal_complete_527_target_test':len(selected)==527,
            'screen_mean_not_a_replacement_for_139_63ms':len(selected)!=527})
        del old_jobs, old
        with torch.inference_mode():
            with PairedCore(backend,module,torch) as runner:
                audit=out/'timing_source'
                audit.mkdir()
                (audit/'original_ground.py').write_text(runner.clock.original_source)
                (audit/'timed_ground.py').write_text(runner.clock.timed_source+'\n')
                (audit/'original_sam_helper.py').write_text(runner.clock.sam_original_source)
                (audit/'timed_sam_helper.py').write_text(runner.clock.sam_timed_source+'\n')
                result=benchmark(runner,selected,images,reference,out)
        for folder,name in [(prior,'PRIOR_FILES_SHA256.json'),(bench,'REFERENCE_BENCH_SHA256.json'),(profile,'PROFILE_REFERENCE_SHA256.json')]:
            verify(folder,read(HERE/'inputs'/name))
        for name in ['source_hashes.json','FROZEN_ATTRIBUTE_SOURCES.json','GROUND_HELPER_SHA256.json']:
            verify(project,read(HERE/'inputs'/name))
        for row in read(prior/'images_inference_only.json'):
            if sha(Path(row['path']))!=row['sha256']:raise RuntimeError('SOURCE_IMAGE_CHANGED')
        precision_after={'float32_matmul_precision':torch.get_float32_matmul_precision(),
            'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}
        if precision_after!=precision_before:raise RuntimeError('PRECISION_SETTINGS_CHANGED')
        result.update(production_sources_unchanged=True,reference_files_unchanged=True,source_images_unchanged=True)
        write(out/'summary.json',result)
        code=0
    except BaseException:
        error=traceback.format_exc()
        print(error,flush=True)
        write(out/'fatal_error.json',{'error':error})
        write(out/'summary.json',{'status':'COMPILE_SCREEN_BLOCKED_OR_INTERRUPTED','complete':False,
            'all_output_checks_pass':False,'speed_change_valid':False,'production_modified':False,'error':error})
    finally:
        archive=out.with_name(out.name+'_UPLOAD_THIS.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.rglob('*')):
                if f.is_file():z.write(f,f.relative_to(out))
        print('UPLOAD_THIS: '+str(archive),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
