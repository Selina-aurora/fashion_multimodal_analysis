"""Both modes use the validated white reuse; candidate adds ROI-local SAM reuse."""
from contextlib import ExitStack
import copy
import time
import numpy as np
from PIL import Image
from e2e_logic import FullBackend, canonical, health
from run_attributes import SIX
from sam_vision_reuse import SamVisionReuse
from white_reuse import patch


class SamBackend(FullBackend):
    def __init__(self, backend, torch, reuse, capture=False):
        super().__init__(backend, True, capture)
        self.torch = torch
        self.reuse = reuse
        self.state = None
        self.image_ref = self.roi_ref = None
        self.states = []
        self.raw_ground = []
        self.vision_events = []
        self.dino_calls = self.sam_calls = 0
        self.expected_dino = self.expected_sam = 0

    def ground(self, image, roi, region):
        # Identity is used only to bound state, never to assume pixel equality.
        if image is not self.image_ref or roi is not self.roi_ref:
            if self.state is not None:
                self.states.append(self.state.record())
                self.state.clear()
            self.image_ref, self.roi_ref = image, roi
            self.state = SamVisionReuse(self.backend.sam_model.vision_encoder,
                                        self.torch, self.reuse, self.capture_raw)
        event_start = len(self.state.events)
        with self.state.activate():
            result, ms = self.backend.ground(image, roi, region)
        self.expected_dino += int(result['status'] != 'not_applicable')
        self.expected_sam += int(result['status'] in ('detected', 'empty_mask'))
        if self.capture_raw:
            self.raw_ground.append({'instance_id': roi.instance_id, 'region': region,
                                    'prediction': copy.deepcopy(result)})
            self.vision_events.append({'instance_id': roi.instance_id, 'region': region,
                                       'events': self.state.events[event_start:]})
        return result, ms

    def finish(self):
        if self.state is not None:
            self.states.append(self.state.record())
            self.state.clear()
            self.state = None
        self.image_ref = self.roi_ref = None

    def cache_stats(self):
        result = {k: sum(x[k] for x in self.states) for k in
                  ('vision_requests', 'actual_vision_forwards', 'cache_hits')}
        return {**result, 'roi_scopes': len(self.states),
                'dino_calls': self.dino_calls, 'sam_calls': self.sam_calls}


def full_call(backend, Pipeline, path, queries, candidate, sync, torch, capture=False):
    adapter = SamBackend(backend, torch, candidate, capture)
    pipe = Pipeline(adapter)
    def counter(model, attribute):
        original = model.forward
        def observe(*args, **kwargs):
            setattr(adapter, attribute, getattr(adapter, attribute) + 1)
            return original(*args, **kwargs)
        return patch(model, 'forward', observe)
    # Patches are process-local, sequential and restored on every exit path.
    with ExitStack() as stack:
        stack.enter_context(counter(backend.ground_model, 'dino_calls'))
        stack.enter_context(counter(backend.sam_model, 'sam_calls'))
        sync()
        start = time.perf_counter()
        try:
            with Image.open(path) as im:
                image = im.convert('RGB')
            result = pipe.run(image, list(queries), list(SIX))
        finally:
            # Cache allocation, exact comparisons, clones and cleanup are timed.
            adapter.finish()
            sync()
        elapsed = (time.perf_counter() - start) * 1000
    snapshot = canonical({k: v for k, v in result.items() if k != 'timings'}, image.size)
    if not health(snapshot, queries)['passed']:
        raise RuntimeError('PIPELINE_FAILED_OR_INVALID_OUTPUT')
    stats = adapter.cache_stats()
    applicable = adapter.expected_dino
    sam_paths = adapter.expected_sam
    if stats['dino_calls'] != applicable or stats['sam_calls'] != sam_paths:
        raise RuntimeError('GROUNDING_MODEL_COVERAGE_CHANGED')
    if stats['vision_requests'] != sam_paths or stats['actual_vision_forwards'] + stats['cache_hits'] != sam_paths:
        raise RuntimeError('VISION_CACHE_ACCOUNTING_CHANGED')
    if not candidate and stats['cache_hits'] != 0:
        raise RuntimeError('BASELINE_REUSE_FORBIDDEN')
    audit = {'sam_cache': stats}
    if capture:
        audit.update(raw_attributes=canonical(adapter.raw, image.size),
                     raw_ground=canonical(adapter.raw_ground, image.size),
                     vision_events=adapter.vision_events,
                     white_cache=adapter.cache)
    return snapshot, elapsed, result.get('timings', []), audit
