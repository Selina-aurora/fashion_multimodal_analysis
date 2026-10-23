"""Experimental SAM vision-output reuse, scoped to ONE predicted garment ROI.

The original SamModel.forward, prompt encoder, mask decoder, processors and
postprocessing still execute per query. Only exact repeated vision inputs may
reuse a cloned vision output. All cache work belongs inside full-image timing.
"""
from contextlib import contextmanager
import hashlib
import inspect
from white_reuse import patch


def scalar_key(value):
    if value is None or type(value) in (bool, str, int, float):
        return (type(value).__name__, value)
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(scalar_key(x) for x in value))
    if isinstance(value, dict):
        return tuple((k, scalar_key(v)) for k, v in sorted(value.items()))
    raise RuntimeError('UNSUPPORTED_NONPIXEL_VISION_ARGUMENT: ' + type(value).__name__)


def clone_tree(value, torch):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        values = {k: clone_tree(v, torch) for k, v in value.items()}
        return values if type(value) is dict else type(value)(**values)
    if isinstance(value, tuple):
        return tuple(clone_tree(x, torch) for x in value)
    if isinstance(value, list):
        return [clone_tree(x, torch) for x in value]
    if value is None or type(value) in (bool, str, int, float):
        return value
    raise RuntimeError('UNSUPPORTED_VISION_OUTPUT: ' + type(value).__name__)


def digest_tree(value, torch):
    """Preflight only. Never materialize vision features in timed runs."""
    if torch.is_tensor(value):
        array = value.detach().cpu().contiguous().numpy()
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'sha256': hashlib.sha256(array.tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {k: digest_tree(v, torch) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [digest_tree(v, torch) for v in value]
    if value is None or type(value) in (bool, str, int, float):
        return value
    raise RuntimeError('UNSUPPORTED_VISION_AUDIT_VALUE')


class SamVisionReuse:
    def __init__(self, vision, torch, enabled, capture=False):
        if vision.training:
            raise RuntimeError('VISION_MODEL_MUST_BE_IN_EVAL_MODE')
        self.vision = vision
        self.torch = torch
        self.enabled = enabled
        self.capture = capture
        self.original = vision.forward
        self.signature = inspect.signature(self.original)
        self.pixels = None
        self.output = None
        self.key = None
        self.calls = self.encodes = self.hits = 0
        self.events = []
        self.active = False

    def clear(self):
        self.pixels = self.output = self.key = None

    def record(self):
        return {'vision_requests': self.calls, 'actual_vision_forwards': self.encodes,
                'cache_hits': self.hits, 'enabled': self.enabled}

    @contextmanager
    def activate(self):
        if self.active:
            raise RuntimeError('NESTED_VISION_REUSE_SCOPE_FORBIDDEN')
        self.active = True

        def forward(*args, **kwargs):
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()
            pixels = bound.arguments.get('pixel_values')
            if not self.torch.is_tensor(pixels):
                raise RuntimeError('VISION_PIXEL_TENSOR_MISSING')
            flags = scalar_key({k: v for k, v in bound.arguments.items() if k != 'pixel_values'})
            key = (tuple(pixels.shape), str(pixels.dtype), str(pixels.device),
                   tuple(pixels.stride()), flags)
            self.calls += 1
            reused = bool(self.enabled and self.output is not None and key == self.key
                          and self.torch.equal(pixels, self.pixels))
            if reused:
                self.hits += 1
                # Never expose the stored copy to downstream in-place mutation.
                output = clone_tree(self.output, self.torch)
            else:
                self.encodes += 1
                output = self.original(*args, **kwargs)
                if self.enabled:
                    self.key = key
                    self.pixels = pixels.detach().clone()
                    self.output = clone_tree(output, self.torch)
            if self.capture:
                self.events.append({'input': digest_tree(pixels, self.torch),
                                    'output': digest_tree(output, self.torch),
                                    'reused': reused})
            return output

        try:
            with patch(self.vision, 'forward', forward):
                yield self
        finally:
            self.active = False


def semantic_vision_events(rows):
    return [{**{k: r[k] for k in ('instance_id', 'region')},
             'events': [{k: e[k] for k in ('input', 'output')} for e in r['events']]}
            for r in rows]
