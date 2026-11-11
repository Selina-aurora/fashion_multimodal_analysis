"""Load the exact GPU-provided guard source without overwriting it."""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path

EXPECTED_SHA256 = "6eb785c71c23a0b20529ac4c391dac007666017f6d0c85fbc948527662594f7c"
GUARD_RELATIVE = Path("src/fashion_multimodal_analysis/integration/prd32_output_guard.py")


def load_guard(path: Path):
    path = Path(path).resolve(strict=True)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA256:
        raise ValueError(f"Guard source SHA changed: {actual}; refuse incompatible integration")
    spec = importlib.util.spec_from_file_location("prd32_verified_gpu_guard", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load verified guard module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.guard_outputs
