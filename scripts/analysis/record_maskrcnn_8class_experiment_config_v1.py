"""Command entrypoint; implementation is in src/fashion_multimodal_analysis."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "src/fashion_multimodal_analysis").is_dir()
)
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    from fashion_multimodal_analysis.cli import run_entry

    run_entry("record_maskrcnn_8class_experiment_config_v1")
