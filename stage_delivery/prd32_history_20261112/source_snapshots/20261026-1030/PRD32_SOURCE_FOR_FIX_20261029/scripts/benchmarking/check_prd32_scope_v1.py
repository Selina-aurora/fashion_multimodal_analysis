"""Run the PRD3.2 draft request and asset preflight from this checkout."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Delegate to the CPU-only implementation, without initializing a model."""
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from fashion_multimodal_analysis.benchmarking.prd32_preflight_v1 import main as run

    return run()


if __name__ == "__main__":
    raise SystemExit(main())
