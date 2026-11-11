#!/usr/bin/env python3
"""Offline PRD3.2 service output guard; no model load and no GT files.

Accept an array of prediction records as JSON or JSONL; write a guarded JSON.
Create a fresh output directory to prevent accidental overwrites.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fashion_multimodal_analysis.integration.prd32_output_guard import guard_outputs


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    args=p.parse_args(argv)
    path=args.input.resolve(strict=True)
    if any(token in str(path).lower() for token in ('humanreview', 'human_review', 'holdout', 'sealed', 'reference_gt')):
        p.error('human-review / reference / sealed evaluation inputs are forbidden')
    raw=path.read_text(encoding='utf-8-sig')
    if path.suffix.lower()=='.jsonl':
        rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        rows=json.loads(raw)
    report=guard_outputs(rows)
    dest=args.output_dir
    dest.mkdir(parents=True,exist_ok=False)
    (dest/'prd32_service_guard_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f"GUARD_COMPLETE: {report['instance_count']} instances, {len(report['geometric_overlap_pairs'])} overlap cues")
    print(f"STATUS: {report['service_status']}; publish=False; PRD_acceptance=False")
    print('OUTPUT:',dest/'prd32_service_guard_result.json')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
