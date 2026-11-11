#!/usr/bin/env python3
"""Read-only, bounded DEV-only Python entrypoint collector; never scans Holdout.

Inspects only explicitly allowed DEV Qwen experiment dirs and the repository's
scripts. Does not export datasets, prompts, response text, images, model weights.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

FORBIDDEN = ("holdout", "sealed", "blind", "reference_gt", "human_review",
             "train", "val", "test", "token", "credential", "secret", "private")
RISK_SECRET = re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|hf[_-]?token|password|secret)\s*[:=]\s*['\"](?!\{)[^'\"]{8,}['\"]")
TOKENS = ("qwen", "transformers", "from_pretrained", "generate(", "model.chat", "apply_chat_template")


def has_forbidden_folder(path: Path) -> bool:
    parts = [p.lower() for p in path.parts]
    return any(any(bad in p for bad in FORBIDDEN) for p in parts)


def collect(workspace: Path, out: Path, max_files: int=250):
    base = workspace.resolve(strict=True)
    if not base.is_dir() or has_forbidden_folder(base.relative_to(base)):
        raise ValueError('invalid workspace')
    roots = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("PRD32_DEV60_QWEN_")]
    repo = base/'fashion_multimodal_analysis'/'scripts'
    if repo.is_dir(): roots.append(repo)
    selected=[]
    inspected=0
    for root in sorted(roots):
        if has_forbidden_folder(root.relative_to(base)): continue
        for f in sorted(root.rglob('*.py')):
            rel=f.relative_to(base)
            if has_forbidden_folder(rel) or f.is_symlink() or not f.is_file():continue
            if f.stat().st_size > 150000:continue
            inspected+=1
            if inspected>max_files:break
            src=f.read_text(encoding='utf8',errors='replace')
            low=src.lower()
            hits=[t for t in TOKENS if t in low]
            if len(hits)<2:continue
            item={"source_path":str(f), "relative_path":str(rel),
                  "sha256":hashlib.sha256(f.read_bytes()).hexdigest(),
                  "matched_tokens":hits, "size_bytes":f.stat().st_size,
                  "source_exported":False,"secret_pattern_skipped":bool(RISK_SECRET.search(src))}
            if not item['secret_pattern_skipped']:
                target=out/'candidate_source'/str(rel)
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():raise RuntimeError('duplicate export path')
                target.write_text(src,encoding='utf8')
                item['source_exported']=True
            selected.append(item)
        if inspected>max_files:break
    manifest={"schema":"prd32_qwen_real_entrypoint_discovery_v1",
              "roots_checked":[str(r) for r in roots],
              "files_inspected":min(inspected,max_files),
              "max_files":max_files,"selected":selected,
              "no_model_calls":True,"no_holdout_scan":True,
              "not_a_deployment":True}
    (out/'qwen_entrypoint_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=Path('/workspace'))
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise SystemExit('output exists; choose fresh directory')
    a.out.mkdir(parents=True)
    m=collect(a.workspace,a.out)
    print(json.dumps({"files_inspected":m['files_inspected'],"qwen_candidates":len(m['selected']),
                      "exported":sum(x['source_exported'] for x in m['selected']),
                      "results_dir":str(a.out)},ensure_ascii=False))
if __name__=='__main__':main()
