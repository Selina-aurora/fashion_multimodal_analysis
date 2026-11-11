"""Read-only inventory of PRD3.2 GPU source entrypoints, without copying code or secrets."""
from __future__ import annotations
import argparse,ast,hashlib,json
from pathlib import Path

DEFAULT_DIRS=[
    '/workspace/PRD32_DEV60_V5_REAL_20261030_run01',
    '/workspace/PRD32_DEV60_QWEN_BASELINE_20261030_run01',
    '/workspace/PRD32_DEV60_QWEN_FOCUS_V1_20261030_run01',
    '/workspace/fashion_multimodal_analysis/src/fashion_multimodal_analysis/integration',
    '/workspace/fashion_multimodal_analysis/scripts',
]
SKIP_DIR={'holdout','blind','sealed','archive','.venv','venv','__pycache__','.git'}

def inspect(roots:list[Path], max_files:int=400)->dict:
    found=[]
    for base in roots:
        if not base.is_dir():continue
        for path in base.rglob('*.py'):
            if any(any(x in part.lower() for x in SKIP_DIR) for part in path.parts):continue
            if len(found)>=max_files:raise RuntimeError('Too many candidate sources; tighten paths')
            if path.stat().st_size>256000:continue
            content=path.read_bytes()
            try:
                tree=ast.parse(content.decode('utf-8-sig'))
                funcs=[node.name for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
            except (SyntaxError,UnicodeError):
                funcs=['[AST_UNAVAILABLE]']
            found.append({'path':str(path),'size_bytes':len(content),'sha256':hashlib.sha256(content).hexdigest(),'top_level_symbols':funcs[:25]})
    return {'schema':'prd32_gpu_runtime_entrypoint_inventory_v1','source_files':len(found),
            'source_code_exported':False,'secret_files_accessed':False,
            'holdout_read':False,'files':sorted(found,key=lambda x:x['path'])}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('/workspace/PRD32_1106_PIPELINE_SHADOW_RESULTS/runtime_entrypoint_inventory.json'))
    args=ap.parse_args()
    report=inspect([Path(p) for p in DEFAULT_DIRS]);args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'INVENTORY_OK: {report["source_files"]} source files; metadata only; no source content exported')
