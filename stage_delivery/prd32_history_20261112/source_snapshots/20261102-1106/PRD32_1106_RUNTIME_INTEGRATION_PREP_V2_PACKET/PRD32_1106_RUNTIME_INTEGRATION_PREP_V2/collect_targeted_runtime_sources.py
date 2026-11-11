"""Collect ONLY two GPU PRD3.2 integration sources for exact patch planning.

Does not walk the workspace, inspect images, read the Holdout, or run models.
By default the source text is collected if the static secrets guard finds no
obvious credential literal. Review copied sources yourself before sharing.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import re
import shutil
from pathlib import Path

SOURCES={
    'prd32_output_guard.py': 'src/fashion_multimodal_analysis/integration/prd32_output_guard.py',
    'prd32_guard_service_outputs.py':'scripts/prd32_guard_service_outputs.py',
}
SUSPECTS=[
    re.compile(r'(?i)\b(?:api_?key|secret|password|access_?token|authorization)\s*[=:]\s*[\'\"][^\'\"]{7,}[\'\"]'),
    re.compile(r'\b(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{12,})\b'),
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
]


def signature(item):
    try: return ast.unparse(item.args)
    except Exception: return ''


def collect(repo: Path,out: Path):
    out.mkdir(parents=True,exist_ok=True)
    raw=out/'verified_source_files';raw.mkdir(exist_ok=True)
    results=[]
    for name,rel in SOURCES.items():
        p=repo/rel
        if not p.is_file():
            results.append({'path':str(p),'status':'MISSING'})
            continue
        content=p.read_bytes()
        digest=hashlib.sha256(content).hexdigest()
        entry={'path':str(p),'size_bytes':len(content),'sha256':digest,'status':'READ_ONLY_INSPECTED'}
        try:
            txt=content.decode('utf-8')
            tree=ast.parse(txt,filename=rel)
            entry['top_level']=[{'name':n.name,'kind':type(n).__name__,
                                 'args':signature(n) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) else '',
                                 'line':getattr(n,'lineno',0)}
                                for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
            entry['function_calls']=sorted({n.func.id if isinstance(n.func,ast.Name) else n.func.attr
                                            for n in ast.walk(tree) if isinstance(n,ast.Call)
                                            if isinstance(n.func,(ast.Name,ast.Attribute))})[:200]
            entry['secret_pattern_detected']=any(x.search(txt) for x in SUSPECTS)
            if entry['secret_pattern_detected']:
                entry['status']='SENSITIVE_LITERAL_DETECTED_SOURCE_NOT_EXPORTED'
            else:
                shutil.copyfile(p,raw/name)
                entry['status']='SOURCE_EXPORTED_FOR_REVIEW'
        except (UnicodeError,SyntaxError) as exc:
            entry['status']='SOURCE_PARSE_FAILED'
            entry['parse_error']=type(exc).__name__
        results.append(entry)
    (out/'runtime_source_manifest.json').write_text(json.dumps({'schema':'prd32_gpu_targeted_source_handoff_v1',
                'no_model_calls':True,'no_holdout_scan':True,'files':results},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='/workspace/fashion_multimodal_analysis');p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(collect(Path(a.repo),Path(a.out)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
