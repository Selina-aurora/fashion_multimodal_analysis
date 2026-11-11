#!/usr/bin/env python3
"""Static, read-only CUDA Qwen decoder configuration audit for pinned PRD32 DEV60.

This script NEVER imports the inspected runner or model, never loads model weights,
never generates text, never reads Holdout, and never alters any original file.
Its AST/config findings are evidence, not a claim about the effective dynamic kwargs.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import sys
import traceback
import zipfile

WORKSPACE = Path('/workspace')
RUNNER = WORKSPACE/'PRD32_DEV60_QWEN_AB_20261030/dev60_qwen_ab.py'
CORE = WORKSPACE/'PRD32_DEV60_QWEN_AB_20261030/reference_12img_runner_READ_ONLY.py'
MODEL_DIR = WORKSPACE/'models/Qwen2.5-VL-7B-Instruct'
EXPECTED = {
    'dev60_qwen_ab.py': '98a833c429da4c39849942a81f1067239273258a8605eac9265d0aeb9defc6f3',
    'reference_12img_runner_READ_ONLY.py': 'fdf1c55583b75f4692ddad07d459ceabff7a337d4f82d90e71a5c908f1323d2f',
}
SENSITIVE_KEYS = {'token', 'api_key', 'access_token', 'auth_token', 'password', 'secret', 'key'}
CONFIG_KEYS = ('use_cache', 'cache_implementation', 'do_sample', 'num_beams', 'top_p', 'top_k',
               'temperature', 'max_length', 'max_new_tokens', 'eos_token_id',
               'bos_token_id', 'pad_token_id', 'sliding_window', 'use_sliding_window',
               'rope_scaling', 'attn_implementation', '_attn_implementation')


def digest(path: Path) -> str:
    sha=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            sha.update(b)
    return sha.hexdigest()


def refuse_restricted(path: Path):
    s=str(path).casefold()
    if any(tok in s for tok in ('holdout','sealed','do_not_open','test_private','prod_secrets')):
        raise ValueError('refuse restricted path: '+str(path))


def static_calls(source: str, file_name: str):
    root=ast.parse(source,filename=file_name)
    lines=source.splitlines()
    out=[]
    functions=[]
    for node in ast.walk(root):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            if node.name in ('run_model_case','run_triplets','main','build_prompt'):
                functions.append({'function':node.name,'line':node.lineno})
        if not isinstance(node,ast.Call):continue
        func=node.func
        name=func.attr if isinstance(func,ast.Attribute) else (func.id if isinstance(func,ast.Name) else '?')
        if name not in {'generate','run_model_case','apply_chat_template','from_pretrained','processor'}:continue
        kw={}
        has_kwargs=False
        for k in node.keywords:
            if k.arg is None:
                has_kwargs=True
                continue
            if k.arg in {'use_cache','cache_implementation','max_new_tokens','num_beams','do_sample',
                         'temperature','top_p','top_k','attn_implementation',
                         'torch_dtype','device_map','max_pixels'}:
                try: v=ast.literal_eval(k.value)
                except (ValueError,TypeError,RecursionError):
                    v='DYNAMIC_EXPR'
                if isinstance(v,(str,int,float,bool)) or v is None:
                    kw[k.arg]=v
                else:
                    kw[k.arg]='COMPLEX_LITERAL'
        out.append({'file':file_name,'line':node.lineno,'call':name,
                    'selected_explicit_kwargs':kw,'has_asterisk_kwargs':has_kwargs,
                    'line_excerpt':lines[node.lineno-1].strip()[:180]})
    flags={'inference_mode_context':bool(re.search(r'\btorch\.inference_mode\b',source)),
           'no_grad_context':bool(re.search(r'\btorch\.no_grad\b',source)),
           'use_cache_literal':bool(re.search(r'\buse_cache\s*=',source)),
           'cache_implementation_literal':bool(re.search(r'\bcache_implementation\s*=',source)),
           'sdpa_literal':('sdpa' in source.lower()),
           'flash_attention_2_literal':('flash_attention_2' in source.lower())}
    return {'file':file_name,'interesting_functions':functions,
            'call_sites':out,'lexical_flags_not_proof_of_effective_runtime':flags}


def sanitized_config(path:Path):
    if not path.exists():return {'status':'NOT_PRESENT'}
    if path.stat().st_size>20_000_000: raise ValueError('unreasonably large config')
    content=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(content,dict):raise ValueError('config must be JSON object')
    found={k:content[k] for k in CONFIG_KEYS if k in content}
    # Don't emit arbitrary extra fields: model configs may have private metadata.
    nested={}
    for key in ('text_config','vision_config'):
        obj=content.get(key,{})
        if isinstance(obj,dict):
            nested[key]={k:obj[k] for k in CONFIG_KEYS if k in obj}
    return {'status':'READ_ONLY','filename':path.name,'sha256':digest(path),
            'safe_selected_fields':found,'safe_selected_nested_fields':nested}


def installed_versions():
    return {n: (metadata.version(n) if metadata.packages_distributions() is not None else None)
            for n in []}


def read_versions():
    v={}
    for p in ('torch','transformers','accelerate','tokenizers','qwen-vl-utils'):
        try:v[p]=metadata.version(p)
        except metadata.PackageNotFoundError:v[p]='NOT_INSTALLED'
    return v


def inspect(runner:Path,core:Path,model_dir:Path):
    for p in (runner,core,model_dir):refuse_restricted(p)
    ret={'schema':'prd32_qwen_decode_static_config_audit_v4','scope':'DEV60_STATIC_READ_ONLY',
         'executes_model':False,'model_generate_called':False,'modifies_pinned_files':False,
         'independent_holdout_read':False,'production_approved':False}
    sources=[]
    for p in (runner,core):
        if not p.is_file():raise FileNotFoundError(p)
        actual=digest(p)
        expected=EXPECTED[p.name]
        if actual!=expected:
            raise ValueError(f'SOURCE_HASH_MISMATCH {p.name}: expected={expected} actual={actual}')
        if p.stat().st_size>3_000_000:raise ValueError('source unexpectedly large')
        parsed=static_calls(p.read_text(encoding='utf-8'),p.name)
        sources.append({'sha256':actual,'analysis':parsed})
    ret['verified_sources']=sources
    if not model_dir.is_dir():raise FileNotFoundError(model_dir)
    ret['model_dir_basename']=model_dir.name
    ret['model_config']=sanitized_config(model_dir/'config.json')
    ret['generation_config']=sanitized_config(model_dir/'generation_config.json')
    ret['python_version']=sys.version.split()[0]
    ret['installed_versions']=read_versions()
    generate_calls=[c for x in sources for c in x['analysis']['call_sites'] if c['call']=='generate']
    ret['generate_calls_detected']=len(generate_calls)
    ret['explicit_cache_kwargs_in_generate_calls']=[c['selected_explicit_kwargs'] for c in generate_calls if any(k in c['selected_explicit_kwargs'] for k in ('use_cache','cache_implementation'))]
    ret['cache_effective_runtime_state']='UNKNOWN_STATIC_ONLY'
    ret['notes']=[
        'AST reports explicit visible kwargs in exactly two pinned source files; dynamic **kwargs and imported dependencies are not resolved.',
        'Config use_cache and cache_implementation defaults cannot establish actual effective runtime behavior.',
        'No GPU calls, no model initialization, no source edits, no holdout, no releases.',
        'Inspect source line locations and config defaults to choose exactly one subsequent controlled candidate.'
    ]
    return ret


def unique_out():
    return WORKSPACE/('PRD32_1110_DECODE_CONFIG_AUDIT_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))


def pack(out:Path):
    z=out.with_suffix('.zip')
    with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED) as f:
        for p in sorted(out.rglob('*')):
            if p.is_file():f.write(p,arcname=p.relative_to(out.parent).as_posix())
    return z


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner',type=Path,default=RUNNER)
    p.add_argument('--core',type=Path,default=CORE)
    p.add_argument('--model-dir',type=Path,default=MODEL_DIR)
    p.add_argument('--out',type=Path,default=None)
    a=p.parse_args(argv)
    out=a.out or unique_out()
    refuse_restricted(out)
    if not out.resolve().is_relative_to(WORKSPACE.resolve()):
        raise ValueError('output must be inside /workspace')
    out.mkdir(parents=True,exist_ok=False)
    code=0
    try:
        result=inspect(a.runner,a.core,a.model_dir)
        (out/'decode_static_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print('DECODE_CONFIG_STATIC_AUDIT_COMPLETE_READ_ONLY',flush=True)
    except Exception as e:
        code=2
        # No source texts/secrets are saved: failure paths are local file names.
        (out/'failure.json').write_text(json.dumps({'status':'BLOCKED_READ_ONLY','error_type':type(e).__name__,
            'message':str(e),'model_calls':0,'production_approved':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('DECODE_CONFIG_AUDIT_BLOCKED',type(e).__name__,str(e),file=sys.stderr,flush=True)
    finally:
        print('UPLOAD_THIS:',pack(out),flush=True)
    return code


if __name__=='__main__':sys.exit(main())
