#!/usr/bin/env python3
"""PRD 3.2 DEV60: read-only AST traceback of actual Qwen generate helper.

No runner imports, model initialization, inference, prompts, or holdout data.
Refuses pinned-source changes and scans only bounded DEV60 code dirs.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import sys
import zipfile

WS=Path('/workspace')
PINNED=WS/'PRD32_DEV60_QWEN_AB_20261030'
EXPECTED={
    'dev60_qwen_ab.py':'98a833c429da4c39849942a81f1067239273258a8605eac9265d0aeb9defc6f3',
    'reference_12img_runner_READ_ONLY.py':'fdf1c55583b75f4692ddad07d459ceabff7a337d4f82d90e71a5c908f1323d2f'
}
SAFE_DIRS=(
    'PRD32_DEV60_QWEN_AB_20261030',
    'PRD32_DEV60_QWEN_BASELINE_20261030_run01',
    'PRD32_1106_LIVE_QWEN_SHADOW_V4',
    'PRD32_1106_DEV12_RUNTIME_V7',
)
CALL_KEYS={'do_sample','temperature','top_p','top_k','num_beams','use_cache','cache_implementation',
           'max_new_tokens','max_length','eos_token_id','pad_token_id','return_dict_in_generate',
           'output_scores','repetition_penalty','typical_p','min_new_tokens','synced_gpus'}
RESTRICTED=('holdout','sealed','do_not_open','test_private','prod_secrets')

def digest(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def safe_path(p:Path):
    rp=p.resolve()
    if not rp.is_relative_to(WS.resolve()):raise ValueError('Outside /workspace')
    if any(x in str(rp).lower() for x in RESTRICTED):raise ValueError('Restricted path')
    return rp

def kw_value(node):
    try:value=ast.literal_eval(node)
    except (ValueError,TypeError,RecursionError):return 'DYNAMIC_EXPRESSION_NOT_RESOLVED'
    if isinstance(value,(bool,int,float,str)) or value is None:
        if isinstance(value,str) and len(value)>64:return 'STRING_REDACTED'
        return value
    return 'COMPLEX_VALUE_NOT_RESOLVED'

def trace_source(src:str,relative:str):
    tree=ast.parse(src,filename=relative)
    defs=[]; generates=[]; imports=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ('run_model_case','prompt_check'):
            args=[a.arg for a in node.args.posonlyargs+node.args.args+node.args.kwonlyargs]
            defs.append({'name':node.name,'line':node.lineno,'parameters':args[:20]})
        if isinstance(node,ast.Call):
            fn=node.func
            func_name=fn.attr if isinstance(fn,ast.Attribute) else (fn.id if isinstance(fn,ast.Name) else '')
            if func_name not in ('generate','run_model_case','GenerationConfig'):continue
            kws={};unpacked=False
            for kw in node.keywords:
                if kw.arg is None:unpacked=True
                elif kw.arg in CALL_KEYS:kws[kw.arg]=kw_value(kw.value)
            generates.append({'call':func_name,'line':node.lineno,'selected_kwargs':kws,'has_unpacked_kwargs':unpacked})
        if isinstance(node,ast.ImportFrom):
            mod=node.module or ''
            if mod and len(mod)<100 and all(x not in mod.casefold() for x in ('secret','token','key')):
                imports.append({'module':mod,'line':node.lineno})
    # AST analysis only; keyword flags do not establish effective runtime behavior.
    return {'relative_file':relative,'definitions':defs,'calls':generates,'imports':imports[:30]}

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args(argv)
    out=args.output or WS/('PRD32_1110_DECODE_HELPER_TRACE_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
    safe_path(out);out.mkdir(parents=True,exist_ok=False)
    code=0
    try:
        verified=[]
        for name,sha in EXPECTED.items():
            p=safe_path(PINNED/name)
            if not p.is_file():raise FileNotFoundError(str(p))
            got=digest(p)
            if got!=sha:raise RuntimeError('PINNED_SOURCE_HASH_MISMATCH:'+name)
            verified.append({'file':name,'sha256':got})
        scanned=[]; skip=[]; candidates=[];count=0
        for folder in SAFE_DIRS:
            root=safe_path(WS/folder)
            if not root.is_dir():continue
            for p in sorted(root.glob('**/*.py')):
                if any(part in {'venv','.venv','site-packages','__pycache__','.git'} for part in p.parts):continue
                if p.is_symlink():continue
                p=safe_path(p)
                if not p.is_file():continue
                count+=1
                if count>350:raise RuntimeError('SOURCE_SCAN_FILE_LIMIT_350')
                if p.stat().st_size>1_000_000:
                    skip.append({'file':p.name,'reason':'MAX_SOURCE_BYTES'});continue
                rel=str(p.relative_to(WS))
                try:a=trace_source(p.read_text('utf-8'),rel)
                except (UnicodeError,SyntaxError):
                    skip.append({'file':p.name,'reason':'NOT_PARSEABLE_PYTHON'});continue
                if any(d['name']=='run_model_case' for d in a['definitions']) or any(c['call']=='generate' for c in a['calls']):
                    candidates.append({'file':rel,'sha256':digest(p),'analysis':a})
                if p.name in EXPECTED:scanned.append({'file':rel,'analysis':a})
        report={'schema':'prd32_decode_helper_trace_v5','status':'STATIC_SCAN_COMPLETED',
                'read_only':True,'model_calls':0,'holdout_accessed':False,'release_allowed':False,
                'pinned_sources_verified':verified,'directories_scanned':[x for x in SAFE_DIRS if (WS/x).is_dir()],
                'num_python_files_scanned':count,'helper_and_generate_candidates':candidates,
                'pinned_entry_call_sites':scanned,'skipped':skip,
                'interpretation':'A visible generate call does not prove effective runtime kwargs. No model was imported or executed. If no helper found, inspect the import chain outside these bounded DEV60 directories.'}
        (out/'decode_helper_trace.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        print('DECODE_HELPER_STATIC_TRACE_COMPLETE_READ_ONLY candidates='+str(len(candidates)))
    except Exception as e:
        code=2
        (out/'failure.json').write_text(json.dumps({'status':'STATIC_TRACE_BLOCKED','error_type':type(e).__name__,'message':str(e),'model_calls':0,'holdout_accessed':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('DECODE_HELPER_STATIC_TRACE_BLOCKED '+type(e).__name__+': '+str(e),file=sys.stderr)
    z=out.with_suffix('.zip')
    with zipfile.ZipFile(z,'x',zipfile.ZIP_DEFLATED) as f:
        for item in out.iterdir():f.write(item,item.relative_to(out.parent).as_posix())
    print('UPLOAD_THIS:',z)
    return code

if __name__=='__main__':sys.exit(main())
