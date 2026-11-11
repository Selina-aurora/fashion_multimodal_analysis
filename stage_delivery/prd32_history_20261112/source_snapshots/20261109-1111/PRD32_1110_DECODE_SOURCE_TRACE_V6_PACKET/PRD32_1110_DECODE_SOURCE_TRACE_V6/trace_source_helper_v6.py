#!/usr/bin/env python3
"""Read-only, hash-pinned AST trace of the Qwen model helper source.

No importing target code; no model calls, image access, Holdout access,
secrets, prompt text, or source-code export. JSON results only.
"""
from __future__ import annotations
import ast
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

WS = Path('/workspace')
FILES = {
 'PRD32_DEV60_QWEN_AB_20261030/dev60_qwen_ab.py': '98a833c429da4c39849942a81f1067239273258a8605eac9265d0aeb9defc6f3',
 'PRD32_DEV60_QWEN_AB_20261030/reference_12img_runner_READ_ONLY.py': 'fdf1c55583b75f4692ddad07d459ceabff7a337d4f82d90e71a5c908f1323d2f',
 'PRD32_20261029_NextBatch_GTFree/v5_to_qwen_bridge.py': 'affee7342646d808b4f0214f135d60d2d2611db920655c5e064268092e2dd02d',
 'PRD32_20261029_NextBatch_GTFree/next_batch.py': '08b35135223b49e2a2a7bbb448cdad3a4ce90a6a1373e57af29d1119102489f6',
 'PRD32_20261029_NextBatch_GTFree/frozen_helpers/prd32_integrated_pipeline_20261027.py': '90dda6974dc533aaae30fe1343095864ee74d536713ad80ce2006617286fa6b4',
}
INTEREST = {'run_model_case', 'load_helper', 'generate', 'GenerationConfig', 'from_pretrained', 'import_file', 'spec_from_file_location', 'load_runtime'}
KWARGS = {'do_sample','temperature','top_p','top_k','num_beams','use_cache','cache_implementation','max_new_tokens','max_length','eos_token_id','pad_token_id','repetition_penalty','generation_config','output_scores','return_dict_in_generate','task_limits'}

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def guarded_path(relative,root=WS):
    p = root/relative
    if p.is_symlink() or any(x.is_symlink() for x in list(p.parents) if x != root and x.is_relative_to(root)):
        raise ValueError('SYMLINK_SOURCE_BLOCKED')
    resolved = p.resolve()
    if not resolved.is_relative_to(root.resolve()): raise ValueError('PATH_OUTSIDE_WORKSPACE')
    if any(term in str(resolved).lower() for term in ('holdout','sealed','do_not_open','secret')):
        raise ValueError('RESTRICTED_SOURCE')
    return resolved

def literal_value(node):
    # This helper only emits primitive decoder configurations, not raw prompt contents.
    try: v = ast.literal_eval(node)
    except (TypeError, ValueError, RecursionError): return 'DYNAMIC_NOT_RESOLVED'
    if isinstance(v,bool) or v is None: return v
    if isinstance(v,(int,float)): return v
    if isinstance(v,str) and len(v) <= 40 and not any(t in v.lower() for t in ('token','secret','password','prompt')): return v
    return 'COMPLEX_OR_REDACTED'

def name_of(fn):
    if isinstance(fn,ast.Name): return fn.id
    if isinstance(fn,ast.Attribute): return fn.attr
    return None

def analyze(src:str,relative:str):
    tree=ast.parse(src,filename=relative)
    definitions=[];calls=[];imports=[]
    for n in ast.walk(tree):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in ('run_model_case','load_helper','load_runtime'):
            params=[a.arg for a in (n.args.posonlyargs+n.args.args+n.args.kwonlyargs)]
            defaults={}
            last=n.args.args
            for k,v in zip(last[-len(n.args.defaults):] if n.args.defaults else [], n.args.defaults): defaults[k.arg]=literal_value(v)
            definitions.append({'name':n.name,'line':n.lineno,'arguments':params[:20],'literal_defaults':defaults})
        if isinstance(n,ast.Call):
            fn=name_of(n.func)
            if fn not in INTEREST:continue
            kw={};spread=False
            for k in n.keywords:
                if k.arg is None:spread=True
                elif k.arg in KWARGS:kw[k.arg]=literal_value(k.value)
            calls.append({'function':fn,'line':n.lineno,'literal_decode_kwargs':kw,'unpacked_kwargs_present':spread,'positional_count':len(n.args)})
        if isinstance(n,ast.ImportFrom) and n.module is not None:
            if 'secret' not in n.module.lower():imports.append({'module':n.module[:90],'line':n.lineno})
        if isinstance(n,ast.Import):
            for a in n.names:
                if 'secret' not in a.name.lower():imports.append({'module':a.name[:90],'line':n.lineno})
    return {'file':relative,'definitions':sorted(definitions,key=lambda x:x['line']), 'interesting_calls':sorted(calls,key=lambda x:x['line']), 'imports':sorted(imports,key=lambda x:x['line'])[:40]}

def main():
    out=WS/('PRD32_1110_DECODE_SOURCE_TRACE_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
    out.mkdir(parents=True,exist_ok=False)
    try:
        verified=[];scans=[]
        for rel,expected in FILES.items():
            p=guarded_path(rel)
            if not p.is_file(): raise FileNotFoundError('PINNED_SOURCE_MISSING:'+rel)
            if p.stat().st_size>1_000_000: raise ValueError('UNEXPECTED_SOURCE_SIZE:'+rel)
            actual=sha(p)
            if actual!=expected:raise ValueError('PINNED_SOURCE_SHA256_MISMATCH:'+rel)
            verified.append({'file':rel,'sha256':actual})
            scans.append(analyze(p.read_text(encoding='utf-8'),rel))
        definitions=[(s['file'],d) for s in scans for d in s['definitions'] if d['name']=='run_model_case']
        generate=[(s['file'],c) for s in scans for c in s['interesting_calls'] if c['function']=='generate']
        report={'schema':'prd32_decode_source_trace_v6','status':'SOURCE_TRACE_FOUND' if definitions and generate else 'SOURCE_TRACE_PARTIAL',
                'read_only':True,'model_calls':0,'holdout_accessed':False,'release_allowed':False,
                'source_verified':verified,'source_analysis':scans,
                'run_model_case_definition_count':len(definitions),'generate_call_count':len(generate),
                'effective_runtime_decoding_confirmed':False,
                'interpretation':'AST only; static source does not prove effective decoding settings, KV cache semantics, or speed.'}
        (out/'source_helper_trace.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print('DECODE_SOURCE_TRACE_V6_'+report['status'], 'definitions='+str(len(definitions)), 'generate_calls='+str(len(generate)))
        ret=0
    except Exception as e:
        ret=2
        (out/'failure.json').write_text(json.dumps({'status':'STATIC_TRACE_BLOCKED','error_type':type(e).__name__,'message':str(e),'read_only':True,'model_calls':0,'holdout_accessed':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('DECODE_SOURCE_TRACE_V6_BLOCKED',type(e).__name__,str(e),file=sys.stderr)
    zip_path=out.with_suffix('.zip')
    with zipfile.ZipFile(zip_path,'x',zipfile.ZIP_DEFLATED) as z:
        for f in out.iterdir():z.write(f,f.relative_to(out.parent).as_posix())
    print('UPLOAD_THIS:',zip_path)
    return ret

if __name__=='__main__':sys.exit(main())
