"""Read-only, offline discovery of local snapshots of the EXACT three pinned HF models.

Never downloads weights or silently substitutes a different checkpoint.
"""
from __future__ import annotations
import os
from pathlib import Path

MODEL_IDS = {
    'grounding_model': 'IDEA-Research/grounding-dino-tiny',
    'sam_model': 'facebook/sam-vit-base',
    'clip_model': 'openai/clip-vit-base-patch32',
}
WEIGHTS = ('model.safetensors', 'pytorch_model.bin', 'model.safetensors.index.json', 'pytorch_model.bin.index.json')

def hub_roots():
    hints = [os.environ.get('HF_HUB_CACHE'), os.environ.get('HUGGINGFACE_HUB_CACHE'),
             os.environ.get('TRANSFORMERS_CACHE')]
    for home in [os.environ.get('HF_HOME'), '/workspace/hf_cache',
                 str(Path.home()/'.cache/huggingface'), '/workspace/.cache/huggingface',
                 '/workspace/models/huggingface', '/workspace/models/hf_cache']:
        if home: hints.extend((str(Path(home)/'hub'),home))
    seen=set(); roots=[]
    for raw in hints:
        if not raw: continue
        path=Path(raw).expanduser()
        if str(path) not in seen:
            roots.append(path);seen.add(str(path))
    return roots

def has_weights(folder: Path):
    return any((folder/n).is_file() for n in WEIGHTS)

def snapshot_report(folder: Path):
    return {'path':str(folder),'config_json':(folder/'config.json').is_file(),
            'processor_json':(folder/'preprocessor_config.json').is_file(),
            'weights':has_weights(folder),
            'tokenizer':any((folder/n).is_file() for n in ('tokenizer.json','tokenizer_config.json','vocab.txt'))}

def discover(cfg):
    """Return exact model-id -> usable local snapshot paths plus diagnostic audit.

    Missing snapshots are documented and should prevent a model run; dry-run
    can be called only to validate data, not to assert model assets are ready.
    """
    roots=hub_roots()
    audit={'search_hub_roots':[str(r) for r in roots], 'models':{}, 'all_model_assets_found':True}
    resolved={}
    for key,model_id in MODEL_IDS.items():
        if cfg.get(key)!=model_id:
            raise ValueError(f'Unexpected pinned model {key}={cfg.get(key)!r} (expected {model_id!r})')
        dirname='models--'+model_id.replace('/','--')
        candidates=[]
        for root in roots:
            # Hugging Face hub layout. Check snapshots only; do not treat a
            # cached ref pointer or incomplete download as model directory.
            d=root/dirname/'snapshots'
            if d.is_dir(): candidates.extend(sorted(p for p in d.iterdir() if p.is_dir()))
        # Also support a deliberately mounted local copy of the exact model.
        env_override=os.environ.get('PRD31_'+key.upper()+'_DIR')
        if env_override:
            candidates.insert(0,Path(env_override).expanduser())
        audit_candidates=[snapshot_report(c) for c in candidates]
        # Basic presence requirements. Full processor compatibility is
        # validated below via transformers from_pretrained on selected path.
        usable=[c for c in candidates if (c/'config.json').is_file() and
                (c/'preprocessor_config.json').is_file() and has_weights(c)]
        if key in ('grounding_model','clip_model'):
            usable=[c for c in usable if snapshot_report(c)['tokenizer']]
        path=next(iter(usable),None)
        if path:
            resolved[key]=str(path)
        else:
            audit['all_model_assets_found']=False
        audit['models'][key]={'expected_model_id':model_id, 'resolved_snapshot':str(path) if path else None,
                              'found_candidates':audit_candidates[:30]}
    return resolved,audit

def verify_processors(resolved):
    """GPU-environment-only processor check, local paths, no networking."""
    from transformers import AutoProcessor,SamProcessor,CLIPProcessor
    # Names correspond precisely to ModelBackends constructor.
    loaders={'grounding_model':AutoProcessor, 'sam_model':SamProcessor,
             'clip_model':CLIPProcessor}
    result={}
    for key,loader in loaders.items():
        try:
            obj=loader.from_pretrained(resolved[key],local_files_only=True)
            result[key]={'processor_load':'ok','class':type(obj).__name__}
            del obj
        except Exception as e:
            result[key]={'processor_load':'failed','error':str(e)}
    return result
