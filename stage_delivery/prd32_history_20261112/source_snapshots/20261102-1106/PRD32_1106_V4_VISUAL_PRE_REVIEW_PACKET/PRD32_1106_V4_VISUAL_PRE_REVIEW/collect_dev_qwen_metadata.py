#!/usr/bin/env python3
"""Read-only DEV60 Qwen metadata discovery. Does not run inference or touch independent holdout."""
from pathlib import Path
import csv, json, re, zipfile, sys

here=Path(__file__).resolve().parent
with (here/'priority12_source.csv').open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.DictReader(f))
cases={r['case_id'] for r in rows}
digests={r['image_sha256'].lower() for r in rows}
fragments=sorted(cases|digests|{d[:16] for d in digests})
roots=[Path('/workspace/PRD32_DEV60_QWEN_BASELINE_20261030_run01'),Path('/workspace/PRD32_DEV60_QWEN_FOCUS_V1_20261030_run01'),Path('/workspace/PRD32_DEV60_QWEN_AB_COMPARE_20261030_run01')]
output=Path('/workspace/PRD32_1106_V4_QWEN_DEV_EVIDENCE.zip')
max_files=4000;max_bytes=15_000_000;count=0;lines=[];candidates=[]
for root in roots:
    if not root.is_dir(): continue
    if 'DEV60' not in root.name or 'QWEN' not in root.name: raise SystemExit('Refuse non-DEV60 QWEN root')
    for p in root.rglob('*'):
        if not p.is_file(): continue
        if any(s in str(p).lower() for s in ('holdout','sealed','blind','independent_test')): continue
        if p.suffix.lower() not in ('.json','.jsonl','.csv','.txt','.md'): continue
        if p.stat().st_size>max_bytes: continue
        count+=1
        if count>max_files: raise SystemExit('Too many files, stop without archive')
        candidates.append(f'{p} ({p.stat().st_size} bytes)')
        with p.open('r',encoding='utf-8',errors='replace') as f:
            for i,line in enumerate(f,1):
                l=line.lower()
                hits=[x for x in fragments if x in l]
                if not hits: continue
                # Record full relevant line up to 24000 characters; never read credentials from env.
                lines.append({'source_path':str(p),'line_number':i,'hits':hits,'text':line[:24000]})
print('DEV Qwen files scanned:',count,'matched lines:',len(lines))
with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as z:
    z.writestr('matched_qwen_dev_lines.jsonl',''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in lines))
    z.writestr('scanned_dev_files.txt','\n'.join(candidates)+'\n')
    z.writestr('README.txt','DEV60-only metadata excerpts. No original image/ROI files copied. Match needs manual verification; tokens or personal data may be present in source text and must be reviewed before external sharing.\n')
print('OUTPUT:',output)
