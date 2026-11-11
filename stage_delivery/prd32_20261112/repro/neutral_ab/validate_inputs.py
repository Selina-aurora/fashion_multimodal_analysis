#!/usr/bin/env python3
"""Fail fast if the five strict A/B pairs lose their pixel-controlled geometry."""
import csv, hashlib
from pathlib import Path
import numpy as np
from PIL import Image

root=Path(__file__).resolve().parent
with (root/'manifest.csv').open(encoding='utf-8-sig',newline='') as f: records=list(csv.DictReader(f))
assert len(records)==5, f'Expected five pairs, got {len(records)}'
for c in records:
    assert '红框' not in c['prompt'] and '青框' not in c['prompt'], c['case_id']
    files=[root/c[f'{arm}_image'] for arm in ('red','cyan')]
    for arm,p in zip(('red','cyan'),files):
        assert p.is_file(),p
        assert hashlib.sha256(p.read_bytes()).hexdigest()==c[f'{arm}_sha256'],p
    a=np.asarray(Image.open(files[0]).convert('RGB'),dtype=np.int16)
    b=np.asarray(Image.open(files[1]).convert('RGB'),dtype=np.int16)
    assert a.shape==b.shape
    assert a.shape[1]==int(c['width']) and a.shape[0]==int(c['height'])
    change=np.any(a!=b,axis=2)
    assert int(change.sum())==int(c['changed_marker_pixels'])
    yy,xx=np.where(change)
    x0,y0,x1,y1=map(int,c['bbox_xyxy'].split(','))
    assert ((xx>=x0-8)&(xx<=x1+8)&(yy>=y0-8)&(yy<=y1+8)).all(),c['case_id']
    assert ((abs(xx-x0)<=8)|(abs(xx-x1)<=8)|(abs(yy-y0)<=8)|(abs(yy-y1)<=8)).all(),c['case_id']
    assert (a[change,0]>a[change,1]).all() and (b[change,1]>b[change,0]).all(),c['case_id']
    print('PASS',c['case_id'],c['target'],'changed-marker-pixels=',int(change.sum()))
print('PASS: 5/5 pairs, prompt neutral, source hashes verified, changed pixels limited to color marker perimeter')
