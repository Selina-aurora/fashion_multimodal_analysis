#!/usr/bin/env python3
"""DEV60-only, SHA-256-checked collection of Qwen marked images for V4 priority12.

No inference, no GT writes, no modifications of original results or holdout.
"""
import argparse
import csv
import hashlib
import html
import json
import shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

DEFAULT_ALLOWED={
    'baseline':Path('/workspace/PRD32_DEV60_QWEN_BASELINE_20261030_run01/marked_inputs'),
    'focus_v1':Path('/workspace/PRD32_DEV60_QWEN_FOCUS_V1_20261030_run01/marked_inputs'),
}
DEFAULT_VISUAL=Path('/workspace/PRD32_1106_V4_VISUAL_PRE_REVIEW')
FORBIDDEN=('holdout','sealed','blind','independent_test')

def sha256_file(path):
    dig=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            dig.update(chunk)
    return dig.hexdigest()

def strict_source(raw_path,allowed_dir):
    p=Path(raw_path).resolve(strict=True)
    root=Path(allowed_dir).resolve(strict=True)
    if not p.is_file() or not p.is_relative_to(root):
        raise ValueError('Not a file within approved DEV marked_inputs directory')
    if p.suffix.lower() not in ('.jpg','.jpeg','.png','.webp'):
        raise ValueError('Not an approved image file type')
    if any(k in str(p).lower() for k in FORBIDDEN):
        raise ValueError('Disallowed Holdout/Blind/Sealed path')
    return p

def try_copy_verified(src,dst,expected):
    actual=sha256_file(src)
    if actual.lower()!=expected.lower():
        raise ValueError('SHA256 mismatch: '+str(src))
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
    if sha256_file(dst)!=actual:
        raise ValueError('Copied file changed: '+str(dst))
    return actual

def make_html(manifest, statuses, outdir):
    rows=[]
    for row in manifest:
        key=(row['case_id'],row['instance_id'])
        case,inst=key
        label=html.escape(case+' / '+inst)
        tag0=html.escape(str(row['variants']['baseline']['category']))
        tag1=html.escape(str(row['variants']['focus_v1']['category']))
        v5=html.escape(str(row['v5_category']))
        cap=html.escape(row['visual_assessment_provisional_not_gt'])
        cells=[]
        for variant in ('baseline','focus_v1'):
            target=Path('marked')/variant/(case+'__'+inst+'.jpg')
            pic=f'<img src="{html.escape(target.as_posix())}" alt="{html.escape(variant)} marked image" />' if (outdir/target).exists() else '<p>MISSING / NOT VERIFIED</p>'
            cells.append('<div><h4>'+html.escape(variant)+'</h4>'+pic+'<p><b>Tag:</b> '+html.escape(str(row['variants'][variant]['category']))+'</p><p><b>Copy:</b> '+html.escape(row['variants'][variant]['copy_response'] or '')+'</p><p><b>VQA:</b> '+html.escape(row['variants'][variant]['vqa_response'] or '')+'</p></div>')
        original=Path('originals')/(case+'.jpg')
        overlay=Path('overlays')/(case+'_'+inst+'_overlay.jpg')
        context=[]
        if (outdir/original).exists(): context.append('<img src="'+original.as_posix()+'" alt="verified original" />')
        if (outdir/overlay).exists():context.append('<img src="'+overlay.as_posix()+'" alt="existing overlay" />')
        rows.append(f'<article><h2>{label} — V5 {v5}, baseline {tag0}, focus {tag1}</h2><p><b>Visual pre-review (NOT GT):</b> {cap}</p><div class="images">{''.join(context)}</div><div class="cols">{"".join(cells)}</div><p>Identity verification: PENDING · Human GT: PENDING · Mainline approval: NO</p></article>')
    h='''<!doctype html><html lang="zh"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>DEV60 Qwen priority12 marked comparison</title><style>body{max-width:1350px;margin:auto;padding:22px;font:15px/1.5 system-ui,Arial,sans-serif;color:#26344b;background:#f7f9fc}article{background:white;border:1px solid #dce3ef;border-radius:12px;padding:16px;margin:20px 0}h1{font-size:24px}h2{font-size:19px}.cols{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.cols>div{padding:12px;background:#f1f5fa;border-radius:9px}.images{display:flex;gap:12px;justify-content:center;flex-wrap:wrap}.images img{max-height:300px;max-width:45%;object-fit:contain}.cols img{max-width:100%;max-height:460px;display:block;margin:auto}p{overflow-wrap:anywhere}small{color:#556} @media(max-width:720px){.cols{grid-template-columns:1fr}.images img{max-width:100%}}</style></head><body><h1>Guard V4 / Qwen 标记图验证（12 目标）</h1><p><b>DEV-only；所有显示分类都是未核验预测。</b> 同一目标的 Baseline、Focus V1 红框图与已有 Mask 可视化并排。不可当作人工 GT 或部署证据。</p>'''
    (outdir/'review_marked12.html').write_text(h+'\n'.join(rows)+'</body></html>',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='/workspace/PRD32_1106_V4_QWEN_MARKED12_EVIDENCE')
    parser.add_argument('--visual-root',default=str(DEFAULT_VISUAL))
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'priority12_marked_manifest.json').read_text(encoding='utf-8'))
    if len(manifest)!=12 or len({(x['case_id'],x['instance_id']) for x in manifest})!=12:
        raise SystemExit('Invalid priority12 manifest')
    dst=Path(args.out).resolve()
    if dst==Path('/workspace') or dst in DEFAULT_ALLOWED.values():
        raise SystemExit('Unsafe output location')
    if any(k in str(dst).lower() for k in FORBIDDEN):
        raise SystemExit('Output cannot be in holdout/blind/sealed directory')
    dst.mkdir(parents=True,exist_ok=True)
    visual=Path(args.visual_root)
    statuses=[]
    cases_original={}
    for row in manifest:
        case,inst=row['case_id'],row['instance_id']
        prior=cases_original.setdefault(case,row['image_sha256'])
        if prior!=row['image_sha256']:raise SystemExit('Conflicting expected original SHA for '+case)
        for variant in ('baseline','focus_v1'):
            srcrow=row['variants'][variant]
            local=dst/'marked'/variant/(case+'__'+inst+'.jpg')
            status={'case_id':case,'instance_id':inst,'variant':variant,
                    'source_path':srcrow['marked_image_path'],
                    'expected_sha256':srcrow['marked_image_sha256'],
                    'status':'UNVERIFIED','copied_path':local.relative_to(dst).as_posix()}
            try:
                source=strict_source(srcrow['marked_image_path'],DEFAULT_ALLOWED[variant])
                status['actual_sha256']=try_copy_verified(source,local,status['expected_sha256'])
                status['status']='SHA256_VERIFIED'
            except (FileNotFoundError,ValueError,OSError) as e:
                status['error']=f'{type(e).__name__}: {e}'
            statuses.append(status)
    originals_verified=0
    for case,expected in cases_original.items():
        source=visual/'originals'/(case+'.jpg')
        if not source.is_file():continue
        try:
            if any(k in str(source.resolve()).lower() for k in FORBIDDEN):continue
            try_copy_verified(source,dst/'originals'/(case+'.jpg'),expected)
            originals_verified+=1
        except (FileNotFoundError,ValueError,OSError):pass
    overlays_copied=0
    for row in manifest:
        case,inst=row['case_id'],row['instance_id']
        s=visual/'overlays'/(case+'_'+inst+'_overlay.jpg')
        if s.is_file() and s.resolve().is_relative_to(visual.resolve()) and not any(k in str(s).lower() for k in FORBIDDEN):
            o=dst/'overlays'/s.name;o.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s,o)
            overlays_copied+=1
    with (dst/'marked_sha256_verification.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['case_id','instance_id','variant','source_path','expected_sha256','actual_sha256','status','copied_path','error']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(statuses)
    verified=sum(s['status']=='SHA256_VERIFIED' for s in statuses)
    summary={'schema':'prd32_dev60_qwen_marked_priority12_v1','marked_verified':verified,
             'marked_expected':len(statuses),'distinct_target_instances':len(manifest),
             'originals_sha256_verified':originals_verified,'overlays_copied':overlays_copied,
             'holdout_not_scanned':True,'gt_verified':False,'publish_allowed':False,
             'status':'EVIDENCE_COLLECTED_NOT_GT' if verified==len(statuses) else 'BLOCKED_INCOMPLETE_EVIDENCE'}
    (dst/'evidence_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    make_html(manifest,statuses,dst)
    (dst/'priority12_marked_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    zname=dst.with_suffix('.zip')
    with ZipFile(zname,'w',compression=ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(dst.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(dst))
    summary['result_zip']=str(zname)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if verified<len(statuses): raise SystemExit('Evidence incomplete: see marked_sha256_verification.csv; do not approve GT')

if __name__=='__main__':main()
