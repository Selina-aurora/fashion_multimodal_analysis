#!/usr/bin/env python3
"""PRD 3.2 DEV60: collect read-only evidence for Guard V4 priority-12 review.

This is NOT a model evaluator. Requires visual GT and source Qwen responses for
identity conclusions. Never scans a directory containing holdout/sealed data.
Standard-library collector; optional Pillow overlay generator.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
EXT = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
FORBIDDEN = {'holdout', 'sealed', 'blind', 'independent_test', 'testset', 'test_set', 'eval_test', 'secret'}
SKIP_SCAN = {'.git', '.cache', 'node_modules', '__pycache__', 'site-packages', 'models', 'weights', 'checkpoints', 'venv', '.venv'}
CATS = ('top', 'pants', 'skirt', 'outerwear', 'dress', 'shoe', 'bag', 'accessory', 'UNCERTAIN')


def is_forbidden_path(p):
    s=str(p).lower().replace('\\','/')
    parts=re.split(r'[/_\-.]+',s)
    return any(x in s for x in ('holdout', 'sealed', 'independent_test', 'testset', 'test_set', 'blind_test')) or 'blind' in parts


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def read_csv(p):
    with open(p, newline='', encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def validate_inputs(rows, full, geom):
    if len(rows)!=12 or len(full)!=171 or len(geom)!=171:
        raise ValueError('Frozen input row counts differ from 12/171/171')
    keys={(r['case_id'],r['instance_id']) for r in full}
    if len(keys)!=171:raise ValueError('Duplicate instance keys')
    for r in rows:
        if (r['case_id'],r['instance_id']) not in keys:
            raise ValueError('Priority review key absent from source')
        if not re.fullmatch(r'fresh_dev_\d{3}',r['case_id']) or not re.fullmatch(r'pred_\d{4}',r['instance_id']):
            raise ValueError('Unexpected case/instance ID')
        if not re.fullmatch(r'[0-9a-f]{64}',r['image_sha256']): raise ValueError('Bad SHA256')


def find_images(root, required, full_hash=False, max_files=250000):
    """Only accept byte-identical originals; never match a model crop as original."""
    matched={}; name_candidates=defaultdict(list); count=0; eligible=0; stopped=False
    root=Path(root).expanduser().resolve()
    if not root.is_dir():raise FileNotFoundError(f'Image search root not found: {root}')
    if is_forbidden_path(root): raise ValueError('Refusing to scan holdout/sealed root')
    sha_to_case={digest:case for case,digest in required.items()}
    case_names=tuple(required)
    for directory, dirs, filenames in os.walk(root, followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in SKIP_SCAN and not is_forbidden_path(Path(directory)/d))
        if is_forbidden_path(directory):dirs[:]=[];continue
        for fn in sorted(filenames):
            count+=1
            if count>max_files:
                stopped=True;break
            p=Path(directory)/fn
            if p.suffix.lower() not in EXT or is_forbidden_path(p) or p.is_symlink():continue
            is_named=any(case in fn.lower() for case in case_names)
            likely_hash=any(digest in fn.lower() for digest in sha_to_case)
            if not (is_named or likely_hash or full_hash):continue
            # Exclude predicted masks, crops and visualizations from original-image search.
            if any(x in fn.lower() for x in ('_mask','_overlay','_crop','_roi','visualize','viz_')):continue
            eligible+=1
            try: actual=sha256(p)
            except (OSError,PermissionError):continue
            if actual in sha_to_case:
                case=sha_to_case[actual]
                matched.setdefault(case, str(p))
            elif is_named:
                for case in case_names:
                    if case in fn.lower() and len(name_candidates[case])<5:
                        name_candidates[case].append({'path':str(p),'actual_sha256':actual,'expected_sha256':required[case]})
        if stopped:break
    return matched,name_candidates,{'walked_files':count,'hashed_candidates':eligible,'max_files_exceeded':stopped,'search_root':str(root),'full_hash':full_hash}


def optional_overlays(rows, geom_by_key, original_by_case, mask_copies, out):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return {}, 'Pillow missing: original and mask evidence copied without overlay images'
    overlays={}; notes=[]
    for r in rows:
        case,ins=r['case_id'],r['instance_id']; imgrel=original_by_case.get(case)
        if not imgrel or (case,ins) not in mask_copies:continue
        try:
            with Image.open(out/imgrel) as source: im=source.convert('RGBA')
            if im.width*im.height>65_000_000:
                notes.append(f'{case}/{ins}: image too large for overlay');continue
            size=im.size
            over=Image.new('RGBA',size,(0,0,0,0))
            def paint(key,color):
                mrel=mask_copies.get(key)
                if not mrel:return
                with Image.open(out/mrel) as mask_img: mask=mask_img.convert('L')
                if mask.size!=size:
                    # Do not imply same-size geometry if resizing was necessary.
                    notes.append(f'{case}/{ins}: {key[1]} mask resized for display only from {mask.size} to {size}')
                    mask=mask.resize(size, resample=Image.Resampling.NEAREST)
                opacity=mask.point(lambda x: 110 if x>=128 else 0)
                layer=Image.new('RGBA',size,color)
                layer.putalpha(opacity)
                over.alpha_composite(layer)
            paint((case,ins),(0,205,100,255))
            dest=r.get('shadow_alias_destination') or ''
            if dest:paint((case,dest),(247,53,144,255))
            im=Image.alpha_composite(im,over)
            draw=ImageDraw.Draw(im)
            def rect(key,color,width=4):
                vals=json.loads(geom_by_key[key].get('v5_bbox_xyxy') or '[]')
                if len(vals)==4:draw.rectangle(tuple(float(v) for v in vals),outline=color,width=width)
            rect((case,ins),'#00d47a')
            if dest and (case,dest) in geom_by_key:rect((case,dest),'#fb4f95')
            target=Path('overlays')/f'{case}_{ins}_overlay.jpg'
            (out/target).parent.mkdir(exist_ok=True)
            im.convert('RGB').save(out/target,quality=85,optimize=True)
            overlays[(case,ins)]=target.as_posix()
        except Exception as e:
            notes.append(f'{case}/{ins}: overlay skipped: {type(e).__name__}: {e}')
    return overlays, notes


def build_html(rows, full_by_case, geom_by_key, img_by_case, masks, overlays, manifest):
    header='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PRD 3.2 Guard V4 · DEV60 视觉复核</title>
<style>body{margin:0;background:#f5f7fa;color:#172335;font:15px/1.6 system-ui,-apple-system,"Microsoft YaHei",sans-serif}.wrap{max-width:1180px;margin:auto;padding:28px}.top{background:#10233d;color:white;padding:30px;border-radius:16px}.top h1{margin:0;font-size:25px}.top p{color:#b9c9dd;margin:10px 0 0}.stat{display:flex;gap:12px;margin:16px 0;flex-wrap:wrap}.chip{padding:8px 12px;border-radius:9px;background:#e8eef5;font-weight:600}.warning{padding:12px 16px;background:#fff3d8;border-left:4px solid #bd7d00;margin:18px 0;border-radius:8px}.card{margin:20px 0;background:#fff;border:1px solid #dbe1e9;border-radius:15px;padding:21px;box-shadow:0 4px 15px #14253a0a}.head{display:flex;justify-content:space-between;gap:8px;align-items:center;flex-wrap:wrap}.head h2{font-size:19px;margin:0}.tag{font-size:12px;background:#f0f3f8;padding:4px 9px;border-radius:6px}.media{display:grid;grid-template-columns:1.4fr 1fr;gap:14px;margin-top:14px}.media img{width:100%;max-height:560px;object-fit:contain;background:#eef0f3;border-radius:8px}.media .small img{max-height:180px}.media figure{margin:0}.media figcaption{font-size:12px;color:#657387;padding:5px 0}.masks{display:flex;gap:7px;flex-wrap:wrap}.masks figure{width:130px}.meta{font-size:13px;line-height:1.7;color:#40536b}.forms{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:15px}.field label{display:block;font-size:12px;font-weight:700;margin-bottom:4px}.field select,.field input,.field textarea{width:100%;padding:9px;border:1px solid #c9d2dd;border-radius:8px;background:#fff;box-sizing:border-box;font:inherit}.field.wide{grid-column:1/-1}button{border:0;background:#124d8a;color:white;padding:11px 16px;border-radius:8px;cursor:pointer;font-size:14px}.muted{color:#728198}.fixed{position:sticky;bottom:0;background:#fff;padding:12px;border:1px solid #d9e1e9;box-shadow:0 -3px 15px #162b4015;display:flex;justify-content:space-between;align-items:center;gap:12px}.missing{background:#fff8e9;color:#986900;border-radius:8px;padding:12px}code{font-size:12px}a{color:#205f9c}@media(max-width:820px){.media,.forms{grid-template-columns:1fr}.wrap{padding:10px}}</style></head><body><div class="wrap"><header class="top"><h1>PRD 3.2 · Guard V4 优先 12 条视觉证据复核</h1><p>冻结 DEV60｜仅人工审核辅助｜不写入 V5/Qwen，不发布类别，不触碰独立保留集</p></header>'''
    copied=sum(bool(img_by_case.get(case)) for case in {r['case_id'] for r in rows})
    out=[header,f'<div class="stat"><span class="chip">12 个优先实例</span><span class="chip">10 张开发图片</span><span class="chip">已校验原图 {copied}/10</span><span class="chip">Qwen 原始响应：未提供</span></div>']
    out.append('<div class="warning"><b>审查边界：</b>V5/Qwen 类别是未验证预测，绿色目标 Mask、粉色 Alias Mask 均不是人工 GT。仅凭原图和 V5 Mask 不能判断 Qwen 实际指向了哪件衣物；需要对应的原始 query、ROI 和 response 证据。未亲自核对请保持字段空白。</div>')
    for i,r in enumerate(rows,1):
        case,ins=r['case_id'],r['instance_id']; img=img_by_case.get(case); overlay=overlays.get((case,ins)); alias=r.get('shadow_alias_destination','')
        a=lambda s:html.escape(str(s),quote=True)
        out += [f'<section class="card review" data-case="{a(case)}" data-instance="{a(ins)}"><div class="head"><h2>{i:02d}. {a(case)} / {a(ins)}</h2><span class="tag">{a(r["v4_policy"])} · 优先分 {a(r["priority_score_audit_only"])}</span></div>']
        out.append(f'<div class="meta"><b>V5 预测：</b>{a(r["v5_category_unverified"])}　 <b>Qwen 预测：</b>{a(r["qwen_category_unverified"])}　 <b>类别一致：</b>{a(r["category_agreement"])}　 <b>Alias 指向：</b>{a(alias or "无")}<br><b>历史 BBox：</b>{a(r["v5_bbox_historical"])}　 <b>历史类别：</b>{a(r["v5_class_historical"])}　 <b>V5 BBox：</b>{a(geom_by_key[(case,ins)].get("v5_bbox_xyxy",""))}</div>')
        out.append('<div class="media"><div>')
        if overlay: out.append(f'<figure><img src="{a(overlay)}" alt="{a(case)} overlay"><figcaption>证据可视化：绿色为当前目标 Mask/BBox，粉色为 Alias 候选（非 GT）</figcaption></figure>')
        elif img: out.append(f'<figure><img src="{a(img)}"><figcaption>SHA256 校验通过的源图；无可生成的叠加图</figcaption></figure>')
        else: out.append('<div class="missing">未找到 SHA256 完全匹配的原图。请查看 evidence_manifest.json 中的候选路径，或提供图片映射后重新收集。不能据此填写视觉 GT。</div>')
        out.append('</div><div class="small">')
        if img:out.append(f'<figure><img src="{a(img)}"><figcaption>原始图片（文件 SHA256 已匹配）</figcaption></figure>')
        out.append('<div class="masks">')
        others=[x for x in full_by_case[case] if (case,x['instance_id']) in masks]
        others.sort(key=lambda x:(x['instance_id']!=ins,x['instance_id']!=alias,x['instance_id']))
        for other in others:
            oid=other['instance_id']; rel=masks[(case,oid)]
            out.append(f'<figure><img src="{a(rel)}"><figcaption>{a(oid)} · {a(other["v5_category_unverified"])} {"(TARGET)" if oid==ins else "(ALIAS)" if oid==alias else ""}</figcaption></figure>')
        out.append('</div></div></div>')
        out.append(f'''<div class="forms"><div class="field"><label>V5 目标 Mask 与图中衣物关系</label><select data-f="v5_mask_visual_alignment"><option value="">尚未核实</option><option>MATCH</option><option>PARTIAL</option><option>WRONG_INSTANCE</option><option>UNCERTAIN</option></select></div>
<div class="field"><label>视觉 GT 类别（基于原图）</label><select data-f="human_visual_category_gt"><option value="">尚未核实</option>{''.join('<option>'+a(c)+'</option>' for c in CATS)}</select></div>
<div class="field"><label>Qwen 真实指向（需要原始推理证据）</label><select data-f="human_visual_identity_result"><option value="">尚未核实</option><option>MATCH</option><option>CROSS_INSTANCE</option><option>UNCERTAIN</option></select></div>
<div class="field"><label>Qwen 实际指向的 pred ID（可空）</label><input data-f="qwen_actual_target_instance_id" placeholder="pred_0001 / UNCERTAIN"></div>
<div class="field"><label>原图/Overlay 证据路径</label><input data-f="human_image_evidence_reference" placeholder="evidence/originals/... or overlays/..."></div>
<div class="field"><label>原始 Qwen query/ROI/response 证据路径</label><input data-f="human_qwen_evidence_reference" placeholder="没有原始证据则留空"></div>
<div class="field wide"><label>人工观察备注（遮挡、裙/上衣歧义、Mask 边界和串扰证据等）</label><textarea data-f="human_review_notes" rows="2" placeholder="未看图不要填"></textarea></div></div>''')
        out.append('</section>')
    out.append('''<div class="fixed"><div><b>复核记录导出</b> <span class="muted">填写后点导出；不会覆盖原始预测</span><div style="display:flex;gap:8px;margin-top:6px"><input id="reviewer" placeholder="审核人" style="padding:6px"><input id="date" type="date"></div></div><button id="export">导出人工审核 CSV</button></div></div><script>
const columns=['case_id','instance_id','v5_mask_visual_alignment','human_visual_category_gt','human_visual_identity_result','qwen_actual_target_instance_id','human_image_evidence_reference','human_qwen_evidence_reference','human_review_notes','reviewer','reviewed_at'];
function csvCell(v){let s=String(v??'');return /[",\n\r]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}
document.getElementById('export').onclick=()=>{
  const reviewer=document.getElementById('reviewer').value.trim();const date=document.getElementById('date').value;
  let out=[columns.join(',')];
  document.querySelectorAll('.review').forEach(card=>{
    let row={case_id:card.dataset.case,instance_id:card.dataset.instance};
    card.querySelectorAll('[data-f]').forEach(el=>row[el.dataset.f]=el.value.trim());
    const attempted=columns.some(k=>!['case_id','instance_id','reviewer','reviewed_at'].includes(k)&&row[k]);
    row.reviewer=attempted?reviewer:'';row.reviewed_at=attempted?date:'';
    out.push(columns.map(k=>csvCell(row[k])).join(','));
  });
  const blob=new Blob(['\uFEFF'+out.join('\r\n')+'\r\n'],{type:'text/csv;charset=utf-8'}),a=document.createElement('a');
  a.href=URL.createObjectURL(blob);a.download='PRD32_V4_PRIORITY12_HUMAN_REVIEW.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),500);
};
</script></body></html>''')
    return '\n'.join(out)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image-root',default='/workspace', help='Development images search root; NEVER point to holdout')
    ap.add_argument('--image-map',help='Optional CSV with case_id,image_path; byte SHA256 still required')
    ap.add_argument('--mask-root',help='Optional fallback mask directory if historical absolute paths do not exist')
    ap.add_argument('--full-hash-scan',action='store_true',help='Hash all images under --image-root; use a small DEV-only directory')
    ap.add_argument('--out',default='/workspace/PRD32_1106_V4_PRIORITY12_EVIDENCE',help='Output directory')
    ap.add_argument('--max-files',type=int,default=250000)
    args=ap.parse_args()
    root=Path(args.image_root).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    if is_forbidden_path(root) or is_forbidden_path(out):raise SystemExit('Refusing holdout/sealed path')
    if args.full_hash_scan and root==Path('/workspace'):
        raise SystemExit('For full hash scan, supply a narrower --image-root containing DEV images only')
    prio=read_csv(BASE/'priority12.csv');full=read_csv(BASE/'routing171.csv');geom=read_csv(BASE/'geometry171.csv')
    validate_inputs(prio,full,geom)
    out.mkdir(parents=True,exist_ok=True)
    for dir in ('originals','masks','overlays'): (out/dir).mkdir(exist_ok=True)
    case_digest={r['case_id']:r['image_sha256'] for r in prio}
    matched,candidates,scan=find_images(root,case_digest,args.full_hash_scan,args.max_files)
    if args.image_map:
        for row in read_csv(args.image_map):
            case=row.get('case_id');s=row.get('image_path') or row.get('path') or ''
            if case not in case_digest or not s:continue
            p=Path(s).expanduser().resolve()
            if is_forbidden_path(p): continue
            if p.is_file() and sha256(p)==case_digest[case]:matched[case]=str(p)
            elif p.is_file():candidates[case].append({'path':str(p),'actual_sha256':sha256(p),'expected_sha256':case_digest[case]})
    originals={}
    for case,path in sorted(matched.items()):
        suffix=Path(path).suffix.lower()
        rel=Path('originals')/(case+suffix)
        shutil.copy2(path,out/rel)
        originals[case]=rel.as_posix()
    geom_by_key={(r['case_id'],r['instance_id']):r for r in geom}
    selected_cases=set(case_digest); masks={}; missing_masks=[]
    fallback=Path(args.mask_root).resolve() if args.mask_root else None
    for (case,ins),r in geom_by_key.items():
        if case not in selected_cases:continue
        p=Path(r['mask_path']);
        if not p.is_file() and fallback: p=fallback/p.name
        if is_forbidden_path(p) or not p.is_file():
            missing_masks.append({'case_id':case,'instance_id':ins,'source':r['mask_path']});continue
        rel=Path('masks')/(case+'_'+ins+'_mask'+p.suffix.lower())
        shutil.copy2(p,out/rel);masks[(case,ins)]=rel.as_posix()
    overlays,overlay_notes=optional_overlays(prio,geom_by_key,originals,masks,out)
    full_by_case=defaultdict(list)
    for r in full:full_by_case[r['case_id']].append(r)
    manifest={
        'schema':'guard_v4_priority12_visual_evidence_collection_v1',
        'scope':'FROZEN_DEV60_ONLY',
        'safety':'READ_ONLY_NO_GT_NO_DEPLOYMENT',
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'priority_instance_count':12,'priority_image_count':len(case_digest),
        'matched_original_images':len(originals),'copied_masks':len(masks),
        'missing_masks':missing_masks,'unverified_filename_candidates':dict(candidates),
        'scan':scan,'overlays_created':len(overlays),'overlay_notes':overlay_notes,
        'source_image_sha256_by_case':case_digest,
        'images_by_case':originals,
        'mask_source_geometry':[{**r,'evidence_copy':masks.get((r['case_id'],r['instance_id']),'')} for r in geom if r['case_id'] in selected_cases],
        'review_required':'Qwen original query, ROI, response, and human visual GT not included; no inference of correctness',
        'independent_holdout_not_scanned_by_policy':True
    }
    (out/'evidence_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'review_priority12.html').write_text(build_html(prio,full_by_case,geom_by_key,originals,masks,overlays,manifest),encoding='utf-8')
    (out/'priority12_source.csv').write_bytes((BASE/'priority12.csv').read_bytes())
    (out/'README_UPLOAD_FIRST.txt').write_text(
        'PRD 3.2 Guard V4 — DEV60 Priority 12 Evidence\n'
        f'Original image matches: {len(originals)}/10 (SHA256 verified); masks: {len(masks)}; overlays: {len(overlays)}.\n'
        'Open review_priority12.html after extraction for visual review.\n'
        'The HTML leaves every manual label blank. No Qwen original response is included.\n'
        'If 0 original images matched, provide a DEV-only directory with --image-root and --full-hash-scan,\n'
        'or provide --image-map CSV with case_id,image_path. Never use independent holdout.\n',encoding='utf-8')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out))
    status={'status':'EVIDENCE_COLLECTED_NOT_GT','original_images_sha256_verified':len(originals),
            'priority_images_total':len(case_digest),'masks_copied':len(masks),
            'overlays_created':len(overlays),'missing_masks':len(missing_masks),
            'result_zip':str(archive),'html':str(out/'review_priority12.html'),
            'holdout_scanning_blocked':True}
    print(json.dumps(status,ensure_ascii=False,indent=2))
    if not originals:
        print('WARNING: No SHA256-verified source images found. Please identify DEV-only originals root. Do NOT enter visual GT.',file=sys.stderr)

if __name__=='__main__':main()
