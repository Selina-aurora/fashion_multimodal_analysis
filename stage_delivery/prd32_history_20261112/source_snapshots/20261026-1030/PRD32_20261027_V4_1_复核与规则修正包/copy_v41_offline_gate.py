#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD 3.2 V4.1: post-process V4 candidate facts conservatively, no GPU/GT.

Outputs are UNVERIFIED candidates, never production-ready approved descriptions.
Usage: python copy_v41_offline_gate.py --input candidate_drafts.jsonl --output result_dir
       python copy_v41_offline_gate.py --zip PRD32_COPY_V4_REVIEW_INPUTS_20261027_run01.zip --output result_dir
"""
import argparse, collections, csv, hashlib, io, json, math, re, zipfile
from pathlib import Path

CLASSES={"top","pants","skirt","outerwear","dress","shoe","bag","accessory"}
PATTERNS={"纯色","条纹","格纹","人字纹","波点","花卉印花","文字印花","字母印花","几何印花","动物纹","拼色"}
SUBTYPES={"dress":{"背带裙","吊带裙","连衣裙"},"skirt":{"半身裙"},"top":{"T恤","衬衫","卫衣","针织衫"},"outerwear":{"夹克","西装外套","大衣"},"pants":{"长裤","短裤"},"shoe":{"运动鞋","皮鞋"},"bag":{"背包","手提包"},"accessory":{"配饰"}}
CLASS_NAMES={"top":"上衣","pants":"裤装","skirt":"半身裙","outerwear":"外套","dress":"连衣裙","shoe":"鞋","bag":"包","accessory":"配饰"}
COLOR_TRANSLATE={"black":"黑色","white":"白色","gray":"灰色","grey":"灰色","red":"红色","blue":"蓝色","navy":"藏蓝色","green":"绿色","yellow":"黄色","brown":"棕色","pink":"粉色","purple":"紫色","orange":"橙色","beige":"米色","cream":"乳白色"}
# This whitelist is copied from V4, not learned from human notes.
STRUCTURES={"背带","交叉肩带","肩带","露背","圆领","V领","翻领","长袖","短袖","无袖","门襟","纽扣","纽扣装饰","口袋","拼接","黑白条纹下摆","条纹下摆","宽松裙摆","褶皱","拉链","抽绳","蝴蝶结","刺绣","字母印花"}
PROHIBITED=("红框","红色框","框内","图片里","模特","100%","纯棉","真丝","面料成分","保证","绝对","显瘦","高品质","蕾丝面料")

def safe_list(v):
    if not isinstance(v,list):return []
    return [x.strip() for x in v if isinstance(x,str) and x.strip()][:20]

def clean_record(row):
    issues=[]
    facts=row.get('facts')
    out={'case_id':row.get('case_id'),'image_id':row.get('image_id'),'target_instance_id':row.get('target_instance_id'),
         'source_status':row.get('status'),'v4_original_draft':row.get('draft'), 'v41_candidate_draft':None,
         'normalized_facts':None,'issues':[], 'release_status':'HOLD_FOR_HUMAN_REVIEW','human_verified':False}
    if not isinstance(facts,dict):
        out['issues']=['MISSING_OR_INVALID_FACTS'];return out
    f={k:facts.get(k) for k in ('category','primary_color','pattern','subtype','visible_structures','distinct_decorations','uncertainty')}
    uncs={x.lower().strip() for x in safe_list(f.get('uncertainty'))}
    category=f.get('category') if isinstance(f.get('category'),str) and f.get('category') in CLASSES else None
    if category is None: issues.append('INVALID_OR_MISSING_CATEGORY')
    if 'category' in uncs: category=None;issues.append('CATEGORY_DECLARED_UNCERTAIN')
    f['category']=category
    color=f.get('primary_color');
    if isinstance(color,str):
        color=color.strip(); conv=COLOR_TRANSLATE.get(color.lower())
        if conv: color=conv;issues.append('COLOR_ENGLISH_NORMALIZED')
        if len(color)>12 or not color or any(w in color for w in PROHIBITED):color=None;issues.append('COLOR_INVALID')
    else: color=None
    if 'primary_color' in uncs or 'color' in uncs:color=None;issues.append('COLOR_DECLARED_UNCERTAIN')
    f['primary_color']=color
    subtype=f.get('subtype')
    if not isinstance(subtype,str):subtype=None
    if subtype and category and subtype not in SUBTYPES.get(category,set()):
        issues.append('SUBTYPE_CATEGORY_CONFLICT_ABSTAINED');subtype=None
    if 'subtype' in uncs and subtype:
        issues.append('SUBTYPE_DECLARED_UNCERTAIN_ABSTAINED');subtype=None
    f['subtype']=subtype
    p=f.get('pattern')
    if p is not None and (not isinstance(p,str) or p not in PATTERNS):
        issues.append('PATTERN_INVALID_ABSTAINED');p=None
    if 'pattern' in uncs and p is not None:
        issues.append('PATTERN_DECLARED_UNCERTAIN_ABSTAINED');p=None
    f['pattern']=p
    strs=[];decs=[]
    for key,target in [('visible_structures',strs),('distinct_decorations',decs)]:
        if key in uncs and f.get(key):
            issues.append(key.upper()+'_DECLARED_UNCERTAIN_ABSTAINED');continue
        for x in safe_list(f.get(key)):
            if x not in STRUCTURES or any(t in x for t in PROHIBITED):
                issues.append(key.upper()+'_UNSUPPORTED_ITEM_ABSTAINED');continue
            if x not in target:target.append(x)
    f['visible_structures']=strs
    f['distinct_decorations']=decs
    f['uncertainty']=sorted(uncs)
    out['normalized_facts']=f
    if category is None or color is None:
        issues.append('INSUFFICIENT_SAFE_FACTS_NO_DRAFT')
    else:
        name=subtype or CLASS_NAMES[category]
        clauses=[]
        if p=='纯色':clauses.append('表面以纯色为主')
        elif p: clauses.append('可见'+p+'图案')
        for x in strs+decs:
            # Deduplicate semantic nouns used as both structure and decoration.
            phrase='带有'+x+'细节'
            if phrase not in clauses:clauses.append(phrase)
        draft='，'.join([color+name]+clauses)+'。'
        if len(draft)>80 or any(t in draft for t in PROHIBITED):
            issues.append('DRAFT_LENGTH_OR_UNSAFE_TEXT');draft=None
        out['v41_candidate_draft']=draft
    if f['pattern'] is None:issues.append('PATTERN_NOT_VERIFIED')
    if f['subtype'] is None:issues.append('SUBTYPE_NOT_VERIFIED')
    if out['v41_candidate_draft'] is not None:
        issues.append('VISUAL_ACCURACY_STILL_UNVERIFIED')
    out['issues']=list(dict.fromkeys(issues))
    return out

def get_input(args):
    if bool(args.input)==bool(args.zip):raise ValueError('provide exactly one: --input or --zip')
    if args.input: raw=Path(args.input).read_bytes()
    else:
        with zipfile.ZipFile(args.zip) as z:raw=z.read('V4_SOURCE/candidate_drafts.jsonl')
    lines=raw.decode('utf-8').splitlines()
    rows=[];seen=set()
    for ix,line in enumerate(lines,1):
        if not line.strip():continue
        try:row=json.loads(line,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        except Exception as e: raise ValueError(f'bad candidate JSON line {ix}: {e}')
        if not isinstance(row,dict) or not isinstance(row.get('case_id'),str):raise ValueError(f'bad case id line {ix}')
        if row['case_id'] in seen:raise ValueError(f'duplicate case_id {row["case_id"]}')
        seen.add(row['case_id']);rows.append(row)
    return rows, hashlib.sha256(raw).hexdigest()

def run(args):
    rows,sourcehash=get_input(args)
    outdir=Path(args.output)
    if outdir.exists():raise FileExistsError(f'output exists; choose a new directory: {outdir}')
    outdir.mkdir(parents=True,exist_ok=False)
    cleaned=[clean_record(x) for x in rows]
    with (outdir/'v41_candidates.jsonl').open('w',encoding='utf8') as f:
        for r in cleaned:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    with (outdir/'v41_review_sheet.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','image_id','v4_original_draft','v41_candidate_draft','issues','v41_human_verdict','review_note'])
        w.writeheader()
        for r in cleaned:w.writerow({**{k:r.get(k) for k in ('case_id','image_id','v4_original_draft','v41_candidate_draft')},'issues':';'.join(r['issues']),'v41_human_verdict':'PENDING','review_note':''})
    counters=collections.Counter(k for r in cleaned for k in r['issues'])
    summary={'schema':'prd32_copy_v41_offline_gate_v1','source_candidate_sha256':sourcehash,
       'source_rows':len(rows),'generated_drafts':sum(r['v41_candidate_draft'] is not None for r in cleaned),
       'human_verified':0,'approved_for_publication':0,'issues':dict(counters),
       'warning':'All results are model-derived candidates. Old human findings are NOT injected into the model or automatic rewrite.',
       'official_prd_acceptance':False,'does_not_rerun_gpu':True}
    (outdir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2));return cleaned

def tests():
    f={'category':'outerwear','primary_color':'white','pattern':'波点','subtype':'夹克','visible_structures':['纽扣'],'distinct_decorations':[],'uncertainty':['pattern']}
    r=clean_record({'case_id':'x','facts':f,'draft':'white夹克，可见波点图案'})
    assert r['v41_candidate_draft']=='白色夹克，带有纽扣细节。'
    assert 'PATTERN_DECLARED_UNCERTAIN_ABSTAINED' in r['issues']
    f['uncertainty']=[]
    assert '波点' in clean_record({'case_id':'x','facts':f})['v41_candidate_draft']
    f['subtype']='运动鞋';assert 'SUBTYPE_CATEGORY_CONFLICT_ABSTAINED' in clean_record({'case_id':'x','facts':f})['issues']
    f['primary_color']='invalidcolorvalueverylong';assert clean_record({'case_id':'x','facts':f})['v41_candidate_draft'] is None
    print('V41_SELF_TEST_PASS')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input');ap.add_argument('--zip');ap.add_argument('--output');ap.add_argument('--self-test',action='store_true')
    a=ap.parse_args()
    if a.self_test:tests();return
    if not a.output:ap.error('--output required')
    run(a)

if __name__=='__main__':main()
