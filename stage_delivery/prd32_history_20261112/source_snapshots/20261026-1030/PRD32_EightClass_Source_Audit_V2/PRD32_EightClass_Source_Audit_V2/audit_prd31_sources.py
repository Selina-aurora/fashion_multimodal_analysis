#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD 3.2 eight-class source schema/provenance auditor (read-only).

Reads CSV TEXT METADATA ONLY, never opens images or modifies the source,
never trains or infers. Conservative with fullval and unknown provenance.
Outputs a JSON technical report and Markdown handoff summary.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

CLASSES = ('top','pants','skirt','outerwear','dress','shoe','bag','accessory')
DEFAULT_FILES = (
    'configs/prd_8class_unified_manifest_v2.csv',
    'configs/prd_3_1_predicted_roi_fullval_v1.csv',
    'configs/prd_3_1_predicted_roi_matched_v2.csv',
)
DANGER_TOKENS = ('holdout','sealed','do_not_open','do-not-open','private_test','blind_test')
CLASS_KEYS = ('predicted_category','pred_category','pred_class','class_pred','detected_class','class_name_pred','category','garment_category','garment_class','class_name','gt_category','gt_class','label','gt_label')
SPLIT_KEYS = ('source_split','dataset_split','data_split','split','set','subset','partition','data_role','source_type')
SHA_KEYS = ('image_sha256','image_hash_sha256','sha256','source_image_sha256')
IMAGE_KEYS = ('image_id','img_id','image_path','img_path','file_path','filepath','image','file_name','filename','img_file','image_file','path','image_name')
BOX_KEYS = ('target_bbox_xyxy','bbox_xyxy','pred_bbox_xyxy','predicted_bbox_xyxy','bbox','pred_bbox','box','x1','xmin','left','bbox_x1')
INSTANCE_KEYS = ('target_instance_id','instance_id','pred_instance_id','object_id','annotation_id','instance_index')
TRAIN_TOKENS = ('train','training','fit','trainval')
SEALED_TOKENS = ('holdout','sealed','blind','private','test_locked','test_sealed')
ALREADY_USED_MARKERS = ('new1022_','dev_02','dev_04','dev_05','dev_06','dev_08','dev_09','dev_12')

def norm_key(k):
    return re.sub(r'[^a-z0-9]+','_',str(k).strip().lower()).strip('_')

def sha256_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):h.update(chunk)
    return h.hexdigest()

def sniff_encoding(p):
    with p.open('rb') as f: b=f.read(8192)
    for enc in ('utf-8-sig','gb18030'):
        try:b.decode(enc);return enc
        except UnicodeDecodeError:continue
    return 'utf-8'   # replacement read below is only for metadata diagnostics

def first_matching(keys, prefs):
    for pref in prefs:
        if pref in keys:return keys[pref]
    return None

def safe_path(p):
    return not any(tok in p.as_posix().lower() for tok in DANGER_TOKENS)

def classify_source(p):
    name=p.name.lower()
    if 'fullval' in name:return 'REFERENCE_ONLY_FULLVAL_NOT_NEW_DEV'
    if 'matched' in name:return 'REFERENCE_OR_MATCHED_PREDICTIONS_UNVERIFIED'
    if 'unified_manifest' in name:return 'EIGHT_CLASS_INVENTORY_NOT_GT_OR_NEW_DEV'
    return 'UNVERIFIED_SOURCE'

def shape(value):
    if value is None or not str(value).strip():return 'empty'
    s=str(value).strip()
    if s.startswith(('[','{')):return 'json_like'
    if re.fullmatch(r'[+-]?\d+',s):return 'integer'
    try:float(s);return 'number'
    except (ValueError,OverflowError):pass
    if '/' in s or '\\' in s:return 'path_like'
    if len(s)==64 and re.fullmatch(r'[a-fA-F0-9]{64}',s):return 'sha256_like'
    return 'text'

def split_flags(value):
    v=str(value).strip().lower()
    if any(t in v for t in SEALED_TOKENS):return 'restricted'
    if any(t in v for t in TRAIN_TOKENS):return 'train'
    if v in ('test','testing','val','validation','fullval','eval','evaluation'):return 'eval_or_test_protect'
    if v in ('dev_new','dev','development','new_dev','development_new'):return 'dev_labeled_not_independent'
    return 'unknown'

def find_identity_columns(keys):
    for names,kind in ((SHA_KEYS,'sha256'), (('image_id','img_id'),'image_id'), (('image_path','img_path','file_path','filepath','image','file_name','filename','img_file','image_file','path','image_name'),'image_location')):
        col=first_matching(keys,names)
        if col:return col,kind
    return None,None

def inspect_source(path, *, row_limit=200000, id_cap=100000):
    result={
        'path':str(path), 'source_role':classify_source(path),
        'read_mode':'CSV_METADATA_ONLY_NO_IMAGE_OPENED',
        'eligibility':'NOT_APPROVED_FOR_DEV_OR_INDEPENDENT_EVALUATION',
        'exists':path.is_file(), 'warnings':[]
    }
    if not safe_path(path):
        result['read_mode']='REFUSED_RESTRICTED_PATH';result['warnings'].append('Restricted/holdout filename; file not opened')
        return result,set(),None
    if not path.is_file():
        result['warnings'].append('File is absent on this runtime; run on AutoDL')
        return result,set(),None
    result['bytes']=path.stat().st_size
    result['file_sha256']=sha256_file(path)
    result['encoding']=sniff_encoding(path)
    csv.field_size_limit(max(csv.field_size_limit(),10*1024*1024))
    category_counts=defaultdict(Counter);split_counter=Counter();shape_per_col=defaultdict(Counter)
    identifier_seen=set();identity_duplicates=0;identity_count=0;restricted_splits=0;train_splits=0
    old_markers=Counter();empty_identifiers=0;first_samples=0
    try:
        with path.open('r',encoding=result['encoding'],errors='replace',newline='') as f:
            head=f.read(4096);f.seek(0)
            try: dialect=csv.Sniffer().sniff(head, delimiters=',\t;')
            except csv.Error:dialect=csv.excel
            reader=csv.DictReader(f,dialect=dialect)
            if not reader.fieldnames:
                result['warnings'].append('No CSV header discovered')
                return result,set(),None
            fields=[str(x).strip() for x in reader.fieldnames if x is not None]
            norm_fields={norm_key(col):col for col in fields}
            ident_col,ident_kind=find_identity_columns(norm_fields)
            class_cols=[norm_fields[x] for x in CLASS_KEYS if x in norm_fields]
            split_cols=[norm_fields[x] for x in SPLIT_KEYS if x in norm_fields]
            image_cols=[norm_fields[x] for x in IMAGE_KEYS if x in norm_fields]
            bbox_cols=[norm_fields[x] for x in BOX_KEYS if x in norm_fields]
            inst_cols=[norm_fields[x] for x in INSTANCE_KEYS if x in norm_fields]
            result.update({'columns':fields, 'columns_count':len(fields),
                'class_columns':class_cols,'split_columns':split_cols,'image_columns':image_cols,
                'bbox_columns':bbox_cols,'instance_columns':inst_cols,
                'identity_column':ident_col,'identity_kind':ident_kind,
                'record_count':0,'csv_dialect_delimiter':dialect.delimiter})
            for index,row in enumerate(reader):
                if index>=row_limit:
                    result['warnings'].append(f'Hit safe row limit {row_limit}; counts are partial')
                    break
                result['record_count']+=1
                for col in class_cols:
                    val=str(row.get(col,'') or '').strip().lower()
                    if val: category_counts[col][val]+=1
                for col in split_cols:
                    val=str(row.get(col,'') or '').strip()
                    if val:
                        split_counter[f'{col}={val}']+=1
                        sf=split_flags(val)
                        if sf=='restricted':restricted_splits+=1
                        if sf=='train':train_splits+=1
                if ident_col:
                    ident=str(row.get(ident_col,'') or '').strip()
                    if ident:
                        identity_count+=1
                        if ident in identifier_seen:identity_duplicates+=1
                        if len(identifier_seen)<id_cap:identifier_seen.add(ident)
                        for marker in ALREADY_USED_MARKERS:
                            if marker in ident.lower():old_markers[marker]+=1
                    else:empty_identifiers+=1
                if first_samples<4:
                    for col in fields:shape_per_col[col][shape(row.get(col))]+=1
                    first_samples+=1
            result.update({'identity_nonempty_count':identity_count,
                'identity_duplicate_rows':identity_duplicates,
                'identity_empty_count':empty_identifiers,
                'category_counts_by_column':{col:dict(ctr.most_common(30)) for col,ctr in category_counts.items()},
                'split_value_counts':dict(split_counter.most_common(30)),
                'redacted_value_shapes_first_four_rows':{col:dict(c) for col,c in shape_per_col.items()},
                'previously_used_image_marker_counts':dict(old_markers),
                'has_train_split_rows':train_splits>0, 'has_restricted_split_rows':restricted_splits>0,
                'has_required_bridge_fields': bool(ident_col and bbox_cols and class_cols and image_cols and inst_cols)})
            if not split_cols:
                result['warnings'].append('No recognizable split column: cannot prove disjoint from training/holdout')
            if not bbox_cols:
                result['warnings'].append('No recognizable bbox column: not directly usable as 3.2 target ROI')
            if not class_cols:
                result['warnings'].append('No recognizable class column')
            if not ident_col:
                result['warnings'].append('No recognizable image identity: cannot crosscheck overlap')
            if result['source_role'].startswith('REFERENCE_ONLY'):
                result['warnings'].append('fullval is protected as reference-only; NEVER use as new development data without separate approval')
            if train_splits:result['warnings'].append('Includes train-marked rows; cannot be used for independent evaluation')
            if restricted_splits:result['warnings'].append('Includes restricted/holdout-marked rows; require separate exclusion before any use')
            return result,identifier_seen,ident_kind
    except Exception as e:
        result['warnings'].append(f'Read error: {type(e).__name__}: {e}')
        return result,set(),None

def path_next(base:Path)->Path:
    if not base.exists():return base
    for i in range(2,100):
        x=base.with_name(base.name+f'_run{i:02d}')
        if not x.exists():return x
    raise FileExistsError('All output suffixes used, supply a new --output-dir')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',default='/workspace/fashion_multimodal_analysis')
    p.add_argument('--output-dir',default='/workspace/PRD32_8CLASS_SOURCE_AUDIT_20261027')
    p.add_argument('--source',action='append',help='Explicit CSV relative to project root or absolute; repeatable')
    p.add_argument('--max-rows',type=int,default=200000)
    a=p.parse_args(argv)
    root=Path(a.project_root)
    if a.max_rows<1:raise ValueError('--max-rows must be positive')
    if not root.is_dir():raise NotADirectoryError(root)
    paths=[Path(x) for x in (a.source or DEFAULT_FILES)]
    paths=[x if x.is_absolute() else root/x for x in paths]
    report={'schema':'prd32_8class_source_schema_audit_v2','generated_by':'audit_prd31_sources.py',
        'objective':'Identify candidate 3.1 CSV schemas/provenance before building 3.2 eight-class datasets',
        'images_opened':0,'model_loaded':False,'frozen_prd31_modified':False,
        'holdout_accessed':False,'independent_evaluation_performed':False,
        'sources':[],'identity_overlap':[]}
    sets=[]
    for path in paths:
        result,ids,kind=inspect_source(path,row_limit=a.max_rows)
        report['sources'].append(result)
        sets.append((path.name,ids,kind))
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            a1,ids1,k1=sets[i];a2,ids2,k2=sets[j]
            comparable=bool(k1 and k1==k2)
            report['identity_overlap'].append({
                'source_a':a1,'source_b':a2,'identity_kind_a':k1,'identity_kind_b':k2,
                'comparable':comparable,
                'exact_overlap_distinct_keys':len(ids1&ids2) if comparable else None,
                'note':'Metadata key overlap only; absence does not prove independent image content' if comparable else 'Cannot compare different identifier kinds'
            })
    out=path_next(Path(a.output_dir))
    if not safe_path(out):raise ValueError('Refusing write into holdout/restricted output path')
    out.mkdir(parents=True,exist_ok=False)
    dest=out/'source_schema_audit.json'
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    md=['# PRD 3.2 八类数据源审计（仅元数据）','','**重要：这不是八类准确率，更不是新的独立测试集。**','',
        '没有读取图片、执行推理或修改 PRD 3.1 冻结；fullval 一律标记为 reference-only。','',
        '| 来源 | 记录数 | 类别字段 | 划分字段 | 图片标识 | bbox字段 | 是否可直接评估 |',
        '|---|---:|---|---|---|---|---|']
    for src in report['sources']:
        def s(k):return ', '.join(src.get(k,[])) or '未知'
        md.append('| {} | {} | {} | {} | {} | {} | 否：来源与人工 GT 未确认 |'.format(
            Path(src['path']).name,src.get('record_count','—'),s('class_columns'),s('split_columns'),
            src.get('identity_column') or '未知',s('bbox_columns')))
        md.append('')
        md.append(f"**{Path(src['path']).name} — {src['source_role']}**")
        for key,count in src.get('category_counts_by_column',{}).items():
            md.append(f'- {key}（预测/GT 含义未确认）: '+', '.join(f'{v}={n}' for v,n in count.items()))
        for note in src.get('warnings',[]):md.append(f'- 注意：{note}')
        md.append('')
    md += ['## 下一步','',
        '1. 将 `source_schema_audit.json` 和本报告提供给助手，依据**实际字段**制定转换映射。',
        '2. 通过真实训练数据清单、明确的数据划分和图片哈希核查来源；不要把“val/fullval/matched”名称视为独立样本证明。',
        '3. 针对确定属于可用新开发集的实例，分别准备 `人工确认参考框`（测3.2独立能力）和 `冻结V5预测框`（测端到端误差传播），两条结果不混计。',
        '4. 补齐人工类别、颜色、图案 GT 后再做 3.2 属性推理和指标核算；封存 holdout 不参与开发。','']
    (out/'README_审计结果.md').write_text('\n'.join(md),encoding='utf-8')
    print('SOURCE_SCHEMA_AUDIT_COMPLETE')
    print('OUTPUT_DIR:',out)
    for src in report['sources']:
        print('SOURCE:',Path(src['path']).name,'rows=',src.get('record_count','NA'),
              'classes=',src.get('class_columns',[]),'split=',src.get('split_columns',[]),
              'identity=',src.get('identity_column'),'bbox=',src.get('bbox_columns',[]))
    print('NEXT: Upload source_schema_audit.json and README_审计结果.md; no original dataset needed.')
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        print('AUDIT_FAILED:',f'{type(e).__name__}: {e}',file=sys.stderr)
        sys.exit(2)
