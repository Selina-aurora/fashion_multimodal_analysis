"""准备隔离 6 张含 jumpsuit 训练图的对照候选，默认只做 dry-run。

Do not overwrite V5 checkpoint or modify validation/thresholds. This candidate
is a data-policy experiment, not a proven improvement or final model.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

EXCLUDED_IMAGES = ['../fashion_data/raw/fashionpedia/images/test/5519b067d1037167b0acd664fcbb918e.jpg', '../fashion_data/raw/fashionpedia/images/test/563c45745e1775a74e3a5b2513079b9d.jpg', '../fashion_data/raw/fashionpedia/images/test/614e7297f89c14e73b691d7f76ed77f4.jpg', '../fashion_data/raw/fashionpedia/images/test/6765044e8079cfe42ae6cac73cbf79c4.jpg', '../fashion_data/raw/fashionpedia/images/test/73b8457583eb2fc453caef8b8307bff9.jpg', '../fashion_data/raw/fashionpedia/images/test/d47fd4ffb3f8559ed563d59f03507e98.jpg']
EXPECTED_TRAIN_SHA256 = 'e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f'
EXPECTED_VAL_SHA256 = '39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true', help='Explicitly start the 15-epoch comparison candidate; default only validates')
    args = parser.parse_args()
    from fashion_multimodal_analysis.common.paths import project_root
    root = project_root()
    source = root / 'configs/prd_8class_train_v5_balanced150_20261008.csv'
    validation = root / 'configs/prd_8class_val_v2_full_targets.csv'
    if sha(source) != EXPECTED_TRAIN_SHA256 or sha(validation) != EXPECTED_VAL_SHA256:
        raise ValueError('清单已与本次审核版本不同，停止，不修改文件')
    with source.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream); fields=reader.fieldnames; rows=list(reader)
    actual={r['source_image'] for r in rows if r['source_image'] in EXCLUDED_IMAGES}
    if actual != set(EXCLUDED_IMAGES):
        raise ValueError('6 张隔离图片与审核清单不一致')
    kept=[r for r in rows if r['source_image'] not in EXCLUDED_IMAGES]
    if len(kept)!=2282 or len({r['source_image'] for r in kept})!=619:
        raise ValueError('候选数量与审核预期不一致')
    candidate=root/'configs/prd_8class_train_v5_no_jumpsuit619_20261008.csv'
    # 新名字保存候选，原清单保持不动。重复执行时必须仍为相同内容。
    import io
    buffer=io.StringIO(newline='')
    writer=csv.DictWriter(buffer,fieldnames=fields);writer.writeheader();writer.writerows(kept)
    payload=buffer.getvalue().encode('utf-8-sig')
    if candidate.exists() and candidate.read_bytes()!=payload:
        raise FileExistsError('候选文件已存在且内容不同：'+str(candidate))
    if not candidate.exists():candidate.write_bytes(payload)
    report=root/'reports/reruns/recovery_20261008/v5_no_jumpsuit619'
    report.mkdir(parents=True,exist_ok=True)
    summary={'original_images':625,'candidate_images':619,'original_instances':2299,
             'candidate_instances':len(kept),'excluded_images':EXCLUDED_IMAGES,
             'class_counts':dict(Counter(r['garment_category'] for r in kept)),
             'original_train_sha256':sha(source),'candidate_train_sha256':sha(candidate),
             'unchanged_val_sha256':sha(validation),'training_requested':args.train,
             'note':'Only a controlled exclusion candidate; effectiveness has not been evaluated.'}
    (report/'candidate_data_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    command=[sys.executable,'-m','fashion_multimodal_analysis.segmentation.training.train_prd_8class_maskrcnn_v3_b1_dataexp',
             '--train-csv',str(candidate),'--val-csv',str(validation),
             '--output-dir',str(root/'outputs/recovery_20261008/v5_no_jumpsuit619'),
             '--report-dir',str(report),'--experiment-name','v5-no-jumpsuit619-20261008',
             '--epochs','15','--workers','2','--sampler-alpha','0.5','--seed','20260921',
             '--batch-size','1','--lr','0.0025','--momentum','0.9','--weight-decay','0.0005',
             '--step-size','5','--gamma','0.1','--min-size','640','--max-size','1024',
             '--verify-isolation']
    if not args.train:command.append('--dry-run')
    subprocess.run(command,cwd=root,check=True)
    if sha(source)!=EXPECTED_TRAIN_SHA256 or sha(validation)!=EXPECTED_VAL_SHA256:
        raise RuntimeError('原始清单意外变化，请停止并检查')
    print('CANDIDATE_TRAIN_COMPLETED' if args.train else 'CANDIDATE_CHECK_PASSED',flush=True)


if __name__=='__main__':
    main()
