#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dev60-only mask asset coverage & candidate mask IoU probe.
Reads ONLY 171 filenames from frozen 10.30 V5 output; does not glob or read held-out images.
Requires an explicitly supplied mask root. Never trains/runs a model.
"""
from pathlib import Path
import argparse,csv,json,sys

def write_rows(path,rs,columns):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rs)

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--mask-root',required=True,help='Directory holding V5 Dev60 instance PNG mask assets, not a holdout directory')
    pa.add_argument('--output-dir',default='DEV60_MASK_PROBE_OUTPUT')
    a=pa.parse_args()
    root=Path(a.mask_root).resolve();out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).parent/'input'/'v5_raw_predictions.jsonl'
    dev=[json.loads(x) for x in source.read_text(encoding='utf-8').splitlines() if x.strip()]
    assert len(dev)==60 and sum(len(p['instances']) for p in dev)==171
    pairs=[];records=[];lookup={};size_lookup={}
    for p in dev:
        for it in p['instances']:
            name=it['mask_path']
            if name!=Path(name).name or not name.endswith('.png'):
                raise ValueError('Unexpected mask filename: '+name)
            target=(root/name)
            exists=target.is_file()
            key=(p['case_id'],it['instance_id']);lookup[key]=target if exists else None
            size_lookup[key]=tuple(p['image_size'])
            records.append({'case_id':p['case_id'],'instance_id':it['instance_id'],
                            'mask_basename':name,'asset_present':'YES' if exists else 'NO',
                            'mask_matches_image_dimensions':'NOT_CHECKED','foreground_fraction':''})
    try:
        from PIL import Image,ImageChops
    except ImportError:
        print('MASK_PROBE_MISSING_PIL: install pillow');return 2
    cache={};ridx={(r['case_id'],r['instance_id']):r for r in records}
    for key,file in lookup.items():
        if not file:continue
        try:
            with Image.open(file) as mask:
                m=mask.convert('L').point(lambda t:255 if t>0 else 0)
                w,h=m.size
                if m.size!=size_lookup[key]:
                    ridx[key]['mask_matches_image_dimensions']='NO';continue
                ridx[key]['mask_matches_image_dimensions']='YES'
                white=m.histogram()[255]
                ridx[key]['foreground_fraction']=round(white/(w*h),5)
                cache[key]=m
        except Exception:
            ridx[key]['mask_matches_image_dimensions']='READ_ERROR'
    # Analyze only the EXACT dev60 cross-class box pair list. No automatic suppression.
    with open(Path(__file__).parent/'output'/'43_GEOMETRIC_OVERLAP_PAIRS_IoU_GE_05.csv',encoding='utf-8-sig',newline='') as f:
        geom=list(csv.DictReader(f))
    assert len(geom)==43
    for q in geom:
        k1=(q['case_id'],q['instance_a']);k2=(q['case_id'],q['instance_b'])
        status='MISSING_OR_BAD_MASK';val=''
        if k1 in cache and k2 in cache:
            m1,m2=cache[k1],cache[k2]
            intersection=ImageChops.darker(m1,m2).histogram()[255]
            union=ImageChops.lighter(m1,m2).histogram()[255]
            val=round(intersection/union,5) if union else ''
            status='MASK_IOU_COMPUTED' if union else 'EMPTY_MASK_UNION'
        pairs.append({'case_id':q['case_id'],'instance_a':q['instance_a'],'instance_b':q['instance_b'],
                      'bbox_iou':q['bbox_iou'],'mask_iou_if_available':val,'status':status,
                      'warning':'Do not call duplicate/merge solely based on mask IoU'})
    write_rows(out/'171_MASK_ASSET_COVERAGE.csv',records,list(records[0]))
    write_rows(out/'43_PAIR_MASK_COMPARISON.csv',pairs,list(pairs[0]))
    measured=sum(r['status']=='MASK_IOU_COMPUTED' for r in pairs)
    print(f'DEV60_MASK_ASSET_PROBE_DONE {len(cache)} / 171 masks dimension-valid; {measured} / 43 pairs mask-IoU computed')
    print('No model inference, no holdout access, no automatic NMS.')
    return 0
if __name__=='__main__':sys.exit(main())
