#!/usr/bin/env python3
"""Recompute same-session full-vs-ROI paired categorical metrics from the raw per-case CSV.
Uses only Python standard library; makes no GPU/model calls.
"""
import argparse,csv,hashlib,json,pathlib,collections


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def evaluate(csv_path):
    with open(csv_path,encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    assert len(rows)==12, f'Expected 12 rows, received {len(rows)}'
    assert len({r['case_id'] for r in rows})==len(rows),'Duplicate case ID'
    scoreable=[r for r in rows if r['scoreable'].lower()=='true']
    excluded=[r for r in rows if r['scoreable'].lower()!='true']
    assert len(scoreable)==9, f'Expected 9 scoreable cases, received {len(scoreable)}'
    res={'schema':'prd32_new12_roi_replication_independent_recompute_v1','scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY_NOT_INDEPENDENT',
         'input_csv_sha256':sha256(csv_path),'n_images':len(rows),'n_scoreable':len(scoreable),'n_excluded':len(excluded),
         'excluded_case_ids':[r['case_id'] for r in excluded],'versions':{},'paired_change_cases':[]}
    for v in ['V2','V3']:
        full=sum(r[v+'_full_correct']=='True' for r in scoreable)
        roi=sum(r[v+'_roi_correct']=='True' for r in scoreable)
        improvements=[r['case_id'] for r in scoreable if r[v+'_group']=='IMPROVED']
        regressions=[r['case_id'] for r in scoreable if r[v+'_group']=='REGRESSED']
        still_wrong=[r['case_id'] for r in scoreable if r[v+'_group']=='STILL_WRONG']
        still_correct=[r['case_id'] for r in scoreable if r[v+'_group']=='STILL_CORRECT']
        assert full+len(improvements)-len(regressions)==roi
        archive_match=sum(r[v+'_archived_matches_full']=='True' for r in rows)
        res['versions'][v]={'full_correct':full,'roi_correct':roi,'full_rate':round(full/len(scoreable),4),
            'roi_rate':round(roi/len(scoreable),4),'improved':improvements,'regressed':regressions,
            'still_wrong':still_wrong,'still_correct_count':len(still_correct),'archived_full_matches_current_full':archive_match,
            'strata_by_overlap':{s:{'n':len(z),'full_correct':sum(x[v+'_full_correct']=='True' for x in z),
                                      'roi_correct':sum(x[v+'_roi_correct']=='True' for x in z)}
                      for s in ['YES','NO'] for z in [[r for r in scoreable if r['overlap_other_garments']==s]]}}
        assert archive_match==len(rows), f'{v} archived full mismatch'
    for r in scoreable:
        if any(r[v+'_group'] in ('IMPROVED','REGRESSED','STILL_WRONG') for v in ('V2','V3')):
            res['paired_change_cases'].append({'case_id':r['case_id'],'human_category':r['human_category'],
                'roi_area_fraction':float(r['roi_area_fraction']),
                **{v:{'full_category':r[v+'_full_category'],'roi_category':r[v+'_roi_category'],
                      'full_color':r[v+'_full_color'],'roi_color':r[v+'_roi_color'],'group':r[v+'_group']}
                   for v in ('V2','V3')}})
    return res

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=evaluate(a.csv);pathlib.Path(a.out).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('RECOMPUTE_OK', json.dumps({k:{'full':x['full_correct'],'roi':x['roi_correct']} for k,x in out['versions'].items()}))
