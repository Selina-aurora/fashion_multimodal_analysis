"""Read-only audit of existing sleeve supervision; no training or label changes.

Fashionpedia native instance attributes are kept verbatim for semantic review.
DeepFashion2 category-derived hints are explicitly weak labels, not sleeve GT.
Train and validation remain separate. No model loading or GPU required.
"""
import csv
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile
from collections import Counter

MANIFESTS = {
 'train': ('configs/prd_8class_train_v5_balanced150_20261008.csv', 'e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f'),
 'val': ('configs/prd_8class_val_v2_full_targets.csv', '39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93'),
}
UPPER = {'top', 'outerwear', 'dress'}
# These names describe dataset classes, not necessarily the visible sleeve end
# after rolling/occlusion. Never convert these hints into confirmed labels.
WEAK_DF2 = {
 'short_sleeve_top':'short', 'short_sleeve_outwear':'short', 'short_sleeve_dress':'short',
 'long_sleeve_top':'long', 'long_sleeve_outwear':'long', 'long_sleeve_dress':'long',
 'vest':'sleeveless', 'sling':'sleeveless', 'vest_dress':'sleeveless', 'sling_dress':'sleeveless',
}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):digest.update(b)
    return digest.hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def annotation_id(row):
    ref=str(row.get('annotation_ref','')).strip()
    if ref.isdigit():return int(ref)
    match=re.search(r'fashionpedia_ann_(\d+)',str(row.get('garment_id','')))
    return int(match.group(1)) if match else None


def basename(value):
    return str(value).replace('\\','/').rsplit('/',1)[-1]


def sleeve_related(attribute):
    # Audit candidates only: sleeves can also occur in a shape/style field.
    # Export the full definition so the actual field semantics can be reviewed.
    return 'sleeve' in json.dumps(attribute,ensure_ascii=False).lower()


def main():
    from fashion_multimodal_analysis.common.paths import project_root,resolve_path,data_root
    root=project_root();rows={};manifest_provenance={}
    for split,(relative,expected) in MANIFESTS.items():
        path=resolve_path(relative)
        if sha(path)!=expected:raise ValueError('Frozen manifest changed: '+relative)
        with path.open(encoding='utf-8-sig',newline='') as stream:
            all_rows=list(csv.DictReader(stream))
        rows[split]=[r for r in all_rows if r['garment_category'] in UPPER]
        if len({r['record_id'] for r in rows[split]})!=len(rows[split]):raise ValueError('Duplicate record IDs')
        manifest_provenance[split]={'path':str(path),'sha256':expected,'all_rows':len(all_rows),'upper_garment_rows':len(rows[split])}
    fp_root=data_root()/'raw/fashionpedia'
    wanted={annotation_id(r) for items in rows.values() for r in items if r['source_dataset']=='Fashionpedia'}
    matches={};definitions=[];raw_files=[];issues=[]
    # Search only native instance annotation files, never images/model caches.
    paths=sorted(fp_root.rglob('instances_attributes*.json')) if fp_root.exists() else []
    for path in paths:
        print('READ_NATIVE_ATTRIBUTES:',path,flush=True)
        try:
            raw=json.loads(path.read_text(encoding='utf-8-sig'))
            if not all(isinstance(raw.get(k),list) for k in ('images','annotations','categories')):
                issues.append({'path':str(path),'issue':'Not an instance annotation JSON'});continue
            attributes={int(a['id']):a for a in raw.get('attributes',[]) if 'id' in a}
            images={int(i['id']):i for i in raw['images']}
            categories={int(c['id']):c for c in raw['categories']}
            raw_files.append({'path':str(path),'sha256':sha(path),'annotations':len(raw['annotations']),'attributes':len(attributes)})
            definitions.extend({'file':str(path),'definition':a,'sleeve_related_candidate':sleeve_related(a)} for a in attributes.values())
            for ann in raw['annotations']:
                aid=int(ann['id'])
                if aid not in wanted:continue
                image=images.get(int(ann['image_id']),{})
                category=categories.get(int(ann['category_id']),{})
                ids=ann.get('attribute_ids')
                if ids is None:ids=ann.get('attributes')
                valid=isinstance(ids,list) and all(isinstance(v,int) for v in ids)
                unknown=[v for v in ids if v not in attributes] if valid else []
                known=[attributes[v] for v in ids if v in attributes] if valid else []
                item={'annotation_id':aid,'image_file':image.get('file_name',''),'source_category':category.get('name',''),
                      'raw_file':str(path),'attribute_ids':ids,'attribute_format_valid':valid,
                      'unknown_attribute_ids':unknown,'all_attributes':known,
                      'sleeve_related_attributes':[a for a in known if sleeve_related(a)]}
                matches.setdefault(aid,[]).append(item)
        except (OSError,ValueError,KeyError,TypeError) as e:
            issues.append({'path':str(path),'issue':type(e).__name__+': '+str(e)})
    records=[]
    for split,items in rows.items():
        for row in items:
            source=row['source_dataset'];fine=row.get('fine_category','')
            record={'split':split,'record_id':row['record_id'],'source_dataset':source,
                    'source_image':row['source_image'],'garment_id':row['garment_id'],
                    'category':row['garment_category'],'fine_category':fine,
                    'bbox':[float(row['bbox_'+k]) for k in ('x1','y1','x2','y2')],
                    'mask_path':row.get('mask_path',''),'native_attributes':[],
                    'sleeve_candidates':[],'weak_category_hint':'','confirmed_sleeve_gt':False}
            if source=='DeepFashion2':
                record['weak_category_hint']=WEAK_DF2.get(fine,'')
                record['status']='CATEGORY_HINT_ONLY' if record['weak_category_hint'] else 'NO_SLEEVE_CATEGORY_HINT'
            elif source=='Fashionpedia':
                # An annotation ID alone is insufficient when two raw splits
                # reuse IDs. Require exact source image basename and category.
                candidates=[a for a in matches.get(annotation_id(row),[]) if
                            basename(a['image_file'])==basename(row['source_image']) and a['source_category']==fine]
                if not candidates:record['status']='NATIVE_ANNOTATION_NOT_JOINED'
                elif len(candidates)>1:
                    record['status']='MULTIPLE_NATIVE_MATCHES_REVIEW';record['native_match_candidates']=candidates
                else:
                    native=candidates[0];record['native_annotation']=native
                    record['native_attributes']=native['all_attributes'];record['sleeve_candidates']=native['sleeve_related_attributes']
                    record['status']='ATTRIBUTE_FORMAT_REVIEW' if not native['attribute_format_valid'] or native['unknown_attribute_ids'] else 'NATIVE_SLEEVE_CANDIDATES_REVIEW' if record['sleeve_candidates'] else 'NO_NATIVE_SLEEVE_CANDIDATE'
            else:record['status']='UNSUPPORTED_SOURCE'
            records.append(record)
    out=root/'reports/reruns/recovery_20261009'/('sleeve_label_audit_'+time.strftime('%Y%m%d_%H%M%S')+f'_{time.time_ns()%1000000:06d}')
    out.mkdir(parents=True,exist_ok=False)
    summary={'status':'READ_ONLY_SLEEVE_LABEL_AUDIT_COMPLETED','manifest_provenance':manifest_provenance,
             'raw_annotation_files':raw_files,'raw_read_issues':issues,'raw_root':str(fp_root),
             'per_split':{},'training_started':False,'formal_prd_acceptance':False,
             'limitations':['Native sleeve-related attributes are candidates requiring taxonomy review, not automatically approved six-class sleeve labels.',
                            'DeepFashion2 short/long/vest/sling categories are weak hints; rolled or occluded sleeves can disagree with visible-end review.',
                            'Do not train on validation rows or the 18 reviewed diagnostic images.',
                            'Image basenames and categories are required for raw annotation association; unresolved/multiple matches are not used.',
                            'No feasibility promise before examining class coverage, annotation semantics and example quality.']}
    for split in rows:
        subset=[r for r in records if r['split']==split]
        names=Counter(a.get('name',json.dumps(a,ensure_ascii=False)) for r in subset for a in r['sleeve_candidates'])
        summary['per_split'][split]={'upper_garments':len(subset),'status_counts':dict(Counter(r['status'] for r in subset)),
                                     'source_counts':dict(Counter(r['source_dataset'] for r in subset)),
                                     'native_sleeve_candidate_names':dict(names),
                                     'weak_hint_counts':dict(Counter(r['weak_category_hint'] for r in subset if r['weak_category_hint']))}
    dump(out/'summary.json',summary);dump(out/'per_garment_label_audit.json',records)
    dump(out/'native_attribute_definitions.json',definitions)
    fields=['split','record_id','source_dataset','source_image','garment_id','category','fine_category','status','weak_category_hint','confirmed_sleeve_gt','sleeve_candidates']
    with (out/'per_garment_label_audit.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in records:
            line={k:r[k] for k in fields};line['sleeve_candidates']=json.dumps(line['sleeve_candidates'],ensure_ascii=False);writer.writerow(line)
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zipped:
        for path in out.iterdir():zipped.write(path,path.relative_to(out))
    print(json.dumps(summary['per_split'],ensure_ascii=False,indent=2),flush=True)
    print('UPLOAD_THIS_ZIP:',archive,flush=True)


if __name__=='__main__':main()
