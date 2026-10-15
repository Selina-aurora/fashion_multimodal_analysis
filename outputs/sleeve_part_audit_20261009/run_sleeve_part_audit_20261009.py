"""Read native Fashionpedia sleeve-part labels; never assume garment linkage.
Source attributes can be attached to a sleeve annotation rather than the whole
garment. Export all same-image garment candidates for later association review.
"""
import hashlib
import json
from pathlib import Path
import time
import zipfile
from collections import Counter

AUDIT='reports/reruns/recovery_20261009/sleeve_label_audit_20261009_065518_852298'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    from fashion_multimodal_analysis.common.paths import project_root
    root=project_root();audit=root/AUDIT
    previous=json.loads((audit/'summary.json').read_text())
    targets=json.loads((audit/'per_garment_label_audit.json').read_text())
    native_targets=[r for r in targets if r['source_dataset']=='Fashionpedia' and r.get('native_annotation')]
    by_source={}
    for target in native_targets:
        n=target['native_annotation']
        by_source.setdefault(n['raw_file'],{}).setdefault(n['image_file'],[]).append(target)
    records=[];definitions=[];coverage=[];all_counts=Counter();all_attribute_counts=Counter()
    for info in previous['raw_annotation_files']:
        path=Path(info['path'])
        if sha(path)!=info['sha256']:raise ValueError('Native file changed')
        raw=json.loads(path.read_text(encoding='utf-8-sig'))
        cats={int(c['id']):c for c in raw['categories']}
        attrs={int(a['id']):a for a in raw.get('attributes',[])}
        images={int(i['id']):i for i in raw['images']}
        definitions.append({'file':str(path),'categories':list(cats.values()),
                            'length_attributes':[a for a in attrs.values() if a.get('supercategory')=='length'],
                            'sleeve_named_attributes':[a for a in attrs.values() if 'sleeve' in a.get('name','').lower()]})
        sleeve_ids={cid for cid,c in cats.items() if 'sleeve' in c.get('name','').lower()}
        if not sleeve_ids:
            coverage.append({'file':str(path),'sleeve_part_categories':[],'native_sleeve_parts':0});continue
        indexed={int(a['id']):a for a in raw['annotations']}
        native_part_count=0
        for ann in raw['annotations']:
            cid=int(ann['category_id'])
            if cid not in sleeve_ids:continue
            native_part_count+=1;name=cats[cid]['name'];all_counts[name]+=1
            ids=ann.get('attribute_ids',[])
            attribute_definitions=[attrs[i] for i in ids if isinstance(i,int) and i in attrs]
            all_attribute_counts.update(a.get('name','') for a in attribute_definitions)
            image=images.get(int(ann['image_id']),{});filename=image.get('file_name','')
            same_image=by_source.get(str(path),{}).get(filename,[])
            if not same_image:continue
            candidates=[]
            for t in same_image:
                native=indexed[t['native_annotation']['annotation_id']]
                candidates.append({'record_id':t['record_id'],'split':t['split'],'category':t['category'],
                                   'garment_id':t['garment_id'],'garment_bbox_xyxy':t['bbox'],
                                   'native_garment_annotation_id':native['id'],'native_garment_bbox_xywh':native.get('bbox')})
            # Do not use same-image proximity as a parent relationship.
            records.append({'raw_file':str(path),'image_file':filename,'part_annotation_id':ann['id'],
                            'part_category':name,'part_bbox_xywh':ann.get('bbox'),'part_attribute_ids':ids,
                            'part_attributes':attribute_definitions,'part_annotation_keys':list(ann),
                            'explicit_relation_fields':{k:v for k,v in ann.items() if any(s in k.lower() for s in ('parent','instance','garment','group'))},
                            'same_image_garment_candidates':candidates,'association_confirmed':False})
        coverage.append({'file':str(path),'sleeve_part_categories':[cats[i] for i in sleeve_ids],'native_sleeve_parts':native_part_count})
    out=root/'reports/reruns/recovery_20261009'/('sleeve_part_audit_'+time.strftime('%Y%m%d_%H%M%S')+f'_{time.time_ns()%1000000:06d}')
    out.mkdir(parents=True,exist_ok=False)
    summary={'status':'NATIVE_SLEEVE_PART_AUDIT_COMPLETED','raw_coverage':coverage,
             'all_native_sleeve_category_counts':dict(all_counts),'all_native_sleeve_attribute_names':dict(all_attribute_counts),
             'sleeve_parts_on_selected_garment_images':len(records),
             'selected_image_part_attribute_names':dict(Counter(a.get('name','') for r in records for a in r['part_attributes'])),
             'train_garments_with_same_image_sleeve_parts':len({c['record_id'] for r in records for c in r['same_image_garment_candidates'] if c['split']=='train'}),
             'val_garments_with_same_image_sleeve_parts':len({c['record_id'] for r in records for c in r['same_image_garment_candidates'] if c['split']=='val'}),
             'training_started':False,'formal_prd_acceptance':False,
             'limitations':['Same-image sleeve and garment annotations do not establish ownership.',
                            'Native length definitions must be mapped carefully; sleeve shape is not sleeve length.',
                            'No automatically generated garment labels; no train/val merging.']}
    dump(out/'summary.json',summary);dump(out/'native_sleeve_parts.json',records);dump(out/'taxonomy.json',definitions)
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zipped:
        for path in out.iterdir():zipped.write(path,path.relative_to(out))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('UPLOAD_THIS_ZIP:',archive,flush=True)


if __name__=='__main__':main()
