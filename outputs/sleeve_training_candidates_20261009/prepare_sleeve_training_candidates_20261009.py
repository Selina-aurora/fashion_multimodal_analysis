"""Prepare native sleeve-supervision candidates for human review, not training.
Compare every upper/whole-body garment in the source image, including garments
outside the selected manifest. Fixed mask-coverage gates only propose ownership.
"""
import base64
from collections import Counter,defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import time
import zipfile

PART_REPORT='reports/reruns/recovery_20261009/sleeve_part_audit_20261009_070132_623305'
LABEL_AUDIT='reports/reruns/recovery_20261009/sleeve_label_audit_20261009_065518_852298'
LABELS=('sleeveless','cap','short','elbow','three_quarter','long')
LENGTH_IDS={156:'sleeveless',157:'short',158:'elbow',159:'three_quarter',160:'long'}


def label_hint(ids):
    labels={LENGTH_IDS[i] for i in ids if i in LENGTH_IDS}
    if 207 in ids:
        # Native cap sleeve shape can coexist with native short length.
        # Propose cap for review, never claim this priority is confirmed GT.
        if labels <= {'short'}:return 'cap'
        return ''
    return next(iter(labels)) if len(labels)==1 else ''


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    import numpy as np
    from PIL import Image,ImageDraw
    from pycocotools import mask as maskutils
    from fashion_multimodal_analysis.common.paths import project_root,resolve_path
    root=project_root();part_report=root/PART_REPORT;audit=root/LABEL_AUDIT
    parts=json.loads((part_report/'native_sleeve_parts.json').read_text())
    targets=json.loads((audit/'per_garment_label_audit.json').read_text())
    previous=json.loads((audit/'summary.json').read_text())
    target_index={int(t['native_annotation']['annotation_id']):t for t in targets if t['source_dataset']=='Fashionpedia' and t.get('native_annotation')}
    assignments=[];groups=defaultdict(list)

    def decode(ann,h,w):
        segmentation=ann['segmentation']
        if isinstance(segmentation,list):rle=maskutils.merge(maskutils.frPyObjects(segmentation,h,w))
        elif isinstance(segmentation,dict) and isinstance(segmentation.get('counts'),list):rle=maskutils.frPyObjects(segmentation,h,w)
        else:rle=segmentation
        mask=maskutils.decode(rle)
        if mask.ndim==3:mask=np.any(mask,axis=2)
        if mask.shape!=(h,w):raise ValueError('Native mask dimensions changed')
        return mask>0

    for info in previous['raw_annotation_files']:
        path=Path(info['path'])
        if sha(path)!=info['sha256']:raise ValueError('Raw annotations changed')
        raw=json.loads(path.read_text(encoding='utf-8-sig'))
        cats={int(c['id']):c for c in raw['categories']}
        images={int(i['id']):i for i in raw['images']}
        annotations={int(a['id']):a for a in raw['annotations']}
        garments=defaultdict(list)
        for a in raw['annotations']:
            if cats[int(a['category_id'])]['supercategory'] in {'upperbody','wholebody'}:
                garments[int(a['image_id'])].append(a)
        by_image=defaultdict(list)
        for p in parts:
            if p['raw_file']==str(path):by_image[int(annotations[p['part_annotation_id']]['image_id'])].append(p)
        for image_id,items in by_image.items():
            meta=images[image_id];h,w=int(meta['height']),int(meta['width'])
            garment_masks={int(a['id']):decode(a,h,w) for a in garments[image_id]}
            for p in items:
                ann=annotations[p['part_annotation_id']];part_mask=decode(ann,h,w);area=int(part_mask.sum())
                ranked=sorted([{'garment_annotation_id':aid,'part_coverage':float(np.logical_and(part_mask,gm).sum()/area) if area else 0.} for aid,gm in garment_masks.items()],key=lambda x:(-x['part_coverage'],x['garment_annotation_id']))
                best=ranked[0] if ranked else None;second=ranked[1]['part_coverage'] if len(ranked)>1 else 0.
                candidate=best['garment_annotation_id'] if best and best['part_coverage']>=.8 and second<=.2 and area else None
                t=target_index.get(candidate)
                status='PROPOSED_SELECTED_GARMENT' if t else 'PROPOSED_NONSELECTED_GARMENT' if candidate is not None else 'AMBIGUOUS_OR_LOW_COVERAGE'
                hint=label_hint(ann.get('attribute_ids',[]))
                evidence={'part_annotation_id':ann['id'],'image_file':meta['file_name'],'part_bbox_xywh':ann['bbox'],
                          'native_attributes':p['part_attributes'],'label_hint':hint,'association_status':status,
                          'ranked_garment_coverages':ranked,'proposed_record_id':t['record_id'] if t else None,
                          'association_confirmed':False}
                assignments.append(evidence)
                if t and hint:groups[t['record_id']].append(evidence)
    candidates=[]
    by_record={t['record_id']:t for t in targets}
    for rid,evidence in groups.items():
        hints={p['label_hint'] for p in evidence}
        if len(hints)!=1:continue
        t=by_record[rid]
        candidates.append({'record_id':rid,'split':t['split'],'category':t['category'],'source_image':t['source_image'],
                           'garment_id':t['garment_id'],'garment_bbox':t['bbox'],'draft_sleeve_length':next(iter(hints)),
                           'parts':evidence,'review_status':'pending','confirmed_training_label':False})
    # A fixed small review sample from TRAIN only; native val stays diagnostic.
    rng=random.Random(20261009);selected=[]
    for label in LABELS:
        pool=sorted([c for c in candidates if c['split']=='train' and c['draft_sleeve_length']==label],key=lambda c:c['record_id'])
        rng.shuffle(pool);selected.extend(pool[:8])
    out=root/'reports/reruns/recovery_20261009'/('sleeve_training_candidates_'+time.strftime('%Y%m%d_%H%M%S')+f'_{time.time_ns()%1000000:06d}')
    out.mkdir(parents=True,exist_ok=False);(out/'review_images').mkdir()
    review=[]
    for number,c in enumerate(selected,1):
        image=Image.open(resolve_path(c['source_image'])).convert('RGB')
        c['source_image_sha256']=sha(resolve_path(c['source_image']))
        original=image.copy();pen=ImageDraw.Draw(image);width=max(2,image.width//220)
        pen.rectangle(c['garment_bbox'],outline='yellow',width=width)
        for p in c['parts']:
            x,y,w,h=p['part_bbox_xywh'];pen.rectangle([x,y,x+w,y+h],outline='red',width=width)
        image.thumbnail((950,1000));image.save(out/'review_images'/f'{number:02d}_{c["record_id"]}.jpg',quality=92)
        buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=88)
        crop=original.crop(tuple(int(x) for x in c['garment_bbox']));crop.thumbnail((650,700));b=io.BytesIO();crop.save(b,format='JPEG',quality=88)
        review.append({**c,'image':'data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode(),
                       'crop':'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()})
    payload={'purpose':'TRAIN-only native sleeve candidate review; not final training data','labels':list(LABELS),
             'association_thresholds':{'best_part_mask_coverage_min':.8,'runner_up_max':.2},'cases':review}
    template=(Path(__file__).parent/'review_template.html').read_text()
    (out/'sleeve_training_review.html').write_text(template.replace('__DATA__',json.dumps(payload,ensure_ascii=False)),encoding='utf-8')
    summary={'status':'SLEEVE_TRAINING_CANDIDATES_PREPARED','assignment_status_counts':dict(Counter(a['association_status'] for a in assignments)),
             'consistent_native_candidate_counts':{split:dict(Counter(c['draft_sleeve_length'] for c in candidates if c['split']==split)) for split in ('train','val')},
             'review_train_samples':len(selected),'review_label_counts':dict(Counter(c['draft_sleeve_length'] for c in selected)),
             'training_started':False,'formal_prd_acceptance':False,
             'limitations':['Mask coverage only proposes sleeve ownership; all review samples start pending.',
                            'Cap shape takes precedence over a short-length hint only for a review draft.',
                            'Conflicting assigned sleeve lengths excluded from consistent candidates; this can bias coverage.',
                            'At most eight training garments per draft class sampled; unreviewed candidates are not approved labels.',
                            'Every upper/whole-body native garment competes for ownership, including nonselected garments.']}
    dump(out/'summary.json',summary);dump(out/'part_association_evidence.json',assignments);dump(out/'all_native_candidates.json',candidates)
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(out))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('UPLOAD_THIS_ZIP:',archive,flush=True)


if __name__=='__main__':main()
