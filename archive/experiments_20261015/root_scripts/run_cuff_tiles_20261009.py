"""Cuff proposal coverage/ranking development diagnostic; no product installation."""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
import zipfile


def tiles(size):
    w, h = size
    if w < 2 or h < 2:
        return []
    tw, th = max(1, math.ceil(w*.65)), max(1, math.ceil(h*.65))
    return [(x, y, x+tw, y+th) for y in (0,h-th) for x in (0,w-tw)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack',default='outputs/prd31_cuff_pair_20261009')
    p.add_argument('--validate-only',action='store_true')
    args=p.parse_args();pack=Path(args.pack).resolve()
    sys.path.insert(0,str(pack/'scripts'))
    from run_cuff_pair import digest, iou
    labels=json.loads((pack/'development_labels.json').read_text())
    config=json.loads((pack/'run_config.json').read_text())
    from PIL import Image,ImageDraw
    for case in labels['cases']:
        assert digest(pack/case['image'])==case['image_sha256']
        with Image.open(pack/case['image']) as im:assert list(im.size)==case['image_size']
    if args.validate_only:
        assert tiles((100,200))==[(0,0,65,130),(35,0,100,130),(0,70,65,200),(35,70,100,200)]
        print('VALIDATED: 8 image hashes/sizes; 4 overlapping tiles; no models called.')
        return
    from fashion_multimodal_analysis.integration import model_backends as mb
    from fashion_multimodal_analysis.integration.pipeline import roi_images
    from fashion_multimodal_analysis.grounding.refinement.run_shoulder_waist_pattern_groundingdino_refinement_v1 import run_grounding_dino
    assert digest(config['segmentation_checkpoint'])==labels['checkpoint_sha256'], 'Wrong checkpoint'
    backend=mb.ModelBackends(config);prompt=mb.REGION_PROMPTS['cuff']
    output=Path('reports/reruns/recovery_20261009')/('cuff_tiles_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    for ci,case in enumerate(labels['cases']):
        image=Image.open(pack/case['image']).convert('RGB');rois,seg_ms=backend.segment(image)
        all_instances=[]
        # Predictions depend only on image, predicted garment ROI, frozen config.
        # Human/draft target boxes are consulted below, after ALL prediction calls.
        for roi in rois:
            if roi.category not in ('top','outerwear','dress'):continue
            crop,mask,origin=roi_images(image,roi)
            views=[('full',(0,0,crop.width,crop.height))]+[(f'tile_{i}',b) for i,b in enumerate(tiles(crop.size))]
            candidates=[];rejected=[];elapsed=[]
            for name,b in views:
                detections,seconds=run_grounding_dino(crop.crop(b),prompt,config['grounding_box_threshold'],config['grounding_text_threshold'],backend.ground_processor,backend.ground_model,backend.device)
                elapsed.append({'view':name,'model_seconds':seconds})
                for det in detections:
                    score=float(det['score']);local=[float(v) for v in det['box']]
                    if not math.isfinite(score) or not mb.valid_box(local):continue
                    if not mb.phrase_matches('cuff',det['label']):continue
                    # Clip to actual search window; convert tile -> ROI -> source.
                    x1=max(0,local[0]);y1=max(0,local[1]);x2=min(b[2]-b[0],local[2]);y2=min(b[3]-b[1],local[3])
                    if x2<=x1 or y2<=y1:continue
                    box=[x1+b[0]+origin[0],y1+b[1]+origin[1],x2+b[0]+origin[0],y2+b[1]+origin[1]]
                    item={'view':name,'bbox':box,'score':score,'phrase':det['label']}
                    # Apply existing 0.65 area guard in each search crop as well.
                    ratio=(x2-x1)*(y2-y1)/(crop.width*crop.height)
                    view_ratio=(x2-x1)*(y2-y1)/((b[2]-b[0])*(b[3]-b[1]))
                    if max(ratio,view_ratio)>config['max_local_region_crop_area_ratio']:
                        rejected.append({**item,'reason':'search_crop_area_guard','parent_area_ratio':ratio,'view_area_ratio':view_ratio});continue
                    candidates.append(item)
            candidates.sort(key=lambda v:-v['score'])
            baseline=[v for v in candidates if v['view']=='full']
            all_instances.append({'instance_id':roi.instance_id,'category':roi.category,'bbox':list(roi.bbox),'score':float(roi.score),'baseline_candidates':baseline,'expanded_candidates':candidates,'rejected':rejected,'grounding_calls':elapsed})
        # Evaluation only; no target coordinates influence tile generation/ranking.
        scores=[(iou(case['garment_bbox'],r['bbox']),i) for i,r in enumerate(all_instances)]
        best=max(scores,default=(0,-1));matched=all_instances[best[1]] if best[0]>=config['bbox_match_iou'] else None
        record={'sample_id':case['sample_id'],'draft_presence':case['draft_presence'],'garment_match_iou':best[0],'category_correct':bool(matched and matched['category']==case['category']),'all_instances':all_instances,'arms':{}}
        draw=ImageDraw.Draw(image);draw.rectangle(case['garment_bbox'],outline='yellow',width=2)
        for b in case['draft_cuff_boxes']:draw.rectangle(b,outline='orange',width=3)
        for arm,key,color in [('baseline','baseline_candidates','lime'),('expanded','expanded_candidates','cyan')]:
            candidates=matched[key] if matched else []
            selected=candidates[0] if candidates else None
            overlaps=[max((iou(v['bbox'],b) for b in case['draft_cuff_boxes']),default=0) for v in candidates]
            record['arms'][arm]={'selected':selected,'proposal_count':len(candidates),'draft_selected_iou':overlaps[0] if overlaps else 0,'draft_any_candidate_iou':max(overlaps,default=0),'draft_selected_hit':bool(case['draft_presence']=='present' and overlaps and overlaps[0]>=.5),'draft_any_candidate_hit':bool(case['draft_presence']=='present' and max(overlaps,default=0)>=.5),'negative_has_proposal':bool(case['draft_presence']=='absent' and candidates)}
            if selected:draw.rectangle(selected['bbox'],outline=color,width=3)
        image.save(output/(case['sample_id']+'_selected.jpg'))
        records.append(record)
        (output/'predictions.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
        print(f'{ci+1}/8 {case["sample_id"]} match={best[0]:.3f}',flush=True)
    summary={'status':'DRAFT_DEVELOPMENT_PROPOSALS_ONLY','formal_accuracy':None,'experiment':'full ROI vs full ROI + four overlapping 65% width/height tiles','prompt':prompt,'checkpoint_sha256':digest(config['segmentation_checkpoint']),'labels_sha256':digest(pack/'development_labels.json'),'backend_sha256':digest(mb.__file__),'script_sha256':digest(__file__),'config':config,'cases':8,'positive_denominator':5,'negative_denominator':3,'segmentation_matches':sum(r['garment_match_iou']>=.5 for r in records),'matched_negative_count':sum(r['draft_presence']=='absent' and r['garment_match_iou']>=.5 for r in records),'arms':{a:{'draft_selected_positive_hits':sum(r['arms'][a]['draft_selected_hit'] for r in records),'draft_any_candidate_positive_hits':sum(r['arms'][a]['draft_any_candidate_hit'] for r in records),'draft_negative_with_proposals':sum(r['arms'][a]['negative_has_proposal'] for r in records)} for a in ('baseline','expanded')},'limitations':['Compares DINO proposal boxes BEFORE SAM, not previous SAM-refined output.','Any-candidate hit uses draft labels as an evaluation oracle, never product selection.','Scores across different crops are not calibrated; highest score selection is diagnostic.','No formal absence prediction; segmentation misses are not correct negatives.','No proposal is automatically promoted to the product pipeline.','Known Core images and assistant draft labels; not independent blind acceptance.','Tile union retains baseline candidates: oracle coverage can only stay or increase; assess selected hits separately.','Four additional DINO calls per eligible garment: no latency acceptance claim.']}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (output/'development_labels.json').write_bytes((pack/'development_labels.json').read_bytes())
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in output.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(output))
    print('RESULT_ZIP='+str(archive.resolve()),flush=True)


if __name__=='__main__':main()
