"""Controlled garment-crop diagnosis. Uses reference garment boxes, not cuff labels, for inference."""
import argparse
import json
import math
import sys
from pathlib import Path
from datetime import datetime
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack',default='outputs/prd31_cuff_pair_20261009')
    parser.add_argument('--baseline-dir')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();pack=Path(args.pack).resolve()
    sys.path.insert(0,str(pack/'scripts'))
    from run_cuff_pair import digest,iou,json_safe
    from PIL import Image,ImageDraw
    labels=json.loads((pack/'development_labels.json').read_text())
    config=json.loads((pack/'run_config.json').read_text())
    if args.baseline_dir:
        baseline=Path(args.baseline_dir).resolve()
    else:
        choices=sorted(Path('reports/reruns/recovery_20261009').glob('cuff_tiles_*'),reverse=True)
        choices=[p for p in choices if p.is_dir() and (p/'summary.json').is_file()]
        if not choices:raise SystemExit('找不到分块测试缓存，请用 --baseline-dir 指定其结果目录。')
        baseline=choices[0]
    summary=json.loads((baseline/'summary.json').read_text())
    if summary.get('labels_sha256')!=digest(pack/'development_labels.json'):
        raise ValueError('Cached baseline labels differ; stopped.')
    if summary.get('checkpoint_sha256')!=labels['checkpoint_sha256']:
        raise ValueError('Cached segmentation checkpoint differs; stopped.')
    if summary.get('status')!='DRAFT_DEVELOPMENT_PROPOSALS_ONLY' or summary.get('cases')!=8:
        raise ValueError('Not the completed eight-image tile diagnostic.')
    old_config=summary['config']
    for key in ('grounding_model','grounding_box_threshold','grounding_text_threshold','bbox_match_iou','max_local_region_crop_area_ratio'):
        if config[key]!=old_config[key]:raise ValueError('Baseline config differs: '+key)
    old={c['sample_id']:c for c in json.loads((baseline/'predictions.json').read_text())}
    if set(old)!={c['sample_id'] for c in labels['cases']}:raise ValueError('Case IDs differ')
    for case in labels['cases']:
        assert digest(pack/case['image'])==case['image_sha256']
        with Image.open(pack/case['image']) as im:assert list(im.size)==case['image_size']
    if args.validate_only:
        print('VALIDATED: eight image hashes, reference crop metadata, cached baseline and fixed settings; no GPU calls.')
        return
    import torch
    from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
    from fashion_multimodal_analysis.integration.pipeline import crop_bounds
    from fashion_multimodal_analysis.integration.region_semantics import phrase_matches
    from fashion_multimodal_analysis.evaluation.protocol import valid_box
    from fashion_multimodal_analysis.grounding.refinement.run_shoulder_waist_pattern_groundingdino_refinement_v1 import run_grounding_dino
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device=torch.device('cuda');processor=AutoProcessor.from_pretrained(config['grounding_model'])
    model=AutoModelForZeroShotObjectDetection.from_pretrained(config['grounding_model']).to(device).eval()
    prompt=summary['prompt'] # EXACT earlier original prompt, not a new wording experiment.
    output=Path('reports/reruns/recovery_20261009')/('cuff_controlled_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False);records=[]
    for n,case in enumerate(labels['cases']):
        image=Image.open(pack/case['image']).convert('RGB')
        bounds=crop_bounds(case['garment_bbox'],image.size);crop=image.crop(bounds)
        # This explicitly uses reference garment crop; it is NOT a GT-free product prediction.
        # Draft cuff boxes/presence never enter model inputs or candidate selection.
        detections,seconds=run_grounding_dino(crop,prompt,config['grounding_box_threshold'],config['grounding_text_threshold'],processor,model,device)
        candidates=[];rejections=[]
        for det in detections:
            local=[float(v) for v in det['box']];score=float(det['score'])
            if not valid_box(local) or not math.isfinite(score) or not phrase_matches('cuff',det['label']):continue
            local=[max(0,local[0]),max(0,local[1]),min(crop.width,local[2]),min(crop.height,local[3])]
            if not valid_box(local):continue
            box=[local[0]+bounds[0],local[1]+bounds[1],local[2]+bounds[0],local[3]+bounds[1]]
            ratio=(local[2]-local[0])*(local[3]-local[1])/(crop.width*crop.height)
            item={'bbox':box,'score':score,'phrase':det['label'],'crop_area_ratio':ratio}
            if ratio>config['max_local_region_crop_area_ratio']:
                rejections.append({**item,'reason':'search_crop_area_guard'});continue
            candidates.append(item)
        candidates.sort(key=lambda d:-d['score'])
        overlaps=[max((iou(v['bbox'],b) for b in case['draft_cuff_boxes']),default=0) for v in candidates]
        selected=candidates[0] if candidates else None
        controlled={'selected':selected,'proposal_count':len(candidates),'draft_selected_iou':overlaps[0] if overlaps else 0,'draft_any_candidate_iou':max(overlaps,default=0),'draft_selected_hit':bool(case['draft_presence']=='present' and overlaps and overlaps[0]>=.5),'draft_any_candidate_hit':bool(case['draft_presence']=='present' and max(overlaps,default=0)>=.5),'negative_has_proposal':bool(case['draft_presence']=='absent' and candidates)}
        previous=old[case['sample_id']]
        record={'sample_id':case['sample_id'],'draft_presence':case['draft_presence'],'review_status':'ASSISTANT_DRAFT_PENDING_HUMAN_REVIEW','controlled_reference_crop':bounds,'all_controlled_candidates':candidates,'rejected':rejections,'model_seconds':seconds,'prior_segmentation_match_iou':previous['garment_match_iou'],'prior_category_correct':previous['category_correct'],'arms':{'predicted_crop':previous['arms']['baseline'],'reference_crop':controlled}}
        draw=ImageDraw.Draw(image);draw.rectangle(bounds,outline='yellow',width=2)
        for box in case['draft_cuff_boxes']:draw.rectangle(box,outline='orange',width=3)
        for arm,color in [('predicted_crop','lime'),('reference_crop','cyan')]:
            candidate=record['arms'][arm]['selected']
            if candidate:draw.rectangle(candidate['bbox'],outline=color,width=3)
        image.save(output/(case['sample_id']+'_controlled.jpg'))
        records.append(record);(output/'predictions.json').write_text(json.dumps(json_safe(records),ensure_ascii=False,indent=2))
        print(f'{n+1}/8 {case["sample_id"]} candidates={len(candidates)}',flush=True)
    result={'status':'CONTROLLED_REFERENCE_GARMENT_CROP_DEVELOPMENT_ONLY','formal_accuracy':None,'prompt':prompt,'config':config,'cases':8,'positive_denominator':5,'negative_denominator':3,'labels_sha256':digest(pack/'development_labels.json'),'script_sha256':digest(__file__),'baseline_summary_sha256':digest(baseline/'summary.json'),'baseline_predictions_sha256':digest(baseline/'predictions.json'),'baseline_dir':str(baseline),'arms':{a:{'draft_selected_positive_hits':sum(r['arms'][a]['draft_selected_hit'] for r in records),'draft_any_candidate_positive_hits':sum(r['arms'][a]['draft_any_candidate_hit'] for r in records),'draft_negative_with_proposals':sum(r['arms'][a]['negative_has_proposal'] for r in records)} for a in ('predicted_crop','reference_crop')},'limitations':['Uses reference garment boxes; never report as end-to-end product accuracy.','Compares DINO proposal boxes before SAM; no region mask accuracy.','Controlled reference crops include background; no reference garment mask is used.','Draft cuff boundaries unchanged from earlier experiments; pending independent confirmation.','Reference cropping also bypasses segmentation misses/category routing; attribution is to upstream ROI setup as a whole.','Cached baseline scores reused; no valid paired latency comparison.','Same embedded JPEGs as previous runs, not newly recovered dataset raw files.']}
    (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (output/'development_labels.json').write_bytes((pack/'development_labels.json').read_bytes())
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in output.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(output))
    print('RESULT_ZIP='+str(archive.resolve()),flush=True)


if __name__=='__main__':main()
