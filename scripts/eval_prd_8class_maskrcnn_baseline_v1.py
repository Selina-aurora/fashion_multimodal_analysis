
from __future__ import annotations

import argparse, csv, math, statistics, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAL_CSV = PROJECT_ROOT / "configs" / "prd_8class_val_v1.csv"
DEFAULT_CKPT = PROJECT_ROOT / "outputs" / "prd_instance_segmentation" / "maskrcnn_8class_baseline_v1" / "checkpoint_last.pth"
REPORT_DIR = PROJECT_ROOT / "reports" / "prd_instance_segmentation" / "maskrcnn_8class_eval_v1"
VIS_DIR = REPORT_DIR / "visualizations"

CLASS_TO_ID = {"top":1,"pants":2,"skirt":3,"outerwear":4,"dress":5,"shoe":6,"bag":7,"accessory":8}
ID_TO_CLASS = {v:k for k,v in CLASS_TO_ID.items()}
NUM_CLASSES = 9
PRD_MASK_IOU_TARGET = 0.85
PRD_TIME_TARGET_MS = 50.0

def args_parse():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p.add_argument("--val-csv", default=str(VAL_CSV))
    p.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    p.add_argument("--score-threshold", type=float, default=0.50)
    p.add_argument("--match-bbox-iou", type=float, default=0.50)
    p.add_argument("--mask-threshold", type=float, default=0.50)
    p.add_argument("--warmup-runs", type=int, default=5)
    p.add_argument("--max-visualizations", type=int, default=32)
    return p.parse_args()

def resolve_path(v):
    p = Path(str(v).strip()).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def group_rows(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["source_dataset"].strip(), r["source_image"].strip())].append(r)
    out=[]
    for (ds,img), items in g.items():
        out.append({"source_dataset":ds,"source_image":img,"instances":items})
    out.sort(key=lambda x:(x["source_dataset"],x["source_image"]))
    return out

def open_mask(path):
    arr = np.asarray(Image.open(path).convert("L"))
    return Image.fromarray(((arr>0).astype(np.uint8)*255), mode="L")

def reconstruct_mask(mask_path, image_size, bbox):
    iw,ih = image_size
    x1,y1,x2,y2 = bbox
    m = open_mask(mask_path)
    if m.size == (iw,ih):
        return m
    bw,bh = max(1,x2-x1), max(1,y2-y1)
    if m.size != (bw,bh):
        m = m.resize((bw,bh), Image.Resampling.NEAREST)
    canvas = Image.new("L",(iw,ih),0)
    cx1,cy1=max(0,min(iw,x1)),max(0,min(ih,y1))
    cx2,cy2=max(0,min(iw,x2)),max(0,min(ih,y2))
    if cx2<=cx1 or cy2<=cy1: return canvas
    mx1,my1 = cx1-x1, cy1-y1
    canvas.paste(m.crop((mx1,my1,mx1+(cx2-cx1),my1+(cy2-cy1))), (cx1,cy1))
    return canvas

class ValDataset(Dataset):
    def __init__(self, csv_path):
        self.samples = group_rows(read_csv(csv_path))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        ip = resolve_path(s["source_image"])
        image = Image.open(ip).convert("RGB")
        iw,ih = image.size
        boxes=[]; labels=[]; masks=[]; meta=[]
        for r in s["instances"]:
            cls=r["garment_category"].strip().lower()
            x1,y1,x2,y2=[int(float(r[k])) for k in ("bbox_x1","bbox_y1","bbox_x2","bbox_y2")]
            x1=max(0,min(iw-1,x1)); y1=max(0,min(ih-1,y1))
            x2=max(x1+1,min(iw,x2)); y2=max(y1+1,min(ih,y2))
            mp=resolve_path(r["mask_path"])
            m=np.asarray(reconstruct_mask(mp,(iw,ih),(x1,y1,x2,y2)))>0
            if not m.any(): raise ValueError(f"Empty GT mask: {mp}")
            boxes.append([x1,y1,x2,y2]); labels.append(CLASS_TO_ID[cls]); masks.append(m.astype(np.uint8))
            meta.append({"record_id":r.get("record_id",""),"garment_id":r.get("garment_id",""),"fine_category":r.get("fine_category",""),"gt_class":cls})
        return F.to_tensor(image), {
            "boxes":torch.tensor(boxes,dtype=torch.float32),
            "labels":torch.tensor(labels,dtype=torch.int64),
            "masks":torch.tensor(np.stack(masks),dtype=torch.uint8),
        }, {"source_dataset":s["source_dataset"],"source_image":s["source_image"],"image_path":ip,"gt_meta":meta}

def build_model(min_size,max_size):
    m=maskrcnn_resnet50_fpn(weights=None,weights_backbone=None)
    inf=m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor=FastRCNNPredictor(inf,NUM_CLASSES)
    infm=m.roi_heads.mask_predictor.conv5_mask.in_channels
    m.roi_heads.mask_predictor=MaskRCNNPredictor(infm,256,NUM_CLASSES)
    m.transform.min_size=(min_size,); m.transform.max_size=max_size
    return m

def choose_device(name):
    if name=="cpu": return torch.device("cpu")
    if name=="cuda":
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def box_iou(a,b):
    ax1,ay1,ax2,ay2=map(float,a); bx1,by1,bx2,by2=map(float,b)
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    ua=max(0,ax2-ax1)*max(0,ay2-ay1)+max(0,bx2-bx1)*max(0,by2-by1)-inter
    return inter/ua if ua>0 else 0.0

def mask_iou(a,b):
    a=a.astype(bool); b=b.astype(bool)
    inter=np.logical_and(a,b).sum(); union=np.logical_or(a,b).sum()
    return float(inter/union) if union else 0.0

def greedy_match(gt_boxes,pred_boxes,thr):
    cand=[]
    for gi in range(len(gt_boxes)):
        for pi in range(len(pred_boxes)):
            i=box_iou(gt_boxes[gi],pred_boxes[pi])
            if i>=thr: cand.append((i,gi,pi))
    cand.sort(reverse=True)
    ug=set(); up=set(); out=[]
    for i,gi,pi in cand:
        if gi in ug or pi in up: continue
        ug.add(gi); up.add(pi); out.append((gi,pi,i))
    return out

def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def ratio(a,b): return a/b if b else float("nan")
def mean(v): return sum(v)/len(v) if v else float("nan")
def fmt(v):
    try:
        v=float(v)
        return "NA" if math.isnan(v) else f"{v:.4f}"
    except Exception:
        return str(v)

def summarize(rows,pred_count):
    localized=sum(bool(r["localized_bbox50"]) for r in rows)
    class_ok=sum(bool(r["class_correct"]) for r in rows)
    mask50=sum(bool(r["mask_iou_ge_0_50"]) for r in rows)
    mask85=sum(bool(r["mask_iou_ge_0_85"]) for r in rows)
    bi=[float(r["bbox_iou"]) for r in rows if r["localized_bbox50"]]
    mi=[float(r["mask_iou"]) for r in rows if r["localized_bbox50"]]
    mic=[float(r["mask_iou"]) for r in rows if r["localized_bbox50"] and r["class_correct"]]
    return {
        "gt_count":len(rows),"prediction_count":pred_count,
        "bbox50_recall":ratio(localized,len(rows)),
        "bbox50_precision":ratio(localized,pred_count),
        "category_accuracy_on_localized":ratio(class_ok,localized),
        "end_to_end_bbox50_class_recall":ratio(class_ok,len(rows)),
        "mean_bbox_iou_localized":mean(bi),
        "mean_mask_iou_localized":mean(mi),
        "mean_mask_iou_correct_class":mean(mic),
        "mask_iou_ge_0_50_rate":ratio(mask50,len(rows)),
        "mask_iou_ge_0_85_rate":ratio(mask85,len(rows)),
        "missed_gt_count":len(rows)-localized,
        "wrong_class_localized_count":localized-class_ok,
        "false_positive_count":max(0,pred_count-localized),
    }

def draw_vis(image_path,gt_boxes,gt_labels,pred_boxes,pred_labels,pred_scores,out_path):
    img=Image.open(image_path).convert("RGB")
    d=ImageDraw.Draw(img); font=ImageFont.load_default()
    for b,l in zip(gt_boxes,gt_labels):
        x1,y1,x2,y2=map(int,b); c=ID_TO_CLASS[int(l)]
        d.rectangle([x1,y1,x2,y2],outline=(0,200,0),width=3)
        d.text((x1+2,max(2,y1-14)),f"GT {c}",fill=(0,160,0),font=font)
    for b,l,s in zip(pred_boxes,pred_labels,pred_scores):
        x1,y1,x2,y2=map(int,b); c=ID_TO_CLASS.get(int(l),str(int(l)))
        d.rectangle([x1,y1,x2,y2],outline=(220,0,0),width=2)
        d.text((x1+2,min(img.height-14,y2+2)),f"P {c} {float(s):.2f}",fill=(180,0,0),font=font)
    out_path.parent.mkdir(parents=True,exist_ok=True); img.save(out_path,quality=92)

def main():
    a=args_parse()
    device=choose_device(a.device)
    ckpt=resolve_path(a.checkpoint); val_csv=resolve_path(a.val_csv)
    REPORT_DIR.mkdir(parents=True,exist_ok=True); VIS_DIR.mkdir(parents=True,exist_ok=True)

    ds=ValDataset(val_csv)
    cp=torch.load(ckpt,map_location="cpu")
    saved=cp.get("args",{})
    min_size=int(saved.get("min_size",640)); max_size=int(saved.get("max_size",1024))
    model=build_model(min_size,max_size)
    model.load_state_dict(cp["model_state_dict"]); model.to(device).eval()
    epoch=int(cp.get("epoch",-1))

    print(f"device={device} | val_images={len(ds)} | checkpoint_epoch={epoch}")

    warm,_,_=ds[0]; warm=warm.to(device)
    with torch.inference_mode():
        for _ in range(a.warmup_runs):
            _=model([warm])
            if device.type=="cuda": torch.cuda.synchronize()

    per_gt=[]; timing=[]; image_stats=[]
    pred_by_source=Counter(); pred_by_class=Counter()

    with torch.inference_mode():
        for idx in range(len(ds)):
            image,target,meta=ds[idx]; img_gpu=image.to(device)
            if device.type=="cuda": torch.cuda.synchronize()
            t0=time.perf_counter(); out=model([img_gpu])[0]
            if device.type=="cuda": torch.cuda.synchronize()
            ms=(time.perf_counter()-t0)*1000

            scores=out["scores"].detach().cpu().numpy()
            keep=scores>=a.score_threshold
            pb=out["boxes"].detach().cpu().numpy()[keep]
            pl=out["labels"].detach().cpu().numpy()[keep]
            ps=scores[keep]
            pm=(out["masks"].detach().cpu().numpy()[keep,0]>=a.mask_threshold).astype(np.uint8)
            gb=target["boxes"].numpy(); gl=target["labels"].numpy(); gm=target["masks"].numpy().astype(np.uint8)

            matches=greedy_match(gb,pb,a.match_bbox_iou)
            by_gt={gi:(pi,bi) for gi,pi,bi in matches}
            pred_by_source[meta["source_dataset"]]+=len(pb)
            for x in pl: pred_by_class[ID_TO_CLASS.get(int(x),str(int(x)))]+=1

            timing.append({"image_index":idx,"source_dataset":meta["source_dataset"],"source_image":meta["source_image"],"gt_instances":len(gb),"predictions_above_threshold":len(pb),"inference_ms":ms})
            image_stats.append({"source_dataset":meta["source_dataset"],"pred_count":len(pb),"match_count":len(matches)})

            vis=VIS_DIR/f"img_{idx:03d}.jpg"
            if idx<a.max_visualizations:
                draw_vis(meta["image_path"],gb,gl,pb,pl,ps,vis)

            for gi in range(len(gb)):
                info=meta["gt_meta"][gi]; gt_class=ID_TO_CLASS[int(gl[gi])]
                if gi in by_gt:
                    pi,bi=by_gt[gi]
                    pred_class=ID_TO_CLASS.get(int(pl[pi]),str(int(pl[pi])))
                    mi=mask_iou(gm[gi],pm[pi]); cc=(int(pl[pi])==int(gl[gi]))
                    outcome="wrong_class" if not cc else ("low_mask_iou" if mi<0.50 else "correct")
                    score=float(ps[pi]); loc=True
                else:
                    bi=mi=score=0.0; pred_class=""; cc=False; outcome="missed"; loc=False
                per_gt.append({
                    "image_index":idx,"source_dataset":meta["source_dataset"],"source_image":meta["source_image"],
                    "record_id":info["record_id"],"garment_id":info["garment_id"],"gt_class":gt_class,
                    "fine_category":info["fine_category"],"pred_class":pred_class,"pred_score":score,
                    "localized_bbox50":loc,"bbox_iou":float(bi),"class_correct":cc,"mask_iou":float(mi),
                    "mask_iou_ge_0_50":bool(loc and cc and mi>=0.50),
                    "mask_iou_ge_0_85":bool(loc and cc and mi>=PRD_MASK_IOU_TARGET),
                    "outcome":outcome,"visualization_path":str(vis.resolve()) if vis.is_file() else ""
                })
            print(f"{idx+1:02d}/{len(ds)} done")

    total_pred=sum(x["pred_count"] for x in image_stats)
    overall=summarize(per_gt,total_pred)

    per_class=[]
    for c in CLASS_TO_ID:
        rows=[r for r in per_gt if r["gt_class"]==c]
        per_class.append({"garment_category":c,**summarize(rows,pred_by_class[c])})

    per_source=[]
    for src in sorted({r["source_dataset"] for r in per_gt}):
        rows=[r for r in per_gt if r["source_dataset"]==src]
        per_source.append({"source_dataset":src,**summarize(rows,pred_by_source[src])})

    errors=[r for r in per_gt if r["outcome"]!="correct"]
    times=[float(r["inference_ms"]) for r in timing]
    mean_ms=mean(times); med_ms=float(statistics.median(times)); p95=float(np.percentile(times,95))

    write_csv(REPORT_DIR/"per_gt_results.csv",per_gt)
    write_csv(REPORT_DIR/"per_class_metrics.csv",per_class)
    write_csv(REPORT_DIR/"per_source_metrics.csv",per_source)
    write_csv(REPORT_DIR/"per_image_timing.csv",timing)
    write_csv(REPORT_DIR/"error_cases.csv",errors)

    oc=Counter(r["outcome"] for r in per_gt)
    summary=[
        "PRD 3.1.1 Mask R-CNN 8-Class Baseline Evaluation v1",
        "====================================================","",
        f"checkpoint={ckpt}",f"checkpoint_epoch={epoch}",f"device={device}",
        f"gpu={torch.cuda.get_device_name(0) if device.type=='cuda' else 'none'}",
        f"score_threshold={a.score_threshold}",f"match_bbox_iou={a.match_bbox_iou}",f"mask_threshold={a.mask_threshold}",
        f"validation_images={len(ds)}",f"gt_instances={overall['gt_count']}",f"predictions_above_threshold={overall['prediction_count']}","",
        f"bbox50_recall={fmt(overall['bbox50_recall'])}",
        f"bbox50_precision={fmt(overall['bbox50_precision'])}",
        f"mean_bbox_iou_localized={fmt(overall['mean_bbox_iou_localized'])}",
        f"category_accuracy_on_localized={fmt(overall['category_accuracy_on_localized'])}",
        f"end_to_end_bbox50_class_recall={fmt(overall['end_to_end_bbox50_class_recall'])}",
        f"mean_mask_iou_localized={fmt(overall['mean_mask_iou_localized'])}",
        f"mean_mask_iou_correct_class={fmt(overall['mean_mask_iou_correct_class'])}",
        f"mask_iou_ge_0_50_rate={fmt(overall['mask_iou_ge_0_50_rate'])}",
        f"mask_iou_ge_0_85_rate={fmt(overall['mask_iou_ge_0_85_rate'])}","",
        f"mean_inference_ms_per_image={fmt(mean_ms)}",
        f"median_inference_ms_per_image={fmt(med_ms)}",
        f"p95_inference_ms_per_image={fmt(p95)}",
        f"mean_inference_time_le_50ms={bool(mean_ms<=PRD_TIME_TARGET_MS)}","",
        f"correct={oc.get('correct',0)}",f"low_mask_iou={oc.get('low_mask_iou',0)}",
        f"wrong_class={oc.get('wrong_class',0)}",f"missed={oc.get('missed',0)}","",
        "Notes:",
        "- Matching is one-to-one by bbox IoU >= 0.50, independent of class.",
        "- Wrong-class localized predictions are counted as wrong_class rather than missed.",
        "- Mask IoU >= 0.85 is reported as a PRD-reference pass rate.",
        "- Timing excludes file I/O and visualization and is measured after warm-up.",
        "- Do not run another GPU-heavy job during timing if you need trustworthy speed numbers."
    ]
    (REPORT_DIR/"evaluation_summary.txt").write_text("\n".join(summary)+"\n",encoding="utf-8")

    print("\n=== EVALUATION FINISHED ===")
    print("bbox50 recall:",fmt(overall["bbox50_recall"]))
    print("category accuracy on localized:",fmt(overall["category_accuracy_on_localized"]))
    print("mean mask IoU correct-class:",fmt(overall["mean_mask_iou_correct_class"]))
    print("mask IoU >= 0.85 rate:",fmt(overall["mask_iou_ge_0_85_rate"]))
    print("mean inference ms/image:",fmt(mean_ms))
    print("summary:",REPORT_DIR/"evaluation_summary.txt")

if __name__=="__main__":
    main()
