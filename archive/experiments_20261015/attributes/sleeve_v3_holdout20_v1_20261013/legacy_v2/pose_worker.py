"""Runs only image-based pose inference in the existing isolated YOLO environment."""
import argparse,json,hashlib
from pathlib import Path
from PIL import Image

def main():
    p=argparse.ArgumentParser(); p.add_argument('--images',required=True);p.add_argument('--weights',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    import torch
    from ultralytics import YOLO
    weight=Path(a.weights)
    if not weight.is_file(): raise FileNotFoundError(weight)
    model=YOLO(str(weight)); output=[]
    for image in json.loads(Path(a.images).read_text()):
        path=Path(image['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=image['sha256']: raise ValueError('Image changed: '+str(path))
        r=model.predict(source=str(path),imgsz=640,device=0 if torch.cuda.is_available() else 'cpu',conf=.25,verbose=False)[0]
        people=[]
        if r.keypoints is not None:
            for j in range(len(r.keypoints.xy)):
                xy=r.keypoints.xy[j].cpu().tolist()
                conf=r.keypoints.conf[j].cpu().tolist() if r.keypoints.conf is not None else [0.]*len(xy)
                people.append({'person_id':j,'bbox':r.boxes.xyxy[j].cpu().tolist(),'score':float(r.boxes.conf[j]),'xy':xy,'conf':conf})
        output.append({'image_id':image['image_id'],'size':list(Image.open(path).size),'people':people})
        print('pose',image['image_id'],'persons',len(people),flush=True)
    Path(a.output).write_text(json.dumps(output,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
