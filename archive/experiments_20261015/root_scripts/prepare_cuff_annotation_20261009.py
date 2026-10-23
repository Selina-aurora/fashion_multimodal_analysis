"""Export training-side cuff annotation pilot; excludes val/Core paths and exact file duplicates."""
import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime
from collections import Counter
import zipfile


HTML = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>袖口标注</title>
<style>body{font:16px sans-serif;margin:20px;background:#f4f5f7}button,select{font-size:16px;padding:8px;margin:4px}canvas{max-width:100%;height:auto;background:white;touch-action:none}#controls{position:sticky;top:0;background:white;padding:10px;z-index:2}#stage{max-width:1000px}p{max-width:1100px}#notice{color:#a32}</style>
<h1>袖口训练侧标注小批次</h1><p>只标黄色框指定的服装。袖口包含短袖末端开口；左右分别画紧框，不框整条袖子、下摆、手表或内搭。边缘遮挡、截断、太模糊时选“无法判断”，不能当成“无袖口”。这些图片来自原分割训练集，不能当独立测试集。</p>
<div id="controls"><button id="prev">上一张</button><button id="next">下一张</button><select id="presence"><option value="pending">待标注</option><option value="present">可见袖口</option><option value="absent">目标服装确实无袖口</option><option value="ambiguous">无法判断 / 遮挡</option></select><button id="undo">撤销一个框</button><button id="clear">清空袖口框</button><button id="confirm">确认当前标注</button><button id="export">导出标注 JSON</button><button id="import">导入已有 JSON</button><input type="file" id="file" accept="application/json" hidden><div id="status"></div><div id="notice"></div></div>
<p id="info"></p><div id="stage"><canvas id="canvas"></canvas></div>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent), cs=data.cases;
const $=id=>document.getElementById(id), cv=$('canvas'),ctx=cv.getContext('2d');let index=0,img=null,start=null,drag=null,token=0;
const key='cuff_annotation_'+data.selection_sha256;
function notice(s){$('notice').textContent=s}
function persist(){try{localStorage.setItem(key,JSON.stringify(cs.map(c=>({record_id:c.record_id,presence:c.presence,cuff_boxes:c.cuff_boxes,review_status:c.review_status,reviewed_at:c.reviewed_at||null}))));}catch(e){notice('浏览器无法自动保存；请及时点击“导出标注 JSON”。')}}
function applyRows(rows){if(rows.length!==cs.length||rows.some((r,i)=>r.record_id!==cs[i].record_id))throw Error('样本顺序或数量不匹配');for(let i=0;i<rows.length;i++){let r=rows[i];if(!['pending','present','absent','ambiguous'].includes(r.presence)||!Array.isArray(r.cuff_boxes))throw Error('标注格式错误');for(let b of r.cuff_boxes){if(b.length!==4||!b.every(Number.isFinite)||!(0<=b[0]&&b[0]<b[2]&&b[2]<=cs[i].image_width&&0<=b[1]&&b[1]<b[3]&&b[3]<=cs[i].image_height))throw Error('框超出图像范围')}if(r.review_status==='reviewed'&&(r.presence==='pending'||(r.presence==='present'&&!r.cuff_boxes.length)||(r.presence!=='present'&&r.cuff_boxes.length)))throw Error('已确认标注的状态与框不一致')};rows.forEach((r,i)=>Object.assign(cs[i],{presence:r.presence,cuff_boxes:r.cuff_boxes,review_status:r.review_status==='reviewed'?'reviewed':'pending',reviewed_at:r.reviewed_at||null}));}
try{let saved=localStorage.getItem(key);if(saved)applyRows(JSON.parse(saved))}catch(e){notice('自动保存记录不可用；可导入之前导出的 JSON。')}
function progress(){let c=cs[index];$('status').textContent=`${index+1}/${cs.length}；已确认 ${cs.filter(v=>v.review_status==='reviewed').length}/${cs.length}；当前：${c.review_status}`}
function draw(){if(!img)return;ctx.clearRect(0,0,cv.width,cv.height);ctx.drawImage(img,0,0);ctx.lineWidth=Math.max(2,cv.width/250);ctx.strokeStyle='#ffe600';let b=cs[index].garment_bbox;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);ctx.strokeStyle='#00e060';for(let b of cs[index].cuff_boxes)ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);if(start&&drag){ctx.strokeStyle='#ff9900';ctx.strokeRect(start[0],start[1],drag[0]-start[0],drag[1]-start[1])}}
function show(){let c=cs[index],t=++token;img=null;start=null;drag=null;cv.width=c.image_width;cv.height=c.image_height;$('presence').value=c.presence;$('info').textContent=`${c.record_id} | ${c.category} | ${c.source_dataset} | 原细类参考：${c.fine_category}。细类名不能代替袖口可见性判断。黄色=目标服装，绿色=你画的袖口。`;progress();let im=new Image();im.onload=()=>{if(t!==token)return;img=im;draw()};im.src=c.image_data}
function point(e){let r=cv.getBoundingClientRect();return [Math.max(0,Math.min(cv.width,(e.clientX-r.left)*cv.width/r.width)),Math.max(0,Math.min(cv.height,(e.clientY-r.top)*cv.height/r.height))]}
function change(){let c=cs[index];c.review_status='pending';c.reviewed_at=null;persist();progress();draw()}
cv.onpointerdown=e=>{if(!img)return;cv.setPointerCapture(e.pointerId);start=point(e);drag=start;e.preventDefault()};cv.onpointermove=e=>{if(start){drag=point(e);draw()}};
cv.onpointerup=e=>{if(!start)return;drag=point(e);let b=[Math.floor(Math.min(start[0],drag[0])),Math.floor(Math.min(start[1],drag[1])),Math.ceil(Math.max(start[0],drag[0])),Math.ceil(Math.max(start[1],drag[1]))];start=null;drag=null;if(b[2]>b[0]&&b[3]>b[1]){cs[index].cuff_boxes.push(b);cs[index].presence='present';$('presence').value='present';change()}else draw()};cv.onpointercancel=()=>{start=null;drag=null;draw()};
$('prev').onclick=()=>{index=Math.max(0,index-1);show()};$('next').onclick=()=>{index=Math.min(cs.length-1,index+1);show()};$('undo').onclick=()=>{cs[index].cuff_boxes.pop();change()};$('clear').onclick=()=>{cs[index].cuff_boxes=[];change()};$('presence').onchange=()=>{cs[index].presence=$('presence').value;change()};
$('confirm').onclick=()=>{let c=cs[index];if(c.presence==='pending')return notice('先选择可见、确实无袖口或无法判断。');if(c.presence==='present'&&!c.cuff_boxes.length)return notice('可见袖口必须至少画一个框。');if(c.presence!=='present'&&c.cuff_boxes.length)return notice('无袖口或无法判断时，请先清空袖口框。');c.review_status='reviewed';c.reviewed_at=new Date().toISOString();notice('当前标注已确认。');persist();progress()};
$('export').onclick=()=>{let result={...data,cases:cs.map(({image_data,...c})=>c),exported_at:new Date().toISOString(),reviewed_count:cs.filter(c=>c.review_status==='reviewed').length};let a=document.createElement('a'),u=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));a.href=u;a.download='cuff_annotation_progress_20261009.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),2000)};
$('import').onclick=()=>$('file').click();$('file').onchange=async()=>{try{let f=$('file').files[0];if(!f)return;let r=JSON.parse(await f.text());if(r.selection_sha256!==data.selection_sha256)throw Error('不属于本批样本');applyRows(r.cases);persist();show();notice('标注已恢复。')}catch(e){notice('导入失败：'+e.message)}finally{$('file').value=''}};show();
</script></html>'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--per-stratum',type=int,default=6);args=p.parse_args()
    if not 1<=args.per_stratum<=10:raise ValueError('per-stratum must be 1..10')
    from PIL import Image
    from fashion_multimodal_analysis.common.paths import project_root,resolve_path
    root=project_root();train=resolve_path('configs/prd_8class_train_v5_balanced150_20261008.csv');val=resolve_path('configs/prd_8class_val_v2_full_targets.csv');core=resolve_path('benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv')
    expected={train:'e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f',val:'39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93',core:'c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89'}
    for path,value in expected.items():
        if sha(path)!=value:raise ValueError('Manifest changed: '+str(path))
    protected_paths={resolve_path(r['source_image']).resolve() for path in (val,core) for r in rows(path)}
    print(f'Checking {len(protected_paths)} protected source files...',flush=True)
    protected_hashes={sha(path) for path in protected_paths} # Fail on missing protected source.
    all_rows=rows(train);selected=[];selected_hashes=set();selected_paths=set();hashes={};excluded=Counter()
    for category in ('top','outerwear','dress'):
        for source_dataset in ('DeepFashion2','Fashionpedia'):
            pool=[r for r in all_rows if r['garment_category']==category and r['source_dataset']==source_dataset]
            # Prefer larger full garment boxes; no sleeve presence is inferred from metadata.
            pool.sort(key=lambda r:(-min(float(r['bbox_width']),float(r['bbox_height'])),hashlib.sha256(('cuff20261009:'+r['record_id']).encode()).hexdigest()))
            count=0
            for r in pool:
                source=resolve_path(r['source_image']).resolve()
                if source in protected_paths:excluded['protected_path']+=1;continue
                if source in selected_paths:continue
                if source not in hashes:hashes[source]=sha(source)
                h=hashes[source]
                if h in protected_hashes:excluded['protected_exact_file_hash']+=1;continue
                if h in selected_hashes:continue
                with Image.open(source) as image:iw,ih=image.size
                if (iw,ih)!=(int(float(r['image_width'])),int(float(r['image_height']))):raise ValueError('Source dimensions differ: '+str(source))
                box=[float(r['bbox_'+k]) for k in ('x1','y1','x2','y2')]
                if not (0<=box[0]<box[2]<=iw and 0<=box[1]<box[3]<=ih):raise ValueError('Invalid native garment bbox')
                selected.append({'record_id':r['record_id'],'source_dataset':source_dataset,'category':category,'fine_category':r['fine_category'],'source_image_reference':r['source_image'],'garment_id':r['garment_id'],'garment_bbox':box,'image_width':iw,'image_height':ih,'source_image_sha256':h,'source_file':source,'presence':'pending','cuff_boxes':[],'review_status':'pending'})
                selected_hashes.add(h);selected_paths.add(source);count+=1
                if count==args.per_stratum:break
            if count<args.per_stratum:raise ValueError(f'Not enough isolated images in {category}/{source_dataset}: {count}')
    output=root/'reports/reruns/recovery_20261009'/('cuff_annotation_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));(output/'source_images').mkdir(parents=True)
    for case in selected:
        source=case.pop('source_file');file_name=case['record_id']+source.suffix.lower();raw=source.read_bytes()
        (output/'source_images'/file_name).write_bytes(raw);case['source_file_relative']='source_images/'+file_name
        mime='image/png' if source.suffix.lower()=='.png' else 'image/jpeg';case['image_data']='data:'+mime+';base64,'+base64.b64encode(raw).decode()
    ids=[(c['record_id'],c['source_image_sha256']) for c in selected]
    data={'status':'TRAINING_SIDE_ANNOTATION_PILOT_PENDING_REVIEW','selection_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'manifest_sha256':{str(k):v for k,v in expected.items()},'script_sha256':sha(__file__),'role':'training_side_annotation_pool_not_independent_test','definition':'Opening edge at target sleeve end, including short sleeves; separate box per visible opening.','leakage_checks':['Resolved source paths exclude val/Core.','Exact source file SHA256 excludes val/Core and duplicates in this batch.'],'limitations':['Does not detect visually similar or re-encoded duplicate images.','These images were used for the segmentation model training.','Do not use Core cases from cuff diagnostics as fine-tuning data.','No new untouched test set or train/validation split is claimed.'],'cases':selected}
    (output/'cuff_annotation.html').write_text(HTML.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    metadata={**data,'cases':[{k:v for k,v in c.items() if k!='image_data'} for c in selected]}
    (output/'selection.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    (output/'summary.json').write_text(json.dumps({'images':len(selected),'strata':dict(Counter(c['category']+'/'+c['source_dataset'] for c in selected)),'protected_unique_paths':len(protected_paths),'protected_unique_file_hashes':len(protected_hashes),'excluded_candidate_reasons':dict(excluded),'reviewed':0,'source_files_exact_byte_copies':True,'status':data['status']},ensure_ascii=False,indent=2))
    (output/'README.txt').write_text('下载结果 zip 后解压，浏览器打开 cuff_annotation.html。黄色框是指定服装实例；绿色框是袖口。拖动画框，选择存在/不存在/无法判断，点击确认。可导出 JSON 并继续导入恢复。全部初始待标注，没有自动生成的真值。导出的 JSON 发回协助核对。训练侧样本不作为独立测试集。原图文件也在 source_images，可把本 zip 发回协助目视预审。')
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in output.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(output))
    print(f'ANNOTATION_IMAGES={len(selected)}',flush=True);print('RESULT_ZIP='+str(archive),flush=True)


if __name__=='__main__':main()
