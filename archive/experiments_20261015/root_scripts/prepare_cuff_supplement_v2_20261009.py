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


PREVIOUS_IMAGE_HASHES = set(['08d9b1b4aa091a8070c8286dcd4e38af3329d73749fad31bffcc16281013d7c1', '0bf3598b752c26f60f53a9069c0daabc91b3b313c74d626134599079871c2dab', '0f05867782d841a778e128e95eb0a2eeeba792f26152a56e0936dc32557a3210', '0f3dacae99c6dcb74e1cb1dfdb3b3ef9e8c2c6435f636ce1ebd27b0060c68cd0', '1da72025cc6181725befcfdbc4e7b0d5b561dd4ba11b0d767b8a44f416566c17', '25e6b962983339d1753774a8949f38180dc23741459832b69dc158f274102041', '27f2a3eb664c1d5ce4c3dbcbbb29c7eec19cf8071098dbe7212ef94230804bb9', '314bff0ecc5cac4b78f9f8b90a9c5eafaab87d9d82440d97fe5911f276a80197', '326374941ed7d55272b4b507714d93636cf1d1701e37214ba5632f2c44295170', '3a91a0bce3353fb89912db0ef27ebf159eac3dc3415f891972e2faa567cebe10', '3b3b16bf99316e32efa406b26b2b6558b7c8383ab11615ebcc35b04303fb7c26', '4024569d6963470f1d2d8273a34bd94375aaa732a9fdb1bdd720fa3cea6f7e17', '445095db54c8919092a64ca2c77ffc64773327c7d4f7e05beff28faa9f6cf63a', '58f622ae2fc17d2a56ac98cce5cc1ec7498ff39ed08386902800fe17f772935a', '5e3d1e85a953a96a88d2b4504cd375052cc691e66ef39f81f9023a9e83dce87b', '6142a23d073c1785bc6ac9e17dcbde04c8acb7fcd4686da69a7c7ae1d421642c', '6166cf50d86b3ac61b7d1c1bb060d9cb96c73be6fc47f97da48f86bdb570c81d', '6c1faa6f3fa57132543f18769fc7a0b3fed698f76adb25a074aa5d6cd28c66d5', '79a74253b3947bb006cf650750d66eb995a27e8ad376e99c6727896dbf790d90', '809588e1a5f33dd125821af573f794915d01f8d1cb0b23f24fd537715ffcbd43', '80e5ef8bc1b6013c2edc97e734277535d8b294230fb134a11660b1e8adb41a61', '8c52210cc74abbe38e8a151bda3ef0a13544c700c31ded02b93a9862e9ccd590', '909c25e98e480aea92dbd7b86e8eaef481fb1d2ca090769e0cee158262cdf383', '987aa5dcd862b406a3addc1f57c32e7f23decd087662c3c1cb78bef8de026d85', 'a36bc372b38756ea3caf5bcc43f4d0c548b75bff677a02e3c293e9bde69aca17', 'a91c3ad8eb22abdd7fc83f30074d541396ead78856e11a16b8066b94c4e99388', 'ad048c59fb04f42bf703b5344d9928d461aad6c925af43f114cdffaf26294013', 'aec42310b5350d9f2465f70edf40929e5556e45ff3435bb6aabe0130b6c68ebb', 'b17c91aa9f687020190918874d3b858ac16210849853bafc4db2883ce9a4028f', 'b7a4d8a5fb0d962c3f75d9263343bb3ee9438874906850d651ec809364f9bd80', 'b94b1b606c7e9c20a26a6941b6de4409dcf8b7e2b485e82e9d8ed3f376ac9ab7', 'bbbaea0f5049b31de205ec5021503b0bf8aa4259fed07b2f02ef1d5db5b053c6', 'c998dd71c37db8bb50915f500f751285cd9969f28915a76c74b4efd45679e106', 'caf9e364eafb4bd482bf78418478a85ac3cdd6f02ded1f288c93728bab3e0c81', 'cdea85f53ca6795504a641eab6742154c4fa6edf67d5b14fe85a2f7631851fe6', 'ced98476915a20cf5ef5bec3ebcc12792a0de656411b6f77f3a39820a7f4c3ea', 'd9aac98bd3fa22494fd98de318f5ab0e7e54c0ffe0b63a990e129afd0d00242c', 'd9d0675eb6291def49d9dba41be99d1acdb0d0cdc078bcb9c9dee90c816c27e7', 'dc04369a95d20bba15701956be1ad23bf2d637510a402f1ba4e31e4f7f5ced77', 'dcf5c21401cabe845aab03f7009a9936b2833fdbef0b5ba9527a9cd75dd6d5ad', 'e644a918096bf290dad38c705f510f66d24e6d77ab3411f1efa284359eb884bc', 'e7114c3546161ec22d8fd5e63e63d6a7c3e23210d05193b7515d0c0c8c5c668d', 'e7fa570eb1d147c3256a81876287ad1cf035998209020a5baa60653987d0f92a', 'e898fce3a3e88aabb692f70fab39557648b11e2caff7b0e22590d9e08ae32d34', 'ed066f78a1ea6b126e5b3be259776ce985f416bff2f7d9ef1a27478b70372759', 'efa0b4274ccfa3102dad25e86f95b924cc714ab6618fc7d6e3ca8a2ed35d86a2', 'fce894e96fea4aa4e80182eb427bd90c69d9ff00923a67c770ebc98add4d8571', 'fff009679bef54ff4fb71b3541fa36690cae777ceaca888d7d955f2e6756924d'])

HTML = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>袖口标注</title>
<style>body{font:16px sans-serif;margin:20px;background:#f4f5f7}button,select{font-size:16px;padding:8px;margin:4px}canvas{max-width:100%;height:auto;background:white;touch-action:none}#controls{position:sticky;top:0;background:white;padding:10px;z-index:2}#stage{max-width:1000px}p{max-width:1100px}#notice{color:#a32}</style>
<h1>袖口补充训练标注（第二批）</h1><p>只标黄色框指定的服装。有独立罗纹、纽扣或翻折袖口时框完整袖口带；没有独立袖口带时框袖子末端开口，包括短袖。左右分别画紧框，不框整条袖子、下摆、手表或内搭。边缘遮挡、截断、太模糊时选“无法判断”，不能当成“无袖口”。这些图片来自原分割训练集，不能当独立测试集。</p>
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
$('export').onclick=()=>{let result={...data,cases:cs.map(({image_data,...c})=>c),exported_at:new Date().toISOString(),reviewed_count:cs.filter(c=>c.review_status==='reviewed').length};let a=document.createElement('a'),u=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));a.href=u;a.download='cuff_supplement_v2_progress_20261009.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),2000)};
$('import').onclick=()=>$('file').click();$('file').onchange=async()=>{try{let f=$('file').files[0];if(!f)return;let r=JSON.parse(await f.text());if(r.selection_sha256!==data.selection_sha256)throw Error('不属于本批样本');applyRows(r.cases);persist();show();notice('标注已恢复。')}catch(e){notice('导入失败：'+e.message)}finally{$('file').value=''}};show();
</script></html>'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--per-stratum',type=int,default=8);args=p.parse_args()
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
            # Deterministic new-image sampling; no selection by development errors or sleeve labels.
            pool.sort(key=lambda r:hashlib.sha256(('cuff_supplement_v2_20261009:'+r['record_id']).encode()).hexdigest())
            count=0
            for r in pool:
                source=resolve_path(r['source_image']).resolve()
                if source in protected_paths:excluded['protected_path']+=1;continue
                if source in selected_paths:continue
                if source not in hashes:hashes[source]=sha(source)
                h=hashes[source]
                if h in protected_hashes:excluded['protected_exact_file_hash']+=1;continue
                if h in PREVIOUS_IMAGE_HASHES:excluded['previous_48_image_hash']+=1;continue
                if h in selected_hashes:continue
                with Image.open(source) as image:iw,ih=image.size
                if (iw,ih)!=(int(float(r['image_width'])),int(float(r['image_height']))):raise ValueError('Source dimensions differ: '+str(source))
                box=[float(r['bbox_'+k]) for k in ('x1','y1','x2','y2')]
                if not (0<=box[0]<box[2]<=iw and 0<=box[1]<box[3]<=ih):raise ValueError('Invalid native garment bbox')
                selected.append({'record_id':r['record_id'],'source_dataset':source_dataset,'category':category,'fine_category':r['fine_category'],'source_image_reference':r['source_image'],'garment_id':r['garment_id'],'garment_bbox':box,'image_width':iw,'image_height':ih,'source_image_sha256':h,'source_file':source,'presence':'pending','cuff_boxes':[],'review_status':'pending'})
                selected_hashes.add(h);selected_paths.add(source);count+=1
                if count==args.per_stratum:break
            if count<args.per_stratum:raise ValueError(f'Not enough isolated images in {category}/{source_dataset}: {count}')
    output=root/'reports/reruns/recovery_20261009'/('cuff_supplement_v2_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));(output/'source_images').mkdir(parents=True)
    for case in selected:
        source=case.pop('source_file');file_name=case['record_id']+source.suffix.lower();raw=source.read_bytes()
        (output/'source_images'/file_name).write_bytes(raw);case['source_file_relative']='source_images/'+file_name
        mime='image/png' if source.suffix.lower()=='.png' else 'image/jpeg';case['image_data']='data:'+mime+';base64,'+base64.b64encode(raw).decode()
    ids=[(c['record_id'],c['source_image_sha256']) for c in selected]
    data={'status':'TRAINING_SIDE_SUPPLEMENT_V2_PENDING_REVIEW','selection_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'manifest_sha256':{str(k):v for k,v in expected.items()},'script_sha256':sha(__file__),'role':'training_side_annotation_pool_not_independent_test','definition':'Entire independent cuff band if present; otherwise visible sleeve-end opening, including short sleeves. Separate box per visible target sleeve opening.','leakage_checks':['Resolved source paths exclude val/Core.','Exact source file SHA256 excludes val/Core, all previous 48 annotation images, and duplicates in this batch.'],'limitations':['Does not detect visually similar or re-encoded duplicate images.','These images were used for the segmentation model training.','Do not use Core cases from cuff diagnostics as fine-tuning data.','No new untouched test set or train/validation split is claimed.'],'cases':selected}
    (output/'cuff_annotation.html').write_text(HTML.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    metadata={**data,'cases':[{k:v for k,v in c.items() if k!='image_data'} for c in selected]}
    (output/'selection.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    (output/'summary.json').write_text(json.dumps({'images':len(selected),'strata':dict(Counter(c['category']+'/'+c['source_dataset'] for c in selected)),'protected_unique_paths':len(protected_paths),'previous_annotation_hashes_excluded':len(PREVIOUS_IMAGE_HASHES),'protected_unique_file_hashes':len(protected_hashes),'excluded_candidate_reasons':dict(excluded),'reviewed':0,'source_files_exact_byte_copies':True,'status':data['status']},ensure_ascii=False,indent=2))
    (output/'README.txt').write_text('下载结果 zip 后解压，浏览器打开 cuff_annotation.html。黄色框是指定服装实例；绿色框是袖口。拖动画框，选择存在/不存在/无法判断，点击确认。可导出 JSON 并继续导入恢复。全部初始待标注，没有自动生成的真值。导出的 JSON 发回协助核对。训练侧样本不作为独立测试集。原图文件也在 source_images，可把本 zip 发回协助目视预审。')
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in output.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(output))
    print(f'ANNOTATION_IMAGES={len(selected)}',flush=True);print('RESULT_ZIP='+str(archive),flush=True)


if __name__=='__main__':main()
