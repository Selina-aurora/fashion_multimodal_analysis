"""Export a new cuff holdout annotation batch from original segmentation val.
Exclude original train, Core, and all prior cuff images by path/hash.
Segmentation used these images for validation: not blind end-to-end evidence.
Freeze V3 before annotation; no sleeve predictions shown or training performed.
"""
import argparse,base64,csv,hashlib,json,zipfile
from pathlib import Path
from datetime import datetime
from collections import Counter
HTML = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>袖口标注</title>
<style>body{font:16px sans-serif;margin:20px;background:#f4f5f7}button,select{font-size:16px;padding:8px;margin:4px}canvas{max-width:100%;height:auto;background:white;touch-action:none}#controls{position:sticky;top:0;background:white;padding:10px;z-index:2}#stage{max-width:1000px}p{max-width:1100px}#notice{color:#a32}</style>
<h1>袖口保留评估标注</h1><p>只标黄色框指定的服装。有独立罗纹、纽扣或翻折袖口时框完整袖口带；没有独立袖口带时框末端开口，包括短袖。左右分别画紧框，不框整条袖子、下摆、手表或内搭。边缘遮挡、截断、太模糊时选“无法判断”，不能当成“无袖口”。这些图片来自原分割验证集，未参与本次袖口训练和调参；不能称整个分割流程的全新盲测。仅用于冻结袖口候选评估，不加入袖口训练。</p>
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
$('export').onclick=()=>{let result={...data,cases:cs.map(({image_data,...c})=>c),exported_at:new Date().toISOString(),reviewed_count:cs.filter(c=>c.review_status==='reviewed').length};let a=document.createElement('a'),u=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));a.href=u;a.download='cuff_holdout_progress_20261009.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),2000)};
$('import').onclick=()=>$('file').click();$('file').onchange=async()=>{try{let f=$('file').files[0];if(!f)return;let r=JSON.parse(await f.text());if(r.selection_sha256!==data.selection_sha256)throw Error('不属于本批样本');applyRows(r.cases);persist();show();notice('标注已恢复。')}catch(e){notice('导入失败：'+e.message)}finally{$('file').value=''}};show();
</script></html>'''


PREVIOUS_HASHES=set(['0149097caf09d96edaeeac426a685c2938ced685d0c8b2be37386390f009319c', '06f46dfcecc69065a9155532f1c72cc9ff9386d83640cc4a2475ee77d524aa9d', '08d9b1b4aa091a8070c8286dcd4e38af3329d73749fad31bffcc16281013d7c1', '0a753a36b79a70e9f892d1da614d252dc6374966ab75885378fc97754c009858', '0bb7d4f41c990e32e822a0e6a4cae732c92b09f7af4f75dd92f0aa572b96f113', '0bf3598b752c26f60f53a9069c0daabc91b3b313c74d626134599079871c2dab', '0e2c63bd53932c0186dc6718771ef2c1f9f7a2000d68d299751fef6e8cf223a2', '0f05867782d841a778e128e95eb0a2eeeba792f26152a56e0936dc32557a3210', '0f3dacae99c6dcb74e1cb1dfdb3b3ef9e8c2c6435f636ce1ebd27b0060c68cd0', '13edbc40d887ab6f2c3f0be4a691f34e361478fcbd398a6d399d3ba40f4d3c47', '1b91c7fb91d0d3fa58958a7b78e757797cdfc3d1cb24b97ef5173f176b0583ec', '1da72025cc6181725befcfdbc4e7b0d5b561dd4ba11b0d767b8a44f416566c17', '210d6b63217b0916f5b4e764abeaca14c4fd8c1e8065be95e1bec7844918ca92', '22014295610a857d62b2f8dffa928ae61325fbd440ae0fc2f52ae0fdf3b7c5b3', '25e6b962983339d1753774a8949f38180dc23741459832b69dc158f274102041', '26255f0d47cb197adb875aa9f7d53347b811fb04f51b9df1e4c7b6d5b1f63eba', '273ef08976d40d5ccddb1d577ba9d66b68294f0d4f263e88863432bf549a8c60', '27f2a3eb664c1d5ce4c3dbcbbb29c7eec19cf8071098dbe7212ef94230804bb9', '314bff0ecc5cac4b78f9f8b90a9c5eafaab87d9d82440d97fe5911f276a80197', '32486f85a17e3150ea267a5e383287cf48ef4db9e2daec31248d29f067c8a3d2', '326374941ed7d55272b4b507714d93636cf1d1701e37214ba5632f2c44295170', '37b741a29840dd1e67ca55f786ee61b52369d9562f6aef4826aabc6969743503', '3996f34402ba1a5a99c7926f46f0deb0ff1128b14ff67cd5e4cf80a028403b89', '3a91a0bce3353fb89912db0ef27ebf159eac3dc3415f891972e2faa567cebe10', '3b3b16bf99316e32efa406b26b2b6558b7c8383ab11615ebcc35b04303fb7c26', '3e4a93fbffaed3dc72d820bb6ffa696938d6a7a59dc9bf84bf67c9edfcb45a37', '4024569d6963470f1d2d8273a34bd94375aaa732a9fdb1bdd720fa3cea6f7e17', '443979d93bd7d8dbe366397f7f99fce742b98e903d9f958739b7665d2d4c6dbd', '445095db54c8919092a64ca2c77ffc64773327c7d4f7e05beff28faa9f6cf63a', '4c78e17f2d40831881362bdc54924382b1b763841efea44878ef169285857e53', '58f622ae2fc17d2a56ac98cce5cc1ec7498ff39ed08386902800fe17f772935a', '5aab878c6dc8d895b2f1d0d16357d9732d62ab87bed9a85eb202524f26c9c3a2', '5e3d1e85a953a96a88d2b4504cd375052cc691e66ef39f81f9023a9e83dce87b', '6142a23d073c1785bc6ac9e17dcbde04c8acb7fcd4686da69a7c7ae1d421642c', '6166cf50d86b3ac61b7d1c1bb060d9cb96c73be6fc47f97da48f86bdb570c81d', '623f67e40c5b22dc2c7d234ab89791bbdd1bf225456053f54489526c4c3026ef', '63036b2070d9db29dec5a1b2a4a04ae717db229fc738671c9f1b203273e15e52', '668737b3fde3e26e6ba813866dd6bfee648918173a9be18200315a32a6db4d9a', '697491391f2b90e6fec840a55252b0facd0e08d7db5426bb4c15cf10ae8d0efc', '6c1faa6f3fa57132543f18769fc7a0b3fed698f76adb25a074aa5d6cd28c66d5', '763deaaf4e1f0cac844f5f8de0e8555a6150cbb541351422f22cbb4ff9c59036', '794153c26f77474628c69a77677b1b486ba690f92e979d0d20247adba902aae1', '79a74253b3947bb006cf650750d66eb995a27e8ad376e99c6727896dbf790d90', '7c7bde53c254aa1d18526176e27c5d4e4367dcff35a806d74f05cb556de93dfa', '809588e1a5f33dd125821af573f794915d01f8d1cb0b23f24fd537715ffcbd43', '80e5ef8bc1b6013c2edc97e734277535d8b294230fb134a11660b1e8adb41a61', '8c52210cc74abbe38e8a151bda3ef0a13544c700c31ded02b93a9862e9ccd590', '8cc97b4225f143db06e95d9474afb150b596000500f6246dcd0333a954398fa2', '8cde0520f1821ee5bb86541752eceb04f184cf6f2a4eb1c90bb6b16313739d74', '909c25e98e480aea92dbd7b86e8eaef481fb1d2ca090769e0cee158262cdf383', '975646698d5662c8efc0ca971286796ca667e23faf3415382d2c7c2d41bdbbf4', '987aa5dcd862b406a3addc1f57c32e7f23decd087662c3c1cb78bef8de026d85', '9cd09ad6078faa67f206c78e5460e2c6c582da3fe459913f61a1a58eafda7456', '9decaffee6d76d1f5378044177c083df84e4aeef2cb62db8ff6f302da89aafca', 'a00860ac801cb6962ff5bdc12e7558625efcea8308dc09f25a85bee957f5157e', 'a36bc372b38756ea3caf5bcc43f4d0c548b75bff677a02e3c293e9bde69aca17', 'a652ebd38dd031433d5fdd14aedbf3d907c9526fca8620517d25e9c43d873ae5', 'a8dc9594bb9af69776e8719b02cdb0b0d00efd26b1e17fc1ffa3f542b24d28b5', 'a91c3ad8eb22abdd7fc83f30074d541396ead78856e11a16b8066b94c4e99388', 'ad048c59fb04f42bf703b5344d9928d461aad6c925af43f114cdffaf26294013', 'aea5d83d7895e612dc8082d756dc3d8ce94ea8a853a42016bbbea2fe31c8b07e', 'aec42310b5350d9f2465f70edf40929e5556e45ff3435bb6aabe0130b6c68ebb', 'b17c91aa9f687020190918874d3b858ac16210849853bafc4db2883ce9a4028f', 'b7a4d8a5fb0d962c3f75d9263343bb3ee9438874906850d651ec809364f9bd80', 'b94b1b606c7e9c20a26a6941b6de4409dcf8b7e2b485e82e9d8ed3f376ac9ab7', 'bbbaea0f5049b31de205ec5021503b0bf8aa4259fed07b2f02ef1d5db5b053c6', 'bd8500b7025471edda619f46e0ed4d9d8258607f8ebd32b299004268e154fe28', 'c2ac35f9e4c0f0ea8b072a678107173ed226c49601df2f08bcdbdaf82fc325fb', 'c441f16ed71e3b716a785d73a73525bb2aacd2f0462452901107d7fbf0652c9b', 'c56eb36234dd90204385b6999e0ed06ed4565444ea8f8fbc05dd307946cbb231', 'c998dd71c37db8bb50915f500f751285cd9969f28915a76c74b4efd45679e106', 'caf9e364eafb4bd482bf78418478a85ac3cdd6f02ded1f288c93728bab3e0c81', 'cdea85f53ca6795504a641eab6742154c4fa6edf67d5b14fe85a2f7631851fe6', 'ce7754f071ca4b0b6718d490d0a00282ed1ccc0201a5c979d6826a84d6d279ea', 'ced98476915a20cf5ef5bec3ebcc12792a0de656411b6f77f3a39820a7f4c3ea', 'd9aac98bd3fa22494fd98de318f5ab0e7e54c0ffe0b63a990e129afd0d00242c', 'd9d0675eb6291def49d9dba41be99d1acdb0d0cdc078bcb9c9dee90c816c27e7', 'db517cab4adc01eefb4dd0a80ce8207d75e41ead76d075ddf27a2812a657798e', 'dc04369a95d20bba15701956be1ad23bf2d637510a402f1ba4e31e4f7f5ced77', 'dcf5c21401cabe845aab03f7009a9936b2833fdbef0b5ba9527a9cd75dd6d5ad', 'dd8e888565919d38b48a5aa8898daef60baa96239f843630db596b3fe94c5501', 'e14ab2225c14d5f90a72aa12df8ad57a935ff5896f75eb54ef86c96f5513d76b', 'e2bff41535439de3654d4f4bf4a6d9798d21fbd0379e664d133aafc787313aa8', 'e3e5d1a1a4aaf793baefe4fa852ba0cb084d27b06b6102f44fd8eaedf31bd1c1', 'e644a918096bf290dad38c705f510f66d24e6d77ab3411f1efa284359eb884bc', 'e7114c3546161ec22d8fd5e63e63d6a7c3e23210d05193b7515d0c0c8c5c668d', 'e7fa570eb1d147c3256a81876287ad1cf035998209020a5baa60653987d0f92a', 'e898fce3a3e88aabb692f70fab39557648b11e2caff7b0e22590d9e08ae32d34', 'ed066f78a1ea6b126e5b3be259776ce985f416bff2f7d9ef1a27478b70372759', 'efa0b4274ccfa3102dad25e86f95b924cc714ab6618fc7d6e3ca8a2ed35d86a2', 'f10ea79ee72b94371b7e7233820e4432d6d656f724e8a042e6062d1d3631e155', 'f230fb79c3dfe4cf36f723aab47ef959c77b02d9b367bd37832b739fdee27376', 'f62112f417dcf217f0ca511bdd0237258507a19ad0b3899d5ecc5f9ef152426d', 'fc463ea26c058b94bfe6f5398bff4951b37f0bd4c1eda2b65fb3bef22f33a5c5', 'fce894e96fea4aa4e80182eb427bd90c69d9ff00923a67c770ebc98add4d8571', 'fff009679bef54ff4fb71b3541fa36690cae777ceaca888d7d955f2e6756924d'])

CANDIDATE_SHA='52f8d78ffa032b9135eea8ba8ea511e942e2f12cd793f49bbca324f720a91c84'
CANDIDATE='reports/reruns/recovery_20261009/cuff_expanded_train_v3_20261009_052743_791119/checkpoint_best.pth'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='')as f:return list(csv.DictReader(f))
def main():
    from PIL import Image
    from fashion_multimodal_analysis.common.paths import project_root,resolve_path
    root=project_root();candidate=root/CANDIDATE
    if sha(candidate)!=CANDIDATE_SHA:raise ValueError('Frozen V3 candidate changed')
    train=resolve_path('configs/prd_8class_train_v5_balanced150_20261008.csv');val=resolve_path('configs/prd_8class_val_v2_full_targets.csv');core=resolve_path('benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv')
    expected={train:'e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f',val:'39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93',core:'c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89'}
    for p,h in expected.items():
        if sha(p)!=h:raise ValueError('Original manifest changed: '+str(p))
    protected={resolve_path(r['source_image']).resolve()for p in [train,core]for r in rows(p)}
    print(f'Checking {len(protected)} train/Core source files...',flush=True)
    protected_hashes={sha(p)for p in protected}|PREVIOUS_HASHES
    all_rows=rows(val);selected=[];used=set();strata_counts={};excluded=Counter()
    for category in ['top','outerwear','dress']:
        for ds in ['DeepFashion2','Fashionpedia']:
            pool=[r for r in all_rows if r['garment_category']==category and r['source_dataset']==ds]
            pool.sort(key=lambda r:hashlib.sha256(('cuff_holdout_20261009:'+r['record_id']).encode()).hexdigest())
            count=0
            for r in pool:
                source=resolve_path(r['source_image']).resolve()
                if source in protected:excluded['protected_path']+=1;continue
                h=sha(source)
                if h in protected_hashes:excluded['protected_or_previous_hash']+=1;continue
                if h in used:excluded['batch_duplicate_source']+=1;continue
                with Image.open(source)as im:w,height=im.size
                if [w,height]!=[int(float(r['image_width'])),int(float(r['image_height']))]:raise ValueError('Native dimensions changed')
                box=[float(r['bbox_'+k])for k in ['x1','y1','x2','y2']]
                if not(0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=height):raise ValueError('Invalid garment bbox')
                selected.append({'record_id':r['record_id'],'source_dataset':ds,'category':category,'fine_category':r['fine_category'],'source_image_reference':r['source_image'],'garment_id':r['garment_id'],'garment_bbox':box,'image_width':w,'image_height':height,'source_image_sha256':h,'source_file':source,'presence':'pending','cuff_boxes':[],'review_status':'pending'})
                used.add(h);count+=1
                if count==3:break
            strata_counts[category+'/'+ds]=count
    if not selected:raise RuntimeError('No eligible new holdout images')
    out=root/'reports/reruns/recovery_20261009'/('cuff_holdout_annotation_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));(out/'source_images').mkdir(parents=True)
    for c in selected:
        source=c.pop('source_file');name=c['record_id']+source.suffix.lower();raw=source.read_bytes();(out/'source_images'/name).write_bytes(raw);c['source_file_relative']='source_images/'+name
        mime='image/png'if source.suffix.lower()=='.png'else'image/jpeg';c['image_data']='data:'+mime+';base64,'+base64.b64encode(raw).decode()
    ids=[(c['record_id'],c['source_image_sha256'])for c in selected]
    data={'status':'FROZEN_CUFF_HOLDOUT_PENDING_ANNOTATION','selection_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'candidate_checkpoint':str(candidate),'candidate_sha256':CANDIDATE_SHA,'candidate_epoch':27,'role':'cuff_holdout_not_used_for_cuff_training_or_tuning; original_segmentation_val','definition':'Whole independent cuff band when present; otherwise visible sleeve-end opening. Target garment only, separate box per visible opening.','limitations':['Original segmentation model used these source images for validation, so not a blind end-to-end benchmark.','Exact file hashes/path exclusion does not establish perceptual deduplication.','Small holdout batch; no formal PRD acceptance claim.','Freeze this candidate and report all reviewed usable cases, not only successes.','Do not train or tune thresholds on this batch; if later used for development it loses holdout role.'],'cases':selected}
    (out/'cuff_holdout_annotation.html').write_text(HTML.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</',r'<\/')),encoding='utf-8')
    meta={**data,'cases':[{k:v for k,v in c.items()if k!='image_data'}for c in selected]};(out/'selection.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    (out/'summary.json').write_text(json.dumps({'images':len(selected),'strata':strata_counts,'original_train_Core_paths_checked':len(protected),'prior_annotation_hashes_excluded':96,'excluded':dict(excluded),'candidate_sha256':CANDIDATE_SHA,'reviewed':0,'model_weights_modified':False},ensure_ascii=False,indent=2))
    (out/'README.txt').write_text('浏览器打开 cuff_holdout_annotation.html。该批只用于冻结袖口第三版的保留评估，不加入训练或调参。请把整个zip发回协助预审，再确认标注。图片来自原分割验证集，因此不能宣称整个流程的独立盲测。无袖口与无法判断必须区分；只标黄色框指定衣物。候选权重在挑选图片前已验证冻结。实际图片数不超过18张，部分分层可能不足3张。')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED)as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(out))
    print('IMAGES='+str(len(selected)),flush=True);print('RESULT_ZIP='+str(archive),flush=True)
if __name__=='__main__':main()
