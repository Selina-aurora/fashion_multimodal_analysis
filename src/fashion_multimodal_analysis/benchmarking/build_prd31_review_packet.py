"""基准集管理：保留训练、开发、固定回归/审核划分，冻结后的标签不能随结果调整。

Build an offline human GT review page. Color labels require untinted source images.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image

from fashion_multimodal_analysis.benchmarking.prepare_prd31_recovery import (
    digest,
    labels_for,
)
from fashion_multimodal_analysis.common.io import read_csv, write_json
from fashion_multimodal_analysis.common.paths import project_root, resolve_path

PAGE = r"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PRD 3.1 人工真值审核</title>
<style>
body{font:16px/1.55 system-ui,sans-serif;margin:0;background:#f3f5f8;color:#18253a}
header{padding:20px 28px;background:#173d70;color:white}h1{font-size:23px;margin:0 0 6px}
main{max-width:1400px;margin:auto;padding:22px}.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:0 0 18px}
button,select,input{font:inherit;padding:8px;border:1px solid #b4c1d2;border-radius:6px;background:white;color:#18253a}
button{cursor:pointer}button:disabled{cursor:default;opacity:.45}.layout{display:grid;grid-template-columns:minmax(320px,1.2fr) minmax(350px,1fr);gap:22px}
section{background:white;padding:18px;border-radius:10px;border:1px solid #d7e0ea}canvas{max-width:100%;height:auto;background:#eee;touch-action:none;cursor:crosshair}
.warn{background:#fff2cf;padding:12px;border-radius:6px;margin:12px 0}.good{color:#087344}small{color:#50647d}
table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #e1e7ef;padding:9px 6px}
.row{border-bottom:1px solid #e1e7ef;margin-bottom:12px;padding-bottom:12px}.row label{display:inline-block;margin:6px 8px 6px 0}
.box{font-size:13px;overflow-wrap:anywhere}#message{min-height:28px;color:#923600}@media(max-width:850px){.layout{grid-template-columns:1fr}main{padding:12px}}
</style>
<header><h1>PRD 3.1 人工真值审核</h1><div>按原图标注局部区域和属性。页面不显示模型预测，避免把预测当作真值。</div></header>
<main><div class="bar"><button id="prev">上一张</button><button id="next">下一张</button><button id="pending">下一张待审核</button>
<select id="jump" aria-label="选择样本"></select><label>审核人 <input id="reviewer" placeholder="姓名或稳定代号"></label><span id="progress"></span></div>
<div class="layout"><section><h2 id="title"></h2><small id="source"></small><div id="warning"></div>
<canvas id="canvas" aria-label="在原图上拖动画出当前目标区域的框"></canvas><p>黄色框是服装 GT 框；红色框是当前局部目标。选择区域后，在原图上拖拽画框。框会自动转换为原图坐标。</p>
<div class="box" id="coords"></div><div id="message" role="status"></div></section>
<section><h3>3.1.2 局部区域</h3><select id="query" aria-label="当前局部区域"></select><div id="ground"></div>
<h3>3.1.3 属性</h3><div id="attrs"></div><button id="save">保存当前样本审核</button>
<p><small>看不清或多种答案都成立时选择“歧义”，并填写原因；不适用的属性已经排除。不会自动补标签。</small></p></section></div>
<section style="margin-top:20px"><div class="bar"><button id="exportGround">导出 grounding_reviewed.csv</button>
<button id="exportAttrs">导出 attribute_reviewed.csv</button><button id="backup">导出审核进度 JSON</button>
<label>恢复进度 <input type="file" id="restore" accept="application/json"></label></div>
<small>完整审核后，把两个 CSV 放到项目 benchmark/prd_3_1_v2/review/，再运行 GPU 脚本的 --phase finish。歧义或不存在目标不会被算成正例。</small></section>
</main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('payload').textContent),$=id=>document.getElementById(id);
const storageKey='prd31-review-'+data.fingerprint;
let ground=data.grounding,attrs=data.attributes,index=0,img=new Image(),drag=null,selectedBox=null,activeQuery='';
function state(){return {fingerprint:data.fingerprint,ground,attrs,index,reviewer:$('reviewer').value};}
function persist(){try{localStorage.setItem(storageKey,JSON.stringify(state()));}catch(e){$('message').textContent='浏览器未保存缓存，请导出进度 JSON。';}}
function load(s){if(s.fingerprint!==data.fingerprint)throw new Error('进度文件属于其他候选集');ground=s.ground;attrs=s.attrs;index=Math.max(0,Math.min(data.cases.length-1,s.index||0));$('reviewer').value=s.reviewer||'';}
try{const cached=localStorage.getItem(storageKey);if(cached)load(JSON.parse(cached));}catch(e){$('message').textContent='缓存无法恢复，可导入进度 JSON。';}
data.cases.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=c.sample_id+' · '+c.garment_category;$('jump').appendChild(o);});
function textNode(tag,text){let n=document.createElement(tag);n.textContent=text;return n;}
function selectOptions(id,options,value){let s=document.createElement('select');s.id=id;options.forEach(([v,t])=>{let o=document.createElement('option');o.value=v;o.textContent=t;s.appendChild(o);});s.value=value;return s;}
function currentGround(){return ground.find(r=>r.query_id===activeQuery);}
function complete(c){return ground.filter(r=>r.sample_id===c.sample_id).every(r=>r.review_status==='reviewed')&&attrs.filter(r=>r.sample_id===c.sample_id&&r.applicability==='eligible').every(r=>r.review_status==='reviewed');}
function draw(){const cv=$('canvas'),ctx=cv.getContext('2d'),c=data.cases[index];ctx.clearRect(0,0,cv.width,cv.height);ctx.drawImage(img,0,0,cv.width,cv.height);
 if(!c.raw_available)return;
 const sx=cv.width/c.image_width,sy=cv.height/c.image_height;
 function box(b,color){ctx.strokeStyle=color;ctx.lineWidth=3;ctx.strokeRect(b[0]*sx,b[1]*sy,(b[2]-b[0])*sx,(b[3]-b[1])*sy);}
 box(c.garment_bbox,'#ffc600');if(selectedBox)box(selectedBox,'#ff355c');
 $('coords').textContent=selectedBox?'局部框（原图）: '+selectedBox.map(x=>x.toFixed(1)).join(', '):'还没有局部框';}
function showGround(){activeQuery=$('query').value;const row=currentGround(),node=$('ground');node.replaceChildren();selectedBox=null;
 if(!row){node.textContent='此样本没有局部区域审核任务。';draw();return;}
 node.appendChild(textNode('p','目标：'+row.query_text));
 node.appendChild(selectOptions('presence',[['','待选择'],['yes','存在目标'],['no','不存在目标']],row.target_present));
 let amb=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.id='groundAmb';check.checked=row.ambiguous==='1';amb.append(check,' 歧义');node.appendChild(amb);
 let note=document.createElement('input');note.id='groundNote';note.placeholder='歧义原因 / 标注说明';note.value=row.exclude_reason||row.annotation_note||'';node.appendChild(note);
 if(row.gt_region_bbox_x1!=='')selectedBox=['x1','y1','x2','y2'].map(k=>Number(row['gt_region_bbox_'+k]));
 node.querySelectorAll('input,select').forEach(n=>n.disabled=!data.cases[index].raw_available);draw();}
function show(){const c=data.cases[index];$('jump').value=index;$('title').textContent=c.sample_id+' · '+c.garment_category;$('source').textContent=c.source_image;
 $('warning').replaceChildren();if(!c.raw_available){let n=textNode('div','原始图片尚未接入。当前只是染色 GT 预览，不能据此审核颜色或完成真值。请在 AutoDL 运行 prepare 重建此页面。');n.className='warn';$('warning').appendChild(n);}
 $('progress').textContent='已审核 '+data.cases.filter(complete).length+' / '+data.cases.length+' 张';
 $('query').replaceChildren();ground.filter(r=>r.sample_id===c.sample_id).forEach(r=>{let o=document.createElement('option');o.value=r.query_id;o.textContent=r.target_region+(r.review_status==='reviewed'?' ✓':'');$('query').appendChild(o);});
 let node=$('attrs');node.replaceChildren();attrs.filter(r=>r.sample_id===c.sample_id).forEach(r=>{
 let d=document.createElement('div');d.className='row';d.dataset.attribute=r.attribute_name;d.appendChild(textNode('strong',r.attribute_name));
 if(r.applicability!=='eligible'){d.appendChild(textNode('p','不适用'));node.appendChild(d);return;}
 let sel=selectOptions('attr_'+r.attribute_name,[['','待标注'],...r.allowed_labels.map(x=>[x,x])],r.gt_label);d.appendChild(sel);
 let l=document.createElement('label'),b=document.createElement('input');b.type='checkbox';b.className='amb';b.checked=r.ambiguous==='1';l.append(b,' 歧义');d.appendChild(l);
 let n=document.createElement('input');n.className='note';n.placeholder='歧义原因 / 说明';n.value=r.exclude_reason||r.annotation_note||'';d.appendChild(n);
 d.querySelectorAll('input,select').forEach(e=>e.disabled=!c.raw_available);node.appendChild(d);});
 $('save').disabled=!c.raw_available;
 img.onload=()=>{let scale=Math.min(1,1000/img.width);$('canvas').width=Math.round(img.width*scale);$('canvas').height=Math.round(img.height*scale);showGround();};
 img.src=c.preview;persist();}
function saveGround(){const r=currentGround();if(!r)return;
 r.target_present=$('presence').value;r.ambiguous=$('groundAmb').checked?'1':'0';r.annotation_note=$('groundNote').value.trim();r.exclude_reason=r.ambiguous==='1'?r.annotation_note:'';
 if(selectedBox)['x1','y1','x2','y2'].forEach((k,i)=>r['gt_region_bbox_'+k]=selectedBox[i].toFixed(2));
 const done=r.ambiguous==='1'?!!r.exclude_reason:r.target_present==='no'||(r.target_present==='yes'&&!!selectedBox);
 r.reviewer=$('reviewer').value.trim();r.review_status=done&&r.reviewer?'reviewed':'pending';}
function save(){if(!data.cases[index].raw_available)return;saveGround();
 attrs.filter(r=>r.sample_id===data.cases[index].sample_id&&r.applicability==='eligible').forEach(r=>{
 let d=$('attr_'+r.attribute_name).parentNode;r.gt_label=$('attr_'+r.attribute_name).value;r.ambiguous=d.querySelector('.amb').checked?'1':'0';
 r.annotation_note=d.querySelector('.note').value.trim();r.exclude_reason=r.ambiguous==='1'?r.annotation_note:'';r.reviewer=$('reviewer').value.trim();
 r.review_status=r.reviewer&&(r.ambiguous==='1'?!!r.exclude_reason:!!r.gt_label)?'reviewed':'pending';});
 persist();$('message').textContent=$('reviewer').value.trim()?'已保存。未填完的任务继续保持待审核。':'请填写审核人。';$('progress').textContent='已审核 '+data.cases.filter(complete).length+' / '+data.cases.length+' 张';}
function position(e){let r=$('canvas').getBoundingClientRect(),c=data.cases[index];return [Math.max(0,Math.min(c.image_width,(e.clientX-r.left)/r.width*c.image_width)),Math.max(0,Math.min(c.image_height,(e.clientY-r.top)/r.height*c.image_height))];}
$('canvas').addEventListener('pointerdown',e=>{if(!data.cases[index].raw_available||!currentGround())return;drag=position(e);$('canvas').setPointerCapture(e.pointerId);});
$('canvas').addEventListener('pointermove',e=>{if(!drag)return;let p=position(e);selectedBox=[Math.min(drag[0],p[0]),Math.min(drag[1],p[1]),Math.max(drag[0],p[0]),Math.max(drag[1],p[1])];draw();});
$('canvas').addEventListener('pointerup',e=>{if(drag){let p=position(e);selectedBox=[Math.min(drag[0],p[0]),Math.min(drag[1],p[1]),Math.max(drag[0],p[0]),Math.max(drag[1],p[1])];if(selectedBox[2]-selectedBox[0]<1||selectedBox[3]-selectedBox[1]<1)selectedBox=null;drag=null;draw();}});
$('query').onchange=()=>{save();showGround();};
$('save').onclick=save;
function move(n){save();index=Math.max(0,Math.min(data.cases.length-1,n));show();}
$('prev').onclick=()=>move(index-1);$('next').onclick=()=>move(index+1);$('jump').onchange=()=>move(Number($('jump').value));
$('pending').onclick=()=>{save();for(let step=1;step<=data.cases.length;step++){let i=(index+step)%data.cases.length;if(!complete(data.cases[i])){move(i);return;}}$('message').textContent='全部候选已审核，导出两个 CSV 后运行冻结检查。';};
function download(name,value,type){let a=document.createElement('a'),url=URL.createObjectURL(new Blob([value],{type}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function csv(rows,fields){let esc=v=>'"'+String(v??'').replaceAll('"','""')+'"';return '\ufeff'+[fields.map(esc).join(','),...rows.map(r=>fields.map(k=>esc(r[k])).join(','))].join('\r\n');}
$('exportGround').onclick=()=>{save();download('grounding_reviewed.csv',csv(ground,data.ground_fields),'text/csv;charset=utf-8');};
$('exportAttrs').onclick=()=>{save();download('attribute_reviewed.csv',csv(attrs,data.attribute_fields),'text/csv;charset=utf-8');};
$('backup').onclick=()=>{save();download('prd31_review_progress.json',JSON.stringify(state()),'application/json');};
$('restore').onchange=async e=>{try{load(JSON.parse(await e.target.files[0].text()));show();}catch(error){$('message').textContent=error.message;}};
$('reviewer').onchange=persist;show();
</script></html>"""


def image_url(path: str | Path) -> Any:
    """图像 url。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((1200, 1200))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def reuse_reviews(
    drafts: Any, path: str | Path, keys: list[str], editable: bool
) -> Any:
    """reuse reviews。

    Args:
        drafts: drafts。
        path: 要读取或写入的文件路径。
        keys: 标识。
        editable: editable。

    Returns:
        返回 output，由函数体中同名变量的计算/收集过程得到。

    Raises:
        ValueError: Existing review belongs to a changed candidate; preserve the CSV and
        resolve identity
    """
    if not path.is_file():
        return drafts
    existing = {tuple(r[k] for k in keys): r for r in read_csv(path)}
    output = []
    for row in drafts:
        review = existing.get(tuple(row[k] for k in keys))
        if review:
            if any(
                str(review.get(k, "")) != str(v)
                for k, v in row.items()
                if k not in editable
            ):
                raise ValueError(
                    "Existing review belongs to a changed candidate; preserve the CSV and resolve identity"
                )
            row = {**row, **{k: review.get(k, "") for k in editable}}
        output.append(row)
    return output


def build(root: Path | None = None, output: Any = None) -> Any:
    """构建。

    Args:
        root: 当前项目根目录。
        output: 模型预测或输出文件位置，具体由本函数的读写操作决定。

    Returns:
        结果字典包含 review_page, cases, source_images_ready, status。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    root = root or project_root()
    base = root / "benchmark/prd_3_1_v2"
    ground = read_csv(base / "drafts/grounding_test.csv")
    attrs = read_csv(base / "drafts/attribute_test.csv")
    from fashion_multimodal_analysis.benchmarking.freeze_prd31_reviewed_tests import (
        EDITABLE_ATTRIBUTE,
        EDITABLE_GROUND,
    )

    ground = reuse_reviews(
        ground, base / "review/grounding_reviewed.csv", ("query_id",), EDITABLE_GROUND
    )
    attrs = reuse_reviews(
        attrs,
        base / "review/attribute_reviewed.csv",
        ("sample_id", "attribute_name"),
        EDITABLE_ATTRIBUTE,
    )
    cases = read_csv(base / "drafts/end_to_end_test.csv")
    rows = []
    for case in cases:
        raw = resolve_path(case["source_image"])
        overlay = resolve_path(case["gt_overlay_path"])
        ready = raw.is_file()
        if ready:
            with Image.open(raw) as original:
                if original.size != (
                    int(case["image_width"]),
                    int(case["image_height"]),
                ):
                    raise ValueError(
                        "Original image dimensions differ from the reviewed Core: "
                        + case["sample_id"]
                    )
        preview = (
            image_url(raw if ready else overlay) if ready or overlay.is_file() else ""
        )
        rows.append(
            {
                "sample_id": case["sample_id"],
                "garment_category": case["garment_category"],
                "source_image": case["source_image"],
                "image_width": int(case["image_width"]),
                "image_height": int(case["image_height"]),
                "garment_bbox": [
                    float(case["gt_bbox_" + k]) for k in ("x1", "y1", "x2", "y2")
                ],
                "raw_available": ready,
                "preview": preview,
            }
        )
    payload = {
        "cases": rows,
        "grounding": ground,
        "attributes": [
            {
                **r,
                "allowed_labels": labels_for(
                    r["attribute_name"], r["garment_category"]
                ),
            }
            for r in attrs
        ],
        "ground_fields": list(ground[0]),
        "attribute_fields": list(attrs[0]),
        "fingerprint": digest(base / "drafts/attribute_test.csv")
        + digest(base / "drafts/grounding_test.csv"),
    }
    html = PAGE.replace(
        "__PAYLOAD__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    )
    target = output or base / "review/prd31_review.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    result = {
        "review_page": str(target.relative_to(root)),
        "cases": len(rows),
        "source_images_ready": sum(r["raw_available"] for r in rows),
        "status": (
            "READY_FOR_HUMAN_REVIEW"
            if all(r["raw_available"] for r in rows)
            else "SOURCE_IMAGES_REQUIRED"
        ),
    }
    write_json(base / "review/packet_status.json", result)
    return result


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output")
    args = p.parse_args()
    print(
        json.dumps(
            build(output=resolve_path(args.output) if args.output else None),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
