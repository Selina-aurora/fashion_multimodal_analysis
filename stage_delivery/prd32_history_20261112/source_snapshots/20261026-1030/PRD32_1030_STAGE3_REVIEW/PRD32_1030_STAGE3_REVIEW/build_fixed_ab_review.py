#!/usr/bin/env python3
"""Generate an offline human-review HTML for paired PRD 3.2 fixed-dev A/B runs.
No inference; no GT injection; copies only source model outputs and marked images.
"""
import argparse
import base64
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import statistics

TASKS=('tags','vqa','copy')
SCOPE='TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY'
META_RE=re.compile(r'红.{0,3}(?:框|矩形|方框|标记)|框内|框中|框选|框出|圈出|定位框|标注框|目标框|红色.{0,4}指向|图中标注|红色边')
ABSTAIN_RE=re.compile(r'无法确定|不能确定|不确定|无法判断|看不清|无法识别')

def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as fp:
        for chunk in iter(lambda:fp.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def normalized(rows,variant):
    got={}
    for r in rows:
        key=(r.get('case_id'),r.get('target_instance_id'),r.get('task'))
        if r.get('model_variant')!=variant or key in got or r.get('status')!='GENERATED_REVIEW_REQUIRED':
            raise ValueError('BAD_VARIANT_DUPLICATE_OR_STATUS:'+str(key))
        if r.get('request_id')!='__'.join(key):raise ValueError('BAD_REQUEST_ID:'+str(key))
        got[key]=r
    return got

def data_uri(path,resize_width=850):
    # Original file was already SHA-checked before resizing for display.
    try:
        from PIL import Image
        with Image.open(path) as im:
            im=im.convert('RGB')
            im.thumbnail((resize_width,950))
            stream=io.BytesIO(); im.save(stream,format='JPEG',quality=82,optimize=True)
            return 'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode('ascii')
    except ImportError:
        return 'data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,default=Path('/workspace/PRD32_FIXED_DEV_BASELINE_20261030_run01'))
    p.add_argument('--focus',type=Path,default=Path('/workspace/PRD32_FIXED_DEV_FOCUS_V1_20261030_run01'))
    p.add_argument('--output',type=Path,default=Path('/workspace/PRD32_FIXED_DEV_AB_HUMAN_REVIEW_20261030.html'))
    a=p.parse_args()
    bpath=a.baseline/'qwen_results.jsonl';fpath=a.focus/'qwen_results.jsonl'
    b=normalized(read(bpath),'baseline');f=normalized(read(fpath),'focus_v1')
    if set(b)!=set(f) or len(b)!=93:raise ValueError('INCOMPLETE_OR_UNPAIRED_93_REQUESTS')
    grouped=sorted(set((c,i) for c,i,t in b))
    if len(grouped)!=31:raise ValueError('EXPECTED_31_INSTANCES')
    content=[]
    hashes=set()
    for cid,iid in grouped:
        bt={task:b[(cid,iid,task)] for task in TASKS}
        ft={task:f[(cid,iid,task)] for task in TASKS}
        for task in TASKS:
            old,new=bt[task],ft[task]
            fields=['image_sha256','marked_image_sha256','v5_predicted_bbox_xyxy','v5_category_unverified','question_text','question_sha256']
            if any(old.get(key)!=new.get(key) for key in fields):raise ValueError('INPUT_MISMATCH:'+cid+' '+iid+' '+task)
            if old['marked_image_sha256']!=bt['tags']['marked_image_sha256']:
                raise ValueError('TASK_IMAGE_MISMATCH:'+cid+' '+iid+' '+task)
        marked=Path(bt['tags']['marked_image'])
        if not marked.is_file():
            alternative=Path(ft['tags']['marked_image'])
            if alternative.is_file():marked=alternative
            else:raise FileNotFoundError('MISSING_MARKED_IMAGE:'+cid+' '+iid+' '+str(marked))
        if digest(marked)!=bt['tags']['marked_image_sha256']:
            raise ValueError('MARKED_IMAGE_SHA_MISMATCH:'+cid+' '+iid)
        hashes.add(bt['tags']['marked_image_sha256'])
        oldcat=bt['tags'].get('tag_parse',{}).get('canonical_service_json',{}).get('category')
        newcat=ft['tags'].get('tag_parse',{}).get('canonical_service_json',{}).get('category')
        oldcopy=bt['copy'].get('raw_text','');newcopy=ft['copy'].get('raw_text','')
        flags=[]
        if oldcat!=newcat:flags.append('Qwen 类别改变：'+str(oldcat)+' → '+str(newcat))
        if bt['tags'].get('v5_category_unverified')!=newcat:flags.append('Focus 类别与 V5 预测不一致（非准确率）')
        if META_RE.search(newcopy):flags.append('Focus 文案仍含定位元文字')
        if ABSTAIN_RE.search(ft['vqa']['raw_text']):flags.append('Focus VQA 显式拒答，需检查是否合理')
        content.append({
            'case_id':cid,'instance_id':iid,'v5_category_unverified':bt['tags'].get('v5_category_unverified'),
            'v5_predicted_bbox_xyxy':bt['tags']['v5_predicted_bbox_xyxy'],
            'image_sha256':bt['tags']['image_sha256'],
            'marked_image_sha256':bt['tags']['marked_image_sha256'],
            'image':data_uri(marked), 'flags':flags,
            'question':ft['vqa'].get('question_text') or '原始问题未提供',
            'baseline':{t:bt[t].get('raw_text','') for t in TASKS},
            'focus':{t:ft[t].get('raw_text','') for t in TASKS},
            'changes':{t:bt[t].get('raw_text','')!=ft[t].get('raw_text','') for t in TASKS},
        })
    if len(hashes)!=31:print('WARNING: distinct marked image count',len(hashes),'(may be duplicated; review)')
    prior_path=Path(__file__).with_name('focus_8_verified_prior_human_labels.json')
    carryover=[]
    if prior_path.is_file():
        prior=json.loads(prior_path.read_text(encoding='utf-8'))
        if prior.get('new_focus_qwen_sha256')==digest(fpath):
            for row in prior.get('records',[]):
                cid,iid=row['case_id'],row['instance_id']
                if not all((cid,iid,task) in f and hashlib.sha256(f[(cid,iid,task)]['raw_text'].encode('utf-8')).hexdigest()==row['raw_text_sha256_per_task'][task] for task in TASKS):
                    raise ValueError('CARRYOVER_RAW_TEXT_MISMATCH:'+cid+' '+iid)
                if f[(cid,iid,'tags')]['marked_image_sha256']!=row['marked_image_sha256']:
                    raise ValueError('CARRYOVER_MARKED_IMAGE_MISMATCH:'+cid+' '+iid)
                carryover.append(row)
        else: print('PRIOR_REVIEW_SKIPPED_DIFFERENT_FOCUS_FILE')
    manifest={
        'schema':'prd32_20261030_fixed_ab_human_review_v1',
        'scope':SCOPE,'baseline_qwen_sha256':digest(bpath),'focus_qwen_sha256':digest(fpath),
        'images':len(set(item['case_id'] for item in content)),'instances':len(content),
        'requests_per_variant':len(b),'images_sha256_verified':True,
        'human_review_completed':False,'official_prd_acceptance':False,'publish_allowed':False
    }
    # Escape HTML script termination and JS-compatible line separators without altering source content.
    data=json.dumps({'manifest':manifest,'items':content,'verified_prior_focus_review':carryover},ensure_ascii=False,separators=(',',':'))
    data=data.replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    html=HTML.replace('/*__REVIEW_DATA__*/','const DATA = '+data+';')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(html,encoding='utf-8')
    print('REVIEW_HTML_READY:',a.output)
    print('IMAGES:',manifest['images'],'INSTANCES:',manifest['instances'],'PAIRED_REQUESTS:',manifest['requests_per_variant'])
    print('MARKED_IMAGE_SHA256_VERIFIED:',len(hashes),'unique files')
    print('VERIFIED_20261029_FOCUS_LABELS_PREFILLED:',len(carryover),'out of 31')
    print('SIZE_MB:',round(a.output.stat().st_size/1024**2,2))
    print('HUMAN_REVIEW_NOT_COMPLETED; NO_MAIN_STRATEGY_CHANGE')

HTML=r'''<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PRD 3.2 | 10.30 固定开发集 A/B 人工复核</title>
<style>
:root{color-scheme:light;font-family:system-ui,"Microsoft YaHei",sans-serif;color:#182238;background:#f3f5f9}
*{box-sizing:border-box}body{max-width:1460px;margin:0 auto;padding:24px 20px}h1{margin-bottom:8px}small,.sub{color:#607085}header{background:white;border:1px solid #d9e1eb;border-radius:12px;padding:20px;margin-bottom:14px}.toolbar{position:sticky;top:0;background:#f3f5f9eb;backdrop-filter:blur(8px);padding:10px 0;display:flex;flex-wrap:wrap;gap:12px;align-items:center;z-index:3}button{background:#193964;color:white;border:0;padding:11px 16px;border-radius:8px;cursor:pointer}button.subtle{background:white;color:#193964;border:1px solid #b3c1d4}select,textarea{border:1px solid #b8c5d7;border-radius:6px;background:#fff;padding:8px;font:inherit;min-width:0}select{width:100%}.item{background:#fff;border-radius:12px;border:1px solid #d9e1eb;margin:16px 0;padding:18px}.item.done{border:2px solid #2c8053}.split{display:grid;grid-template-columns:minmax(230px,.75fr) minmax(0,1.8fr);gap:16px}.photo{display:block;width:100%;height:auto;max-height:700px;object-fit:contain;background:#ebeff7;border-radius:7px;border:1px solid #dce3ef}.flag{display:inline-block;background:#fff2dc;color:#80500b;padding:4px 8px;border-radius:4px;margin:3px 5px 3px 0;font-size:12px}.answer-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.answer{background:#f4f6fa;border-radius:7px;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere;min-height:82px;font-size:13px}.answer h4{margin:0 0 8px;font-size:14px;color:#354966}.task-head{margin:12px 0 6px;font-weight:700;color:#1c304b}.field-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0}.field-grid label{display:block;font-size:13px;font-weight:600}.field-grid label select{display:block;margin-top:4px}.note{width:100%;min-height:60px;margin-top:8px}.confirm{padding:12px;background:#eaf2fb;border-radius:7px;display:flex;gap:8px;align-items:center}.case-title{font-size:20px;font-weight:750;margin:0 0 8px}.status{font-size:13px;color:#4a627d}.none{display:none}@media(max-width:900px){.split,.answer-grid,.field-grid{grid-template-columns:1fr}}@media print{.toolbar,button{display:none}}
</style></head><body>
<header><h1>PRD 3.2｜10 月 30 日固定开发集 A/B 人工复核</h1><div class="sub">12 张图 · 31 个 V5 预测实例 · Baseline / Focus V1 各 93 条请求</div>
<p>这份页面供逐条核对原始标框图、Tags / VQA / Copy 输出。<strong>全部选项初始均为未审核</strong>；机器诊断标签不是真值。保留测试集未参与，禁止把这次人工复核当成独立测试准确率或 PRD 验收。</p>
<p>审核两个版本时采用相同口径。相似的预测框可能对应同一件衣物；重复标注不会自动合并。每条审核完成后勾选确认；若随后改动选项，该条会自动恢复待审核。若与 10.29 已人工复核的 8 条 Focus 输出及图像哈希完全一致，将预填原审核意见供你复查；<strong>不会自动标记本轮审核完成</strong>。</p>
<small id="source"></small></header>
<div class="toolbar"><label>筛选图片 <select id="filterCase"><option value="">全部图片</option></select></label><label><input type="checkbox" id="onlyIncomplete"/>只看待审核</label><b id="progress">已确认 0/31</b><button id="export">导出审核 JSON</button><button id="clear" class="subtle">清空本地预填</button></div>
<main id="items"></main>
<script>
/*__REVIEW_DATA__*/
const PREFIX='prd32_1030_review_'+DATA.manifest.focus_qwen_sha256.slice(0,16);
const TASKS=['tags','vqa','copy'];
const choices={
 tags:[['','未审核'],['CORRECT','CORRECT 正确'],['PARTIALLY_CORRECT','PARTIALLY_CORRECT 部分正确'],['INCORRECT','INCORRECT 错误'],['UNCERTAIN','UNCERTAIN 不确定']],
 vqa:[['','未审核'],['CORRECT','CORRECT 正确'],['INCORRECT','INCORRECT 错误'],['REASONABLE_ABSTENTION','REASONABLE_ABSTENTION 合理拒答'],['UNCERTAIN','UNCERTAIN 不确定']],
 copy:[['','未审核'],['CORRECT','CORRECT 正确'],['DEBUG_TEXT','DEBUG_TEXT 定位调试词'],['UNSUPPORTED_CLAIM','UNSUPPORTED_CLAIM 无依据断言'],['OTHER_ERROR','OTHER_ERROR 其他错误'],['UNCERTAIN','UNCERTAIN 不确定']],
 binding:[['','未审核'],['CORRECT','CORRECT 目标正确'],['INCORRECT','INCORRECT 目标错误'],['UNCERTAIN','UNCERTAIN 不确定']],
 duplicate:[['','未审核'],['DISTINCT','DISTINCT 独立实例'],['DUPLICATE','DUPLICATE 重复候选'],['UNCERTAIN','UNCERTAIN 不确定']]
};
let state={};try{state=JSON.parse(localStorage.getItem(PREFIX)||'{}')}catch{}
for(const p of DATA.verified_prior_focus_review||[]){const k=p.case_id+'__'+p.instance_id;if(!state[k]){state[k]={binding:'',duplicate:'',baseline:{tags:'',vqa:'',copy:''},focus:p.focus_quality,note:'【10.29 已人工复核且本次图像/输出逐字匹配；本轮仍需重新确认】'+p.note,confirmed:false}}}
const container=document.getElementById('items');
function el(name,cl,txt){const x=document.createElement(name);if(cl)x.className=cl;if(txt!==undefined)x.textContent=txt;return x}
function options(select,items,value){for(const [v,t] of items){const o=document.createElement('option');o.value=v;o.textContent=t;select.append(o)}select.value=value||''}
function key(item){return item.case_id+'__'+item.instance_id}
function blank(){return {binding:'',duplicate:'',baseline:{tags:'',vqa:'',copy:''},focus:{tags:'',vqa:'',copy:''},note:'',confirmed:false}}
function current(item){return state[key(item)]||blank()}
function save(item,fn){const k=key(item);const st=structuredClone(current(item));fn(st);st.confirmed=false;state[k]=st;try{localStorage.setItem(PREFIX,JSON.stringify(state))}catch{}refresh()}
function complete(st){return !!(st.binding&&st.duplicate&&TASKS.every(t=>st.baseline[t]&&st.focus[t]))}
function render(){
 const cases=[...new Set(DATA.items.map(x=>x.case_id))];for(const c of cases){const o=document.createElement('option');o.value=c;o.textContent=c;document.getElementById('filterCase').append(o)}
 document.getElementById('source').textContent='Baseline SHA: '+DATA.manifest.baseline_qwen_sha256.slice(0,16)+'… / Focus SHA: '+DATA.manifest.focus_qwen_sha256.slice(0,16)+'…';
 for(const item of DATA.items){const st=current(item);const sec=el('section','item');sec.dataset.key=key(item);sec.dataset.case=item.case_id;
 sec.append(el('h2','case-title',key(item)));const status=el('div','status','V5 预测类别（非 GT）：'+item.v5_category_unverified+'｜bbox：'+item.v5_predicted_bbox_xyxy.join(', '));sec.append(status);
 const split=el('div','split');const left=el('div');const img=el('img','photo');img.src=item.image;img.alt='冻结预测实例的红框图';left.append(img);
 const flags=el('div');if(item.flags.length){item.flags.forEach(x=>flags.append(el('span','flag',x)))}else flags.append(el('small',null,'自动检查暂未触发重点风险；仍需人工审核'));
 left.append(flags);const right=el('div');
 for(const t of TASKS){right.append(el('div','task-head',t.toUpperCase()+'｜'+(item.changes[t]?'输出变化':'原文未变化')));const two=el('div','answer-grid');
 for(const [v,name] of [['baseline','Baseline'],['focus','Focus V1']]){const cell=el('div','answer');cell.append(el('h4',null,name));cell.append(el('div',null,item[v][t]));two.append(cell)}right.append(two);
 if(t==='vqa')right.append(el('small',null,'固定问题：'+item.question));}
 split.append(left,right);sec.append(split);
 const grid=el('div','field-grid');
 function addField(title,opts,at,task){const wrap=el('label',null,title);const s=el('select');options(s,opts,task?current(item)[at][task]:current(item)[at]);s.addEventListener('change',()=>save(item,x=>{if(task)x[at][task]=s.value;else x[at]=s.value}));wrap.append(s);grid.append(wrap)}
 addField('实例绑定质量（两版共同输入）',choices.binding,'binding');addField('重复候选判断（两版共同输入）',choices.duplicate,'duplicate');
 for(const t of TASKS){addField('Baseline '+t.toUpperCase()+' 质量',choices[t],'baseline',t);addField('Focus '+t.toUpperCase()+' 质量',choices[t],'focus',t)}
 sec.append(grid);const nl=el('label',null,'复核备注（如类别边界、定位缺陷、跨实例串扰）');const note=el('textarea','note');note.value=current(item).note||'';note.addEventListener('input',()=>save(item,x=>x.note=note.value));nl.append(note);sec.append(nl);
 const confirm=el('label','confirm');const cb=document.createElement('input');cb.type='checkbox';cb.checked=!!st.confirmed;
 cb.addEventListener('change',()=>{if(cb.checked&&!complete(current(item))){alert('请先填写该实例全部 8 项质量判断，再确认。');cb.checked=false;return;}const k=key(item);const s=structuredClone(current(item));s.confirmed=cb.checked;state[k]=s;try{localStorage.setItem(PREFIX,JSON.stringify(state))}catch{}refresh()});
 confirm.append(cb,el('span',null,'我已逐项对照图片与两版输出，并确认这条审核结果'));sec.append(confirm);container.append(sec)}
 refresh();
}
function refresh(){let done=0;for(const item of DATA.items){const sec=document.querySelector('[data-key="'+key(item)+'"]');const st=current(item);if(st.confirmed&&complete(st))done++;sec.classList.toggle('done',!!(st.confirmed&&complete(st)));
 const shouldHide=(document.getElementById('filterCase').value&&document.getElementById('filterCase').value!==item.case_id)||(document.getElementById('onlyIncomplete').checked&&st.confirmed&&complete(st));sec.classList.toggle('none',!!shouldHide)}document.getElementById('progress').textContent='已确认 '+done+'/'+DATA.items.length}
document.getElementById('filterCase').addEventListener('change',refresh);document.getElementById('onlyIncomplete').addEventListener('change',refresh);
document.getElementById('export').addEventListener('click',()=>{
 const rows=DATA.items.map(item=>{const s=current(item);return {case_id:item.case_id,instance_id:item.instance_id,image_sha256:item.image_sha256,marked_image_sha256:item.marked_image_sha256,instance_binding:s.binding,duplicate_assessment:s.duplicate,baseline_quality:s.baseline,focus_quality:s.focus,reviewer_note:s.note,instance_review_confirmed:!!(s.confirmed&&complete(s))}});
 const all=rows.every(r=>r.instance_review_confirmed);const data={schema:'prd32_fixed_dev_paired_human_review_20261030_v1',scope:'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',baseline_qwen_sha256:DATA.manifest.baseline_qwen_sha256,focus_qwen_sha256:DATA.manifest.focus_qwen_sha256,reviewed_by_human:all,review_status:all?'HUMAN_REVIEW_COMPLETED_DIAGNOSTIC_ONLY':'PARTIAL_REVIEW_NOT_COMPLETE',publish_allowed:false,official_prd_acceptance:false,rows};
 const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='PRD32_FIXED_DEV_AB_HumanReview_20261030.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),10000);
});
document.getElementById('clear').addEventListener('click',()=>{if(!confirm('确定清空本页已填的审核选项吗？（不可撤销）'))return;state={};try{localStorage.removeItem(PREFIX)}catch{}location.reload()});
render();
</script></body></html>'''

if __name__=='__main__':main()
