#!/usr/bin/env python3
"""Conservative, GT-free, posthoc semantic audit for fixed PRD3.2 V5/Qwen outputs.
No model inference. Never assert semantic correctness or modify frozen predictions.
"""
from __future__ import annotations
import argparse, base64, collections, csv, hashlib, html, json, math, re, statistics
from pathlib import Path
from typing import Any

TASKS = ('tags', 'vqa', 'copy')
COLOR_WORDS = {
 '白': ('白色','米白','乳白','象牙白'), '黑': ('黑色','纯黑'),
 '蓝': ('蓝色','浅蓝','深蓝','牛仔蓝','湖蓝','天蓝'),
 '黄': ('黄色','亮黄','浅黄','姜黄','明黄'), '红': ('红色','酒红','玫红','正红'),
 '绿': ('绿色','墨绿','草绿'), '粉': ('粉色','粉红'), '灰': ('灰色','浅灰','深灰'),
 '棕': ('棕色','咖啡色','褐色','卡其'), '紫': ('紫色',), '橙': ('橙色',),
}
COPY_DEBUG_PATTERNS = ('红色边框','红色方框','红框','标注框','红色矩形框','红色框','被框选','框选区域','bounding box','bbox')
ABSTAIN_MARKERS = ('无法确定','不能确定','看不清','无法判断','不确定','不知道','无法辨认')
BRAND_TERMS = ('MIU MIU', 'PRADA', 'GUCCI', 'CHANEL', 'LOUIS VUITTON')


def digest(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load_jsonl(p: Path) -> list[dict[str,Any]]:
    with p.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def box_iou(a:list[float], b:list[float])->float:
    if not (len(a)==len(b)==4):return 0.0
    if not all(math.isfinite(float(v)) for v in a+b):return 0.0
    x=max(0.,min(a[2],b[2])-max(a[0],b[0]));y=max(0.,min(a[3],b[3])-max(a[1],b[1]))
    area_a=max(0.,a[2]-a[0])*max(0.,a[3]-a[1]);area_b=max(0.,b[2]-b[0])*max(0.,b[3]-b[1]); inter=x*y
    return inter/(area_a+area_b-inter) if area_a+area_b>inter else 0.0


def color_set(s: str, *, copy:bool=False)->set[str]:
    if copy:
        for word in COPY_DEBUG_PATTERNS:
            s=s.replace(word,'')
    return {name for name, aliases in COLOR_WORDS.items() if any(a in s for a in aliases)}


def copy_category(s:str)->str|None:
    if any(k in s for k in ('连衣裙','连身裙')):return 'dress'
    if any(k in s for k in ('裙裤','裤裙')):return None
    if any(k in s for k in ('短裤','长裤','牛仔裤','裤装','裤子','阔腿裤')):return 'pants'
    if any(k in s for k in ('半身裙','短裙','百褶裙','蓬蓬裙','裙子','裙装')):return 'skirt'
    if any(k in s for k in ('外套','夹克','大衣','风衣')):return 'outerwear'
    if any(k in s for k in ('上衣','背心','T恤','衬衫','卫衣','针织衫','短袖')):return 'top'
    if any(k in s for k in ('运动鞋','皮鞋','鞋子','高跟鞋')):return 'shoe'
    if any(k in s for k in ('手提包','背包','挎包','包包')):return 'bag'
    return None


def tags_struct(row:dict)->dict|None:
    v=row.get('tag_parse',{}).get('canonical_service_json')
    if isinstance(v,dict):return v
    return row.get('tag_parse',{}).get('raw_parsed')


def make_html(instances:list[dict],duplicates:list[dict], source_rows:list[dict], img_embed:bool=False)->str:
    dup_map=collections.defaultdict(list)
    for p in duplicates:
        dup_map[(p['case_id'],p['instance_a'])].append(f"{p['instance_b']} (IoU {p['bbox_iou']:.3f})")
        dup_map[(p['case_id'],p['instance_b'])].append(f"{p['instance_a']} (IoU {p['bbox_iou']:.3f})")
    by_req={(r['case_id'],r['target_instance_id'],r['task']):r for r in source_rows}
    cards=[]
    for i,x in enumerate(instances):
        case,inst=x['case_id'],x['instance_id']
        d={t:by_req.get((case,inst,t),{}) for t in TASKS}
        img_path=Path(next((v.get('marked_image','') for v in d.values() if v.get('marked_image')), ''))
        preview='原始红框图片未随 JSONL 上传。若在 AutoDL 本地执行并加 --embed-images，审核页会显示对应图片。'
        if img_embed and img_path.is_file() and img_path.suffix.lower() in ('.png','.jpg','.jpeg','.webp') and img_path.stat().st_size<3_000_000:
            mime={'png':'image/png','jpg':'image/jpeg','jpeg':'image/jpeg','webp':'image/webp'}[img_path.suffix.lower().lstrip('.')]
            preview=f'<img class="photo" src="data:{mime};base64,{base64.b64encode(img_path.read_bytes()).decode()}" alt="{html.escape(case+inst)}" />'
        tag=tags_struct(d['tags']) or {}
        cls=x['v5_category']; score=x['v5_score']
        cards.append(f'''<section class="case" data-case="{html.escape(case)}" data-inst="{html.escape(inst)}">
<h2>{html.escape(case)} / {html.escape(inst)}</h2>
<p><b>V5 未核验类别</b> {html.escape(str(cls))} · <b>score</b> {score:.3f} · <b>bbox</b> {html.escape(str(x['bbox']))}</p>
<p><b>疑似高重叠候选</b> {html.escape('; '.join(dup_map.get((case,inst), [])) or '无')}</p>
<div class="media">{preview}</div>
<div class="triplet"><div><h3>Tags</h3><pre>{html.escape(json.dumps(tag,ensure_ascii=False,indent=2))}</pre></div>
<div><h3>VQA</h3><pre>{html.escape(d['vqa'].get('raw_text',''))}</pre></div>
<div><h3>Copy</h3><pre>{html.escape(d['copy'].get('raw_text',''))}</pre></div></div>
<p class="flags"><b>自动提示：</b> {html.escape(' / '.join(x['flags']) or '无自动提示；内容仍须人工确认')}</p>
<div class="controls">
<label>是否正确绑定实例？<select class="binding"><option value="UNREVIEWED">未审核</option><option value="CORRECT">正确</option><option value="WRONG_INSTANCE">描述其他服饰</option><option value="UNCERTAIN">无法确认</option></select></label>
<label>是否重复检测同一件衣服？<select class="duplicate"><option value="UNREVIEWED">未审核</option><option value="DUPLICATE">重复</option><option value="DISTINCT">不同衣物</option><option value="UNCERTAIN">无法确认</option></select></label>
<label>标签质量<select class="tags_quality"><option value="UNREVIEWED">未审核</option><option value="CORRECT">正确</option><option value="INCORRECT">有误</option><option value="UNCERTAIN">无法确认</option></select></label>
<label>问答质量<select class="vqa_quality"><option value="UNREVIEWED">未审核</option><option value="CORRECT">正确</option><option value="INCORRECT">有误</option><option value="ABSTENTION_REASONABLE">合理拒答</option><option value="UNCERTAIN">无法确认</option></select></label>
<label>文案质量<select class="copy_quality"><option value="UNREVIEWED">未审核</option><option value="CORRECT">正确</option><option value="WRONG_INSTANCE">串实例</option><option value="UNSUPPORTED_CLAIM">缺乏证据</option><option value="DEBUG_TEXT">包含调试信息</option><option value="OTHER_ERROR">其他错误</option><option value="UNCERTAIN">无法确认</option></select></label>
<label class="notes">审核备注<textarea rows="2" placeholder="请根据红框图片核验。不要将自动标记视为人工真值。"></textarea></label></div></section>''')
    js_items=[{'case_id':x['case_id'],'instance_id':x['instance_id'],'flags':x['flags']} for x in instances]
    # JavaScript only exports human-entered values. Do not prefill outcomes as human.
    return '''<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>PRD 3.2 V5→Qwen 5实例人工复核</title>
<style>body{font-family:system-ui,'Microsoft YaHei',sans-serif;background:#f6f7fa;color:#252a33;max-width:1250px;margin:25px auto;padding:0 14px}h1{font-size:25px}.case{background:white;border:1px solid #d5d8df;border-radius:12px;margin:20px 0;padding:20px}.media{margin:12px 0;line-height:1.5;background:#f0f3f6;padding:12px;border-radius:8px}.photo{max-width:700px;max-height:700px;object-fit:contain}.triplet{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.triplet>div{background:#f5f6f8;padding:12px;border-radius:7px;min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.flags{color:#9a391f}.controls{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.controls label{display:grid;gap:5px}select,textarea{padding:7px;font:inherit}.notes{grid-column:1/-1}button{border:0;background:#172d58;color:white;padding:12px 20px;border-radius:8px;cursor:pointer}@media(max-width:700px){.triplet,.controls{grid-template-columns:1fr}}</style></head><body>
<h1>PRD 3.2 V5 → Qwen 逐实例复核</h1><p>2张开发图 / 5个预测实例；自动判断只供筛查。原始输出均未经过人工质量确认；当前没有发布权限。</p>
<p><b>特别检查：</b> fresh_dev_004 / pred_0001 的文案是否串到蓝色短裤；fresh_dev_001 双重预测、fresh_dev_004 裤装/裙装双重预测是否重复；图片中品牌文字是否确实可见。</p>
<button onclick="exportReview()">导出人工审核 JSON</button><p id="status"></p>
''' + ''.join(cards)+ '''<script>
function exportReview(){const rows=[...document.querySelectorAll('section.case')].map(x=>({case_id:x.dataset.case,instance_id:x.dataset.inst,instance_binding:x.querySelector('.binding').value,duplicate_assessment:x.querySelector('.duplicate').value,tags_quality:x.querySelector('.tags_quality').value,vqa_quality:x.querySelector('.vqa_quality').value,copy_quality:x.querySelector('.copy_quality').value,reviewer_note:x.querySelector('textarea').value}));
const payload={schema:'prd32_v5_qwen_human_review_v1',scope:'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',reviewed_by_human:false,rows};
const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}); const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='PRD32_V5_Qwen_5instance_Review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),500);document.getElementById('status').textContent='已导出审核数据；请确认每条判断后再上传复核。';}
</script></body></html>'''


def evaluate(summary:dict,rows:list[dict],source_hash:str,embed_images:bool=False)->tuple[dict,list[dict],list[dict],str]:
    warnings=[]
    identities={}
    for row in rows:
        k=(row['case_id'],row['target_instance_id'])
        if row['task'] not in TASKS: warnings.append(f"UNKNOWN_TASK:{row.get('task')}");continue
        obj=identities.setdefault(k,{})
        if row['task'] in obj: warnings.append(f"DUPLICATE_REQUEST:{k}/{row['task']}")
        obj[row['task']]=row
        if row.get('human_label_or_gt_bbox_used') is not False: warnings.append(f"GT_POLICY_VIOLATION:{k}")
        if row.get('publish_allowed') is not False: warnings.append(f"PUBLISH_POLICY_VIOLATION:{k}")
        if row.get('target_bbox_origin')!='prd31_v5_live_prediction': warnings.append(f"UNEXPECTED_BBOX_ORIGIN:{k}")
    instances=[]
    for (case,inst), d in sorted(identities.items()):
        if set(d)!=set(TASKS): warnings.append(f"INCOMPLETE_TASKS:{case}/{inst}")
        if 'tags' not in d:continue
        tagrow=d['tags']; vqarow=d.get('vqa',{}); copyrow=d.get('copy',{})
        cat=tagrow['v5_category_unverified']; score=tagrow['v5_predicted_score_uncalibrated']; bbox=tagrow['v5_predicted_bbox_xyxy']; t=tags_struct(tagrow) or {}; copytext=copyrow.get('raw_text','');vqa=vqarow.get('raw_text','')
        flags=[]
        for row in d.values():
            if row.get('v5_category_unverified')!=cat or row.get('v5_predicted_bbox_xyxy')!=bbox:
                flags.append('TASK_METADATA_INCONSISTENT')
        if t.get('category') and t['category']!=cat:flags.append('V5_QWEN_CATEGORY_CONFLICT')
        if not tagrow.get('tag_parse',{}).get('schema_valid',False):flags.append('TAGS_SCHEMA_UNVERIFIED')
        if not tagrow.get('tag_parse',{}).get('raw_native_json',False):flags.append('TAGS_MARKDOWN_FENCE_NON_NATIVE_JSON')
        if any(s.lower() in copytext.lower() for s in COPY_DEBUG_PATTERNS): flags.append('COPY_DEBUG_TARGET_MARKER')
        cp_cat=copy_category(copytext)
        if t.get('category') and cp_cat and cp_cat!=t['category']:flags.append('COPY_TAG_CATEGORY_CONFLICT')
        cp_colors=color_set(copytext,copy=True)
        tg_colors=color_set(str(t.get('primary_color','')))
        if cp_colors and tg_colors and cp_colors.isdisjoint(tg_colors):flags.append('COPY_TAG_COLOR_CONFLICT')
        vqa_abstain=bool(vqarow.get('vqa_visible_abstention_marker')) or any(s in vqa for s in ABSTAIN_MARKERS)
        if vqa_abstain:flags.append('VQA_ABSTAIN_NOT_SUCCESSFUL_ANSWER')
        else:
            vq_colors=color_set(vqa)
            if vq_colors and tg_colors and vq_colors.isdisjoint(tg_colors):flags.append('VQA_TAG_COLOR_CONFLICT')
        if any(t.upper() in copytext.upper() for t in BRAND_TERMS):flags.append('BRAND_TEXT_REQUIRES_VISUAL_CHECK')
        original_flags=set(copyrow.get('copy_review_flags',[]))
        if 'COPY_DEBUG_TARGET_MARKER' in flags and 'DEBUG_MARKER_IN_COPY' not in original_flags:
            flags.append('ORIGINAL_COPY_DEBUG_GUARD_MISSED')
        severe={'COPY_DEBUG_TARGET_MARKER','COPY_TAG_CATEGORY_CONFLICT','COPY_TAG_COLOR_CONFLICT','VQA_TAG_COLOR_CONFLICT','TASK_METADATA_INCONSISTENT'}
        x={'case_id':case,'instance_id':inst,'v5_category':cat,'v5_score':score,'bbox':bbox,'qwen_category':t.get('category'), 'vqa_text':vqa,
        'copy_text':copytext,'flags':sorted(set(flags)),'copy_content_disposition':'BLOCK_FOR_DISPLAY_REWRITE' if severe.intersection(flags) else 'REVIEW_REQUIRED',
        'publish_allowed':False,'task_request_ids':{name:row['request_id'] for name,row in d.items()}}
        instances.append(x)
    dupe=[]
    for i,a in enumerate(instances):
        for b in instances[i+1:]:
            if a['case_id']!=b['case_id']:continue
            score=box_iou(a['bbox'],b['bbox'])
            if score>=.8:
                dupe.append({'case_id':a['case_id'],'instance_a':a['instance_id'],'instance_b':b['instance_id'],'bbox_iou':round(score,6),'v5_classes':[a['v5_category'],b['v5_category']], 'status':'POSSIBLE_DUPLICATE_REQUIRES_IMAGE_MASK_REVIEW'})
                a['flags'].append('HIGH_OVERLAP_POSSIBLE_DUPLICATE');b['flags'].append('HIGH_OVERLAP_POSSIBLE_DUPLICATE')
    if len(rows)!=summary.get('n_requests'):warnings.append('SUMMARY_N_REQUESTS_MISMATCH')
    if len(instances)!=summary.get('n_predicted_instances_processed'):warnings.append('SUMMARY_N_INSTANCES_MISMATCH')
    if len({x['case_id'] for x in instances})!=summary.get('n_images'):warnings.append('SUMMARY_N_IMAGES_MISMATCH')
    if summary.get('successful_outputs')!=sum(str(r.get('status','')).startswith('GENERATED') for r in rows):warnings.append('SUCCESS_COUNT_MISMATCH')
    flag_counter=collections.Counter(flag for x in instances for flag in set(x['flags']))
    bytask={}
    for task in TASKS:
        vals=[float(r['timing']['end_to_end_ms']) for r in rows if r.get('task')==task and 'timing'in r]
        bytask[task]={'n':len(vals),'median_ms':round(statistics.median(vals),2) if vals else None}
    all_lat=[float(r['timing']['end_to_end_ms']) for r in rows if 'timing' in r]
    audit={'schema':'prd32_v5_qwen_gtfree_semantic_audit_v1','scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY_NOT_INDEPENDENT',
           'source_qwen_jsonl_sha256':source_hash,'v5_raw_predictions_sha256_claim_from_summary':summary.get('v5_raw_sha256'),
           'requests':len(rows),'predicted_instances':len(instances),'images':len({x['case_id'] for x in instances}),
           'generation_success':summary.get('successful_outputs'),'syntactic_tags_schema_valid':sum(bool(r.get('tag_parse',{}).get('schema_valid')) for r in rows if r['task']=='tags'),
           'v5_qwen_category_conflicts':flag_counter['V5_QWEN_CATEGORY_CONFLICT'],
           'copy_tag_category_conflicts':flag_counter['COPY_TAG_CATEGORY_CONFLICT'],
           'copy_tag_color_conflicts':flag_counter['COPY_TAG_COLOR_CONFLICT'],
           'copy_debug_mentions':flag_counter['COPY_DEBUG_TARGET_MARKER'],
           'original_debug_guard_misses':flag_counter['ORIGINAL_COPY_DEBUG_GUARD_MISSED'],
           'vqa_abstentions':flag_counter['VQA_ABSTAIN_NOT_SUCCESSFUL_ANSWER'],
           'possible_duplicate_pairs':len(dupe),
           'copy_display_blocked_instances':sum(x['copy_content_disposition']=='BLOCK_FOR_DISPLAY_REWRITE' for x in instances),
           'median_all_request_ms':round(statistics.median(all_lat),2) if all_lat else None,
           'median_by_task_ms':bytask,
           'warnings':warnings,'publish_allowed':False,'human_verified':False,'visual_inspection_completed':False,
           'note':'This is string/metadata/box-overlap triage only. bbox IoU != mask IoU; Qwen agreement does not imply ground truth.'}
    return audit,instances,dupe,make_html(instances,dupe,rows,img_embed=embed_images)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    here=Path(__file__).resolve().parent
    p.add_argument('--summary',type=Path,default=here/'inputs/gt_free_qwen_smoke_summary.json')
    p.add_argument('--results',type=Path,default=here/'inputs/qwen_results.jsonl')
    p.add_argument('--output',type=Path,default=here/'outputs')
    p.add_argument('--embed-images',action='store_true',help='Embed source marked images when referenced paths exist locally (e.g. on AutoDL)')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    summary=json.loads(a.summary.read_text(encoding='utf-8'));rows=load_jsonl(a.results)
    result, instances, dupes, report=evaluate(summary,rows,digest(a.results),embed_images=a.embed_images)
    (a.output/'semantic_gate_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output/'instance_evidence.json').write_text(json.dumps({'instances':instances,'duplicate_suspicions':dupes},ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output/'Review_5Instances.html').write_text(report,encoding='utf-8')
    fields=['case_id','instance_id','v5_category','v5_score','qwen_category','vqa_text','copy_text','flags','copy_content_disposition','publish_allowed']
    with (a.output/'instance_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for x in instances:w.writerow({**x,'flags':' ; '.join(x['flags'])})
    print('SEMANTIC_GATE_AUDIT_GENERATED',json.dumps({k:result[k] for k in ('images','requests','predicted_instances','v5_qwen_category_conflicts','copy_tag_category_conflicts','copy_debug_mentions','possible_duplicate_pairs','copy_display_blocked_instances')},ensure_ascii=False))
    print('REVIEW_HTML', a.output/'Review_5Instances.html')

if __name__=='__main__':main()
