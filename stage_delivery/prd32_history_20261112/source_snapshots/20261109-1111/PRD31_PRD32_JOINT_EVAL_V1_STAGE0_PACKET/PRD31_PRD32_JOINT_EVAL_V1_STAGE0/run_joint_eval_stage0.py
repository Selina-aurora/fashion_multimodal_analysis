#!/usr/bin/env python3
"""PRD31+PRD32 DEV60 joint Stage-0 read-only evidence audit.
No GPU, no model calls, no holdout access. Requires Python 3.9+ standard library.
"""
from __future__ import annotations
import argparse, csv, json, hashlib, statistics
from pathlib import Path
from collections import Counter, defaultdict
from datetime import date


def rows(path):
    with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def csvout(path, data, columns):
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        w.writeheader();w.writerows(data)

def truth(v):return str(v).strip().lower() in ('true','1','yes')

def fail(msg):raise RuntimeError('INTEGRITY_CHECK_FAILED: '+msg)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',type=Path,default=Path(__file__).resolve().parent/'inputs')
    ap.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'results')
    a=ap.parse_args()
    i=a.input_dir;o=a.output_dir;o.mkdir(parents=True,exist_ok=True)
    r=rows(i/'dev60_review_171.csv');t=rows(i/'warm48_per_request_timing.csv');m=rows(i/'focus45_mask_review_original.csv');sc=rows(i/'scorecard_v5_original.csv')
    q=[json.loads(x) for x in (i/'warm48_qwen_calls.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    jsummary=json.loads((i/'warm48_summary_original.json').read_text(encoding='utf-8'))
    idx={x['review_key']:x for x in r}
    if len(idx)!=len(r):fail('duplicate review_key in DEV60 review')
    if len(r)!=171:fail(f'expected 171 reviewed V5 prediction rows, got {len(r)}')
    cases=sorted(set(x['case_id'] for x in r))
    if len(cases)!=60:fail(f'expected 60 DEV images, got {len(cases)}')
    h=defaultdict(set)
    for x in r:
        if x['review_key']!=x['case_id']+'/'+x['instance_id']:fail('review_key not equal to case_id/instance_id')
        h[x['case_id']].add(x['image_sha256'])
        if not x['image_sha256']:fail('empty source image SHA')
    if any(len(v)!=1 for v in h.values()):fail('inconsistent image sha within DEV image')
    determinate=[x for x in r if x['category_gt_status']=='DETERMINATE']
    unresolved=[x for x in r if x['category_gt_status']=='UNRESOLVED']
    if len(determinate)+len(unresolved)!=len(r):fail('unrecognized category status')
    reviewed=Counter(x['review_confirmed'] for x in r)
    counts=Counter(x['case_id']+'/'+x['instance_id'] for x in t)
    if len(t)!=48 or len(counts)!=24 or any(v!=2 for v in counts.values()):fail('warm48 expected 24 targets x 2')
    if any(k not in idx for k in counts):fail('a warm target is missing from reviewed DEV60')
    nonmatched=[]
    for x in t:
        k=x['case_id']+'/'+x['instance_id'];rr=idx[k]
        if x['source_image_sha256']!=rr['image_sha256']:nonmatched.append((k,'image_sha256'))
        if x['producer_category']!=rr['v5_original_category']:nonmatched.append((k,'v5_category'))
    if nonmatched:fail(f'warm linkage mismatches: {nonmatched[:10]}')
    warm_cohorts=set(x['cohort'] for x in t)
    actual_q=[x for x in q if x['cohort']!='warmup_excluded']
    by_cohort=defaultdict(list)
    for x in actual_q:by_cohort[x['cohort']].append(x)
    if len(actual_q)!=144 or set(by_cohort)!=warm_cohorts:fail('wrong Qwen call count or cohorts')
    if any(set(v['task'] for v in xx)!={'tags','vqa','copy'} or len(xx)!=3 for xx in by_cohort.values()):fail('incomplete per-cohort Tags/VQA/Copy')
    for x in t:
        key=(x['case_id'],x['instance_id'])
        if any((v['case_id'],v['instance_id'])!=key for v in by_cohort[x['cohort']]):fail('Qwen target/cohort mismatch')
    maskidx={x['review_key']:x for x in m}
    if len(m)!=45 or len(maskidx)!=45:fail('unexpected Mask45 cardinality')
    wrongkeys={x['review_key'] for x in r if x['focus_category_correct']=='0'}
    if wrongkeys!=set(maskidx):fail('Mask45 cases are not the complete Focus45-error set')
    allowed={'PILOT_CANDIDATE','REVIEW_FIRST','EXCLUDE'}
    if any(x['suggested_mask_ab_action'] not in allowed for x in m):fail('unexpected Mask action')
    if any(truth(x['publish_allowed']) or truth(x['holdout_accessed']) for x in t):fail('unsafe warm record')
    # Original Mask source remains AI-prefill/pending in CSV: chat-level user confirmation does not alter file provenance.
    bykey=defaultdict(list)
    for x in t:bykey[x['case_id']+'/'+x['instance_id']].append(x)
    linked=[]
    for x in sorted(r,key=lambda z:(z['case_id'],z['instance_id'])):
        k=x['review_key'];tt=bykey[k];mask=maskidx.get(k,{})
        linked.append({
          'review_key':k,'case_id':x['case_id'],'instance_id':x['instance_id'],
          'image_sha256':x['image_sha256'],'v5_pred_category':x['v5_original_category'],
          'confirmed_category_user_chat':x['confirmed_category'],
          'category_gt_status':x['category_gt_status'],
          'focus_correct_on_reviewed_pred':x['focus_category_correct'],
          'vqa_color_correct_if_evaluable':x['vqa_color_correct'],
          'full_tags_suggestion_correct_if_evaluable':x['full_tags_suggestion_correct'],
          'copy_rating_if_evaluable':x['copy_satisfaction_1_to_5'],
          'review_confirmed':x['review_confirmed'],
          'review_provenance':x['confirmation_provenance'],
          'warm_live_repeat_count':len(tt),
          'warm_live_mean_e2e_ms':round(statistics.mean(float(y['total_host_wall_ms']) for y in tt),3) if tt else '',
          'warm_live_all_guard_blocked': all(y['guard_status']=='REVIEW_REQUIRED_NOT_RELEASED' for y in tt) if tt else '',
          'focus45_mask_preflight_action_AI_prefill':mask.get('suggested_mask_ab_action',''),
          'focus45_mask_source_status':mask.get('new_mask_visual_confirmation',''),
          'requires_complete_true_object_gt':'YES',
          'data_scope':'DEV60_PREDICTED_INSTANCES_ONLY'
        })
    csvout(o/'JOINT_171_EVIDENCE_LINKAGE.csv',linked,list(linked[0]))
    per_image=[]
    for c in cases:
        rr=[x for x in r if x['case_id']==c]
        per_image.append({'case_id':c,'image_sha256':next(iter(h[c])),'v5_predicted_instance_count':len(rr),
          'true_garment_count_from_complete_GT':'','complete_GT_available':'NO',
          'true_garment_instance_ids':'','unmatched_true_garments':'',
          'duplicate_prediction_notes':'','gt_annotation_source':'','reviewer':'','review_date':'',
          'notes':'待独立标注/核对；空白不代表零漏检'})
    csvout(o/'DEV60_TRUE_OBJECT_GT_INTAKE_TEMPLATE.csv',per_image,list(per_image[0]))
    n_category_correct=sum(x['focus_category_correct']=='1' for x in determinate)
    n_baseline_correct=sum(x['baseline_category_correct']=='1' for x in determinate)
    n_v5_correct=sum(x['v5_original_category']==x['confirmed_category'] for x in determinate)
    tmeans={k:round(statistics.mean(float(x[k]) for x in t),3) for k in ['total_host_wall_ms','v5_host_wall_ms','tags_task_host_wall_ms','vqa_task_host_wall_ms','copy_task_host_wall_ms','guard_host_wall_ms']}
    maskactions=dict(Counter(x['suggested_mask_ab_action'] for x in m))
    s={
      'schema':'joint_dev60_stage0_evidence_audit_v1',
      'scope':'read_only_existing_DEV60_171_reviews_and_24target_warmE2E',
      'status':'STAGE0_COMPLETED_NOT_JOINT_E2E_ACCEPTANCE',
      'evidence':{'reviewed_predicted_instances':len(r),'source_images':len(cases),
        'user_chat_confirmed_reviews':reviewed.get('YES',0),
        'category_determinate':len(determinate),'category_unresolved':len(unresolved),
        'v5_category_correct_in_determinate_predicted_instances':n_v5_correct,
        'qwen_baseline_category_correct_in_determinate_predicted_instances':n_baseline_correct,
        'qwen_focus_category_correct_in_determinate_predicted_instances':n_category_correct,
        'focus_wrong':len(wrongkeys),'warm_e2e_unique_targets':len(counts),
        'warm_e2e_source_images':len(set(x['case_id'] for x in t)),
        'warm_e2e_requests':len(t),'warm_qwen_calls':len(actual_q),
        'warm_all_three_task_cohorts':len(by_cohort),
        'warm_request_mean_ms':tmeans,'warm_all_guard_blocked':all(x['guard_status']=='REVIEW_REQUIRED_NOT_RELEASED' for x in t),
        'warm_all_holdout_accessed_false':all(not truth(x['holdout_accessed']) for x in t),
        'mask45_action_counts_AI_prefill':maskactions},
      'coverage_gaps':{'complete_true_object_GT':'NOT_AVAILABLE',
        'v5_missed_true_garment_recall':'NOT_COMPUTABLE',
        'v5_mask_iou_vs_GT':'NOT_COMPUTABLE',
        'full_DEV60_real_warm_pipeline':'NOT_RUN',
        'Oracle_vs_Predicted':'NOT_RUN',
        'PRD31_3_1_2_localization_in_joint_pipeline':'NOT_VERIFIED',
        'PRD31_3_1_3_attributes_in_joint_pipeline':'NOT_VERIFIED',
        'full_multitype_VQA_accuracy':'NOT_COMPUTABLE_FROM_THIS_DATA',
        'formal_PRD32_tags_accuracy':'NOT_COMPUTABLE_FROM_THIS_DATA',
        'formal_PRD32_copy_satisfaction':'NOT_COMPUTABLE_FROM_THIS_DATA',
        'independent_holdout':'SEALED_NOT_ACCESSED'},
      'source_caveats':['171 rows are V5 predicted instances, not true garment census',
         '171 reviews were user-chat confirmations of AI-prefilled suggestions, not independent blind GT',
         'Mask45 original CSV still carries PENDING_USER_REVIEW despite subsequent chat-level confirmation; original is not rewritten',
         'warm 48 requests are 24 targets x 2 repeats; they cannot be substituted for 171 unique targets',
         'no experiment or source files modified; no GPU calls executed in this stage']}
    (o/'SUMMARY.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
    mdr=f"""# PRD 3.1 + 3.2 联合评估 V1｜今晚 Stage 0：证据链核对（已完成）

> 状态：**离线证据核对完成；不等于完成 DEV60 全量联合推理，也不等于 PRD 验收。**
> 范围：只读现有 DEV60 记录，**未调用 GPU，未修改模型、Prompt、Guard 或主流程，未访问 Holdout**。

## 今晚实际核验到的结果

- **60 张 DEV 图片 / 171 条 V5 预测实例**，171 条均有用户在聊天中的批量确认记录；其中 **140 条类别可判定，31 条类别未定**。这里的 171 **不是完整真实服装数量**。
- Focus V1 在上述 140 条类别可判定预测实例中正确 **{n_category_correct}/140**；Baseline **{n_baseline_correct}/140**；V5 **{n_v5_correct}/140**。三者均不能替代 3.1 实例召回或者 PRD 3.2 完整标签验收准确率。
- 核对了今天的真实常驻 E2E 记录：**24 个不同目标，来自 {len(set(x['case_id'] for x in t))} 张图，2 次/目标，共 48 次请求与 144 次 Qwen 实测调用**。全部目标键在 171 条审核中找到；V5 类别与原图 SHA 对应一致；每个请求都有 Tags、VQA、Copy 三项。
- 48 次真实请求平均总时延 **{tmeans['total_host_wall_ms']:.2f} ms**；48/48 Guard 阻断，无自动发布；这一组只覆盖 24 个目标，不能称为 DEV60 全量联测。
- 已有的 45 条 Focus 错误与 Mask 预审表逐键核对成功，首轮候选 **{maskactions.get('PILOT_CANDIDATE',0)}** 条；需复核 **{maskactions.get('REVIEW_FIRST',0)}** 条；排除 **{maskactions.get('EXCLUDE',0)}** 条。注意原 Mask CSV 仍标为 AI 预填待确认，尽管用户已在聊天中口头确认判断，文件没有被篡改为独立 GT。

## 今晚确定的评估边界

1. **不能从 171 条预测实例推断 3.1 漏检率**。生成了 `DEV60_TRUE_OBJECT_GT_INTAKE_TEMPLATE.csv`（60 行），用来补充每图完整真实服装 GT、漏检、重复预测。空白不能当作 0。
2. **目前的 24×2 warm E2E 是性能与调用溯源基线，不是 60 图 / 171 目标的全量质量基线**。后续必须单独跑 DEV60 所有目标，并对失败案例留痕。
3. **Tags/VQA/Copy 的正式 PRD 指标不能仅凭当前代理分数算出**：需要固定问题类型、字段 GT、文案人工评分规则与统一分母。
4. 此联合链路目前能确认 V5→标记整图→Qwen→Guard；**3.1.2 局部定位、3.1.3 细粒度属性提取是否在该链路实际调用，尚无充分运行证据**。
5. **独立 Holdout 继续封存**；候选方案进入独立验证前，不能依据开发集汇报生产达标。

## 下一步只做一件最有价值的事

以当前冻结 V5、Qwen 和 Guard 为基线，补齐完整 DEV60 真实目标清单并检查 171 个预测实例的匹配、重复和漏检；随后进行 60 图全量真实链路回归。先检查数据完整性，不急于重新训练模型。

**输出文件**：`JOINT_171_EVIDENCE_LINKAGE.csv`（171 条对应关系）；`DEV60_TRUE_OBJECT_GT_INTAKE_TEMPLATE.csv`（60 图真实 GT 模板）；`SUMMARY.json`（可机读状态和未完成项）。
"""
    (o/'JOINT_EVAL_STAGE0_REPORT.md').write_text(mdr,encoding='utf-8')
    print('STAGE0_OK',json.dumps({'images':len(cases),'reviewed':len(r),'warm_requests':len(t),'qwen_calls':len(actual_q),'focus_errors':len(wrongkeys),'mask_actions':maskactions,'mean_ms':tmeans['total_host_wall_ms']},ensure_ascii=False))
if __name__=='__main__':main()
