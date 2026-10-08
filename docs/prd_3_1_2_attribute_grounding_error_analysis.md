# PRD 3.1.2 局部定位：核对后的分析

## A. sleeve/collar/button/zipper 错误审查

来源：[manual_localization_audit_reviewed.csv](../reports/grounding_group_evaluation/manual_localization_audit_reviewed.csv)。四类各 10 例，总计 40。localization_quality 为 coarse 6、wrong 34。

| error_type | 例数 |
|---|---:|
| whole_garment | 4 |
| wrong_part | 5 |
| target_absent | 15 |
| face_head | 4 |
| non_clothing_object | 4 |
| background_object | 2 |
| whole_image | 2 |
| footwear | 4 |
| 合计 | 40 |

旧文字遗漏 footwear 的 4 例，本版补齐。按人工 error_type 排除 target_absent 的 15 例后剩 25 例，这只是 oracle 可见性筛选诊断；不代表已有可部署的自动可见性模块，也没有据此新增模型准确率提升。

可复算脚本：`scripts/analysis/analyze_visibility_filtering.py`；派生输出在 `reports/repository_audit/visibility_filtering/`。上传的 `reports/grounding_visibility_filtering/` 保留不变。

## B. PRD 八区域正式 40-case baseline

来源：[prd_40case_formal_manual_summary_final.md](../reports/prd_region_coverage/prd_40case_formal_manual_summary_final.md)。每区域 5 例，严格 correct、coarse、wrong、missed 分开计数。

| Region | Correct | Coarse | Wrong | Missed |
|---|---:|---:|---:|---:|
| collar | 0 | 5 | 0 | 0 |
| cuff | 0 | 4 | 0 | 1 |
| hem | 0 | 3 | 0 | 2 |
| pocket | 0 | 0 | 0 | 5 |
| shoulder | 0 | 2 | 0 | 3 |
| waist | 0 | 3 | 0 | 2 |
| pattern | 1 | 2 | 0 | 2 |
| decoration | 0 | 0 | 0 | 5 |
| 合计 | 1 | 19 | 0 | 20 |

严格正确率为 1/40=2.5%，correct+coarse 的宽松可用率为 20/40=50%。返回了框不能直接等同于细粒度定位正确。`pattern_019` 为唯一严格正确案例，依据是其条纹本身覆盖几乎整个裤装区域。

## 后续定向诊断

定向实验保留了空间高分辨率、肩腰/图案细化、口袋/装饰提示尺度诊断等结果。汇总表已从 scripts 移至 [final_summary_v1](../reports/prd_region_coverage/final_summary_v1/)。

| 定向条件 | 目标 | Correct | Coarse | Missed |
|---|---|---:|---:|---:|
| spatial HR | collar | 0 | 4 | 1 |
| spatial HR | cuff | 0 | 5 | 0 |
| spatial HR | hem | 0 | 0 | 5 |
| 定向细化 | shoulder | 0 | 5 | 0 |
| 定向细化 | waist | 0 | 5 | 0 |
| 定向细化 | pattern | 1 | 4 | 0 |
| prompt/scale | pocket | 0 | 5 | 0 |
| prompt/scale | decoration | 1 | 4 | 0 |

SAM 9-case 审查为 correct 0、coarse 8、wrong 1。以上不同诊断不能合并成一个统一模型的 40 例新成绩；主要证据仍指向“能够给出响应，但细粒度边界与语义准确性不足”。
