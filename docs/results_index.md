# 全部已取得结果索引

目录保留实验名称和版本，不把旧结果混成最新成绩。原始结果文件在 reports/outputs/benchmark，汇总工具只写 `reports/repository_summary/saved_evaluations.csv`。

## 先看完整 V8

[全量训练解释](full_dataset_training_v8_report.md) 与 [原始 V8 目录](../reports/reruns/recovery_v2/v8_full_dataset/)：准备说明、排除清单、训练摘要、最终状态、旧进度/失败日志、四组 V5/V8 × Core/开发对照均保留。

## V4–V8 和速度修复

| 版本 / 实验 | 结果位置 | 阅读重点 |
|---|---|---|
| V4 clean | `reports/reruns/recovery_v2/train_v4_clean/`、`core_v4_clean/` | 清理已知重叠后的训练/固定 Core 回归 |
| V4 诊断 | `diagnosis/` 及对应 analysis 入口 | 误差类别与协议对照 |
| V5 full targets | `reports/reruns/recovery_v2/v5_full_targets/`、`core_v5_full_targets/` | 全目标清单、15 epoch 版本及后续基线 |
| V5 speed fix | `reports/reruns/recovery_v2/v5_speed_fix/` | 固定精度核对、候选 runtime 与筛选后 CPU 传输计时 |
| V6 coverage | `reports/reruns/recovery_v2/v6_coverage/` | 扩展覆盖、逐类回归及退化诊断 |
| V7 FP heads | `reports/reruns/recovery_v2/v7_fp_heads/` | FP 类别头尝试，不据目录名判断改进成功 |
| V8 full dataset | `reports/reruns/recovery_v2/v8_full_dataset/` | 全量一轮、真实范围、四组完整对照、最终状态 |

## 主报告树

| 目录 | 文件数（生成索引时） | 内容 |
|---|---:|---|
| [category_coverage_v1](../reports/category_coverage_v1/) | 2 | 原始实验/采样/诊断结果，按目录和文件查看 |
| [daily](../reports/daily/) | 1 | 原始实验/采样/诊断结果，按目录和文件查看 |
| [environment](../reports/environment/) | 2 | 上传过的运行环境记录 |
| [evaluation_sampling](../reports/evaluation_sampling/) | 6 | 原始实验/采样/诊断结果，按目录和文件查看 |
| [grounding_failure_analysis](../reports/grounding_failure_analysis/) | 5 | 局部定位错误归因与案例 |
| [grounding_group_evaluation](../reports/grounding_group_evaluation/) | 4 | 按组/目标比较局部区域 |
| [grounding_visibility_filtering](../reports/grounding_visibility_filtering/) | 1 | 可见性筛选及局部结果 |
| [prd_3_1_v1](../reports/prd_3_1_v1/) | 2,107 | 原 PRD 对照、审核及基准说明 |
| [prd_attribute_extraction](../reports/prd_attribute_extraction/) | 106 | 3.1.3 颜色、几何、花纹、设计属性 |
| [prd_instance_segmentation](../reports/prd_instance_segmentation/) | 649 | 3.1.1 分割各版训练、评估、阈值扫描与误差 |
| [prd_integration](../reports/prd_integration/) | 104 | 集成链路及输入来源说明 |
| [prd_region_coverage](../reports/prd_region_coverage/) | 79 | 3.1.2 PRD 局部区域覆盖 |
| [prompt_diagnostic_2026_09_14](../reports/prompt_diagnostic_2026_09_14/) | 3 | 原始实验/采样/诊断结果，按目录和文件查看 |
| [repository_audit](../reports/repository_audit/) | 28 | 原项目整理检查及来源审计 |
| [reruns](../reports/reruns/) | 296 | 后续恢复、V4–V8 固定协议及诊断 |
| [roi_conditioned_grounding](../reports/roi_conditioned_grounding/) | 2 | 衣物 ROI 条件的局部区域定位 |
| [segmentation_first_diagnostic_2026_09_14](../reports/segmentation_first_diagnostic_2026_09_14/) | 3 | 原始实验/采样/诊断结果，按目录和文件查看 |
| [target_balanced_candidates](../reports/target_balanced_candidates/) | 5 | 原始实验/采样/诊断结果，按目录和文件查看 |
| [uploaded_summaries](../reports/uploaded_summaries/) | 8 | 单独上传评估摘要 |
| [verified_positive_benchmark](../reports/verified_positive_benchmark/) | 2 | 原始实验/采样/诊断结果，按目录和文件查看 |
| [verified_positive_evaluation](../reports/verified_positive_evaluation/) | 9 | 原始实验/采样/诊断结果，按目录和文件查看 |
| [verified_positive_evaluation_pilot_v2](../reports/verified_positive_evaluation_pilot_v2/) | 6 | 原始实验/采样/诊断结果，按目录和文件查看 |

## 预测、基准和审核

`outputs/` 保留取得的分割、局部定位、属性和集成输出；`benchmark/` 保存固定清单、项目内 GT 掩码、候选/审核 CSV、联系表与网页。AI 预审核及 AI annotations 保留来源身份，不能替代人工最终审核。

运行 `python scripts/maintenance/summarize_saved_results.py` 可重建已保存评估表。统一协议与 legacy_summary 分开标记，旧实验没有 TP/FP 等字段时留空，不推造缺失指标。逐 GT、逐图片计时、错误清单仍需回原目录阅读。

每个实际交付文件及 SHA 见根目录 `package_contents_sha256.csv`，来源见 `docs/repository_organization/source_file_inventory.csv`。V8 没有随结果 ZIP 提供可视化 PNG，缺失模型也不计成已交付。
