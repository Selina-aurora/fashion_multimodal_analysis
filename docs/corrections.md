# 版本与结论核对记录

整理日期：2026-09-30。原始数据、清单、benchmark 与报告不回填新结论；修正在当前文档和新增审计中说明。历史说明原文保存在 `archive/`。

| 核对项 | 原资料中的问题 | 本版处理 |
|---|---|---|
| B1 与 Core 的数据独立性 | `train_v3` 与 Core 有 33 张相同源图路径 | 保留原清单；新 v4-clean 排除全部 Core/val 源图路径，共 547 实例 / 475 图，重训待执行 |
| B1 训练规模 | 容易被写成原始数据集全量训练 | 明确为 580 实例 / 508 图、15 epochs，固定 val 43 实例 / 32 图 |
| Core mask IoU | 旧 `prd_3_1_final_summary.md` 将 val43 的 0.7442 标为 Core | Core 已定位均值 0.7221、正确类别子集均值 0.7677，分开写明 |
| fullval_v1 模型归属 | 部分文字说明沿用 B1 名称 | 按实际 ROI summary 的 checkpoint 确认为 V2-balanced；与 B1 v2 pilot 分列 |
| 属性指标名称 | ROI 一致性容易写成属性准确率 | 明确为 GT-ROI 输出与 predicted-ROI 输出的标签一致率，无人工属性真值准确率结论 |
| 两组 40 例 | sleeve/collar/button/zipper 与 PRD 八区域审查容易混用 | 分别标识 4×10 与 8×5 数据；不合并分母 |
| 4 类定位错误分布 | 旧文字遗漏 footwear 4 例，合计只有 36 | 按 reviewed.csv 补齐，合计 40 |
| visibility filtering | 使用人工 target_absent 标签的模拟可能被写成自动模块收益 | 标为 oracle 诊断；40 中排除 15 后余 25，不新增准确率提升结论 |
| 推理时间 | 24.7143 ms 容易被解释为完整流程耗时 | 保留历史值；新运行计入预处理、模型和后处理，10 预热 / 5 passes / CUDA 同步 |
| 预测匹配 | 旧评估按 IoU 全局贪心，违反冻结置信度优先定义 | 八个活动评估入口共用置信度降序一对一匹配；新报告目录不覆盖历史记录 |
| 人工属性真值 | 传播一致性和人工准确率容易混用 | 另接入已上传颜色 32/37、图案 34/40 人工审核，保留为开发证据；新审核集独立统计 |
| 完整流水线 | 只统计匹配子集，容易漏掉上游失败 | 处理全部预测实例；事后匹配评价，所有适用 GT 保留在 Track B 分母；领口使用预测局部 ROI |
| 默认 checkpoint | 多个 v2/v3 评估入口仍默认 baseline_v1 权重 | 改为与实验版本对应；增加 `--report-dir`，运行示例显式指定清单和阈值 |
| 目录迁移 | `__file__` 层级、跨脚本 import、GPU 打包仅拷脚本会失效 | 共用根目录解析、绝对包 import、完整 runtime 复制；验证可迁移入口 |

## 33 张源图重叠

比较时使用规范化的完整 `source_image` 路径，不只比较文件名；同一源图里的不同实例仍构成源图重叠。结果位于 [summary.json](../reports/repository_audit/data_overlap/summary.json) 和 [train_core_overlaps.csv](../reports/repository_audit/data_overlap/train_core_overlaps.csv)。

Core 类别分布：shoe 12、bag 9、accessory 12；全部来自 Fashionpedia。训练 v1/v2 与 Core 的同口径交集为 0；三个训练版本与固定 val、val 与 Core 的同口径交集也为 0。该方法不能证明不存在改名、复制或感知相似的图片。

新排除集合与训练清单已经生成，路径交集为 0，详见 [新数据清单审计](../reports/repository_audit/recovery_v2/data_isolation/summary.json)。实际重训、同协议测量、人工 GT 及严格定位语义审核仍需执行；现有代码入口见 [RECOVERY_GUIDE](recovery_guide.md)。不修改原实验成绩，不恢复 Core 的盲测独立性。

## 证据优先级

模型归属以具体 run 的 checkpoint 字段为准；指标以该 run 的 summary 和逐实例/汇总 CSV 为准；阶段性汇报文字只作历史背景。对应来源见 [RESULTS_INDEX](results_index.md)。
