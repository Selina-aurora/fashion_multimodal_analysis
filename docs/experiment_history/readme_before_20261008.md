# Fashion Multimodal Analysis

服装图像分析研究工程，覆盖服装实例分割、语言引导局部区域定位，以及颜色、图案和款式属性提取。当前 PRD 范围为 3.1.1—3.1.3；材质、面料和工艺属性不纳入本阶段完成项。

本版本按功能整理原工程：`scripts/` 提供分类命令入口，`src/fashion_multimodal_analysis/` 保存实现和共用逻辑，`reports/` 保存实验记录。最终上传的 reports 和 outputs 已合并，冻结 benchmark 与原始结果文件保持字节一致。

**本次修复：**已新增无已知 Core/val 源图路径交集的 v4-clean 训练清单（547 实例 / 475 图）、修正置信度匹配和完整模块计时、接通分割→局部定位→属性的预测流水线，并提供人工 GT 审核、冻结和验收报告入口。续改版补充 checkpoint 原子保存、完整轮次断点续训、阶段进度及验收模型校验。39 项 CPU 检查通过，2 项真实 PyTorch 加载/恢复检查因当前未安装 torch 而跳过；本包没有新的模型验收成绩。按 [AutoDL 开始步骤](docs/AUTODL_QUICK_START.md) 和 [修复运行指南](docs/RECOVERY_GUIDE.md) 执行：

```bash
python scripts/maintenance/run_prd31_recovery.py --phase all
```

## 当前进展与结论边界

| 模块 | 已有工作 | 当前结论 |
|---|---|---|
| 3.1.1 实例分割 | Mask R-CNN 8 类训练、消融、Core400 回归、错误诊断 | 原记录选定 V3-B1 Data Expansion，V2-balanced 为对照；尚不能据此声明独立盲测通过 |
| 3.1.2 局部定位 | Grounding DINO、ROI/空间先验、高分辨率、SAM 与人工审查 | PRD 八区域 40 例 baseline 严格正确 1/40；主要问题仍是粗框与漏检 |
| 3.1.3 属性提取 | 颜色、图案、连续几何、离散款式标签及预测 ROI 传播；保留 80 例历史人工审核 | 颜色 32/37、图案 34/40 为历史开发证据；新八类人工 GT 与实测验收待完成 |

**历史结果的已知问题：**按完整源图路径核对，B1 的 `train_v3` 与冻结 Core400 存在 **33 张源图重叠**（Core 中 shoe 12、bag 9、accessory 12）。历史清单和指标保持原样；新 `train_v4_clean` 已排除这些源图，重训尚未执行。Core 已有使用历史，不能仅凭排除训练路径就声称恢复为独立盲测。该检查没有覆盖改名图片或内容级重复。见 [原重叠审计](reports/repository_audit/data_overlap/summary.json)、[新清单审计](reports/repository_audit/recovery_v2/data_isolation/summary.json) 与 [修正记录](docs/CORRECTIONS.md)。

## 目录职责

| 目录 | 用途 |
|---|---|
| `src/fashion_multimodal_analysis/` | 算法实现、数据集、掩码处理、模型构建、采样和评估逻辑 |
| `scripts/` | 170 个分类入口：原 161 个命令、2 个整理检查、7 个 PRD 修复命令 |
| `configs/` | 数据清单、类别映射、属性 schema |
| `benchmark/` | 原样保留的 v1；新增 v2 开发证据、待审核清单、离线页面和冻结入口 |
| `reports/` | 最终上传的结果、表格和可视化；新整理审计在 `repository_audit/` |
| `outputs/` | 上传的裁剪、掩码与示例输出；模型权重单独保管 |
| `docs/` | 安装、命令、实验结果索引、版本核对与迁移说明 |
| `models/` | 权重清单与放置说明 |
| `tests/` | 共用逻辑与迁移后的 CPU 回归检查 |
| `archive/` | 原说明文档和 3 份备份脚本，便于追溯 |

共用逻辑已经从重复脚本中抽取：`datasets/prd8.py`、`image_processing/masks.py`、`segmentation/sampling.py`、`segmentation/modeling.py`、`evaluation/metrics.py`。85 处定义以原 AST 完全相同为前提进行合并；不同实验的 anchors 等差异继续保留。详见 [架构说明](docs/ARCHITECTURE.md)。

## 快速开始

在解压后的项目根目录执行。只查看命令或检查目录，无需加载任何模型：

```bash
python scripts/maintenance/check_project.py
python scripts/segmentation/training/train_prd_8class_maskrcnn_v3_b1_dataexp.py --help
```

安装基础依赖后，可运行命令目录、数据审计和 CPU 检查：

```bash
python -m pip install -e .
fashion-analysis list
python scripts/analysis/audit_dataset_overlap.py
python -m unittest discover -s tests -v
```

训练和推理另需模型依赖、外部数据及相应 checkpoint；完整步骤见 [SETUP](docs/SETUP.md) 和 [数据与权重](docs/DATA_AND_WEIGHTS.md)。`--help` 是从参数定义提取的静态帮助，通过帮助检查不代表模型已运行。

## 3.1.1 已记录结果

以下为上传报告中的历史 Core400 结果，score=0.40、mask=0.50、bbox 匹配 IoU=0.50。B1 存在上文所述源图重叠；此表仅用于准确整理历史记录。

| 指标 | V2-balanced | V3-B1 Data Expansion |
|---|---:|---:|
| bbox50 recall | 0.3625 | 0.3600 |
| 已定位实例分类正确率 | 0.6621 | 0.6944 |
| bbox50 且类别正确的端到端召回 | 0.2400 | 0.2500 |
| 已定位且类别正确子集 mean mask IoU | 0.7391 | 0.7677 |

B1 为 **580 个训练实例 / 508 张源图、15 epochs**；固定验证集为 43 个实例 / 32 张图。这里的训练规模指所选 manifest，不能写成 DeepFashion2/Fashionpedia 全量数据训练。

B1 全部已定位实例的 mean mask IoU 为 0.7221，与表中正确类别子集的 0.7677 分母不同；旧文档的 0.7442 属于 val43，不能标为 Core400。记录的平均 24.7143 ms 为模型调用计时，未覆盖完整端到端流程。

## 阅读顺序

1. [修复运行指南](docs/RECOVERY_GUIDE.md)、[当前总览](docs/PRD_3_1_FINAL_SUMMARY.md) 与 [结果索引](docs/RESULTS_INDEX.md)
2. [安装与复现](docs/SETUP.md)、[全部命令](docs/COMMANDS.md)
3. [重构与迁移](docs/ARCHITECTURE.md)、[旧新路径对照](docs/SCRIPT_MIGRATION.csv)
4. [核对修正](docs/CORRECTIONS.md)、[验证记录](docs/VALIDATION.md)
5. [上传 GitHub](docs/GITHUB_UPLOAD.md)

原始阶段性文档在 `archive/docs_original/`；与当前汇总有冲突时，请结合 `CORRECTIONS.md` 回到对应的实验 summary/CSV 核对。本仓库没有新增模型成绩，也没有将诊断结果改写成最终验收结论。

## AutoDL 运行反馈与加载修正

用户 2026-09-30 的终端截图确认 v4-clean 已完成 15 轮，最后一轮平均训练 loss 为 0.2713。随后 Core 评估因 PyTorch 默认仅加载权重而无法读取新增 NumPy 随机状态。本版已显式加载工程训练产生的完整 checkpoint，可复用现有模型继续评估；训练 loss 不是 mask IoU。详细恢复命令见 [加载修正说明](docs/CHECKPOINT_LOADING_FIX.md)。
