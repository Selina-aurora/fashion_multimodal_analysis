# 服饰多模态分析实习项目

本项目面向电商服饰图片的结构化解析，结合视觉模型与文本语义信息，开展服饰实例分割、局部区域定位和可见属性提取。当前主要工作集中在 PRD 3.1 模块，连接实例预测、ROI 处理与属性分析，并通过版本对照、人工审核和错误案例分析定位质量问题。

## 当前阶段提交（2026-10-23）

当前采用未编译 `white-sam` 候选，保留 `white` 回退。固定 24 图 × 5 轮完整请求均值 4272.34 → 3590.79ms（下降 15.95%）；单查询核心均值 139.63ms，30ms 目标仍未达到。新增 20 图工程一致性与每图一个选定目标的 160 项人工定性复核完成；正式精度仍缺完整同版本 GT。

SAM 非空候选回退暂不采用，静态编译因第二图特征超限停止；人工矩形 ROI 方向关闭。当前阶段说明、复现命令、源码来源与遗留问题见 [阶段提交入口](stage_delivery/prd31_20261023/README.md)。候选 37 个原文件保持字节一致；6 个活动集成文件同步到 GPU 已验证的固定源码。数据、权重及离线模型缓存须另行提供，没有重新执行 GPU 推理或宣称全部 PRD 验收通过。

下方 10 月 15 日和 V8 内容保留为历史记录。

## 模块与当前范围

| 模块 | 作用与输出 | 当前状态 |
|---|---|---|
| 3.1.1 实例分割 | 八类服饰的类别、置信度、边界框与实例掩码 | 已保存基线、V5/V8 对照及全量一轮训练结果；质量仍需优化 |
| 3.1.2 局部区域 | 基于服饰实例开展细粒度区域定位、裁剪与区域评估 | 已有定位实验和审核材料，覆盖与边界质量继续验证 |
| 3.1.3 属性提取 | 基于 ROI 提取颜色、图案及适用的设计标签 | 已有基线与预测 ROI 联调；当前不以面料、材质、工艺为重点 |

流程基本连接不等于全部 PRD 验收通过；其他需求范围以 项目原始 PRD（本次文字包未含原件） 为准，不能由本页推断其已实现。

## 历史阶段 2026年10月15日

当前 GPU 实验基线为恢复后的 V5 Balanced150 epoch 15。3.1 流程已连接，已完成 Core400 回归、局部查询批处理对照和同批 12 图全链路运行；人工真值支持的速度精度对照仍未收齐，不能据此宣布完整 PRD 验收通过。批处理候选仍保留为实验方案。

| 已核对的结果 | 数值与范围 |
|---|---|
| Core400 类别正确召回 | 202/400，50.50%；仅定位召回 281/400，70.25% |
| Core400 类别正确匹配的平均 mask IoU | 0.7546，分母为202个匹配；不能当作全部GT平均值 |
| Core400 分割平均 / P95 耗时 | 30.6055 / 46.5236 ms/图 |
| Stage14 每件服饰4个查询的平均耗时 | 830.9668 → 544.9605 ms/组，下降34.42% |
| Stage14 输出一致性 | 48/48在容差内一致；33个双方检出的mask中32个逐像素相同；不是人工真值精度 |
| 同批12图完整链路 | 36次计时调用，平均5948.59 ms/图，P95 12510.23 ms/图；零失败调用 |

完整链路包括每图全部预测服饰、配置区域和属性，不能与单区域或单属性耗时直接比较。8图连续袖长诊断未显示稳定收益；软标签已有候选及隔离核查，尚未完成六类模型训练。

本次更新保留 GitHub 已有的核心说明和注释，新增 GPU 当前实现与实验文字证据；独立实验按模块归档，原始数据文件不改写。**此更新包不是 GPU 全量备份**：图片、mask、HTML审核页、权重和缓存不在本次文字包内；远端已有图片不会被删除。

当前 checkpoint 位于 GPU 的 `outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth`，运行记录中的 SHA-256 为 `c5ba795225349b1c521aac0517ac7c493d9818dcc17d558ac4a3f6f8669452f0`。本次上传未包含权重字节。历史 V8 报告继续留档，不能将历史 V8 与当前恢复的 V5 Balanced150 混为同一次实验。

优先阅读 [本次更新与结果入口](docs/updates/update_20261015.md)、[关键代码说明](docs/updates/code_notes_20261015.md) 和 [文件对应清单](docs/updates/file_map_20261015.csv)。下方 V8 表格及旧说明为历史实验记录。

## 先看哪些文件

1. [V8 全量训练报告](docs/full_dataset_training_v8_report.md)：实际范围、训练设置、全部对照结果和可解释的限制。
2. [全部结果索引](docs/results_index.md)：早期版本、V4–V8、3.1.2/3.1.3 及审核材料的位置。
3. [模块说明](docs/module_guide.md)：输入输出、坐标系统、采样、掩码和计时设计。
4. [GPU 恢复与运行](docs/gpu_restore_and_run.md)、[GitHub 上传](docs/github_upload.md)：按工作副本执行的操作步骤。
5. [编码规范落实](docs/coding_standard_compliance.md)、[验证报告](docs/validation_report.md)：具体检查范围与未做的 GPU 检查。

## 从哪里阅读核心代码

`scripts/` 是可运行的薄入口，所以多个脚本结构相似；训练、数据处理和属性逻辑位于 `src/fashion_multimodal_analysis/`。阅读时从对应实现开始，运行时仍使用原 scripts 命令。

| 阅读内容 | 核心实现 |
|---|---|
| V8 数据准备、训练、恢复与回归 | [全量工作流](src/fashion_multimodal_analysis/segmentation/training/run_prd31_v8_full_dataset.py) |
| 类别权重、逐图采样及 V8 的差别 | [采样实现](src/fashion_multimodal_analysis/segmentation/sampling.py) |
| COCO 初始化与八类预测头 | [模型构造](src/fashion_multimodal_analysis/segmentation/modeling.py) |
| 二值掩码与裁剪到整图坐标转换 | [掩码处理](src/fashion_multimodal_analysis/image_processing/masks.py) |
| 预测 ROI 属性传递及一致性统计 | [属性联调](src/fashion_multimodal_analysis/integration/run_prd31_predicted_roi_attributes_v2.py) |

## 错误分析入口

- [V8 全量报告](docs/full_dataset_training_v8_report.md)：训练类别分布、V5/V8 逐类变化及解释限制。
- [V8 逐类别指标](reports/reruns/recovery_v2/v8_full_dataset/eval_core400_regression_v8_epoch_01/per_class_metrics.csv)：定位漏检、定位后错类及误检计数。
- [V8 逐实例结果](reports/reruns/recovery_v2/v8_full_dataset/eval_core400_regression_v8_epoch_01/per_gt_results.csv) 与 [错误案例](reports/reruns/recovery_v2/v8_full_dataset/eval_core400_regression_v8_epoch_01/error_cases.csv)：实例类别、匹配、预测类别及掩码表现。
- [历史 grounding 错误分析](reports/grounding_failure_analysis/failure_analysis_summary.md)：属于此前定位实验，不作为 V8 错误分析的替代。

V8 的鞋、包、配饰主要表现为漏检，外套主要表现为定位后错类。类别不平衡是待验证的解释，尚未通过单因素实验确认。CSV 中的历史 visualization_path 不代表对应图片已包含在当前备份中。

## 分类与命名

| 目录 | 用途 |
|---|---|
| `src/fashion_multimodal_analysis/` | 可复用实现，按 segmentation、grounding、attributes 等职责分类 |
| `scripts/` | 与实现对应的薄入口；不复制训练与推理逻辑 |
| `configs/` | 训练/验证清单和配置；冻结清单保留原始字节 |
| `benchmark/` | 固定测试、实例掩码、人工审核与 AI 预审核材料 |
| `reports/` | 指标、逐例结果、计时、诊断、历史日志和 V4–V8 回归 |
| `outputs/` | 预测、属性表、可视化和训练输出的可取得部分 |
| `models/` | 权重说明及目录；可取得的三个权重仅随 GPU 包提供 |
| `docs/` | 使用说明、技术解释、来源清单和上传过的规范/PRD |
| `tests/` | 几何、采样、坐标、清单身份、状态和文件迁移测试 |
| `docker/` | 保留原上传容器文件；GPU 使用前按实际环境调整 |
| `archive/` | 不同分支、补丁快照、冲突版本及原始算法参考，不作为活动入口 |

目录和普通文件统一为小写加下划线。`__init__.py`、`.gitignore`、`.github/` 等工具协议名按其规定保留。项目路径相对项目根目录；外部图片默认在同级 `../fashion_data`，可通过 `FASHION_DATA_ROOT` 指定。冻结 CSV 中原来使用的大写掩码名通过别名映射解析，避免改写测试集哈希。

## 安装和只读检查

从解压后的项目根目录执行，建议 Python 3.10 及以上：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/maintenance/check_assets.py
python scripts/maintenance/summarize_saved_results.py
fashion-analysis list
python scripts/segmentation/training/run_prd31_v8_full_dataset.py --help
```

Windows 激活环境使用 `.venv\Scripts\Activate.ps1`。模型依赖与 CUDA 安装见 [GPU 指南](docs/gpu_restore_and_run.md)。`--help` 是静态帮助，不加载模型。当前注册 179 个分类命令，完整入口见 [命令目录](docs/commands.md)。

## V8 结果概览

| Core400 指标 | V5（epoch 15，V8 流程内对照） | V8（全量一轮） |
|---|---:|---:|
| TP / FP / FN | 188 / 1970 / 212 | 166 / 1001 / 234 |
| 类别正确召回（全部 GT） | 0.4700 | 0.4150 |
| 类别正确精度 | 0.0871 | 0.1422 |
| Micro F1 | 0.1470 | 0.2119 |
| 仅定位召回 | 0.6775 | 0.6325 |
| 定位成功后的类别准确率 | 0.6937 | 0.6561 |
| 类别正确匹配上的平均掩码 IoU | 0.7539 | 0.8413 |
| 掩码 IoU ≥ 0.85，通过数 / 全部 GT | 75/400 = 18.75% | 111/400 = 27.75% |
| 平均推理时间 / ms | 32.4909 | 25.8134 |
| 中位推理时间 / ms | 29.8438 | 23.7871 |
| P95 推理时间 / ms | 55.6647 | 39.0417 |

全量训练后，Core400 的 FP 减少、micro F1 和条件平均掩码 IoU 提升；类别正确召回从 47.00% 降至 41.50%。鞋、包、配饰出现明显退化，不能只报告上升的指标。Core400 已参与过开发与回归，不能当作未知盲测集。

32 张图 / 88 GT 的开发集：V5 类别正确召回 69.32%，V8 为 44.32%。V8 开发集 P95 为 78.9621 ms，因此平均时间低于 50 ms 不代表所有图片都低于 50 ms。

原始结果入口：[V8 目录](reports/reruns/recovery_v2/v8_full_dataset/)、[训练摘要](reports/reruns/recovery_v2/v8_full_dataset/training_summary.json)、[最终工作流状态](reports/reruns/recovery_v2/v8_full_dataset/workflow_status.json)。原始 JSON/TXT/CSV 不改写；本说明是基于它们整理的解读。

## 开发检查

```bash
python -m pip install -r requirements_dev.txt
python -m black --check src scripts tests fix_checkpoint_loading.py
python -m isort --check-only src scripts tests fix_checkpoint_loading.py
python -m ruff check src scripts tests fix_checkpoint_loading.py
python -m mypy
python -m unittest discover -s tests -v
```

规范原件见 编码规范原件（当前仓库未附PDF），项目原始需求见 PRD原件（当前仓库未附PDF）。历史 `.py.txt` 参考保留上传内容，不参与格式检查与命令注册。此次整理没有重新训练或测出新的 GPU 指标。


