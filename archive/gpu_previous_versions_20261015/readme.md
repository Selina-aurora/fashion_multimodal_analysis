# 服饰多模态分析实习项目

本次交付日期：2026-10-08。整理上传过的可取得项目文件、补丁、评估结果和审核材料，进度截至 V8 全量一轮训练及回归完成。保留 3.1.1 实例分割 → 3.1.2 局部区域 → 3.1.3 属性提取的分类，旧实验与新实验分别留档。

**最新状态：V8 训练及回归完成；正式验收未宣称通过。** V8 使用 187,311 张图片、305,452 个实例，从 V5 epoch 15 微调一轮。实际范围是 DF2 原始 train 加 395 张既有 Fashionpedia 训练图片，不是两套原始训练集都全量。

**资产边界：这批上传备份没有 V5/V8 checkpoint、V8 SQLite 索引和完整开发集清单的文件字节。** 历史回执与 SHA-256 均保留；它们不能代替模型。GPU 包额外包含取得的三个 core-part detector 权重，属于局部区域模型，不能代替八类分割模型。详见 [缺失资产及恢复办法](docs/gpu_restore_and_run.md)。

## 先看哪些文件

1. [V8 全量训练报告](docs/full_dataset_training_v8_report.md)：实际范围、训练设置、全部对照结果和可解释的限制。
2. [全部结果索引](docs/results_index.md)：早期版本、V4–V8、3.1.2/3.1.3 及审核材料的位置。
3. [模块说明](docs/module_guide.md)：输入输出、坐标系统、采样、掩码和计时设计。
4. [GPU 恢复与运行](docs/gpu_restore_and_run.md)、[GitHub 上传](docs/github_upload.md)：按工作副本执行的操作步骤。
5. [编码规范落实](docs/coding_standard_compliance.md)、[验证报告](docs/validation_report.md)：具体检查范围与未做的 GPU 检查。

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

规范原件见 [编码规范](docs/standards/coding_standard.pdf)，项目原始需求见 [PRD](docs/standards/product_requirements.pdf)。历史 `.py.txt` 参考保留上传内容，不参与格式检查与命令注册。此次整理没有重新训练或测出新的 GPU 指标。
