# 安装与运行

## 1. 基础环境

项目元数据要求 Python >=3.10。原 GPU 记录使用 Python 3.12.11；建议复现时先使用原服务器环境。没有完整的原始 pip freeze，因此本包不提供声称已复现验证的依赖锁文件。

在项目根目录建立虚拟环境并安装基础包：

```bash
python -m venv .venv
# Linux/macOS：
source .venv/bin/activate
# Windows PowerShell 使用：.venv\Scripts\Activate.ps1
python -m pip install -e .
python scripts/maintenance/check_project.py
python -m unittest discover -s tests -v
```

基础依赖为 numpy、Pillow、pandas、tqdm。模型依赖按模块安装：Mask R-CNN 使用 torch/torchvision；Grounding DINO、CLIP、Mask2Former、SAM 相关实现还使用 transformers；部分数据/几何工具使用 opencv-python、pycocotools。全套依赖入口为：

```bash
python -m pip install -e ".[vision]"
```

GPU 环境请先用 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 按操作系统和计算平台安装匹配的 torch/torchvision，再安装其他依赖。这里没有硬编码 CUDA 安装索引，也没有验证任意最新依赖组合。模型首次加载可能需要下载上游预训练权重；离线环境需提前准备相应缓存。

原上传环境记录（仅转述，非本次验证）：Python 3.12.11、PyTorch 2.13.0+cu132、Torchvision 0.28.0+cu132、RTX 4090。来源：[gpu_server_config_2026_09_21.txt](../reports/environment/gpu_server_config_2026_09_21.txt)。

## 2. 数据与 checkpoint

按 [DATA_AND_WEIGHTS](data_and_weights.md) 恢复外部数据和 checkpoint。训练集中的路径既包括仓库内的 outputs，也包括同级 fashion_data 中的 raw/processed 文件。仓库附带的裁剪和报告不能替代原始训练图片。

```bash
python scripts/maintenance/check_project.py --assets --checkpoint outputs/prd_instance_segmentation/maskrcnn_8class_v3_b1_dataexp/checkpoint_last.pth
```

这个检查在未准备外部数据/权重时会返回 1，并打印缺失数量；这是资产缺失，不是模型运行失败。可选环境变量示例：

```bash
export FASHION_DATA_ROOT=/absolute/path/to/fashion_data
```

## 3. B1 历史训练的运行入口

以下用于复现已有 train_v3 方案。该清单有 33 张 Core 源图重叠，不能直接用于产生新的独立验收结论。后续正式实验应先建立排除全部冻结测试源图的新训练清单，并保留新版本编号。

```bash
python scripts/segmentation/training/train_prd_8class_maskrcnn_v3_b1_dataexp.py --dry-run
python scripts/segmentation/training/train_prd_8class_maskrcnn_v3_b1_dataexp.py --epochs 15 --device cuda
```

`--dry-run` 仍会导入 torch/torchvision 并检查真实图像、掩码与采样器。训练程序沿用对应版本的固定输出目录；需要保留历史文件时，应在另一个工作副本中运行。此包没有自动重新训练或覆盖上传结果。

## 4. 明确指定 Core400 与阈值

评估版本的默认 checkpoint 已与名称对应。通用 `eval_prd_8class_*` 入口仍允许 val43/Core400 两种清单，不能只靠脚本名判断评估集；使用以下显式参数：

```bash
python scripts/segmentation/evaluation/eval_prd_8class_maskrcnn_v3_b1_dataexp.py \
  --checkpoint outputs/prd_instance_segmentation/maskrcnn_8class_v3_b1_dataexp/checkpoint_last.pth \
  --val-csv benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv \
  --score-threshold 0.40 --mask-threshold 0.50 --match-bbox-iou 0.50 \
  --device cuda --report-dir reports/reruns/core_b1_reproduction
```

V2 对照使用 `eval_prd_8class_maskrcnn_baseline_v2.py`、`maskrcnn_8class_baseline_v2/checkpoint_last.pth`，其余 Core 参数相同，另写入 `reports/reruns/core_v2_reproduction`。较早的 `eval_prd31_core_regression_v2_balanced_v1.py` 保留历史分组统计实现；新比较应使用上述支持修正后统计口径的 baseline_v2 入口。

报告中的计时来自模型调用区间，包含模型内部处理与 CUDA 同步，不包含外部读图、CPU 预处理/后处理、保存和可视化。不要将这一数字与完整端到端验收延迟直接等同；冻结协议里的更完整计时应另外执行并记录。

## 5. 定位与属性分析

先查 [命令目录](commands.md)，再查看所需命令的 `--help`。不带 argparse 的旧交互/诊断脚本在帮助中明确标识，输入路径在实现常量中。需要人工审查的脚本应在有图形界面的本地环境使用。

B1 对应 `build_prd31_predicted_roi_pilot_v2` / `build_prd31_matched_roi_manifest_v2` / `run_prd31_predicted_roi_attributes_v2`。`run_prd31_fullval_expansion_stage1` 及现有 fullval_v1 结果使用的是 V2-balanced；不要将该分支改称 B1 实验。

## 6. 不运行 GPU 的审计

```bash
python scripts/analysis/audit_dataset_overlap.py
python scripts/analysis/analyze_visibility_filtering.py
```

新审计写入 `reports/repository_audit/`。添加 `--fail-on-overlap` 可让源图重叠审计在发现问题时返回 2，便于后续训练前设立检查门槛。Visibility filtering 是按人工 `target_absent` 标签进行的 oracle 分析，不是已经训练完成的自动可见性模型。
