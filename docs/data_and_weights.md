# 数据、权重与文件范围

## 本包包含什么

保留上传的源码、configs、冻结 benchmark、最终 reports、非权重 outputs，以及少量 data 测试资源。原始证据文件的 SHA-256 见 [original_evidence_sha256.csv](original_evidence_sha256.csv)。

大型训练权重、原始数据集、旧 GPU 重复工作包、Python 缓存和本地虚拟环境不进入 GitHub 工作树。完整证据版仓库仍含大量报告图片；本包用于源码与证据交付，不是含全部训练数据/模型的离线运行镜像。

## 外部数据布局

| 相对于项目的位置 | 内容 |
|---|---|
| `../fashion_data/raw/train/train/image/` | DeepFashion2 图片 |
| `../fashion_data/raw/train/train/annos/` | DeepFashion2 实例标注 |
| `../fashion_data/raw/fashionpedia/images/test/` | manifest 引用的 Fashionpedia 图片 |
| `../fashion_data/raw/fashionpedia/annotations/` | Fashionpedia 实例/属性标注 |
| `../fashion_data/processed/` | 各版 manifest 使用的外部裁剪与掩码 |
| `outputs/garment_instances/` 等 | 本包保留的实例裁剪/掩码，按原路径存放 |

`configs/*.csv` 的 `source_image`、`mask_path`、`crop_path` 等字段是实际输入位置；不同实验可能使用不同的 processed 子目录。以 manifest 为准，不要只复制一个 train 文件夹就假定资产齐全。`check_project --assets` 检查 B1 train、固定 val 和 Core 的图片/掩码，不检查所有历史实验的可选输入。

## 权重放置

完整清单：[models/weight_inventory.csv](../models/weight_inventory.csv)。清单基于上传 ZIP 的目录信息，含路径、大小与 ZIP CRC32；CRC32 是传输完整性信息，不是内容 SHA-256 或模型效果校验。本次不重新加载这些大权重。

| 用途 | 恢复位置 |
|---|---|
| 原记录选定的 B1 epoch 15 | `outputs/prd_instance_segmentation/maskrcnn_8class_v3_b1_dataexp/checkpoint_last.pth` |
| V2-balanced 对照 epoch 15 | `outputs/prd_instance_segmentation/maskrcnn_8class_baseline_v2/checkpoint_last.pth` |
| 其他 3.1.1 训练版本 | `outputs/prd_instance_segmentation/<实验名>/checkpoint_*.pth` |
| 核心局部检测器 | `models/core_part_detector_v1/{best_loss_model,best_model,last_model}.pt` |

最新 outputs 共 9 个 Mask R-CNN 实验目录、36 个 checkpoint，权重合计 12,648,849,284 字节；非权重图片 1,105 个、40,602,000 字节已保留。另记录核心局部检测器的 3 个权重和 1 个旧备份，共 40 条清单记录。权重从你保留的原上传包取回后按相对路径恢复；本仓库不依赖会过期的临时分享链接。

`maskrcnn_8class_final_v1` 目录虽然包含权重，但本次没有将其名称当成最终选型证据。选型依据是 `reports/prd_3_1_v1/final_3_1_1_v1/selected_model_evaluation_summary.txt`，其中 checkpoint 明确指向 B1。

## 模型与数据授权记录

本次未给原项目擅自新增开源许可证；原资料未提供统一 LICENSE。上游模型、外部数据与代码应各自保留其来源和实际授权信息。此整理不改变这些资料的归属。
