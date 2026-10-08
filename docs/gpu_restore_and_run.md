# GPU 上传、缺失资产与运行

## 解压到新工作副本

GPU ZIP 包含与 GitHub ZIP 相同的代码、配置、全部可取得结果及文档，另包含三个可取得的 core-part detector 权重。它不包含完整原始 DF2/Fashionpedia 数据，也不包含未上传的 V5/V8 权重。

在 GPU 机器新建目录，避免覆盖已在机器上的模型和历史结果：

```bash
mkdir fashion_delivery_20261008
cd fashion_delivery_20261008
unzip ../fashion_multimodal_analysis_gpu_20261008.zip
cd fashion_multimodal_analysis
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export FASHION_DATA_ROOT=../fashion_data
python scripts/maintenance/check_assets.py --hash-weights
```

`../fashion_data` 是示例相对布局。实际外部数据在别处时，将环境变量指向其真实目录。可使用 `--data-root` 显式传入；不需要修改源码中的服务器绝对路径。

安装与现有 CUDA/驱动匹配的 PyTorch、torchvision，再安装需要的其他模型依赖：

```bash
python -m pip install -e '.[vision]'
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果 GPU 镜像已提供成套 PyTorch/torchvision，先核对其匹配情况；`.[vision]` 只声明项目模型依赖，并不能替代 GPU 镜像自身的 CUDA 配置。原项目环境资料保留在 `reports/environment/` 和 `docs/setup.md`。

## 当前确实缺哪些资产

| 资产 | 状态及作用 |
|---|---|
| V5 `checkpoint_last.pth` | 缺字节；V8 从 V5 epoch 15 初始化与对照都需要它 |
| V8 `checkpoint_last.pth` | 缺字节；评估已完成全量模型或继续该训练需要它 |
| V8 `dataset.sqlite` | 缺字节；历史摘要记录了身份，不能靠摘要还原索引 |
| 完整开发集 `prd_8class_val_v2_full_targets.csv` | 缺原文件；32 图片 / 88 GT 的结果表已保留，但不含重建全部 GT 所需信息 |
| V5 派生训练/开发掩码 | 训练清单引用的是外部 fashion_data 派生文件，不随本 ZIP 提供 |
| 原始 DF2/Fashionpedia 图片、标注 | 不随本 ZIP 提供；沿用已有 GPU 数据或从合法原数据重新准备 |
| Core400 清单及项目内掩码 | 已保留，原清单 SHA 不变，重命名由别名处理 |
| V5 训练清单 | 从 V6 清单前 2,039 行恢复，序列化后的 SHA 与原回执完全一致 |
| 三个 core-part detector `.pt` | GPU ZIP 已包含，属于 3.1.2，不能当作 V5/V8 分割权重 |

较早单独上传过一份 2026-09-23 `checkpoint_last.pth`，本次读取其字节返回 HTTP 403，未将它伪装成已备份权重；也不能据文件名认定它是 V5 或 V8。

检查命令默认只读，报告 MISSING/PRESENT_UNHASHED/HASH_MATCH/HASH_MISMATCH。`--hash-weights` 才读取完整大模型摘要。`--require-ready` 在所列资产不齐备时返回 2。ready 只覆盖所列身份文件和目录存在性，训练准备还会检查实际图片、标注与派生掩码。

## 用已有权重复核 Core400

只有取得并校验 V8 权重后，才运行下面的回归。它生成独立目录，不覆盖 2026-09-30 的回执：

```bash
python scripts/segmentation/evaluation/eval_prd_8class_maskrcnn_v3_b1_dataexp.py \
  --checkpoint outputs/prd_instance_segmentation/maskrcnn_8class_v8_full_dataset/checkpoint_last.pth \
  --val-csv benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv \
  --report-dir reports/reruns/recovery_20261008/core_v8 \
  --device cuda --score-threshold 0.4 --match-bbox-iou 0.5 \
  --mask-threshold 0.5 --warmup-runs 10 --timed-passes 5
```

这里复用分类入口的评估实现，通过 checkpoint 参数选择 V8，并不是重新用旧模型训练。计时期间保持 GPU 无其他重任务；新环境速度须看新报告，不能直接沿用旧 4090 时间。

## 缺少清单/掩码时重新准备

已有 V4 基础清单、原始标注和图片齐备后，可在新的工作副本执行 V5 准备阶段：

```bash
python scripts/segmentation/training/run_prd31_v5_full_targets.py \
  --project . --data-root ../fashion_data --phase prepare
python scripts/maintenance/check_assets.py --hash-weights
```

准备会生成派生清单和掩码，应核对它们的统计和摘要。若与历史摘要不同，它就是新准备的数据身份；保留两份结果，不能在现有回执里替换哈希。逐 GT 评估表不能用来补造缺失 GT bbox 或掩码。

## 在新目录启动全量实验

取得 V5 初始权重、原始数据和完整开发集后，使用新 run-name。下面是与历史范围相同的“既有 Fashionpedia 训练图片”模式：

```bash
python scripts/segmentation/training/run_prd31_v8_full_dataset.py \
  --phase all --project . --data-root ../fashion_data \
  --fp-mode existing-train --run-name v8_full_dataset_recovery_20261008 \
  --epochs 1 --batch-size 1 --workers 2 --prepare-workers 4 \
  --lr 0.0001 --seed 20260930 --allow-bad-images
```

`--allow-bad-images` 显式记录并排除无效整图，对应历史曾排除 1 张图的处理；默认不使用时会报错停止。先阅读本次生成的 `data_preparation.json`，确认实际入选范围，再解释训练结果。

两套原始 train 都取得时，可以显式使用 `--fp-mode native-train`；那是不同数据范围，应使用另一个新 run-name，不能仍标成旧 V8 的同次训练。

相同新 run-name 的断点恢复会核对索引、初始权重、训练设置和随机状态；`--epochs` 是总轮数。整理后的运行时代码增加了注释和类型提示，文件摘要与历史运行时代码不同，旧回执保留原摘要；新运行应在新目录记录本次摘要，不跳过身份检查。

运行前读 `--help` 和 [V8 报告](full_dataset_training_v8_report.md)。缺少模型时，上传此 ZIP 仍可审阅代码及全部已保存结果，但不能直接恢复训练。
