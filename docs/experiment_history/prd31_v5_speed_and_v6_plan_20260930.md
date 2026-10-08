# PRD 3.1.1：V5 提速验证与 V6 数据覆盖扩充

日期：2026-09-30。证据：用户上传的 `prd31_v5_speed_results.zip`（168 个文件，ZIP CRC 校验通过），以及已保留的 V4 Core400 报告和数据清单。

## 已确认的结果

V5 的先筛选、再传输掩码实现已在 AutoDL / RTX 4090 上通过一致性验证。400 张固定 Core 图的平均耗时从 59.1453 ms 降到 32.7758 ms，约减少 44.6%。平均耗时达到 PRD 的 ≤50 ms 参考目标。

两个版本的 400 条逐目标结果除可视化文件路径外完全一致，protocol_metrics.json 全部内容也完全一致。每个版本均有 400 张图 × 5 次 = 2000 条计时记录；逐图 5 次覆盖完整，重新计算的均值、中位数及 P95 与摘要相符。

GPU 上还针对相同模型输出，逐项比较了两个后处理器在 32 张开发图上的 boxes、labels、scores 和二值 masks，均完全相等。该开发集模型输出 420 个候选，冻结阈值保留 138 个。空输出、无保留目标、NaN/inf 分数及阈值边界检查也已通过。模型权重、图像缩放、0.40 分数阈值、0.50 mask 阈值和 0.50 bbox 匹配阈值均未改变。

运行状态：`ACCURACY_EQUAL_FASTER_RUNTIME_INSTALLED`。AutoDL 已保留提速实现。此处的速度对比引用原评测与新评测两次独立运行，不是随机交替计时实验；它证明当前模块实测的平均耗时达标，并提供输出一致性证据。

## Core400 数值

| 指标 | V4 原实现 | V5 原实现 | V5 提速实现 |
|---|---:|---:|---:|
| 定位召回率 | 35.00%（140/400） | 67.75%（271/400） | 67.75%（271/400） |
| 定位且类别正确 | 25.00%（100/400） | 47.00%（188/400） | 47.00%（188/400） |
| 类别正确且 mask IoU ≥0.50 | 91/400 | 168/400 | 168/400 |
| 类别正确匹配实例的平均 mask IoU | 0.7481 | 0.7539 | 0.7539 |
| 类别正确且 mask IoU ≥0.85，占全部 GT | 34/400（8.50%） | 75/400（18.75%） | 75/400（18.75%） |
| 漏检 | 260 | 129 | 129 |
| 定位但分类错误 | 40 | 83 | 83 |
| 类别正确但 mask IoU <0.50 | 9 | 20 | 20 |
| 置信度 ≥0.40 的输出 | 685 | 2158 | 2158 |
| 平均耗时 | 35.7518 ms | 59.1453 ms | 32.7758 ms |
| 中位耗时 | 27.7804 ms | 45.3369 ms | 30.4329 ms |
| P95 耗时 | 78.7392 ms | 118.3744 ms | 50.5716 ms |

平均 mask IoU 仍低于 0.85，PRD 整体验收尚未达标。400 张图已经有历史检查与调优记录，结果属于固定回归证据，不能宣称为未接触的盲测。上表 V4/V5 最后一列使用不同后处理实现，因此不能由速度差单独推断两模型架构的效率差异。

Core 每图只保留一个指定 GT。2158 个输出中未匹配的其他目标，可能包含同图内真实但未纳入清单的服装；不能把协议统计的 1970 个未正确匹配输出逐个认定为真实误检。总体 bbox50_precision 12.56% 是定位匹配数/预测数；类别感知 precision 是 188/2158=8.71%，两者口径不同，且都受选择性 GT 的范围影响。

在 V4/V5 都定位且类别正确的相同 84 个目标上，平均 mask IoU 从 0.7878 增加到 0.7991。总体均值变化还受到新增匹配目标集合的影响。

## 分类、分割与训练覆盖诊断

| 类别 | V5 定位/50 | 定位且类别正确/50 | 类别正确实例平均 mask IoU | 漏检 | 分类错误 |
|---|---:|---:|---:|---:|---:|
| top | 38 | 28 | 0.6995 | 12 | 10 |
| pants | 32 | 22 | 0.7843 | 18 | 10 |
| skirt | 36 | 23 | 0.8465 | 14 | 13 |
| outerwear | 40 | 20 | 0.6252 | 10 | 20 |
| dress | 35 | 21 | 0.8621 | 15 | 14 |
| shoe | 37 | 35 | 0.7573 | 13 | 2 |
| bag | 32 | 25 | 0.7398 | 18 | 7 |
| accessory | 21 | 14 | 0.7016 | 29 | 7 |

外套有 20 个已定位目标类别错误，其中 12 个被预测为 top、6 个为 dress。大目标已定位 92/104，但仅 50/104 定位且类别正确，分类仍是明显瓶颈。小目标已定位 66/136，仍漏检 70/136。不能用扩大输入尺寸单独解决这两类问题。

Fashionpedia Core 的 150 个现有 GT 掩码已逐个与官方原始标注重新解码并比较，所有像素完全一致（IoU=1.0）。此检查只覆盖这 150 个 Fashionpedia 目标；未在本地核验缺少原始 JSON 的 250 个 DeepFashion2 GT。没有改动 Core GT。

V5 的训练目标为 2039 个，其中 Fashionpedia 1914 个、DeepFashion2 125 个；DF2 的 80 张训练图与 V4 相同，原来的细类别覆盖缺口仍在。当前 DF2 独立训练图数如下：

| DF2 原始标签 | 已有训练图 |
|---|---:|
| short_sleeve_top | 26 |
| long_sleeve_top | 14 |
| short_sleeve_outwear | 1 |
| long_sleeve_outwear | 9 |
| vest | 2 |
| sling | 0 |
| shorts | 7 |
| trousers | 24 |
| skirt | 13 |
| short_sleeve_dress | 7 |
| long_sleeve_dress | 5 |
| vest_dress | 8 |
| sling_dress | 7 |

这些缺口给出了扩充训练数据的具体理由；它们不是对错误唯一原因的证明。

## V6 的改动和使用方法

`RUN_PRD31_V6_COVERAGE.py` 保留 V5 的完整训练目标，从已下载的 DF2 原始训练集按固定随机种子补充图片，使 13 个原始标签各至少出现在 40 张训练图中。每张新增图的全部可用目标均被解码并保留，训练框由完整 mask 生成。

新增图排除 Core、开发验证以及脚本列出的现有 benchmark manifest/review 图片。V5 的原始训练图保留；旧图与其他 benchmark 的继承重叠数量会在 data_preparation.json 中披露，不能据此宣称所有历史评测都无泄漏。隔离按源图路径进行，没有进行图像内容去重。

数据变化是 V6 的主要实验变量。仍从 COCO DEFAULT 开始训练 15 轮，保留现有 SGD、缩放、采样及增强参数。V6 使用补全后的开发清单作为 checkpoint 的验证清单，训练数据检查显示 88 个开发目标；训练程序不做逐轮验证选择。V5/V6 在同一份补全开发集、同一提速 runtime、同一冻结阈值下重新评测，再决定候选。脚本此阶段不运行 Core，也不替换验收主流程模型。

本地已使用真实生成的多目标图像和掩码夹具验证：13 类覆盖、每图多个目标保留、开发/Core/grounding 图片排除、原清单保留、重复运行内容一致、不足覆盖时拒绝发布、已有 Core 重叠时拒绝执行。语法和入口检查通过。本地没有 PyTorch/CUDA；实际训练、断点恢复与 GPU 比较需由 AutoDL 执行，尚未声明 V6 提升。

下载 V6 脚本，放到 AutoDL 项目的 `scripts` 文件夹。之前的 `RUN_PRD31_V5_FULL_TARGETS.py` 和 `DIAGNOSE_PRD31_V4.py` 也需在该目录（或项目根目录）。在 `(py312)` 终端执行：

```bash
cd /workspace/prd31_recovery_20260930/fashion_multimodal_analysis
nohup python -u scripts/RUN_PRD31_V6_COVERAGE.py --phase all > reports/reruns/recovery_v2/v6_console.log 2>&1 &
tail -f reports/reruns/recovery_v2/v6_console.log
```

开始会输出 SCAN、SELECTED 和 PREPARED，随后进入 15 轮训练及两个模型的开发集评测。Ctrl+C 只会退出上面的日志查看，后台任务会继续。

完成后，发送项目根目录生成的 `prd31_v6_results.zip`。如果同名包已存在，新包会加时间后缀，日志中的 `RESULTS_ZIP=` 指向本次文件。任务中断后重新执行同一命令，会校验数据和参数后续训或复用已完成 checkpoint。

可选的只准备数据模式：`python -u scripts/RUN_PRD31_V6_COVERAGE.py --phase prepare`。默认最多扫描 50000 个原始标注；若覆盖仍不足，程序会明确列出缺口，并保持训练清单不发布。使用相同配置扩大 --max-scan 后可继续准备；默认 40 张目标与训练 seed 是固定记录，不用 Core 调阈值。

新增路径：

```text
configs/prd_8class_train_v6_coverage.csv
fashion_data/processed/prd8_full_targets_v6/train/
outputs/prd_instance_segmentation/maskrcnn_8class_v6_coverage/checkpoint_last.pth
reports/reruns/recovery_v2/v6_coverage/
```

关键版本证据：

```text
V5 checkpoint SHA256: ea5a4ddf0f002256f1ce5164fc1b4b5b4d2263698f9525c9fbe7d8cf98bc1a65
Core manifest SHA256: c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89
Complete development manifest SHA256: 39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93
Verified runtime SHA256: 60021df022b16fbe42c88ed57635e180cfd397077803403cbb6513169daabaca
```
