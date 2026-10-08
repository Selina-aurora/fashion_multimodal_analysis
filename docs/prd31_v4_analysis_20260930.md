# PRD 3.1.1 v4-clean 结果分析

来源：AutoDL 实际 15 轮训练与 Core400 回归；checkpoint SHA-256 `f1f662d45ab1bdbb851dbcb9560f3318844ebd4add837f6a097cf34379284678`。

400 个目标中定位 140 个，定位且类别正确 100 个，漏检 260 个，错类 40 个。
正确类别匹配子集 mean mask IoU=0.7481；满足类别正确且 mask IoU≥0.85 的目标 34/400。

平均耗时 35.7518 ms，中位数 27.7804 ms，p95 78.7392 ms；平均耗时满足 50 ms 目标。
训练 loss 从 0.8171 降至 0.2713。训练集合固定为 547 实例 / 475 图，不能表述为全量 Fashionpedia/DeepFashion2 训练。

## 八类结果

| 类别 | 目标数 | 定位 | 类别正确 | 漏检 | 匹配正确类 mask IoU |
|---|---:|---:|---:|---:|---:|
| top | 50 | 13 | 7 | 37 | 0.7209 |
| pants | 50 | 20 | 11 | 30 | 0.7784 |
| skirt | 50 | 19 | 12 | 31 | 0.9001 |
| outerwear | 50 | 20 | 13 | 30 | 0.6642 |
| dress | 50 | 18 | 11 | 32 | 0.8357 |
| shoe | 50 | 22 | 22 | 28 | 0.7484 |
| bag | 50 | 21 | 20 | 29 | 0.6497 |
| accessory | 50 | 7 | 4 | 43 | 0.7800 |

## 目标大小

| 大小 | 目标数 | 定位召回 | 漏检 |
|---|---:|---:|---:|
| small | 136 | 17.65% | 112 |
| medium | 160 | 33.75% | 106 |
| large | 104 | 59.62% | 42 |

## 训练细类覆盖

| 数据来源 | 细类 | 训练实例 |
|---|---|---:|
| DeepFashion2 | long_sleeve_dress | 6 |
| DeepFashion2 | long_sleeve_outwear | 9 |
| DeepFashion2 | long_sleeve_top | 14 |
| DeepFashion2 | short_sleeve_dress | 8 |
| DeepFashion2 | short_sleeve_outwear | 1 |
| DeepFashion2 | short_sleeve_top | 26 |
| DeepFashion2 | shorts | 7 |
| DeepFashion2 | skirt | 13 |
| DeepFashion2 | sling | 0 |
| DeepFashion2 | sling_dress | 7 |
| DeepFashion2 | trousers | 24 |
| DeepFashion2 | vest | 2 |
| DeepFashion2 | vest_dress | 8 |
| Fashionpedia | bag, wallet | 100 |
| Fashionpedia | belt | 11 |
| Fashionpedia | cape | 2 |
| Fashionpedia | cardigan | 4 |
| Fashionpedia | coat | 14 |
| Fashionpedia | dress | 27 |
| Fashionpedia | glasses | 10 |
| Fashionpedia | glove | 10 |
| Fashionpedia | hat | 10 |
| Fashionpedia | headband, head covering, hair accessory | 11 |
| Fashionpedia | jacket | 12 |
| Fashionpedia | leg warmer | 5 |
| Fashionpedia | pants | 9 |
| Fashionpedia | scarf | 11 |
| Fashionpedia | shirt, blouse | 4 |
| Fashionpedia | shoe | 100 |
| Fashionpedia | shorts | 11 |
| Fashionpedia | skirt | 28 |
| Fashionpedia | sock | 11 |
| Fashionpedia | sweater | 3 |
| Fashionpedia | tie | 3 |
| Fashionpedia | tights, stockings | 9 |
| Fashionpedia | top, t-shirt, sweatshirt | 4 |
| Fashionpedia | vest | 4 |
| Fashionpedia | watch | 9 |

## 原始标注完整性检查

| 集合 | 源图 | 已核对 | 清单外可用原标注实例 | 状态分布 |
|---|---:|---:|---:|---|
| train | 475 | 395 | 尚未完整核对 | {"RAW_ANNOTATION_FILE_MISSING": 80, "ADDITIONAL_RAW_TARGETS": 387, "COMPLETE_WITHIN_SCOPE": 8} |
| validation | 32 | 12 | 尚未完整核对 | {"RAW_ANNOTATION_FILE_MISSING": 20, "ADDITIONAL_RAW_TARGETS": 12} |
| core_selected | 400 | 150 | 尚未完整核对 | {"RAW_ANNOTATION_FILE_MISSING": 250, "ADDITIONAL_RAW_TARGETS": 150} |

原标注缺失时，数量为尚未核对，而不是证明清单完整。可用原标注指八类范围内、非 crowd、有合法框和非空分割描述；尚未验证新增 mask 是否可解码。

### 按来源核对

| 集合 / 来源 | 源图 | 已核对 | 已选目标 | 原始可用目标 | 清单遗漏目标 |
|---|---:|---:|---:|---:|---:|
| train / DeepFashion2 | 80 | 0 | 125 | 未核对 | 未核对 |
| train / Fashionpedia | 395 | 395 | 422 | 1914 | 1492 |
| validation / DeepFashion2 | 20 | 0 | 31 | 未核对 | 未核对 |
| validation / Fashionpedia | 12 | 12 | 12 | 57 | 45 |
| core_selected / DeepFashion2 | 250 | 0 | 250 | 未核对 | 未核对 |
| core_selected / Fashionpedia | 150 | 150 | 150 | 704 | 554 |

## 判断与下一步

1. 当前瓶颈包括小目标漏检、配饰覆盖、服装细类错分，以及包/外套 mask 质量。统计相关性不能证明单一原因。
2. 原训练清单只有 547 实例 / 475 图。检查每张训练原图是否保留了八类范围内所有可用标注；若有遗漏，先建立新版本完整目标清单，再扩充薄弱细类。
3. 使用开发验证集选择候选设置；保留这次 Core400 回归及现有阈值，后续模型使用新的输出目录。Core 已被诊断使用，不声明独立盲测。
4. Core400 每图只选一个评估目标；没有命中的额外预测可能对应未列入 Core 的真实服饰。precision/FP 仅是对所选 GT 的表观统计，不能据此判定所有额外预测都是误报。
5. mean mask IoU 的分母是定位且类别正确的匹配目标，不是全部 400 个；某类子集 IoU 达到 0.85 不代表该类整体验收通过。

参考：
- https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py
- https://github.com/cvdfoundation/fashionpedia
- https://github.com/switchablenorms/DeepFashion2

## 本次已确认的问题与修复

Fashionpedia 的 395 张训练图中，387 张存在清单外的八类目标。原清单保留 422 个实例，原始标注有 1,914 个可用实例，遗漏 1,492 个。官方公开标注的 SHA-256 为 `f831f415edad12d52acde8b138fe7ebd1592b57bf0dcd6a3b63ba8cabd208223`。逐实例身份和类别核对未发现冲突。

遗漏实例按类别为 shoe 486、accessory 316、top 239、pants 148、outerwear 113、dress 108、skirt 47、bag 35。已实际解码全部 1,914 个训练目标，1,842 个多边形与 72 个 RLE 均成功。DeepFashion2 的原始 JSON 尚未上传到本地；AutoDL 运行时会继续验证该部分。

整张图进入 TorchVision Mask R-CNN 训练时，未与已给 GT 匹配的低 IoU proposal 会被标为背景。因此，清单遗漏原图上其他服饰目标可能影响模型学习。这是由官方实现推导的潜在训练机制，不证明全部漏检由单一原因造成。

Core 的小目标定位召回为 24/136=17.65%，中目标 54/160=33.75%，大目标 62/104=59.62%。配饰仅 7/50 定位、4/50 类别正确。Core 的 25 件 DeepFashion2 短裤中，8 件定位后全部被分为裙子，17 件漏检。当前训练清单含 DeepFashion2 短裤 7 件、Fashionpedia 短裤 11 件，共 18 件；DeepFashion2 吊带上衣 0 件、短袖外套 1 件。

V5 首先对现有 475 张训练图补全八类范围内所有可用原标注，同时解码全尺寸 mask，以 mask 外接框建立训练 box。训练保持 COCO DEFAULT 初始化、15 epochs、640/1024 输入设置、原 SGD/StepLR/采样设置；训练 checkpoint 仍记录原 val_v1 身份。改动包括目标补全与根据完整 mask 重建训练框，不能表述为只改一个数量参数。

另为现有 32 张开发验证图建立完整目标清单，用同样的图、GT、0.4 阈值及 10 次预热/5 次计时比较 V4 与 V5。原 Core400 的 GT 和 V4 本轮回归结果保留。此开发比较不等于正式 PRD 验收，也不恢复独立盲测。

## AutoDL 运行

将 `DIAGNOSE_PRD31_V4.py` 和 `RUN_PRD31_V5_FULL_TARGETS.py` 都放到项目的 `scripts/`。

```bash
cd /workspace/prd31_recovery_20260930/fashion_multimodal_analysis
python -m pip install --no-deps pycocotools==2.0.11
nohup python -u scripts/RUN_PRD31_V5_FULL_TARGETS.py --phase all > reports/reruns/recovery_v2/v5_console.log 2>&1 &
tail -f reports/reruns/recovery_v2/v5_console.log
```

脚本使用默认数据根 `/workspace/fashion_data`。完整原标注或图片缺失、mask 无法解码、目标身份不一致、与 Core/val 源图重叠时，会停止，不发布新的训练清单。新数据写到 `configs/prd_8class_train_v5_full_targets.csv`、`configs/prd_8class_val_v2_full_targets.csv` 和数据目录的 `processed/prd8_full_targets_v5/`。新权重写到 `outputs/prd_instance_segmentation/maskrcnn_8class_v5_full_targets/`，新报告在 `reports/reruns/recovery_v2/v5_full_targets/`。相同已生成的数据可重复使用；已有但内容不同的 V5 文件会被保护。

运行结束显示 `V5_DEVELOPMENT_COMPARISON_COMPLETED`，项目根目录生成 `prd31_v5_results.zip`。用这个包判断下一轮效果。恢复同一轮训练时再次运行原命令，未完成 checkpoint 会按完整训练状态校验后续跑，已完成 15 轮则复用。

若只做原标注库存排查：

```bash
python scripts/DIAGNOSE_PRD31_V4.py --project . --data-root /workspace/fashion_data
```

## 验证边界

已完成：上传结果的 400 目标身份匹配、2,000 条计时覆盖检查、官方 Fashionpedia 目标库存对比、1,914 个真实 mask 解码、模拟数据的整图目标补全/重复运行/原文件保留/验证图隔离/不同输出保护检查，以及两份脚本的语法和参数入口检查。

当前执行环境未安装 torch/torchvision、无用户 AutoDL CUDA。V5 的训练、真实断点续训、开发比较及最终 PRD 数值效果尚未执行；不能保证此次修复后达到 IoU 0.85。人工 GT 审核仍需完成 3.1.2 和 3.1.3。
