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
