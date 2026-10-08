# PRD 3.1 当前实验总览

本版根据最终上传的实验记录核对，日期为 2026-09-30。整理代码与证据，不新增训练/推理成绩。3.1.1 原记录选型为 V3-B1 Data Expansion，V2-balanced 为参考；整套 PRD 尚无足够证据声明全部验收通过。

修复版已完成数据排除、新训练入口、冻结协议实现、完整预测流水线和人工审核入口，30 项 CPU 检查通过。新实验按 [RECOVERY_GUIDE](recovery_guide.md) 执行；以下数值均为保留的历史结果。

## 3.1.1：服装实例分割

目标类别为 top、pants、skirt、outerwear、dress、shoe、bag、accessory。B1 使用 Mask R-CNN ResNet50-FPN、COCO 初始化，训练清单 `prd_8class_train_v3.csv` 共 580 个实例、508 张图，训练 15 epochs；固定验证集 43 个实例、32 张图。

| B1 训练设置 | 记录值 |
|---|---|
| 图像组加权采样 | replacement=True，alpha=0.5 |
| 学习率 / momentum / weight decay | 0.0025 / 0.9 / 0.0005 |
| LR step / gamma | 5 / 0.1 |
| min_size / max_size | 640 / 1024 |
| 水平翻转 / seed | 开启 / 20260921 |
| final_train_loss | 0.264722 |

冻结 Core400 每类 50 例；DeepFashion2 250、Fashionpedia 150。历史 B1 Core 结果如下：

| 指标 | 值 | 分母/含义 |
|---|---:|---|
| bbox50 recall | 0.3600 | 144/400，忽略预测类别的定位匹配 |
| bbox50 precision | 0.2124 | 144/678，忽略类别的定位匹配 |
| mean bbox IoU localized | 0.8084 | 已定位子集 |
| category accuracy localized | 0.6944 | 100/144 |
| bbox50 + class recall | 0.2500 | 100/400 |
| mean mask IoU localized | 0.7221 | 已定位子集，包含类别错误 |
| mean mask IoU correct class | 0.7677 | 已定位且类别正确子集 |
| mask IoU >=0.50 rate | 0.2350 | 94/400，要求定位及类别正确 |
| mask IoU >=0.85 rate | 0.0975 | 39/400，要求定位及类别正确 |
| 平均 / P95 模型计时 | 24.7143 / 72.3295 ms | RTX 4090，非完整端到端延迟 |

错误分解为 correct 94、low_mask_iou 6、wrong_class 44、missed 256。平均耗时低于 50 ms 不代表每例均达标，也不能替代 benchmark 协议要求的完整流程计时。

**数据独立性问题：**B1 的 train_v3 与 Core 有 33 张完整源图路径重叠。此处指标仅如实保留历史记录，不能用作独立盲测泛化成绩。原 v1/v2 清单与 Core 的路径交集为 0，尚未进行内容级去重。见 [CORRECTIONS](corrections.md)。

## 3.1.2：语言引导局部区域定位

已经完成 Grounding DINO 提示词、ROI、空间/高分辨率、SAM 等诊断及人工核查，但粗框与小部件漏检仍明显。两组 40 例数据必须区分：

| 审查组 | 样本构成 | 结论 |
|---|---|---|
| 四类定位错误审查 | sleeve/collar/button/zipper 各 10 | coarse 6、wrong 34；错误类型包含 target_absent 15 |
| PRD 八区域 baseline | collar/cuff/hem/pocket/shoulder/waist/pattern/decoration 各 5 | correct 1、coarse 19、wrong 0、missed 20；严格正确率 2.5% |

目标不可见筛除只是利用人工标签的 oracle 分析。后续定向诊断条件不一致，不能与 baseline 拼成统一最终验收分数。详见 [定位分析](prd_3_1_2_attribute_grounding_error_analysis.md)。

## 3.1.3：属性提取与预测 ROI 传播

颜色、图案、袖长、领口、轮廓和风格已有实现及诊断。传播评估比较同一属性算法在 GT-ROI 与预测 ROI 上输出的标签，测的是一致性，不是人工真值准确率。

另外有可复用人工审核：颜色 40 例中 37 例明确单主色、32 正确（86.49%）；图案 40 例中 34 正确（85%）。已作为历史开发证据接入 v2，不能把曾经审核或调参使用的这些样本重新称为新盲测。新候选覆盖八类、六属性，适用行 160，人工标签保持待填。

| 实验 | ROI 模型 | 配对规模 | 范围 |
|---|---|---|---|
| predicted_roi_attributes_v2 | B1 epoch 15 | 12 对；袖长/领口可比 7 对 | top/pants/skirt/outerwear/dress，未覆盖后三类 |
| predicted_roi_attributes_fullval_v1 | V2-balanced epoch 15 | 22 对；常规属性可比 18 对、袖长/领口 13 对 | 缺 skirt；各属性有效样本数不同 |

fullval_v1 不能写成 B1 实验，也不能把其一致率当作全八类属性准确率。详见 [属性传播分析](prd_3_1_3_attribute_propagation_analysis.md)。

## 下一阶段需要补足的实验

新 train_v4_clean 已排除已知交集，547 实例 / 475 图、八类保留。匹配顺序、计时口径和全部 GT 分母已经修复；分割 ROI/mask→局部 ROI/mask→属性输入已接通。待实际完成：在 AutoDL 干净重训/复测，审核新的局部与属性 GT，并审核模型预测是否严格语义正确。脚本随后生成 PRD PASS/FAIL、Track A/B 和端到端失败阶段报告，不能在得到真实结果前宣布达标。

原始结果路径和版本映射见 [RESULTS_INDEX](results_index.md)。
