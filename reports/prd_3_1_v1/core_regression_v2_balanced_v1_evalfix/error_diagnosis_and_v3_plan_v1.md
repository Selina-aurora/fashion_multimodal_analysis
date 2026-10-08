# ERROR_DIAGNOSIS_AND_V3_PLAN_v1

**项目：** 多模态驱动的电商服饰细粒度语义增强与智能解析系统  
**模块：** PRD 3.1.1 服饰实例分割 / Core Regression  
**日期：** 2026-09-24  
**当前模型：** Mask R-CNN 8-Class v2-balanced  
**评估集：** Frozen Core-400（8 类 × 50）  
**评估阈值：** score=0.40，bbox IoU=0.50，mask=0.50，mask reference IoU=0.85

---

## 1. 当前正式 Regression 结果

v2-balanced 在 Frozen Core-400 上的正式结果如下：

| Metric | Result |
|---|---:|
| BBox50 Recall | 0.3625 |
| Category Accuracy on Localized | 0.6621 |
| End-to-End BBox50 + Class Recall | 0.2400 |
| Mean Mask IoU on Correct-Class Localized | 0.7391 |
| Mask IoU ≥ 0.85 Rate | 0.0850 |
| Mean Inference Time | 约 24.47 ms/image |

400 个 GT 的错误构成为：

- Correct：84
- Missed：255
- Wrong Class：49
- Low Mask IoU：12

**阶段判断：当前首要瓶颈是漏检，其次是类别混淆；Mask 质量问题数量相对较少。**

---

## 2. Error Diagnosis

### 2.1 目标尺寸对漏检的影响

按目标面积划分：

| Area Bin | N | Miss Rate | Correct Rate |
|---|---:|---:|---:|
| Large ≥10% | 104 | 44.23% | 29.81% |
| Medium 2–10% | 160 | 62.50% | 20.63% |
| Small 0.5–2% | 136 | 80.15% | 14.71% |

Small object 的 miss rate 达到 **80.15%**，相比 large object 高约 **35.9 个百分点**。

按 `absolute_small_object` 统计：

| Small Object | N | Miss Rate | Correct Rate |
|---|---:|---:|---:|
| No | 264 | 55.30% | 24.24% |
| Yes | 136 | 80.15% | 14.71% |

### 结论

目标尺寸是当前 detection miss 的重要风险因素之一。

该趋势并非只由 accessory 类别造成。在多个类别内部也能观察到明显的尺寸效应：

- Dress：Large 23.5% → Medium 66.7% → Small 100% miss
- Pants：Large 29.4% → Medium 43.8% → Small 100% miss
- Skirt：Large 37.5% → Medium 64.7% → Small 94.1% miss
- Top：Large 50.0% → Medium 75.0% → Small 94.4% miss
- Bag：Large 33.3% → Medium 48.3% → Small 66.7% miss

因此，**small-object localization failure 是模型当前的系统性问题，而不是单一类别问题。**

需要注意：shoe 不呈现相同单调趋势，因此尺寸不是唯一影响因素。

---

## 2.2 可见度与场景复杂度

### Visibility

| Visibility | N | Miss Rate |
|---|---:|---:|
| Full | 88 | 51.14% |
| Partial | 312 | 67.31% |

部分可见样本的漏检率比完整可见样本高约 16 个百分点。

### Scene Type

| Scene Type | N | Miss Rate |
|---|---:|---:|
| Complex Scene | 128 | 74.22% |
| Worn Person | 192 | 59.90% |
| Partial View | 72 | 54.17% |
| Product Display | 8 | 75.00% |

Complex scene 的 miss rate 较高，说明复杂背景、多物体场景可能进一步放大检测困难。

`product_display` 仅有 8 个样本，当前不用于形成稳定结论。

---

## 2.3 遮挡因素

| Occlusion | N | Miss Rate |
|---|---:|---:|
| None | 144 | 68.75% |
| Partial | 254 | 61.02% |
| Heavy | 2 | 50.00% |

当前结果不支持直接把 occlusion 作为主要原因。

`none` 组的 miss rate 高于 `partial`，更可能受到类别、目标尺寸、场景组成等混杂因素影响，因此：

**现阶段只把 occlusion 作为辅助诊断变量，不作独立因果结论。**

Heavy occlusion 仅有 2 个样本，不具备统计代表性。

---

## 2.4 Accessory 检测问题

Accessory 整体结果：

- N = 50
- Correct = 4
- Missed = 42
- Wrong Class = 4
- Miss Rate = 84%

各 fine category：

| Fine Category | N | Missed | Miss Rate |
|---|---:|---:|---:|
| Belt | 9 | 8 | 88.9% |
| Glasses | 5 | 4 | 80.0% |
| Glove | 3 | 3 | 100% |
| Hat | 6 | 5 | 83.3% |
| Hair accessory / head covering | 8 | 6 | 75.0% |
| Scarf | 7 | 5 | 71.4% |
| Sock | 5 | 4 | 80.0% |
| Tights / stockings | 5 | 5 | 100% |
| Watch | 2 | 2 | 100% |

### 结论

Accessory 的困难不能只解释为“小目标”。

该类别同时存在：

1. 小目标比例高；
2. 类内视觉差异大；
3. 细长、贴身、边界弱目标较多；
4. 部分目标与人体/服装主体高度融合。

因此 accessory 更适合被视为：

**small-object difficulty + high intra-class variation 的组合问题。**

对于 glove、watch 等小样本 subtype，目前仅作案例提示，不单独下稳定结论。

---

## 2.5 类别混淆问题

按类别统计：

| GT Class | Correct | Missed | Wrong Class | Low Mask IoU |
|---|---:|---:|---:|---:|
| Accessory | 4 | 42 | 4 | 0 |
| Bag | 19 | 25 | 1 | 5 |
| Dress | 9 | 31 | 10 | 0 |
| Outerwear | 2 | 34 | 14 | 0 |
| Pants | 15 | 29 | 4 | 2 |
| Shoe | 19 | 24 | 2 | 5 |
| Skirt | 5 | 33 | 12 | 0 |
| Top | 11 | 37 | 2 | 0 |

Wrong-class rate 较明显的类别：

- Outerwear：28%
- Skirt：24%
- Dress：20%

主要 confusion group：

- `outerwear ↔ top / dress`
- `skirt ↔ pants`
- `dress ↔ top`

典型可视化中还观察到：

- Layered clothing 导致 outerwear 与内部 top 混淆；
- 服饰整体轮廓不完整时，dress 容易被识别为 top；
- 坐姿、遮挡、窄版版型会增加 skirt / pants 混淆；
- 部分高置信度 wrong-class 表明问题不是简单提高 score threshold 即可解决。

### 结论

Outerwear、Skirt、Dress 存在独立于 detection miss 的 **category-boundary learning problem**。

---

## 2.6 Mask 质量问题

Low Mask IoU 一共只有 12 个：

- Bag：5
- Shoe：5
- Pants：2

部分案例 bbox IoU 已较高但 mask IoU 仍较低，说明这些案例存在独立的 mask boundary / foreground segmentation 问题。

但从错误数量看：

**Mask refinement 暂时不是 v3 第一优先级。**

---

# 3. v3 优化优先级

## Priority 1 — Small-Object Detection Enhancement

### 目标

优先降低 small / medium object 的 missed detection，重点关注：

- Top
- Pants
- Skirt
- Dress
- Accessory

### 候选实验方向

1. **提高训练/推理输入分辨率**
   - 对比当前设置与更高 `min_size / max_size`
   - 重点观察 small-object recall 和推理成本变化

2. **Multi-scale Training**
   - 增强不同尺寸服饰的尺度鲁棒性

3. **RPN / Anchor 针对小目标优化**
   - 检查 anchor scales 与当前小目标尺寸是否匹配
   - 增加更小尺度 anchor 的对照实验

4. **Small-object / Hard-example Sampling**
   - 训练阶段增加 small-object 样本出现频率
   - 避免只增加总样本量而不改变有效困难样本比例

5. **Crop / Enlargement Strategy**
   - 针对局部小目标进行局部裁剪放大
   - 作为独立 ablation，不直接替换 baseline

### 核心评估指标

除了 overall 指标之外，v3 必须单独比较：

- Small-object miss rate
- Medium-object miss rate
- Per-class small-object recall
- Overall BBox50 Recall
- End-to-End BBox50 + Class Recall
- Inference latency

---

## Priority 2 — Category Confusion Improvement

### 目标类别

重点处理：

- Outerwear vs Top / Dress
- Skirt vs Pants
- Dress vs Top

### 候选实验方向

1. Hard confusion sample sampling
2. 对混淆类别进行 targeted rebalancing
3. 增加相似类别边界样本
4. 针对 layered clothing 加入困难样本
5. 检查训练标签映射与 coarse class 定义的一致性
6. 必要时再评估额外 classification constraint / coarse-to-fine 方案

### 核心评估指标

- Per-class wrong-class rate
- Outerwear class-correct recall
- Skirt class-correct recall
- Dress class-correct recall
- Confusion pair count

---

## Priority 3 — Mask Refinement

暂不作为 v3 首轮主要目标。

后续主要针对：

- Bag
- Shoe

检查：

- bbox 已正确定位但 mask boundary 明显偏差的案例
- mask head 分辨率
- 边界细节
- 小区域/细长区域 mask 稳定性

---

# 4. v3 实验原则

为保证 regression 可解释，后续实验遵循以下原则：

1. **Frozen Core-400 不再修改**
2. 所有训练改动在 train / validation 阶段完成
3. 不根据 Core-400 单个案例逐图调参
4. 每次只改变一组主要因素，保留可解释的 ablation
5. 每个 v3 candidate 与 v2-balanced 使用相同 Core、相同 evaluator、相同阈值比较
6. 除 overall metrics 外，必须保留：
   - class-wise metrics
   - area-bin metrics
   - error cases
   - inference speed
7. 优化不能只看 overall recall，需要确认是否真正改善 small-object / confusion target，而不是通过大量 FP 换 recall

---

# 5. 推荐 v3 实验顺序

建议不要一次同时修改多个模块。

### V3-A：Small-object baseline enhancement

先只针对 detection：

- Input resolution / multi-scale / RPN-small-anchor 中选择一个最基础、最可控方案
- 训练后重新跑 Frozen Core-400

如果 small-object miss 有明确下降，再进入下一阶段。

### V3-B：Small-object + hard sampling

在 V3-A 有效的基础上加入 small/hard-example sampling。

### V3-C：Category confusion targeted improvement

再针对：

- outerwear / top / dress
- skirt / pants

做 targeted sampling 或类别边界优化。

### V3-D：Mask refinement

仅在 detection / classification 的主要瓶颈缓解后再处理 bag / shoe mask。

---

# 6. v3 成功判定

v3 不以“某一个 overall 数字上涨”作为唯一成功标准。

建议至少满足：

1. Overall BBox50 Recall 不下降；
2. End-to-End BBox50 + Class Recall 提升；
3. Small-object miss rate 明显下降；
4. Outerwear / Skirt / Dress wrong-class 不恶化；
5. FP 不出现不可接受增长；
6. Mean inference time 仍满足当前 PRD 速度要求；
7. Core-400 上的改善应能对应到预先定义的优化目标。

---

# 7. 当前阶段结论

v2-balanced 的主要瓶颈已经从“模型效果不好”进一步定位为两个具体问题：

**A. Detection 层面：**
small object、partial visibility、complex scene 下的漏检明显。

**B. Classification 层面：**
outerwear–top–dress 和 skirt–pants 存在稳定混淆。

Mask 问题存在，但当前数量较少。

因此下一版 v3 不建议直接做大范围结构修改，而应优先采用：

> **Small-object detection enhancement → targeted confusion improvement → mask refinement**

的顺序进行可控实验。

该文件作为 v2-balanced Core Regression 的 error diagnosis 与 v3 实验设计依据。
