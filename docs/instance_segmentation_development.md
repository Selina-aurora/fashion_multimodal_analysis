# 服饰实例分割模块开发记录

## 1. 文档说明

本文档用于记录“多模态驱动的电商服饰细粒度语义增强与智能解析系统”中服饰实例分割模块的开发过程，包括数据分析、模型选型、Baseline 实验、ROI 方案、训练数据构造、问题记录与当前技术决策。

本文档不是每日工作日志，重点记录：
- 为什么采用当前方案；
- 实验如何设计；
- 实验结果说明了什么；
- 遇到了哪些问题；
- 当前做出了哪些技术决策；
- 后续还需要验证什么。

## 2. 模块目标

### 2.1 功能目标

当前开发对应 PRD 3.1.1“服饰实例分割功能”。

输入：
- RGB 格式服饰/商品图片。

输出：
- 服饰实例分割掩码（mask）；
- 边界框（bounding box）；
- 类别标签（category）。

PRD 目标类别包括：
- 上衣；
- 裤子；
- 裙子；
- 外套；
- 连衣裙；
- 鞋子；
- 包包；
- 配饰。

性能目标：
- 单张图片分割时间 ≤ 50 ms；
- 分割 IoU ≥ 0.85。

### 2.2 当前范围说明

当前阶段优先使用 DeepFashion2 完成服饰实例分割主流程验证。

已确认：
- DeepFashion2 可覆盖上衣、裤子、裙子、外套、连衣裙等主要服饰类别；
- 鞋子、包包、配饰后续需要补充自建数据；
- 细粒度属性中的“面料”和“工艺”当前不纳入实现范围。

## 3. 数据集分析

### 3.1 DeepFashion2 数据结构

当前主要使用 DeepFashion2 训练集：

```text
fashion_data/
└── raw/
    └── train/
        └── train/
            ├── image/
            └── annos/
```

图片与 JSON 标注按文件名一一对应，例如：

```text
image/000001.jpg
annos/000001.json
```

### 3.2 核心标注字段

DeepFashion2 单个服饰实例主要使用以下字段：

```text
category_name
category_id
bounding_box
segmentation
landmarks
```

当前 3.1.1 主要关注：
- `category_name`：服饰类别名称；
- `category_id`：类别编号；
- `bounding_box`：实例边界框；
- `segmentation`：实例 polygon 标注。

`landmarks` 暂不用于当前实例分割主流程。

### 3.3 标注可视化验证

编写：

```text
scripts/visualize_annotation.py
```

实现：
- 读取原始图片；
- 读取对应 JSON；
- 绘制 category；
- 绘制 bounding box；
- 绘制 segmentation polygon。

目的：
- 确认数据路径正确；
- 确认图片与 JSON 对应关系正确；
- 确认 bbox 和 polygon 解析逻辑正确；
- 在模型训练前人工检查数据标注。

### 3.4 类别分布统计

编写：

```text
scripts/analyze_deepfashion2_categories.py
```

统计训练 annotation。

当前结果：
- annotation 文件：191,961；
- 服饰实例：312,186；
- 原始类别：13；
- 读取失败文件：0。

主要类别分布：

| 类别 | 实例数 | 占比 |
|---|---:|---:|
| short sleeve top | 71,645 | 22.95% |
| trousers | 55,387 | 17.74% |
| shorts | 36,616 | 11.73% |
| long sleeve top | 36,064 | 11.55% |
| skirt | 30,835 | 9.88% |
| vest dress | 17,949 | 5.75% |
| short sleeve dress | 17,211 | 5.51% |
| vest | 16,095 | 5.16% |
| long sleeve outwear | 13,457 | 4.31% |
| long sleeve dress | 7,907 | 2.53% |
| sling dress | 6,492 | 2.08% |
| sling | 1,985 | 0.64% |
| short sleeve outwear | 543 | 0.17% |

结论：
- 原始 13 类存在明显类别不平衡；
- 最大类别与最小类别实例数差距较大；
- 正式训练前需要结合最终业务类别映射，进一步评估是否采用类别合并、采样策略或 loss weighting。

## 4. Baseline 方案

### 4.1 模型选型

PRD 中计算机视觉技术栈包含：
- OpenCV；
- PyTorch；
- SAM-HQ；
- Mask2Former；
- DINOv2。

当前 3.1.1 优先验证 Mask2Former。

初步判断：
- Mask2Former 可用于实例分割，并可输出实例 mask 与类别；
- bounding box 可根据预测 mask 的像素范围计算；
- SAM-HQ 更适合作为后续高质量 mask 精修或提示式分割方案，不作为当前第一版主模型。

### 4.2 COCO 预训练 Baseline 的目的

使用：

```text
facebook/mask2former-swin-tiny-coco-instance
```

作为初始 Baseline。

COCO 并不是项目最终训练数据集，也不是 PRD 指定的业务数据。

使用 COCO 预训练权重的目的：
1. 验证 Mask2Former 推理代码和依赖环境；
2. 检查模型输入、输出与后处理流程；
3. 观察未经过 DeepFashion2 微调时模型的原始能力；
4. 为后续 DeepFashion2 fine-tuning 提供训练前参照。

### 4.3 Baseline 样例结果

对 DeepFashion2 样例 `000001.jpg` 推理。

Ground Truth：

```text
short sleeve top
trousers
```

COCO 预训练模型主要输出：

```text
person
bed
```

结论：
- 模型具备通用实例分割能力；
- COCO 标签空间无法直接覆盖 DeepFashion2 细粒度服饰类别；
- 需要基于 DeepFashion2 进行类别适配及 fine-tuning。

## 5. Person ROI 方案

### 5.1 引入 ROI 的原因

当前业务主要关注人物及人物服饰区域。

带教建议：
- 先定位图片中的人物；
- 提取 person ROI；
- 再在人物区域内进行服饰分割。

预期目的：
- 减少背景干扰；
- 缩小后续服饰模型的有效搜索区域；
- 避免床、墙、家具等背景目标影响服饰解析。

### 5.2 Person Mask 初步实验

对 Mask2Former COCO Baseline 的 `person` 结果单独可视化，包括：
- person mask；
- person bbox；
- person overlay；
- person ROI。

单样例中 person 主体区域识别较准确。

随后随机抽取 10 张 DeepFashion2 图片进行批量验证。

初步观察：
- 大多数样例能够检测到人物；
- 部分图片存在多人；
- 裙摆、鞋子、包等服饰边缘区域可能超出 person mask；
- 因此不适合直接使用 person mask 做硬裁剪。

### 5.3 Bbox + Margin 实验

为降低 person mask 漏掉服饰边缘的风险，测试：

```text
person bbox
+
15% margin
```

将 person bbox 四周向外扩展后再提取 ROI。

初步结果：
- 裙摆、鞋子、包等人物周边服饰区域保留更完整；
- 相比直接使用 person mask 裁剪更稳定；
- 对近景人物，原始 bbox 已接近整图，固定 15% margin 后 ROI 可能接近整张图片；
- 多人图片中，扩大 ROI 也可能包含其他人物。

### 5.4 当前 ROI 技术决策

当前阶段：
- 不使用 person mask 作为硬裁剪区域；
- `person bbox + margin` 作为更优的第一版 ROI 候选方案；
- 固定 15% 仅用于实验验证，不直接作为最终参数；
- 后续需要考虑自适应 margin；
- 多人场景需要进一步设计主人物选择规则。

后续主人物选择可考虑综合：
- confidence；
- person bbox 面积；
- 与图片中心的距离。

### 5.5 Person ROI 实验迭代记录（2026-09-03—2026-09-04）

本节按照“实验设置 → 结果 → 错误案例 → 原因分析 → 优化 → 下一轮验证”的方式记录 Person ROI 的迭代过程。

#### 5.5.1 Experiment 01：直接使用 Person Mask 作为 ROI

**实验目的**

验证 COCO 预训练 Mask2Former 输出的 `person` mask 是否可以直接作为后续服饰实例分割的前置 ROI。

**实验设置**

- 模型：`facebook/mask2former-swin-tiny-coco-instance`
- 数据：DeepFashion2 训练集随机样例
- 批量样本数：10
- 随机种子：42
- 多人场景当前规则：选择置信度最高的 `person`
- 输出：person mask、person bbox、ROI 可视化及统计 CSV

**实验结果**

10 张样例中：

- 9 张成功检测到 `person`
- 1 张未检测到 `person`
- 成功样例平均 person confidence：**0.9656**

整体上，模型对人物主体的定位较稳定，但直接使用 person mask 作为硬裁剪区域存在明显风险：person mask 更关注人体主体，并不能保证把所有穿戴物和服饰边缘完整包含在内。

**典型错误案例**

1. `167622.jpg`
   - 人物上半身区域识别较完整；
   - 花裙部分未完整进入 person mask；
   - 如果直接按照 person mask 裁剪，会损失后续服饰分割所需的信息。

2. `072098.jpg`
   - 人物主体能够正常定位；
   - 手提包区域不属于 person mask；
   - 说明 person mask 不能替代“人物及其穿戴物”的业务 ROI。

3. `177393.jpg`
   - 本轮未检测到 person；
   - 说明 person detector 本身也存在 failure case，后续流程需要保留 fallback 方案。

**原因分析**

当前使用的是 COCO 预训练模型，`person` 类的语义目标是“人物实例”，而不是“人物 + 所有服饰/配饰”。因此裙摆、包、鞋等区域可能不被 person mask 完整覆盖。

**本轮结论**

不使用 person mask 直接进行硬裁剪。保留 person mask 用于人物定位与可视化，同时改为基于 person bbox 构造更宽松的 ROI。

---

#### 5.5.2 Experiment 02：Person Bbox + 固定 15% Margin

**优化动机**

为减少 person mask 漏掉裙摆、鞋、包等区域的问题，在 person bbox 的四个方向分别按 bbox 宽高扩展 15%，并裁剪到图像边界内。

**实验设置**

与 Experiment 01 使用相同的 10 张样例和随机种子，确保前后结果可直接比较。

当前规则：

```text
person mask
↓
计算 person bbox
↓
bbox 宽高各向外扩展 15%
↓
限制在原图边界
↓
得到 ROI
```

**定量结果**

成功检测 person 的 9 张样例中：

- 平均原始 person bbox 占图比例：**44.60%**
- 平均扩展后 ROI 占图比例：**58.12%**

| Image | Person Count | Score | 原 Bbox 占图 | 15% 扩展后占图 | 观察 |
|---|---:|---:|---:|---:|---|
| `167622.jpg` | 1 | 0.9710 | 23.27% | 34.63% | 扩框能够保留 person mask 外的裙摆区域 |
| `029185.jpg` | 1 | 0.9534 | 98.56% | 100.00% | 近景人物，扩展后 ROI 退化为整图 |
| `006557.jpg` | 1 | 0.9625 | 25.33% | 37.76% | 全身/长裙场景，扩展范围较合理 |
| `072098.jpg` | 1 | 0.9650 | 34.55% | 53.20% | 扩框可保留手提包所在周边区域 |
| `064197.jpg` | 1 | 0.9700 | 65.53% | 86.88% | 人物占图较大，扩展后 ROI 过宽 |
| `058514.jpg` | 1 | 0.9700 | 38.39% | 51.72% | 整体范围较合理 |
| `036580.jpg` | 2 | 0.9674 | 33.08% | 48.50% | 检测到多人，当前仅选择最高置信度 person |
| `026869.jpg` | 1 | 0.9644 | 65.17% | 84.40% | 近景人物，扩展后 ROI 占图明显偏大 |
| `142965.jpg` | 3 | 0.9665 | 17.53% | 26.00% | 检测到 3 个 person，主目标选择仍需进一步验证 |

**有效案例**

- `167622.jpg`：person mask 对裙摆覆盖不足，而扩展 bbox 能够把裙摆所在区域保留下来。
- `006557.jpg`：人物为全身/长裙场景，扩展后 ROI 能够保留完整人物及服饰，同时没有退化为整图。
- `072098.jpg`：手提包不属于 person mask，但 bbox 扩展可以保留包所在周边区域。

**新的错误案例**

1. `029185.jpg`
   - 原始 bbox 已占整图 **98.56%**；
   - 固定 15% 扩展后占图 **100%**；
   - ROI 完全退化成原图，失去缩小搜索区域的意义。

2. `064197.jpg`
   - 原始 bbox 占图 **65.53%**；
   - 扩展后升至 **86.88%**；
   - 对近景人物而言，固定 margin 明显偏大。

3. `026869.jpg`
   - 原始 bbox 占图 **65.17%**；
   - 扩展后升至 **84.40%**；
   - 与 `064197.jpg` 表现一致，说明该问题具有重复性。

4. 多人场景
   - `036580.jpg` 检测到 2 个 person；
   - `142965.jpg` 检测到 3 个 person；
   - 当前仅按最高 confidence 选择目标，不能保证始终对应业务上的主要人物。

**原因分析**

固定 15% margin 没有考虑人物在整张图中的尺度差异：

```text
小人物
→ 需要更大的扩展比例
→ 避免鞋、裙摆、包等被截断

大人物 / 近景人物
→ bbox 已经很大
→ 继续固定扩展 15% 会使 ROI 接近整图
```

因此，固定 margin 的问题不是“15% 一定错误”，而是同一个 margin 无法适配不同尺度的人物。

**本轮结论**

`person bbox + margin` 的方向优于直接使用 person mask 硬裁剪，但固定 15% 不适合作为最终策略。下一轮应根据 person bbox 占图比例动态调整 margin。

---

#### 5.5.3 Experiment 03：Adaptive Margin

**实验目的**

针对 Experiment 02 中固定 15% margin 在不同人物尺度下表现不一致的问题，引入基于 person bbox 占图比例的自适应 margin。该轮实验仅修改 margin 策略，人物检测模型、样本、随机种子及主人物选择规则保持不变，以保证前后结果可直接比较。

**实验设置**

- 模型：`facebook/mask2former-swin-tiny-coco-instance`
- 样本数：10
- 随机种子：42
- 主人物规则：仍选择最高置信度 `person`
- 对比基线：固定 15% margin
- 新策略：

```text
bbox_area_ratio < 0.30
→ margin = 20%

0.30 ≤ bbox_area_ratio < 0.60
→ margin = 10%

bbox_area_ratio ≥ 0.60
→ margin = 5%
```

**定量结果**

与固定 15% margin 相比，Adaptive Margin 在 9 张成功检测到 person 的样例中：

- 固定 15% 时平均扩展后 ROI 占图比例：**58.12%**
- Adaptive Margin 时平均扩展后 ROI 占图比例：**55.10%**
- 平均减少：**3.02%** 个百分点
- 相对固定 15% 的平均 ROI 面积下降约：**5.20%**

逐样例结果如下：

| Image | 原 Bbox 占图 | Fixed Margin | Fixed ROI 占图 | Adaptive Margin | Adaptive ROI 占图 | 变化 |
|---|---:|---:|---:|---:|---:|---|
| `167622.jpg` | 23.27% | 15% | 34.63% | 20% | 38.87% | 扩大 4.24% |
| `029185.jpg` | 98.56% | 15% | 100.00% | 5% | 100.00% | 无变化 |
| `006557.jpg` | 25.33% | 15% | 37.76% | 20% | 42.51% | 扩大 4.75% |
| `072098.jpg` | 34.55% | 15% | 53.20% | 10% | 46.91% | 缩小 6.29% |
| `064197.jpg` | 65.53% | 15% | 86.88% | 5% | 75.62% | 缩小 11.26% |
| `058514.jpg` | 38.39% | 15% | 51.72% | 10% | 47.66% | 缩小 4.06% |
| `036580.jpg` | 33.08% | 15% | 48.50% | 10% | 44.59% | 缩小 3.91% |
| `026869.jpg` | 65.17% | 15% | 84.40% | 5% | 71.58% | 缩小 12.82% |
| `142965.jpg` | 17.53% | 15% | 26.00% | 20% | 28.13% | 扩大 2.13% |

**结果分析**

1. **中等人物尺度样例**

   `072098.jpg`、`058514.jpg`、`036580.jpg` 的 margin 从 15% 调整为 10%，扩展后 ROI 均有所缩小。

   其中：
   - `072098.jpg`：53.20% → 46.91%
   - `058514.jpg`：51.72% → 47.66%
   - `036580.jpg`：48.50% → 44.59%

   从可视化结果看，人物主体和主要服饰区域仍被保留，同时减少了部分无关背景。

2. **大人物 / 近景样例**

   `064197.jpg` 与 `026869.jpg` 的 margin 从 15% 降低到 5%，ROI 面积下降最明显：

   - `064197.jpg`：86.88% → 75.62%
   - `026869.jpg`：84.40% → 71.58%

   这说明基于人物尺度降低 margin 可以缓解固定 15% 在近景人物上的过度扩张问题。

3. **小人物样例**

   `167622.jpg`、`006557.jpg`、`142965.jpg` 的 margin 从 15% 提高到 20%，ROI 面积有所增加。

   该调整是有意设计：人物较小时优先保留更多周边区域，用于减少裙摆、鞋子、包等服饰信息被裁掉的风险。

   当前可视化结果表明主要人物及周边服饰区域仍被完整包含，但“服饰完整性是否确实提升”尚未建立单独的定量指标，因此本轮只能作定性判断。

4. **极端近景样例仍未解决**

   `029185.jpg` 原始 person bbox 已占整图 **98.56%**。即使 margin 降为 5%，扩展后的 ROI 仍为 **100%**。

   这说明该场景的问题并不只是 margin 过大，而是 person bbox 本身已经接近整图。继续调小 margin 的收益有限。

5. **多人问题没有改善**

   `036580.jpg` 和 `142965.jpg` 仍分别检测到多个 person。本轮刻意保持“最高置信度 person”选择规则不变，因此 Adaptive Margin 只优化 ROI 尺度，不解决主人物选择问题。

**本轮结论**

Adaptive Margin 相比固定 15% margin 有明确改善：

- 对中等和大尺度人物减少了不必要背景；
- 对小尺度人物保留了更宽松的周边区域；
- 整体平均 ROI 占图比例由 **58.12%** 降至 **55.10%**；
- 同时未改变 person 检测结果，便于确认效果变化来自 margin 策略本身。

但当前策略仍存在两个未解决问题：

1. person bbox 已接近整图时，margin 调整无法产生有效 ROI；
2. 多人场景中最高置信度 person 不一定是业务主目标。

因此 Adaptive Margin 可以作为当前 ROI 方案的第二版，但还不能视为最终方案。

---

#### 5.5.4 下一轮优化方向

下一轮不再继续微调 20% / 10% / 5% 三档参数，而优先处理当前更明显的结构性问题。

**方向 A：极端近景 / 全图人物 fallback**

当：

```text
bbox_area_ratio > 0.85
```

可考虑：

```text
不继续扩框
→ margin = 0

或

直接标记为 full-frame person
→ 跳过 ROI 裁剪
→ 使用原图进入后续服饰分割
```

原因是此时 ROI 已无法有效缩小搜索空间，继续扩框没有实际意义。

**方向 B：多人主目标选择**

保持 ROI margin 方案不变，单独比较主人物选择规则：

```text
highest confidence
vs.
confidence + bbox area
vs.
confidence + bbox area + centrality
```

该实验应与 margin 调整分开进行，避免多个变量同时变化导致结果难以解释。

### 5.6 ROI 第一阶段迭代结论

截至 Adaptive Margin 实验，ROI 优化链路为：

```text
COCO person mask
↓
人物主体定位总体稳定
↓
发现裙摆 / 鞋 / 包等区域可能漏分
↓
Experiment 02：person bbox + fixed 15% margin
↓
服饰周边区域保留更完整
↓
发现近景人物 ROI 过大
↓
Experiment 03：adaptive margin
↓
平均 ROI 占图 58.12% → 55.10%
中/大尺度人物背景区域减少
↓
仍存在：
极端近景 ROI 退化为整图
多人主目标选择不稳定
↓
下一轮：
full-frame fallback + multi-person selection
```

当前阶段技术决策：

1. 不直接使用 person mask 作为硬裁剪 ROI；
2. person bbox 继续作为 ROI 基础；
3. 固定 15% margin 不再作为最终候选；
4. Adaptive Margin 作为当前第二版 ROI 方案；
5. 极端近景场景引入 fallback，而不是继续无限调 margin；
6. 多人主人物选择单独建立下一轮实验。


## 6. Polygon 到训练 Mask 的转换

### 6.1 转换目的

DeepFashion2 的 `segmentation` 为 polygon 坐标。

Mask2Former 训练需要像素级 mask 标签，因此需要完成：

```text
polygon coordinates
↓
binary instance mask
```

### 6.2 Binary Mask 生成

使用 OpenCV：

```text
cv2.fillPoly()
```

将 polygon 内部填充为前景区域。

定义：

```text
0 = background
1 = clothing instance
```

或保存可视化图片时使用：

```text
0 = black
255 = white
```

### 6.3 单样例验证

`000001.jpg` 成功生成：
- short sleeve top binary mask；
- trousers binary mask；
- mask overlay。

可视化结果与 DeepFashion2 polygon 标注基本一致。

## 7. 多实例 Mask Overlap 问题

### 7.1 问题发现

在将多个实例转换成单一 instance segmentation map 时，发现不同服饰实例的 mask 存在重叠。

`000001` 样例检测到约：

```text
20,142 overlapping pixels
```

### 7.2 单一 Instance Map 的问题

单一 instance map 中，一个像素只能存一个 instance ID。

如果直接将多个 mask 写入同一 map：
- 后写入的实例会覆盖前一个实例；
- 原始 DeepFashion2 重叠标注信息会丢失。

### 7.3 当前处理方案

正式 Dataset 中保留每个实例的独立 binary mask：

```text
mask_labels
├── instance 1 mask
├── instance 2 mask
└── ...
```

数据形态：

```text
mask_labels: [N, H, W]
class_labels: [N]
```

优点：
- 保留实例之间的重叠关系；
- 不强制所有实例像素互斥；
- 与 Mask2Former 实例级训练目标更匹配。

## 8. DeepFashion2Dataset

### 8.1 Dataset 目标

实现统一数据读取流程：

```text
image + JSON
↓
读取实例类别
↓
polygon → independent binary masks
↓
图片与 masks 同步 resize
↓
图像预处理
↓
返回训练样本
```

单个 Dataset sample 当前返回：

```text
pixel_values
mask_labels
class_labels
category_names
image_name
```

### 8.2 Resize 策略

当前目标输入尺寸：

```text
384 × 384
```

图片：
- 使用 bilinear resize。

mask：
- 使用 nearest-neighbor resize。

原因：
- RGB 图像属于连续像素值，适合 bilinear；
- mask 为离散标签，必须使用 nearest-neighbor，防止产生非标签值。

### 8.3 单样本结果

样例输出：

```text
Pixel values shape:
(3, 384, 384)

Mask labels shape:
(2, 384, 384)

Class labels:
[0, 7]

Mask values:
[0.0, 1.0]
```

resize 后仍保留实例间重叠关系。

## 9. DataLoader 与 Batch

### 9.1 问题

不同图片包含的服饰实例数量不同，因此 `mask_labels` 不能直接在实例维度使用普通 `torch.stack()`。

### 9.2 Collate 策略

当前：
- `pixel_values`：统一尺寸，stack 为 batch Tensor；
- `mask_labels`：保持为 `list[Tensor]`；
- `class_labels`：保持为 `list[Tensor]`。

示例：

```text
Pixel values:
[2, 3, 384, 384]

Sample 0:
mask_labels [2, 384, 384]

Sample 1:
mask_labels [3, 384, 384]
```

## 10. Mask2Former 13 类训练适配

### 10.1 类别调整

COCO 预训练模型原始分类头：

```text
80 COCO classes + no-object
```

DeepFashion2 当前训练目标：

```text
13 clothing classes + no-object
```

因此：
- 保留预训练视觉与分割相关参数；
- 重新初始化 13 类对应的分类头。

DeepFashion2 原始 `category_id` 为 1~13，模型内部转换为 0~12。

例如：

```text
short sleeve top: 1 → 0
trousers: 8 → 7
```

### 10.2 Forward 与 Loss 验证

完成 batch forward。

结果：

```text
Model class count: 13
Loss: 114.976166
Class logits shape: (2, 100, 14)
Mask logits shape: (2, 100, 96, 96)
```

说明：
- Dataset 输出可正常传入 Mask2Former；
- class labels 正常参与计算；
- mask labels 正常参与计算；
- 模型能够正常产生训练 loss。

当前 loss 数值仅用于训练链路 sanity check，不用于判断模型精度。

## 11. 单步训练验证

完成一次完整训练 step：

```text
forward
↓
loss
↓
backward
↓
gradient
↓
optimizer.step()
```

本地 CPU 环境仅执行一个 batch、一个 step。

结果：

```text
Loss before optimization: 114.791336
Class predictor gradient norm: 247.081696
Parameter change: 0.0358398966
```

结果表明：
- loss 正常计算；
- backward 正常生成梯度；
- optimizer 能正常更新模型参数；
- DeepFashion2 → Mask2Former 基础训练链路已打通。

当前不在本地 CPU 上进行完整训练，正式 fine-tuning 计划在 GPU 服务器执行。

## 12. 当前技术结论

截至当前阶段：

1. DeepFashion2 图片与 annotation 解析流程正常；
2. Polygon → binary instance mask 转换正确；
3. DeepFashion2 13 类存在明显类别不平衡；
4. COCO Mask2Former 可作为训练前 Baseline，但不能直接完成服饰细分类；
5. Person 检测可作为 ROI 前置步骤进行进一步验证；
6. Person mask 不适合直接作为硬裁剪区域；
7. `person bbox + margin` 更适合作为第一版 ROI 方案；
8. 多实例存在 mask overlap，应保留独立 instance masks；
9. DeepFashion2Dataset 与 DataLoader 已完成基础验证；
10. Mask2Former 13 类 forward、loss、backward 和 optimizer step 均已跑通；
11. 当前已经具备进入 GPU fine-tuning 阶段的基础数据和训练链路。

## 13. 当前未解决问题

### 13.1 业务类别映射

DeepFashion2 为 13 类，而项目最终需要 8 大业务类别。

当前尚未正式确定：
- 13 类如何合并到上衣、裤子、裙子、外套、连衣裙；
- vest、sling 等类别的最终业务归属。

该映射应在正式训练配置前确认。

### 13.2 鞋、包、配饰数据

DeepFashion2 无法完整覆盖：
- shoes；
- bags；
- accessories。

后续需要补充自建数据。

### 13.3 类别不平衡

部分原始类别样本数量差距较大。

后续需评估：
- 是否使用最终业务类别合并；
- class weighting；
- sampler；
- 数据增强。

### 13.4 ROI 稳定性

仍需验证：
- 更多样本；
- 多人图片；
- 小人物；
- 遮挡；
- 人物靠边；
- person 检测失败场景。

### 13.5 正式模型指标

当前仅完成训练链路验证，尚未完成：
- GPU fine-tuning；
- validation；
- mask IoU；
- segmentation AP；
- category accuracy；
- inference latency；
- error analysis。

## 14. 下一步计划

### 14.1 代码与工程整理

按照项目编码规范统一整理当前代码：
- 模块结构；
- 函数职责；
- 类型注解；
- Google 风格 docstring；
- 异常处理；
- logging；
- Black；
- isort；
- flake8。

### 14.2 Train / Validation 数据接入

分别建立：
- training dataset；
- validation dataset；
- training DataLoader；
- validation DataLoader。

### 14.3 GPU Fine-tuning

在服务器 GPU 环境：
- 配置训练参数；
- 建立 optimizer / scheduler；
- 完成 checkpoint 保存；
- 记录 loss；
- 进行初步 fine-tuning。

### 14.4 模型评估

逐步加入：
- mask IoU；
- 类别预测指标；
- 实例分割相关指标；
- 典型错误样例分析；
- 推理延迟统计。

### 14.5 后续模块

完成 3.1.1 基础模型验证后，继续进入：
- 3.1.2 语言引导局部区域定位；
- 3.1.3 细粒度属性提取。

## 15. 实验记录维护约定

后续每次形成有意义的实验结论时，更新本文档。

建议重点记录：
- 实验日期；
- 实验目的；
- 模型/checkpoint；
- 数据范围；
- 参数；
- 输出指标；
- 典型成功/失败案例；
- 结论；
- 下一步技术决策。

避免只记录“运行了什么命令”，应重点记录：

```text
问题
↓
实验
↓
结果
↓
分析
↓
技术决策
```
