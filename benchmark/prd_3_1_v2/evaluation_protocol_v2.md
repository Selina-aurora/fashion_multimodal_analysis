# PRD 3.1 v2 评价定义

v2 遵循原 v1 分割匹配与计时定义，补充真实完整流水线与人工 GT 的可执行实现。类别映射及门槛不变。配置在 `configs/prd_3_1_recovery_v2.json`，审核后快照、清单及哈希写入 frozen。

1. 分割以置信度降序、类别无关 bbox IoU≥0.50 一对一匹配；保留浮点预测框，在原图坐标下计算二值 mask IoU。错误类别同时计 FP 与 FN。报告 micro/macro、八类、mask 均值/中位数/0.85 率和失败代码。
2. 分割 score=0.40、mask=0.50，Core400 原成员不改。局部定位默认 Grounding DINO box=0.30、text=0.25，SAM 输出与输入服装 mask 相交。所有局部框和 mask 回到原图坐标，裁剪来源被记录。
3. 局部 GT 必须人工确定存在性和目标区域框；各区域至少 5 正例。严格正确要求人工 correct、bbox IoU≥0.50、检测成功；Track B 还要求上游服装类别正确。coarse 不算成功。未审核预测不自动置 correct；缺语义判断时严格准确率是 NOT_EVALUATED。
4. 属性 GT 按适用类别、人工审核且非歧义计算；全部适用真值保留在分母。Track A 用 GT 服装 ROI/mask，领口固定几何代理；Track B 使用预测服装 ROI/mask，领口必须来自预测 3.1.2 领口，不回退到 GT。两条轨道分开统计。
5. 采用已有颜色鲁棒多背景、图案四桶 gate/subtype 逻辑及款式提示词；同一 CLIP 模型常驻，文本原型缓存。离散标签采用现有模型词表，包括 geometric_abstract、other_pattern；连续几何工具继续独立保留，不替代人工离散 GT。
6. 所有模型在计时前加载；float32、batch=1、10 预热、5 计时 passes、CUDA 前后同步。包含裁剪/预处理、模型和 CPU 后处理；排除磁盘解码、GT 处理、报告及图片写出。均值/中位数/P95、计时数量、GPU、依赖版本及解析到的模型 revision 均记录。分割单位 image，定位 query-instance，属性 instance。
7. 端到端统计所有冻结目标的分割/分类覆盖、完整成功数、人工 GT 属性准确率与失败阶段；额外预测不依据 GT 预先过滤。没有新的端到端数字门槛。
8. 数值门槛：mask 均值≥0.85、分割均值≤50ms；严格定位≥0.92、定位均值≤30ms；Track A 属性 micro≥0.88、属性均值≤20ms。状态使用 PASS、FAIL、NOT_EVALUATED、INSUFFICIENT_COVERAGE。没有对应八类/属性/区域覆盖或真实 GPU 计时，不能声明全部 PASS。

这些测试源图已用于历史分割回归。清除训练路径交集后，仍不能将其描述为完全未见盲测。换门槛、改标签或挑选困难样本都需要新的 benchmark 版本。
