# PRD 3.1.3 袖长实验检查与下一步（2026-10-12）

## 现有证据

提供的 `sleeve_eval_materials_20261012.zip` 含10月9日实际结果。旧线性分类头训练已运行完成，44条人工确认训练来源、32条内部训练、12条内部开发。**旧模型不等于已经上线。**

| 范围 | 线性分类头 | 同样原框视角的CLIP提示词基线 | 适用解释 |
| --- | --- | --- | --- |
| 32条训练 | 32/32（100%） | 7/32（21.9%） | 不能作为泛化证据 |
| 12条开发 | 7/12（58.3%） | 4/12（33.3%） | 分类头按此集选epoch，乐观偏差 |
| 15条诊断GT ROI | 5/15（33.3%） | 7/15（46.7%） | 已多次检查的诊断样本，不是新的盲测 |
| 15条诊断预测ROI | 4/15（26.7%） | 4/15（26.7%） | 有2个定位失败；均计为错误 |

15条 GT ROI：头独对3、提示词独对5、双方都对2、双方都错5。
15条预测 ROI：头独对3、提示词独对3、双方都对1、双方都错8。

## 当前证据的限制

- `benchmark/prd_3_1_v1/manifests/attribute_test_v1.csv` 只有表头，没有正式测试样本（0行）。
- `reports/prd_attribute_extraction/continuous_v1/sleeve_continuous_predictions.csv` 有101条旧版连续数值，但它是由旧 CLIP 锚点方案得到的模型输出；**不等于10月12日四点人工测量的真实袖长比例**，不可直接当作新方案验证。
- `continuous_v2_geometry/sleeve_geometry_features.csv` 也有101条几何形状特征，不等于肩肘腕袖口连续比例预测模型。
- 10月12日pilot8的`ratio`是人工GT，不是算法预测；直接和CLIP预测精度比较属于信息泄漏。
- 本次冻结验证集及Core400与pilot8的隔离检查已由之前 `pilot8_preflight_core400_report.json` 证实；本脚本运行时再次检查。

## 下一步：只做历史冻结方案在8张新候选上的推理诊断

1. 把 `sleeve_eval_next_20261012.zip` 上传到 GPU 的 `/workspace`。
2. 在终端运行：

```bash
cd /workspace
python -m zipfile -e sleeve_eval_next_20261012.zip /workspace
cd /workspace/fashion_multimodal_analysis
PYTHONPATH=src python /workspace/sleeve_eval_next_20261012/run_pilot8_frozen_baselines.py
```

3. 将终端 `UPLOAD_THIS_ZIP` 指向的文件发回。

此脚本使用10月9日既存模型权重 `reports/reruns/recovery_20261009/sleeve_linear_train_20261009_073022_027703/sleeve_head_best.pth`。**没有该文件时脚本会停止，不会偷偷重新训练。**

预计会离线加载GPU缓存里的 `openai/clip-vit-base-patch32`；若GPU无模型缓存则会提示缺失。使用与旧实验相同的**原始服装GT bbox，无扩边**输入，做线性头与提示词基线的逐样本公平比较；不会更新原始训练CSV、验证集或生产代码。

**即使8样本结果好，也只能报告小样本诊断，不代表连续比例方案提升，也不构成PRD验收。** 后续需独立开发连续比例自动预测和更多独立标签，再对比连续比值误差（MAE）和袖长分类指标。
