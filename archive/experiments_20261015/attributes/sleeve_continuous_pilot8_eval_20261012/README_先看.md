# 10月12日 3.1.3 连续袖长诊断（只读，不训练）

实验用**同一批8张已有人工四点标注图片**，和此前8张 CLIP/线性分类头实验一致的**原始服饰 GT bbox**。调用现有 `src/fashion_multimodal_analysis/attributes/geometry/run_sleeve_length_continuous_v1.py` 中 `infer_continuous_ratio`，对比其自动产生的 6-anchor CLIP probability weighted expectation 和人工 2D coverage ratio。

特别注意：现有代码没有自动肩/肘/腕/袖口测量。产生的是 **CLIP 语义连续评分**，并非物理袖长比例；其 MAE 仅作探索性数值偏差，不可直接代表几何测量性能。重点看 Spearman / 成对排序准确率，不能把这个8样本实验当 PRD 验收。

## GPU运行

先上传 `sleeve_continuous_pilot8_eval_20261012.zip` 到 `/workspace`，执行：

```bash
cd /workspace
python -m zipfile -e sleeve_continuous_pilot8_eval_20261012.zip /workspace
cd /workspace/fashion_multimodal_analysis
PYTHONPATH=src python /workspace/sleeve_continuous_pilot8_eval_20261012/run_pilot8_continuous_diagnostic.py
```

会打印 `UPLOAD_THIS_ZIP:`；下载该结果 ZIP 上传即可。也会验证 Core400 已通过隔离检查、8张原图 SHA256、旧记录一致性。不会改动任何训练 CSV、Core400、模型权重或生产代码。

如果报错，原样发送最后15行错误日志，不要直接改项目源码或重训。

## 输出

- `summary.json`: 数值误差、相关性、成对排序、每类别偏差
- `predictions.json`: 人工比例与自动 CLIP 分数及此前分类头/提示词预测逐样本对齐
- `per_case_comparison.csv`: 可直接打开查看的逐样本清单

**8张人工四点GT不会作为推理输入，只用于推理后的对照。**
