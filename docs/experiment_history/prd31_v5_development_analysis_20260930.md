# PRD 3.1.1：V5 补全目标训练的开发集结果

日期：2026-09-30。证据：用户上传的 `prd31_v5_results.zip`，共 21 个文件，ZIP 校验通过。

## 结论

V5 的 15 轮 GPU 训练及 V4/V5 开发集对比已完成。补全目标后的版本在这份开发集上明显减少漏检，并提高正确分类实例的平均 mask IoU；类别感知精确率下降，耗时增加。建议将 V5 作为下一次固定 Core400 回归评测的候选。

这些结果属于开发集证据，不能替代 Core400 结果或 PRD 正式验收。开发集只有 32 张图、88 个目标，部分类别样本很少。

## 数据与训练

- 训练图仍为 475 张：DeepFashion2 80 张，Fashionpedia 395 张。
- 训练目标从 547 个增加到 2039 个；增加 1492 个可用原始标注目标。原来已经选中的目标保留。
- 开发验证图仍为 32 张：DeepFashion2 20 张、31 个目标；Fashionpedia 12 张、57 个目标。验证目标从 43 个增加到 88 个。
- 八类完整训练数量：top 296、pants 199、skirt 88、outerwear 155、dress 164、shoe 586、bag 135、accessory 416。
- V5 同时根据完整解码的掩码重建训练框，因此该对比验证的是“补全目标并校正目标几何”的整体修复，不能把全部提升归因于单一因素。
- 训练清单保留原有图像划分。构建程序检查训练与开发/Core 的原图路径无交集；这不等同于图像内容去重。
- Core GT 没有改动。本次 ZIP 没有 V5 的 Core400 评测结果。
- COCO DEFAULT 初始化；15 轮；batch size 1；SGD 初始学习率 0.0025、momentum 0.9、weight decay 0.0005；StepLR 每 5 轮乘 0.1；输入 min/max size 640/1024；水平翻转；seed 20260921；balanced sampler alpha 0.5。
- 训练记录包含连续的 epoch 1–15，每轮采样 475 张图，无断点续训；累计 epoch 耗时 543.76 秒，总训练耗时 555.37 秒。总 loss 从 1.189387 降到 0.397412。标注数量及监督目标已经变化，V4/V5 训练 loss 不能直接用于比较模型优劣。

训练摘要中的 `fixed_val_manifest=configs/prd_8class_val_v1.csv` 是训练程序保留的原验证配置，数据检查因此显示 43 个验证目标。实际比较时两个模型都使用 `configs/prd_8class_val_v2_full_targets.csv` 的 88 个目标；已通过逐条记录和 provenance 核对。

## 同一开发集、同一阈值的对比

共同协议：score threshold 0.40、bbox IoU 匹配阈值 0.50、mask threshold 0.50、预热 10 次、每张图计时 5 次；CUDA / RTX 4090。匹配按置信度降序、一对一、先定位再判断类别。

| 指标 | V4 | V5 |
|---|---:|---:|
| 定位召回率，类别无关 | 53.41%（47/88） | 81.82%（72/88） |
| 定位且类别正确的召回率 | 43.18%（38/88） | 69.32%（61/88） |
| 定位且类别正确、mask IoU ≥ 0.50 | 37/88（42.05%） | 61/88（69.32%） |
| 类别正确匹配实例的平均 mask IoU | 0.7491 | 0.8102 |
| 类别正确且 mask IoU ≥ 0.85，占全部 GT | 10/88（11.36%） | 27/88（30.68%） |
| 类别感知 micro precision | 55.07%（38/69） | 44.20%（61/138） |
| 类别感知 micro F1 | 0.4841 | 0.5398 |
| 置信度 ≥ 0.40 的预测数量 | 69 | 138 |
| 漏检 | 41 | 16 |
| 已定位但类别错误 | 9 | 11 |
| 类别正确但 mask IoU < 0.50 | 1 | 0 |
| 平均每图耗时 | 38.60 ms | 47.53 ms |
| 中位每图耗时 | 30.72 ms | 32.02 ms |
| P95 每图耗时 | 85.81 ms | 96.02 ms |

注意：evaluation_summary 中总体 `bbox50_precision` 是定位匹配数 / 全部预测数，V4 为 68.12%，V5 为 52.17%；它与类别感知 precision 是两个不同指标。上表使用 protocol_metrics.json 的类别感知 precision，避免把定位正确误读成类别正确。

“完整目标”指两个原始数据集在固定八类映射内的可用标注。DeepFashion2 的原始标注类别范围不覆盖所有鞋、包、配饰，因此协议中未匹配的预测不能逐个直接认定为真实误检；需要结合数据集范围与可视化检查。不过，V5 输出的候选数量翻倍、类别感知精确率下降、后处理耗时增加，都是本次比较实际观察到的变化。

在 V4/V5 都定位且类别正确的相同 34 个目标上，平均 mask IoU 从 0.7694 增加到 0.8018。这个比较控制了匹配目标集合的变化。

## 逐类结果

| 类别 | 开发 GT 数 | V4 定位且类别正确 | V5 定位且类别正确 | V4 漏检 | V5 漏检 |
|---|---:|---:|---:|---:|---:|
| top | 14 | 6 | 12 | 6 | 1 |
| pants | 13 | 5 | 7 | 6 | 4 |
| skirt | 3 | 1 | 1 | 1 | 1 |
| outerwear | 7 | 2 | 3 | 3 | 0 |
| dress | 12 | 10 | 10 | 0 | 0 |
| shoe | 22 | 9 | 19 | 13 | 3 |
| bag | 5 | 3 | 3 | 2 | 2 |
| accessory | 12 | 2 | 6 | 10 | 5 |

提升主要来自 shoe、top 和 accessory。outerwear 7 个目标全部被定位，但仅 3 个类别正确；这是后续需要检查的分类问题。skirt 只有 3 个 GT、bag 只有 5 个 GT，不能据此得出稳定的泛化结论。

按来源，DeepFashion2 的定位且类别正确召回率从 58.06% 增到 64.52%，Fashionpedia 从 35.09% 增到 71.93%。这与新增目标主要来自 Fashionpedia 相符，但不是对单一原因的证明。

逐条 outcome 的变化：原先 missed 的 41 个目标中，24 个变为 correct、3 个变为 wrong_class、14 个仍 missed；原先 correct 的 37 个中，34 个保持 correct、2 个变为 wrong_class、1 个变为 missed。另有 3 个 wrong_class 变为 correct，1 个 low_mask_iou 变为 missed。

## 已完成的报告核对

两个模型的 88 个 GT 标识、原图、类别、细分类别及顺序完全一致；同一个开发清单 SHA256；所有冻结评测参数一致。逐条 GT 的定位、类别、IoU ≥ 0.85 计数与 protocol_metrics 相符。每个模型都有 32 张图 × 5 次 = 160 条有效计时记录，逐图计时覆盖完整；重新计算的均值、中位数和 P95 与摘要一致。训练 epoch 连续且每轮采样总数正确。

这些检查针对上传的报告内容和一致性，未在当前机器重新执行 GPU 训练或推理。

关键 SHA256：

```text
V4 checkpoint: f1f662d45ab1bdbb851dbcb9560f3318844ebd4add837f6a097cf34379284678
V5 checkpoint: ea5a4ddf0f002256f1ce5164fc1b4b5b4d2263698f9525c9fbe7d8cf98bc1a65
V5 train manifest: 36ff78831b20e492d787c66161f05747433f8e85f7a089fd5df981c2bdf476e6
Full development manifest: 39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93
Original Core manifest: c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89
```

## 下一步：固定 Core400 回归评测

开发集的 V5 mean mask IoU 为 0.8102，仍低于 PRD 参考目标 0.85；平均耗时距 50 ms 只有约 2.47 ms。先获取同一 Core400 的固定阈值结果，再决定下一轮优化。Core 已有历史检查与调优记录，结果应表述为固定回归证据，不声明为未接触的盲测验收。

在 AutoDL 的 `(py312)` 终端执行以下命令。这是评测，读取已有 V5 checkpoint。使用独立的 V5 报告目录保存结果。计时期间让其他 GPU 训练或评测任务结束。

```bash
cd /workspace/prd31_recovery_20260930/fashion_multimodal_analysis
export FASHION_DATA_ROOT=/workspace/fashion_data
python -u scripts/segmentation/evaluation/eval_prd_8class_maskrcnn_v3_b1_dataexp.py \
  --checkpoint outputs/prd_instance_segmentation/maskrcnn_8class_v5_full_targets/checkpoint_last.pth \
  --val-csv benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv \
  --report-dir reports/reruns/recovery_v2/core_v5_full_targets \
  --device cuda \
  --score-threshold 0.4 \
  --mask-threshold 0.5 \
  --match-bbox-iou 0.5 \
  --warmup-runs 10 \
  --timed-passes 5
```

执行结束后，可直接发送 `reports/reruns/recovery_v2/core_v5_full_targets/evaluation_summary.txt` 的内容，或用以下命令打包完整指标（含已生成的可视化），发送项目根目录的 `prd31_v5_core_results.zip`：

```bash
python - <<'PY'
from pathlib import Path
import zipfile
root = Path.cwd()
folder = root / 'reports/reruns/recovery_v2/core_v5_full_targets'
required = ['evaluation_summary.txt', 'protocol_metrics.json', 'per_gt_results.csv',
            'per_class_metrics.csv', 'per_source_metrics.csv', 'per_image_timing.csv', 'error_cases.csv']
missing = [name for name in required if not (folder / name).is_file()]
if missing:
    raise SystemExit('评测尚未完成，缺少文件：' + ', '.join(missing))
with zipfile.ZipFile(root / 'prd31_v5_core_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(folder.rglob('*')):
        if p.is_file():
            z.write(p, p.relative_to(root))
print('已生成：' + str(root / 'prd31_v5_core_results.zip'))
PY
```

人工 GT 审核及 `--phase finish` 属于后续验收流程。现有 recovery 主流程配置仍指向 V4；Core 回归完成后，需要再决定候选模型，并将 checkpoint、对应训练清单、评测报告及 provenance 一起接入验收配置，避免结果与模型版本不一致。
