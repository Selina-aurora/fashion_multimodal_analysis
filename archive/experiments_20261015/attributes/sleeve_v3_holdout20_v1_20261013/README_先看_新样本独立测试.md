# PRD 3.1.3 袖口 V3 dev_v1：20张新样本固定规则验证（2026-10-13）

**目的：**在不重新训练、不修改 V1 几何规则的情况下，从未参与此前 V1 开发的一批新原图中，固定选样并运行全流程 V2 自动分割/姿态 + V3 端点候选预测。此包尚未包含新样本的人工 GT，因此**当前只能完成冻结预测，不能报告新集 A/B 或中点准确率**。

## 已完成的准备工作

- V3 `geometry_v3_proposals.py` 从既有 `sleeve_v3_auto_endpoints_dev_v1_20261013.zip` **原样复制**，不针对新样本修改阈值。
- `legacy_v2/` 包含原 V2 GPU 运行脚本与几何代码，模型权重不包含在 ZIP，继续使用你 GPU 工作区里的已存在权重。
- 附有 Core400 的400条哈希、既有8张开发图索引，以及10月12日24候选与先前44条父样本记录的文件名/哈希排除表。文件选择前，还逐字节验证训练、验证及Core400 **原 CSV SHA**，然后根据训练/验证 CSV中的图片字段排除引用的图片（可解析源文件时同时核验内容SHA）。如果 CSV 格式未知或 SHA 改变，直接停止，不自动放宽。
- 固定选图种子+相对路径散列决定候选顺序；不读取人工袖长、服装类别真值、模型输出，也不按易难程度挑图。
- 保留所有服装实例、两侧手臂和每一个拒绝原因；不筛选“只展示成功”的结果。
- 13项几何/选样/模拟输入本地单元测试通过。**GPU上的真实新图推理尚未运行，不能假设权重和原始数据必然存在。**

## 在 GPU 上执行（复制下面整段）

先将下载的 `sleeve_v3_holdout20_v1_20261013.zip` 上传到 GPU 的 `/workspace/`：

```bash
set -e
cd /workspace
python -m zipfile -e sleeve_v3_holdout20_v1_20261013.zip /workspace
cd /workspace/sleeve_v3_holdout20_v1_20261013

PY=/workspace/sleeve_pose_sandbox/.venv/bin/python
"$PY" -m unittest discover -s . -p 'test_*.py' -v

"$PY" prepare_holdout.py \
  --root /workspace/fashion_multimodal_analysis \
  --candidate-dir /workspace/fashion_data/raw/fashionpedia/images/test \
  --n 20 \
  --out-dir /workspace/sleeve_v3_holdout20_20261013

"$PY" run_frozen_inference.py \
  --root /workspace/fashion_multimodal_analysis \
  --out-dir /workspace/sleeve_v3_holdout20_20261013 \
  --pose-python /workspace/sleeve_pose_sandbox/.venv/bin/python \
  --pose-weights /workspace/sleeve_pose_sandbox/yolo11n-pose.pt \
  --checkpoint outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth
```

完成后上传下面两个文件到 ChatGPT：

1. `/workspace/sleeve_v3_holdout20_20261013/UPLOAD_01_PREDICTIONS.zip`：原始预测、掩码、姿态、全部实例双臂 V3 A/B/拒绝原因、冻结SHA及可视化。
2. `/workspace/sleeve_v3_holdout20_20261013/UPLOAD_02_ORIGINALS.zip`：对应新图片原始字节、固定清单及用于**不看预测**的原图审核总览。

**提示：**如不方便一次上传原图 ZIP，可先给我`UPLOAD_01_PREDICTIONS.zip`，但之后仍需原始图片进行盲标。任何失败请发终端错误及 `/workspace/sleeve_v3_holdout20_20261013/predictions_frozen/ERROR.txt`（若已产生）。若输出目录已经存在，不要删除或覆盖以往文件；将 `--out-dir` 改成一个新目录名，两个命令保持一致。若训练/验证CSV校验失败，不应自行跳过或重新抽取“漂亮结果”，应先核查环境和来源。

## GPU端预期输出不等于准确率

这一步能算：总图片数、全部被预测的服装实例和左右臂数、完整/部分/拒绝端点建议的数量与原因、全量图片上任何完整候选的覆盖。

这一步**不能算**：人工端点误差、可测/不可测真值下的识别率、准确率及PRD验收。需要对本批原图进行**先不显示预测结果的人工标注**，记录A/B、不可测、目标服装实例/侧别、未检测衣物，以及标注置信度，再读取冻结的自动预测进行匹配和误差统计。所有20张都在固定分母内；非目标或完全不可测也需明确记录，不能直接删除难图。`holdout_001` 等ID不代表任何人工袖长类别。

## 非泄漏与局限

1. 原8张先前已看过多次，仅作开发集；这里使用全新选样规则筛出的候选，不再用它们报告独立性能。
2. 与训练/验证CSV引用的原图做文件名和有源文件时的内容SHA排除；与Core400/历史开发图做SHA或源文件名排除。但**不同文件名重新编码、近重复图和预训练模型的外部训练语料重叠**并不能被此机制完全排除。
3. 数据来源单一且随机图可能没有完整可见袖口。完成后先进行“自然分布下的端到端覆盖/拒绝”分析；还需单独建立结构化袖长类别/难例集做诊断，不能把二者混为一谈。
4. V3 目前只是需要人工复核的候选提取器，尚无可推广的端点误差结论。不要因此宣称软标签训练已完成。
5. 固定20张本批画像用于一次验收式验证后，若据此修改 V3 阈值，本批立即转为新的开发集；下一次性能声明需另取新的未见样本。

## 文件组成

- `prepare_holdout.py`：新图片随机顺序固定、排除及冻结清单；只读取原数据、训练/验证/Core清单；只在新输出目录写入。
- `run_frozen_inference.py`：GPU V2 + 原样 V3 dev_v1，完整留存所有实例双臂，输出两份传输ZIP。
- `geometry_v3_proposals.py`：固定的 V3 V1 算法。
- `legacy_v2/`：复用旧已运行的 GPU 分割与姿态流程，不含新模型训练。
- `inputs/`：隔离索引（无人工端点GT）。
- `test_geometry_v3.py`、`test_holdout_pipeline.py`：一共13项本地测试。
- `SHA256SUMS.txt`：关键文件完整性哈希。
