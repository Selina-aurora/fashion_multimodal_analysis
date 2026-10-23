# 10.12 袖长人工审核：下一步 GPU 原图核验（只读）

## 为什么还要这一步
24 条 JSON 均已标记 confirmed，但上一轮 QA 仅 7 条初步保留、9 条待复核，另 8 条不纳入本轮训练。`confirmed_training_labels_added=19` 不代表 19 条训练真值。`v5_train_01719` 的 measurable / sleeveless_confirmed=true 冲突没有擅自修复。

## 操作（GPU终端，只需执行一次）
把整个 `sleeve_source_audit_20261012.zip` 上传到 `/workspace`，然后粘贴：

```bash
conda activate py312
cd /workspace
python -m zipfile -e sleeve_source_audit_20261012.zip /workspace/sleeve_source_audit_20261012
python /workspace/sleeve_source_audit_20261012/extract_source_review.py
```

终端打印 `UPLOAD_THIS_ZIP: /workspace/sleeve_source_audit_20261012_...zip`，**下载那个带时间戳的 ZIP 发回聊天**。如显示 `BLOCKED`，把终端报错发回，不要自行改动 CSV 和验证集。

## 导出内容与保障
- 16 条样本：9 条待复核 + 7 条初步保留。
- 原始无黄框图（校验 SHA256 才导出），四点可视化图以及两张总览图。
- `audit_report.json` 记录 frozen train/val CSV 核验、源图哈希、旧审核数据隔离和每张目标框一致性。
- 不修改 GitHub、GPU 项目数据、模型权重；不训练；不写入任何新训练标签。
- 只读取原训练候选清单 `configs/prd_8class_train_v5_balanced150_20261008.csv` 与冻结验证清单 `configs/prd_8class_val_v2_full_targets.csv`；后者不可改。
- 如果显存昂贵，无需开训；只用 CPU+Pillow。

## 审核局限
源图核验不能独立证明每个袖型类别正确，仍需对照16张图检查点位；当前 7 条初步保留 **不等于最终可训练**。如果存储空间限制，超过 38MiB 将只输出 1400 像素无框缩略图。发回后我可以继续完成精确复核并整理安全的候选标签集。旧两批标注及冻结验证集不动。
