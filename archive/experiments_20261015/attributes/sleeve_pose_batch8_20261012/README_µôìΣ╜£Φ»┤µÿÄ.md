# 10月12日 — 袖长姿态检测 8 样本诊断

在已有的 `sleeve_pose_sandbox/.venv` 中执行 YOLO11n-pose，不安装任何新包；不修改标注、冻结数据、训练 CSV 或模型权重。

## 操作

将整个 ZIP 上传到 `/workspace` 后运行：

```bash
cd /workspace
python -m zipfile -e sleeve_pose_batch8_20261012.zip /workspace
cd /workspace/fashion_multimodal_analysis
/workspace/sleeve_pose_sandbox/.venv/bin/python /workspace/sleeve_pose_batch8_20261012/run_pose_batch8.py
```

成功后上传终端中 `UPLOAD_THIS_ZIP` 对应的文件（通常为 `/workspace/sleeve_pose_batch8_results_20261012.zip`）。

脚本会核验原始 8 条人工标注 SHA256 和各图 SHA256，确保和先前受控试验一致。仅使用已有的 YOLO11n-pose 权重文件（不会自动下载）；找不到权重会给出查找命令，**不要擅自删掉 SHA256 检查**。

## 输出和解释

- `pose_8_summary.json`：检测到有效手臂的数量、关键点平均误差等。
- `pose_8_details.json`、`pose_8_results.csv`：逐张结果和各关节点误差。
- `visuals/`：8张左为 YOLO 骨架，右为人工四点（黄色）与对应 YOLO 肩、肘、腕点（青色）的叠加图。

**注意：** 本次只验证姿态检测；人工点位仅用于离线匹配和评估，不用于模型推理。无法由本次结果声称已经实现自动袖口定位或自动预测真实袖长覆盖比例。图像左右与 COCO 人体左右不是同一个定义，脚本在诊断时通过人工点匹配；生产流程仍需自动目标人物和手臂选择策略。
