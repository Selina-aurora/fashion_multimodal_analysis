# 软标签10条候选：GPU训练准入隔离检查

## 已完成
- 4类开发用软标签、18条soft target、10条后续训练候选已在上一阶段产生，本包保留其源文件。
- 本次新增 `verify_softlabel_train_gate.py` **仅检查**：训练/验证CSV字节级SHA256是否仍等于2026-10-12来源快照；10条候选的原始图像SHA是否与人工标注匹配；文件名是否确实出现于训练CSV且不出现于验证CSV；候选图像SHA是否与 Core400 和冻结20张重合；候选是否仍属priority且可测。
- 门禁失败会停止，并打印具体原因；不修改数据和模型，不训练任何head。

## GPU上运行
```bash
cd /workspace
python -m zipfile -e prd31_stage2_pocket_softlabel_gate_20261013.zip /workspace/prd31_stage2_20261013
python /workspace/prd31_stage2_20261013/softlabel_gate/verify_softlabel_train_gate.py
```
执行后上传 `/workspace/softlabel_gate_report_20261013.json`。

即使状态是 `CANDIDATE_SOURCE_GATE_PASSED_NO_TRAINING`，也**不代表训练已批准**。四类别锚点极少、七分袖/长袖缺失，需经人工确认样本用途与六类定义后才可做正式 hard-vs-soft 训练对照；这次只验证当前候选的快照来源安全性。

如果因CSV SHA变化而失败，先上传报告排查，不要覆盖当前CSV或跳过哈希检查。
