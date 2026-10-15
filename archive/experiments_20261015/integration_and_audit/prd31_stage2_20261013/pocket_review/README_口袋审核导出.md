# PRD 3.1.2 口袋候选复核 — 第二阶段

## 数据范围与已完成事项
- 依据原 `pocket_candidates_v2_20261009` 的24条 `review_cases.json`，内含42个原生口袋部件、22张不同原图；训练来源19条、验证来源5条。审核**尚未完成**。
- 已对历史数据进行静态检查，并重新核对冻结Core400哈希、冻结20张袖长图片哈希：这24条候选的22张原图与两者均无重复，队列内部训练/验证也没有同图交叉。
- 原始包没有24张图片文件。本包提供GPU侧**原图校验→可视化导出→本地离线人工审核**代码，已经用合成图片端到端演练，尚未在用户GPU的原图上真正运行。
- 绿色框标出目标服装 bbox，黄色框是原生口袋部件的**未确认归属提议**；图上框并非正式GT。
- `pocket24_triage_pending.csv` 的风险提示只帮助安排审核优先级，**绝非定位模型准确率**。

## 现在在GPU运行（第一项）
把总ZIP传到 `/workspace`，然后：
```bash
cd /workspace
python -m zipfile -e prd31_stage2_pocket_softlabel_gate_20261013.zip /workspace/prd31_stage2_20261013
cd /workspace/prd31_stage2_20261013/pocket_review
python -m unittest -v test_prepare_pocket_review.py test_validate_review.py
python prepare_pocket_review.py \
  --project-root /workspace/fashion_multimodal_analysis \
  --out-dir /workspace/prd312_pocket24_review_20261013
```
若成功，终端出现 `POCKET24_IMAGES_AND_OFFLINE_REVIEW_READY`，上传终端 `UPLOAD_THIS_ZIP` 路径对应的文件，一般是：
`/workspace/prd312_pocket24_review_20261013_UPLOAD_THIS.zip`。

该ZIP包含22张不同来源的原图副本（按24条候选记录排列）、24张口袋与衣物候选框叠加图、单文件离线HTML审核工具、预检查报告和风险提示CSV。

如出现 `PREFLIGHT_FAILED`，上传：
`/workspace/prd312_pocket24_review_20261013/preflight.json`。
**切勿**绕过图片SHA256核对或把缺失的图片替换为相似图。

## 第二步：离线审核
收到生成的ZIP后，可以先把文件交给助手协助分析。审核员在浏览器打开 `prd312_pocket24_review_OFFLINE.html`，对每个口袋 P1/P2/P3 单独判定：
1. 是否确属绿色框目标衣物；
2. 黄色部件框位置是否正确；
3. 部件类型是否合理。

允许标记无法确定；可中途导出JSON并重新导入恢复。最终导出 `prd312_pocket24_HUMAN_REVIEW.json` 并上传，不要改原始JSON。

## 数据边界
- 所有 `association_confirmed` 当前为 false，208条大池与24条审核队列都不能算作口袋识别准确率。
- 24条样本是正例优先队列，未覆盖负例，不能计算完整召回率。
- 不能让 `val` 来源5条参与训练；也不能与冻结20张V3评估数据混用。
- `validate_review.py` 可检查人工导出是否结构完整，不会回写原始数据：
```bash
python validate_review.py --review prd312_pocket24_HUMAN_REVIEW.json --report pocket_review_check.json
```
