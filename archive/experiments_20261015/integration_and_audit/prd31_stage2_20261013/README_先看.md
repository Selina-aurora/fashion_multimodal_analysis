# 10月13日 PRD3.1 提前推进包 — 下一步两项操作

本包可以在无网络的GPU实例运行，**不训练模型，不写入代码仓库、模型checkpoint、Core400或冻结20图。**

A. 口袋审核包：运行 `pocket_review/prepare_pocket_review.py`，从已有Fashionpedia原图导出24条候选的原图+叠加图及离线审核HTML。**文件导出 ≠ 人工审核完成**。可在导出后的页面集中判断42个口袋部件；验证5条只用于诊断。

B. 软标签准入检查：运行 `softlabel_gate/verify_softlabel_train_gate.py`，校验10条候选与旧训练/验证CSV快照、原始图像和冻结基准的隔离关系。**准入检查通过 ≠ 已完成软标签训练**。

## GPU一键操作（建议分步观察报错）
先将 `prd31_stage2_pocket_softlabel_gate_20261013.zip` 上传到 `/workspace`，执行：
```bash
cd /workspace
python -m zipfile -e prd31_stage2_pocket_softlabel_gate_20261013.zip /workspace/prd31_stage2_20261013
cd /workspace/prd31_stage2_20261013/pocket_review
python -m unittest -v test_prepare_pocket_review.py test_validate_review.py
python prepare_pocket_review.py --project-root /workspace/fashion_multimodal_analysis --out-dir /workspace/prd312_pocket24_review_20261013
python /workspace/prd31_stage2_20261013/softlabel_gate/verify_softlabel_train_gate.py
```

## 发回给助手的文件
- `/workspace/prd312_pocket24_review_20261013_UPLOAD_THIS.zip`（如果生成成功）；
- `/workspace/softlabel_gate_report_20261013.json`；
- 若原图缺失或不一致，则发 `.../preflight.json` 和终端报错。

两个检查互不依赖，可以先完成一个。请勿为“通过”而跳过哈希校验，也不要把开发样本转成正式验收样本。

`3_1_2_下摆腰部错误归因说明.md` 说明剩余证据缺口；未获得逐例文件前只保留先前28案例的聚合结论，不能生成单例分数。
