# PRD3.1 输出一致性候选补丁，2026-10-09

本包包括可安装代码和已完成的八图缓存验证结果 `cached_review/`。用户无需为本次验证再次运行模型或训练。

## 修改与局限

- 同一实例的不同局部名称共享几乎相同的框（IoU >= 0.90）时，两者标为 requires_review。部位可能接近，该阈值只是审核线索，不能证明哪一个错误。
- DINO 明确命名的图案与 CLIP 图案标签矛盾时，保留两个候选并转入审核。泛称 printed pattern、pattern 和多类型混合短语不用于强判图案类型。
- 依赖冲突领口的领型也转入审核。
- 所有原始类别、框、mask 和模型证据保留，不自动合并实例，不按旧预审意见修正 top/dress/jumpsuit 等归属。

这是交付状态和数据流的一致性保护。未新增语义 GT、未修复模型识别能力、未证明精度提升，不能按冲突数量计算 PRD 准确率。requires_review 和空正式属性 label 是弃权，不能计为正确或部位不存在。下游应读取 status，并展示模型候选供审核。

真实八图缓存中：25 条图案跨模块冲突、9 对几乎重合的部位框。34 个服装实例及类别/框/掩码均保留。`consistency_review_summary.json` 是本次审核统计；复制过来的 `summary.json` 描述原始模型推理，不是本次步骤的推理耗时或验收统计。

## GPU 安装

把 ZIP 上传到项目根目录，用 Python zipfile 解压到 patches，然后执行：

```bash
cd /workspace/fashion_multimodal_analysis
python patches/prd31_consistency_patch_20261009/install_patch.py
```

安装器核对现有 pipeline 版本并备份，版本不符时不会覆盖。备份在 `reports/patch_backups/semantic_consistency_*`。

运行时默认为关闭，仅在 backend config 中显式设置 `semantic_consistency_review=True` 才启用候选逻辑。因此其他现有模型推理不会因安装而自动采用新弃权策略。

若想在 GPU 对同一缓存结果导出审核版，可运行：

```bash
PYTHONPATH=src python scripts/review_prd31_consistency_20261009.py
```

该脚本自动找到今日 neckline-context 八图报告，写入独立目录；不加载模型、不修改原报告。此步骤与本包已经完成的 cached_review 验证相同，无需再次上传结果。

## 已验证

9 项规则及 pipeline 分支测试通过，原有10项几何保护测试通过。模型之外的测试不证明模型精度。缓存复核检查原类别/框、所有 PNG mask 和输入目录内容保持不变。已有输出新增 raw_regions_before_consistency_review / raw_attributes_before_consistency_review，提供逐实例原始证据。

## 查看结果

每个 case 的 consistency_review.jpg 中橙色框为需复核区域。predictions.json 保存审核状态及原候选。全部冲突记录在 consistency_review_evidence.json。主要遗留问题仍为上游实例类别错误、局部语义定位错误以及袖长/图案误判；本包不能代替人工冻结 GT 或正式验收。
