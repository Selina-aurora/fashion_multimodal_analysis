# PRD 3.1 阶段冻结与遗留问题移交（2026-10-26）

> 仓库建议位置：`docs/handoffs/PRD31_STAGE_FREEZE_HANDOFF_20261026.md`。本文件是可提交的交接记录，**不表示本次已执行 git commit / push**。

## 1. 决策

- **已记录检查状态**：`PRD31_STAGE_FREEZE_READY`；冻结候选 `PRD31_CANDIDATE_20261022_WHITE_SAM_REUSE`。
- 固定 **3.1 Mask R-CNN V5 checkpoint + white-sam 未编译复用候选 + white 回退**；不采纳未获证实的 SAM 非空回退/静态编译，也不恢复效果劣化的人工矩形 ROI 方向。
- 这只是阶段性**接口、资产与身份**就绪；**不是**原 PRD 的精度/延迟、完整生产链路和 3.2 正式功能验收。
- 2026-10-26 **3.2 V2 留作较优开发基线**；3.2 V3 结构化设计字段实验是负向消融，**不晋级**。不要把 3.1 的 V5 冻结候选和 3.2 单图 V5 诊断脚本混为一谈。

## 2. 核验范围和证据级别

| 项目 | 结果 | 证据级别 |
|---|---|---|
| 冻结脚本内部 20 张图、109 实例、12 条 3.2 桥接请求 | `interface_blockers=0` | 本次已从归档源代码嵌入快照独立重算；未运行 3.1 GPU 模型 |
| 冻结源码/权重/4张图片身份 | 历史 AutoDL 执行记录报告通过 | 原 AutoDL `summary.json` 未入包，需另补 |
| 可选质量警告 | 83 条字段/区域提醒 | 不同字段同图可重复出现，不是独立缺陷或准确率 |
| 3.2 V3 结构化输出 | Schema 9/14；非法5/14；原生严格JSON 0/14 | 已从原始 Dev7 V3 14 行 CSV 复算 |

当前原始冻结 `summary.json`、`freeze_record.json` 和 `bridge_inputs.jsonl` **不在**可恢复归档中，不能凭本 MD/`freeze_version_record.json` 假冒 GPU 原证据。

## 3. 固定版本身份证据

**候选**：`PRD31_CANDIDATE_20261022_WHITE_SAM_REUSE`  
**V5 checkpoint SHA256（原检查脚本预期值）**：`c5ba795225349b1c521aac0517ac7c493d9818dcc17d558ac4a3f6f8669452f0`

| 源码文件 | 预期 SHA256 |
|---|---|
| `src/fashion_multimodal_analysis/integration/quality_guards.py` | `5252a7a2c52e81ee655a467fb45f924dd09bc6bfd82de5b26eb789f70589514a` |
| `src/fashion_multimodal_analysis/integration/region_semantics.py` | `c21cedf6d9bb450eea8de9b5f9fa7ad45d55f463ea1240211484a936b8208255` |
| `src/fashion_multimodal_analysis/integration/model_backends.py` | `0efd47f3a29e6c25d7396717cb4fc1b1540ed30c16df171a9ad7b93fade7f7ec` |
| `src/fashion_multimodal_analysis/integration/pipeline.py` | `8c6e8b3278cc94fdbc934defbd7f51aa5081dabb1d767a03c9160854dc22da68` |

历史 GitHub release：<https://github.com/Selina-aurora/fashion_multimodal_analysis/releases/tag/prd31-stage-20261023>。此处仅引用已知历史标签；**本次未改动仓库、未声称新 tag 已发布**。当前完整 Git commit hash 需补录。

## 4. 3.1 → 3.2 字段门槛

- 必须验证图片文件存在且身份 hash 对齐，`source_size` 合理，`instance_id` 唯一，可按请求关联目标，`bbox` 为有效 `xyxy` 且处于原图范围。
- 模型预测的颜色/图案/袖长需注明 `predicted / unverified`；不能把历史 reference ROI 当 frozen V5 predicted ROI，也不能把人工 GT 喂回推理并冒充无监督预测。
- `localization_missing`、空 ROI、属性缺省需显式未知或降级，**但若具体业务接口把该字段设为必填，则对该接口仍是阻塞**。
- 3.2 返回文本还需独立的 Schema 验证，不可因 3.1 已 READY 就豁免 3.2 协议失败。

## 5. P0 阻塞风险与 P1 效果遗留

| 等级 | 项目 | 判定 |
|---|---|---|
| P0（有条件阻塞） | 字段/实例关联失效、图片身份不符、bbox非法、生产加载失败 | 可能阻断 3.2 特定请求；需新端到端 smoke |
| P0（3.2严格消费者） | V3 的 `shirt` 非法类别 3 条，`"null"` 字符串 2 条 | 严格 Schema 9/14；是 3.2 协议问题，不是 3.1 新发现全局阻塞 |
| P0（局部样本） | dev_10 类别冲突、dev_11 路径/SHA 冲突、dev_03 历史参考审核不全 | 保持隔离，影响对应个案评分 |
| P1（效果） | 局部 ROI 漏检/边界、上游错选目标、袖口/下摆/领口、预测属性误判 | 不掩盖错误，交接后持续改进 |
| P1（速度） | 历史局部定位核心均值 139.63ms / P95 178.55ms vs 30ms | 运行可继续，但时延不达标 |
| P1（验收） | 3.2 当前 n=7 非盲，主要 top/outerwear，八类/QA/文案/P95/QPS 未验收 | 不能称完整 PRD 通过 |

详细责任分级、验证条件见 `PRD31_issues_handoff.csv`。

## 6. 3.2 版本和 5 条失败明细

| 输入 | V2 类别/颜色/图案 | V3 类别/颜色/图案（严格门控） | V3 合法 | V2→V3 中位 ms |
|---|---|---|---|---|
| 标框整图 | 6/7,5/7,6/7 | 4/7,3/7,3/7 | 4/7 | 1370.53→2237.39 |
| 历史参考 ROI | 6/7,5/7,6/7 | 5/7,3/7,3/7 | 5/7 | 1305.97→2250.30 |

不合法具体为 `dev_04|marked_full_image`、`dev_04|reference_roi`、`dev_05|marked_full_image` 输出 `shirt`，以及 `dev_12` 两种输入的设计 `strap="null"`。V2 服务端规范化 14/14，但**模型原生严格 JSON 是 0/14**；V3 同样原生 0/14。因此解析成功和原生遵循不可混用。

## 7. 复现（必须在 AutoDL 对应源码/权重/数据仍可用）

```bash
cd /workspace/fashion_multimodal_analysis
python prd31_stage_freeze_check_20261026.py \
  --project . \
  --output /workspace/prd31_stage_freeze_20261026
```

**应补入归档**：`/workspace/prd31_stage_freeze_20261026/{summary.json,freeze_record.json,bridge_inputs.jsonl,optional_quality_warnings.json}`、实际模型权重与源码 SHA 验证日志、Git commit/branch、统一 3.2 实时服务 smoke。千万不要把本地重算的 `evidence_recheck.json` 当原 GPU 报告。

## 8. 后续执行次序

1. 在 frozen V5 真正输出上对接 3.2 统一 `image + target + query + optional predicted attrs`，加入空值/越界/路径及异常回退测试。
2. 基于 3.2 V2 实现 QA / 受控 JSON / 商品文案独立质量验收；保存模型原文、canonical JSON、时延与错误来源。
3. 新增非盲开发样本覆盖八类，最后开展独立评估；**8 张封存 W7 holdout 不用于调参**。
4. 在相同测量协议下评价精度—时延权衡；保留所有未达标项作为下一阶段风险。

复杂面料、工艺和高难度设计属性按带教确认的暂缓范围，不机械以原始 14 类 200+ 属性追责当前 3.1 阶段。
