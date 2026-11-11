# GitHub 分支管理与 PRD 3.2 开发状态说明

> 仓库：[`Selina-aurora/fashion_multimodal_analysis`](https://github.com/Selina-aurora/fashion_multimodal_analysis)  
> 核对日期：2026-11-12  
> 文档类型：仓库管理与阶段开发状态说明；**不代表 PRD 3.2 验收通过**。

## 1. 当前结论

本仓库采用“**稳定主分支 + 阶段快照 + 研发归档分支**”的管理方式。`main` 保留 PRD 3.1 已整理的阶段性成果；PRD 3.2 的最新可审阅代码快照、实验记录、GPU 核对与审核结论已上传到 `work/prd32-full-history-20261112`。该新分支已成功推送到 GitHub，**尚未合并到 `main`**。

PRD 3.2 当前仍处于开发与验证阶段。DEV60 等固定开发集实验用于分析候选方案；实验结果不得直接视为独立保留集盲测或正式上线结论。需要进入主流程的候选实现，必须先明确功能覆盖、质量基线、失败案例与独立验证结果。

## 2. 分支一览（2026-11-12 远端核对）

| 远端分支 | 作用 | 当前建议 |
|---|---|---|
| [`main`](https://github.com/Selina-aurora/fashion_multimodal_analysis/tree/main) | 默认主分支；保留 PRD 3.1 阶段交付版本（2026-10-26） | 保持稳定，仅合入经审核的文档或已通过准入的代码 |
| [`stage/prd32-20261027-closeout`](https://github.com/Selina-aurora/fashion_multimodal_analysis/tree/stage/prd32-20261027-closeout) | 2026-10-27 的 PRD 3.2 预检与阶段性证据快照 | 保留为历史检查点，不作为当前主开发入口 |
| [`work/prd32-full-history-20261112`](https://github.com/Selina-aurora/fashion_multimodal_analysis/tree/work/prd32-full-history-20261112) | PRD 3.2 截至 11 月 11 日的代码与实验历史、11 月 12 日 GPU 核对及 Oracle8 审核状态 | **当前最新的 GitHub 可审阅研发归档分支**；后续若继续实际开发，应区分稳定代码与实验快照 |
| `update/prd31-stage-20261023` | PRD 3.1 较早阶段交付记录 | 历史保留 |
| `update/prd31-20261015` | PRD 3.1 早期代码更新记录 | 历史保留 |
| `organized-20261008` | 2026-10-08 仓库整理记录 | 历史保留 |
| `backup-before-overwrite-20261008-1248` | 覆盖操作前的独立备份历史 | **保留，不直接合并或删除**；与 `main` 不具有常规可直接比较的共同祖先 |

截至本次检查，`main` 指向提交 `4a82e3d`，新归档分支指向提交 `0a68ab7`；后者相对 `main` **领先 2 个提交、落后 0 个提交**。这仅说明 Git 历史关系，**不等于 PRD 3.2 具备合并上线资格**。

## 3. PRD 3.2 当前工作与证据位置

| 路径（位于 `work/prd32-full-history-20261112`） | 内容 |
|---|---|
| [`stage_delivery/prd32_20261112/README.md`](https://github.com/Selina-aurora/fashion_multimodal_analysis/blob/work/prd32-full-history-20261112/stage_delivery/prd32_20261112/README.md) | 固定 DEV60、性能瓶颈、架构 A/B、VQA-only、精简提示、多类型试验等阶段结论及限制 |
| [`stage_delivery/prd32_20261112/GPU_RECONCILIATION_AND_REVIEW_STATUS.md`](https://github.com/Selina-aurora/fashion_multimodal_analysis/blob/work/prd32-full-history-20261112/stage_delivery/prd32_20261112/GPU_RECONCILIATION_AND_REVIEW_STATUS.md) | 两轮 GPU 文件核对、Oracle8 5 组用户确认及未完成验证状态 |
| [`stage_delivery/prd32_20261112/QUALITY_BASELINE_AND_PARALLEL_PLAN.md`](https://github.com/Selina-aurora/fashion_multimodal_analysis/blob/work/prd32-full-history-20261112/stage_delivery/prd32_20261112/QUALITY_BASELINE_AND_PARALLEL_PLAN.md) | 质量基线和并行多任务方案的后续验证计划 |
| `stage_delivery/prd32_history_20261112/source_snapshots/` | 按历史来源保存的代码快照，供追溯；**并非每个脚本都是当前推荐执行入口** |

阶段性已核实事实包括：固定 DEV60 涉及 60 张开发图片、171 个预测实例；Oracle8 中性提问 A/B 的 5 组 GPU 结果已由用户确认。该 5 组定性审核中，红框相对较好 2 组、青框相对较好 1 组、两者均有明显问题 2 组；样本量及审核性质不足以证明更改框色能稳定提高效果。

两轮 GPU 文本采集中，第一轮汇总 23,342 个扫描文件，实际打包 4,054 个；第二轮补回此前受容量限制跳过的 3,825 个文件。以上是**文本与代码对账范围**，不代表模型权重、所有原始图像或 GPU 二进制资产已完整备份。完整的私人对账清单与部分图片审核资料未公开存入仓库。

## 4. 当前开发状态与质量准入

**最新已上传分支是研发归档/审阅分支，而非已验收的生产部署分支。** 目前尚未建立正式的 `develop/prd32` 远端分支，也没有把 11 月的实验候选晋升为 `main` 正式实现。

- **统一基线**：候选改动先与固定 DEV60 对照，记录任务定义、输入、版本及质量指标。
- **错误归因**：优先区分上游实例错误、目标绑定偏差、空间定位漂移、类别串扰和无依据属性推断。
- **架构评估**：模板文案、VQA-only、多任务并行等方案需要首先证明功能覆盖和质量不劣于基线；不能只凭耗时更低就宣布改善。
- **独立验证**：任何计划进入正式主流程的方案，应先完成隔离保留集盲测，明确通过标准并复核失败案例。
- **状态门禁**：截至当前，相关实验保持 `SHADOW_BLOCKED`；用户确认的人工审核记录不等于独立盲测完成。

## 5. 推荐的后续 Git 工作流

1. **先将本说明文档提交到 `main`**：建议从 `main` 新建一个仅修改 `docs/BRANCH_STATUS_20261112.md` 的文档分支，提交 Pull Request 审核后合并。**不要**借此次文档更新将整个 PRD 3.2 历史分支合入 `main`。
2. **保留现有 PRD 3.2 历史分支**：`work/prd32-full-history-20261112` 用于保存可追溯实验与当前证据，不删除旧快照。
3. **后续持续开发单独管理**：当确定可维护的 PRD 3.2 实现入口后，可考虑建立 `develop/prd32`，从经过选择的代码和测试开始；该分支目前只是建议，**尚不存在**。
4. **分阶段评审**：先完成代码整理、固定基线复现和实验误差审核；再做独立保留集验证，最后才考虑将通过质量门禁的实现以 PR 形式合并到 `main`。
5. **公开仓库数据边界**：只提交适宜公开的代码、脱敏配置和小体积证据。不得推送 GitHub Token、SSH 私钥、私有原始图片、模型权重及未获授权的数据。

## 6. 已完成与待完成事项

| 项目 | 状态 |
|---|---|
| PRD 3.2 历史代码与阶段材料推送到新远端分支 | **已完成** |
| GPU 两轮文本/代码核对 | **已完成相应范围的对账** |
| Oracle8 中性提问 5 组人工审核 | **用户已确认** |
| 将这份更新后的分支管理说明放进 `main` | **待提交/审核**（本文件尚需上传） |
| PRD 3.2 正式独立保留集准入与主流程晋升 | **未完成** |
| 将整个 PRD 3.2 工作分支合并到 `main` | **未执行，当前不建议** |

---

**简要说明（供汇报）**：`main` 保持 PRD 3.1 的稳定阶段版本；PRD 3.2 的历史代码和实验审核证据已经统一上传至 `work/prd32-full-history-20261112`。目前已完成开发集实验整理、GPU 文件对账和部分人工审核，但尚未完成候选方案的独立保留集准入验证，因此暂不合并实验代码到 `main`。后续将以质量基线、错误归因和独立验证为准，逐步形成可复现、可审核的稳定实现。
