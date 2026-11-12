# GitHub 分支管理说明与 PRD 3.2 开发进度（2026-11-12 更新）

**仓库：** [Selina-aurora/fashion_multimodal_analysis](https://github.com/Selina-aurora/fashion_multimodal_analysis)（公开）  
**默认分支：** `main`  
**PRD 3.2 当前工作与归档分支：** `work/prd32-full-history-20261112`  
**说明性质：** 本文件为分支状态、代码归档和研发计划说明；不表示 PRD 3.2 已验收，也不执行任何分支合并、删除或重命名。

## 一、当前分支及其用途

> 以下状态依据 2026-11-12 的 GitHub 远端检查；提交和领先/落后数量可能随着后续推送发生变化。

| 分支 | 现阶段用途 | 管理建议 |
|---|---|---|
| `main` | PRD 3.1 的稳定阶段交付版本；检查时提交为 `4a82e3d`（2026-10-26）。现有主 README 主要说明 PRD 3.1。 | **继续保持默认主分支**。可以先合入分支说明等纯文档更新，暂不合并未经验证的 PRD 3.2 实验实现。 |
| `work/prd32-full-history-20261112` | **已经创建并推送成功**的 PRD 3.2 工作与历史证据归档分支。包含现有 PRD 3.2 代码快照、实验记录、复现说明及 GPU 核对/部分人工审核状态。检查时提交为 `0c9dcf4`，相对 `main` **领先 3 个提交、落后 0 个**。 | 目前作为 PRD 3.2 的主要远端参考分支；后续代码须区分可复现基线、候选实验和已确认审核结论。 |
| `stage/prd32-20261027-closeout` | 2026-10-27 PRD 3.2 诊断与阶段交付快照，保留原始来源；其后建立了上述 `work/` 分支。 | 保留，不作为后续主开发入口。 |
| `update/prd31-stage-20261023` | 2026-10-23 PRD 3.1 阶段更新历史。 | 历史参考，不继续开发。 |
| `update/prd31-20261015` | 2026-10-15 PRD 3.1 旧版更新。 | 历史参考，不继续开发。 |
| `organized-20261008` | 2026-10-08 项目整理快照。 | 历史参考，不继续开发。 |
| `backup-before-overwrite-20261008-1248` | 覆盖操作前的独立历史备份；此前 GitHub 对比未找到与 `main` 共同的祖先提交。 | **继续保留**，不直接合并或删除。 |

截至检查时，GitHub 远端共有上述 **7 个分支**。`develop/prd32` 等名称属于后续可考虑的规范化计划，**目前尚未创建**。

## 二、目前主要在哪个分支开发？

- 远端 PRD 3.2 的**工作/历史证据参考分支**是 [`work/prd32-full-history-20261112`](https://github.com/Selina-aurora/fashion_multimodal_analysis/tree/work/prd32-full-history-20261112)。
- GPU 上仍存在多个不同日期的 PRD 3.2 实验工作目录。当前远端分支主要汇总来源可追溯的代码和证据，**不等于 GPU 上所有实验已整理成一个完整、可直接上线的统一 pipeline**。
- `stage/prd32-20261027-closeout` 用于保留旧阶段快照；`main` 用于稳定集成。后续若建立长期开发分支，应基于已核对的基线明确迁移，而不直接把全量实验输出覆盖 `main`。

## 三、PRD 3.2 已完成的阶段性验证（开发集）

### 1. 文案与类别质量

- 在冻结 **DEV60（60 张图片、171 个 V5 预测实例）**上进行了类别输出对账。
- 对 **24 条 Copy / 模板 Copy** 候选进行了开发集审核：20 条原版相对更好，4 条两者均有明显问题；**0 条模板候选被准许直接替代原版**。
- 模板方案的主要瓶颈仍涉及实例类别、目标绑定和属性细节遗漏；目前不采用仅凭耗时下降就替换原版的做法。

### 2. 重复实例与 Mask 风险定位

- 在 DEV60 的 171 个 V5 预测实例中，筛出 **23 组高重叠候选**，涉及 **20 张图片**，并完成逐对开发集分析。
- 针对其中 **13 组**进行了影子去重离线模拟、Mask IoU / 独有区域计算及差异图可视化。
- 发现最高置信度候选仍可能类别错误，局部/整体粒度、相邻配饰和掩码边界也可能造成误抑制。**13 组视觉与几何判断已在聊天中由用户确认**，但此确认不等于独立逐对 GT 或盲测。
- 现有重点风险案例包括 `fresh_dev_001`、`fresh_dev_015`、`fresh_dev_025`、`fresh_dev_040`、`fresh_dev_041` 等，后续需要真实类别、物件数量及实例边界核验。

### 3. 并行多任务实验的判断原则

- 先前架构 A/B 中，不同路径具有不同的功能范围；因此其耗时差不能直接解释为**功能等价的并行加速收益**。
- 后续应先验证模板质量和目标绑定效果，在**相同任务功能、输入与评价口径**下，再比较串行/并行端到端耗时、输出质量及错误率。

**统一质量门禁：** 当前相关候选保持 `SHADOW_BLOCKED`，尚未完成独立 GT 核验及独立保留集盲测；未自动删除 V5 实例，未将去重候选合入正式推理流程，也不宣称 PRD 3.2 已达到验收标准。

## 四、建议的分支管理与后续工作

1. **`main`：** 保持 PRD 3.1 稳定版本；分支说明等经过检查的纯文档可以单独 PR 合入。
2. **`work/prd32-full-history-20261112`：** 保存已核对的 PRD 3.2 阶段代码、实验来源及审计证据。请对原始图片、私有 GPU 备份、密钥和模型权重等实行公开仓库发布前检查，不直接上传私人归档包。
3. **开发入口：** 后续可在完成代码与实验来源整理后新建 `develop/prd32`，集中维护明确的可复现开发基线；在此之前，不把历史归档分支等同于生产主线。
4. **质量优先：** 优先核查 P0/P1 高风险实例的 GT、真实物件数、误抑制风险与召回影响；之后在独立保留集上盲测，再决定是否准入主流程。
5. **性能对照：** 模板质量和目标绑定达到门禁后，开展功能等价的串行/并行多任务基准测试。现在不继续在解码超参数层面反复试错。

## 五、快速查看链接

- [主分支 `main`](https://github.com/Selina-aurora/fashion_multimodal_analysis/tree/main)
- [当前 PRD 3.2 工作与归档分支](https://github.com/Selina-aurora/fashion_multimodal_analysis/tree/work/prd32-full-history-20261112)
- [PRD 3.2 阶段 README](https://github.com/Selina-aurora/fashion_multimodal_analysis/blob/work/prd32-full-history-20261112/stage_delivery/prd32_20261112/README.md)
- [GPU 核对及审核状态说明](https://github.com/Selina-aurora/fashion_multimodal_analysis/blob/work/prd32-full-history-20261112/stage_delivery/prd32_20261112/GPU_RECONCILIATION_AND_REVIEW_STATUS.md)
- [GitHub 主分支与 PRD 3.2 分支比较](https://github.com/Selina-aurora/fashion_multimodal_analysis/compare/main...work/prd32-full-history-20261112)

---

**文档更新日期：** 2026-11-12。  
**注意：** 这份 Markdown 是更新后的说明文件；只有用户提交到 GitHub 后，远端路径 `main/docs/BRANCH_STATUS_20261112.md` 才会显示此新版内容。当前旧分支里同名文档可能仍为过时快照，上传时应以本文件覆盖。
