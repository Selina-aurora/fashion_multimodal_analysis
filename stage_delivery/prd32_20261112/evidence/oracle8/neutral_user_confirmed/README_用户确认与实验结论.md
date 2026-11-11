# PRD 3.2 · Oracle8 中性提问五组 A/B 复测 — 用户确认版

确认时间：2026-11-12（聊天中用户明确回复“全部没有问题”）。
审核范围：03、04、05、06、07 五组；每组红框和青框各一次，共 10 条真实 GPU 回答。
来源：原图、GPU `neutral_results.jsonl` 与此前 AI 辅助逐图审核草稿。

| 案例 | 已确认结论 | 主要错误 |
|---|---|---|
| 03 | 两者均有实质错误 | 目标绑定错到上衣，目标实际为黑色凉鞋 |
| 04 | 红框更好 | 青框增加缺乏证据的结构描述；红框亦非完全正确 |
| 05 | 两者均有实质错误 | 遗漏白色背心上可见的 `miu miu` 字样 |
| 06 | 红框更好 | 青框额外假设牛仔材质 |
| 07 | 青框更好 | 青框颜色描述更接近目标，但图案描述仍缺乏依据 |

汇总：红框更好 2 / 5；青框更好 1 / 5；双方明显有误 2 / 5。

**审计状态**：`USER_CONFIRMED_CHAT_REVIEW`，但 `independent_blind_GT=false`，`production_release_allowed=false`，`SHADOW_BLOCKED` 保持不变。
该复测只能支持开发集上的定性判断，不能证明框线颜色具有稳定收益、不能用作独立盲测，也不能推广至正式主流程。

文件说明：
- `ORACLE8_NEUTRAL_5_PAIRS_USER_CONFIRMED.html`：用户确认版审核表（内嵌图片与答案）
- `ORACLE8_NEUTRAL_5_PAIRS_USER_CONFIRMED.csv`：五组结论及解释
- `neutral_results_GPU_ORIGINAL.jsonl`：未经改写的 GPU 输出
- `provenance_and_sha256.json`：核对来源、哈希与状态
