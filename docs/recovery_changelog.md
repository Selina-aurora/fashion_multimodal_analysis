# 本次 PRD 3.1 修复交付

| 问题 | 已落实 | 仍需真实工作 |
|---|---|---|
| B1 train/Core 重叠 | 新清单排除全部 Core/val 源图，547 实例 / 475 图；训练前强制检查，checkpoint 记录清单 SHA | AutoDL 15 epochs 干净重训、同协议测量 |
| 评估不符合冻结定义 | 八个分割入口改为置信度优先匹配；类别错误同时进入 FP/FN；micro/macro/八类；全模块计时 | 新模型真实 GPU 结果 |
| 历史报告容易被覆盖 | 活动评估的新默认报告目录在 reports/reruns/protocol_v2；修复流程在 recovery_v2 | 保留后续每个模型的独立 run 目录 |
| 局部粗框/漏检 | 服装预测 ROI/mask→掩码隔离的 Grounding DINO→SAM；局部结果转为原图框/mask；新增真值与语义审核入口 | 补人工目标 GT，评估新推理结果；不保证已经达到 92% |
| 属性人工 GT 不完整 | 复用颜色 32/37、图案 34/40 开发证据；新八类/六属性集中审核表（160 适用） | 人工完成新真值；真实 88% 准确率和 20ms 计时 |
| 只看匹配子集 | 全部预测实例处理、完整 GT 分母、Track A/B、领口来自预测局部 ROI、明确失败阶段 | 冻结后运行完整验收与局部语义审核 |

本地通过 386 个 Python 文件的语法检查、170 个跨目录静态帮助、30 项 CPU 测试及 5,570 份原始证据 SHA 校验。全流程测试采用临时合成图片和模拟后端，其结果没有作为项目实测成绩保存。

## 对话恢复后的续改

- 新版未完成 checkpoint 可按已完成轮次恢复模型、SGD、学习率计划、随机状态、加权采样器及逐轮日志；checkpoint 通过临时文件原子替换保存。
- 正式验收检查完成 15 轮、两份数据清单及训练设置；旧报告与当前 checkpoint 不一致时拒绝复用。
- 增加 `--phase check`、只读 `--status`、阶段进度、子进程日志和无输出时的 30 秒提示；AutoDL 同工程并行启动会被阻止。
- 新增 [AutoDL 开始步骤](autodl_quick_start.md)。续改后的验证为 39 项通过、1 项因 torch 缺失跳过；真实训练恢复与 GPU 推理仍需在 AutoDL 检查。

上一段的 386/30 为前次版本记录；本次结果单独保存在 `reports/repository_audit/continuation_20260930/validation/`。

当前本地预检查没有 torch/torchvision/transformers 和 CUDA，1,341 个外部图片或 mask 路径不可访问；审核页面原图就绪数为 0/101。原图及处理过的掩码需要在原 AutoDL 数据环境中恢复。这些状态说明为什么 GPU/人工步骤尚未执行，不代表 AutoDL 环境也缺文件。

操作入口：[RECOVERY_GUIDE](recovery_guide.md)。机器验证：[checks.json](../reports/repository_audit/recovery_v2/validation/checks.json)。数据隔离：[summary.json](../reports/repository_audit/recovery_v2/data_isolation/summary.json)。

## checkpoint 加载兼容修正

修正了 17 个历史加载入口的默认模式，使工程自产完整 checkpoint 中的 NumPy 随机状态可被读取。新增真实 Core 入口加载回归检查。当前共 41 项检查：39 项通过、2 项因本地缺少 torch/torchvision 而跳过；AutoDL 真实模型继续评估的结果待回传。本包未包含用户刚训练的权重。
