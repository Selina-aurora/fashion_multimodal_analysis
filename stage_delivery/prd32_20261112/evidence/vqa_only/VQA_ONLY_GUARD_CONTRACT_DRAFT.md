# VQA-only 独立 Guard 契约草案（仅供设计，未实现/未批准）

## 问题
原 Guard 接收 Tags、VQA、Copy 三任务完整输出。VQA-only 没有 Tags 与 Copy，若复用旧 Guard 的成功/放行逻辑会造成契约缺失。因此当前必须在 Shadow 状态强制禁止发布。

## 提议输入（所有字段必须有明确上游出处）
- `request_id`, `image_id`, `target_instance_id`, `target_provenance`: 保留原始图片和目标的关联
- `query_text`, `question_type`: 不把单一颜色问题当成通用 VQA
- `target_marking_provenance`: 框/Mask 来自 V5 还是已审核标注；“与冻结 Mask 一致”不能充当 GT
- `raw_answer`, `model_version`, `prompt_hash`, `runtime_version`
- `target_identity_status`: 目标身份是否清楚；未知必须阻断
- `schema_version`, `fallback_reason`

## 提议状态机
- `BLOCKED_CONTRACT_INCOMPLETE`：缺少必要输入、版本或映射
- `BLOCKED_TARGET_UNCERTAIN`：目标身份不清，或目标绑定未验证
- `BLOCKED_MODEL_UNCERTAIN`：模型明确回答“无法确定”等
- `BLOCKED_UNSUPPORTED_QUESTION`：超出当前验证范围的问题类型
- `BLOCKED_FORMAT_OR_EVIDENCE`：不合法格式、可见证据不足或内容不符
- `SHADOW_CANDIDATE_FOR_REVIEW`：满足初步校验但仍需人为/独立验收验证，**不允许发布**

## 重要原则
- **本草案所有状态 `publish_allowed=false`**，没有成功发布状态；以后发布必须经带教批准的新 Guard 实现与独立验证。
- VQA 任务不应以固定颜色问答的通过率推断所有问答类型质量。
- 可将冗长回答规范化为结构化颜色，但必须保留双色和花纹等可见细节；重格式化不能算提高准确率。
- 记录缺失/不确定/阻断的分母，不得只报告放行样本的准确率。
