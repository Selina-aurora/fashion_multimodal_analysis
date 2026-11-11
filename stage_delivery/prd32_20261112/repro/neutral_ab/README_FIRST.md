# PRD 3.2 · ORACLE8 中性提问、严格标记颜色 A/B（2026-11-11）

## 本轮目标

原 Oracle8 7 组人工确认：**2 组 geometry_only，5 组 marker_color_only**。
5 组红/青框的 B 版本仍使用“红框”的提问，无法将结果变化单独归因于标记颜色。
本包只重测这 **5 组 × 2 颜色 = 10 次 Qwen 推理**，不重复混用另 2 组框几何变化试验。

## 已完成（无需 GPU）

- `reference/ORACLE8_7_PAIRS_USER_ACCEPTED.csv`：上一轮 7 组结论按用户本轮“没问题”确认；**不是独立盲测 GT**。
- `images/*_RED_control.png` 与 `*_CYAN_control.png`：同一个 JPEG 底图解码后生成无损 PNG，**除了框线颜色像素外，所有像素逐点完全相同**；不直接拿两张不同 JPEG 当颜色 A/B。
- `manifest.csv` / `manifest.json`：5 个目标的固定中性提问、图像 SHA256、来源标识和已确认的旧观察。
- `marker_control_integrity.json`：逐对像素一致性核验记录。

## GPU 容器运行（优先使用已有 py312 环境）

```bash
cd /workspace
python -m zipfile -e PRD32_ORACLE8_NEUTRAL_AB_V2_20261111_UPLOAD_THIS.zip .
cd PRD32_ORACLE8_NEUTRAL_AB_V2_20261111
python validate_inputs.py
python run_neutral_ab.py --model /workspace/models/Qwen2.5-VL-7B-Instruct --output outputs/neutral_results.jsonl
python build_review.py --results outputs/neutral_results.jsonl --output outputs/NEUTRAL_5_PAIRS_REVIEW.html
```

若模型目录不同，将 `--model` 指向实际模型路径；**运行前不要改提问或切换其它模型/权重**。
仅在 GPU 环境运行（本包未包含模型权重，也没有预先生成的推理答案）。
如缺依赖，建议在隔离环境中安装 `transformers>=4.49,<5`、`accelerate`、`qwen-vl-utils`、`Pillow`；尽可能沿用当前项目已锁定的 torch/transformers 版本，避免环境漂移。

## 返传这两个文件

- `outputs/neutral_results.jsonl`（真实 GPU 回答、耗时、配置）
- `outputs/NEUTRAL_5_PAIRS_REVIEW.html`（可人工核对并导出 CSV）

## 如何判定

- **主结果：** 同一中性提问下 red 和 cyan 对目标的绑定、类别、可见属性支持是否存在可复核的差异。
- **额外记录：** 拒答、未经支持的材质断言、文字/Logo遗漏、配饰类别错误、对衣物类别/目标的串扰。
- **不要把两边答案相同视作正确，也不要把更长、更自信的答复自动视作提升。**
- 仍处于 `SHADOW_BLOCKED`：任何方向都不直接改正式生产路径，不使用已经曝光的 Oracle8 样本充当独立保留集，候选进入主流程前必须在封存 holdout 完成盲测。

## 数据完整性与公平性

- 同一个模型实例、同一个 processor、相同 generation 参数（默认 greedy、max_new_tokens=96）。
- 同一套中性提问、图片尺寸相同、框坐标相同、除了框线颜色变动外全部像素相同。
- 每条结果写入 `case_id`, `arm`, `prompt`, `model`, `sha256`, `duration_ms`, `generated_tokens`。
- 5 组是定位混杂因素的探索性样本，不能报告泛化精度或显著性结论。

## 成功判据

本阶段成功 = 10 条输出全部形成、可复核、两种颜色对应的图像完整性检查通过。
“青框更好”**不是**预设成功条件。5 组跑完先归因，之后优先去 DEV60 固定集检查实例绑定与类别串扰，并在独立封存集盲测。
