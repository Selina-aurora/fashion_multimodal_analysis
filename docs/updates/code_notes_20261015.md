# 当前主流程的关键代码说明

这份说明补充代码的调用关系和设计理由；代码内既有注释保留。它不是模型精度证明。

## 实例与局部区域

`integration/pipeline.py` 将每个预测实例传给区域与属性后端。实例保持各自ID和原图坐标；重叠框记录复核标志，不直接删除叠穿实例。袖长预测为无袖时，仅提示检查袖口适用性，不将模型判断转换成人工真值。

`integration/model_backends.py` 在 strict_region_grounding 开启时使用原始服饰裁剪保留上下文，通过类别适用性、提示词匹配和候选面积检查过滤区域。候选显式按置信度排序；SAM结果受服饰mask及局部候选框约束，再回填整图坐标。裁剪错误仍可能传播，几何规则不是语义正确的保证。

`integration/quality_guards.py` 处理候选框面积与mask裁剪，并生成重叠实例复核提示；`region_semantics.py` 定义部位提示词与类别适用性。拒绝结果与真实部位不存在应分开统计。

## 属性与一致性

`integration/attribute_context.py` 在服饰与图像范围内扩展领口上下文，保留原裁剪与上下文预测；两者冲突时要求复核。该候选由配置开关控制。

`integration/semantic_consistency.py` 在属性计算后检查袖口/下摆重叠、无袖与袖口等冲突。规则只标记矛盾，不覆盖服饰类别或伪造标签。标为requires_review不能计作正确预测。

## 冻结与复现

初次同步提交c2847e1的pipeline、model_backends、semantic_consistency与10月14日成功运行记录的SHA对应。后续CI修复统一排版、补充类型标注与说明，因此活动代码的文件SHA已变化。原始六份集成代码保存在archive/runtime_snapshot_20261014并记录SHA；历史门禁和原始结果未改写。旧实验复现应使用原提交或恢复独立工作副本，不能跳过旧SHA校验。GitHub原有采样、模型构造、mask回填及ROI属性配对说明继续保留。

批处理加速候选只在归档实验中；同步当前GPU主实现不等于启用批处理。正式替换前仍需同样本人工真值精度对照。
