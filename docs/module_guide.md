# 模块职责与阅读路线

活动实现位于 `src/fashion_multimodal_analysis/`，`scripts/` 只做入口转交。所有活动 Python 模块、类、函数都补充文档，公共参数与返回值补充类型提示。后端对象使用 Any 的位置仍需结合相应模型库理解，不能把它当作所有类型已严格证明。

## 3.1.1 实例分割

`segmentation/training/` 保留各版本。V4 清理已知重叠；V5 将已选图片的目标实例补齐；V6 扩展覆盖；V7 针对 Fashionpedia 类别头；V8 扩展原始训练范围并无放回遍历。它们是不同实验，不把旧目录改名冒充新版本。

V8 的 `discover_df2` / `discover_fp` 发现数据源，`protected_inventory` 建立受保护图片身份，`prepare` 生成 SQLite 和范围说明，`NativeDataset` 按需读图/解码标注，`epoch_order` 产生每轮确定性次序，`train` 保存断点与完整统计，`evaluate` 保存固定协议对照。这些函数的 Args/Returns/Raises 和关键状态说明位于实现中。

`datasets/prd8.py` 负责 CSV → 图片级实例集合；同图 target 是多个 bbox、labels、masks，而不是每个实例独立读取整图。`image_processing/masks.py` 将 crop mask 放回原图；缩放和边界裁剪必须一起处理，否则掩码与 bbox 的坐标会错位。

`segmentation/sampling.py` 的早期类别均衡采样，作用于图片的抽样权重；不要把 WeightedRandomSampler 的样本数说成已无放回看完所有图片。V8 采用 shuffle_without_replacement，两种采样口径不同。

`evaluation/metrics.py` 计算 bbox/mask IoU 和固定匹配统计。空掩码、退化框、无匹配类别都需要显式处理；类别错误的定位匹配不计类别正确 TP。报告同时给定位召回、类别正确召回和条件均值，避免只报告有利指标。

`evaluation/runtime.py` 先在 GPU 按 score 筛选，再搬运保留的输出到 CPU，减少整批低分掩码传输。计时同步包住 preprocess + forward + postprocess，排除读盘和 GT；与只测模型 forward 的时间不能直接比较。加速验证工具继续校验固定模型和固定测试集身份。

## 3.1.2 局部区域定位

`grounding/baselines/` 保留提示、ROI 和检测基线；`grounding/refinement/` 做衣物条件下的区域调整；`grounding/core_parts/` 训练和推理 core-part detector；`grounding/evaluation/` 比较目标、可见性和阈值。

局部预测的坐标必须明确来自原图还是 garment crop。将 crop 中的 ROI 放回原图时需要恢复偏移及缩放，再裁剪到衣物/图像范围。`integration/pipeline.py` 连接前一步的衣物实例和后一步局部结果；单步使用 GT ROI 的成绩不能直接作为端到端预测 ROI 的成绩。

`common/schema.py` 定义共享字段/类别约定，`integration/model_backends.py` 封装各模型输出，`integration/run_prd31_acceptance.py` 按指定输入与报告目录执行集成协议。先读这些文件的接口文档，再阅读具体实验。

## 3.1.3 属性提取

`attributes/color/` 使用服饰区域内像素提取颜色；`attributes/geometry/` 提取长度、形状及轮廓特征；`attributes/pattern/` 保留分层花纹版本及修正；`attributes/design/` 保留图案/设计输出和人工审计。

属性值应能追溯到衣物实例、局部区域及版本。可见性不足、掩码为空或某项不适用时，按实现返回的空值/状态处理，不自动当成一个有效负例。比较 v1/v2/v3 时同时检查标签定义、人工审核和测试划分。

## 数据、审核和工具

`benchmarking/` 负责候选、质量检查、冻结和审核包；`analysis/` 汇总分布、错误和数据重叠；`visualization/` 生成标注/结果对照；`data_tools/` 整理输入数据；`maintenance/` 做恢复、打包、资产检查和保存结果的索引。

AI 预审核和教师/人工最终审核是不同来源。`benchmark/prd_3_1_v2/ai_pre_review/` 与 `ai_annotations/` 保留前者，不据此擅自升级为正式验收 GT。`benchmark/prd_3_1_v2/review/prd31_review.html` 使用取得的较完整上传版本。

## 逐模块和逐接口查找

全部分类入口见 [commands.md](commands.md)，公共及内部接口的位置见 [api_documentation_index.csv](repository_organization/api_documentation_index.csv)。文件重命名前后映射见 [file_name_map.csv](repository_organization/file_name_map.csv)，来源和冲突保存见 [source_provenance.md](source_provenance.md)。

核心算法参考保存在 `archive/reference_algorithms/*.py.txt`，来自上传前的文件字节。测试先核对原 AST 指纹，再剥除 docstring/type hints 比较活动算法的执行结构，防止补说明时悄悄改变 IoU、采样或模型构建逻辑。
