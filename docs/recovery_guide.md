# PRD 3.1 修复版运行指南

这版已经修复代码和数据清单问题，并提供 GPU 重训、人工审核、冻结及完整验收入口。当前交付没有新增真实模型成绩。原始报告和 v1 Core400 保留；新实验写到 `reports/reruns/recovery_v2/`，新权重写到 `outputs/prd_instance_segmentation/maskrcnn_8class_v4_clean/`。

## 先在 AutoDL 放好项目

完整 ZIP 用于首次整理和 GitHub 上传；如果已经使用上次整理版，可使用小型修复 ZIP，按照包内说明安装到该整理版。原先平铺脚本的工程需要先使用完整 ZIP 建立整理版工作目录。

在 AutoDL 保留原数据目录 `/workspace/fashion_data` 和原权重目录。解压完整包到新的工作目录，例如 `/workspace/prd31_recovery/fashion_multimodal_analysis`；避免覆盖旧实验记录。基础示例：

```bash
cd /workspace/prd31_recovery/fashion_multimodal_analysis
export FASHION_DATA_ROOT=/workspace/fashion_data
python -m pip install -e .
python scripts/maintenance/check_project.py
python scripts/maintenance/run_prd31_recovery.py --phase prepare
```

原工程中 `outputs/garment_instances/` 等原图裁剪、mask 文件，以及 `/workspace/fashion_data/processed/` 的训练掩码，需要在工作副本中按原相对路径可访问。完整包已保留用户上传的 outputs；未上传的大型数据仍从原数据目录读取。脚本会逐项检查并导出缺失文件列表，不会跳过缺失样本开始训练。原绝对工程路径和 Windows 分隔符已统一解析到当前工程。

模型运行需要原 AutoDL 环境中的 torch、torchvision、transformers 及 CUDA；这版不会更换你的 CUDA 依赖组合。预检查会实际调用 NMS，发现 torch/torchvision 不匹配时停止。Mask R-CNN COCO 初始化和 Grounding DINO、SAM、CLIP 的预训练模型需要已缓存或可下载。模型与依赖下载不计入模块延迟。

## 第一次运行：干净重训和 Core 回归

```bash
python scripts/maintenance/run_prd31_recovery.py --phase all
```

它依次准备数据、检查外部文件和 GPU、训练 v4-clean 15 epochs、执行 Core400 回归、生成原图人工审核页面。若缺人工 GT，状态为 `HUMAN_GT_REVIEW_REQUIRED`，不会把空表当作验收通过。已完成且哈希、epoch 和参数相符的 checkpoint 可复用；续改版生成的未完成 checkpoint 通过完整状态校验后，自动从下一轮继续。其他已有 checkpoint 不会被覆盖。

先检查环境、查看运行进度：

```bash
python scripts/maintenance/run_prd31_recovery.py --phase check
python scripts/maintenance/run_prd31_recovery.py --status
```

`check` 只检查依赖、原图、掩码与 CUDA，不训练、不改人工 GT；`status` 只读取最后保存的阶段、训练轮次和最近子进程日志。完整缺失文件列表保存在 `gpu_preflight.json`，终端仅显示前 8 项。训练和 Core 评估子进程使用无缓冲输出，日志追加到 `reports/reruns/recovery_v2/workflow.log`；超过 30 秒没有新输出时会显示子进程仍在运行。阶段文件为 `workflow_progress.json`。AutoDL 同一个工程只能同时启动一条修复流程。

断点恢复粒度是**已完成的 epoch**。新版 checkpoint 同时保存模型、SGD、学习率计划、Python/NumPy/PyTorch/CUDA 随机状态、加权采样器状态和已有逐轮日志。写入失败会保留上一个完整 checkpoint。训练中途停止后，再运行原 `--phase all` 或 `--phase gpu` 命令即可；本轮未完成的 batch 会重跑。恢复要求清单、类别、参数和可见 CUDA 设备数量一致；不承诺跨硬件或依赖版本的逐位一致。旧版未完成 checkpoint 因缺少随机与采样状态，会明确停止，需保留它并在草稿配置中选择新的输出目录。

正式验收要求完成全部 15 轮；直接调用验收入口也会检查两个清单哈希与训练设置。已有验收报告若对应另一个 checkpoint，不会被当前模型复用，需选择新的 `--report-dir`。只整理已有报告仍可用 `--phase report`，无需重新加载模型。

v4-clean 从原 train_v3 排除所有与冻结 Core/固定 val 相同源图路径的记录：580→547 实例，508→475 源图，八类均保留。训练设置沿用 B1 的模型、15 epochs、采样、优化器、分辨率、翻转和 seed，以便观察去掉源图重叠后的变化。减少重叠不保证模型指标提高；新结果以真实报告为准。

查看：

- `reports/reruns/recovery_v2/gpu_preflight.json`：缺失文件、依赖及 CUDA 状态。
- `reports/reruns/recovery_v2/train_v4_clean/`：训练过程和数据检查。
- `reports/reruns/recovery_v2/core_v4_clean/protocol_metrics.json`：新匹配与计时口径的 micro/macro/per-class 指标。
- `benchmark/prd_3_1_v2/review/prd31_review.html`：下一步人工真值审核页面。

## 一次集中补齐人工真值

从 AutoDL 下载 `prd31_review.html`，本地浏览器直接打开。页面内嵌图片，标注无需联网。

候选集为 40 件服装（每类 5），240 个属性行中 160 行适用；局部区域为八类、每类 10 个候选，共 80 个查询，审核后每区域固定取前 5 个有效正例。选样只依据人工真值，不依据模型是否预测成功。不存在目标、看不清或多义样本均保留明确原因；有效正例不足时必须补新候选并建立新草稿，不能伪造 GT 或降低覆盖要求。

若某区域不足 5 个明确正例，先把已经导出的两个 reviewed CSV 放回 review 目录，再执行（以 pocket 为例）：

```bash
python scripts/benchmarking/prepare_prd31_recovery.py --extra-grounding 10 --region pocket
```

它保留旧候选，追加未使用源图，并重建含已有审核答案的页面。只审核新增项后重新导出。不能在 frozen 已存在后追加。

逐图填写审核人，给适用属性选择真实标签；选择局部区域后，在原图上拖拽画目标框，点击保存。切换区域、图片时也会保存。多主色、风格无法唯一确定等情况标“歧义”并说明原因。袖长、领口仅对 top/outerwear/dress；轮廓、风格仅对五类服装；颜色、图案覆盖八类。

导出两个 CSV，上传回 AutoDL 的 `benchmark/prd_3_1_v2/review/`：

```text
grounding_reviewed.csv
attribute_reviewed.csv
```

当前交付中的页面只有上传材料里的染色 GT 预览，原图未附带时会禁用标注。必须在原数据环境运行 prepare/all 重建原图页面，再审核颜色。浏览器缓存可能被清理，建议导出进度 JSON 作备份。

## 冻结并运行完整验收

```bash
python scripts/maintenance/run_prd31_recovery.py --phase finish
```

脚本检查所有候选身份、原图坐标、审核人、类别适用性和覆盖；通过后建立 `benchmark/prd_3_1_v2/frozen/` 及 SHA-256。冻结后不覆盖 GT。它随后执行：

- 分割：Core400，按置信度排序的类别无关一对一 bbox 匹配，再评价类别和 mask。
- Track A：GT 服装 ROI/mask 下的局部定位和人工 GT 属性准确率。领口使用固定几何代理裁剪，单独记录输入来源。
- Track B：处理全部预测实例，3.1.1 的预测 ROI/mask 进入 3.1.2，领口属性使用 3.1.2 的预测领口 mask/ROI。匹配只用于事后评价，预测阶段不接收 GT。
- 完整分母：分割漏检、类别错误、局部漏检和属性缺失均进入报告，歧义及不适用项按冻结规则排除。
- 延迟：模型先加载；10 次预热、5 次计时，CUDA 同步；计入预处理、模型和后处理，排除磁盘加载/写报告；按 image、query-instance、instance 分别记录。

输出到 `reports/reruns/recovery_v2/acceptance/`。已有该次运行报告时不重复推理；比较另一模型时使用新的 `--report-dir`，并保留对应 checkpoint 和冻结协议。

## 严格定位语义审核与最终报告

打开本次报告目录的 `grounding_semantic_review.html`，查看绿色 GT 框、红色预测框，判断 correct/coarse/wrong/missed。这一步针对模型是否找对语义位置；GT 框和 bbox IoU 不能替代语义审核。coarse 不能算成功。导出的 CSV 上传回本次报告目录，然后运行：

```bash
python scripts/maintenance/run_prd31_recovery.py --phase report
```

它只更新统计，不重新运行模型。最终阅读 `ACCEPTANCE_REPORT.md`、`acceptance_gates.csv` 和 `end_to_end_failure_stages.csv`。没有严格定位审核时，已检测案例仍保持待评价；实际漏检可以直接计为失败。

PRD 目标保持：mean mask IoU≥0.85/分割≤50ms；严格定位≥92%/≤30ms；人工 GT 属性 micro≥88%/≤20ms。Track B 和端到端另报，不新增端到端数值门槛。CPU/mock 检查不作为 GPU 达标证明。

## 已有人工结果与结论边界

颜色已有 40 例审核，37 例明确单主色中正确 32，86.49%；图案 40 例正确 34，85%。保留为 `benchmark/prd_3_1_v2/development/` 的历史开发证据；这些样本已有研究/调参历史，不重新包装成新盲测。

新训练清单解决已知源图路径交集，但尚未做内容级去重，Core 也已经用于先前模型诊断。这一轮是固定协议回归；正式未见数据泛化结论仍需要独立采集的测试集。代码流程完整并不意味着 3.1 数值目标已经达到。
