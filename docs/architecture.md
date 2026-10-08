# 代码结构与迁移

## scripts 和 src 如何分工

`scripts/` 负责可发现的命令入口：定位项目、引入包、将参数交给对应实现。算法函数、模型构建、数据读取与实验流程位于 `src/fashion_multimodal_analysis/`。新增算法优先写在可复用模块中；新增实验可以组合这些模块，入口仍保持简短。

原有 164 个 Python 脚本中，3 个 `_backup` 版本归档为 `.py.txt`；其余 161 个保留实验身份并分类。新增 `audit_dataset_overlap` 和 `check_project`，总计 163 个入口。GPU/图片诊断用的旧 `test_*` 已改为 `diagnose_*` 或 `check_*`，避免和自动化测试混淆。原实验的 v1/v2/fixed 标识保留，便于对应历史输出。

| scripts / src 子目录 | 职责 |
|---|---|
| `data_tools` | 数据抽样、标注转换、实例清单与训练集构建 |
| `image_processing` | 通用图像裁剪与掩码处理 |
| `segmentation/training` | PRD8 训练与训练消融 |
| `segmentation/evaluation` | 验证、Core 回归、阈值扫描与结果对比 |
| `segmentation/baselines`、`diagnostics` | Mask2Former 基线及小目标诊断 |
| `grounding/baselines`、`evaluation` | 提示词、ROI 定位基线与评估 |
| `grounding/refinement`、`core_parts` | 空间/SAM 等诊断及局部检测器 |
| `attributes/color`、`pattern` | 主色与图案识别 |
| `attributes/geometry`、`design` | 连续几何、轮廓与款式标签 |
| `integration` | 预测 ROI、配对与属性传播分析 |
| `benchmarking` | 数据审计、人工复核包与 benchmark 建立工具 |
| `analysis`、`visualization` | 结果汇总、误差分析、可视化 |
| `maintenance/packaging` | GPU 工作包；同时复制 src、scripts、pyproject |

## 已抽取的共用模块

| 模块 | 内容 |
|---|---|
| `common/paths.py` | 仓库根目录、数据根目录和历史路径转换 |
| `common/schema.py` | 固定 PRD8 类别编号（背景为 0） |
| `datasets/prd8.py` | manifest 读取、按源图分组、训练 Dataset、collate |
| `image_processing/masks.py` | 二值掩码、裁剪掩码还原到整图 |
| `segmentation/sampling.py` | 实例类别频次与图像组采样权重 |
| `segmentation/modeling.py` | 标准训练/评估模型工厂；anchors 消融保留专用实现 |
| `evaluation/metrics.py` | bbox IoU 与 mask IoU |
| `cli.py`、`command_registry.json` | 命令发现、静态帮助和惰性加载 |

本轮只合并与参考定义 AST 完全一致的 85 处重复定义，没有统一改写全部历史实验。`refactor_record.json` 保存原始函数 AST 指纹，测试校验抽取前后算法定义一致。项目仍是研究工程，实验模块中保留版本特有的常量和流程。

## 路径约定

默认项目与 `fashion_data/` 是同级目录。`FASHION_PROJECT_ROOT` 可指定包含 `configs/` 的仓库根目录，`FASHION_DATA_ROOT` 可指定外部数据目录。共用路径解析兼容历史 `/workspace/fashion_multimodal_analysis/` 和 `/workspace/fashion_data/` 前缀。通过分类脚本或 `fashion-analysis run` 运行时，工作目录自动切换到项目根目录，并在结束后恢复。

历史报告中的服务器绝对路径保留为溯源信息；它们不代表本机当前存在该文件。个别较早的交互/示例脚本依赖固定图片或人工审查表，需先看实现中的输入常量。默认同级数据布局对所有历史脚本最直接；自定义数据布局请先执行资产检查。

## 迁移方式

完整对照表：[script_migration.csv](script_migration.csv)。例如：

```bash
# 原命令：python scripts/train_prd_8class_maskrcnn_v3_b1_dataexp.py ...
python scripts/segmentation/training/train_prd_8class_maskrcnn_v3_b1_dataexp.py --help
# 安装后也可使用原命令名：
fashion-analysis run train_prd_8class_maskrcnn_v3_b1_dataexp --help
```

避免从旧 scripts 目录直接 `import xxx`；新的跨模块引用使用 `fashion_multimodal_analysis.<功能>.<模块>`。`src/` 下的实现文件用于包导入，运行入口使用 scripts 或 CLI。

PRD 3.1.2 汇总生成器原来把结果写进 scripts 根目录，现写到 `reports/prd_region_coverage/final_summary_v1/`。三种历史 GPU 打包命令已同步复制整个代码运行时，避免打出的包仅包含无法独立运行的薄入口。
