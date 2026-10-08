# 来源、合并与版本保留

本工作副本以取得的 `fashion_multimodal_analysis_GitHub_20260930.zip` 为分类基础，补充 `fashion_sources_restored_20260930.zip`、PRD31 fix 包、单独上传的 configs/scripts/src/docs/benchmark/data/models/outputs 等有用内容，以及 V4–V8 的补丁、诊断、速度修复与结果。

新增纳入 V8 全量运行脚本、完整训练与准备摘要、排除图片清单、最终工作流、V5/V8 在开发集和 Core400 上的四组对照。V6 coverage 和诊断、V7 FP heads、V5 full targets 及 speed fix 均单独保留。PRD31 审核 HTML 采用取得的更完整单独上传版本，较早版本作为冲突副本保留。

不同路径内容冲突时，未静默覆盖唯一版本：原差异保存在 `archive/source_variants/`，冲突记录见 `docs/repository_organization/merge_conflicts.json`。早期 `alibaba-ai-main(1).zip` 属于另一套 fashion_mm 实现，留在 `archive/previous_repository/`，不混进当前八类 Mask R-CNN 入口。

补丁内源代码保留在 `archive/result_source_code/` 或相应快照目录。归档 Python 改为 `.py.txt`，表示它是原始参考，避免与活动源码同名时误运行。活动源码位于 src，入口位于 scripts，文档已按本次规范补齐。

## 精确恢复的 V5 清单

V6 训练清单在 V5 的 2,039 条记录后追加覆盖实例。恢复其前 2,039 行，使用原字段顺序、UTF-8 BOM 与 CRLF 序列化后，得到与原回执完全一致的 SHA：

`36ff78831b20e492d787c66161f05747433f8e85f7a089fd5df981c2bdf476e6`

恢复文件是 `configs/prd_8class_train_v5_full_targets.csv`，475 张图片，包括 395 张 Fashionpedia 图片。它不是猜测标签的重建。完整开发集清单和派生掩码缺失时，没有从预测结果补造 GT。

## 仍不可取得的模型/文件

V5/V8 权重和 V8 SQLite 不在已取得结果包内。单独较早 checkpoint 的下载失败也有记录；结果、日志和哈希原样保留，但缺字节的文件不被标记为已恢复。详见 [GPU 指南](gpu_restore_and_run.md)。

`source_file_inventory.csv` 保留上传包内文件来源，`file_name_map.csv` 记录整理名，最终 `package_contents_sha256.csv` 记录本次交付文件的实际字节。前两份属于来源视图，不等于每个原始重复文件都需在当前目录再次复制一遍。

此次没有在 GitHub push，也没有在新 GPU 上重跑训练。所有关于已完成训练的数字都来自上传的真实结果文件。
