# GitHub 上传说明

GitHub ZIP 保存源代码、配置、测试、文档、基准资产、预测输出和取得的所有结果，包括 `reports/reruns/recovery_v2/v8_full_dataset/`。其中不打包大模型权重；GPU ZIP 多出取得的三个 `.pt`。

先解压，把项目目录中的内容放进新仓库或现有仓库的新工作分支。不要把整个交付 ZIP 当成源代码提交，也不要覆盖唯一的原始 GPU 工作目录。

新仓库示例（在解压后的项目根目录）：

```bash
git init
git add .
git status --short
git commit -m "docs: organize internship experiments through full dataset training"
git branch -M main
git remote add origin <你的GitHub仓库URL>
git push -u origin main
```

现有仓库使用新分支，并先检查差异：

```bash
git switch -c docs/internship_archive_20261008
git add .
git diff --cached --stat
git commit -m "docs: document full dataset training and preserve experiment results"
git push -u origin docs/internship_archive_20261008
```

上述提交与 push 由你在目标仓库执行，本次整理没有操作任何远端仓库。仓库 URL 占位符必须替换为你的真实地址。

`.gitignore` 忽略权重、原始数据、环境、缓存及临时归档，已经移除旧规则对整个 `reports/reruns/` 的忽略，确保全量结果会进入 Git。可检查：

```bash
git check-ignore reports/reruns/recovery_v2/v8_full_dataset/training_summary.json
git ls-files reports/reruns/recovery_v2/v8_full_dataset
```

第一条应没有忽略规则匹配；第二条在 git add 后应能看到 V8 文件。模型采用独立传输或 Git LFS 等适当方式管理，不将不同模型重命名成同一个 checkpoint。

GitHub 对普通 Git 大文件的现行规则见官方说明：[About large files on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。本交付的普通单文件大小也在最终验证报告中核对；上传整个 ZIP 与上传其中的普通文件是不同操作。

展示入口建议为根目录 `readme.md`、`docs/full_dataset_training_v8_report.md`、`docs/results_index.md` 和实际 V8 原始结果。不要将 AI 预审核写成老师已经审核，也不要把 Core 回归写成独立盲测验收。
