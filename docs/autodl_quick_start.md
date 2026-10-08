# AutoDL 开始步骤

第一次使用分类整理版时，上传完整 `fashion_multimodal_analysis_GitHub_20260930.zip`。已有分类整理版可按小型修复包的 `README_先看这里.md` 安装；原先平铺 scripts 的工程使用完整包。

## 1. 建立工作副本

将 ZIP 上传到 `/workspace/` 后，在原 AutoDL 的 `(py312)` 环境执行。以下目标目录必须尚不存在，原工程保留。

```bash
python - <<'PY'
from pathlib import Path
import zipfile
source = Path('/workspace/fashion_multimodal_analysis_GitHub_20260930.zip')
destination = Path('/workspace/prd31_recovery')
if destination.exists():
    raise SystemExit('工作目录已存在。已解压则直接进入它；需要另一份副本时更换目标目录。')
with zipfile.ZipFile(source) as archive:
    for item in archive.infolist():
        if not (destination / item.filename).resolve().is_relative_to(destination.resolve()):
            raise ValueError('压缩包路径无效')
    archive.extractall(destination)
print('解压完成：', destination / 'fashion_multimodal_analysis')
PY
cd /workspace/prd31_recovery/fashion_multimodal_analysis
export FASHION_DATA_ROOT=/workspace/fashion_data
python -m pip install -e . --no-build-isolation
python scripts/maintenance/run_prd31_recovery.py --phase check
```

看到 `READY_FOR_GPU_AND_ASSETS` 后继续。如果显示 `BLOCKED_GPU_OR_ASSETS`，先看 `reports/reruns/recovery_v2/gpu_preflight.json` 中的具体缺项。

## 2. 后台运行并查看进度

在同一项目根目录执行：

```bash
mkdir -p reports/reruns/recovery_v2
nohup python -u scripts/maintenance/run_prd31_recovery.py --phase all > reports/reruns/recovery_v2/console.log 2>&1 &
```

之后用以下两条查看阶段和日志；这两条不会另开训练：

```bash
python scripts/maintenance/run_prd31_recovery.py --status
tail -n 40 reports/reruns/recovery_v2/console.log
```

`nohup` 使任务在关闭终端后继续运行；AutoDL 实例停机仍会停止任务。任务确实停止后，再运行原命令可从新版 checkpoint 的最近完整轮次继续。同一工程已有流程运行时，新流程会停止并提示查看状态。

如果当前 AutoDL 已安装 torch，可先执行一次恢复检查，包含小型 CPU 模型恢复实验，不运行服装训练：

```bash
python -m unittest discover -s tests -p test_checkpoint_state.py -v
```

## 3. 补人工 GT 和完成验收

看到 `HUMAN_GT_REVIEW_REQUIRED` 后，下载 `benchmark/prd_3_1_v2/review/prd31_review.html`，在本地浏览器填写真实属性与区域框，导出两个 CSV 放回同一 review 目录：

```text
grounding_reviewed.csv
attribute_reviewed.csv
```

再执行：

```bash
python scripts/maintenance/run_prd31_recovery.py --phase finish
```

随后下载本次 `acceptance/grounding_semantic_review.html`，审核 correct/coarse/wrong/missed。把导出的 `grounding_semantic_audit_reviewed.csv` 放回本次 acceptance 目录，执行：

```bash
python scripts/maintenance/run_prd31_recovery.py --phase report
```

最终查看 `reports/reruns/recovery_v2/acceptance/ACCEPTANCE_REPORT.md`。阶段代码已补齐；是否达到 PRD 数值目标，以真实重训、推理与人工审核后的报告为准。当前交付没有新模型成绩。
