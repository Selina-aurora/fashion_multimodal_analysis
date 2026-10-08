# Core 评估 checkpoint 加载修正

2026-09-30 AutoDL 反馈：v4-clean 的 15 epochs 已训练完成；Core 评估在 torch.load 默认 weights_only 模式下遇到 NumPy 随机状态反序列化错误。本次仅修正工程自产 checkpoint 的读取方式，保留权重、数据清单与报告。

本版源码已将相关入口显式设置 weights_only=False。已有旧工作副本可将 fix_checkpoint_loading.py 上传至项目根目录，再执行：

```bash
cd /workspace/prd31_recovery_20260930/fashion_multimodal_analysis
python fix_checkpoint_loading.py --project .
export FASHION_DATA_ROOT=/workspace/fashion_data
nohup python -u scripts/maintenance/run_prd31_recovery.py --phase all >> reports/reruns/recovery_v2/console.log 2>&1 &
```

修复脚本先备份源文件，再修改工程内缺少显式加载参数的入口；重复执行会跳过已经修正的文件。修复流程会校验并复用已完成 15 轮的 v4-clean checkpoint，继续 Core400 评估和重建原图审核页面。

参考：PyTorch 官方 serialization notes，https://docs.pytorch.org/docs/2.14/notes/serialization.html#torch-load-with-weights-only-true 。这种读取用于本工程自产的完整训练 checkpoint。
