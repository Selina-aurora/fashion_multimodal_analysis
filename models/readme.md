# 模型文件说明

GitHub ZIP 不打包权重；GPU ZIP 包含取得的 `core_part_detector_v1/last_model.pt`、`best_loss_model.pt`、`best_model.pt`，它们属于局部区域检测。

V5/V8 八类分割 checkpoint 不在已取得备份内。最终训练及评估回执保留它们的 SHA，但不能靠回执恢复模型字节；不把本目录的 core-part 权重改名当成分割模型。

当前资产状态见 [GPU 指南](../docs/gpu_restore_and_run.md) 及 [只读检查](../scripts/maintenance/check_assets.py)。旧上传权重记录 [weight_inventory.csv](weight_inventory.csv) 是当时的来源视图；最终 ZIP 内容以 `package_contents_sha256.csv` 为准。
