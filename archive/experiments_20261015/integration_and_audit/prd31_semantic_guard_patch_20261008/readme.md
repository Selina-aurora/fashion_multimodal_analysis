# PRD3.1 第一轮局部定位保护补丁

基于 2026-10-08 上传的 model_backends.py / pipeline.py 制作；不修改训练权重、不读取审核答案来替换模型类别。

## 已改内容

- 按分数排序 DINO 候选，对越界候选继续尝试下一项。
- 严格模式下，局部部位候选超过服装裁剪面积的 0.65 时拦截，保留诊断原因。阈值是可调开发规则，不是验收阈值；可能增加漏检，需比较召回。
- collar/cuff/hem/pocket/shoulder/waist 的 SAM 输出同时约束在候选框和预测服装掩码内，阻止整衣扩张。pattern/decoration 不使用这一局部框限制。
- 重叠实例标记人工复核，保留实例；叠穿不能仅凭框重叠删除。
- 预测无袖时添加袖口适用性复核提示，不自动把 cuff missed 当作已确认漏检或不存在。
- 固定原来的 8 张输入图片及 SHA-256；导出 review_flags、几何诊断和带原图的结果 ZIP。

## 在 GPU 上使用

在当前项目根目录（有 src、scripts、configs）打开终端，把 ZIP 上传到这里，执行：

```bash
unzip -q prd31_semantic_guard_patch_20261008.zip
python prd31_semantic_guard_patch_20261008/install_patch.py --project-root .
PYTHONPATH=src python scripts/run_prd31_review_regression_20261008.py --dry-run
PYTHONPATH=src python scripts/run_prd31_review_regression_20261008.py
```

安装脚本先核对上传时两个文件的哈希，存在差异则停止；已有文件保存在 reports/patch_backups。仅在 PATCH_INSTALLED 后继续。

若 dry-run 显示 DRY_RUN_PASSED，继续最后一条。完成后上传终端 UPLOAD_THIS_ZIP 指向的文件。

本脚本沿用今天 V5 balanced150 checkpoint、train_manifest 和 val_manifest 的校验及离线模型缓存要求。

## 验证及限制

10 项模型外回归测试通过：候选排序、拦截整衣级局部框、SAM 边界、原图坐标、保留分层实例、无袖复核提示等。未在此环境执行 GPU 模型推理。测试中的模型/语义依赖使用 mock，不能作为模型精度验证。

本轮不能修复所有上游类别错误、裤腿拆分、误判图案、不可见属性或包遮挡。几何合法也不代表语义正确，SAM 框约束也不保证部位边界正确。重跑后仍需原图审核。

这 8 张图已经用于开发诊断，只作为回归集；正式 PRD 验收仍需冻结标注及独立评估数据。
