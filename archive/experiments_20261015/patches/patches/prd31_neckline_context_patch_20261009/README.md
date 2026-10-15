# 10.9 领型上下文候选

## 本次修改

昨日领型只使用局部 SAM mask 涂白后的紧裁剪。新候选同时分类该原输入和扩展后的原图裁剪，保留领口附近皮肤、边界及衣料。两种裁剪标签冲突时，输出 `requires_review`、空正式 label 和两个候选，不把模型分歧当作部位不存在。

扩展倍数 2.0，最小宽度/高度为服装框的 25%/15%，均为未经校准的开发参数。即使两种输入一致，语义正确性仍为 pending。CLIP 分数不是准确率。

这是领型候选修正；本包未修复袖长、图案、分类混淆或领口定位语义错误。错误的领口仍可能导致错误的领型。默认配置不开启新行为，只有本包候选运行脚本显式开启；controlled GT 路径保留旧行为。

## 在 GPU 执行

上传 ZIP 到 `/workspace/fashion_multimodal_analysis`，确保 py312 环境已激活，再执行聊天中的解压和运行命令。运行入口为 `run_patch_20261009.py`。它会设置昨日缓存路径，安装前核对已有 backend 哈希并备份，先 dry-run，后运行固定 8 图。

dry-run 会按原图、模型权重、训练和验证清单哈希找到昨日成功的 guard-regression 目录。找不到时不会加载模型。若目录移走了，可直接执行 `PYTHONPATH=src python scripts/run_prd31_neckline_context_20261009.py --baseline-dir <原目录>`。

安装前版本不匹配时不覆盖；需要提供提示以核对代码。备份目录在 `reports/patch_backups/neckline_context_*`。

运行入口用于首次安装。已经安装后再次运行回归，应直接执行 `PYTHONPATH=src python scripts/run_prd31_neckline_context_20261009.py`，并保留 HF_HOME/HF_HUB_CACHE 与 offline 环境设置。

## 输出与验证

结果保存到 `reports/reruns/recovery_20261009/prd31_neckline_context_<时间>`，新目录不会覆盖旧报告。最后输出 `UPLOAD_THIS_ZIP`，把该 ZIP 上传即可。

包含预测、原图、两种领型裁剪以及 `baseline_comparison.json`。比较会核对实例类别/框、服装与部位掩码及其他属性，单独统计领型分歧和旧裁剪标签漂移。任何非领型变化都需要复核，不能默认为本次改善。

固定 8 图已用于开发，结果不是独立盲测或 PRD 验收。候选正式启用前需人工复核原图、裁剪和领口归属。

本地验证：11 项裁剪及 backend 分支测试通过；原有 10 项几何保护测试另行复跑。测试使用 mock，不验证真实模型识别能力。GPU 实验由用户运行，尚未获取结果。
