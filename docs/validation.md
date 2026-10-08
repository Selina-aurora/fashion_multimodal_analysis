# 整理验证记录

验证日期：2026-09-30。验证目标包括目录重构、协议修复、流水线逻辑和证据完整性；没有运行真实 GPU 训练/推理，也不声明科学验收通过。

## 对话恢复后的新增验证

本次追加 10 项检查，总计 40 项：**39 项通过、1 项跳过**。新增覆盖跨实验 checkpoint 拒绝、未完成模型不能正式验收、恢复所需随机/采样状态和日志完整性、保存失败保留旧 checkpoint、只读状态、同工程进程互斥及子进程错误日志。真实 PyTorch 小模型恢复检查因当前没有 torch 而跳过，已提供 AutoDL 执行命令。

本次机器记录和完整日志在 `reports/repository_audit/continuation_20260930/validation/`；下表保留前次整理版的验证记录。

| 检查 | 结果 |
|---|---|
| Python 语法编译 | 386 个源码/入口/测试文件通过 |
| 分类命令的静态帮助 | 170/170 从其他工作目录启动成功；不加载模型 |
| 内部包 import 目标 | 全部指向存在的模块/包 |
| CPU 回归测试 | 30 项通过，包括临时合成数据上的冻结→流水线→报告检查 |
| 安装包构建 | 离线 wheel 构建成功，包含命令注册表和实现 |
| 原始证据字节校验 | 5,570 个 benchmark/reports/configs/outputs/data 文件 SHA-256 相同 |
| 当前说明文档链接 | 检查 19 个主要文档，仓库内链接可达 |
| 单文件大小 | 无超过 100 MiB 的文件；大型权重与 ZIP 工作包已排除 |
| 常见私钥/令牌格式扫描 | 未发现匹配；不是完整安全审计 |
| 原数据集与 checkpoint | 未包含于源码证据包，资产检查按预期报告缺失 |

CPU 测试覆盖 IoU 几何/空集、掩码还原与越界裁剪、采样权重、根目录与数据目录迁移、运行入口异常后的工作目录恢复、GPU 工作包运行时复制，以及源图交集审计。抽取的共用定义与上传版本 AST 指纹一致，保留原算法表达式。

最新机器结果：[checks.json](../reports/repository_audit/recovery_v2/validation/checks.json)；最新测试日志：[unittest.txt](../reports/repository_audit/recovery_v2/validation/unittest.txt)。先前整理验证保留在 validation 目录。

## 资产检查的解释

| 清单 | 独立图片/掩码路径总数 | 当前未附带数量 |
|---|---:|---:|
| B1 train_v3 | 1,088 | 963 |
| 固定 val_v1 | 75 | 44 |
| 冻结 Core400 | 800 | 400 |

这些数量来自本包的实际路径检查：Core 的 400 份掩码已保留，400 张外部源图需从原数据目录恢复；训练/验证集还引用外部 processed 掩码。权重也需按原路径恢复。详见 [external_assets.txt](../reports/repository_audit/validation/external_assets.txt)。

## 验证限制

当前验证环境有 numpy/Pillow/pandas，没有 torch、torchvision、transformers 和 GPU 实验所需外部数据/权重。因此，没有执行模型工厂、训练 DataLoader 全流程、checkpoint 加载、模型推理和速度复现；静态帮助通过不代表这些流程已经跑通。

旧 train_v3 的交集审计仍返回 `OVERLAP_FOUND`，该历史证据没有被改写。新 train_v4_clean 已排除全部 Core/val 路径交集。它没有完成内容级去重，也没有重新训练，因此不能解释为新模型已经通过或 Core 已恢复盲测独立性。

## checkpoint 加载兼容修正

修正了 17 个历史加载入口的默认模式，使工程自产完整 checkpoint 中的 NumPy 随机状态可被读取。新增真实 Core 入口加载回归检查。当前共 41 项检查：39 项通过、2 项因本地缺少 torch/torchvision 而跳过；AutoDL 真实模型继续评估的结果待回传。本包未包含用户刚训练的权重。
