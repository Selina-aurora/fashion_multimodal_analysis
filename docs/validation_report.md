# 最终交付验证报告

交付日期：2026-10-08。检查使用 Python 3.12.14，对象是本次整理后的活动代码和可取得的原始结果。

## 规范及入口

| 检查 | 实际结果 |
|---|---|
| Black 88 字符格式 | 408 个 Python 文件通过 |
| isort 分组导入 | 通过 |
| Ruff 基本错误、相邻字符串、文档及类型提示规则 | 通过 |
| mypy 共享核心模块 | 5 个模块通过；其余后端接口仍采用渐进类型检查 |
| 可编辑安装 | 通过，安装入口在项目外目录可列出 179 命令 |
| 分类入口静态 --help | 179 / 179 通过 |
| 文档覆盖 | 408 个模块、2042 个类/函数定义全部有文档 |
| 公共函数参数/返回提示 | 1944 个公共函数通过 |
| Markdown 项目内文件链接 | 旧名引用修复后无未解析链接 |

开发工具版本：black 26.10.0, isort 9.0.2, ruff 0.16.10, mypy 2.4.0, pytest 9.1.1。`requirements_dev.txt` 使用这些实际检查过的版本，CI 使用 Python 3.12。

## 核心与文件身份

单元测试运行 47 项，45 项通过，2 项因缺少 PyTorch 跳过。覆盖 IoU/掩码回填、类别采样、匹配顺序、ROI 往返、验收分母、审核冻结、断点状态、工作流锁、原算法执行结构、冻结清单身份及 V8 逐 GT 指标一致性。

跳过项原始记录：

```text
验证 Core 入口在模型初始化前正确加载含 NumPy 随机状态的断点。 ... skipped 'PyTorch/torchvision are not installed; check the actual Core loader in AutoDL'
验证真实 PyTorch 小模型的续跑与不中断训练一致。 ... skipped 'PyTorch is not installed; actual training-state restore must be checked in AutoDL'
```

V4–V8 七个结果包中 324 个报告/配置/输出成员与当前主文件的原始字节一致。Core400 清单与恢复的 V5 清单 SHA 保持历史身份。12 个抽取算法首先验证上传参考的 AST 指纹，再去除注释与类型提示比较执行结构；筛选后传输 runtime 的函数也与上传候选保持执行结构一致。

历史 runtime SHA：`60021df022b16fbe42c88ed57635e180cfd397077803403cbb6513169daabaca`。

注释版 runtime SHA：`ff85ca020695498ab287d73e8afd915b44345ecd2168b38eb24410fb8ddddfb8`。

现有历史回执没有改写，新的训练脚本身份检查使用注释版 SHA；新运行目录记录新身份。原文件比对详情、入口检查、接口索引和运行日志位于 `docs/repository_organization/`。

## 压缩包内容与边界

GitHub ZIP 和 GPU ZIP 的共同文件一致；GPU ZIP 额外包含取得的三个 core-part detector 权重。两个包均包含 V8 训练摘要、准备/排除清单、最终状态、四组 V5/V8 对照及早期有用结果。

打包时排除 Python 字节码、工具缓存、虚拟环境和安装元数据。根目录 `package_contents_sha256.csv` 给出实际文件 SHA、大小及分别属于哪个 ZIP；清单自身不自引用计算 SHA。ZIP 生成后检查 CRC、安全路径、成员数量、V8 必需文件及 GitHub/GPU 文件差异，检查记录随交付保留。

此次未在 GPU 重跑训练或计时；旧速度来自上传的 RTX 4090 回执。V5/V8 权重、SQLite、完整开发集清单以及外部原始数据/派生掩码的字节缺失已在恢复指南中列出。结果记录齐备不代表已恢复这些训练资产，也不表示正式验收通过。
