# 10月22日第一轮推进包

已接入昨天真实GPU验证的白底特征复用＋SAM视觉复用作为单图入口默认模式，保留white模式回退。新增图片采样与编译筛选分阶段运行。11项本地CPU替身测试通过，真实GPU入口、新样本回归、编译结果尚待本轮运行，不能提前写成完成。

## AutoDL运行

在原py312环境，把ZIP放到/workspace：

```bash
cd /workspace
python -m zipfile -e PRD31_W7_1022_FIRST.zip /workspace
python -u /workspace/PRD31_W7_1022_FIRST/scripts/run_1022.py
```

无需安装或联网下载模型，使用原工程、离线缓存和现有数据。编译需要原torch环境的Inductor/Triton依赖，若缺失、输出变化或未真实执行编译图，则记录失败，不自动安装或放宽阈值。结束后上传最后UPLOAD_THIS指向的ZIP。子阶段日志同步显示并保存在输出目录；异常、中断和超时保留已有证据，打包部分结果。

## 依次执行四步

1. **新增20张回归候选**。从现有训练图片所在的原始数据目录抽取没有出现在已发现项目训练/验证清单及已知历史开发集合中的图片。排除训练/验证源图、Core400、旧袖长/开发批次及W7封存身份；封存图片名称在读取字节前跳过。对可访问的已知源图做SHA和dHash近重复筛查，近重复距离不超过4则排除；新图在任何模型推理前固定。生成images.json、预览、review.html和待填写的scene_review.csv。
2. **独立单图CLI检查**。新进程从原模型缓存初始化，运行scripts/run_candidate.py默认white-sam模式，对063238.jpg的完整预测对照已冻结输出，并逐个核验导出的mask。调用与092427完全相同的完整链路函数；新增包装只保留原始返回用于导出。原white_reuse.py、sam_vision_reuse.py、sam_reuse_logic.py逐字节保留并校验SHA。
3. **新增图片配对回归**。若选足20张且入口通过，另起进程，在新固定图片上对比white与white-sam两版，检查完整输出一致、实例/查询/属性及导出mask，保存实际模型编码次数。2次/版预热后只做单轮配对；耗时是观察值，不替代昨天固定24图的5轮正式测速，也不计算语义精度。
4. **核心编译加速筛选**。另起进程，固定选择64条原始适用查询，按局部区域×原始返回状态分层，不根据编译结果挑样本。DINO forward和SAM视觉编码尝试Inductor编译；每次都重新计算视觉/文本特征，没有跨查询嵌入缓存，dtype、batch、模型、像素和查询不改。10次/版预热、64对预检查、5轮×64对活动墙钟计时；固定数值容差1e-4，mask摘要精确相同，任何漂移停止。

首次编译期间可能较慢。本轮编译阶段默认最多20分钟，届时结束该阶段并保留日志，不影响已有SAM复用候选。新增图片回归阶段也设20分钟上限，单图入口10分钟。若只先跑交付入口与新增图片：

```bash
python -u /workspace/PRD31_W7_1022_FIRST/scripts/run_1022.py --skip-compile
```

## 对新增样本的独立性说明

当前V5训练与验证清单SHA固定。自动发现现有工程及archive中遵循prd_8class_train*.csv和prd_8class_val*.csv命名的清单，输出清单路径、SHA及排除数量。若其他历史训练使用不同命名或未保留清单，脚本无法自动认证。缺少已知源图时，字节/近重复审计标为不完整。

因此本轮始终称“新增回归候选”，完整历史隔离和场景复核待确认，不提前称正式独立盲测。DeepFashion2/Fashionpedia来源也不意味着DINO/SAM/CLIP预训练从未见过。模型输出一致不等于定位/六属性正确；若要精度报告，还需对应人工GT。场景覆盖由review.html和scene_review.csv核对，抽样本身不保证多实例、遮挡、宽松等类别均已补齐。

W7封存清单仅用于身份字段排除，不使用/导出标签，不打开封存图片。找不到或无法解析8张身份时，新采样停止，交付入口仍可检查；不擅自打开封存集凑数。可用--w7-identities指定原身份清单，或--pool-root多次指定已有原始图片文件夹。

## 核心编译计时的边界与限制

计时沿用085326的活动墙钟定义：GPU像素/文本输入就绪，模型视觉/文本计算、候选筛选和原CPU mask生成计入；规定的动态SAM输入准备、数据拷贝/物化明确暂停，终点在ROI局部bbox/二值mask生成后、原图canvas格式化前。不是纯GPU算子或连续全GPU窗口。

编译只缓存代码与执行计划，每个请求的图像/文本嵌入均重算，SAM视觉编码实际调用数仍逐次检查。本轮保持Tensor包装和模型分派函数在整个配对实验中的身份稳定，避免每次更换包装导致额外Dynamo guard失效。支持部分编译区域，但每条候选查询必须实际执行DINO编译图，需要SAM的查询还必须执行SAM视觉编译图；仅调用torch.compile后实际回退eager的情况不能当成功。任何正式计时期间新增编译都计入时间并单列记录。

64条属于诊断筛选，其构成与原527条总体不同。必须比较本轮相同64条的eager/compiled两版，不能把筛选均值直接与旧139.63ms相减，不能据此填写全527条30ms指标达标。若后续确有一致且稳定收益，可按预定参数覆盖完整527条：

```bash
python -u /workspace/PRD31_W7_1022_FIRST/scripts/core_compile_probe.py --count 527
```

编译结果不会自动成为交付默认。原SAM复用完整请求3.59秒/图及原核心139.63ms记录保留。PyTorch自定义编译后端官方接口说明：https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_custom_backends.html 。当前GPU实际兼容性以运行结果为准。

## 日常单图入口及回退

默认white-sam，独立于实验prior/benchmark/profile文件夹，仅依赖原工程、原离线模型和本包的固定配置：

```bash
python -u /workspace/PRD31_W7_1022_FIRST/scripts/run_candidate.py --image /workspace/fashion_data/raw/train/train/image/063238.jpg
```

回退原白底复用版：

```bash
python -u /workspace/PRD31_W7_1022_FIRST/scripts/run_candidate.py --image /workspace/fashion_data/raw/train/train/image/063238.jpg --mode white
```

输出prediction.json、masks、runtime_observation.json、环境/版本与summary。单次观察值不是正式性能测速；初始化、模型资产完整性和输出检查在本轮GPU回传后确认。临时适配器仅适用于单进程顺序调用，未做并发服务化验证，生产源码不修改。

## 回传与当前状态

summary.json和stage_status.json分别记录入口、新样本、编译阶段真实状态。任何一个阶段失败不会被其他阶段的成功覆盖。选样不足、训练清单无法解析、缺模型资产、编译超时或数值漂移都保留实际原因；不宣称全部完成或全部PRD通过。11项CPU替身测试覆盖缓存入口、mask导出、封存排除、重复图片、编译分派/实际执行计数、原方法恢复、分层选样、漂移门槛和子进程超时清理；这些检查不使用真实torch/CUDA。
