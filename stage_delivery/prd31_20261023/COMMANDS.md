# 候选复现命令与资产要求

使用现有py312环境。候选目录可换位置，但原工程源文件、配置、分割检查点和transformers实现会由入口核对指纹。保持固定输入与参数，出现指纹不匹配应先确认使用的工程版本，不能跳过校验。

## 默认模式与回退

默认white-sam复用白底和同ROI SAM视觉结果：

```bash
python -u /workspace/fashion_multimodal_analysis/stage_delivery/prd31_20261023/candidate/PRD31_W7_1022_FIRST/scripts/run_candidate.py --project /workspace/fashion_multimodal_analysis --image /workspace/fashion_data/raw/train/train/image/063238.jpg --mode white-sam --check-smoke-reference
```

white回退只保留白底复用，不启用SAM视觉缓存：

```bash
python -u /workspace/fashion_multimodal_analysis/stage_delivery/prd31_20261023/candidate/PRD31_W7_1022_FIRST/scripts/run_candidate.py --project /workspace/fashion_multimodal_analysis --image /workspace/fashion_data/raw/train/train/image/063238.jpg --mode white --check-smoke-reference
```

任意其他图片使用绝对路径替换 --image 并去掉 --check-smoke-reference。该开关仅适用于固定063238.jpg，不能用于任意新图。实际调用会做新推理，不从证据文件读取本次结果。默认输出每次创建新目录，不覆盖旧结果；导出与摘要检查位于计时外，单次runtime_observation不是正式性能基准。

## 必需资产

| 资产 | 固定位置或标识 |
|---|---|
| 原工程 | /workspace/fashion_multimodal_analysis，含src及相关属性与定位辅助文件 |
| V5权重 | outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth，SHA256 c5ba795225349b1c521aac0517ac7c493d9818dcc17d558ac4a3f6f8669452f0 |
| DINO离线模型 | IDEA-Research/grounding-dino-tiny，snapshot a2bb814dd30d776dcf7e30523b00659f4f141c71 |
| SAM离线模型 | facebook/sam-vit-base，snapshot 70c1a07f894ebb5b307fd9eaaee97b9dfc16068f |
| CLIP离线模型 | openai/clip-vit-base-patch32，snapshot 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 |
| HF目录 | 工程下hf_cache/hub，包含配置、处理器、权重及需要的tokenizer文件 |
| 参考图 | /workspace/fashion_data/raw/train/train/image/063238.jpg，SHA256 dee7dcd6c7d14a3f5bdf7fb8e4fc1ed98421e0b9c9bb144d02ddd839fdd1ef86 |

离线模型发现与处理器加载的已测记录在reference_environment/model_cache_audit.json和processor_audit.json。资产存在性与完整模型字节认证不是同一检查：本包未复制或完整哈希所有HF大模型权重，迁移时需独立核对。

## GitHub 源码检查

在仓库根目录运行：

```bash
python stage_delivery/prd31_20261023/verify_sources.py --project .
```

该命令核对候选原清单与 11 个活动依赖，不加载模型或核对所有权重。完整阶段 ZIP 另有 verify_delivery.py 和原始审计附件；这些二进制附件、PPT、稿子、原图与模型权重未发布到本次 GitHub 更新。
