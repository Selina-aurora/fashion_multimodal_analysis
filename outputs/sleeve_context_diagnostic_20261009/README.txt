袖长上下文对照：保留旧白底遮罩基线，对比原图衣服框和每侧扩10%的原图框。
固定模型、提示词、标签和关联失败；不训练、不修改生产配置。
本地已验证裁剪与分母逻辑；GPU推理待运行。
GPU：
cd /workspace/fashion_multimodal_analysis
python -m zipfile -e sleeve_context_diagnostic_20261009.zip outputs
PYTHONPATH=src python outputs/sleeve_context_diagnostic_20261009/run_sleeve_context_diagnostic_20261009.py
上传 UPLOAD_THIS_ZIP 后的ZIP。
