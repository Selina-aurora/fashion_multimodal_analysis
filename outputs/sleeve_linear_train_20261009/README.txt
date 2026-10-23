44件有效人工审核训练来源，6类袖长。冻结CLIP，只训练512维线性分类头。
按原图哈希分组划分内部训练/开发；开发集选epoch，18张既有诊断图仅在选好后评估。
原图衣服框无扩边输入；训练集做水平翻转，验证不做增强。150epoch仅分类头。
不替换正式系统；小样本和既有诊断图结果不能当作盲测验收。
本地已验证真实44件的分组划分和指标逻辑，并编译；GPU特征提取和训练待运行。

cd /workspace/fashion_multimodal_analysis
python -m zipfile -e sleeve_linear_train_20261009.zip outputs
PYTHONPATH=src python outputs/sleeve_linear_train_20261009/train_sleeve_linear_20261009.py

上传 UPLOAD_THIS_ZIP 后的ZIP，含小型分类头权重，不含CLIP大模型权重。
