导出8件训练集无袖类别候选，核对后才可用于训练。已有37件有效人工标签保持原样；3件未知跳过。
排除之前40件及全部验证图片的文件哈希，不混入18张属性诊断。
本地语法检查通过；候选提取待GPU数据运行，不训练。

cd /workspace/fashion_multimodal_analysis
python -m zipfile -e sleeveless_supplement_20261009.zip outputs
PYTHONPATH=src python outputs/sleeveless_supplement_20261009/prepare_sleeveless_supplement_20261009.py
上传 UPLOAD_THIS_ZIP 后面的ZIP。
