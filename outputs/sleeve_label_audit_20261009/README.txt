只读检查现有袖长监督来源，不训练，不改现有标签。
Fashionpedia原生属性按实例ID+图片名+类别关联；原样保留属性定义，待核对后才映射六类袖长。
DeepFashion2服装细类仅统计弱标签提示，不能当作已审核真值。训练/验证分开。
不需要重新标注，不加载模型，不需要显卡。
本地已通过语法和ID/字段识别检查；GPU现有原生属性文件是否齐全待实际运行。

cd /workspace/fashion_multimodal_analysis
python -m zipfile -e sleeve_label_audit_20261009.zip outputs
PYTHONPATH=src python outputs/sleeve_label_audit_20261009/run_sleeve_label_audit_20261009.py

上传 UPLOAD_THIS_ZIP 后面的ZIP。
