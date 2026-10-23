查原始Fashionpedia袖子部件标注及其长度属性，避免只查整件服装而漏标。
不训练、不修改标签。相同图片中的袖子不能自动认作某件衣服的袖子；所有关联候选原样导出。
脚本语法检查已通过，实际原始部件数量待运行。

cd /workspace/fashion_multimodal_analysis
python -m zipfile -e sleeve_part_audit_20261009.zip outputs
PYTHONPATH=src python outputs/sleeve_part_audit_20261009/run_sleeve_part_audit_20261009.py
