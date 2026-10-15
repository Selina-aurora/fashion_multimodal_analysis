通过原生掩码重叠提出袖子归属候选。所有同图上身/全身衣服参与比较，未选入清单的衣服也参与。
阈值固定为第一覆盖>=0.8、第二<=0.2，仅生成待审核草稿；不训练、不自动确认。
训练集每个草稿类别最多8件，最多48件；验证集不进入审核训练样本。
脚本已编译，标签映射冲突逻辑已验证，实际掩码关联待GPU运行。

cd /workspace/fashion_multimodal_analysis
python -m zipfile -e sleeve_training_candidates_20261009.zip outputs
PYTHONPATH=src python outputs/sleeve_training_candidates_20261009/prepare_sleeve_training_candidates_20261009.py

完成后上传 UPLOAD_THIS_ZIP 后面的ZIP。
