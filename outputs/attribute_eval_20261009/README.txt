18张属性诊断评估：主色18、图案18、袖长15；3项未知袖长排除。
保持用户确认文件原样。第2张素色/袖条纹备注保留，另输出排除该张的图案敏感性结果。
真实框+真实mask用于属性诊断；V5预测框+mask用于带定位误差的结果。定位缺失计错，不删除失败样本。
仅使用既有CLIP属性方法，不训练、不修改生产代码、不加载袖口模型。
原分割验证集上的18张小样本，不能作为正式PRD验收。
本地已验证标注与指标分母逻辑并编译；实际GPU推理待运行。

GPU执行：
cd /workspace/fashion_multimodal_analysis
unzip -o attribute_eval_20261009.zip -d outputs
PYTHONPATH=src python outputs/attribute_eval_20261009/run_attribute_eval_20261009.py

完成后上传终端 UPLOAD_THIS_ZIP 后面打印的ZIP。
