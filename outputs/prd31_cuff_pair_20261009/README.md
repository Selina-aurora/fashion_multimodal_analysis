# 袖口单变量对照开发实验

包含 5 张袖口可见的目视草稿正例和 3 张目标服装无袖的草稿负例。先打开 review_drafts.html 复核橙色框与黄色目标实例。标注尚未人工确认，不用于正式验收或训练。照片来自已有审核 HTML 中内嵌 JPEG，保留原嵌入字节；没有宣称它们与数据集原始文件字节相同。

原方案读取当前项目 REGION_PROMPTS["cuff"]；新方案只改为 `short sleeve cuff . long sleeve cuff .`。两组复用分割 ROI、权重、阈值、短语过滤、几何保护及 SAM。全部预测服装先推理，之后才用目标服装框进行评价匹配。提示词在 finally 中恢复；不修改产品源文件，不自动上线。

## 执行

把 zip 放到 /workspace/fashion_multimodal_analysis，然后执行：

```bash
cd /workspace/fashion_multimodal_analysis
unzip -q prd31_cuff_pair_20261009.zip -d outputs
PYTHONPATH=src HF_HUB_CACHE=/workspace/hf_cache/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python outputs/prd31_cuff_pair_20261009/scripts/run_cuff_pair.py
```

结果末行打印 RESULT_ZIP=绝对路径。将该结果 zip 发回。每张对照图：黄色=目标服装，橙色=待复核袖口草稿，绿色=原方案，蓝色=新提示词。脚本对五个草稿正例保留固定分母；分割漏检单独列出。负例只报告检出数，不把漏检或拒绝算作正确不存在。

本地已验证图像哈希、尺寸、框范围、脚本语法和提示词恢复逻辑。GPU 对照尚未运行；不能据此宣称效果提高。时间为少量顺序调用耗时，不是 PRD 延迟验收。
