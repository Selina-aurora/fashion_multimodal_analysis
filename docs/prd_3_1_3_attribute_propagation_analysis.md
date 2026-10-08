# PRD 3.1.3 属性传播：区分模型与分母

本节测量从 GT-ROI 切换为 predicted-ROI 后，属性算法输出是否一致。GT-ROI 上的属性输出仍是算法结果，不能作为人工属性真值。缺失、不适用属性不进入该属性的可比分母。

## B1 pilot v2

模型来源：[predicted_roi_pilot20_v2/summary.txt](../reports/prd_integration/predicted_roi_pilot20_v2/summary.txt)，checkpoint 为 V3-B1 Data Expansion epoch 15。文件夹名虽然有 pilot20，实际运行记录为 selected_images=32；bbox50 匹配 39、bbox50 且类别匹配 24，最终属性分析使用 12 个配对。

12 对类别为 top 3、pants 4、skirt 1、outerwear 1、dress 3，没有 shoe/bag/accessory。来源：[predicted_roi_attributes_v2](../reports/prd_integration/predicted_roi_attributes_v2/summary.txt)。

## V2-balanced fullval_v1

模型来源：[predicted_roi_fullval_v1/summary.txt](../reports/prd_integration/predicted_roi_fullval_v1/summary.txt)，checkpoint 明确为 baseline_v2 epoch 15。32 张验证图片、bbox50 匹配 30、bbox50 且类别匹配 22，最终有 22 个 ROI 配对。

22 对中 DeepFashion2 18、Fashionpedia 4；top 7、pants 5、skirt 0、outerwear 1、dress 5、shoe 1、bag 2、accessory 1。可用属性基准输出减少了分母：常规属性为 18 对，袖长/领口为 13 对。不能把 22 作为每一属性的分母。

## 一致性结果

| 属性 | B1 v2 pilot | V2-balanced fullval_v1 |
|---|---:|---:|
| primary_color | 12/12 = 100.00% | 15/18 = 83.33% |
| pattern | 7/12 = 58.33% | 14/18 = 77.78% |
| sleeve_length | 4/7 = 57.14% | 8/13 = 61.54% |
| neckline | 1/7 = 14.29% | 7/13 = 53.85% |
| silhouette_fit | 9/12 = 75.00% | 16/18 = 88.89% |
| fashion_style | 9/12 = 75.00% | 11/18 = 61.11% |

两个实验的模型、配对集合与样本规模不同，不能直接把列间差值解释成某种方法的提升。原始汇总分别见 [v2 CSV](../reports/prd_integration/predicted_roi_attributes_v2/attribute_agreement_summary.csv) 和 [fullval CSV](../reports/prd_integration/predicted_roi_attributes_fullval_v1/attribute_agreement_summary.csv)。

现有证据支持对 ROI 误差传播进行分析，还不足以给出完整八类属性准确率或产品验收结论。材质、面料、工艺不属于当前阶段已完成范围。
