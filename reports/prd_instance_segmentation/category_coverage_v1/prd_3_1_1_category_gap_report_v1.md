# PRD 3.1.1 类别覆盖检查 v1

## 当前 156 个 garment instances

| PRD 类别 | 当前数量 | 状态 |
|---|---:|---|
| 上衣 top | 52 | 已覆盖 |
| 裤子 pants | 39 | 已覆盖 |
| 裙子 skirt | 16 | 已覆盖 |
| 外套 outerwear | 13 | 已覆盖 |
| 连衣裙 dress | 36 | 已覆盖 |
| 鞋子 shoe | 0 | 缺失 |
| 包包 bag | 0 | 缺失 |
| 配饰 accessory | 0 | 缺失 |

当前 156 个实例只能覆盖 PRD 3.1.1 的 5/8 大类。

## 下一步

不要继续在 DeepFashion2 当前 156 个实例里“硬找”鞋子、包包和配饰，因为当前数据源本身没有这三类正式样本。

建议保留现有 DeepFashion2 五类主干，同时增加一个补充数据源，只用于：
- shoe
- bag
- accessory

优先候选：Fashionpedia。它提供实例分割 mask，并包含 shoe、bag/wallet 以及多种 accessory 类别，因此与 3.1.1 的 mask + bbox + category 输出形式比较匹配。

## 建议训练/验证策略

1. DeepFashion2 五类保持不动，不重做。
2. Fashionpedia 只抽取缺失三类做补充。
3. 统一映射成 PRD 8 类，不直接保留 Fashionpedia 的全部细分类。
4. 先做小规模 pilot，确认数据读取、mask 映射和类别映射都没问题，再决定是否扩大。
5. 正式训练时按 8 类统一输出：
   - mask
   - bbox
   - category label
   - confidence
