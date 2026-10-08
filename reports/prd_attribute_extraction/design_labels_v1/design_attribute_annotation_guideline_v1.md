# PRD 3.1.3 人工审核标准 v1（2026-09-17）

## 目的
用于后续 3.1.3 人工抽样/全量复核，统一判断标准，降低人为因素。适用于 sleeve_length、neckline、silhouette_fit、fashion_style、pattern、primary_color。

通用规则：只根据图像可见信息判断；不确定允许标记 `ambiguous=1`；人工判断时不使用模型 score 反推 GT；正式评估前冻结本说明；条件允许时抽取部分样本由第二位审核者独立复核。

## 袖长 sleeve_length
- `sleeveless`：无袖，无明显袖管。
- `cap`：仅覆盖肩缘的小袖。
- `short`：袖口在上臂，明显高于肘部。
- `elbow`：袖口大致到肘部。
- `three_quarter`：肘部以下、手腕以上。
- `long`：到手腕附近。

边界：宽袖/泡泡袖仍按袖口终止位置判断；遮挡严重无法判断时标记 ambiguous。

## 领型 neckline
- `crew`：浅、贴近颈根的圆领。
- `v_neck`：明显 V 形。
- `round_scoop`：较深或较宽的圆弧领。
- `square`：明显方形。
- `boat`：横向较宽、较浅。
- `halter`：挂脖结构。
- `off_shoulder`：明显露双肩。
- `one_shoulder`：单肩非对称。
- `turtleneck`：高领覆盖大部分颈部。
- `mock_neck`：短立领。
- `polo`：Polo 式领口。
- `shirt_collar`：衬衫式翻领。
- `hooded`：有帽结构。
- `other`：可见但无法归入以上类别。

边界：头发、叠穿、遮挡导致领口无法确认时标记 ambiguous。最终 end-to-end 优先使用 3.1.2 neckline ROI。

## 版型 silhouette_fit
### top / outerwear
- `slim_fitted`：明显贴身。
- `regular_straight`：常规、较直。
- `loose_oversized`：明显宽松/oversized。

### pants
- `slim_tapered`：窄或向裤脚收窄。
- `straight`：上下宽度变化小。
- `wide_leg`：裤腿整体较宽。
- `flared`：裤脚相对膝部明显外扩。

### skirt
- `straight`：腰到下摆变化小。
- `a_line`：逐渐扩张。
- `flared`：下摆明显大幅展开。

### dress
- `fitted`：整体贴身。
- `straight`：纵向较直。
- `a_line`：下半部分逐渐扩张。
- `flared`：裙摆明显外扩。

若 `upper_geometry_quality=weak` 或 `trouser_geometry_quality=weak`，相关几何特征不作为可信依据；遮挡/姿态导致轮廓不可判断时标记 ambiguous。

## 风格 fashion_style
- `casual`：日常、轻松、实穿。
- `formal`：商务/正式感明显。
- `sporty`：运动或 activewear 风格明显。
- `streetwear`：城市街头/潮流视觉语言明显。
- `elegant`：精致、成熟、优雅。
- `minimalist`：线条简洁、装饰克制。
- `vintage`：明显复古视觉语言。
- `romantic`：柔美、浪漫、女性化装饰明显。

同一件服饰同时符合多个风格且无明显主导时标记 ambiguous；不要只凭模特姿势和背景判断风格。

## Pattern
水洗、褶皱、破洞、拼接、结构线本身不算真实图案。文字/Logo/插画 -> graphic_logo；条纹 -> striped；格纹 -> checked_plaid；花卉 -> floral；波点 -> polka_dot；动物纹 -> animal_print；迷彩 -> camouflage；无真实图案 -> solid。

## Primary Color
当前仅审核单一主色。两个或多个大面积颜色接近时标记 ambiguous；cream/ivory 归入 beige；secondary color 不混入当前单主色准确率。

## 审核字段
至少保留：source_image、garment_id、garment_category、attribute、predicted_label、manual_label、manual_correct、ambiguous、error_type、notes、reviewer。正式评估还应记录 guideline 版本、模型版本、数据版本、审核日期和 holdout seed。
