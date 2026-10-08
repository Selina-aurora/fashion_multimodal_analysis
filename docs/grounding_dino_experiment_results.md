> 历史阶段记录：正文保留当时的实验背景；当前模型归属、指标修正和限制见项目根 README 与 docs/corrections.md。文中的运行入口已按新目录调整。

Grounding DINO Baseline Experimental Results

1. Experimental Setup

（1）Model

Model: Grounding DINO Tiny

Framework:

1）PyTorch

2）Hugging Face Transformers

（2）Task Definition

This experiment evaluates the capability of vision-language grounding models for language-guided local region localization in fashion images.

The objective is to investigate whether a pretrained vision-language model can localize clothing regions based on flexible natural language descriptions instead of fixed predefined categories.

（3）Input and Output

Input:

1）Fashion image

2）Natural language prompt

Examples:

1）"shirt"

2）"the sleeve of the shirt"

3）"blue denim shirt"

4）"left sleeve"

Output:

The model generates:

1）Target region bounding box

2）Confidence score

3）Predicted semantic label



2. Prompt Difficulty Design

To evaluate localization ability under different levels of semantic complexity, prompts were divided into several categories.

| Level | Prompt Type | Examples | Purpose |

|---|---|---|---|

| Easy | Category-level | shirt | Overall garment localization |

| Medium | Clothing part | sleeve, collar | Common clothing component localization |

| Hard | Fine-grained detail | button, zipper | Small object localization |

| Hard | Sub-part | sleeve cuff | Fine-grained region understanding |

| Medium-Hard | Attribute description | blue denim shirt | Attribute understanding |

| Hard | Spatial description | left sleeve | Spatial relationship understanding |

The prompt design aims to evaluate whether the model can generalize from simple clothing categories to more flexible language descriptions.

3. Single-image Prompt Evaluation

（1） Category-level Prompt

Prompt：shirt

Observation：

The model can identify general clothing regions when the semantic category is clear.

However, fashion images contain diverse garment categories, such as jackets, dresses, and tops. Therefore, category ambiguity may influence localization performance.

（2）Clothing Part Prompt

&#x20;Prompt：sleeve

Observation：

The model shows relatively stable localization ability for common clothing components.

The results indicate that Grounding DINO can effectively utilize visual-language alignment for recognizable clothing structures.

（3）Fine-grained Detail Prompt

Prompts

1）button

2）zipper

3）sleeve cuff

Observation：

Performance decreases for small-scale clothing details.

Possible reasons:

1）Limited target size

2）Weak visual features

3）Difficulty distinguishing fine-grained structures

These results indicate that additional segmentation refinement may be required for precise clothing detail extraction.

（4）Attribute-based Prompt

Prompt：blue denim shirt

Observation：

The model can utilize additional semantic information such as color and material descriptions.

However, attribute understanding does not always guarantee precise localization, and predicted regions may remain coarse.

（5）Spatial-aware Prompt

Prompt：left sleeve

Observation：

The model successfully identifies the sleeve concept.

However, the spatial constraint "left" is not always strictly reflected in the predicted region.

This indicates that the model has semantic understanding but limited spatial reasoning ability.



4. Threshold Sensitivity Analysis

（1）Experimental Setting

The same image and prompt were tested with different confidence thresholds.

Prompt：sleeve

（2）Results

| Threshold | Detection Result | Observation |

|---|---|---|

|0.2|Detected|Valid detection retained|

|0.3|Detected|Stable result|

|0.4|Detected|Stable result|

|0.5|Not detected|Prediction filtered out|

（3）Analysis

The detection result remains stable when the threshold is between 0.2 and 0.4.

When the threshold increases to 0.5, the prediction is removed.

This indicates that overly strict confidence filtering may reduce recall and cause missed detections.



5. Multi-image Evaluation

（1）Purpose

To evaluate whether Grounding DINO can generalize beyond a single image, a multi-image testing pipeline was developed.

Pipeline:

Image Input

↓

Natural Language Prompt

↓

Grounding DINO Inference

↓

Bounding Box + Confidence Score

↓

Result Saving

（2）Sleeve Localization Evaluation

Prompt: sleeve

Number of images: 10

Results:

| Metric | Value |

|---|---|

| Number of images | 10 |

| Successful detections | 5 |

| Detection rate | 50% |

Observation：

The model successfully detects sleeves in images with clear sleeve structures.

Failures mainly occur in:

1）sleeveless garments

2）unclear clothing structures

3）challenging viewpoints

The results indicate that Grounding DINO can recognize common clothing components but remains sensitive to visual ambiguity.

（3）Shirt Localization Evaluation

Prompt: shirt

Observation：

Compared with part-level prompts, category-level prompts show larger variation.

Possible reasons:

1）Different garment categories have overlapping visual characteristics

2）Fashion images contain diverse clothing styles

This demonstrates that prompt semantics significantly influence grounding performance.



6. Overall Findings

Based on the experiments, several observations are summarized.

（1）Semantic Grounding Ability

Grounding DINO can associate natural language descriptions with clothing regions.

（2）Prompt Complexity Influence

Localization performance decreases as prompt complexity increases.

Category-level

↓

Part-level

↓

Fine-grained detail

↓

Spatial relationship

（3）Current Limitations

The baseline model has limitations in:

1）Small object localization

2）Fine-grained clothing details

3）Precise spatial relationship understanding

（4）Future Improvement Direction

Future work will investigate:

1）Segmentation refinement methods

2）Grounding + segmentation pipelines

3）More comprehensive multi-image evaluation

4）Automatic evaluation metrics



7. Future Work

The next stage will focus on integrating grounding detection with segmentation models.

Expected pipeline:

Image

↓

Grounding DINO

↓

Bounding Box Localization

↓

Segmentation Model

↓

Fine-grained Clothing Region Mask



8. Grounding DINO Zero-shot Localization Results

A preliminary evaluation was conducted using six natural language prompts with different levels of clothing semantic granularity.

The prompts were divided into three difficulty levels:
1）Level 1: Basic clothing category
shirt
2）Level 2: Structural components
sleeve
collar
3）Level 3: Fine-grained details
button
sleeve cuff
zipper

| Prompt | Type | Detection Rate | Average Confidence |

|---|---|---|---|

| shirt | Category | 27.27% | 0.3436 |

| sleeve | Structural part | 54.55% | 0.3654 |

| collar | Structural part | 36.36% | 0.3746 |

| button | Fine detail | 9.09% | 0.3093 |

| sleeve cuff | Fine detail | 0.00% | - |

| zipper | Fine detail | 0.00% | - |

The results show that Grounding DINO can better localize relatively visible clothing structures such as sleeves and collars, while performance decreases significantly for small-scale details such as buttons, sleeve cuffs, and zippers.

This indicates that zero-shot vision-language localization models have limitations when dealing with fine-grained fashion attributes, especially when targets occupy only a small image region or are affected by occlusion and image quality.

Therefore, the following experiments will focus on improving fine-grained clothing understanding through additional segmentation and visual feature enhancement methods.

