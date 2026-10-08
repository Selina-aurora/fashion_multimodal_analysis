Grounding DINO Baseline Evaluation for Language-guided Clothing Region Localization



1\. Objective

The objective of this experiment is to evaluate the capability of Grounding DINO as a baseline model for language-guided local region localization in fashion images.

Different from traditional clothing segmentation methods that rely on predefined categories, this experiment investigates whether a vision-language grounding model can locate clothing regions based on flexible natural language descriptions.



The model takes:

Input:

1）Image

2）Natural language prompt

Output:

1）Candidate bounding box

2）Confidence score



2\. Experimental Setup

2.1 Model

Baseline Model: Grounding DINO

Grounding DINO is selected as the first baseline because it supports open-vocabulary object detection and allows users to specify target regions through natural language descriptions.

Advantages:

1）Supports text-guided detection

2）Does not require predefined clothing categories

3）Can potentially generalize to unseen clothing concepts

4）Provides bounding box localization results



2.2 Test Image

A fashion image containing a denim shirt is selected for preliminary evaluation.

The purpose of this experiment is not to measure final model performance, but to analyze the localization ability and limitations of open-vocabulary grounding under different prompt granularities.



3\. Prompt Design and Difficulty Levels

To evaluate different levels of clothing understanding, six natural language prompts are designed.

The prompts are grouped according to localization difficulty.

（1）Level 1: Overall Clothing Category

Prompt: shirt

Description:

This prompt represents a high-level clothing concept.

The target region is relatively large and visually obvious, which tests whether the model can understand and localize general garment categories.

Expected difficulty: Easy

（2）Level 2: Common Clothing Components

Prompt: sleeve，collar

Description:

These prompts represent common structural components of clothing.

Compared with the whole garment, these regions require the model to understand the internal structure of clothing.

Expected difficulty: Medium

（3）Level 3: Fine-grained Clothing Details

Prompt: button，sleeve cuff`，zipper

Description:

These prompts describe small-scale or detailed clothing elements.

They require stronger visual understanding because the targets are smaller, less salient, and sometimes require structural reasoning.

Expected difficulty: Hard



4\. Test Prompts Summary

| Prompt | Difficulty Level | Target Type | Description |

|---|---|---|---|

| shirt | Easy | Whole garment | Overall clothing region |

| sleeve | Medium | Clothing component | Common garment part |

| collar | Medium | Clothing component | Neckline structure |

| button | Hard | Fine-grained detail | Small clothing element |

| sleeve cuff | Hard | Fine-grained sub-part | Specific sleeve region |

| zipper | Hard | Fine-grained detail | Narrow clothing component |

| blue denim shirt | Medium | Attribute + Category | Clothing with visual attributes |

| left sleeve | Hard | Spatial + Part | Part with location constraint |



5\. Preliminary Results and Observations

（1）shirt

Observation:

The model can recognize the general clothing category.

However, the predicted bounding box is relatively large and includes surrounding areas.

This indicates that Grounding DINO has strong semantic understanding of general garment concepts, but localization precision is limited.

（2）sleeve

Observation:

The model provides a relatively better localization result compared with other prompts.

This suggests that common and visually distinctive clothing components are easier for the model to identify.

（3）collar

Observation:

The model shows some understanding of the collar region.

However, the bounding box is still coarse and may cover a larger upper-body area.

This indicates that part-level localization remains challenging.

（4）button

Observation:

The model has difficulty accurately locating buttons.

Possible reasons:

1）Small object size

2）Limited visual information

3）Similar appearance with surrounding textures

This demonstrates the challenge of fine-grained clothing detail localization.

（5） sleeve cuff

Observation:

The model performance decreases when the prompt becomes more specific.

Compared with "sleeve", "sleeve cuff" requires understanding a smaller sub-region and more detailed clothing structure.

（6）zipper

Observation:

The model fails to consistently localize the zipper region.

This suggests that narrow and low-visibility clothing details remain challenging for the current grounding baseline.

（7）Attribute-based Prompt

Prompt: blue denim shirt

Observation:

The model successfully matches the visual attribute information with the clothing category.

Compared with the simple category prompt "shirt", the attribute-based description obtains a higher confidence score.

This indicates that Grounding DINO can utilize additional semantic information such as color and material descriptions during grounding.

However, the predicted bounding box remains relatively coarse, suggesting that attribute understanding does not necessarily guarantee precise local localization.

（8）Spatial-aware Prompt

Prompt: left sleeve

Observation:

The model successfully identifies the sleeve concept from the prompt.

However, the output label is generalized as "sleeve" rather than explicitly maintaining the spatial constraint "left".

The predicted region is still relatively broad, indicating that while the model understands the target concept, precise spatial reasoning remains challenging.

（9）Threshold Sensitivity Analysis

The effect of different confidence thresholds was evaluated using the same image and the prompt "sleeve".

| Prompt | Threshold | Detection Result | Confidence |

|---|---|---|---|

| sleeve | 0.2 | Detected | 0.485 |

| sleeve | 0.3 | Detected | 0.485 |

| sleeve | 0.4 | Detected | 0.485 |

| sleeve | 0.5 | Not detected | - |

Observation：

The detection result remains stable under thresholds from 0.2 to 0.4.

However, when the threshold increases to 0.5, the detection is removed.

This indicates that overly strict confidence filtering may reduce recall and cause missed detections.



6\. Overall Analysis

The preliminary experiment shows that Grounding DINO demonstrates different localization capabilities under different prompt granularities.

The model performs better on:

1）Whole garment categories

2）Large and visually distinctive clothing components

However, performance decreases for:

1）Small clothing details

2）Specific sub-parts

3）Fine-grained structural descriptions

These observations indicate that open-vocabulary grounding provides flexible semantic localization ability, but additional refinement methods may be required for precise clothing region extraction.



7\. Future Work

Based on the preliminary results, future experiments will focus on:

（1） Soft Keyword Evaluation

Instead of only using fixed category names, more natural descriptions will be tested.

Examples:

Easy: "the blue denim shirt"

Medium: "the folded collar area"，"the left sleeve"

Hard: "the fabric near the wrist"，"the zipper on the jacket"

This will evaluate whether the model can understand more flexible human descriptions.

（2） Threshold Analysis

Different confidence thresholds will be tested to analyze:

1）Detection stability

2）False positive cases

3）Missing detection cases

（3） Segmentation Refinement

Since Grounding DINO only provides bounding boxes, future work may combine grounding models with segmentation models to obtain more accurate local masks.

Possible pipeline:

Image + Text Prompt  

↓  

Grounding DINO  

↓  

Bounding Box Localization  

↓  

Segmentation Model  

↓  

Precise Region Mask



