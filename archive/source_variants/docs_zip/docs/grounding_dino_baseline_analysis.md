Grounding DINO Baseline Analysis

1\. Background

For the language-guided local region localization module (3.1.2), the goal is not to perform fixed-category clothing part segmentation. Instead, the system should support open-ended natural language descriptions and locate corresponding regions in fashion images.

Compared with predefined clothing part segmentation, language-guided localization provides better scalability because new concepts can potentially be introduced through natural language descriptions without requiring additional category definitions and retraining.

The expected input is:

Image + Natural Language Description

The expected output is:

Target Region Bounding Box + Confidence Score



2\. Candidate Baseline Selection

Among existing vision-language models, Grounding DINO is selected as the first-stage baseline for language-guided local region localization.

The main reason is that Grounding DINO is designed for open-vocabulary object detection and can directly associate natural language descriptions with visual regions.

The basic workflow is:

Image+Natural Language Prompt

&#x20;       ↓

Grounding DINO

&#x20;       ↓

Candidate Bounding Boxes+Confidence Scores

&#x20;       ↓

Region Selection

&#x20;       ↓

Local Region Output



3\. Why Choose Grounding DINO

（1）Support for Open Vocabulary

Unlike traditional object detection models that rely on predefined classes, Grounding DINO can use arbitrary text descriptions as input.

Examples:

"the left sleeve cuff"

"the floral pattern on the shirt"

"the zipper of the jacket"

"the overlap between jacket and inner shirt"

This matches the requirement of 3.1.2, where the input should not be limited to fixed clothing parts.



3.2 Better Scalability

For fixed part segmentation:

Adding a new concept such as: button，zipper，embroidery

usually requires: new annotation data，category modification，model retraining

For language-guided localization:The model can potentially leverage pretrained vision-language knowledge to understand new descriptions.



4\. Comparison with Other Models



| Model | Main Capability | Input | Output | Advantages | Limitations |

|---|---|---|---|---|---|

| Grounding DINO | Open-vocabulary localization | Image + Text | Bounding Box + Confidence | Direct text-guided region detection; supports arbitrary descriptions | Mainly outputs bbox; fine-grained mask requires additional segmentation |

| CLIP | Vision-language similarity matching | Image Region + Text | Similarity Score | Strong semantic alignment ability | Does not directly provide location |

| DINOv2 | Visual feature extraction | Image | Visual Feature Embedding | Strong visual representation ability | No direct text-image alignment |



5\. Preliminary Evaluation Plan

The evaluation will not restrict the system to predefined clothing categories.

Different types of natural language descriptions will be tested.

Part Description

Examples: collar，zipper，button，sleeve cuff

Attribute Description

Examples: floral pattern on the shirt，embroidered area，logo on the chest

Spatial Description

Examples: left sleeve，right pocket，lower part of the skirt

Relationship Description

Examples: area where jacket overlaps with inner shirt，button near the collar



6\. Future Improvement

The first stage focuses on validating the complete localization pipeline.

After the baseline is established, possible improvements include:

1）combining Person ROI to reduce background interference

2）adjusting confidence thresholds

3）refining bounding boxes

4）adding segmentation models to obtain more accurate masks

