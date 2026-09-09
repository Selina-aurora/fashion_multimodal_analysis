Grounding DINO Prompt Ablation Analysis

1\. Objective

The prompt optimization experiment demonstrated that adding clothing-aware descriptions can improve Grounding DINO localization performance.

To further investigate the influence of language formulation, a prompt ablation study was conducted by comparing different levels of prompt specificity.

The objective is to identify effective prompt designs for fine-grained fashion component localization.



2\. Experimental Setup

Model

1）Model: Grounding DINO Tiny

2）Framework: PyTorch + Hugging Face Transformers

Dataset

Test images: 11 fashion images

Categories

The experiment evaluates four clothing attributes:

1）Sleeve

2）Collar

3）Button

4）Zipper

Evaluation Metrics

The following metrics are used:

Detection Rate

The percentage of images where the model produces at least one detection result.

Average Confidence

The average confidence score of detected regions.

Matched Label Rate

The percentage of images where the detected label matches the target clothing category.



3\. Prompt Variants

Four prompt styles are compared:

| Type | Example |

|---|---|

| Object-level | `sleeve` |

| Clothing-context | `shirt sleeve` |

| Detailed description | `long sleeve part of a shirt` |

| Spatial description | `upper arm clothing sleeve` |

The purpose is to investigate whether additional semantic information improves visual-language grounding.



4\. Results

（1）Sleeve Localization

| Prompt | Detection Rate | Average Confidence |

|---|---:|---:|

| `sleeve` | 81.82% | 0.3364 |

| `shirt sleeve` | 90.91% | 0.3511 |

| `long sleeve part of a shirt` | 100.00% | 0.3626 |

| `upper arm clothing sleeve` | 100.00% | 0.3508 |

The results show that adding clothing context improves sleeve localization.

The detailed description：long sleeve part of a shirt

achieves the highest confidence score, indicating stronger semantic alignment between the prompt and visual region.

（2）Collar Localization

| Prompt | Detection Rate | Average Confidence |

|---|---:|---:|

| `collar` | 90.91% | 0.3504 |

| `shirt collar area` | 100.00% | 0.3475 |

| `collar part of clothing` | 100.00% | 0.3527 |

| `upper clothing collar` | 100.00% | 0.3484 |

The results indicate that collar localization is relatively stable.

Adding clothing context improves detection coverage, while:：collar part of clothing

achieves the highest confidence.

（3）Button Localization

| Prompt | Detection Rate | Average Confidence |

|---|---:|---:|

| `button` | 36.36% | 0.2731 |

| `small button on clothing` | 54.55% | 0.3036 |

| `shirt button detail` | 90.91% | 0.3259 |

| `clothing fastener button` | 54.55% | 0.3055 |

Button localization benefits significantly from fashion-specific descriptions.

The prompt：shirt button detail

achieves the best performance.

This indicates that generic object names may introduce ambiguity, while garment-specific descriptions provide stronger guidance.

（4）Zipper Localization

| Prompt | Detection Rate | Average Confidence |

|---|---:|---:|

| `zipper` | 27.27% | 0.2685 |

| `zipper on a jacket or shirt` | 100.00% | 0.3017 |

| `clothing zipper detail` | 27.27% | 0.2750 |

| `front zipper closure` | 100.00% | 0.3380 |

Zipper localization shows the strongest improvement.

The simple prompt：zipper

provides insufficient semantic context.

However, the prompt：front zipper closure

achieves both higher detection rate and confidence.

This suggests that spatial and functional descriptions can improve localization of ambiguous clothing details.



5\. Discussion

（1）Effect of Semantic Context

The experiments demonstrate that prompt specificity strongly influences zero-shot localization performance.

For clothing components with strong visual structures, such as sleeves and collars, additional semantic information provides moderate improvement.

For small details, such as buttons and zippers, descriptive prompts provide larger gains.

（2）Optimal Prompt Design

Based on the ablation results, the selected prompts are:

| Category | Selected Prompt |

|---|---|

| Sleeve | `long sleeve part of a shirt` |

| Collar | `collar part of clothing` |

| Button | `shirt button detail` |

| Zipper | `front zipper closure` |

These prompts will be used in subsequent experiments.



6\. Conclusion

The prompt ablation study verifies that language formulation plays an important role in Grounding DINO-based fashion localization.

More specific clothing-aware descriptions generally improve detection performance, especially for fine-grained attributes such as buttons and zippers.

However, excessive or irrelevant semantic expansion may not provide additional benefits.

The selected prompts provide a stronger foundation for subsequent region refinement and segmentation experiments.

