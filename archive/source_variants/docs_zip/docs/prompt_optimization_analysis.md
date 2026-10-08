Grounding DINO Prompt Optimization Analysis

1\. Objective

The baseline experiment evaluated the ability of Grounding DINO for language-guided local region localization in fashion images.

The results showed that simple object-level prompts can identify visible clothing components, such as sleeves and collars, but performance decreases for fine-grained details such as buttons, sleeve cuffs, and zippers.

Therefore, a prompt optimization experiment was conducted by replacing short object names with more descriptive clothing-aware prompts.

The objective is to investigate whether additional semantic context can improve zero-shot clothing region localization.



2\. Experimental Setup

Model：

1）Model: Grounding DINO Tiny

2）Framework: PyTorch + Hugging Face Transformers

Task：

Language-guided local region localization for fashion images.

Dataset：

Test images: 11 fashion images

Evaluation Metrics

Two metrics are used:

Detection Rate

The percentage of images where the model generates at least one detection result.

Average Confidence

The average confidence score of detected regions.



3\. Prompt Optimization Design

The baseline experiment uses simple object-level prompts.

The optimized prompts introduce additional clothing context and semantic descriptions to improve visual-language alignment.

| Category | Baseline Prompt | Optimized Prompt |

|---|---|---|

| Shirt | `shirt` | `shirt area` |

| Sleeve | `sleeve` | `shirt sleeve` |

| Collar | `collar` | `shirt collar area` |

| Button | `button` | `small button on clothing` |

| Sleeve Cuff | `sleeve cuff` | `cuff at the end of a shirt sleeve` |

| Zipper | `zipper` | `zipper on a jacket or shirt` |

The optimized prompts aim to provide additional semantic information and reduce ambiguity between general concepts and fashion-specific regions.



4\. Experimental Results

| Category | Baseline Detection Rate | Optimized Detection Rate | Change |

|---|---:|---:|---:|

| Shirt | 27.27% | 18.18% | -9.09% |

| Sleeve | 54.55% | 90.91% | +36.36% |

| Collar | 36.36% | 72.73% | +36.37% |

| Button | 9.09% | 36.36% | +27.27% |

| Sleeve Cuff | 0.00% | 0.00% | 0.00% |

| Zipper | 0.00% | 63.64% | +63.64% |



5\. Analysis

（1）Structural Clothing Components

Prompt optimization significantly improves localization performance for visible clothing structures.

For sleeve localization, the prompt was changed from:

Original Prompt：sleeve

Optimized Prompt：shirt sleeve

The detection rate increased from 54.55% to 90.91%.

This indicates that adding clothing context helps Grounding DINO establish stronger semantic alignment between the language description and visual region.

For collar localization, the prompt was changed from:

Original Prompt：collar

Optimized Prompt：shirt collar area

The detection rate increased from 36.36% to 72.73%.

The additional garment context reduces ambiguity and improves grounding performance.

（2）Fine-grained Clothing Details

Prompt optimization also improves several fine-grained clothing categories.

For button localization:

Original Prompt：button

Optimized Prompt：small button on clothing

The detection rate increased from 9.09% to 36.36%.

For zipper localization:

Original Prompt：zipper

Optimized Prompt：zipper on a jacket or shirt

The detection rate increased from 0.00% to 63.64%.

These results indicate that isolated object names may not provide sufficient fashion-specific semantic information.

Adding clothing-related descriptions helps the model better associate textual concepts with visual regions.

（3）Limitations

Although prompt optimization improves several categories, some limitations remain.

For sleeve cuff localization:

Original Prompt：sleeve cuff

Optimized Prompt：cuff at the end of a shirt sleeve

The detection rate remains 0.00%.

This indicates that extremely small regions with limited visual information remain challenging for zero-shot localization models.

（4）Negative Effect of Over-generalized Descriptions

Not all prompt expansions improve performance.

For shirt localization:

Original Prompt：shirt

Detection Rate：27.27%

Optimized Prompt：shirt area

Detection Rate：18.18%

The performance decrease suggests that overly general descriptions may introduce ambiguity.

The word "area" does not provide stronger object semantics and may make the target region less clearly defined.



6\. Overall Findings

The prompt optimization experiment demonstrates that language formulation has a significant impact on Grounding DINO localization ability.

Adding clothing-aware semantic information improves localization performance for:

1）sleeves

2）collars

3）buttons

4）zippers

The largest improvement is observed for zipper localization：0.00% → 63.64%

This indicates that descriptive prompts can partially compensate for the limitations of zero-shot localization models when dealing with fashion-specific concepts.

However, extremely fine-grained regions such as sleeve cuffs remain difficult due to:

1）small target size

2）limited visual features

3）insufficient spatial resolution



7\. Conclusion

The experiment verifies that prompt optimization is an effective strategy for improving language-guided clothing region localization.

Compared with simple category-level prompts, clothing-aware descriptions provide stronger semantic guidance and improve the model's ability to identify fashion-related regions.

However, prompt optimization alone is insufficient for precise pixel-level fine-grained parsing.

Therefore, future work will focus on combining Grounding DINO with segmentation refinement methods to obtain accurate clothing region masks.

