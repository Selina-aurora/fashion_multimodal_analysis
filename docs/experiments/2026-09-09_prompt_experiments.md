# Prompt Optimization and Ablation — 2026-09-09

## Objective

Test whether adding clothing context or more explicit part descriptions changes Grounding DINO prediction coverage for fine-grained fashion targets.

## Prompt families

| Type | Example |
| --- | --- |
| Object-level | `sleeve` |
| Clothing-context | `shirt sleeve` |
| Detailed description | `long sleeve part of a shirt` |
| Spatial description | `upper arm clothing sleeve` |

The experiment also covered collar, button and zipper variants.

## Pilot results

Small-sample prompt ablation produced higher emitted-prediction rates for several context-rich prompts. Examples from the pilot summary include:

| Target | Baseline-style prompt | Higher-coverage pilot prompt |
| --- | --- | --- |
| sleeve | `sleeve` | `long sleeve part of a shirt` |
| collar | `collar` | `collar part of clothing` |
| button | `button` | `shirt button detail` |
| zipper | `zipper` | `front zipper closure` |

## Important interpretation update

These early percentages measured whether the model emitted a target-related prediction; they did **not** establish fine-grained localization accuracy. The later 300-image evaluation and manual audit showed that context prompts can increase prediction coverage while also producing large or coarse boxes.

Therefore the maintained conclusion is:

> Prompt wording matters, but improved prediction coverage must be validated with localization-quality review before it is treated as an improvement.

See the 2026-09-10 and 2026-09-11 experiment records for the current evidence.
