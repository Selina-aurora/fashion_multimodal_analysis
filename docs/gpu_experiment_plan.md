GPU Experiment Plan

1\. Current Development Status

The first-stage development has been completed locally:

1）DeepFashion2 dataset structure analysis

2）Annotation parsing

3）Mask2Former baseline inference

4）Person mask extraction

5）Person ROI generation

6）ROI quality evaluation and failure case analysis

The current stage focuses on validating the complete pipeline and preparing the next-stage experiments.



2\. Purpose of GPU Usage

GPU resources will mainly be used for computationally intensive experiments.

The main purposes include:

（1）Large-scale Inference

Current local testing only uses a small number of images.

GPU will be used for:

1）running inference on larger subsets of DeepFashion2

2）evaluating ROI extraction performance

3）accelerating vision-language model testing

（2）Model Fine-tuning

GPU will be required for:

1）DeepFashion2-based Mask2Former fine-tuning

2）possible vision-language model adaptation experiments

（3）Language-guided Localization Experiments

For module 3.1.2, GPU will be used for:

1）Grounding DINO inference

2）prompt evaluation

3）comparison between different localization strategies



3\. Current GPU Deployment Strategy

The GPU environment will not be deployed continuously during the design stage.

Reason:

1）The current priority is to determine the technical route.

2）Early deployment may waste resources during environment debugging.

3）GPU experiments will start after the baseline implementation is prepared.

The planned workflow is:

Local Development

&#x20;       ↓

Pipeline Validation

&#x20;       ↓

GPU Environment Deployment

&#x20;       ↓

Large-scale Experiment

&#x20;       ↓

Optimization



4\. Planned GPU Environment

Target environment:

1）GPU: NVIDIA RTX 4090 24GB

2）CUDA: 12.1

3）PyTorch: 2.3.0

The environment will be used for:

1）Mask2Former experiments

2）Grounding DINO baseline testing

3）DeepFashion2 related experiments



5\. Resource Management Strategy

To reduce unnecessary cost:

1）Develop and debug code locally whenever possible.

2）Use GPU only for inference acceleration and model training.

3）Save experiment results and checkpoints before shutting down instances.

4）Avoid long-term idle GPU usage.

