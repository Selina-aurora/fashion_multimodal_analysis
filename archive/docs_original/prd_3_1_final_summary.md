# PRD 3.1 Instance Segmentation Final Summary

## Overview

This project implements an instance segmentation pipeline for fashion item understanding.

The goal of PRD 3.1 is to build a garment-level perception module for automatic detection, classification, and pixel-level segmentation of fashion items.

## Model Architecture

Final model:

**Mask R-CNN with ResNet50-FPN backbone**

Configuration:
- Framework: PyTorch / torchvision
- Initialization: COCO pretrained
- Task: Instance Segmentation
- Categories: 8 garment categories
- Training Epochs: 15

Final selected version:

**V3-B1 Data Expansion**

## Dataset and Benchmark

Datasets:
- DeepFashion2
- Fashionpedia

Evaluation includes:
- Core benchmark evaluation
- Detection performance analysis
- Classification accuracy analysis
- Mask quality evaluation
- Error case review

## Core Regression Evaluation

| Metric | Result |
|---|---|
| BBox50 Recall | 0.7442 |
| Localized Instance Classification Accuracy | 0.7188 |
| Mean Mask IoU (Correct Class) | 0.7897 |
| Mask IoU >= 0.85 Rate | 0.2093 |
| Average Inference Time | ~44 ms/image |

## Error Case Analysis

Error distribution:
- Missed Detection: 22
- Wrong Classification: 4
- Low Mask IoU: 1

Main challenges:
- Small object detection (shoe, bag, accessory)
- Fine-grained category confusion

Observed confusion:
- Dress → Pants
- Skirt → Top
- Outerwear → Top

## Small Object Analysis

Pipeline:

Original Image → GT Region Crop → Resize 640×640 → Model Inference

Test categories:
- Shoe
- Bag
- Accessory

Observation:
Local crop improves object visibility, but improvement is limited when original visual information is insufficient.

## Repository Structure

fashion_multimodal_analysis/

- benchmark/
- configs/
- scripts/
- reports/
- docs/
- src/

## Final Status

Completed:
- Mask R-CNN training pipeline
- V3-B1 model selection
- Core benchmark evaluation
- Error case analysis
- Small object investigation
- Experiment documentation

## Future Directions

1. Improve recall of small object categories.
2. Enhance fine-grained garment category discrimination.
3. Build attribute understanding modules based on instance-level regions.

## Conclusion

PRD 3.1 established a complete fashion instance segmentation pipeline, including model training, benchmark evaluation, failure analysis, and documentation.

The V3-B1 Data Expansion model is selected as the final model for this stage.
