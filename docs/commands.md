# 命令目录

当前 179 个分类入口。静态 --help 不加载模型；历史准备/冻结命令有写入行为，应在工作副本中执行。

优先阅读 [GPU 恢复指南](gpu_restore_and_run.md) 和 [V8 报告](full_dataset_training_v8_report.md)。

## analysis

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `analyze_deepfashion2_categories` | [入口](../scripts/analysis/analyze_deepfashion2_categories.py) · [实现](../src/fashion_multimodal_analysis/analysis/analyze_deepfashion2_categories.py) | Analyze category distribution in DeepFashion2 annotations. |
| `analyze_grounding_bbox_quality` | [入口](../scripts/analysis/analyze_grounding_bbox_quality.py) · [实现](../src/fashion_multimodal_analysis/analysis/analyze_grounding_bbox_quality.py) | Analyze Grounding DINO bounding-box size quality. |
| `analyze_grounding_failure_patterns` | [入口](../scripts/analysis/analyze_grounding_failure_patterns.py) · [实现](../src/fashion_multimodal_analysis/analysis/analyze_grounding_failure_patterns.py) | Analyze Grounding DINO localization failure patterns. |
| `analyze_visibility_filtering` | [入口](../scripts/analysis/analyze_visibility_filtering.py) · [实现](../src/fashion_multimodal_analysis/analysis/analyze_visibility_filtering.py) | analyze visibility filtering |
| `audit_dataset_overlap` | [入口](../scripts/analysis/audit_dataset_overlap.py) · [实现](../src/fashion_multimodal_analysis/analysis/audit_dataset_overlap.py) | Audit train/validation/Core overlap by normalized source image path. |
| `build_prd_3_1_2_final_summary_v1` | [入口](../scripts/analysis/build_prd_3_1_2_final_summary_v1.py) · [实现](../src/fashion_multimodal_analysis/analysis/build_prd_3_1_2_final_summary_v1.py) | build prd 3 1 2 final summary v1 |
| `diagnose_prd31_v4` | [入口](../scripts/analysis/diagnose_prd31_v4.py) · [实现](../src/fashion_multimodal_analysis/analysis/diagnose_prd31_v4.py) | Read-only PRD31 result analysis and native-annotation completeness audit. |
| `diagnose_prd31_v6_regressions` | [入口](../scripts/analysis/diagnose_prd31_v6_regressions.py) · [实现](../src/fashion_multimodal_analysis/analysis/diagnose_prd31_v6_regressions.py) | Read-only inference diagnosis of development cases that regressed from V5 to V6. |
| `finalize_prd_3_1_1_v1` | [入口](../scripts/analysis/finalize_prd_3_1_1_v1.py) · [实现](../src/fashion_multimodal_analysis/analysis/finalize_prd_3_1_1_v1.py) | finalize prd 3 1 1 v1 |
| `record_maskrcnn_8class_experiment_config_v1` | [入口](../scripts/analysis/record_maskrcnn_8class_experiment_config_v1.py) · [实现](../src/fashion_multimodal_analysis/analysis/record_maskrcnn_8class_experiment_config_v1.py) | record maskrcnn 8class experiment config v1 |
| `summarize_attribute_manual_audit` | [入口](../scripts/analysis/summarize_attribute_manual_audit.py) · [实现](../src/fashion_multimodal_analysis/analysis/summarize_attribute_manual_audit.py) | Summarize a completed PRD 3.1.3 manual audit CSV. |
| `summarize_prd31_acceptance` | [入口](../scripts/analysis/summarize_prd31_acceptance.py) · [实现](../src/fashion_multimodal_analysis/analysis/summarize_prd31_acceptance.py) | Finalize strict semantic reviews, PRD gates and complete-chain failure attribution. |
| `summarize_prompt_diagnostic_no_pandas` | [入口](../scripts/analysis/summarize_prompt_diagnostic_no_pandas.py) · [实现](../src/fashion_multimodal_analysis/analysis/summarize_prompt_diagnostic_no_pandas.py) | Summarize prompt diagnostic results without pandas. |

## attributes/color

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `run_primary_color_baseline_v1` | [入口](../scripts/attributes/color/run_primary_color_baseline_v1.py) · [实现](../src/fashion_multimodal_analysis/attributes/color/run_primary_color_baseline_v1.py) | PRD 3.1.3 primary color baseline v1. |

## attributes/design

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `build_unified_garment_attribute_vector_v1` | [入口](../scripts/attributes/design/build_unified_garment_attribute_vector_v1.py) · [实现](../src/fashion_multimodal_analysis/attributes/design/build_unified_garment_attribute_vector_v1.py) | PRD 3.1.3 unified garment attribute output v1. |
| `build_unified_garment_attribute_vector_v1_fixed` | [入口](../scripts/attributes/design/build_unified_garment_attribute_vector_v1_fixed.py) · [实现](../src/fashion_multimodal_analysis/attributes/design/build_unified_garment_attribute_vector_v1_fixed.py) | PRD 3.1.3 unified garment attribute output v1. |
| `repair_design_labels_v1_outputs` | [入口](../scripts/attributes/design/repair_design_labels_v1_outputs.py) · [实现](../src/fashion_multimodal_analysis/attributes/design/repair_design_labels_v1_outputs.py) | Repair PRD 3.1.3 design-label v1 outputs without rerunning CLIP. |
| `run_attribute_baseline_clip` | [入口](../scripts/attributes/design/run_attribute_baseline_clip.py) · [实现](../src/fashion_multimodal_analysis/attributes/design/run_attribute_baseline_clip.py) | PRD 3.1.3 local CPU baseline using CLIP. |
| `run_attribute_baseline_clip_batch` | [入口](../scripts/attributes/design/run_attribute_baseline_clip_batch.py) · [实现](../src/fashion_multimodal_analysis/attributes/design/run_attribute_baseline_clip_batch.py) | PRD 3.1.3 CLIP batch pilot. |
| `run_attribute_baseline_clip_instances` | [入口](../scripts/attributes/design/run_attribute_baseline_clip_instances.py) · [实现](../src/fashion_multimodal_analysis/attributes/design/run_attribute_baseline_clip_instances.py) | PRD 3.1.3 per-garment CLIP attribute baseline. |
| `run_attribute_baseline_clip_instances_v2` | [入口](../scripts/attributes/design/run_attribute_baseline_clip_instances_v2.py) · [实现](../src/fashion_multimodal_analysis/attributes/design/run_attribute_baseline_clip_instances_v2.py) | PRD 3.1.3 per-garment attribute baseline v2. |
| `run_attribute_baseline_cpu` | [入口](../scripts/attributes/design/run_attribute_baseline_cpu.py) · [实现](../src/fashion_multimodal_analysis/attributes/design/run_attribute_baseline_cpu.py) | PRD 3.1.3 local CPU attribute baseline. |
| `run_attribute_baseline_cpu_v2` | [入口](../scripts/attributes/design/run_attribute_baseline_cpu_v2.py) · [实现](../src/fashion_multimodal_analysis/attributes/design/run_attribute_baseline_cpu_v2.py) | PRD 3.1.3 lightweight local CPU baseline (isolated-env version). |
| `run_attribute_baseline_instances_v3` | [入口](../scripts/attributes/design/run_attribute_baseline_instances_v3.py) · [实现](../src/fashion_multimodal_analysis/attributes/design/run_attribute_baseline_instances_v3.py) | PRD 3.1.3 per-garment baseline v3. |
| `run_attribute_baseline_v1` | [入口](../scripts/attributes/design/run_attribute_baseline_v1.py) · [实现](../src/fashion_multimodal_analysis/attributes/design/run_attribute_baseline_v1.py) | PRD 3.1.3 fine-grained attribute extraction baseline v1. |
| `run_attribute_baseline_v2` | [入口](../scripts/attributes/design/run_attribute_baseline_v2.py) · [实现](../src/fashion_multimodal_analysis/attributes/design/run_attribute_baseline_v2.py) | PRD 3.1.3 attribute extraction baseline v2. |
| `run_design_attribute_labels_v1` | [入口](../scripts/attributes/design/run_design_attribute_labels_v1.py) · [实现](../src/fashion_multimodal_analysis/attributes/design/run_design_attribute_labels_v1.py) | run design attribute labels v1 |

## attributes/geometry

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `apply_style_feature_validity_gating_v1` | [入口](../scripts/attributes/geometry/apply_style_feature_validity_gating_v1.py) · [实现](../src/fashion_multimodal_analysis/attributes/geometry/apply_style_feature_validity_gating_v1.py) | Apply validity/quality gating to PRD 3.1.3 style feature vector v1. |
| `build_style_feature_vector_v1` | [入口](../scripts/attributes/geometry/build_style_feature_vector_v1.py) · [实现](../src/fashion_multimodal_analysis/attributes/geometry/build_style_feature_vector_v1.py) | Build PRD 3.1.3 unified continuous style feature vector v1. |
| `build_style_feature_vector_v1_nopandas` | [入口](../scripts/attributes/geometry/build_style_feature_vector_v1_nopandas.py) · [实现](../src/fashion_multimodal_analysis/attributes/geometry/build_style_feature_vector_v1_nopandas.py) | Build PRD 3.1.3 unified continuous style feature vector v1 (no pandas). |
| `extract_neckline_geometry_continuous_v1` | [入口](../scripts/attributes/geometry/extract_neckline_geometry_continuous_v1.py) · [实现](../src/fashion_multimodal_analysis/attributes/geometry/extract_neckline_geometry_continuous_v1.py) | PRD 3.1.3 continuous neckline geometry pilot v1. |
| `extract_pants_silhouette_continuous_v1` | [入口](../scripts/attributes/geometry/extract_pants_silhouette_continuous_v1.py) · [实现](../src/fashion_multimodal_analysis/attributes/geometry/extract_pants_silhouette_continuous_v1.py) | PRD 3.1.3 pants silhouette continuous geometry v1. |
| `extract_skirt_flare_continuous_v1` | [入口](../scripts/attributes/geometry/extract_skirt_flare_continuous_v1.py) · [实现](../src/fashion_multimodal_analysis/attributes/geometry/extract_skirt_flare_continuous_v1.py) | PRD 3.1.3 continuous skirt-flare geometry feature pilot v1. |
| `extract_sleeve_geometry_features_v2` | [入口](../scripts/attributes/geometry/extract_sleeve_geometry_features_v2.py) · [实现](../src/fashion_multimodal_analysis/attributes/geometry/extract_sleeve_geometry_features_v2.py) | PRD 3.1.3 sleeve continuous geometry feature pilot v2. |
| `extract_trouser_fit_continuous_v2` | [入口](../scripts/attributes/geometry/extract_trouser_fit_continuous_v2.py) · [实现](../src/fashion_multimodal_analysis/attributes/geometry/extract_trouser_fit_continuous_v2.py) | PRD 3.1.3 long-trouser fit continuous geometry v2. |
| `extract_upper_silhouette_continuous_v1` | [入口](../scripts/attributes/geometry/extract_upper_silhouette_continuous_v1.py) · [实现](../src/fashion_multimodal_analysis/attributes/geometry/extract_upper_silhouette_continuous_v1.py) | PRD 3.1.3 upper-garment silhouette continuous features v1. |
| `extract_upper_silhouette_continuous_v2` | [入口](../scripts/attributes/geometry/extract_upper_silhouette_continuous_v2.py) · [实现](../src/fashion_multimodal_analysis/attributes/geometry/extract_upper_silhouette_continuous_v2.py) | PRD 3.1.3 upper-garment silhouette continuous features v2. |
| `run_neckline_semantic_continuous_v2` | [入口](../scripts/attributes/geometry/run_neckline_semantic_continuous_v2.py) · [实现](../src/fashion_multimodal_analysis/attributes/geometry/run_neckline_semantic_continuous_v2.py) | PRD 3.1.3 local-neckline continuous semantic pilot v2. |
| `run_pants_length_continuous_v1` | [入口](../scripts/attributes/geometry/run_pants_length_continuous_v1.py) · [实现](../src/fashion_multimodal_analysis/attributes/geometry/run_pants_length_continuous_v1.py) | PRD 3.1.3 continuous pants-length feature baseline v1. |
| `run_skirt_dress_length_continuous_v1` | [入口](../scripts/attributes/geometry/run_skirt_dress_length_continuous_v1.py) · [实现](../src/fashion_multimodal_analysis/attributes/geometry/run_skirt_dress_length_continuous_v1.py) | PRD 3.1.3 skirt/dress length continuous semantic baseline v1. |
| `run_sleeve_length_continuous_v1` | [入口](../scripts/attributes/geometry/run_sleeve_length_continuous_v1.py) · [实现](../src/fashion_multimodal_analysis/attributes/geometry/run_sleeve_length_continuous_v1.py) | PRD 3.1.3 continuous sleeve-length feature baseline v1. |

## attributes/pattern

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `run_pattern_baseline_v1_fixed` | [入口](../scripts/attributes/pattern/run_pattern_baseline_v1_fixed.py) · [实现](../src/fashion_multimodal_analysis/attributes/pattern/run_pattern_baseline_v1_fixed.py) | PRD 3.1.3 garment pattern baseline v1. |
| `run_pattern_hierarchical_v2` | [入口](../scripts/attributes/pattern/run_pattern_hierarchical_v2.py) · [实现](../src/fashion_multimodal_analysis/attributes/pattern/run_pattern_hierarchical_v2.py) | PRD 3.1.3 pattern recognition v2: hierarchical CLIP baseline. |
| `run_pattern_hierarchical_v2_fixed` | [入口](../scripts/attributes/pattern/run_pattern_hierarchical_v2_fixed.py) · [实现](../src/fashion_multimodal_analysis/attributes/pattern/run_pattern_hierarchical_v2_fixed.py) | PRD 3.1.3 pattern recognition v2 - fixed for newer Transformers CLIP APIs. |
| `run_pattern_hierarchical_v3` | [入口](../scripts/attributes/pattern/run_pattern_hierarchical_v3.py) · [实现](../src/fashion_multimodal_analysis/attributes/pattern/run_pattern_hierarchical_v3.py) | PRD 3.1.3 pattern recognition v3: four-way presence gate + subtype classifier. |

## benchmarking

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `audit_fashionpedia_prd_mapping_v1` | [入口](../scripts/benchmarking/audit_fashionpedia_prd_mapping_v1.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/audit_fashionpedia_prd_mapping_v1.py) | Audit the EXACT Fashionpedia -> PRD class mapping already used by this project. |
| `audit_prd31_benchmark_difficulty_v1` | [入口](../scripts/benchmarking/audit_prd31_benchmark_difficulty_v1.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/audit_prd31_benchmark_difficulty_v1.py) | Audit geometry / difficulty bias before freezing PRD 3.1.1 benchmark. |
| `audit_prd31_data_for_benchmark_v1` | [入口](../scripts/benchmarking/audit_prd31_data_for_benchmark_v1.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/audit_prd31_data_for_benchmark_v1.py) | Audit PRD 3.1 data usage before freezing a real benchmark_test split. |
| `audit_prd31_segmentation_gt_qc_v1` | [入口](../scripts/benchmarking/audit_prd31_segmentation_gt_qc_v1.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/audit_prd31_segmentation_gt_qc_v1.py) | Automated GT annotation QC for PRD 3.1.1 Core + Stress review pools. |
| `build_prd31_core_coverage_review_v1` | [入口](../scripts/benchmarking/build_prd31_core_coverage_review_v1.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/build_prd31_core_coverage_review_v1.py) | Build PRD 3.1.1 Core coverage review packet v1. |
| `build_prd31_core_stress_review_pool_v1` | [入口](../scripts/benchmarking/build_prd31_core_stress_review_pool_v1.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/build_prd31_core_stress_review_pool_v1.py) | Build PRD 3.1.1 difficulty-controlled Core + Stress review pools. |
| `build_prd31_human_review_packet_v1` | [入口](../scripts/benchmarking/build_prd31_human_review_packet_v1.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/build_prd31_human_review_packet_v1.py) | Build a focused human-review packet for PRD 3.1.1 QC v2. |
| `build_prd31_minority_scene_review_packet_v1` | [入口](../scripts/benchmarking/build_prd31_minority_scene_review_packet_v1.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/build_prd31_minority_scene_review_packet_v1.py) | Build targeted human-review packet for remaining minority-scene Core cases. |
| `build_prd31_provisional_core_selection_v1` | [入口](../scripts/benchmarking/build_prd31_provisional_core_selection_v1.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/build_prd31_provisional_core_selection_v1.py) | Build a PROVISIONAL 400-case Core selection for PRD 3.1.1. |
| `build_prd31_review_packet` | [入口](../scripts/benchmarking/build_prd31_review_packet.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/build_prd31_review_packet.py) | Build an offline human GT review page. Color labels require untinted source images. |
| `build_prd31_segmentation_review_pool_v1` | [入口](../scripts/benchmarking/build_prd31_segmentation_review_pool_v1.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/build_prd31_segmentation_review_pool_v1.py) | Freeze the Fashionpedia -> PRD mapping and build a LEAKAGE-SAFE |
| `build_prd31_segmentation_review_pool_v2` | [入口](../scripts/benchmarking/build_prd31_segmentation_review_pool_v2.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/build_prd31_segmentation_review_pool_v2.py) | Freeze the Fashionpedia -> PRD mapping and build a LEAKAGE-SAFE |
| `discover_prd31_raw_benchmark_pool_v1` | [入口](../scripts/benchmarking/discover_prd31_raw_benchmark_pool_v1.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/discover_prd31_raw_benchmark_pool_v1.py) | Discover untouched raw-data pools for PRD 3.1 benchmark_test v1. |
| `freeze_prd31_reviewed_tests` | [入口](../scripts/benchmarking/freeze_prd31_reviewed_tests.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/freeze_prd31_reviewed_tests.py) | Validate human reviews and freeze v2 manifests without replacing frozen v1. |
| `freeze_prd31_segmentation_test_v1` | [入口](../scripts/benchmarking/freeze_prd31_segmentation_test_v1.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/freeze_prd31_segmentation_test_v1.py) | Freeze PRD 3.1.1 Core benchmark v1 after full scene review. |
| `prepare_prd31_core_freeze_and_protocol_v1` | [入口](../scripts/benchmarking/prepare_prd31_core_freeze_and_protocol_v1.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/prepare_prd31_core_freeze_and_protocol_v1.py) | Prepare PRD 3.1.1 final Core-freeze review + freeze Evaluation Protocol v1. |
| `prepare_prd31_recovery` | [入口](../scripts/benchmarking/prepare_prd31_recovery.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/prepare_prd31_recovery.py) | Prepare clean training, reusable legacy reviews and new review candidates. |
| `refine_prd31_segmentation_gt_qc_v2` | [入口](../scripts/benchmarking/refine_prd31_segmentation_gt_qc_v2.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/refine_prd31_segmentation_gt_qc_v2.py) | Refine PRD 3.1.1 automated GT QC after diagnosing an overly strict v1 rule. |
| `setup_prd31_benchmark_v1` | [入口](../scripts/benchmarking/setup_prd31_benchmark_v1.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/setup_prd31_benchmark_v1.py) | Install PRD 3.1 Standard Benchmark & Acceptance Suite v1 |
| `suggest_prd31_core_scene_metadata_v1` | [入口](../scripts/benchmarking/suggest_prd31_core_scene_metadata_v1.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/suggest_prd31_core_scene_metadata_v1.py) | Generate CLIP-assisted coverage metadata suggestions for PRD 3.1.1 Core. |
| `suggest_prd31_core_scene_metadata_v1_fixed` | [入口](../scripts/benchmarking/suggest_prd31_core_scene_metadata_v1_fixed.py) · [实现](../src/fashion_multimodal_analysis/benchmarking/suggest_prd31_core_scene_metadata_v1_fixed.py) | Generate CLIP-assisted coverage metadata suggestions for PRD 3.1.1 Core. |

## data_tools

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `build_deepfashion2_sample` | [入口](../scripts/data_tools/build_deepfashion2_sample.py) · [实现](../src/fashion_multimodal_analysis/data_tools/build_deepfashion2_sample.py) | Build one DeepFashion2 training sample. |
| `build_fashionpedia_missing3_pilot_v1` | [入口](../scripts/data_tools/build_fashionpedia_missing3_pilot_v1.py) · [实现](../src/fashion_multimodal_analysis/data_tools/build_fashionpedia_missing3_pilot_v1.py) | Build a small Fashionpedia pilot for the 3 PRD 3.1.1 categories |
| `build_fashionpedia_missing3_pilot_v2` | [入口](../scripts/data_tools/build_fashionpedia_missing3_pilot_v2.py) · [实现](../src/fashion_multimodal_analysis/data_tools/build_fashionpedia_missing3_pilot_v2.py) | Fashionpedia -> PRD 3.1.1 missing-3-class pilot |
| `build_prd_8class_train_v2_expand_fashionpedia` | [入口](../scripts/data_tools/build_prd_8class_train_v2_expand_fashionpedia.py) · [实现](../src/fashion_multimodal_analysis/data_tools/build_prd_8class_train_v2_expand_fashionpedia.py) | Expand the Fashionpedia training portion for PRD 3.1.1 baseline v2. |
| `build_prd_8class_train_v2_expand_fashionpedia_fixed` | [入口](../scripts/data_tools/build_prd_8class_train_v2_expand_fashionpedia_fixed.py) · [实现](../src/fashion_multimodal_analysis/data_tools/build_prd_8class_train_v2_expand_fashionpedia_fixed.py) | Build PRD 3.1.1 train_v2 with exactly 100 VALID Fashionpedia training |
| `build_prd_8class_train_v3_expand_underrepresented` | [入口](../scripts/data_tools/build_prd_8class_train_v3_expand_underrepresented.py) · [实现](../src/fashion_multimodal_analysis/data_tools/build_prd_8class_train_v3_expand_underrepresented.py) | Build PRD 3.1.1 train_v2 with exactly 100 VALID Fashionpedia training |
| `build_prd_8class_train_val_split_v1` | [入口](../scripts/data_tools/build_prd_8class_train_val_split_v1.py) · [实现](../src/fashion_multimodal_analysis/data_tools/build_prd_8class_train_val_split_v1.py) | Build an image-grouped train/validation split for the unified PRD 3.1.1 |
| `build_prd_8class_unified_manifest_v1` | [入口](../scripts/data_tools/build_prd_8class_unified_manifest_v1.py) · [实现](../src/fashion_multimodal_analysis/data_tools/build_prd_8class_unified_manifest_v1.py) | Build a unified PRD 3.1.1 8-class manifest from: |
| `build_prd_8class_unified_manifest_v2` | [入口](../scripts/data_tools/build_prd_8class_unified_manifest_v2.py) · [实现](../src/fashion_multimodal_analysis/data_tools/build_prd_8class_unified_manifest_v2.py) | Corrected PRD 3.1.1 8-class unified manifest validator. |
| `build_target_balanced_candidates` | [入口](../scripts/data_tools/build_target_balanced_candidates.py) · [实现](../src/fashion_multimodal_analysis/data_tools/build_target_balanced_candidates.py) | Build target-balanced candidate pools for manual verification. |
| `check_deepfashion2_dataloader` | [入口](../scripts/data_tools/check_deepfashion2_dataloader.py) · [实现](../src/fashion_multimodal_analysis/data_tools/check_deepfashion2_dataloader.py) | Check batched DeepFashion2 data loading. |
| `check_deepfashion2_dataset` | [入口](../scripts/data_tools/check_deepfashion2_dataset.py) · [实现](../src/fashion_multimodal_analysis/data_tools/check_deepfashion2_dataset.py) | Check DeepFashion2Dataset output with one sample. |
| `convert_annotation_to_masks` | [入口](../scripts/data_tools/convert_annotation_to_masks.py) · [实现](../src/fashion_multimodal_analysis/data_tools/convert_annotation_to_masks.py) | Convert DeepFashion2 polygon annotations to binary instance masks. |
| `prepare_garment_instances_from_deepfashion2` | [入口](../scripts/data_tools/prepare_garment_instances_from_deepfashion2.py) · [实现](../src/fashion_multimodal_analysis/data_tools/prepare_garment_instances_from_deepfashion2.py) | Prepare per-garment crops/masks from DeepFashion2 annotations. |

## grounding/baselines

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `build_grounding_eval_samples` | [入口](../scripts/grounding/baselines/build_grounding_eval_samples.py) · [实现](../src/fashion_multimodal_analysis/grounding/baselines/build_grounding_eval_samples.py) | Build reproducible evaluation samples from DeepFashion2. |
| `build_verified_positive_benchmark` | [入口](../scripts/grounding/baselines/build_verified_positive_benchmark.py) · [实现](../src/fashion_multimodal_analysis/grounding/baselines/build_verified_positive_benchmark.py) | Build the final verified-positive localization benchmark. |
| `run_grounding_dino_baseline` | [入口](../scripts/grounding/baselines/run_grounding_dino_baseline.py) · [实现](../src/fashion_multimodal_analysis/grounding/baselines/run_grounding_dino_baseline.py) | run grounding dino baseline |
| `run_grounding_dino_multi_test` | [入口](../scripts/grounding/baselines/run_grounding_dino_multi_test.py) · [实现](../src/fashion_multimodal_analysis/grounding/baselines/run_grounding_dino_multi_test.py) | run grounding dino multi test |
| `run_grounding_dino_prompt_optimization` | [入口](../scripts/grounding/baselines/run_grounding_dino_prompt_optimization.py) · [实现](../src/fashion_multimodal_analysis/grounding/baselines/run_grounding_dino_prompt_optimization.py) | run grounding dino prompt optimization |
| `run_roi_conditioned_grounding` | [入口](../scripts/grounding/baselines/run_roi_conditioned_grounding.py) · [实现](../src/fashion_multimodal_analysis/grounding/baselines/run_roi_conditioned_grounding.py) | Run annotation-assisted ROI-conditioned Grounding DINO evaluation. |
| `run_verified_positive_prompt_diagnostic` | [入口](../scripts/grounding/baselines/run_verified_positive_prompt_diagnostic.py) · [实现](../src/fashion_multimodal_analysis/grounding/baselines/run_verified_positive_prompt_diagnostic.py) | Run a controlled prompt diagnostic on verified-positive cases. |
| `test_language_prompts` | [入口](../scripts/grounding/baselines/check_language_prompts.py) · [实现](../src/fashion_multimodal_analysis/grounding/baselines/check_language_prompts.py) | test language prompts |
| `test_prompt_variants` | [入口](../scripts/grounding/baselines/diagnose_prompt_variants.py) · [实现](../src/fashion_multimodal_analysis/grounding/baselines/diagnose_prompt_variants.py) | test prompt variants |

## grounding/core_parts

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `annotate_core_part_boxes` | [入口](../scripts/grounding/core_parts/annotate_core_part_boxes.py) · [实现](../src/fashion_multimodal_analysis/grounding/core_parts/annotate_core_part_boxes.py) | OpenCV bbox annotator with previous/next navigation for the core part pilot. |
| `diagnose_core_part_detector_topk` | [入口](../scripts/grounding/core_parts/diagnose_core_part_detector_topk.py) · [实现](../src/fashion_multimodal_analysis/grounding/core_parts/diagnose_core_part_detector_topk.py) | Top-k candidate-ranking diagnostic for core_part_detector_v1. |
| `diagnose_core_part_detector_train_vs_val` | [入口](../scripts/grounding/core_parts/diagnose_core_part_detector_train_vs_val.py) · [实现](../src/fashion_multimodal_analysis/grounding/core_parts/diagnose_core_part_detector_train_vs_val.py) | Train-vs-validation diagnosis for core_part_detector_v1. |
| `diagnose_core_part_detector_v1` | [入口](../scripts/grounding/core_parts/diagnose_core_part_detector_v1.py) · [实现](../src/fashion_multimodal_analysis/grounding/core_parts/diagnose_core_part_detector_v1.py) | Diagnose validation predictions for core_part_detector_v1. |
| `prepare_core_part_annotation_pilot` | [入口](../scripts/grounding/core_parts/prepare_core_part_annotation_pilot.py) · [实现](../src/fashion_multimodal_analysis/grounding/core_parts/prepare_core_part_annotation_pilot.py) | Prepare a leakage-safe core part annotation pilot. |
| `review_flagged_core_part_boxes` | [入口](../scripts/grounding/core_parts/review_flagged_core_part_boxes.py) · [实现](../src/fashion_multimodal_analysis/grounding/core_parts/review_flagged_core_part_boxes.py) | Review only the semantic-QC flagged core-part annotations. |
| `train_core_part_detector_v1` | [入口](../scripts/grounding/core_parts/train_core_part_detector_v1.py) · [实现](../src/fashion_multimodal_analysis/grounding/core_parts/train_core_part_detector_v1.py) | Train a small supervised part detector for collar / cuff / hem. |
| `visualize_core_part_annotations` | [入口](../scripts/grounding/core_parts/visualize_core_part_annotations.py) · [实现](../src/fashion_multimodal_analysis/grounding/core_parts/visualize_core_part_annotations.py) | Generate visual QC sheets for the core part bbox annotations. |

## grounding/evaluation

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `evaluate_grounding_results` | [入口](../scripts/grounding/evaluation/evaluate_grounding_results.py) · [实现](../src/fashion_multimodal_analysis/grounding/evaluation/evaluate_grounding_results.py) | evaluate grounding results |
| `evaluate_person_roi_adaptive_batch` | [入口](../scripts/grounding/evaluation/evaluate_person_roi_adaptive_batch.py) · [实现](../src/fashion_multimodal_analysis/grounding/evaluation/evaluate_person_roi_adaptive_batch.py) | Evaluate adaptive person ROI extraction on DeepFashion2 samples. |
| `evaluate_person_roi_batch` | [入口](../scripts/grounding/evaluation/evaluate_person_roi_batch.py) · [实现](../src/fashion_multimodal_analysis/grounding/evaluation/evaluate_person_roi_batch.py) | Evaluate person ROI extraction on a batch of DeepFashion2 images. |
| `evaluate_person_roi_fallback_batch` | [入口](../scripts/grounding/evaluation/evaluate_person_roi_fallback_batch.py) · [实现](../src/fashion_multimodal_analysis/grounding/evaluation/evaluate_person_roi_fallback_batch.py) | Evaluate person ROI extraction with a large-person fallback rule. |
| `evaluate_person_roi_quality_batch` | [入口](../scripts/grounding/evaluation/evaluate_person_roi_quality_batch.py) · [实现](../src/fashion_multimodal_analysis/grounding/evaluation/evaluate_person_roi_quality_batch.py) | Evaluate person ROI quality using connected-component diagnostics. |
| `evaluate_prd_40case_coverage` | [入口](../scripts/grounding/evaluation/evaluate_prd_40case_coverage.py) · [实现](../src/fashion_multimodal_analysis/grounding/evaluation/evaluate_prd_40case_coverage.py) | Evaluate the 40-case PRD 3.1.2 region-coverage pilot with Grounding DINO. |
| `evaluate_prompt_variants` | [入口](../scripts/grounding/evaluation/evaluate_prompt_variants.py) · [实现](../src/fashion_multimodal_analysis/grounding/evaluation/evaluate_prompt_variants.py) | evaluate prompt variants |
| `evaluate_verified_positive_benchmark` | [入口](../scripts/grounding/evaluation/evaluate_verified_positive_benchmark.py) · [实现](../src/fashion_multimodal_analysis/grounding/evaluation/evaluate_verified_positive_benchmark.py) | Evaluate the verified-positive benchmark with Grounding DINO. |
| `evaluate_verified_positive_benchmark_v2` | [入口](../scripts/grounding/evaluation/evaluate_verified_positive_benchmark_v2.py) · [实现](../src/fashion_multimodal_analysis/grounding/evaluation/evaluate_verified_positive_benchmark_v2.py) | Evaluate the verified-positive benchmark with Grounding DINO. |
| `evaluate_verified_positive_benchmark_v3` | [入口](../scripts/grounding/evaluation/evaluate_verified_positive_benchmark_v3.py) · [实现](../src/fashion_multimodal_analysis/grounding/evaluation/evaluate_verified_positive_benchmark_v3.py) | Evaluate the verified-positive benchmark with Grounding DINO. |
| `run_grounding_group_evaluation` | [入口](../scripts/grounding/evaluation/run_grounding_group_evaluation.py) · [实现](../src/fashion_multimodal_analysis/grounding/evaluation/run_grounding_group_evaluation.py) | Run Grounding DINO evaluation on a sampled image group. |

## grounding/refinement

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `build_remaining5_region_diagnostic_v1` | [入口](../scripts/grounding/refinement/build_remaining5_region_diagnostic_v1.py) · [实现](../src/fashion_multimodal_analysis/grounding/refinement/build_remaining5_region_diagnostic_v1.py) | build remaining5 region diagnostic v1 |
| `build_shoulder_waist_pattern_refinement_inputs_v1` | [入口](../scripts/grounding/refinement/build_shoulder_waist_pattern_refinement_inputs_v1.py) · [实现](../src/fashion_multimodal_analysis/grounding/refinement/build_shoulder_waist_pattern_refinement_inputs_v1.py) | build shoulder waist pattern refinement inputs v1 |
| `build_shoulder_waist_pattern_refinement_inputs_v1_fixed` | [入口](../scripts/grounding/refinement/build_shoulder_waist_pattern_refinement_inputs_v1_fixed.py) · [实现](../src/fashion_multimodal_analysis/grounding/refinement/build_shoulder_waist_pattern_refinement_inputs_v1_fixed.py) | build shoulder waist pattern refinement inputs v1 fixed |
| `build_shoulder_waist_pattern_refinement_inputs_v1_v2` | [入口](../scripts/grounding/refinement/build_shoulder_waist_pattern_refinement_inputs_v1_v2.py) · [实现](../src/fashion_multimodal_analysis/grounding/refinement/build_shoulder_waist_pattern_refinement_inputs_v1_v2.py) | build shoulder waist pattern refinement inputs v1 v2 |
| `evaluate_core_regions_spatial_hr` | [入口](../scripts/grounding/refinement/evaluate_core_regions_spatial_hr.py) · [实现](../src/fashion_multimodal_analysis/grounding/refinement/evaluate_core_regions_spatial_hr.py) | Compare BASELINE vs target-specific SPATIAL_HR on 15 core-region cases. |
| `evaluate_sam_refinement_9case` | [入口](../scripts/grounding/refinement/evaluate_sam_refinement_9case.py) · [实现](../src/fashion_multimodal_analysis/grounding/refinement/evaluate_sam_refinement_9case.py) | Run a 9-case SAM refinement pilot on Grounding DINO region proposals. |
| `run_pocket_decoration_prompt_scale_diagnostic_v1` | [入口](../scripts/grounding/refinement/run_pocket_decoration_prompt_scale_diagnostic_v1.py) · [实现](../src/fashion_multimodal_analysis/grounding/refinement/run_pocket_decoration_prompt_scale_diagnostic_v1.py) | run pocket decoration prompt scale diagnostic v1 |
| `run_segmentation_first_diagnostic` | [入口](../scripts/grounding/refinement/run_segmentation_first_diagnostic.py) · [实现](../src/fashion_multimodal_analysis/grounding/refinement/run_segmentation_first_diagnostic.py) | Run a segmentation-first tight-crop diagnostic for Grounding DINO. |
| `run_shoulder_waist_pattern_groundingdino_refinement_v1` | [入口](../scripts/grounding/refinement/run_shoulder_waist_pattern_groundingdino_refinement_v1.py) · [实现](../src/fashion_multimodal_analysis/grounding/refinement/run_shoulder_waist_pattern_groundingdino_refinement_v1.py) | run shoulder waist pattern groundingdino refinement v1 |

## image_processing

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `build_small_object_crop` | [入口](../scripts/image_processing/build_small_object_crop.py) · [实现](../src/fashion_multimodal_analysis/image_processing/build_small_object_crop.py) | build small object crop |

## integration

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `analyze_prd31_roi_quality_vs_attributes_v2` | [入口](../scripts/integration/analyze_prd31_roi_quality_vs_attributes_v2.py) · [实现](../src/fashion_multimodal_analysis/integration/analyze_prd31_roi_quality_vs_attributes_v2.py) | PRD 3.1 propagation diagnosis v2: |
| `build_prd31_matched_roi_manifest_v1` | [入口](../scripts/integration/build_prd31_matched_roi_manifest_v1.py) · [实现](../src/fashion_multimodal_analysis/integration/build_prd31_matched_roi_manifest_v1.py) | Build a paired predicted-ROI manifest for PRD 3.1 propagation analysis. |
| `build_prd31_matched_roi_manifest_v2` | [入口](../scripts/integration/build_prd31_matched_roi_manifest_v2.py) · [实现](../src/fashion_multimodal_analysis/integration/build_prd31_matched_roi_manifest_v2.py) | Build a paired predicted-ROI manifest for PRD 3.1 propagation analysis. |
| `build_prd31_predicted_roi_pilot_v1` | [入口](../scripts/integration/build_prd31_predicted_roi_pilot_v1.py) · [实现](../src/fashion_multimodal_analysis/integration/build_prd31_predicted_roi_pilot_v1.py) | PRD 3.1 end-to-end integration pilot v1: |
| `build_prd31_predicted_roi_pilot_v2` | [入口](../scripts/integration/build_prd31_predicted_roi_pilot_v2.py) · [实现](../src/fashion_multimodal_analysis/integration/build_prd31_predicted_roi_pilot_v2.py) | PRD 3.1 end-to-end integration pilot v2: |
| `run_prd31_acceptance` | [入口](../scripts/integration/run_prd31_acceptance.py) · [实现](../src/fashion_multimodal_analysis/integration/run_prd31_acceptance.py) | Run frozen v2 Track A and complete Track B; leave semantic grading to humans. |
| `run_prd31_bbox_mask_isolation_ablation_v1` | [入口](../scripts/integration/run_prd31_bbox_mask_isolation_ablation_v1.py) · [实现](../src/fashion_multimodal_analysis/integration/run_prd31_bbox_mask_isolation_ablation_v1.py) | PRD 3.1 bbox/mask isolation ablation v1 |
| `run_prd31_fullval_expansion_stage1` | [入口](../scripts/integration/run_prd31_fullval_expansion_stage1.py) · [实现](../src/fashion_multimodal_analysis/integration/run_prd31_fullval_expansion_stage1.py) | PRD 3.1 full-validation propagation expansion — Stage 1 |
| `run_prd31_predicted_roi_attributes_fullval_v1` | [入口](../scripts/integration/run_prd31_predicted_roi_attributes_fullval_v1.py) · [实现](../src/fashion_multimodal_analysis/integration/run_prd31_predicted_roi_attributes_fullval_v1.py) | PRD 3.1 full-validation predicted-ROI -> 3.1.3 attribute propagation. |
| `run_prd31_predicted_roi_attributes_v1` | [入口](../scripts/integration/run_prd31_predicted_roi_attributes_v1.py) · [实现](../src/fashion_multimodal_analysis/integration/run_prd31_predicted_roi_attributes_v1.py) | Run 3.1.3 attributes on the matched PREDICTED ROI manifest without overwriting |
| `run_prd31_predicted_roi_attributes_v1_fixed` | [入口](../scripts/integration/run_prd31_predicted_roi_attributes_v1_fixed.py) · [实现](../src/fashion_multimodal_analysis/integration/run_prd31_predicted_roi_attributes_v1_fixed.py) | Compatibility-fixed wrapper for PRD 3.1 predicted-ROI attribute pilot. |
| `run_prd31_predicted_roi_attributes_v2` | [入口](../scripts/integration/run_prd31_predicted_roi_attributes_v2.py) · [实现](../src/fashion_multimodal_analysis/integration/run_prd31_predicted_roi_attributes_v2.py) | Run 3.1.3 attributes on the matched PREDICTED ROI manifest (v2) without overwriting |

## maintenance

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `check_assets` | [入口](../scripts/maintenance/check_assets.py) · [实现](../src/fashion_multimodal_analysis/maintenance/check_assets.py) | 只读检查 V5/V8 权重、冻结清单、SQLite 索引及外部数据。 |
| `check_project` | [入口](../scripts/maintenance/check_project.py) · [实现](../src/fashion_multimodal_analysis/maintenance/check_project.py) | Check package structure and optional manifest asset availability without loading models. |
| `run_prd31_recovery` | [入口](../scripts/maintenance/run_prd31_recovery.py) · [实现](../src/fashion_multimodal_analysis/maintenance/run_prd31_recovery.py) | One workflow for clean retraining, reviewed tests and reproducible PRD reports. |
| `summarize_saved_results` | [入口](../scripts/maintenance/summarize_saved_results.py) · [实现](../src/fashion_multimodal_analysis/maintenance/summarize_saved_results.py) | 从全部已保存评估建立独立汇总表，不重新运行 GPU。 |
| `verify_prd31_v5_speed` | [入口](../scripts/maintenance/verify_prd31_v5_speed.py) · [实现](../src/fashion_multimodal_analysis/maintenance/verify_prd31_v5_speed.py) | Verify filtered mask transfers on CUDA, then evaluate the existing V5 checkpoint. |

## maintenance/packaging

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `build_gpu_bundle_v3` | [入口](../scripts/maintenance/packaging/build_gpu_bundle_v3.py) · [实现](../src/fashion_multimodal_analysis/maintenance/packaging/build_gpu_bundle_v3.py) | build gpu bundle v3 |
| `build_gpu_upload_bundle` | [入口](../scripts/maintenance/packaging/build_gpu_upload_bundle.py) · [实现](../src/fashion_multimodal_analysis/maintenance/packaging/build_gpu_upload_bundle.py) | build gpu upload bundle |
| `prepare_prd8_gpu_package_v1` | [入口](../scripts/maintenance/packaging/prepare_prd8_gpu_package_v1.py) · [实现](../src/fashion_multimodal_analysis/maintenance/packaging/prepare_prd8_gpu_package_v1.py) | Prepare a compact, portable GPU package for the PRD 3.1.1 Mask R-CNN baseline. |

## segmentation/baselines

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `check_mask2former_forward` | [入口](../scripts/segmentation/baselines/check_mask2former_forward.py) · [实现](../src/fashion_multimodal_analysis/segmentation/baselines/check_mask2former_forward.py) | Check one Mask2Former training forward pass on DeepFashion2. |
| `check_mask2former_train_step` | [入口](../scripts/segmentation/baselines/check_mask2former_train_step.py) · [实现](../src/fashion_multimodal_analysis/segmentation/baselines/check_mask2former_train_step.py) | Check one Mask2Former optimization step on DeepFashion2. |
| `run_mask2former_baseline` | [入口](../scripts/segmentation/baselines/run_mask2former_baseline.py) · [实现](../src/fashion_multimodal_analysis/segmentation/baselines/run_mask2former_baseline.py) | Run a Mask2Former pretrained baseline on one DeepFashion2 image. |
| `test_mask2former_training_input` | [入口](../scripts/segmentation/baselines/check_mask2former_training_input.py) · [实现](../src/fashion_multimodal_analysis/segmentation/baselines/check_mask2former_training_input.py) | Test Mask2Former preprocessing with one DeepFashion2 sample. |

## segmentation/diagnostics

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `infer_small_crop` | [入口](../scripts/segmentation/diagnostics/infer_small_crop.py) · [实现](../src/fashion_multimodal_analysis/segmentation/diagnostics/infer_small_crop.py) | infer small crop |
| `test_resize_crop_infer` | [入口](../scripts/segmentation/diagnostics/diagnose_resized_crop_inference.py) · [实现](../src/fashion_multimodal_analysis/segmentation/diagnostics/diagnose_resized_crop_inference.py) | test resize crop infer |
| `test_small_crop_resize` | [入口](../scripts/segmentation/diagnostics/diagnose_small_crop_resize.py) · [实现](../src/fashion_multimodal_analysis/segmentation/diagnostics/diagnose_small_crop_resize.py) | test small crop resize |

## segmentation/evaluation

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `compare_prd_8class_balanced_vs_nobalance` | [入口](../scripts/segmentation/evaluation/compare_prd_8class_balanced_vs_nobalance.py) · [实现](../src/fashion_multimodal_analysis/segmentation/evaluation/compare_prd_8class_balanced_vs_nobalance.py) | compare prd 8class balanced vs nobalance |
| `eval_prd31_core_regression_v2_balanced_v1` | [入口](../scripts/segmentation/evaluation/eval_prd31_core_regression_v2_balanced_v1.py) · [实现](../src/fashion_multimodal_analysis/segmentation/evaluation/eval_prd31_core_regression_v2_balanced_v1.py) | eval prd31 core regression v2 balanced v1 |
| `eval_prd_8class_maskrcnn_baseline_v1` | [入口](../scripts/segmentation/evaluation/eval_prd_8class_maskrcnn_baseline_v1.py) · [实现](../src/fashion_multimodal_analysis/segmentation/evaluation/eval_prd_8class_maskrcnn_baseline_v1.py) | eval prd 8class maskrcnn baseline v1 |
| `eval_prd_8class_maskrcnn_baseline_v2` | [入口](../scripts/segmentation/evaluation/eval_prd_8class_maskrcnn_baseline_v2.py) · [实现](../src/fashion_multimodal_analysis/segmentation/evaluation/eval_prd_8class_maskrcnn_baseline_v2.py) | eval prd 8class maskrcnn baseline v2 |
| `eval_prd_8class_maskrcnn_v3_a1_highres` | [入口](../scripts/segmentation/evaluation/eval_prd_8class_maskrcnn_v3_a1_highres.py) · [实现](../src/fashion_multimodal_analysis/segmentation/evaluation/eval_prd_8class_maskrcnn_v3_a1_highres.py) | eval prd 8class maskrcnn v3 a1 highres |
| `eval_prd_8class_maskrcnn_v3_a2_smallanchors` | [入口](../scripts/segmentation/evaluation/eval_prd_8class_maskrcnn_v3_a2_smallanchors.py) · [实现](../src/fashion_multimodal_analysis/segmentation/evaluation/eval_prd_8class_maskrcnn_v3_a2_smallanchors.py) | eval prd 8class maskrcnn v3 a2 smallanchors |
| `eval_prd_8class_maskrcnn_v3_a2b_p2small` | [入口](../scripts/segmentation/evaluation/eval_prd_8class_maskrcnn_v3_a2b_p2small.py) · [实现](../src/fashion_multimodal_analysis/segmentation/evaluation/eval_prd_8class_maskrcnn_v3_a2b_p2small.py) | eval prd 8class maskrcnn v3 a2b p2small |
| `eval_prd_8class_maskrcnn_v3_a3_smallsampling` | [入口](../scripts/segmentation/evaluation/eval_prd_8class_maskrcnn_v3_a3_smallsampling.py) · [实现](../src/fashion_multimodal_analysis/segmentation/evaluation/eval_prd_8class_maskrcnn_v3_a3_smallsampling.py) | eval prd 8class maskrcnn v3 a3 smallsampling |
| `eval_prd_8class_maskrcnn_v3_b1_dataexp` | [入口](../scripts/segmentation/evaluation/eval_prd_8class_maskrcnn_v3_b1_dataexp.py) · [实现](../src/fashion_multimodal_analysis/segmentation/evaluation/eval_prd_8class_maskrcnn_v3_b1_dataexp.py) | eval prd 8class maskrcnn v3 b1 dataexp |
| `sweep_prd_8class_maskrcnn_thresholds_v1` | [入口](../scripts/segmentation/evaluation/sweep_prd_8class_maskrcnn_thresholds_v1.py) · [实现](../src/fashion_multimodal_analysis/segmentation/evaluation/sweep_prd_8class_maskrcnn_thresholds_v1.py) | Run a score-threshold sweep on the already-trained PRD 3.1.1 8-class |

## segmentation/training

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `run_prd31_v5_full_targets` | [入口](../scripts/segmentation/training/run_prd31_v5_full_targets.py) · [实现](../src/fashion_multimodal_analysis/segmentation/training/run_prd31_v5_full_targets.py) | Complete annotations on existing training images, train v5, compare on development images. |
| `run_prd31_v6_coverage` | [入口](../scripts/segmentation/training/run_prd31_v6_coverage.py) · [实现](../src/fashion_multimodal_analysis/segmentation/training/run_prd31_v6_coverage.py) | Add complete DF2 training images for all 13 native labels; compare V5/V6 on development. |
| `run_prd31_v7_fp_heads` | [入口](../scripts/segmentation/training/run_prd31_v7_fp_heads.py) · [实现](../src/fashion_multimodal_analysis/segmentation/training/run_prd31_v7_fp_heads.py) | Fine-tune V5 ROI heads for 3 epochs on its existing Fashionpedia training images. |
| `run_prd31_v8_full_dataset` | [入口](../scripts/segmentation/training/run_prd31_v8_full_dataset.py) · [实现](../src/fashion_multimodal_analysis/segmentation/training/run_prd31_v8_full_dataset.py) | 八类原始训练数据：准备索引、从 V5 微调、断点续跑、固定口径评估。 |
| `train_prd_8class_maskrcnn_baseline_v1` | [入口](../scripts/segmentation/training/train_prd_8class_maskrcnn_baseline_v1.py) · [实现](../src/fashion_multimodal_analysis/segmentation/training/train_prd_8class_maskrcnn_baseline_v1.py) | Train PRD 3.1.1 8-class Mask R-CNN baseline. |
| `train_prd_8class_maskrcnn_baseline_v2_balanced` | [入口](../scripts/segmentation/training/train_prd_8class_maskrcnn_baseline_v2_balanced.py) · [实现](../src/fashion_multimodal_analysis/segmentation/training/train_prd_8class_maskrcnn_baseline_v2_balanced.py) | Train PRD 3.1.1 Mask R-CNN baseline v2 using the expanded train_v2 manifest. |
| `train_prd_8class_maskrcnn_baseline_v2_nobalance` | [入口](../scripts/segmentation/training/train_prd_8class_maskrcnn_baseline_v2_nobalance.py) · [实现](../src/fashion_multimodal_analysis/segmentation/training/train_prd_8class_maskrcnn_baseline_v2_nobalance.py) | Train PRD 3.1.1 Mask R-CNN baseline v2 using the expanded train_v2 manifest. |
| `train_prd_8class_maskrcnn_v3_a1_highres` | [入口](../scripts/segmentation/training/train_prd_8class_maskrcnn_v3_a1_highres.py) · [实现](../src/fashion_multimodal_analysis/segmentation/training/train_prd_8class_maskrcnn_v3_a1_highres.py) | Train PRD 3.1.1 Mask R-CNN baseline v2 using the expanded train_v2 manifest. |
| `train_prd_8class_maskrcnn_v3_a2_smallanchors` | [入口](../scripts/segmentation/training/train_prd_8class_maskrcnn_v3_a2_smallanchors.py) · [实现](../src/fashion_multimodal_analysis/segmentation/training/train_prd_8class_maskrcnn_v3_a2_smallanchors.py) | Train PRD 3.1.1 Mask R-CNN baseline v2 using the expanded train_v2 manifest. |
| `train_prd_8class_maskrcnn_v3_a2b_p2small` | [入口](../scripts/segmentation/training/train_prd_8class_maskrcnn_v3_a2b_p2small.py) · [实现](../src/fashion_multimodal_analysis/segmentation/training/train_prd_8class_maskrcnn_v3_a2b_p2small.py) | Train PRD 3.1.1 Mask R-CNN baseline v2 using the expanded train_v2 manifest. |
| `train_prd_8class_maskrcnn_v3_a3_smallsampling` | [入口](../scripts/segmentation/training/train_prd_8class_maskrcnn_v3_a3_smallsampling.py) · [实现](../src/fashion_multimodal_analysis/segmentation/training/train_prd_8class_maskrcnn_v3_a3_smallsampling.py) | Train PRD 3.1.1 Mask R-CNN baseline v2 using the expanded train_v2 manifest. |
| `train_prd_8class_maskrcnn_v3_b1_dataexp` | [入口](../scripts/segmentation/training/train_prd_8class_maskrcnn_v3_b1_dataexp.py) · [实现](../src/fashion_multimodal_analysis/segmentation/training/train_prd_8class_maskrcnn_v3_b1_dataexp.py) | Train PRD 3.1.1 Mask R-CNN v3-b1 data expansion using the expanded train_v3 manifest. |
| `train_prd_8class_maskrcnn_v4_clean` | [入口](../scripts/segmentation/training/train_prd_8class_maskrcnn_v4_clean.py) · [实现](../src/fashion_multimodal_analysis/segmentation/training/train_prd_8class_maskrcnn_v4_clean.py) | Train the path-isolated v4 candidate using the existing B1 architecture/settings. Inherits B1 trainer options; --verify-isolation is enabled by default. |

## visualization

| 命令 | 入口 / 实现 | 说明 |
|---|---|---|
| `build_design_audit_contact_sheets_v1` | [入口](../scripts/visualization/build_design_audit_contact_sheets_v1.py) · [实现](../src/fashion_multimodal_analysis/visualization/build_design_audit_contact_sheets_v1.py) | Build visual contact sheets for the 40-row design-attribute manual audit. |
| `build_grounding_annotation_sheet` | [入口](../scripts/visualization/build_grounding_annotation_sheet.py) · [实现](../src/fashion_multimodal_analysis/visualization/build_grounding_annotation_sheet.py) | Build a manual annotation sheet for grounding evaluation. |
| `build_grounding_manual_audit` | [入口](../scripts/visualization/build_grounding_manual_audit.py) · [实现](../src/fashion_multimodal_analysis/visualization/build_grounding_manual_audit.py) | Build a stratified manual audit set for grounding localization. |
| `build_pattern_focus_audit_pack` | [入口](../scripts/visualization/build_pattern_focus_audit_pack.py) · [实现](../src/fashion_multimodal_analysis/visualization/build_pattern_focus_audit_pack.py) | Build an ordered focused-audit image pack for pattern baseline v1. |
| `build_pattern_v2_holdout_audit_pack` | [入口](../scripts/visualization/build_pattern_v2_holdout_audit_pack.py) · [实现](../src/fashion_multimodal_analysis/visualization/build_pattern_v2_holdout_audit_pack.py) | Build a row-numbered image pack for the NEW pattern-v2 holdout audit. |
| `build_pattern_v3_holdout_audit_pack` | [入口](../scripts/visualization/build_pattern_v3_holdout_audit_pack.py) · [实现](../src/fashion_multimodal_analysis/visualization/build_pattern_v3_holdout_audit_pack.py) | Build a row-numbered image pack for the fresh pattern-v3 holdout audit. |
| `build_prd_region_candidate_sheets` | [入口](../scripts/visualization/build_prd_region_candidate_sheets.py) · [实现](../src/fashion_multimodal_analysis/visualization/build_prd_region_candidate_sheets.py) | Build manual-screening candidate sheets for PRD 3.1.2 regions. |
| `generate_grounding_contact_sheets` | [入口](../scripts/visualization/generate_grounding_contact_sheets.py) · [实现](../src/fashion_multimodal_analysis/visualization/generate_grounding_contact_sheets.py) | Generate contact sheets for manual grounding annotations. |
| `visualize_annotation` | [入口](../scripts/visualization/visualize_annotation.py) · [实现](../src/fashion_multimodal_analysis/visualization/visualize_annotation.py) | visualize annotation |
| `visualize_grounding_group_results` | [入口](../scripts/visualization/visualize_grounding_group_results.py) · [实现](../src/fashion_multimodal_analysis/visualization/visualize_grounding_group_results.py) | Visualize Grounding DINO group evaluation predictions. |
| `visualize_result` | [入口](../scripts/visualization/visualize_result.py) · [实现](../src/fashion_multimodal_analysis/visualization/visualize_result.py) | visualize result |
