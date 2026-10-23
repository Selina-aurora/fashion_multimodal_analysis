"""3.1 接口联调：区分预测 ROI 传递和使用 GT 的受控诊断，保留对象 ID 与坐标对应关系。

Reuse historical Mask R-CNN, Grounding DINO, SAM and CLIP implementations.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from PIL import Image

from fashion_multimodal_analysis.benchmarking.prepare_prd31_recovery import (
    PROMPTS,
    applies,
    digest,
)
from fashion_multimodal_analysis.common.paths import resolve_path
from fashion_multimodal_analysis.common.schema import CLASS_TO_ID, ID_TO_CLASS
from fashion_multimodal_analysis.evaluation.protocol import original_box, valid_box
from fashion_multimodal_analysis.integration.pipeline import (
    GarmentROI,
    crop_bounds,
    roi_images,
)
from fashion_multimodal_analysis.integration.region_semantics import (
    REGION_PROMPTS,
    phrase_matches,
    supports_region,
)


from fashion_multimodal_analysis.integration.quality_guards import (
    LOCAL_REGIONS, constrain_local_mask, proposal_area_ratio,
)


class ModelBackends:
    """保存 ModelBackends 的数据/运行职责。

    Attributes:
        torch: 调用方提供的 PyTorch 模块，避免帮助命令触发依赖加载。
        device: 当前计算设备，与输入张量和模型设备保持一致。
        config: 记录字段，使用 segmentation_checkpoint, grounding_model, sam_model, clip_model,
        train_manifest, val_manifest。
        segment_model: segment 模型。
        ground_processor: ground processor。
        ground_model: ground 模型。
        sam_processor: SAM processor。
        sam_model: SAM 模型。
        clip_processor: CLIP processor。
        clip_model: CLIP 模型。
        color_proto: 颜色 proto。
        pattern_proto1: 图案 proto1。
    """

    def __init__(self, config: dict[str, Any], device: str = "cuda") -> None:
        """保存当前对象需要的配置、数据引用和状态。

        Args:
            config: 记录字段，使用 segmentation_checkpoint, grounding_model, sam_model,
            clip_model, train_manifest, val_manifest。
            device: 当前计算设备，与输入张量和模型设备保持一致。

        Raises:
            RuntimeError: CUDA is unavailable
        """
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            CLIPModel,
            CLIPProcessor,
            SamModel,
            SamProcessor,
        )

        from fashion_multimodal_analysis.attributes.color import (
            run_primary_color_baseline_v1 as color,
        )
        from fashion_multimodal_analysis.attributes.design import (
            run_design_attribute_labels_v1 as design,
        )
        from fashion_multimodal_analysis.attributes.pattern import (
            run_pattern_hierarchical_v3 as pattern,
        )
        from fashion_multimodal_analysis.segmentation.modeling import (
            build_evaluation_model,
        )

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.config = config
        checkpoint = resolve_path(config["segmentation_checkpoint"])
        cp = torch.load(checkpoint, map_location="cpu", weights_only=False)
        from fashion_multimodal_analysis.segmentation.training.checkpoint_state import (
            validate_checkpoint,
        )

        validate_checkpoint(
            cp,
            digest(resolve_path(config["train_manifest"])),
            digest(resolve_path(config["val_manifest"])),
            settings=config.get("checkpoint_settings"),
        )
        saved = cp.get("args", {})
        self.segment_model = build_evaluation_model(
            int(saved.get("min_size", 640)), int(saved.get("max_size", 1024))
        )
        self.segment_model.load_state_dict(cp["model_state_dict"])
        self.segment_model.to(self.device).eval()
        del cp
        self.ground_processor = AutoProcessor.from_pretrained(config["grounding_model"])
        self.ground_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                config["grounding_model"]
            )
            .to(self.device)
            .eval()
        )
        self.sam_processor = SamProcessor.from_pretrained(config["sam_model"])
        self.sam_model = (
            SamModel.from_pretrained(config["sam_model"]).to(self.device).eval()
        )
        self.clip_processor = CLIPProcessor.from_pretrained(config["clip_model"])
        self.clip_model = (
            CLIPModel.from_pretrained(config["clip_model"]).to(self.device).eval()
        )
        self.color, self.pattern, self.design = color, pattern, design
        self.color_proto = color.build_color_prototypes(
            self.clip_model, self.clip_processor, self.device
        )
        self.pattern_proto1 = pattern.build_class_prototypes(
            pattern.STAGE1_PROMPTS, self.clip_model, self.clip_processor, self.device
        )
        self.pattern_proto2 = pattern.build_class_prototypes(
            pattern.STAGE2_PROMPTS, self.clip_model, self.clip_processor, self.device
        )
        self.design_proto = {}
        for name, prompts in [
            ("sleeve_length", design.SLEEVE_PROMPTS),
            ("neckline", design.NECKLINE_PROMPTS),
            ("fashion_style", design.STYLE_PROMPTS),
            *[("fit_" + c, p) for c, p in design.FIT_PROMPTS_BY_CATEGORY.items()],
        ]:
            self.design_proto[name] = pattern.build_class_prototypes(
                {k: [v] for k, v in prompts.items()},
                self.clip_model,
                self.clip_processor,
                self.device,
            )

    def synchronize(self) -> None:
        """synchronize。"""
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()

    def measure(self, action: Any) -> tuple[Any, ...]:
        """measure。

        Args:
            action: action。

        Returns:
            按顺序返回 result 等结果。
        """
        self.synchronize()
        start = time.perf_counter()
        result = action()
        self.synchronize()
        return result, (time.perf_counter() - start) * 1000

    def segment(self, image: Any) -> Any:
        """segment。

        Args:
            image: 本步骤处理的图像对象。

        Returns:
            本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
        """
        from fashion_multimodal_analysis.evaluation.runtime import predict_timed

        def infer() -> Any:
            """推理。

            Returns:
                返回 result，由函数体中同名变量的计算/收集过程得到。
            """
            boxes, labels, scores, masks, _ = predict_timed(
                image,
                self.segment_model,
                self.device,
                self.config["score_threshold"],
                self.config["mask_threshold"],
            )
            result = []
            for index, (box, label, score, mask) in enumerate(
                zip(boxes, labels, scores, masks)
            ):
                if int(label) not in ID_TO_CLASS or not valid_box(box):
                    continue
                clipped = [
                    max(0.0, float(box[0])),
                    max(0.0, float(box[1])),
                    min(float(image.width), float(box[2])),
                    min(float(image.height), float(box[3])),
                ]
                if not valid_box(clipped):
                    continue
                result.append(
                    GarmentROI(
                        f"pred_{index:04d}",
                        ID_TO_CLASS[int(label)],
                        float(score),
                        list(map(float, clipped)),
                        mask,
                    )
                )
            return result

        return self.measure(infer)

    def ground(self, image: Any, roi: Any, region: str) -> Any:
        """ground。

        Args:
            image: 本步骤处理的图像对象。
            roi: roi。
            region: 区域。

        Returns:
            本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
        """
        def infer() -> dict[str, Any]:
            """推理。

            Returns:
                结果字典，主要字段为 status, target_region, model_phrase, score, bbox, mask,
                proposal_bbox, sam_score, coordinate_space, crop_origin。

            Raises:
                ValueError: SAM mask shape differs from garment crop
            """
            strict_regions = self.config.get("strict_region_grounding", False)
            if strict_regions and not supports_region(roi.category, region):
                return {
                    "status": "not_applicable",
                    "target_region": region,
                    "bbox": None,
                    "mask": None,
                    "routing_note": "Structural clothing query excluded for predicted category; not an absence label.",
                }
            from fashion_multimodal_analysis.grounding.refinement.evaluate_sam_refinement_9case import (
                run_sam,
            )
            from fashion_multimodal_analysis.grounding.refinement.run_shoulder_waist_pattern_groundingdino_refinement_v1 import (
                run_grounding_dino,
            )
            crop, garment_mask, origin = roi_images(image, roi)
            isolated = Image.composite(
                crop, Image.new("RGB", crop.size, "white"), garment_mask
            )
            detections, _ = run_grounding_dino(
                # In the strict diagnostic path retain neck/skin/background context.
                # The predicted garment mask still constrains the SAM output below.
                crop if strict_regions else isolated,
                REGION_PROMPTS[region] if strict_regions else PROMPTS[region],
                self.config["grounding_box_threshold"],
                self.config["grounding_text_threshold"],
                self.ground_processor,
                self.ground_model,
                self.device,
            )
            detections = [
                d
                for d in detections
                if valid_box(d["box"]) and math.isfinite(d["score"])
            ]
            if strict_regions:
                rejected = [d for d in detections if not phrase_matches(region, d["label"])]
                detections = [d for d in detections if phrase_matches(region, d["label"])]
                if not detections and rejected:
                    return {
                        "status": "rejected_wrong_phrase",
                        "target_region": region,
                        "bbox": None,
                        "mask": None,
                        "rejected_proposals": rejected,
                    }
            if not detections:
                return {
                    "status": "missed",
                    "target_region": region,
                    "bbox": None,
                    "mask": None,
                }
            # DINO 的输出顺序不是排序接口；显式按分数选择合法候选。
            # 阈值只控制几何保护，不改变 PRD 验收目标。
            detections.sort(key=lambda candidate: -float(candidate["score"]))
            geometry_rejections = []
            chosen = None
            limit = float(self.config.get("max_local_region_crop_area_ratio", 0.65))
            if not 0 < limit <= 1:
                raise ValueError("max_local_region_crop_area_ratio must be in (0, 1]")
            for candidate in detections:
                try:
                    candidate_box = crop_bounds(candidate["box"], crop.size)
                except ValueError:
                    geometry_rejections.append({"reason": "outside_crop", "proposal": candidate})
                    continue
                area_ratio = proposal_area_ratio(candidate_box, crop.size)
                if strict_regions and region in LOCAL_REGIONS and area_ratio > limit:
                    geometry_rejections.append({
                        "reason": "local_region_covers_whole_garment",
                        "crop_area_ratio": float(area_ratio), "proposal": candidate,
                    })
                    continue
                chosen = candidate, candidate_box
                break
            if chosen is None:
                return {
                    "status": "rejected_region_geometry", "target_region": region,
                    "bbox": None, "mask": None,
                    "rejected_proposals": geometry_rejections,
                    "semantic_review_status": "pending",
                    "semantic_note": "Rejected geometry is not an absence label or a correct detection.",
                }
            proposal, local_box = chosen
            mask, sam_score, _ = run_sam(
                crop, local_box, self.sam_processor, self.sam_model, self.device
            )
            raw_mask = np.asarray(mask, dtype=bool)
            garment_array = np.asarray(garment_mask) > 0
            if raw_mask.shape != garment_array.shape:
                raise ValueError("SAM mask shape differs from garment crop")
            mask = raw_mask & garment_array
            raw_garment_pixels = int(mask.sum())
            if strict_regions and region in LOCAL_REGIONS:
                mask = constrain_local_mask(raw_mask, garment_mask, local_box)
            clipped_pixels = raw_garment_pixels - int(mask.sum())
            # Both the local mask and bbox stay bound to this crop origin. No GT is used.
            ys, xs = np.nonzero(mask)
            if not len(xs):
                return {
                    "status": "empty_mask",
                    "target_region": region,
                    "bbox": None,
                    "mask": None,
                }
            refined = [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            ]
            full = np.zeros((image.height, image.width), dtype=np.uint8)
            x1, y1, x2, y2 = origin
            if mask.shape != (y2 - y1, x2 - x1):
                raise ValueError("SAM mask shape differs from garment crop")
            full[y1:y2, x1:x2] = mask
            return {
                "status": "detected",
                "target_region": region,
                "model_phrase": proposal["label"],
                "score": proposal["score"],
                "bbox": original_box(refined, origin),
                "mask": full,
                "proposal_bbox": original_box(local_box, origin),
                "proposal_crop_area_ratio": float(proposal_area_ratio(local_box, crop.size)),
                "local_mask_constrained_to_proposal": bool(strict_regions and region in LOCAL_REGIONS),
                "sam_pixels_removed_outside_proposal": int(clipped_pixels),
                "geometry_rejections": geometry_rejections,
                "sam_score": float(sam_score),
                "coordinate_space": "original_image",
                "crop_origin": origin,
                "semantic_target_correct": None,
                "semantic_review_status": "pending",
                "grounding_input_view": "raw_crop" if strict_regions else "masked_crop",
                "semantic_note": "Strict semantic correctness needs reviewed output; no automatic manual grade is fabricated.",
            }

        return self.measure(infer)

    def attributes(
        self,
        image: Any,
        roi: Any,
        regions: Any,
        names: list[str],
        controlled: bool = False,
    ) -> Any:
        """属性。

        Args:
            image: 本步骤处理的图像对象。
            roi: roi。
            regions: 区域。
            names: names。
            controlled: controlled。

        Returns:
            本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
        """

        def infer() -> Any:
            """推理。

            Returns:
                返回 result，由函数体中同名变量的计算/收集过程得到。
            """
            crop, mask, origin = roi_images(image, roi)
            whole = Image.composite(crop, Image.new("RGB", crop.size, "white"), mask)
            result = {}
            cached_feature = None
            for name in names:
                if not applies(name, roi.category):
                    result[name] = {"status": "not_applicable", "label": ""}
                    continue
                if name == "primary_color":
                    feature = self.color.robust_garment_embedding(
                        crop, mask, self.clip_model, self.clip_processor, self.device
                    )
                    pred = self.color.classify_color(
                        feature, *self.color_proto, self.clip_model
                    )
                    result[name] = {
                        "status": "predicted",
                        "label": pred["primary_color"],
                        "confidence": pred["top1_score"],
                        "roi_provenance": roi.provenance,
                    }
                elif name == "pattern":
                    if cached_feature is None:
                        cached_feature = self.pattern.encode_image(
                            whole, self.clip_model, self.clip_processor, self.device
                        )
                    gate = self.pattern.classify_with_prototypes(
                        cached_feature, *self.pattern_proto1, self.clip_model
                    )
                    if (
                        self.pattern.binary_route(gate["top1_label"])
                        == "visible_pattern"
                    ):
                        pred = self.pattern.classify_with_prototypes(
                            cached_feature, *self.pattern_proto2, self.clip_model
                        )
                        label = pred["top1_label"]
                        confidence = pred["top1_score"]
                    else:
                        label = "solid"
                        confidence = gate["top1_score"]
                    result[name] = {
                        "status": "predicted",
                        "label": label,
                        "confidence": confidence,
                        "stage1_bucket": gate["top1_label"],
                        "roi_provenance": roi.provenance,
                    }
                else:
                    feature = cached_feature
                    provenance = roi.provenance
                    if name == "neckline":
                        if controlled:
                            local = self.design.neckline_crop(crop, mask)
                            provenance = (
                                "controlled_gt_garment_roi_with_fixed_neckline_proxy"
                            )
                        else:
                            collar = regions.get("collar", {})
                            if collar.get("status") != "detected" or not collar.get(
                                "bbox"
                            ):
                                result[name] = {
                                    "status": "localization_missing",
                                    "label": "",
                                    "roi_provenance": "predicted_collar_missing",
                                }
                                continue
                            bounds = crop_bounds(collar["bbox"], image.size)
                            part_mask = Image.fromarray(
                                (np.asarray(collar["mask"]) > 0).astype("uint8") * 255
                            ).crop(bounds)
                            local = Image.composite(
                                image.crop(bounds),
                                Image.new("RGB", part_mask.size, "white"),
                                part_mask,
                            )
                            provenance = "3_1_2_predicted_collar_roi"
                        feature = self.pattern.encode_image(
                            local, self.clip_model, self.clip_processor, self.device
                        )
                        # 仅新候选启用；默认和 controlled GT 路径保持原行为。
                        # 保存原裁剪预测，与保留皮肤/边界上下文的原图裁剪配对。
                        if not controlled and self.config.get("neckline_context_pair", False):
                            from fashion_multimodal_analysis.integration.attribute_context import (
                                neckline_context_bounds, paired_neckline_result,
                            )
                            context_bounds = neckline_context_bounds(
                                collar["bbox"], roi.bbox, image.size,
                                float(self.config.get("neckline_context_expansion", 2.0)),
                            )
                            context_feature = self.pattern.encode_image(
                                image.crop(context_bounds), self.clip_model,
                                self.clip_processor, self.device,
                            )
                            legacy_pred = self.pattern.classify_with_prototypes(
                                feature, *self.design_proto["neckline"], self.clip_model,
                            )
                            context_pred = self.pattern.classify_with_prototypes(
                                context_feature, *self.design_proto["neckline"], self.clip_model,
                            )
                            result[name] = paired_neckline_result(
                                legacy_pred, context_pred, bounds, context_bounds,
                            )
                            continue
                    elif feature is None:
                        cached_feature = self.pattern.encode_image(
                            whole, self.clip_model, self.clip_processor, self.device
                        )
                        feature = cached_feature
                    key = "fit_" + roi.category if name == "silhouette_fit" else name
                    pred = self.pattern.classify_with_prototypes(
                        feature, *self.design_proto[key], self.clip_model
                    )
                    result[name] = {
                        "status": "predicted",
                        "label": pred["top1_label"],
                        "confidence": pred["top1_score"],
                        "roi_provenance": provenance,
                    }
            return result

        return self.measure(infer)
