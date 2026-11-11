"""NON-EXECUTABLE hook sketch for the actual PRD3.2 Qwen host.

Do NOT copy into the 3.1 pipeline verbatim. Bind stage source metadata from
real host objects; do not assign the target ID into all fields by convention.
"""

# from runtime_bridge_v3 import RuntimeGuardBridge
# from verified_guard_loader import load_guard, GUARD_RELATIVE
# from hashlib import sha256
#
# bridge = RuntimeGuardBridge(load_guard(repo_root / GUARD_RELATIVE))
#
# BEFORE calling Qwen for each (case, segmented garment, variant, task):
#   image_bytes     = actual_source_image_file.read_bytes()
#   roi_bytes       = actual_roi_crop_as_transmitted_bytes
#   mask_bytes      = actual_mask_artifact_bytes
#   marked_bytes    = actual_marked_image_bytes_sent_to_qwen
#   origin_from_roi = crop_producer_record.source_instance_id     # independently observed
#   origin_from_mask = mask_producer_record.source_instance_id    # independently observed
#   job_id          = queue_job_handle.id  # host scheduler, NOT model prose
#   request_id = bridge.dispatch(
#       case_id=case_id, instance_id=segmented_instance_id,
#       source_image_bytes=image_bytes, expected_image_sha256=sha256(image_bytes).hexdigest(),
#       bbox_xyxy=predicted_bbox, v5_category=v5_category,
#       roi_source_instance_id=origin_from_roi,
#       mask_source_instance_id=origin_from_mask,
#       roi_crop_bytes=roi_bytes, expected_roi_sha256=sha256(roi_bytes).hexdigest(),
#       mask_bytes=mask_bytes, expected_mask_sha256=sha256(mask_bytes).hexdigest(),
#       marked_image_bytes=marked_bytes, task=task, variant=variant,
#       job_id=job_id, attempt_id=attempt_id,
#       vqa_question=question if task=='vqa' else None)
#
# AFTER completion, use host scheduler IDs, NOT content parsed from Qwen reply:
#   bridge.complete(host_job_id=job_id, response_host_job_id=completed_handle.id,
#                   completed_request_id=request_id, raw_text=model_text)
#
# On DEV batch end, one variant at a time:
#   audit = bridge.finalize_variant('baseline')
#   # Write audit to a NEW directory and retain original V5/Qwen output untouched.
#   # audit['release_allowed'] remains False regardless of flags.
