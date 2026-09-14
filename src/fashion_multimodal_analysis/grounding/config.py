"""Frozen configuration used by the verified-positive Grounding DINO v3 evaluation."""

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_THRESHOLD = 0.30
DEFAULT_ROI_MARGIN_RATIO = 0.05
DEFAULT_LOCAL_ENLARGE_FACTOR = 1.0
DEFAULT_LOCAL_NMS_IOU = 0.50

TARGET_ORDER = ["sleeve", "collar", "button", "zipper"]
SCALE_GROUPS = {
    "sleeve": "larger_part",
    "collar": "larger_part",
    "button": "small_object",
    "zipper": "small_object",
}
CONDITIONS = ["full_image", "roi_conditioned", "local_enlarged"]

# Ratios are relative to the garment ROI. These are coarse deterministic
# spatial priors and are not fine-grained ground-truth boxes.
LOCAL_WINDOW_RULES = {
    "sleeve": [(0.00, 0.02, 0.36, 0.75), (0.64, 0.02, 1.00, 0.75)],
    "collar": [(0.20, 0.00, 0.80, 0.34)],
    "button": [
        (0.28, 0.06, 0.50, 0.94),
        (0.39, 0.06, 0.61, 0.94),
        (0.50, 0.06, 0.72, 0.94),
    ],
    "zipper": [
        (0.28, 0.03, 0.50, 0.97),
        (0.39, 0.03, 0.61, 0.97),
        (0.50, 0.03, 0.72, 0.97),
    ],
}
