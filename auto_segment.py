"""
drawing-2p5d — Auto-segmentation for preview overlay

Two backends:
  - contour: fast classical CV (grayscale → binary → contours), realtime 30fps
  - sam2: SAM2 auto-mask (accurate but slow ~1-2s per frame)
"""

import os

import cv2
import numpy as np

import config

# Distinct colors for segment overlay (BGR)
SEGMENT_COLORS = [
    (255, 80, 80),    # blue
    (80, 255, 80),    # green
    (80, 80, 255),    # red
    (255, 255, 80),   # cyan
    (255, 80, 255),   # magenta
    (80, 255, 255),   # yellow
    (200, 150, 80),   # teal
    (80, 150, 200),   # orange
    (200, 80, 200),   # purple
    (150, 200, 80),   # lime
]


# ──────────────────────────────────────
# Contour-based (fast, realtime)
# ──────────────────────────────────────

def generate_masks_contour(image_bgr, min_area=800):
    """Find drawing objects via adaptive threshold + contour detection.

    Works well for dark drawings on white/light paper.
    Returns list of dicts compatible with masks_to_overlay / find_segment_at.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold: handles uneven lighting on paper
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )

    # Morphological close to connect nearby strokes into one blob
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Optional dilate to merge close objects
    closed = cv2.dilate(closed, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = image_bgr.shape[:2]
    total_area = h * w
    masks = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter: too small or too large (background)
        if area < min_area or area > total_area * 0.8:
            continue

        # Create filled mask for this contour
        seg_mask = np.zeros((h, w), dtype=bool)
        cv2.drawContours(seg_mask.view(np.uint8), [cnt], -1, 1, -1)

        x, y, bw, bh = cv2.boundingRect(cnt)
        masks.append({
            "segmentation": seg_mask,
            "area": int(area),
            "bbox": [x, y, bw, bh],
        })

    # Sort by area descending
    masks.sort(key=lambda m: m["area"], reverse=True)
    return masks


# ──────────────────────────────────────
# SAM2-based (accurate, slow)
# ──────────────────────────────────────

_sam2_model = None
_mask_generator = None


def load_sam2():
    """Load SAM2 model and create mask generator. Cached globally."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    global _sam2_model, _mask_generator
    if _sam2_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = os.path.join(os.path.dirname(__file__), config.SAM2_CHECKPOINT)
        print(f"  Loading SAM2 ({device}) ...")
        _sam2_model = build_sam2(
            config.SAM2_CONFIG,
            ckpt,
            device=device,
        )
        _mask_generator = SAM2AutomaticMaskGenerator(
            _sam2_model,
            points_per_side=config.SAM2_POINTS_PER_SIDE,
            pred_iou_thresh=config.SAM2_IOU_THRESH,
            stability_score_thresh=config.SAM2_STABILITY_THRESH,
            min_mask_region_area=config.SAM2_MIN_AREA,
        )
    return _mask_generator


def generate_masks_sam2(image_rgb):
    """Run SAM2 auto-mask on an RGB image."""
    mask_gen = load_sam2()
    masks = mask_gen.generate(image_rgb)
    masks = sorted(masks, key=lambda m: m["area"], reverse=True)
    if len(masks) > 1:
        masks = masks[1:]  # exclude background
    return masks


# ──────────────────────────────────────
# Common utilities
# ──────────────────────────────────────

def masks_to_overlay(image_bgr, masks, alpha=0.4, highlight_idx=None):
    """Create a colored overlay showing all segments."""
    overlay = image_bgr.copy()
    for i, m in enumerate(masks):
        seg = m["segmentation"]
        color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]

        a = 0.6 if i == highlight_idx else alpha
        overlay[seg] = (
            np.clip(overlay[seg].astype(int) * (1 - a) + np.array(color) * a, 0, 255)
            .astype(np.uint8)
        )

    return overlay


def find_segment_at(masks, x, y):
    """Find which segment index contains pixel (x, y).
    Prefers smaller segments if overlapping.
    """
    for i in reversed(range(len(masks))):
        seg = masks[i]["segmentation"]
        if 0 <= y < seg.shape[0] and 0 <= x < seg.shape[1]:
            if seg[y, x]:
                return i
    return -1


def get_segment_center(mask_dict):
    """Get the center point of a segment mask."""
    seg = mask_dict["segmentation"]
    ys, xs = np.where(seg)
    if len(xs) == 0:
        return 0, 0
    return int(xs.mean()), int(ys.mean())
