"""
drawing-2p5d — Depth map & normal map generation

Uses MiDaS (via HuggingFace transformers) for monocular depth estimation,
then derives a normal map from the depth map via Sobel gradients.
"""

import cv2
import numpy as np
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor


_model = None
_processor = None


def load_depth_model(model_name=None):
    """Load MiDaS/DPT depth model. Caches globally for reuse."""
    import config as cfg
    global _model, _processor
    if _model is None:
        model_name = model_name or cfg.DEPTH_MODEL_NAME
        print(f"  Loading depth model: {model_name} ...")
        _processor = DPTImageProcessor.from_pretrained(model_name)
        _model = DPTForDepthEstimation.from_pretrained(model_name)
        _model.eval()
    return _model, _processor


def estimate_depth(image_bgr, mask=None):
    """Estimate relative depth from a BGR image.

    Args:
        image_bgr: BGR numpy array
        mask: optional binary mask (0/255). If provided, depth outside mask is zeroed.

    Returns:
        depth_map: float32 numpy array, normalized to [0, 1], same size as input.
                   Higher values = closer to camera.
    """
    model, processor = load_depth_model()
    h, w = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    # Normalize to [0, 1]
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)

    # Apply mask
    if mask is not None:
        depth[mask == 0] = 0.0

    return depth.astype(np.float32)


def depth_to_normal(depth, strength=1.0):
    """Convert a depth map to a normal map using Sobel gradients.

    Args:
        depth: float32 [0, 1] depth map
        strength: how pronounced the normals are (higher = more dramatic)

    Returns:
        normal_map: uint8 (H, W, 3) RGB normal map.
                    R = X normal, G = Y normal, B = Z normal.
                    Neutral (flat) = (128, 128, 255).
    """
    # Sobel gradients
    dz_dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) * strength
    dz_dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) * strength

    # Normal vector: (-dz/dx, -dz/dy, 1), then normalize
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(depth)

    length = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    length[length < 1e-6] = 1.0
    nx /= length
    ny /= length
    nz /= length

    # Map [-1, 1] to [0, 255]
    normal_map = np.zeros((*depth.shape, 3), dtype=np.uint8)
    normal_map[:, :, 0] = ((nx + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)  # R = X
    normal_map[:, :, 1] = ((ny + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)  # G = Y
    normal_map[:, :, 2] = ((nz + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)  # B = Z

    return normal_map
