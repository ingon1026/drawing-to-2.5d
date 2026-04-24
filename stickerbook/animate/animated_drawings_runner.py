"""Runs AnimatedDrawings image_to_animation.py and parses the result."""
from __future__ import annotations

import numpy as np


def composite_on_white_bg(texture_bgra: np.ndarray) -> np.ndarray:
    """Alpha-composite an RGBA/BGRA sticker over solid white, returning BGR uint8.

    AnimatedDrawings expects a full-opaque image. Transparent pixels become white.
    """
    tex = np.asarray(texture_bgra, dtype=np.float32)
    if tex.shape[-1] != 4:
        raise ValueError("texture_bgra must have 4 channels (BGRA)")
    rgb = tex[..., :3]
    alpha = tex[..., 3:4] / 255.0
    white = np.full_like(rgb, 255.0)
    out = rgb * alpha + white * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)
