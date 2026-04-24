"""Rotating-dots spinner drawn over a sticker region while AD is processing."""
from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np


def draw_spinner(
    frame: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    phase: float,
    num_dots: int = 8,
    dot_radius: int = 3,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    cx, cy = int(center[0]), int(center[1])
    for i in range(num_dots):
        theta = phase + (i * 2.0 * math.pi / num_dots)
        x = int(cx + radius * math.cos(theta))
        y = int(cy + radius * math.sin(theta))
        # Fade per dot by index (gives rotation illusion)
        fade = int(255 * (i + 1) / num_dots)
        c = tuple(int(v * fade / 255) for v in color)
        cv2.circle(frame, (x, y), dot_radius, c, thickness=-1)
