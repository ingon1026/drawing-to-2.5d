"""
drawing-2p5d — Mask postprocessing
"""

import cv2
import numpy as np

import config


def morphological_cleanup(mask):
    """Close then open to fill gaps and remove noise."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=config.MORPH_CLOSE_ITER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=config.MORPH_OPEN_ITER)
    return mask


def keep_largest_component(mask):
    """Keep only the largest connected component."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    # label 0 is background; find largest among 1..N
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)
    return ((labels == largest_label).astype(np.uint8) * 255)


def fill_holes(mask):
    """Fill interior holes using flood fill from the border."""
    h, w = mask.shape[:2]
    flood = mask.copy()
    fill_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, fill_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, flood_inv)


def smooth_edges(mask, blur_size=5):
    """Slight Gaussian blur + re-threshold for smoother edges."""
    blurred = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return smoothed


def clean_mask(mask):
    """Full postprocessing pipeline."""
    mask = morphological_cleanup(mask)
    mask = keep_largest_component(mask)
    mask = fill_holes(mask)
    mask = smooth_edges(mask)
    return mask
