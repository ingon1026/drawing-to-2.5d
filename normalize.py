"""
drawing-2p5d — Input normalization (white balance, shadow removal)
"""

import cv2
import numpy as np

import config


def white_balance(image_bgr):
    """Simple gray-world white balance."""
    result = image_bgr.astype(np.float32)
    avg_b, avg_g, avg_r = result.mean(axis=(0, 1))
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    if avg_b > 0:
        result[:, :, 0] *= avg_gray / avg_b
    if avg_g > 0:
        result[:, :, 1] *= avg_gray / avg_g
    if avg_r > 0:
        result[:, :, 2] *= avg_gray / avg_r
    return np.clip(result, 0, 255).astype(np.uint8)


def shadow_remove(image_bgr):
    """Shadow removal via divide-by-Gaussian-blur.

    Based on drawing-classifier/preprocess_compare.py _shadow_remove().
    Returns a single-channel normalized grayscale image.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(gray, (config.GAUSSIAN_BLUR_KSIZE, config.GAUSSIAN_BLUR_KSIZE), 0)
    bg[bg < 1] = 1
    normalized = cv2.divide(gray, bg, scale=255)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def normalize_input(image_path):
    """Load an image and apply white balance.

    Returns BGR image suitable for the segmenter (not grayscale).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    image = white_balance(image)
    return image
