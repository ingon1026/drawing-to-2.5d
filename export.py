"""
drawing-2p5d — Export mask and transparent PNG
"""

import os

import cv2
import numpy as np

import config


def export_mask(mask, output_dir=None):
    """Save binary mask as mask.png."""
    output_dir = output_dir or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, config.MASK_FILENAME)
    cv2.imwrite(path, mask)
    return path


def export_object(image_bgr, mask, output_dir=None):
    """Apply mask as alpha channel and save as BGRA PNG."""
    output_dir = output_dir or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    b, g, r = cv2.split(image_bgr)
    bgra = cv2.merge([b, g, r, mask])
    path = os.path.join(output_dir, config.OBJECT_FILENAME)
    cv2.imwrite(path, bgra)
    return path


def export_depth(depth_map, output_dir=None):
    """Save depth map as grayscale PNG (0=far, 255=near)."""
    output_dir = output_dir or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    depth_u8 = (depth_map * 255).clip(0, 255).astype(np.uint8)
    path = os.path.join(output_dir, config.DEPTH_FILENAME)
    cv2.imwrite(path, depth_u8)
    return path


def export_normal(normal_map, output_dir=None):
    """Save normal map as RGB PNG."""
    output_dir = output_dir or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    # normal_map is RGB, cv2 expects BGR
    normal_bgr = cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR)
    path = os.path.join(output_dir, config.NORMAL_FILENAME)
    cv2.imwrite(path, normal_bgr)
    return path


def export_debug_overlay(image_bgr, mask, output_dir=None, alpha=0.4):
    """Save a semi-transparent red overlay on the original image."""
    output_dir = output_dir or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    overlay = image_bgr.copy()
    overlay[mask > 0] = (0, 0, 255)  # red in BGR
    blended = cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)
    path = os.path.join(output_dir, config.DEBUG_OVERLAY_FILENAME)
    cv2.imwrite(path, blended)
    return path
