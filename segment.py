"""
drawing-2p5d — Interactive Segmenter via TFLite runtime

Runs magic_touch.tflite directly to avoid MediaPipe Tasks API's
OpenGL dependency (which segfaults in WSL2/headless environments).

Model input:  [1, 512, 512, 4] float32  (RGB + keypoint heatmap)
Model output: [1, 512, 512, 1] float32  (segmentation confidence)
"""

import os
import urllib.request

import cv2
import numpy as np
import tensorflow as tf

import config

MODEL_INPUT_SIZE = 512


def download_model_if_needed(model_path=None, url=None):
    """Download the segmenter model if not present."""
    model_path = model_path or config.MODEL_PATH
    url = url or config.MODEL_URL
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"Downloading model to {model_path} ...")
        urllib.request.urlretrieve(url, model_path)
        print("Done.")
    return model_path


def load_segmenter(model_path=None):
    """Load the TFLite interpreter."""
    model_path = model_path or config.MODEL_PATH
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def _make_keypoint_heatmap(h, w, x, y, sigma=10.0):
    """Create a Gaussian heatmap centered at (x, y) in normalized coords."""
    cx = x * w
    cy = y * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return heatmap


def segment_at_point(interpreter, image_bgr, x, y, threshold=None):
    """Run interactive segmentation at a normalized (x, y) point.

    Args:
        interpreter: TFLite Interpreter instance
        image_bgr: BGR numpy array (cv2.imread format)
        x: normalized x coordinate (0.0 ~ 1.0, left to right)
        y: normalized y coordinate (0.0 ~ 1.0, top to bottom)
        threshold: confidence threshold for binary mask

    Returns:
        binary mask as uint8 numpy array (0 or 255), same size as input image
    """
    if threshold is None:
        threshold = config.CONFIDENCE_THRESHOLD

    orig_h, orig_w = image_bgr.shape[:2]
    size = MODEL_INPUT_SIZE

    # Prepare RGB input, resize to 512x512, normalize to [0, 1]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (size, size)).astype(np.float32) / 255.0

    # Create keypoint heatmap channel
    heatmap = _make_keypoint_heatmap(size, size, x, y)

    # Stack: [H, W, 4] = RGB + heatmap
    input_tensor = np.concatenate([resized, heatmap[:, :, np.newaxis]], axis=2)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # [1, 512, 512, 4]

    # Run inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])  # [1, 512, 512, 1]

    # Squeeze and threshold
    confidence = output.squeeze()  # [512, 512]
    binary_small = (confidence > threshold).astype(np.uint8) * 255

    # Resize back to original image size
    binary = cv2.resize(binary_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return binary
