"""
drawing-2p5d — Depth map & normal map generation

Two backends:
  - pytorch: DPT-hybrid-midas via HuggingFace (high quality, ~800ms on PC)
  - tflite:  MiDaS v2.1 small (fast, ~68ms on PC, mobile-ready)
"""

import os

import cv2
import numpy as np

import config

# --- PyTorch backend ---
_pt_model = None
_pt_processor = None

# --- TFLite backend ---
_tflite_interpreter = None


def load_depth_model(backend=None):
    """Load depth model. backend='pytorch' or 'tflite'."""
    backend = backend or config.DEPTH_BACKEND
    if backend == "tflite":
        _load_tflite()
    else:
        _load_pytorch()


def _load_pytorch():
    global _pt_model, _pt_processor
    if _pt_model is None:
        import torch
        from transformers import DPTForDepthEstimation, DPTImageProcessor
        name = config.DEPTH_MODEL_NAME
        print(f"  Loading depth model (PyTorch): {name} ...")
        _pt_processor = DPTImageProcessor.from_pretrained(name)
        _pt_model = DPTForDepthEstimation.from_pretrained(name)
        _pt_model.eval()


def _load_tflite():
    global _tflite_interpreter
    if _tflite_interpreter is None:
        import tensorflow as tf
        path = config.DEPTH_TFLITE_PATH
        if not os.path.exists(path):
            _download_tflite_model(path)
        print(f"  Loading depth model (TFLite): {path} ...")
        _tflite_interpreter = tf.lite.Interpreter(model_path=path)
        _tflite_interpreter.allocate_tensors()


def _download_tflite_model(path):
    """Download MiDaS v2.1 small TFLite model."""
    import urllib.request
    url = config.DEPTH_TFLITE_URL
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"  Downloading depth TFLite model ...")
    urllib.request.urlretrieve(url, path)
    print(f"  Done.")


def estimate_depth(image_bgr, mask=None, backend=None):
    """Estimate relative depth from a BGR image.

    Args:
        image_bgr: BGR numpy array
        mask: optional binary mask (0/255)
        backend: 'pytorch' or 'tflite' (default: config.DEPTH_BACKEND)

    Returns:
        depth_map: float32 [0, 1], same size as input
    """
    backend = backend or config.DEPTH_BACKEND
    if backend == "tflite":
        depth = _estimate_tflite(image_bgr)
    else:
        depth = _estimate_pytorch(image_bgr)

    if mask is not None:
        depth[mask == 0] = 0.0

    return depth.astype(np.float32)


def _estimate_pytorch(image_bgr):
    import torch
    _load_pytorch()
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = _pt_processor(images=image_rgb, return_tensors="pt")

    with torch.no_grad():
        outputs = _pt_model(**inputs)
        predicted_depth = outputs.predicted_depth

    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)

    return depth


def _estimate_tflite(image_bgr):
    _load_tflite()
    h, w = image_bgr.shape[:2]
    inp = _tflite_interpreter.get_input_details()[0]
    out = _tflite_interpreter.get_output_details()[0]
    in_h, in_w = inp["shape"][1], inp["shape"][2]

    # Preprocess: BGR → RGB, resize, normalize to [0, 1]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (in_w, in_h)).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(resized, 0)

    _tflite_interpreter.set_tensor(inp["index"], input_tensor)
    _tflite_interpreter.invoke()
    depth = _tflite_interpreter.get_tensor(out["index"]).squeeze()

    # Normalize to [0, 1]
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)

    # Resize back to original
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    return depth


def depth_to_normal(depth, strength=1.0):
    """Convert a depth map to a normal map using Sobel gradients."""
    dz_dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) * strength
    dz_dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) * strength

    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(depth)

    length = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    length[length < 1e-6] = 1.0
    nx /= length
    ny /= length
    nz /= length

    normal_map = np.zeros((*depth.shape, 3), dtype=np.uint8)
    normal_map[:, :, 0] = ((nx + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    normal_map[:, :, 1] = ((ny + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    normal_map[:, :, 2] = ((nz + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)

    return normal_map
