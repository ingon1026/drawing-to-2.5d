"""MediaPipe Pose Tasks API wrapper. Each frame returns 33 3D landmarks (or None).

Why a thin wrapper: callers can mock this without pulling MediaPipe into
their test setups. This module uses the Tasks API (mp.tasks.vision.PoseLandmarker)
since modern mediapipe wheels (0.10.33+ on Python 3.12) are Tasks-only builds
without the legacy mp.solutions API.

Model: pose_landmarker_lite.task — auto-downloaded on first use to
~/.cache/stickerbook/models/. Lite variant chosen for demo speed (~5MB).

WSL2 GL crash workaround: MediaPipe's PoseLandmarker tries to spin up an
EGL/GL context for image preprocessing even when inference runs on CPU.
On WSL2 (DRI3 unsupported), that GL init segfaults inside the C++ layer.
Unsetting DISPLAY/WAYLAND_DISPLAY before create_from_options forces the
pure CPU path. Done at module import — safe because mediapipe's GL init
is lazy (happens at create_from_options, not at `import mediapipe`).
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

# Scrub display env BEFORE the heavy mediapipe imports so the C++ layer
# never tries to bind an EGL/GL context (segfaults on WSL2). Pose inference
# runs CPU-only via XNNPACK regardless.
os.environ.pop("DISPLAY", None)
os.environ.pop("WAYLAND_DISPLAY", None)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


_MODEL_FILENAME = "pose_landmarker_lite.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


@dataclass(frozen=True)
class PoseLandmarks:
    """33 normalized world-space points (x, y, z)."""
    points: np.ndarray  # shape (33, 3), dtype float32


def _ensure_model(model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / _MODEL_FILENAME
    if not model_path.exists():
        print(f"[pose_estimator] downloading {_MODEL_FILENAME} (~5MB) …")
        urllib.request.urlretrieve(_MODEL_URL, model_path)
    return model_path


class PoseEstimator:
    def __init__(
        self,
        model_dir: Optional[Path] = None,
        num_poses: int = 1,
    ) -> None:
        if model_dir is None:
            model_dir = Path.home() / ".cache" / "stickerbook" / "models"
        model_path = _ensure_model(model_dir)

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=num_poses,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def estimate_batch(
        self, frames: List[np.ndarray]
    ) -> List[Optional[PoseLandmarks]]:
        results: List[Optional[PoseLandmarks]] = []
        for frame_bgr in frames:
            # MediaPipe expects RGB
            frame_rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            res = self._landmarker.detect(mp_image)
            if not res.pose_world_landmarks:
                results.append(None)
                continue
            landmarks = res.pose_world_landmarks[0]  # first detected pose
            pts = np.array(
                [(lm.x, lm.y, lm.z) for lm in landmarks],
                dtype=np.float32,
            )
            results.append(PoseLandmarks(points=pts))
        return results

    def close(self) -> None:
        self._landmarker.close()
