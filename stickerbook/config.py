import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
CAPTURES_DIR = ASSETS_DIR / "captures"
SAMPLES_DIR = ASSETS_DIR / "samples"
DEBUG_DIR = ASSETS_DIR / "debug"


@dataclass(frozen=True)
class Config:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720

    yolo_weights: str = str(MODELS_DIR / "yolo26n.pt")
    mobile_sam_weights: str = str(MODELS_DIR / "mobile_sam.pt")

    homography_min_inliers: int = 10
    homography_min_inlier_ratio: float = 0.3
    homography_lost_frames_threshold: int = 15

    sam_input_size: int = 1024
    max_concurrent_sam: int = 2

    # Approximate webcam horizontal FOV (degrees). Used to build camera intrinsics K.
    # True value varies by device; 60° is a reasonable default for integrated webcams.
    camera_fov_deg: float = 60.0


DEFAULT = Config()


def approximate_camera_intrinsics(frame_shape: Tuple[int, int], fov_deg: float = DEFAULT.camera_fov_deg) -> np.ndarray:
    """Build a 3x3 camera intrinsics matrix K from frame size and horizontal FOV.

    K = [[f, 0, cx], [0, f, cy], [0, 0, 1]] where f = W / (2 * tan(FOV/2)).
    """
    h, w = frame_shape[:2]
    f = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cx = w / 2.0
    cy = h / 2.0
    return np.array(
        [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
