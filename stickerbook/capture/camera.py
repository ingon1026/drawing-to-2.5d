from __future__ import annotations

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, index: int = 0) -> None:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise CameraError(f"failed to open camera at index {index}")

    def read(self) -> np.ndarray:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError("camera read failed")
        return frame

    def release(self) -> None:
        self._cap.release()
