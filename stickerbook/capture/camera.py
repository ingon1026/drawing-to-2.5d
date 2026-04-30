from __future__ import annotations

from typing import Union

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


Source = Union[int, str]


class Camera:
    def __init__(
        self,
        source: Source = 0,
        *,
        loop_video: bool = True,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self._source = source
        self._is_file = isinstance(source, str)
        self._loop_video = loop_video if self._is_file else False
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise CameraError(f"failed to open camera at source {source!r}")
        if not self._is_file:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray:
        ok, frame = self._cap.read()
        if (not ok or frame is None) and self._loop_video:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError("camera read failed")
        return frame

    def release(self) -> None:
        self._cap.release()
