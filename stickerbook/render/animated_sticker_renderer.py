"""Plays back an AnimatedDrawings video as a BGRA frame source for billboard rendering.

Per M9.1 (docs/M9_1_OUTPUT_FORMAT.md): AD output videos have WHITE background;
chroma-key treats near-white pixels (all channels >= threshold) as transparent.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class AnimatedStickerRenderer:
    def __init__(
        self,
        video_path: Path,
        chroma_key_threshold: int = 240,
    ) -> None:
        self._video_path = Path(video_path)
        self._cap = cv2.VideoCapture(str(self._video_path))
        if not self._cap.isOpened():
            raise IOError(f"cannot open video: {self._video_path}")
        self._chroma_threshold = int(chroma_key_threshold)

    def next_frame_bgra(self) -> np.ndarray:
        ok, frame_bgr = self._cap.read()
        if not ok or frame_bgr is None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame_bgr = self._cap.read()
            if not ok or frame_bgr is None:
                raise IOError(f"video has no decodable frames: {self._video_path}")
        return self._bgr_to_bgra_chroma(frame_bgr)

    def _bgr_to_bgra_chroma(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        bgra = np.zeros((h, w, 4), dtype=np.uint8)
        bgra[..., :3] = bgr
        # Alpha 0 where all BGR channels >= threshold (white chroma key)
        is_bg = np.all(bgr >= self._chroma_threshold, axis=-1)
        bgra[..., 3] = np.where(is_bg, 0, 255).astype(np.uint8)
        return bgra

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
