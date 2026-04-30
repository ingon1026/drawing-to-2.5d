"""Frame buffer for M-key triggered motion recording.

Main loop calls add_frame() each tick; toggle handlers call start()/stop().
"""
from __future__ import annotations

from typing import List

import numpy as np


class FrameRecorder:
    def __init__(self) -> None:
        self._recording: bool = False
        self._buffer: List[np.ndarray] = []

    def start(self) -> None:
        self._recording = True
        self._buffer = []

    def stop(self) -> List[np.ndarray]:
        self._recording = False
        frames = self._buffer
        self._buffer = []
        return frames

    def is_recording(self) -> bool:
        return self._recording

    def add_frame(self, frame: np.ndarray) -> None:
        if not self._recording:
            return
        self._buffer.append(frame.copy())
