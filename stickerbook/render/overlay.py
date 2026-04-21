from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from detect.candidate_detector import CandidateBox

_COLORS_BGR = {
    "yolo": (0, 255, 0),
    "contour": (255, 0, 0),
}


def draw_candidate_boxes(frame: np.ndarray, boxes: Iterable[CandidateBox]) -> None:
    for box in boxes:
        color = _COLORS_BGR.get(box.source, (255, 255, 255))
        cv2.rectangle(
            frame,
            (box.x, box.y),
            (box.x + box.w, box.y + box.h),
            color,
            thickness=2,
        )
