from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# COCO class ids considered drawing candidates (humanoid / animal).
_DEFAULT_YOLO_CLASSES = frozenset({0, 15, 16, 17, 18, 19, 20, 21, 22, 23})


@dataclass(frozen=True)
class CandidateBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    source: Literal["yolo", "contour"]


def detect_contour_candidates(
    frame: np.ndarray,
    *,
    min_area_ratio: float = 0.003,
    max_area_ratio: float = 0.8,
    adaptive_block_size: int = 41,
    adaptive_c: int = 15,
    closing_kernel: int = 5,
) -> List[CandidateBox]:
    h, w = frame.shape[:2]
    frame_area = h * w
    min_area = frame_area * min_area_ratio
    max_area = frame_area * max_area_ratio

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=adaptive_block_size,
        C=adaptive_c,
    )
    kernel = np.ones((closing_kernel, closing_kernel), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[CandidateBox] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        boxes.append(
            CandidateBox(x=x, y=y, w=bw, h=bh, confidence=1.0, source="contour")
        )
    return boxes


class CandidateDetector:
    def __init__(
        self,
        yolo_weights: Optional[str] = None,
        yolo_conf_threshold: float = 0.25,
        yolo_keep_classes: Optional[frozenset[int]] = None,
        contour_min_area_ratio: float = 0.003,
        contour_max_area_ratio: float = 0.8,
    ) -> None:
        self._yolo = YOLO(yolo_weights) if yolo_weights else None
        self._yolo_conf_threshold = yolo_conf_threshold
        self._yolo_keep_classes = (
            yolo_keep_classes if yolo_keep_classes is not None else _DEFAULT_YOLO_CLASSES
        )
        self._contour_min_area_ratio = contour_min_area_ratio
        self._contour_max_area_ratio = contour_max_area_ratio

    def detect(self, frame: np.ndarray) -> List[CandidateBox]:
        boxes: List[CandidateBox] = []
        if self._yolo is not None:
            boxes.extend(self._detect_yolo(frame))
        boxes.extend(
            detect_contour_candidates(
                frame,
                min_area_ratio=self._contour_min_area_ratio,
                max_area_ratio=self._contour_max_area_ratio,
            )
        )
        return boxes

    def _detect_yolo(self, frame: np.ndarray) -> List[CandidateBox]:
        assert self._yolo is not None
        results = self._yolo(frame, verbose=False)
        boxes: List[CandidateBox] = []
        for result in results:
            if result.boxes is None:
                continue
            for i in range(len(result.boxes)):
                conf = float(result.boxes.conf[i])
                if conf < self._yolo_conf_threshold:
                    continue
                cls = int(result.boxes.cls[i])
                if cls not in self._yolo_keep_classes:
                    continue
                x1, y1, x2, y2 = (int(v) for v in result.boxes.xyxy[i].tolist())
                boxes.append(
                    CandidateBox(
                        x=x1,
                        y=y1,
                        w=x2 - x1,
                        h=y2 - y1,
                        confidence=conf,
                        source="yolo",
                    )
                )
        return boxes
