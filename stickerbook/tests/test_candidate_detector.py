from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from detect.candidate_detector import (
    CandidateBox,
    CandidateDetector,
    detect_contour_candidates,
)


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)
        self.cls = torch.tensor(cls, dtype=torch.int64)

    def __len__(self):
        return len(self.xyxy)


class _FakeResult:
    def __init__(self, xyxy=None, conf=None, cls=None):
        self.boxes = _FakeBoxes(xyxy, conf, cls) if xyxy is not None else None


def _white_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def test_contour_detects_dark_rectangle_on_white_background() -> None:
    frame = _white_frame()
    frame[150:250, 200:300] = 0  # 100x100 dark square

    boxes = detect_contour_candidates(frame, min_area_ratio=0.001, max_area_ratio=0.8)

    assert any(b.source == "contour" for b in boxes)
    target = next(b for b in boxes if abs(b.w - 100) <= 3 and abs(b.h - 100) <= 3)
    assert abs(target.x - 200) <= 3
    assert abs(target.y - 150) <= 3


def test_contour_detects_faint_sketch_when_darker_objects_also_present() -> None:
    """Tri-modal scene: bright paper + mid-tone pencil sketch + dark background.

    Global Otsu splits at ~midpoint and loses the faint sketch; adaptive threshold
    catches it locally.
    """
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    frame[:, 500:] = 30  # dark background region (simulating keyboard/monitor)
    cv2.rectangle(frame, (200, 150), (300, 250), (170, 170, 170), thickness=2)

    boxes = detect_contour_candidates(frame, min_area_ratio=0.001, max_area_ratio=0.5)

    has_sketch_box = any(
        abs(b.x - 200) <= 15
        and abs(b.y - 150) <= 15
        and abs(b.w - 100) <= 25
        and abs(b.h - 100) <= 25
        for b in boxes
    )
    assert has_sketch_box, f"expected a box near sketch (200,150,100x100); got {boxes}"


def test_contour_ignores_shapes_smaller_than_min_ratio() -> None:
    frame = _white_frame()
    frame[100:105, 100:105] = 0  # tiny 5x5

    boxes = detect_contour_candidates(frame, min_area_ratio=0.01, max_area_ratio=0.8)

    assert len(boxes) == 0


def test_contour_ignores_shapes_larger_than_max_ratio() -> None:
    frame = _white_frame()
    frame[10:-10, 10:-10] = 0  # fills ~95% of frame

    boxes = detect_contour_candidates(frame, min_area_ratio=0.001, max_area_ratio=0.5)

    for b in boxes:
        assert b.w * b.h < frame.shape[0] * frame.shape[1] * 0.5


@patch("detect.candidate_detector.YOLO")
def test_yolo_returns_boxes_for_kept_class(mock_YOLO: MagicMock) -> None:
    mock_model = MagicMock()
    mock_YOLO.return_value = mock_model
    mock_model.return_value = [
        _FakeResult(xyxy=[[100.0, 100.0, 200.0, 250.0]], conf=[0.8], cls=[0]),
    ]

    det = CandidateDetector(
        yolo_weights="fake.pt",
        yolo_conf_threshold=0.25,
        yolo_keep_classes=frozenset({0}),
        contour_min_area_ratio=0.99,  # disable contour via impossible ratio
    )
    boxes = [b for b in det.detect(_white_frame()) if b.source == "yolo"]

    assert len(boxes) == 1
    b = boxes[0]
    assert (b.x, b.y, b.w, b.h) == (100, 100, 100, 150)
    assert b.confidence == pytest.approx(0.8)


@patch("detect.candidate_detector.YOLO")
def test_yolo_filters_low_confidence(mock_YOLO: MagicMock) -> None:
    mock_model = MagicMock()
    mock_YOLO.return_value = mock_model
    mock_model.return_value = [
        _FakeResult(xyxy=[[10.0, 10.0, 60.0, 60.0]], conf=[0.1], cls=[0]),
    ]

    det = CandidateDetector(
        yolo_weights="fake.pt",
        yolo_conf_threshold=0.25,
        yolo_keep_classes=frozenset({0}),
        contour_min_area_ratio=0.99,
    )
    boxes = [b for b in det.detect(_white_frame()) if b.source == "yolo"]

    assert len(boxes) == 0


@patch("detect.candidate_detector.YOLO")
def test_yolo_filters_unwanted_classes(mock_YOLO: MagicMock) -> None:
    mock_model = MagicMock()
    mock_YOLO.return_value = mock_model
    mock_model.return_value = [
        _FakeResult(xyxy=[[10.0, 10.0, 60.0, 60.0]], conf=[0.9], cls=[2]),  # car
    ]

    det = CandidateDetector(
        yolo_weights="fake.pt",
        yolo_conf_threshold=0.25,
        yolo_keep_classes=frozenset({0}),  # only person
        contour_min_area_ratio=0.99,
    )
    boxes = [b for b in det.detect(_white_frame()) if b.source == "yolo"]

    assert len(boxes) == 0
