import numpy as np

from detect.candidate_detector import CandidateBox
from render.overlay import draw_candidate_boxes


def test_yolo_box_drawn_in_green() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box = CandidateBox(x=10, y=10, w=20, h=20, confidence=0.9, source="yolo")

    draw_candidate_boxes(frame, [box])

    has_green = np.any(np.all(frame == [0, 255, 0], axis=-1))
    has_blue = np.any(np.all(frame == [255, 0, 0], axis=-1))
    assert has_green
    assert not has_blue


def test_contour_box_drawn_in_blue() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box = CandidateBox(x=10, y=10, w=20, h=20, confidence=1.0, source="contour")

    draw_candidate_boxes(frame, [box])

    has_green = np.any(np.all(frame == [0, 255, 0], axis=-1))
    has_blue = np.any(np.all(frame == [255, 0, 0], axis=-1))
    assert has_blue
    assert not has_green


def test_no_boxes_leaves_frame_unchanged() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    original = frame.copy()

    draw_candidate_boxes(frame, [])

    assert np.array_equal(frame, original)
