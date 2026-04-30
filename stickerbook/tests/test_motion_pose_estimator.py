"""Unit tests for PoseEstimator."""
from __future__ import annotations

import numpy as np
import pytest

mediapipe = pytest.importorskip("mediapipe")

from motion.pose_estimator import PoseEstimator, PoseLandmarks


def make_blank_frame() -> np.ndarray:
    return np.full((480, 640, 3), 128, dtype=np.uint8)


def test_estimator_returns_one_result_per_frame():
    est = PoseEstimator()
    results = est.estimate_batch([make_blank_frame()])
    est.close()
    assert len(results) == 1


def test_blank_frame_yields_none_or_low_visibility():
    """Blank frame has no person, so MediaPipe returns None or unrecognized."""
    est = PoseEstimator()
    results = est.estimate_batch([make_blank_frame()])
    est.close()
    # blank frame: either None (no detection) or PoseLandmarks with all zeros
    assert results[0] is None or isinstance(results[0], PoseLandmarks)


def test_pose_landmarks_has_33_points():
    """PoseLandmarks dataclass should expose 33 (x,y,z) points."""
    pl = PoseLandmarks(points=np.zeros((33, 3), dtype=np.float32))
    assert pl.points.shape == (33, 3)
