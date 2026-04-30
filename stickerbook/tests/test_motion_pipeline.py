"""Integration test for MotionPipeline.toggle()."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from motion.pipeline import MotionPipeline
from motion.pose_estimator import PoseLandmarks


def _t_pose_landmarks() -> PoseLandmarks:
    pts = np.zeros((33, 3), dtype=np.float32)
    pts[11] = [-0.2, 1.5, 0]; pts[12] = [0.2, 1.5, 0]
    pts[13] = [-0.5, 1.5, 0]; pts[14] = [0.5, 1.5, 0]
    pts[15] = [-0.7, 1.5, 0]; pts[16] = [0.7, 1.5, 0]
    pts[23] = [-0.15, 1.0, 0]; pts[24] = [0.15, 1.0, 0]
    pts[25] = [-0.15, 0.5, 0]; pts[26] = [0.15, 0.5, 0]
    pts[27] = [-0.15, 0.0, 0]; pts[28] = [0.15, 0.0, 0]
    pts[31] = [-0.15, 0.0, 0.1]; pts[32] = [0.15, 0.0, 0.1]
    pts[0] = [0, 1.7, 0]
    return PoseLandmarks(points=pts)


def test_toggle_starts_recording_first_call(tmp_path: Path):
    rec = MagicMock()
    rec.is_recording.return_value = False
    est = MagicMock()
    lib = MagicMock()

    pipeline = MotionPipeline(
        recorder=rec, estimator=est, library=lib,
        tmp_dir=tmp_path, fps=30.0,
    )
    name = pipeline.toggle()
    rec.start.assert_called_once()
    assert name is None  # 녹화만 시작


def test_toggle_stops_and_processes_second_call(tmp_path: Path):
    """Stop, run pose estimator, write BVH, add to library, set active."""
    rec = MagicMock()
    rec.is_recording.return_value = True
    rec.stop.return_value = [
        np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(60)
    ]
    est = MagicMock()
    est.estimate_batch.return_value = [_t_pose_landmarks()] * 60
    lib = MagicMock()
    lib.add.return_value = "motion_001"

    pipeline = MotionPipeline(
        recorder=rec, estimator=est, library=lib,
        tmp_dir=tmp_path, fps=30.0,
    )
    name = pipeline.toggle()

    assert name == "motion_001"
    rec.stop.assert_called_once()
    est.estimate_batch.assert_called_once()
    lib.add.assert_called_once()
    lib.set_active.assert_called_once_with("motion_001")


def test_toggle_aborts_on_too_few_frames(tmp_path: Path):
    rec = MagicMock()
    rec.is_recording.return_value = True
    rec.stop.return_value = [
        np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)
    ]  # < 30
    est = MagicMock()
    lib = MagicMock()

    pipeline = MotionPipeline(
        recorder=rec, estimator=est, library=lib,
        tmp_dir=tmp_path, fps=30.0,
    )
    name = pipeline.toggle()
    assert name is None
    est.estimate_batch.assert_not_called()
    lib.add.assert_not_called()


def test_toggle_aborts_on_high_recognition_failure(tmp_path: Path):
    rec = MagicMock()
    rec.is_recording.return_value = True
    rec.stop.return_value = [
        np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(60)
    ]
    est = MagicMock()
    # 80% 실패율 (None 48 / 60)
    est.estimate_batch.return_value = [None] * 48 + [_t_pose_landmarks()] * 12
    lib = MagicMock()

    pipeline = MotionPipeline(
        recorder=rec, estimator=est, library=lib,
        tmp_dir=tmp_path, fps=30.0,
    )
    name = pipeline.toggle()
    assert name is None
    lib.add.assert_not_called()
