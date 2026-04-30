"""Unit tests for FrameRecorder."""
from __future__ import annotations

import numpy as np
import pytest

from motion.recorder import FrameRecorder


def make_dummy_frame() -> np.ndarray:
    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_recorder_starts_idle():
    rec = FrameRecorder()
    assert rec.is_recording() is False


def test_recorder_start_then_stop_returns_buffered_frames():
    rec = FrameRecorder()
    rec.start()
    assert rec.is_recording() is True

    rec.add_frame(make_dummy_frame())
    rec.add_frame(make_dummy_frame())
    rec.add_frame(make_dummy_frame())

    frames = rec.stop()
    assert rec.is_recording() is False
    assert len(frames) == 3
    assert frames[0].shape == (10, 10, 3)


def test_add_frame_ignored_when_not_recording():
    rec = FrameRecorder()
    rec.add_frame(make_dummy_frame())
    rec.start()
    rec.stop()
    rec.add_frame(make_dummy_frame())  # idle 상태
    assert rec.is_recording() is False


def test_start_clears_previous_buffer():
    rec = FrameRecorder()
    rec.start()
    rec.add_frame(make_dummy_frame())
    rec.stop()

    rec.start()
    rec.add_frame(make_dummy_frame())
    frames = rec.stop()
    assert len(frames) == 1
