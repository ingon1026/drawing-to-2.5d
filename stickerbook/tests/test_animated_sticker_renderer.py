from pathlib import Path

import cv2
import numpy as np
import pytest

from render.animated_sticker_renderer import AnimatedStickerRenderer


@pytest.fixture()
def sample_video(tmp_path: Path) -> Path:
    """Write a small 3-frame BGR video for testing."""
    path = tmp_path / "video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (20, 20))
    frames = []
    for i in range(3):
        f = np.zeros((20, 20, 3), dtype=np.uint8)
        f[..., 0] = (i + 1) * 50
        writer.write(f)
        frames.append(f)
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return path


def test_renderer_reads_first_frame_as_bgra(sample_video: Path) -> None:
    r = AnimatedStickerRenderer(video_path=sample_video)
    try:
        bgra = r.next_frame_bgra()
    finally:
        r.release()
    assert bgra.shape == (20, 20, 4)
    assert bgra.dtype == np.uint8


def test_renderer_loops_back_to_zero_after_last_frame(sample_video: Path) -> None:
    r = AnimatedStickerRenderer(video_path=sample_video)
    try:
        seen = [tuple(r.next_frame_bgra()[0, 0, :3]) for _ in range(6)]
    finally:
        r.release()
    # First 3 unique, then wraps: frame 3 should equal frame 0
    assert seen[0] == seen[3]
    assert seen[1] == seen[4]


def test_renderer_applies_chroma_key_to_white_background(tmp_path: Path) -> None:
    """AD outputs have white BG per M9.1 findings — alpha=0 for near-white pixels."""
    path = tmp_path / "ck.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (10, 10))
    f = np.full((10, 10, 3), 255, dtype=np.uint8)  # white frame
    f[3:7, 3:7] = (0, 255, 0)  # green square on white bg
    for _ in range(2):
        writer.write(f)
    writer.release()

    r = AnimatedStickerRenderer(video_path=path, chroma_key_threshold=240)
    try:
        bgra = r.next_frame_bgra()
    finally:
        r.release()

    # Background (white) -> alpha 0
    assert bgra[0, 0, 3] == 0
    # Foreground (green) -> alpha 255
    assert bgra[5, 5, 3] == 255
