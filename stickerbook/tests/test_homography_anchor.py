import cv2
import numpy as np

from track.homography_anchor import HomographyAnchor


def _textured_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Synthetic frame with enough corners/text for ORB feature extraction."""
    frame = np.full((h, w, 3), 200, dtype=np.uint8)
    cv2.rectangle(frame, (50, 50), (100, 100), (255, 0, 0), -1)
    cv2.rectangle(frame, (150, 80), (220, 150), (0, 255, 0), -1)
    cv2.circle(frame, (80, 180), 30, (0, 0, 255), -1)
    cv2.putText(frame, "stickerbook", (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.line(frame, (10, 10), (w - 10, h - 10), (100, 50, 150), 2)
    return frame


def test_anchor_returns_near_identity_homography_when_frame_unchanged() -> None:
    frame = _textured_frame()
    anchor = HomographyAnchor()
    anchor.initialize(frame, region=(40, 40, 200, 170))

    state = anchor.update(frame)

    assert not state.lost
    assert state.homography is not None
    H = state.homography / state.homography[2, 2]
    assert np.allclose(H, np.eye(3), atol=0.5)


def test_anchor_tracks_translation_in_warped_frame() -> None:
    ref = _textured_frame()
    anchor = HomographyAnchor()
    anchor.initialize(ref, region=(40, 40, 200, 170))

    # shift the whole frame by (+20 x, +10 y)
    M = np.float32([[1, 0, 20], [0, 1, 10]])
    shifted = cv2.warpAffine(ref, M, (ref.shape[1], ref.shape[0]))

    state = anchor.update(shifted)

    assert not state.lost
    H = state.homography / state.homography[2, 2]
    assert abs(H[0, 2] - 20) < 4
    assert abs(H[1, 2] - 10) < 4
    assert abs(H[0, 0] - 1) < 0.05
    assert abs(H[1, 1] - 1) < 0.05


def test_anchor_reports_lost_after_threshold_on_blank_frames() -> None:
    ref = _textured_frame()
    anchor = HomographyAnchor(lost_frames_threshold=3)
    anchor.initialize(ref, region=(40, 40, 200, 170))

    blank = np.zeros_like(ref)
    state = None
    for _ in range(5):
        state = anchor.update(blank)

    assert state is not None
    assert state.lost
    assert state.homography is None
    assert anchor.is_lost()


def test_anchor_reacquires_after_lost_when_content_returns() -> None:
    ref = _textured_frame()
    anchor = HomographyAnchor(lost_frames_threshold=3, retry_interval=5)
    anchor.initialize(ref, region=(40, 40, 200, 170))

    blank = np.zeros_like(ref)
    for _ in range(5):
        state = anchor.update(blank)
    assert state.lost

    # Original content returns — anchor should re-acquire within retry_interval frames
    state = None
    for _ in range(10):
        state = anchor.update(ref)

    assert state is not None
    assert not state.lost
    assert state.homography is not None
    assert not anchor.is_lost()


def test_anchor_initialize_on_featureless_frame_marks_lost() -> None:
    anchor = HomographyAnchor()
    blank = np.full((240, 320, 3), 128, dtype=np.uint8)

    anchor.initialize(blank, region=(100, 100, 50, 50))

    assert anchor.is_lost()
    state = anchor.update(blank)
    assert state.lost
