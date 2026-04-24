import numpy as np

from render.spinner_overlay import draw_spinner


def test_draw_spinner_modifies_frame_in_region() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    before = frame.copy()

    draw_spinner(frame, center=(100, 100), radius=20, phase=0.0)

    assert not np.array_equal(frame, before)
    # Far-away pixels untouched
    assert tuple(frame[0, 0]) == (0, 0, 0)
    assert tuple(frame[199, 199]) == (0, 0, 0)


def test_draw_spinner_different_phase_produces_different_pixels() -> None:
    f1 = np.zeros((200, 200, 3), dtype=np.uint8)
    f2 = np.zeros((200, 200, 3), dtype=np.uint8)
    draw_spinner(f1, center=(100, 100), radius=20, phase=0.0)
    draw_spinner(f2, center=(100, 100), radius=20, phase=1.5)
    assert not np.array_equal(f1, f2)


def test_draw_spinner_does_not_raise_when_center_near_frame_edge() -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    # Should clip instead of raising
    draw_spinner(frame, center=(5, 5), radius=15, phase=0.0)
    draw_spinner(frame, center=(48, 48), radius=15, phase=0.0)
