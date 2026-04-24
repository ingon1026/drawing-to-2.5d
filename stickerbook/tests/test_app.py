from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from animate.animated_drawings_runner import AnimationResult
from app import AnchoredSticker, App, AppAction, AnimationState, _PerfTracker
from track.homography_anchor import HomographyAnchor


def test_handle_key_quit_on_lowercase_q() -> None:
    app = App(camera_index=0)
    assert app._handle_key(ord("q")) == AppAction.QUIT


def test_handle_key_quit_on_uppercase_q() -> None:
    app = App(camera_index=0)
    assert app._handle_key(ord("Q")) == AppAction.QUIT


def test_handle_key_quit_on_escape() -> None:
    app = App(camera_index=0)
    assert app._handle_key(27) == AppAction.QUIT


def test_handle_key_returns_none_when_no_key_pressed() -> None:
    app = App(camera_index=0)
    assert app._handle_key(-1) is None


def test_handle_key_reset_on_lowercase_r() -> None:
    app = App(camera_index=0)
    assert app._handle_key(ord("r")) == AppAction.RESET


def test_handle_key_reset_on_uppercase_r() -> None:
    app = App(camera_index=0)
    assert app._handle_key(ord("R")) == AppAction.RESET


def test_handle_key_save_on_lowercase_s() -> None:
    app = App(camera_index=0)
    assert app._handle_key(ord("s")) == AppAction.SAVE


def test_handle_key_save_on_uppercase_s() -> None:
    app = App(camera_index=0)
    assert app._handle_key(ord("S")) == AppAction.SAVE


def test_handle_key_returns_none_for_other_keys() -> None:
    app = App(camera_index=0)
    assert app._handle_key(ord("a")) is None
    assert app._handle_key(ord("x")) is None


def test_on_click_promotes_to_live_and_submits_to_animation_worker() -> None:
    app = App()
    app._animation_worker = MagicMock()
    fut = MagicMock()
    app._animation_worker.submit.return_value = fut

    sticker_asset = MagicMock()
    sticker_asset.source_region = (0, 0, 10, 10)
    sticker_asset.texture_bgra = np.zeros((10, 10, 4), dtype=np.uint8)
    anchor = HomographyAnchor()

    item = app._promote_to_live(sticker_asset, anchor)

    assert item.animation_state is AnimationState.PREPARING
    app._animation_worker.submit.assert_called_once()
    assert item.animation_future is fut


def test_promote_to_live_without_worker_stays_static() -> None:
    app = App()
    app._animation_worker = None  # animation unavailable

    sticker_asset = MagicMock()
    sticker_asset.source_region = (0, 0, 10, 10)
    sticker_asset.texture_bgra = np.zeros((10, 10, 4), dtype=np.uint8)
    anchor = HomographyAnchor()

    item = app._promote_to_live(sticker_asset, anchor)

    assert item.animation_state is AnimationState.STATIC
    assert item.animation_future is None


def test_poll_animation_transitions_to_animated_on_success(tmp_path: Path) -> None:
    app = App()
    app._anchored = []
    mock_future = MagicMock()
    mock_future.done.return_value = True
    vid = tmp_path / "video.mp4"
    vid.write_bytes(b"x")
    mock_future.result.return_value = AnimationResult(
        success=True, video_path=vid, char_cfg_path=None,
        duration_sec=0.1, error=None,
    )
    asset = MagicMock()
    asset.source_region = (0, 0, 10, 10)
    asset.texture_bgra = np.zeros((10, 10, 4), dtype=np.uint8)
    item = AnchoredSticker(
        sticker=asset, anchor=HomographyAnchor(),
        animation_state=AnimationState.PREPARING,
        animation_future=mock_future,
    )
    app._anchored.append(item)

    app._poll_animations(_PerfTracker())

    assert item.animation_state is AnimationState.ANIMATED
    assert item.animation_video_path == vid


def test_poll_animation_transitions_to_failed_on_error() -> None:
    app = App()
    mock_future = MagicMock()
    mock_future.done.return_value = True
    mock_future.result.return_value = AnimationResult(
        success=False, video_path=None, char_cfg_path=None,
        duration_sec=0.1, error="joint spread low",
    )
    asset = MagicMock()
    asset.source_region = (0, 0, 10, 10)
    asset.texture_bgra = np.zeros((10, 10, 4), dtype=np.uint8)
    item = AnchoredSticker(
        sticker=asset, anchor=HomographyAnchor(),
        animation_state=AnimationState.PREPARING,
        animation_future=mock_future,
    )
    app._anchored = [item]

    app._poll_animations(_PerfTracker())

    assert item.animation_state is AnimationState.FAILED
    assert item.animation_video_path is None


def test_perf_report_includes_animation_metrics_when_present() -> None:
    from app import _PerfTracker
    pt = _PerfTracker()
    pt.record("animation_success_sec", 8.5)
    pt.record("animation_failure_sec", 3.2)
    report = pt.report()
    assert "animation_success_sec" in report
    assert "animation_failure_sec" in report
