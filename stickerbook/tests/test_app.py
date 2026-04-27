from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from animate.animated_drawings_runner import AnimationResult
from app import AnchoredSticker, App, AppAction, AnimationState, _PerfTracker
from track.homography_anchor import HomographyAnchor


def test_handle_key_quit_on_lowercase_q() -> None:
    app = App(camera_source=0)
    assert app._handle_key(ord("q")) == AppAction.QUIT


def test_handle_key_quit_on_uppercase_q() -> None:
    app = App(camera_source=0)
    assert app._handle_key(ord("Q")) == AppAction.QUIT


def test_handle_key_quit_on_escape() -> None:
    app = App(camera_source=0)
    assert app._handle_key(27) == AppAction.QUIT


def test_handle_key_returns_none_when_no_key_pressed() -> None:
    app = App(camera_source=0)
    assert app._handle_key(-1) is None


def test_handle_key_reset_on_lowercase_r() -> None:
    app = App(camera_source=0)
    assert app._handle_key(ord("r")) == AppAction.RESET


def test_handle_key_reset_on_uppercase_r() -> None:
    app = App(camera_source=0)
    assert app._handle_key(ord("R")) == AppAction.RESET


def test_handle_key_save_on_lowercase_s() -> None:
    app = App(camera_source=0)
    assert app._handle_key(ord("s")) == AppAction.SAVE


def test_handle_key_save_on_uppercase_s() -> None:
    app = App(camera_source=0)
    assert app._handle_key(ord("S")) == AppAction.SAVE


def test_handle_key_returns_none_for_other_keys() -> None:
    app = App(camera_source=0)
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


def test_poll_animation_cleans_work_dir_on_failure(tmp_path: Path) -> None:
    app = App()
    work_dir = tmp_path / "dead-job"
    work_dir.mkdir()
    (work_dir / "input.png").write_bytes(b"png")

    mock_future = MagicMock()
    mock_future.done.return_value = True
    mock_future.result.return_value = AnimationResult(
        success=False, video_path=None, char_cfg_path=None,
        duration_sec=0.1, error="boom", work_dir=work_dir,
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
    assert not work_dir.exists()  # cleaned up


def test_reset_stickers_cleans_each_work_dir(tmp_path: Path) -> None:
    app = App()
    work_a = tmp_path / "a"; work_a.mkdir()
    work_b = tmp_path / "b"; work_b.mkdir()

    def _stub_sticker_with_workdir(wd: Path) -> AnchoredSticker:
        asset = MagicMock()
        asset.source_region = (0, 0, 10, 10)
        asset.texture_bgra = np.zeros((10, 10, 4), dtype=np.uint8)
        return AnchoredSticker(
            sticker=asset, anchor=HomographyAnchor(),
            animation_state=AnimationState.ANIMATED,
            animation_work_dir=wd,
        )

    app._anchored = [_stub_sticker_with_workdir(work_a), _stub_sticker_with_workdir(work_b)]
    app._reset_stickers()

    assert not work_a.exists()
    assert not work_b.exists()
    assert app._anchored == []


def test_perf_report_includes_animation_metrics_when_present() -> None:
    from app import _PerfTracker
    pt = _PerfTracker()
    pt.record("animation_success_sec", 8.5)
    pt.record("animation_failure_sec", 3.2)
    report = pt.report()
    assert "animation_success_sec" in report
    assert "animation_failure_sec" in report


def test_choose_popup_lift_ratio_full_when_sy_at_least_sh() -> None:
    from app import _choose_popup_lift_ratio
    # sy=300, sh=200 -> max_ratio=1.5, capped to 1.0
    assert _choose_popup_lift_ratio((100, 300, 200, 200)) == 1.0


def test_choose_popup_lift_ratio_proportional_when_sy_less_than_sh() -> None:
    from app import _choose_popup_lift_ratio
    # sy=50, sh=200 -> ratio=0.25
    assert _choose_popup_lift_ratio((100, 50, 200, 200)) == pytest.approx(0.25)


def test_choose_popup_lift_ratio_zero_when_at_top_edge() -> None:
    from app import _choose_popup_lift_ratio
    # sy=0 -> ratio=0 (billboard sits flat at source)
    assert _choose_popup_lift_ratio((100, 0, 200, 200)) == 0.0


def test_promote_to_live_caps_popup_for_high_source_region() -> None:
    app = App()
    asset = MagicMock()
    asset.source_region = (100, 30, 200, 250)  # sy=30, sh=250 -> ratio=0.12
    asset.texture_bgra = np.zeros((250, 200, 4), dtype=np.uint8)
    anchor = HomographyAnchor()

    item = app._promote_to_live(asset, anchor)

    assert item.popup_lift_ratio == pytest.approx(30.0 / 250.0)


def test_promote_to_live_keeps_full_popup_for_centered_source_region() -> None:
    app = App()
    asset = MagicMock()
    asset.source_region = (100, 400, 200, 200)  # sy=400, sh=200 -> ratio=1.0
    asset.texture_bgra = np.zeros((200, 200, 4), dtype=np.uint8)
    anchor = HomographyAnchor()

    item = app._promote_to_live(asset, anchor)

    assert item.popup_lift_ratio == 1.0
