import subprocess
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from animate.animated_drawings_runner import (
    AnimationResult,
    JointSpreadError,
    composite_on_white_bg,
    joint_spread_ratio,
    run_animated_drawings,
)


def test_composite_places_opaque_pixels_over_white() -> None:
    tex = np.zeros((10, 10, 4), dtype=np.uint8)
    tex[2:5, 2:5, :3] = (0, 0, 255)  # red BGR
    tex[2:5, 2:5, 3] = 255            # opaque

    out = composite_on_white_bg(tex)

    assert out.shape == (10, 10, 3)
    assert out.dtype == np.uint8
    # transparent area is white
    assert tuple(out[0, 0]) == (255, 255, 255)
    # opaque area keeps red
    assert tuple(out[3, 3]) == (0, 0, 255)


def test_composite_blends_semitransparent_pixels_with_white() -> None:
    tex = np.zeros((4, 4, 4), dtype=np.uint8)
    tex[1, 1, :3] = (0, 0, 255)
    tex[1, 1, 3] = 128  # ~50%

    out = composite_on_white_bg(tex)

    # Semi-transparent red over white: B/G blend from 0 toward 255; R stays 255.
    px = out[1, 1]
    # Pure red foreground: R channel stays 255 (both fg and bg are 255)
    assert int(px[2]) == 255
    # B and G blend from 0 → ~127 against white
    assert 120 <= int(px[1]) <= 135
    assert 120 <= int(px[0]) <= 135


def test_composite_accepts_bgra_float_and_returns_uint8() -> None:
    tex = np.zeros((4, 4, 4), dtype=np.float32)
    tex[..., 3] = 255.0
    out = composite_on_white_bg(tex)
    assert out.dtype == np.uint8


def test_run_returns_success_when_video_and_cfg_produced(tmp_path: Path) -> None:
    tex = np.zeros((32, 32, 4), dtype=np.uint8)
    tex[..., :3] = 200
    tex[..., 3] = 255
    ad_repo = tmp_path / "ad"
    ad_repo.mkdir()
    (ad_repo / "examples").mkdir()
    (ad_repo / "examples" / "image_to_animation.py").write_text("# placeholder")
    work_dir = tmp_path / "work"

    def fake_run(cmd, *args, **kwargs):
        # Simulate AD producing artifacts inside out dir
        out_dir = Path(cmd[cmd.index("--output") + 1]) if "--output" in cmd else Path(cmd[-1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "video.gif").write_bytes(b"fake-gif")
        (out_dir / "char_cfg.yaml").write_text(
            "skeleton:\n"
            "  - {name: root, loc: [10, 10]}\n"
            "  - {name: end, loc: [90, 90]}\n"
        )
        result = subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")
        return result

    with patch("animate.animated_drawings_runner.subprocess.run", side_effect=fake_run):
        result = run_animated_drawings(
            texture_bgra=tex,
            motion="dab",
            ad_repo_path=ad_repo,
            work_dir=work_dir,
            ad_python=Path("/fake/python"),
            timeout_sec=5.0,
        )

    assert isinstance(result, AnimationResult)
    assert result.success is True
    assert result.video_path is not None
    assert result.video_path.exists()
    assert result.char_cfg_path is not None
    assert result.error is None
    assert result.duration_sec > 0


def test_run_returns_failure_when_subprocess_nonzero_exit(tmp_path: Path) -> None:
    tex = np.zeros((32, 32, 4), dtype=np.uint8)
    tex[..., 3] = 255
    ad_repo = tmp_path / "ad"
    (ad_repo / "examples").mkdir(parents=True)
    (ad_repo / "examples" / "image_to_animation.py").write_text("#")
    work_dir = tmp_path / "work"

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=2, stdout="", stderr="boom")

    with patch("animate.animated_drawings_runner.subprocess.run", side_effect=fake_run):
        result = run_animated_drawings(
            texture_bgra=tex, motion="dab", ad_repo_path=ad_repo,
            work_dir=work_dir, ad_python=Path("/fake/python"), timeout_sec=5.0,
        )

    assert result.success is False
    assert result.video_path is None
    assert "boom" in (result.error or "") or "exit" in (result.error or "").lower()


def test_run_returns_failure_on_timeout(tmp_path: Path) -> None:
    tex = np.zeros((32, 32, 4), dtype=np.uint8)
    tex[..., 3] = 255
    ad_repo = tmp_path / "ad"
    (ad_repo / "examples").mkdir(parents=True)
    (ad_repo / "examples" / "image_to_animation.py").write_text("#")
    work_dir = tmp_path / "work"

    def raise_timeout(cmd, *args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 1))

    with patch("animate.animated_drawings_runner.subprocess.run", side_effect=raise_timeout):
        result = run_animated_drawings(
            texture_bgra=tex, motion="dab", ad_repo_path=ad_repo,
            work_dir=work_dir, ad_python=Path("/fake/python"), timeout_sec=0.1,
        )

    assert result.success is False
    assert "timeout" in (result.error or "").lower()


def test_run_writes_input_png_and_passes_path_to_subprocess(tmp_path: Path) -> None:
    tex = np.zeros((32, 32, 4), dtype=np.uint8)
    tex[..., :3] = 100
    tex[..., 3] = 255
    ad_repo = tmp_path / "ad"
    (ad_repo / "examples").mkdir(parents=True)
    (ad_repo / "examples" / "image_to_animation.py").write_text("#")
    work_dir = tmp_path / "work"

    captured_cmd: list = []

    def fake_run(cmd, *args, **kwargs):
        captured_cmd.extend(cmd)
        out_dir = Path(cmd[-1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "video.gif").write_bytes(b"g")
        (out_dir / "char_cfg.yaml").write_text(
            "skeleton:\n"
            "  - {name: root, loc: [10, 10]}\n"
            "  - {name: end, loc: [90, 90]}\n"
        )
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    with patch("animate.animated_drawings_runner.subprocess.run", side_effect=fake_run):
        run_animated_drawings(
            texture_bgra=tex, motion="dab", ad_repo_path=ad_repo,
            work_dir=work_dir, ad_python=Path("/fake/python"), timeout_sec=5.0,
        )

    input_png_path = work_dir / "input.png"
    assert input_png_path.exists()
    assert str(input_png_path) in captured_cmd
    loaded = cv2.imread(str(input_png_path))
    assert loaded.shape == (32, 32, 3)


def _write_char_cfg(path: Path, joints: list) -> None:
    path.write_text(yaml.safe_dump({"skeleton": joints}))


def test_joint_spread_ratio_large_for_spread_skeleton(tmp_path: Path) -> None:
    cfg = tmp_path / "char_cfg.yaml"
    _write_char_cfg(cfg, [
        {"name": "root", "loc": [50, 50]},
        {"name": "left_hand", "loc": [10, 10]},
        {"name": "right_hand", "loc": [90, 90]},
    ])
    assert joint_spread_ratio(cfg, image_size=(100, 100)) > 0.5


def test_joint_spread_ratio_small_for_bunched_skeleton(tmp_path: Path) -> None:
    cfg = tmp_path / "char_cfg.yaml"
    _write_char_cfg(cfg, [
        {"name": "root", "loc": [50, 50]},
        {"name": "left_hand", "loc": [51, 50]},
        {"name": "right_hand", "loc": [50, 51]},
    ])
    assert joint_spread_ratio(cfg, image_size=(100, 100)) < 0.1


def test_run_passes_motion_cfg_when_present(tmp_path: Path) -> None:
    tex = np.zeros((32, 32, 4), dtype=np.uint8)
    tex[..., 3] = 255
    ad_repo = tmp_path / "ad"
    (ad_repo / "examples").mkdir(parents=True)
    (ad_repo / "examples" / "image_to_animation.py").write_text("#")
    motion_dir = ad_repo / "animated_drawings" / "config" / "motion"
    motion_dir.mkdir(parents=True)
    motion_cfg = motion_dir / "zombie.yaml"
    motion_cfg.write_text("# fake zombie motion\n")

    work_dir = tmp_path / "work"
    captured_cmd: list = []

    def fake_run(cmd, *args, **kwargs):
        captured_cmd.extend(cmd)
        out_dir = Path(cmd[-2])  # motion cfg is now the last arg, out_dir is second-to-last
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "video.gif").write_bytes(b"g")
        (out_dir / "char_cfg.yaml").write_text(
            "skeleton:\n  - {name: root, loc: [10, 10]}\n  - {name: end, loc: [90, 90]}\n"
        )
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    with patch("animate.animated_drawings_runner.subprocess.run", side_effect=fake_run):
        result = run_animated_drawings(
            texture_bgra=tex, motion="zombie", ad_repo_path=ad_repo,
            work_dir=work_dir, ad_python=Path("/fake/python"), timeout_sec=5.0,
        )

    assert result.success is True
    assert str(motion_cfg) in captured_cmd
    # Order: [python, script, input.png, out_dir, motion_cfg]
    assert captured_cmd[-1] == str(motion_cfg)


def test_run_falls_back_when_motion_cfg_missing(tmp_path: Path, capsys) -> None:
    tex = np.zeros((32, 32, 4), dtype=np.uint8)
    tex[..., 3] = 255
    ad_repo = tmp_path / "ad"
    (ad_repo / "examples").mkdir(parents=True)
    (ad_repo / "examples" / "image_to_animation.py").write_text("#")
    # Deliberately do NOT create the motion dir/file
    work_dir = tmp_path / "work"
    captured_cmd: list = []

    def fake_run(cmd, *args, **kwargs):
        captured_cmd.extend(cmd)
        out_dir = Path(cmd[-1])  # motion missing → out_dir is last
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "video.gif").write_bytes(b"g")
        (out_dir / "char_cfg.yaml").write_text(
            "skeleton:\n  - {name: root, loc: [10, 10]}\n  - {name: end, loc: [90, 90]}\n"
        )
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    with patch("animate.animated_drawings_runner.subprocess.run", side_effect=fake_run):
        result = run_animated_drawings(
            texture_bgra=tex, motion="nonexistent_motion", ad_repo_path=ad_repo,
            work_dir=work_dir, ad_python=Path("/fake/python"), timeout_sec=5.0,
        )

    assert result.success is True
    # argv should be exactly 4 items: python, script, input_png, out_dir
    assert len(captured_cmd) == 4
    # Warning was printed to stdout
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_run_downgrades_success_to_failure_when_joints_bunched(tmp_path: Path) -> None:
    tex = np.zeros((32, 32, 4), dtype=np.uint8)
    tex[..., 3] = 255
    ad_repo = tmp_path / "ad"
    (ad_repo / "examples").mkdir(parents=True)
    (ad_repo / "examples" / "image_to_animation.py").write_text("#")
    work_dir = tmp_path / "work"

    def fake_run(cmd, *args, **kwargs):
        out_dir = Path(cmd[-1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "video.gif").write_bytes(b"g")
        # bunched joints
        _write_char_cfg(out_dir / "char_cfg.yaml", [
            {"name": "root", "loc": [50, 50]},
            {"name": "left_hand", "loc": [50, 50]},
            {"name": "right_hand", "loc": [50, 50]},
        ])
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    with patch("animate.animated_drawings_runner.subprocess.run", side_effect=fake_run):
        result = run_animated_drawings(
            texture_bgra=tex, motion="dab", ad_repo_path=ad_repo,
            work_dir=work_dir, ad_python=Path("/fake/python"), timeout_sec=5.0,
        )

    assert result.success is False
    assert "bunched" in (result.error or "").lower() or "spread" in (result.error or "").lower()
