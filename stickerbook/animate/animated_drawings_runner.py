"""Runs AnimatedDrawings image_to_animation.py and parses the result."""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml


class JointSpreadError(RuntimeError):
    pass


MIN_JOINT_SPREAD_RATIO = 0.15  # 이미지 대각선 대비 joint 범위


def joint_spread_ratio(char_cfg_path: Path, image_size: Tuple[int, int]) -> float:
    data = yaml.safe_load(Path(char_cfg_path).read_text())
    joints = data.get("skeleton", []) if isinstance(data, dict) else []
    locs = [j.get("loc") for j in joints if isinstance(j, dict) and "loc" in j]
    if not locs:
        return 0.0
    xs = [p[0] for p in locs]
    ys = [p[1] for p in locs]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    w, h = image_size
    diag = (w * w + h * h) ** 0.5
    return ((dx * dx + dy * dy) ** 0.5) / max(diag, 1.0)


def composite_on_white_bg(texture_bgra: np.ndarray) -> np.ndarray:
    """Alpha-composite an RGBA/BGRA sticker over solid white, returning BGR uint8.

    AnimatedDrawings expects a full-opaque image. Transparent pixels become white.
    """
    tex = np.asarray(texture_bgra, dtype=np.float32)
    if tex.shape[-1] != 4:
        raise ValueError("texture_bgra must have 4 channels (BGRA)")
    rgb = tex[..., :3]
    alpha = tex[..., 3:4] / 255.0
    white = np.full_like(rgb, 255.0)
    out = rgb * alpha + white * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


@dataclass
class AnimationResult:
    success: bool
    video_path: Optional[Path]
    char_cfg_path: Optional[Path]
    duration_sec: float
    error: Optional[str]
    work_dir: Optional[Path] = None  # job's isolated work directory (for cleanup)


def run_animated_drawings(
    texture_bgra: np.ndarray,
    motion: str,
    ad_repo_path: Path,
    work_dir: Path,
    ad_python: Path,
    timeout_sec: float = 30.0,
) -> AnimationResult:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    input_png = work_dir / "input.png"
    rgb = composite_on_white_bg(texture_bgra)
    cv2.imwrite(str(input_png), rgb)

    script = Path(ad_repo_path) / "examples" / "image_to_animation.py"
    out_dir = work_dir / "out"

    cmd = [str(ad_python), str(script), str(input_png), str(out_dir)]
    motion_cfg = Path(ad_repo_path) / "examples" / "config" / "motion" / f"{motion}.yaml"
    if motion_cfg.is_file():
        cmd.append(str(motion_cfg))
        # auto-pair retarget config sharing the same name; falls back to AD
        # default (fair1_ppf) when no custom retarget yaml is supplied.
        retarget_cfg = Path(ad_repo_path) / "examples" / "config" / "retarget" / f"{motion}.yaml"
        if retarget_cfg.is_file():
            cmd.append(str(retarget_cfg))
    else:
        print(f"[ad] motion cfg not found: {motion_cfg}; using AD default")
    start = time.monotonic()
    # PyOpenGL otherwise probes EGL first under WSLg, breaking AD's GLX context.
    ad_env = {**os.environ, "PYOPENGL_PLATFORM": "glx"}
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=ad_env,
        )
    except subprocess.TimeoutExpired:
        return AnimationResult(
            success=False, video_path=None, char_cfg_path=None,
            duration_sec=time.monotonic() - start,
            error=f"timeout after {timeout_sec}s",
            work_dir=work_dir,
        )
    duration = time.monotonic() - start

    if result.returncode != 0:
        return AnimationResult(
            success=False, video_path=None, char_cfg_path=None,
            duration_sec=duration,
            error=f"exit {result.returncode}: {result.stderr[-500:]}",
            work_dir=work_dir,
        )

    video_gif = out_dir / "video.gif"
    video_mp4 = out_dir / "video.mp4"
    video = video_gif if video_gif.exists() else (video_mp4 if video_mp4.exists() else None)
    cfg = out_dir / "char_cfg.yaml"

    if video is None:
        return AnimationResult(
            success=False, video_path=None, char_cfg_path=None,
            duration_sec=duration,
            error="no video artifact produced",
            work_dir=work_dir,
        )

    # Joint spread sanity check (R13)
    h, w = texture_bgra.shape[:2]
    spread = joint_spread_ratio(cfg, image_size=(w, h)) if cfg.exists() else 0.0
    if spread < MIN_JOINT_SPREAD_RATIO:
        return AnimationResult(
            success=False, video_path=None, char_cfg_path=cfg if cfg.exists() else None,
            duration_sec=duration,
            error=f"joint spread {spread:.3f} below threshold {MIN_JOINT_SPREAD_RATIO} (bunched skeleton)",
            work_dir=work_dir,
        )

    return AnimationResult(
        success=True,
        video_path=video,
        char_cfg_path=cfg if cfg.exists() else None,
        duration_sec=duration,
        error=None,
        work_dir=work_dir,
    )
