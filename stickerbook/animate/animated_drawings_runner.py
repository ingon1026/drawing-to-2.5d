"""Runs AnimatedDrawings image_to_animation.py and parses the result."""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


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
    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return AnimationResult(
            success=False, video_path=None, char_cfg_path=None,
            duration_sec=time.monotonic() - start,
            error=f"timeout after {timeout_sec}s",
        )
    duration = time.monotonic() - start

    if result.returncode != 0:
        return AnimationResult(
            success=False, video_path=None, char_cfg_path=None,
            duration_sec=duration,
            error=f"exit {result.returncode}: {result.stderr[-500:]}",
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
        )

    return AnimationResult(
        success=True,
        video_path=video,
        char_cfg_path=cfg if cfg.exists() else None,
        duration_sec=duration,
        error=None,
    )
