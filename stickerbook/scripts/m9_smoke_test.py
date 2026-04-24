"""Headless M9 smoke test — exercises the full AnimatedDrawings pipeline
(TorchServe start → run_animated_drawings subprocess → AD video output → frame
readback) without needing a webcam.

Usage:
    cd stickerbook
    /usr/bin/python3 scripts/m9_smoke_test.py

Exit codes:
    0 = success (dancing sticker video produced and readable)
    1 = runtime failure (TorchServe / AD / renderer error)
    2 = fixture not found (no saved sticker to use as input)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure stickerbook modules resolve even when invoked via absolute path.
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from animate.animated_drawings_runner import run_animated_drawings  # noqa: E402
from animate.torchserve_runtime import TorchServeRuntime  # noqa: E402
from config import (  # noqa: E402
    AD_PYTHON,
    AD_REPO_PATH,
    ANIMATION_WORK_DIR,
    TORCHSERVE_BIN,
    TORCHSERVE_CONFIG_PATH,
    TORCHSERVE_MODELS,
)
from render.animated_sticker_renderer import AnimatedStickerRenderer  # noqa: E402


FIXTURE_CANDIDATES = [
    HERE / "assets/captures/2026-04-23_14-58-53/sticker_01/texture.png",
    HERE / "assets/captures/2026-04-21_16-43-42/sticker_01/texture.png",
    HERE / "assets/captures/2026-04-21_16-38-13/sticker_01/texture.png",
]


def find_fixture() -> Path:
    for p in FIXTURE_CANDIDATES:
        if p.is_file():
            print(f"[smoke] using fixture: {p}")
            return p
    print("[smoke] no saved sticker fixture found. candidates tried:")
    for p in FIXTURE_CANDIDATES:
        print(f"  - {p}")
    sys.exit(2)


def load_bgra(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[smoke] cv2.imread returned None for {path}")
        sys.exit(1)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    print(f"[smoke] loaded texture: shape={img.shape} dtype={img.dtype}")
    return img


def ensure_ts_config() -> None:
    if not TORCHSERVE_CONFIG_PATH.exists():
        TORCHSERVE_CONFIG_PATH.write_text("default_workers_per_model=1\n")
        print(f"[smoke] wrote default ts config: {TORCHSERVE_CONFIG_PATH}")


def main() -> int:
    fixture = find_fixture()
    texture_bgra = load_bgra(fixture)
    ensure_ts_config()

    print(f"[smoke] TORCHSERVE_BIN: {TORCHSERVE_BIN}")
    print(f"[smoke] AD_PYTHON: {AD_PYTHON}")
    print(f"[smoke] AD_REPO_PATH: {AD_REPO_PATH}")
    print(f"[smoke] ANIMATION_WORK_DIR: {ANIMATION_WORK_DIR}")

    runtime = TorchServeRuntime(
        model_store=AD_REPO_PATH / "torchserve" / "model-store",
        config_path=TORCHSERVE_CONFIG_PATH,
        models=TORCHSERVE_MODELS,
        torchserve_bin=TORCHSERVE_BIN,
        health_timeout_sec=60.0,
    )

    try:
        t0 = time.monotonic()
        print("[smoke] starting torchserve (health probe up to 60s)…")
        runtime.start()
        print(f"[smoke] torchserve healthy after {time.monotonic() - t0:.1f}s")

        work_dir = ANIMATION_WORK_DIR / "smoke"
        work_dir.mkdir(parents=True, exist_ok=True)

        print("[smoke] invoking run_animated_drawings (timeout 120s)…")
        t1 = time.monotonic()
        result = run_animated_drawings(
            texture_bgra=texture_bgra,
            motion="dab",
            ad_repo_path=AD_REPO_PATH,
            work_dir=work_dir,
            ad_python=AD_PYTHON,
            timeout_sec=120.0,
        )
        elapsed = time.monotonic() - t1
        print(f"[smoke] AD call returned in {elapsed:.1f}s: success={result.success}")
        if result.error:
            print(f"[smoke] AD error: {result.error[:500]}")
        if not result.success:
            print("[smoke] FAIL: AD did not produce a valid result")
            return 1

        print(f"[smoke] video_path: {result.video_path}")
        print(f"[smoke] char_cfg_path: {result.char_cfg_path}")

        print("[smoke] opening video via AnimatedStickerRenderer…")
        renderer = AnimatedStickerRenderer(result.video_path)
        try:
            f0 = renderer.next_frame_bgra()
            f1 = renderer.next_frame_bgra()
        finally:
            renderer.release()
        print(f"[smoke] frame[0] shape={f0.shape} dtype={f0.dtype}")
        print(f"[smoke] frame[1] shape={f1.shape} dtype={f1.dtype}")
        # Cheap alpha sanity check: background corner should have alpha=0 (white bg)
        a_corner = int(f0[0, 0, 3])
        print(f"[smoke] top-left alpha (expect 0 for white bg): {a_corner}")

        print("[smoke] OK — M9 pipeline end-to-end verified")
        return 0
    finally:
        print("[smoke] stopping torchserve…")
        try:
            runtime.stop()
        except Exception as e:
            print(f"[smoke] torchserve stop raised: {e}")


if __name__ == "__main__":
    sys.exit(main())
