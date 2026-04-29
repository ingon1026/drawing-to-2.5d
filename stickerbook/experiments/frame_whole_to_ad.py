"""Send the camera frame as-is to AnimatedDrawings (no detection, no SAM).

User aims the camera at one drawing, presses SPACE, and the whole frame is
piped into AD. AD's own segmenter+pose detector picks up the character.
Repeat for N samples.

Usage:
    cd stickerbook
    /usr/bin/python3 experiments/frame_whole_to_ad.py --samples 5 --camera 1

Keys:
    SPACE   capture current frame and run AD
    q       quit
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from animate.animated_drawings_runner import run_animated_drawings  # noqa: E402
from animate.torchserve_runtime import TorchServeRuntime  # noqa: E402
from capture.camera import Camera  # noqa: E402
from config import (  # noqa: E402
    AD_PYTHON,
    AD_REPO_PATH,
    DEFAULT,
    TORCHSERVE_BIN,
    TORCHSERVE_CONFIG_PATH,
    TORCHSERVE_MODELS,
)


AD_TIMEOUT_SEC = 180.0


def ensure_ts_config() -> None:
    if not TORCHSERVE_CONFIG_PATH.exists():
        TORCHSERVE_CONFIG_PATH.write_text("default_workers_per_model=1\n")


def to_bgra_full_alpha(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
    return np.concatenate([bgr, alpha], axis=-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--motion", default="dab")
    parser.add_argument("--camera", type=int, default=DEFAULT.camera_index)
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "experiments" / "results_whole",
    )
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = args.out / timestamp
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[exp] output root: {out_root}")

    ensure_ts_config()
    print("[exp] starting TorchServe …")
    runtime = TorchServeRuntime(
        model_store=AD_REPO_PATH / "torchserve" / "model-store",
        config_path=TORCHSERVE_CONFIG_PATH,
        models=TORCHSERVE_MODELS,
        torchserve_bin=TORCHSERVE_BIN,
        health_timeout_sec=60.0,
    )
    runtime.start()
    print("[exp] TorchServe healthy")

    cam = Camera(args.camera)
    win = "frame_whole_to_ad"
    cv2.namedWindow(win)

    captured = 0
    summary: list[dict] = []
    try:
        while captured < args.samples:
            try:
                frame = cam.read()
            except Exception as e:
                print(f"[exp] camera read failed: {e}")
                break

            disp = frame.copy()
            cv2.putText(
                disp,
                f"sample {captured + 1}/{args.samples}  SPACE=capture  q=quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != 32:  # SPACE
                continue

            captured += 1
            sample_dir = out_root / f"sample_{captured:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[exp] === sample {captured}/{args.samples} ===")

            cv2.imwrite(str(sample_dir / "frame.png"), frame)
            bgra = to_bgra_full_alpha(frame)

            cv2.putText(disp, "BUSY ... AD running", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(win, disp)
            cv2.waitKey(1)

            t0 = time.monotonic()
            result = run_animated_drawings(
                texture_bgra=bgra,
                motion=args.motion,
                ad_repo_path=AD_REPO_PATH,
                work_dir=sample_dir / "ad",
                ad_python=AD_PYTHON,
                timeout_sec=AD_TIMEOUT_SEC,
            )
            dt = time.monotonic() - t0
            ok = result.success and result.video_path is not None
            print(f"[exp] sample {captured}: success={ok} in {dt:.1f}s")
            if not ok:
                print(f"[exp]   error: {(result.error or '')[:300]}")

            if ok:
                gif = sample_dir / "ad" / "out" / "video.gif"
                mp4 = sample_dir / "ad" / "out" / "video.mp4"
                cfg = sample_dir / "ad" / "out" / "char_cfg.yaml"
                inp = sample_dir / "ad" / "input.png"
                for src, dst in [
                    (gif, sample_dir / "video.gif"),
                    (mp4, sample_dir / "video.mp4"),
                    (cfg, sample_dir / "char_cfg.yaml"),
                    (inp, sample_dir / "ad_input.png"),
                ]:
                    if src.exists():
                        shutil.copyfile(src, dst)

            summary.append({
                "sample": captured,
                "ok": ok,
                "duration_sec": dt,
                "error": (result.error or "")[:200] if not ok else "",
            })
    finally:
        cam.release()
        cv2.destroyAllWindows()
        try:
            runtime.stop()
        except Exception as e:
            print(f"[exp] torchserve stop raised: {e}")

    print("\n[exp] === SUMMARY ===")
    for row in summary:
        flag = "OK  " if row["ok"] else "FAIL"
        print(f"  sample {row['sample']:02d}: {flag}  {row['duration_sec']:.1f}s  {row['error']}")
    print(f"[exp] artifacts: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
