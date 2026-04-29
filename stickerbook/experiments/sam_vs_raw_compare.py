"""SAM vs raw bbox crop comparison.

For each click on a live camera frame, runs the AnimatedDrawings pipeline
twice on the same source_region:
  (a) SAM-segmented BGRA  -> AD  -> sam.gif    (current V1 behavior)
  (b) raw BGR crop (alpha=255 fill, no segmentation) -> AD -> raw.gif
Both inputs use the identical bbox so the only varying factor is "SAM mask vs
raw rectangle". Outputs go to experiments/results/<timestamp>/sample_NN/.

Usage:
    cd stickerbook
    /usr/bin/python3 experiments/sam_vs_raw_compare.py --samples 5

Keys (camera window):
    left-click  capture this frame & run both AD pipelines
    q           quit early
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
from extract.segmenter import Segmenter  # noqa: E402


AD_TIMEOUT_SEC = 120.0


def ensure_ts_config() -> None:
    if not TORCHSERVE_CONFIG_PATH.exists():
        TORCHSERVE_CONFIG_PATH.write_text("default_workers_per_model=1\n")


def to_bgra_full_alpha(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
    return np.concatenate([bgr, alpha], axis=-1)


def run_one_pipeline(
    label: str,
    bgra: np.ndarray,
    motion: str,
    work_dir: Path,
) -> tuple[bool, str]:
    print(f"[exp]   [{label}] AD start (input {bgra.shape}) …")
    t0 = time.monotonic()
    result = run_animated_drawings(
        texture_bgra=bgra,
        motion=motion,
        ad_repo_path=AD_REPO_PATH,
        work_dir=work_dir,
        ad_python=AD_PYTHON,
        timeout_sec=AD_TIMEOUT_SEC,
    )
    dt = time.monotonic() - t0
    if result.success and result.video_path is not None:
        print(f"[exp]   [{label}] OK in {dt:.1f}s -> {result.video_path}")
        return True, str(result.video_path)
    err = (result.error or "no error msg")[:300]
    print(f"[exp]   [{label}] FAIL in {dt:.1f}s -> {err}")
    return False, err


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--motion", default="dab")
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "experiments" / "results",
    )
    parser.add_argument("--camera", type=int, default=DEFAULT.camera_index)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = args.out / timestamp
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[exp] output root: {out_root}")
    print(f"[exp] motion: {args.motion}")
    print(f"[exp] samples target: {args.samples}")

    ensure_ts_config()
    print("[exp] starting TorchServe (up to 60s) …")
    runtime = TorchServeRuntime(
        model_store=AD_REPO_PATH / "torchserve" / "model-store",
        config_path=TORCHSERVE_CONFIG_PATH,
        models=TORCHSERVE_MODELS,
        torchserve_bin=TORCHSERVE_BIN,
        health_timeout_sec=60.0,
    )
    runtime.start()
    print("[exp] TorchServe healthy")

    print("[exp] loading SAM …")
    segmenter = Segmenter(DEFAULT.mobile_sam_weights)

    cam = Camera(args.camera)
    win = "sam_vs_raw"
    cv2.namedWindow(win)

    state: dict = {"frame": None, "click": None, "busy": False}

    def on_mouse(event: int, x: int, y: int, flags: int, _):
        if event == cv2.EVENT_LBUTTONDOWN and not state["busy"] and state["frame"] is not None:
            state["click"] = (x, y)

    cv2.setMouseCallback(win, on_mouse)

    captured = 0
    summary: list[dict] = []
    try:
        while captured < args.samples:
            try:
                frame = cam.read()
            except Exception as e:
                print(f"[exp] camera read failed: {e}")
                break
            state["frame"] = frame

            disp = frame.copy()
            status = "BUSY (running AD …)" if state["busy"] else "click drawing"
            cv2.putText(
                disp,
                f"sample {captured + 1}/{args.samples}  {status}  (q=quit)",
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

            if state["click"] is None or state["busy"]:
                continue

            click = state["click"]
            state["click"] = None
            state["busy"] = True
            captured += 1
            sample_dir = out_root / f"sample_{captured:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[exp] === sample {captured}/{args.samples}  click={click} ===")

            cv2.imwrite(str(sample_dir / "frame.png"), frame)
            (sample_dir / "click.txt").write_text(f"{click[0]},{click[1]}\n")

            try:
                sticker = segmenter.segment(frame, click)
            except Exception as e:
                print(f"[exp]   SAM segment failed: {e}")
                state["busy"] = False
                summary.append({"sample": captured, "sam_ok": False, "raw_ok": False, "note": f"sam fail: {e}"})
                continue

            x, y, w, h = sticker.source_region
            print(f"[exp]   source_region (x,y,w,h)=({x},{y},{w},{h})")
            cv2.imwrite(str(sample_dir / "sam_texture.png"), sticker.texture_bgra)

            sam_work = sample_dir / "ad_sam"
            sam_ok, sam_note = run_one_pipeline("SAM", sticker.texture_bgra, args.motion, sam_work)

            raw_bgr = frame[y : y + h, x : x + w].copy()
            raw_bgra = to_bgra_full_alpha(raw_bgr)
            cv2.imwrite(str(sample_dir / "raw_crop.png"), raw_bgr)

            raw_work = sample_dir / "ad_raw"
            raw_ok, raw_note = run_one_pipeline("RAW", raw_bgra, args.motion, raw_work)

            for label, ok, work in (("sam", sam_ok, sam_work), ("raw", raw_ok, raw_work)):
                if not ok:
                    continue
                gif_src = work / "out" / "video.gif"
                mp4_src = work / "out" / "video.mp4"
                cfg_src = work / "out" / "char_cfg.yaml"
                input_src = work / "input.png"
                if input_src.exists():
                    shutil.copyfile(input_src, sample_dir / f"{label}_input.png")
                if gif_src.exists():
                    shutil.copyfile(gif_src, sample_dir / f"{label}.gif")
                elif mp4_src.exists():
                    shutil.copyfile(mp4_src, sample_dir / f"{label}.mp4")
                if cfg_src.exists():
                    shutil.copyfile(cfg_src, sample_dir / f"{label}_char_cfg.yaml")

            summary.append({
                "sample": captured,
                "sam_ok": sam_ok,
                "raw_ok": raw_ok,
                "sam_note": sam_note,
                "raw_note": raw_note,
            })
            print(f"[exp]   sample {captured} done -> {sample_dir}")
            state["busy"] = False
    finally:
        cam.release()
        cv2.destroyAllWindows()
        try:
            runtime.stop()
        except Exception as e:
            print(f"[exp] torchserve stop raised: {e}")

    print("\n[exp] === SUMMARY ===")
    for row in summary:
        print(
            f"  sample {row['sample']:02d}: "
            f"SAM={'OK' if row['sam_ok'] else 'FAIL'}  "
            f"RAW={'OK' if row['raw_ok'] else 'FAIL'}"
        )
    print(f"[exp] artifacts: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
