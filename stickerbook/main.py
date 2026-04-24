"""Entry point.

Examples:
    python main.py --camera 1            # webcam at /dev/video1
    python main.py --video clip.mp4      # looped playback of a recorded clip
"""
from __future__ import annotations

import argparse

from app import App
from config import DEFAULT


def main() -> None:
    parser = argparse.ArgumentParser(description="stickerbook — 2.5D AR stickerization PoC")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=None,
                     help=f"webcam index (default: {DEFAULT.camera_index})")
    src.add_argument("--video", type=str, default=None,
                     help="path to a recorded video file (looped)")
    parser.add_argument("--yolo-weights", type=str, default=DEFAULT.yolo_weights)
    parser.add_argument("--sam-weights", type=str, default=DEFAULT.mobile_sam_weights)
    args = parser.parse_args()

    camera_source: int | str
    if args.video is not None:
        camera_source = args.video
    elif args.camera is not None:
        camera_source = args.camera
    else:
        camera_source = DEFAULT.camera_index

    app = App(
        camera_source=camera_source,
        yolo_weights=args.yolo_weights,
        sam_weights=args.sam_weights,
    )
    app.run()


if __name__ == "__main__":
    main()
