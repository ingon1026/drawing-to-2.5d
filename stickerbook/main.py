"""Entry point. Run from the stickerbook/ directory: `python main.py --camera 1`."""
from __future__ import annotations

import argparse

from app import App
from config import DEFAULT


def main() -> None:
    parser = argparse.ArgumentParser(description="stickerbook — 2.5D AR stickerization PoC")
    parser.add_argument("--camera", type=int, default=DEFAULT.camera_index)
    parser.add_argument("--yolo-weights", type=str, default=DEFAULT.yolo_weights)
    parser.add_argument("--sam-weights", type=str, default=DEFAULT.mobile_sam_weights)
    args = parser.parse_args()

    app = App(
        camera_index=args.camera,
        yolo_weights=args.yolo_weights,
        sam_weights=args.sam_weights,
    )
    app.run()


if __name__ == "__main__":
    main()
