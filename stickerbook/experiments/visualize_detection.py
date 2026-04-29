"""Overlay detection boxes on the 5 captured frames so we can judge whether
the YOLO+contour pipeline produces a "whole stickman" box (raw-bbox approach
viable) or only fragmented part-level boxes.

Output (next to each frame.png):
    detection_overlay.png   green=YOLO, blue=contour, yellow=winner box
                            (smallest box containing the click), red dot=click

Usage:
    cd stickerbook
    /usr/bin/python3 experiments/visualize_detection.py \
        experiments/results/<timestamp>/
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from config import DEFAULT  # noqa: E402
from detect.candidate_detector import CandidateDetector  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        latest = sorted((HERE / "experiments" / "results").glob("*"))[-1]
        print(f"[viz] no path given, using latest: {latest}")
        results_dir = latest
    else:
        results_dir = Path(sys.argv[1])

    samples = sorted(results_dir.glob("sample_*"))
    if not samples:
        print(f"[viz] no sample_* dirs under {results_dir}")
        return 1

    print(f"[viz] loading YOLO ({DEFAULT.yolo_weights}) …")
    det = CandidateDetector(
        yolo_weights=DEFAULT.yolo_weights,
        yolo_conf_threshold=0.25,
    )

    for sdir in samples:
        frame_path = sdir / "frame.png"
        click_path = sdir / "click.txt"
        if not frame_path.exists() or not click_path.exists():
            print(f"[viz] skip {sdir.name}: missing frame.png or click.txt")
            continue
        frame = cv2.imread(str(frame_path))
        cx, cy = (int(v) for v in click_path.read_text().strip().split(","))

        boxes = det.detect(frame)
        n_yolo = sum(1 for b in boxes if b.source == "yolo")
        n_contour = sum(1 for b in boxes if b.source == "contour")

        winners = [
            b for b in boxes
            if b.x <= cx <= b.x + b.w and b.y <= cy <= b.y + b.h
        ]
        winner = min(winners, key=lambda b: b.w * b.h, default=None)

        overlay = frame.copy()
        for b in boxes:
            color = (0, 255, 0) if b.source == "yolo" else (255, 100, 0)
            cv2.rectangle(overlay, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 1)
        if winner is not None:
            cv2.rectangle(
                overlay,
                (winner.x, winner.y),
                (winner.x + winner.w, winner.y + winner.h),
                (0, 255, 255),
                3,
            )
        cv2.circle(overlay, (cx, cy), 8, (0, 0, 255), -1)
        label = f"{len(boxes)} boxes  yolo={n_yolo} contour={n_contour}"
        cv2.putText(
            overlay, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )

        out = sdir / "detection_overlay.png"
        cv2.imwrite(str(out), overlay)
        wbox = (
            f"({winner.x},{winner.y},{winner.w},{winner.h}) src={winner.source}"
            if winner else "NO WINNER (click not inside any box)"
        )
        print(f"[viz] {sdir.name}: yolo={n_yolo} contour={n_contour} winner={wbox}")

    print(f"[viz] done. open: {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
