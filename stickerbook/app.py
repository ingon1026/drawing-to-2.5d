"""Main application state machine."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from capture.camera import Camera
from config import CAPTURES_DIR
from detect.candidate_detector import CandidateDetector
from export.animated_drawings import save_sticker
from extract.segmenter import Segmenter, StickerAsset
from render.overlay import draw_candidate_boxes
from render.tilt_renderer import render_sticker_as_billboard
from track.homography_anchor import HomographyAnchor


class AppState(Enum):
    SCAN = auto()
    EXTRACTING = auto()
    LIVE = auto()


class AppAction(Enum):
    QUIT = auto()
    RESET = auto()
    SAVE = auto()


class AnimationState(Enum):
    STATIC = auto()
    PREPARING = auto()
    ANIMATED = auto()
    FAILED = auto()


WINDOW_NAME = "stickerbook"


@dataclass
class AnchoredSticker:
    sticker: StickerAsset
    anchor: HomographyAnchor
    animation_state: AnimationState = AnimationState.STATIC
    animation_video_path: Optional[Path] = None
    animation_started_at: Optional[float] = None
    animation_future: Optional["Future"] = None  # set while PREPARING


class _PerfTracker:
    def __init__(self, window: int = 120) -> None:
        self._samples: Dict[str, List[float]] = defaultdict(list)
        self.window = window

    def record(self, name: str, seconds: float) -> None:
        s = self._samples[name]
        s.append(seconds)
        if len(s) > self.window:
            s.pop(0)

    def report(self) -> str:
        if not self._samples:
            return "  (no samples)"
        lines = []
        order = ["capture", "poll", "detect", "track_render", "iter"]
        ordered_names = [n for n in order if n in self._samples]
        ordered_names += [n for n in sorted(self._samples) if n not in order]
        for name in ordered_names:
            samples = self._samples[name]
            if not samples:
                continue
            arr = np.asarray(samples)
            avg_ms = float(arr.mean()) * 1000
            p95_ms = float(np.percentile(arr, 95)) * 1000
            peak_ms = float(arr.max()) * 1000
            lines.append(
                f"  {name:13s} avg={avg_ms:6.1f}ms  p95={p95_ms:6.1f}ms  "
                f"peak={peak_ms:6.1f}ms  (n={len(samples)})"
            )
        return "\n".join(lines)


class App:
    def __init__(
        self,
        camera_index: int = 0,
        yolo_weights: str = "yolo26n.pt",
        sam_weights: Optional[str] = None,
    ) -> None:
        self.camera_index = camera_index
        self.state = AppState.SCAN
        self._yolo_weights = yolo_weights
        self._sam_weights = sam_weights
        self._current_frame: Optional[np.ndarray] = None
        self._pending: List[Tuple[Future, np.ndarray]] = []
        self._anchored: List[AnchoredSticker] = []
        self._segmenter: Optional[Segmenter] = None
        self._executor: Optional[ThreadPoolExecutor] = None

    def _handle_key(self, key: int) -> Optional[AppAction]:
        if key == -1:
            return None
        masked = key & 0xFF
        if masked in (ord("q"), ord("Q"), 27):
            return AppAction.QUIT
        if masked in (ord("r"), ord("R")):
            return AppAction.RESET
        if masked in (ord("s"), ord("S")):
            return AppAction.SAVE
        return None

    def _reset_stickers(self) -> None:
        count = len(self._anchored)
        self._anchored.clear()
        self.state = AppState.SCAN
        print(f"[app] reset — cleared {count} sticker(s)")

    def _save_stickers(self) -> None:
        if not self._anchored:
            print("[app] save: no stickers to save")
            return
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_dir = Path(CAPTURES_DIR) / ts
        for i, item in enumerate(self._anchored, start=1):
            save_sticker(item.sticker, session_dir / f"sticker_{i:02d}")
        print(f"[app] saved {len(self._anchored)} sticker(s) to {session_dir}")

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._on_click(x, y)

    def _on_click(self, x: int, y: int) -> None:
        if self._current_frame is None or self._segmenter is None or self._executor is None:
            return
        frame_copy = self._current_frame.copy()
        future = self._executor.submit(self._segmenter.segment, frame_copy, (x, y))
        self._pending.append((future, frame_copy))
        self.state = AppState.EXTRACTING
        print(f"[app] click ({x},{y}) — SAM task submitted (pending={len(self._pending)})")

    def _poll_pending(self) -> None:
        still_pending: List[Tuple[Future, np.ndarray]] = []
        for future, ref_frame in self._pending:
            if future.done():
                try:
                    sticker = future.result()
                    anchor = HomographyAnchor()
                    anchor.initialize(ref_frame, sticker.source_region)
                    if anchor.is_lost():
                        print(
                            f"[app] sticker created but anchor init failed "
                            f"(featureless region); sticker discarded"
                        )
                    else:
                        self._anchored.append(AnchoredSticker(sticker=sticker, anchor=anchor))
                        x, y, w, h = sticker.source_region
                        print(
                            f"[app] sticker #{len(self._anchored)} created + anchored "
                            f"at ({x},{y}) size {w}x{h}"
                        )
                except Exception as e:
                    print(f"[app] segmentation failed: {e}")
            else:
                still_pending.append((future, ref_frame))
        self._pending = still_pending
        if self._pending:
            self.state = AppState.EXTRACTING
        elif self._anchored:
            self.state = AppState.LIVE
        else:
            self.state = AppState.SCAN

    def run(self) -> None:
        camera = Camera(index=self.camera_index)
        detector = CandidateDetector(yolo_weights=self._yolo_weights)
        if self._sam_weights is not None:
            self._segmenter = Segmenter(self._sam_weights)
            self._executor = ThreadPoolExecutor(max_workers=1)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        perf = _PerfTracker()
        try:
            while True:
                t_iter = perf_counter()

                t0 = perf_counter()
                raw = camera.read()
                self._current_frame = raw
                display = raw.copy()
                perf.record("capture", perf_counter() - t0)

                t0 = perf_counter()
                self._poll_pending()
                perf.record("poll", perf_counter() - t0)

                t0 = perf_counter()
                boxes = detector.detect(raw)
                perf.record("detect", perf_counter() - t0)
                draw_candidate_boxes(display, boxes)

                t0 = perf_counter()
                for item in self._anchored:
                    state = item.anchor.update(raw)
                    if state.homography is None:
                        continue
                    render_sticker_as_billboard(display, item.sticker, state.homography)
                perf.record("track_render", perf_counter() - t0)

                cv2.imshow(WINDOW_NAME, display)
                perf.record("iter", perf_counter() - t_iter)

                action = self._handle_key(cv2.waitKey(1))
                if action is AppAction.QUIT:
                    break
                if action is AppAction.RESET:
                    self._reset_stickers()
                if action is AppAction.SAVE:
                    self._save_stickers()
        finally:
            print(
                f"[perf] last {perf.window} frames "
                f"(stickers={len(self._anchored)}):"
            )
            print(perf.report())
            if self._executor is not None:
                self._executor.shutdown(wait=False)
            camera.release()
            cv2.destroyWindow(WINDOW_NAME)
