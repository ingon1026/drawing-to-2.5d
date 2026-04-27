"""Main application state machine."""
from __future__ import annotations

import shutil
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from animate.animated_drawings_runner import AnimationResult, run_animated_drawings
from animate.animation_worker import AnimationWorker
from animate.torchserve_runtime import TorchServeRuntime
from capture.camera import Camera
from config import (
    AD_PYTHON,
    AD_REPO_PATH,
    ANIMATION_WORK_DIR,
    CAPTURES_DIR,
    TORCHSERVE_BIN,
    TORCHSERVE_CONFIG_PATH,
    TORCHSERVE_MODELS,
)
from detect.candidate_detector import CandidateDetector
from export.animated_drawings import save_sticker
from extract.segmenter import Segmenter, StickerAsset
from render.animated_sticker_renderer import AnimatedStickerRenderer
from render.overlay import draw_candidate_boxes
from render.spinner_overlay import draw_spinner
from render.tilt_renderer import render_bgra_as_billboard, render_sticker_as_billboard
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
    animation_work_dir: Optional[Path] = None  # for I2 cleanup
    popup_lift_ratio: float = 1.0  # set at creation by _promote_to_live


def _choose_popup_lift_ratio(source_region: Tuple[int, int, int, int]) -> float:
    """Choose popup_lift_ratio so the billboard's top edge stays >= 0 in image y.

    Returns 1.0 when there's full headroom (sy >= sh), reduces toward 0.0 as the
    drawing approaches the top of the frame. Floor at 0.0 (billboard sits flat
    at the source position, no popup).
    """
    sx, sy, sw, sh = source_region
    if sh <= 0:
        return 1.0
    return float(min(1.0, max(0.0, sy / sh)))


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
        order = [
            "capture", "poll", "detect", "track_render", "iter",
            "animation_success_sec", "animation_failure_sec",
        ]
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
        camera_source: Union[int, str] = 0,
        yolo_weights: str = "yolo26n.pt",
        sam_weights: Optional[str] = None,
    ) -> None:
        self.camera_source = camera_source
        self.state = AppState.SCAN
        self._yolo_weights = yolo_weights
        self._sam_weights = sam_weights
        self._current_frame: Optional[np.ndarray] = None
        self._pending: List[Tuple[Future, np.ndarray]] = []
        self._anchored: List[AnchoredSticker] = []
        self._segmenter: Optional[Segmenter] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._torchserve: Optional[TorchServeRuntime] = None
        self._animation_worker: Optional[AnimationWorker] = None
        self._animated_renderers: Dict[int, AnimatedStickerRenderer] = {}
        self._spinner_phase: float = 0.0

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

    def _cleanup_work_dir(self, work_dir: Optional[Path]) -> None:
        if work_dir is None:
            return
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception as e:
            print(f"[app] work_dir cleanup failed ({work_dir}): {e}")

    def _reset_stickers(self) -> None:
        for r in self._animated_renderers.values():
            r.release()
        self._animated_renderers.clear()
        count = len(self._anchored)
        for item in self._anchored:
            self._cleanup_work_dir(item.animation_work_dir)
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
                        item = self._promote_to_live(sticker, anchor)
                        x, y, w, h = sticker.source_region
                        print(
                            f"[app] sticker #{len(self._anchored)} created + anchored "
                            f"at ({x},{y}) size {w}x{h}, "
                            f"animation_state={item.animation_state.name}"
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

    def _promote_to_live(
        self, sticker_asset: StickerAsset, anchor: HomographyAnchor
    ) -> AnchoredSticker:
        item = AnchoredSticker(
            sticker=sticker_asset,
            anchor=anchor,
            popup_lift_ratio=_choose_popup_lift_ratio(sticker_asset.source_region),
        )
        if self._animation_worker is not None:
            item.animation_future = self._animation_worker.submit(sticker_asset.texture_bgra)
            item.animation_state = AnimationState.PREPARING
            item.animation_started_at = perf_counter()
        self._anchored.append(item)
        return item

    def _poll_animations(self, perf: "_PerfTracker") -> None:
        for item in self._anchored:
            if item.animation_state is not AnimationState.PREPARING:
                continue
            fut = item.animation_future
            if fut is None or not fut.done():
                continue
            try:
                result: AnimationResult = fut.result()
            except Exception as e:
                print(f"[app] animation worker raised: {e}")
                item.animation_state = AnimationState.FAILED
                item.animation_future = None
                continue
            if result.success and result.video_path is not None:
                item.animation_state = AnimationState.ANIMATED
                item.animation_video_path = result.video_path
                item.animation_work_dir = result.work_dir
                perf.record("animation_success_sec", result.duration_sec)
                print(
                    f"[app] sticker {id(item)} animated "
                    f"({result.duration_sec:.1f}s)"
                )
            else:
                item.animation_state = AnimationState.FAILED
                perf.record("animation_failure_sec", result.duration_sec)
                print(f"[app] animation failed: {result.error}")
                self._cleanup_work_dir(result.work_dir)
            item.animation_future = None

    def run(self) -> None:
        camera = Camera(source=self.camera_source)
        detector = CandidateDetector(yolo_weights=self._yolo_weights)
        if self._sam_weights is not None:
            self._segmenter = Segmenter(self._sam_weights)
            self._executor = ThreadPoolExecutor(max_workers=1)
            if not TORCHSERVE_CONFIG_PATH.exists():
                TORCHSERVE_CONFIG_PATH.write_text(
                    "default_workers_per_model=1\n"
                    "enable_metrics_api=false\n"
                )
            existing = TORCHSERVE_CONFIG_PATH.read_text() if TORCHSERVE_CONFIG_PATH.exists() else ""
            if "enable_metrics_api=false" not in existing:
                print(
                    f"[app] hint: append 'enable_metrics_api=false' to "
                    f"{TORCHSERVE_CONFIG_PATH} to silence nvgpu warnings"
                )
            self._torchserve = TorchServeRuntime(
                model_store=AD_REPO_PATH / "torchserve" / "model-store",
                config_path=TORCHSERVE_CONFIG_PATH,
                models=TORCHSERVE_MODELS,
                torchserve_bin=TORCHSERVE_BIN,
            )
            try:
                self._torchserve.start()
                self._animation_worker = AnimationWorker(
                    runner=run_animated_drawings,
                    ad_repo_path=AD_REPO_PATH,
                    work_dir_base=ANIMATION_WORK_DIR,
                    ad_python=AD_PYTHON,
                )
            except Exception as e:
                print(f"[app] WARNING: animation unavailable ({e}); stickers will remain STATIC")
                self._torchserve = None
                self._animation_worker = None

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
                self._poll_animations(perf)
                perf.record("poll_anim", perf_counter() - t0)

                t0 = perf_counter()
                boxes = detector.detect(raw)
                perf.record("detect", perf_counter() - t0)
                draw_candidate_boxes(display, boxes)

                t0 = perf_counter()
                for item in self._anchored:
                    state = item.anchor.update(raw)
                    if state.homography is None:
                        continue
                    if item.animation_state is AnimationState.ANIMATED:
                        renderer = self._animated_renderers.get(id(item))
                        if renderer is None and item.animation_video_path is not None:
                            renderer = AnimatedStickerRenderer(item.animation_video_path)
                            self._animated_renderers[id(item)] = renderer
                        if renderer is not None:
                            bgra = renderer.next_frame_bgra()
                            render_bgra_as_billboard(
                                frame=display,
                                texture_bgra=bgra,
                                source_region=item.sticker.source_region,
                                homography=state.homography,
                                popup_lift_ratio=item.popup_lift_ratio,
                            )
                        else:
                            render_sticker_as_billboard(
                                display,
                                item.sticker,
                                state.homography,
                                popup_lift_ratio=item.popup_lift_ratio,
                            )
                    else:
                        render_sticker_as_billboard(
                            display,
                            item.sticker,
                            state.homography,
                            popup_lift_ratio=item.popup_lift_ratio,
                        )
                        if item.animation_state is AnimationState.PREPARING:
                            x, y, w, h = item.sticker.source_region
                            cx, cy = x + w // 2, y + h // 2
                            self._spinner_phase += 0.1
                            draw_spinner(display, (cx, cy), min(w, h) // 4, self._spinner_phase)
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
            for item in self._anchored:
                self._cleanup_work_dir(item.animation_work_dir)
            for r in self._animated_renderers.values():
                r.release()
            if self._animation_worker is not None:
                self._animation_worker.shutdown(wait=False)
            if self._torchserve is not None:
                self._torchserve.stop()
            if self._executor is not None:
                self._executor.shutdown(wait=False)
            camera.release()
            cv2.destroyWindow(WINDOW_NAME)
