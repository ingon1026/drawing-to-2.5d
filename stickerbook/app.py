"""Main application state machine."""
from __future__ import annotations

import shutil
import time
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
import yaml

from animate.animated_drawings_runner import AnimationResult, run_animated_drawings
from animate.animation_worker import AnimationWorker
from animate.torchserve_runtime import TorchServeRuntime
from capture.camera import Camera
from config import (
    AD_PYTHON,
    AD_REPO_PATH,
    ANIMATION_WORK_DIR,
    CAPTURES_DIR,
    ROOT,
    TORCHSERVE_BIN,
    TORCHSERVE_CONFIG_PATH,
    TORCHSERVE_MODELS,
)
from detect.candidate_detector import CandidateDetector
from export.animated_drawings import save_sticker
from extract.segmenter import Segmenter, StickerAsset
from motion.library import MotionLibrary
from motion.pipeline import MotionPipeline
from motion.pose_estimator import PoseEstimator
from motion.recorder import FrameRecorder
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
    CAPTURE = auto()  # SPACE: send current frame to AD pipeline
    RECORD_TOGGLE = auto()  # M 키
    SELECT_MOTION_1 = auto()
    SELECT_MOTION_2 = auto()
    SELECT_MOTION_3 = auto()
    SELECT_MOTION_4 = auto()
    SELECT_MOTION_5 = auto()
    TOGGLE_LIBRARY_VIEW = auto()  # L key


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
        self._motion_library: Optional[MotionLibrary] = None
        self._motion_pipeline: Optional[MotionPipeline] = None
        self._motion_pose_estimator: Optional[PoseEstimator] = None
        self._motion_recorder: Optional[FrameRecorder] = None
        self._show_library: bool = False

    @staticmethod
    def _ask_motion_name() -> Optional[str]:
        """Pop a tkinter dialog asking for a motion name. None on cancel/empty."""
        try:
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk()
            root.withdraw()
            try:
                name = simpledialog.askstring(
                    "Motion name",
                    "이 동작 이름은? (취소 시 자동 motion_NNN):",
                    parent=root,
                )
            finally:
                root.destroy()
            if name is None:
                return None
            name = name.strip()
            return name if name else None
        except Exception as e:
            print(f"[app] motion name dialog failed: {e}")
            return None

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
        if masked == 32:  # SPACE
            return AppAction.CAPTURE
        if masked == ord("m") or masked == ord("M"):
            return AppAction.RECORD_TOGGLE
        if masked == ord("1"):
            return AppAction.SELECT_MOTION_1
        if masked == ord("2"):
            return AppAction.SELECT_MOTION_2
        if masked == ord("3"):
            return AppAction.SELECT_MOTION_3
        if masked == ord("4"):
            return AppAction.SELECT_MOTION_4
        if masked == ord("5"):
            return AppAction.SELECT_MOTION_5
        if masked == ord("l") or masked == ord("L"):
            return AppAction.TOGGLE_LIBRARY_VIEW
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
        # Click input is disabled in the new "frame-whole" mode; SPACE replaces it.
        return

    def _on_space(self) -> None:
        """SPACE: pipe the current camera frame as-is into AnimatedDrawings."""
        if self._current_frame is None or self._executor is None:
            return
        if self._animation_worker is None:
            print("[app] SPACE ignored: AD pipeline not available (TorchServe failed?)")
            return
        frame_copy = self._current_frame.copy()
        future = self._executor.submit(self._run_ad_pipeline, frame_copy)
        self._pending.append((future, frame_copy))
        self.state = AppState.EXTRACTING
        print(f"[app] SPACE — AD task submitted (pending={len(self._pending)})")

    def _run_ad_pipeline(
        self, frame_bgr: np.ndarray
    ) -> Tuple[StickerAsset, AnimationResult]:
        """Background task: send whole BGR frame to AD, return (StickerAsset, AnimationResult).

        StickerAsset.source_region is built from AD's bounding_box.yaml so the
        existing HomographyAnchor / billboard renderer keep working unchanged.
        """
        h, w = frame_bgr.shape[:2]
        bgra = np.dstack([frame_bgr, np.full((h, w), 255, dtype=np.uint8)])

        work_dir = ANIMATION_WORK_DIR / f"sticker_{int(time.time() * 1000)}"
        result = run_animated_drawings(
            texture_bgra=bgra,
            motion=(
                self._motion_library.active()
                if self._motion_library is not None and self._motion_library.active()
                else "my_dance_3"
            ),
            ad_repo_path=AD_REPO_PATH,
            work_dir=work_dir,
            ad_python=AD_PYTHON,
            timeout_sec=180.0,
        )
        if not result.success or result.video_path is None:
            raise RuntimeError(result.error or "AD failed without error message")

        out_dir = work_dir / "out"
        bbox_path = out_dir / "bounding_box.yaml"
        if not bbox_path.is_file():
            raise RuntimeError(f"missing bounding_box.yaml at {bbox_path}")
        bbox_data = yaml.safe_load(bbox_path.read_text())
        left = int(bbox_data["left"])
        top = int(bbox_data["top"])
        right = int(bbox_data["right"])
        bottom = int(bbox_data["bottom"])
        source_region = (left, top, right - left, bottom - top)

        texture_path = out_dir / "texture.png"
        texture_bgra = cv2.imread(str(texture_path), cv2.IMREAD_UNCHANGED)
        if texture_bgra is None:
            raise RuntimeError(f"failed to read AD texture at {texture_path}")
        if texture_bgra.ndim == 2:
            texture_bgra = cv2.cvtColor(texture_bgra, cv2.COLOR_GRAY2BGRA)
        elif texture_bgra.shape[2] == 3:
            texture_bgra = cv2.cvtColor(texture_bgra, cv2.COLOR_BGR2BGRA)

        mask_path = out_dir / "mask.png"
        mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_u8 is None:
            mask_u8 = (texture_bgra[..., 3] > 0).astype(np.uint8) * 255

        sticker_asset = StickerAsset(
            texture_bgra=texture_bgra,
            mask_u8=mask_u8,
            source_region=source_region,
        )
        return sticker_asset, result

    def _poll_pending(self) -> None:
        still_pending: List[Tuple[Future, np.ndarray]] = []
        for future, ref_frame in self._pending:
            if future.done():
                try:
                    sticker_asset, anim_result = future.result()
                    anchor = HomographyAnchor()
                    anchor.initialize(ref_frame, sticker_asset.source_region)
                    if anchor.is_lost():
                        print(
                            f"[app] sticker created but anchor init failed "
                            f"(featureless region); sticker discarded"
                        )
                        self._cleanup_work_dir(anim_result.work_dir)
                    else:
                        item = self._promote_to_live(sticker_asset, anchor, anim_result)
                        x, y, w, h = sticker_asset.source_region
                        print(
                            f"[app] sticker #{len(self._anchored)} created + anchored "
                            f"at ({x},{y}) size {w}x{h}, "
                            f"animation_state={item.animation_state.name}"
                        )
                except Exception as e:
                    print(f"[app] AD pipeline failed: {e}")
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
        self,
        sticker_asset: StickerAsset,
        anchor: HomographyAnchor,
        anim_result: AnimationResult,
    ) -> AnchoredSticker:
        # AD has already finished by the time we get here (one-shot pipeline),
        # so the sticker enters ANIMATED state directly. PREPARING/spinner is
        # bypassed.
        item = AnchoredSticker(
            sticker=sticker_asset,
            anchor=anchor,
            popup_lift_ratio=_choose_popup_lift_ratio(sticker_asset.source_region),
            animation_state=AnimationState.ANIMATED,
            animation_video_path=anim_result.video_path,
            animation_work_dir=anim_result.work_dir,
        )
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
        # New input mode (SPACE → frame → AD) does not use SAM; segmenter stays None.
        # AnimationWorker is also bypassed; AD is invoked directly inside _run_ad_pipeline.
        # We keep the executor + TorchServe so SPACE can dispatch background AD jobs.
        self._executor = ThreadPoolExecutor(max_workers=2)
        if not TORCHSERVE_CONFIG_PATH.exists():
            TORCHSERVE_CONFIG_PATH.write_text(
                "default_workers_per_model=1\n"
                "enable_metrics_api=false\n"
            )
        self._torchserve = TorchServeRuntime(
            model_store=AD_REPO_PATH / "torchserve" / "model-store",
            config_path=TORCHSERVE_CONFIG_PATH,
            models=TORCHSERVE_MODELS,
            torchserve_bin=TORCHSERVE_BIN,
        )
        try:
            self._torchserve.start()
            self._animation_worker = "ready"  # marker only; SPACE checks this
        except Exception as e:
            print(f"[app] WARNING: AD unavailable ({e}); SPACE will be a no-op")
            self._torchserve = None
            self._animation_worker = None

        # Motion recording pipeline (mediapipe optional dep)
        try:
            self._motion_recorder = FrameRecorder()
            self._motion_pose_estimator = PoseEstimator()
            self._motion_library = MotionLibrary(
                library_dir=ROOT / "assets" / "motions" / "library",
                ad_repo_path=AD_REPO_PATH,
            )
            self._motion_pipeline = MotionPipeline(
                recorder=self._motion_recorder,
                estimator=self._motion_pose_estimator,
                library=self._motion_library,
                tmp_dir=Path("/tmp/stickerbook_motion"),
                fps=30.0,
            )
            print(f"[app] motion library ready ({len(self._motion_library.list())} motions)")
        except Exception as e:
            print(f"[app] WARNING: motion pipeline unavailable ({e}); M/1-5 keys disabled")
            self._motion_pipeline = None

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
                if self._motion_recorder is not None:
                    self._motion_recorder.add_frame(raw)
                perf.record("capture", perf_counter() - t0)

                t0 = perf_counter()
                self._poll_pending()
                perf.record("poll", perf_counter() - t0)

                t0 = perf_counter()
                self._poll_animations(perf)
                perf.record("poll_anim", perf_counter() - t0)

                # Frame-whole input mode: detection candidates are not used,
                # so we don't run them or draw boxes on the live view.

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

                # HUD: motion library status
                if self._motion_recorder is not None and self._motion_recorder.is_recording():
                    cv2.putText(
                        display, "REC", (display.shape[1] - 100, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3,
                    )
                if self._motion_library is not None:
                    active = self._motion_library.active()
                    n = len(self._motion_library.list())
                    label = f"motion: {active or 'default'}  ({n} in lib)"
                    cv2.putText(
                        display, label, (10, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
                    )

                if self._show_library and self._motion_library is not None:
                    names = self._motion_library.list()
                    active = self._motion_library.active()
                    cv2.putText(
                        display, "Library:", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1,
                    )
                    for i, n in enumerate(names[:10]):  # max 10 shown
                        marker = "*" if n == active else " "
                        line = f"{marker} {i+1}. {n}"
                        cv2.putText(
                            display, line, (10, 105 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
                        )

                cv2.imshow(WINDOW_NAME, display)
                perf.record("iter", perf_counter() - t_iter)

                action = self._handle_key(cv2.waitKey(1))
                if action is AppAction.QUIT:
                    break
                if action is AppAction.RESET:
                    self._reset_stickers()
                if action is AppAction.SAVE:
                    self._save_stickers()
                if action is AppAction.CAPTURE:
                    self._on_space()
                if action is AppAction.RECORD_TOGGLE:
                    if self._motion_pipeline is not None:
                        is_stopping = (
                            self._motion_recorder is not None
                            and self._motion_recorder.is_recording()
                        )
                        if is_stopping:
                            chosen_name = self._ask_motion_name()
                        else:
                            chosen_name = None
                        self._motion_pipeline.toggle(name=chosen_name)
                if action is AppAction.TOGGLE_LIBRARY_VIEW:
                    self._show_library = not self._show_library
                if action in (
                    AppAction.SELECT_MOTION_1, AppAction.SELECT_MOTION_2,
                    AppAction.SELECT_MOTION_3, AppAction.SELECT_MOTION_4,
                    AppAction.SELECT_MOTION_5,
                ):
                    if self._motion_library is not None:
                        idx = {
                            AppAction.SELECT_MOTION_1: 1,
                            AppAction.SELECT_MOTION_2: 2,
                            AppAction.SELECT_MOTION_3: 3,
                            AppAction.SELECT_MOTION_4: 4,
                            AppAction.SELECT_MOTION_5: 5,
                        }[action]
                        name = self._motion_library.get_by_index(idx)
                        if name is not None:
                            self._motion_library.set_active(name)
                            print(f"[app] active motion = {name}")
                        else:
                            print(f"[app] no motion at index {idx}")
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
            # _animation_worker is now just a "ready" marker (str) under the
            # frame-whole input mode, not the AnimationWorker class — no shutdown call.
            if self._torchserve is not None:
                self._torchserve.stop()
            if self._executor is not None:
                self._executor.shutdown(wait=False)
            if self._motion_pose_estimator is not None:
                self._motion_pose_estimator.close()
            camera.release()
            cv2.destroyWindow(WINDOW_NAME)
