"""MotionPipeline — M toggle entry. Wires recorder + estimator + bvh_writer + library.

First call: recorder.start().
Second call: stop -> estimate -> write bvh -> library.add -> library.set_active.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from motion.bvh_writer import write_bvh
from motion.library import MotionLibrary
from motion.pose_estimator import PoseEstimator
from motion.recorder import FrameRecorder


MIN_FRAMES = 30           # 1초 미만 녹화는 거부
MAX_FAIL_RATE = 0.5       # 인식 실패율 50% 초과 시 거부


class MotionPipeline:
    def __init__(
        self,
        recorder: FrameRecorder,
        estimator: PoseEstimator,
        library: MotionLibrary,
        tmp_dir: Path,
        fps: float = 30.0,
    ) -> None:
        self._recorder = recorder
        self._estimator = estimator
        self._library = library
        self._tmp_dir = Path(tmp_dir)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self._fps = fps

    def toggle(self) -> Optional[str]:
        if not self._recorder.is_recording():
            self._recorder.start()
            print("[motion] REC start")
            return None
        return self._stop_and_process()

    def _stop_and_process(self) -> Optional[str]:
        frames = self._recorder.stop()
        print(f"[motion] REC stop ({len(frames)} frames)")

        if len(frames) < MIN_FRAMES:
            print(f"[motion] aborted: only {len(frames)} frames (< {MIN_FRAMES})")
            return None

        landmarks = self._estimator.estimate_batch(frames)
        n_fail = sum(1 for lm in landmarks if lm is None)
        fail_rate = n_fail / len(landmarks)
        if fail_rate > MAX_FAIL_RATE:
            print(
                f"[motion] aborted: pose recognition failure rate "
                f"{fail_rate:.0%} (> {MAX_FAIL_RATE:.0%})"
            )
            return None

        tmp_bvh = self._tmp_dir / f"motion_{int(time.time())}.bvh"
        try:
            write_bvh(landmarks, fps=self._fps, output_path=tmp_bvh)
        except Exception as e:
            print(f"[motion] aborted: bvh write failed: {e}")
            return None

        try:
            name = self._library.add(tmp_bvh)
        except Exception as e:
            print(f"[motion] aborted: library add failed: {e}")
            return None

        self._library.set_active(name)
        print(f"[motion] saved {name}, active")
        return name
