"""Single-worker queue for AnimatedDrawings jobs."""
from __future__ import annotations

import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from animate.animated_drawings_runner import AnimationResult


RunnerFn = Callable[[np.ndarray, str, Path, Path, Path, float], AnimationResult]


class AnimationWorker:
    def __init__(
        self,
        runner: RunnerFn,
        ad_repo_path: Path,
        work_dir_base: Path,
        ad_python: Path,
        motion: str = "dab",
        timeout_sec: float = 30.0,
    ) -> None:
        self._runner = runner
        self._ad_repo_path = Path(ad_repo_path)
        self._work_dir_base = Path(work_dir_base)
        self._ad_python = Path(ad_python)
        self._motion = motion
        self._timeout_sec = timeout_sec
        self._executor = ThreadPoolExecutor(max_workers=1)

    def submit(self, texture_bgra: np.ndarray) -> Future:
        work_dir = self._work_dir_base / uuid.uuid4().hex
        return self._executor.submit(
            self._runner,
            texture_bgra,
            self._motion,
            self._ad_repo_path,
            work_dir,
            self._ad_python,
            self._timeout_sec,
        )

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)
