import time
from pathlib import Path

import numpy as np

from animate.animated_drawings_runner import AnimationResult
from animate.animation_worker import AnimationWorker


def _stub_runner_success(tex, motion, ad_repo_path, work_dir, ad_python, timeout_sec):
    return AnimationResult(
        success=True, video_path=Path("/dev/null"),
        char_cfg_path=None, duration_sec=0.01, error=None,
    )


def _stub_runner_raises(tex, motion, ad_repo_path, work_dir, ad_python, timeout_sec):
    raise RuntimeError("boom")


def test_submit_returns_future_resolving_to_animation_result() -> None:
    worker = AnimationWorker(
        runner=_stub_runner_success,
        ad_repo_path=Path("/tmp"),
        work_dir_base=Path("/tmp/wdir"),
        ad_python=Path("/fake/python"),
    )
    try:
        tex = np.zeros((4, 4, 4), dtype=np.uint8)
        fut = worker.submit(tex)
        result = fut.result(timeout=2.0)
    finally:
        worker.shutdown()

    assert isinstance(result, AnimationResult)
    assert result.success is True


def test_submit_processes_jobs_sequentially_not_in_parallel() -> None:
    started: list = []
    finished: list = []

    def slow_runner(tex, motion, ad_repo_path, work_dir, ad_python, timeout_sec):
        started.append(time.monotonic())
        time.sleep(0.05)
        finished.append(time.monotonic())
        return AnimationResult(
            success=True, video_path=Path("/dev/null"),
            char_cfg_path=None, duration_sec=0.05, error=None,
        )

    worker = AnimationWorker(
        runner=slow_runner,
        ad_repo_path=Path("/tmp"),
        work_dir_base=Path("/tmp/wdir"),
        ad_python=Path("/fake/python"),
    )
    try:
        tex = np.zeros((4, 4, 4), dtype=np.uint8)
        f1 = worker.submit(tex)
        f2 = worker.submit(tex)
        f3 = worker.submit(tex)
        for f in (f1, f2, f3):
            f.result(timeout=2.0)
    finally:
        worker.shutdown()

    # Second job cannot start before first one finishes
    assert started[1] >= finished[0] - 0.001
    assert started[2] >= finished[1] - 0.001


def test_worker_survives_runner_exception_and_keeps_processing_next() -> None:
    attempts = {"count": 0}

    def flaky(tex, motion, ad_repo_path, work_dir, ad_python, timeout_sec):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("first one fails")
        return AnimationResult(
            success=True, video_path=Path("/dev/null"),
            char_cfg_path=None, duration_sec=0.01, error=None,
        )

    worker = AnimationWorker(
        runner=flaky,
        ad_repo_path=Path("/tmp"),
        work_dir_base=Path("/tmp/wdir"),
        ad_python=Path("/fake/python"),
    )
    try:
        tex = np.zeros((4, 4, 4), dtype=np.uint8)
        f1 = worker.submit(tex)
        f2 = worker.submit(tex)
        r1 = f1.exception(timeout=2.0)
        r2 = f2.result(timeout=2.0)
    finally:
        worker.shutdown()

    assert r1 is not None  # exception propagates to caller
    assert r2.success is True
