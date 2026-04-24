# M9 — AnimatedDrawings 라이브 통합 (춤추는 딱지) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 클릭으로 생성된 stickerbook 딱지에 AnimatedDrawings dab 댄스 애니메이션을 자동 적용하여, 종이 위에서 춤추는 2.5D 캐릭터를 라이브 카메라 화면에 렌더한다.

**Architecture:** TorchServe 를 Docker 없이 네이티브로 띄우고, AD `image_to_animation.py` 스크립트를 subprocess 로 호출한다. 결과 비디오를 cv2.VideoCapture 로 스트림하면서 기존 `billboard_corners_2d + warpPerspective` 빌보드 렌더링 체인에 프레임 단위로 합성한다. 단일 워커 큐로 다중 클릭을 순차 처리하고, 실패는 조용히 정지 딱지로 폴백한다.

**Tech Stack:** Python 3.10+, OpenCV (cv2.VideoCapture / warpPerspective), numpy, TorchServe 네이티브 (pip + openjdk-17), AnimatedDrawings (ARAP + OpenGL 렌더링, 기존 `/home/ingon/AR_book/AnimatedDrawings/`), concurrent.futures.ThreadPoolExecutor, pytest.

**Spec reference:** `stickerbook/docs/DESIGN.md` § "M9 — AnimatedDrawings 라이브 통합 (춤추는 딱지)"

---

## File Structure (M9 추가분)

신규:
- `animate/__init__.py`
- `animate/torchserve_runtime.py` — TorchServe 네이티브 기동/정지 + health check
- `animate/animated_drawings_runner.py` — AD 스크립트 shell out + 흰 배경 합성 + 결과 파싱 + joint spread sanity check
- `animate/animation_worker.py` — 단일 워커 큐
- `render/animated_sticker_renderer.py` — 비디오 프레임 루프 재생 + 알파 처리
- `render/spinner_overlay.py` — PREPARING 상태 시각화
- `docs/M9_1_OUTPUT_FORMAT.md` — M9.1 검증 결과 기록 (커밋 산출물)
- `tests/test_torchserve_runtime.py`
- `tests/test_animated_drawings_runner.py`
- `tests/test_animation_worker.py`
- `tests/test_animated_sticker_renderer.py`
- `tests/test_spinner_overlay.py`

수정:
- `app.py` — TorchServeRuntime 생명주기, 클릭 시 AnimationWorker.submit, 상태별 렌더 분기, 퍼프 기록
- `tests/test_app.py` — 새 상태 전환 검증
- `requirements.txt` — torchserve, torch-model-archiver 추가 (주석으로 갤탭 포팅 시 제거 표시)
- `README.md` — Docker 섹션 제거, 네이티브 TorchServe 설치 가이드, M9 사용법
- `docs/DESIGN.md` — (이미 기록됨) M9 마일스톤 진행 상태 업데이트

---

## Task 1: M9.1 — AD 출력 포맷 검증

**목적:** R8 (알파 채널/프레임 소스) 해소. 코드 작성 전 manual 검증으로 재생 전략 확정.

**Files:**
- Create: `docs/M9_1_OUTPUT_FORMAT.md`

- [ ] **Step 1: AD 환경 확인 (Docker 아직 살아있으면 사용, 아니면 M9.2 이후로 이동)**

Run:
```bash
docker ps --filter name=docker_torchserve --format '{{.Status}}'
```
Expected output: `Up X minutes` 또는 empty.

If empty:
```bash
# Skip to Task 2 first (TorchServe native install), then come back
```

- [ ] **Step 2: AD CLI 재실행 (boy sticker) 후 출력물 나열**

Run:
```bash
conda activate animated_drawings 2>/dev/null || source ~/AR_book/AnimatedDrawings/.venv/bin/activate
mkdir -p /tmp/m9_1_out
cd ~/AR_book/AnimatedDrawings
python examples/image_to_animation.py \
  stickerbook_test_inputs/boy_white_bg.png \
  /tmp/m9_1_out
ls -la /tmp/m9_1_out/
file /tmp/m9_1_out/video.* 2>/dev/null
```

**Note:** `stickerbook_test_inputs/boy_white_bg.png` 가 없으면 M7.5 에서 만든 composite 경로로 대체 (`/tmp/ad_input/sticker_01_on_paper.png` 재생성 필요). 이 스텝의 목적은 출력 포맷 관찰이므로 입력은 어떤 유효 PNG 든 OK.

Expected outputs (확인 항목):
- `video.gif` 존재? 크기 (~3MB 이전 관찰)
- `video.mp4` 존재? 알파 채널 유무
- `char_cfg.yaml` joint 개수 16 개?
- 프레임 수 / FPS

- [ ] **Step 3: cv2.VideoCapture 로 프레임 읽기 테스트**

Run:
```bash
python3 -c "
import cv2
for p in ['/tmp/m9_1_out/video.mp4', '/tmp/m9_1_out/video.gif']:
    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        print(f'{p}: CANNOT open')
        continue
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, f = cap.read()
    print(f'{p}: {w}x{h} {fps}fps {n}frames, first frame shape={f.shape if ok else None}, dtype={f.dtype if ok else None}')
    if ok:
        # 전경/배경 색 확인 (좌상단 vs 중앙)
        print(f'  top-left pixel (bg): {f[0,0]}')
        print(f'  center pixel: {f[h//2, w//2]}')
    cap.release()
"
```

Expected observations (가능성):
- MP4: BGR 3채널, 알파 없음 → 배경이 검정 또는 흰색일 것. 크로마키 필요.
- GIF: BGR 3채널 (cv2.VideoCapture 가 GIF 도 BGR 로 디코드), 배경은 투명 → 검정으로 디코드될 가능성.

- [ ] **Step 4: 재생 전략 확정 및 기록**

Create `/home/ingon/AR_book/stickerbook/docs/M9_1_OUTPUT_FORMAT.md`:

```markdown
# M9.1 — AnimatedDrawings 출력 포맷 검증 결과

검증일: 2026-04-24 (Task 1 수행 시점으로 업데이트)
검증 대상: AnimatedDrawings `image_to_animation.py` 기본 설정 (motion_cfg=dab)

## 출력물 관찰

| 파일 | 존재 | 크기 | 해상도 | FPS | 프레임 수 | 채널 | 배경 |
|---|---|---|---|---|---|---|---|
| video.mp4 | 예/아니오 | (실측) | (실측) | (실측) | (실측) | BGR 3ch | (실측 색) |
| video.gif | 예/아니오 | (실측) | (실측) | (실측) | (실측) | BGR 3ch | (실측 색) |
| char_cfg.yaml | 예/아니오 | — | — | — | — | — | — |
| joint_overlay.png | 예/아니오 | — | — | — | — | — | — |

## cv2.VideoCapture 호환성

- MP4: (ok/nok + 비고)
- GIF: (ok/nok + 비고)

## 재생 전략 결정

**선택:** (아래 중 하나)
- (A) MP4 + 크로마키 (검정→투명 변환)
- (B) GIF + cv2.VideoCapture
- (C) PNG 시퀀스 추출 (ffmpeg 로 프레임 뽑기)

**이유:** (실측 근거)

## M9.5 에 반영할 사항

- `AnimatedStickerRenderer._bgr_to_bgra_with_chroma_key` 의 chroma threshold 값: ...
- 비디오 프레임 루프 범위: 0 ~ N-1
```

(실측 값으로 채워서 커밋)

- [ ] **Step 5: Commit**

```bash
cd /home/ingon/AR_book/stickerbook
git add docs/M9_1_OUTPUT_FORMAT.md
git commit -m "docs(m9): record AnimatedDrawings output format verification"
```

---

## Task 2: M9.2 — TorchServe 네이티브 설치 검증

**목적:** Docker 없이 `java + torchserve` 로 M7.5 결과를 재현할 수 있는지 확인. 이후 코드가 의존할 수 있는 환경을 만든다.

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Java 설치 확인**

Run:
```bash
java -version 2>&1
```
Expected: openjdk 11 또는 17 이상. 없으면:
```bash
sudo apt install openjdk-17-jre-headless
java -version 2>&1
```
Expected after install: `openjdk version "17.x.x"`.

- [ ] **Step 2: TorchServe + archiver pip 설치**

현재 venv/conda 환경 확인 (stickerbook 용):
```bash
which python && python --version
```

설치:
```bash
pip install torchserve torch-model-archiver
which torchserve
torchserve --version
```
Expected: `TorchServe Version is X.Y.Z` (≥ 0.9).

- [ ] **Step 3: 기존 `.mar` 모델 재사용 확인**

Run:
```bash
ls -la ~/AR_book/AnimatedDrawings/torchserve/model-store/ 2>/dev/null
```
Expected: `drawn_humanoid_detector.mar`, `drawn_humanoid_pose_estimator.mar` 둘 다 존재.

- [ ] **Step 4: 수동 기동 + health 확인**

Run:
```bash
mkdir -p /tmp/ts_logs
torchserve --start \
  --model-store ~/AR_book/AnimatedDrawings/torchserve/model-store \
  --ts-config /tmp/ts_config.properties \
  --models drawn_humanoid_detector.mar drawn_humanoid_pose_estimator.mar \
  --disable-token-auth \
  --no-config-snapshots \
  --foreground 2>&1 | tee /tmp/ts_logs/stdout.log &
TS_PID=$!
sleep 10
curl -s -4 http://127.0.0.1:8080/ping
```
Expected: `{"status": "Healthy"}`.

Cleanup:
```bash
torchserve --stop
```

- [ ] **Step 5: requirements.txt 업데이트**

Modify `/home/ingon/AR_book/stickerbook/requirements.txt` — add:
```
# Animation (M9): runs AnimatedDrawings via TorchServe natively.
# System deps: openjdk-17-jre-headless (apt).
# Galaxy 포팅 시 전부 제거 예정 — onnxruntime-mobile 로 대체.
torchserve>=0.9
torch-model-archiver>=0.9
```

- [ ] **Step 6: Commit**

```bash
cd /home/ingon/AR_book/stickerbook
git add requirements.txt
git commit -m "chore(m9): add torchserve native deps for AD integration"
```

---

## Task 3: `animate/torchserve_runtime.py` — 데이터 타입 + 환경 체크

**Files:**
- Create: `animate/__init__.py`
- Create: `animate/torchserve_runtime.py`
- Create: `tests/test_torchserve_runtime.py`

- [ ] **Step 1: 빈 패키지 생성**

Create `/home/ingon/AR_book/stickerbook/animate/__init__.py` (empty file).

- [ ] **Step 2: Write failing test for environment check helper**

Create `/home/ingon/AR_book/stickerbook/tests/test_torchserve_runtime.py`:
```python
import subprocess
from unittest.mock import patch

import pytest

from animate.torchserve_runtime import (
    EnvironmentCheckResult,
    check_environment,
)


def test_check_environment_reports_ok_when_java_and_torchserve_present() -> None:
    def fake_which(name: str) -> str | None:
        return {"java": "/usr/bin/java", "torchserve": "/usr/local/bin/torchserve"}.get(name)

    with patch("animate.torchserve_runtime.shutil.which", side_effect=fake_which):
        result = check_environment()

    assert isinstance(result, EnvironmentCheckResult)
    assert result.ok is True
    assert result.missing == []


def test_check_environment_reports_missing_java() -> None:
    def fake_which(name: str) -> str | None:
        return {"torchserve": "/usr/local/bin/torchserve"}.get(name)

    with patch("animate.torchserve_runtime.shutil.which", side_effect=fake_which):
        result = check_environment()

    assert result.ok is False
    assert "java" in result.missing
    # 설치 가이드 메시지에 apt install 언급
    assert "apt" in result.install_hint.lower()


def test_check_environment_reports_missing_torchserve() -> None:
    def fake_which(name: str) -> str | None:
        return {"java": "/usr/bin/java"}.get(name)

    with patch("animate.torchserve_runtime.shutil.which", side_effect=fake_which):
        result = check_environment()

    assert result.ok is False
    assert "torchserve" in result.missing
    assert "pip install torchserve" in result.install_hint
```

- [ ] **Step 3: Run test to verify it fails**

Run:
```bash
cd /home/ingon/AR_book/stickerbook
pytest tests/test_torchserve_runtime.py -v
```
Expected: `ModuleNotFoundError: No module named 'animate.torchserve_runtime'`.

- [ ] **Step 4: Write minimal implementation**

Create `/home/ingon/AR_book/stickerbook/animate/torchserve_runtime.py`:
```python
"""TorchServe native lifecycle for AnimatedDrawings inference."""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import List


@dataclass
class EnvironmentCheckResult:
    ok: bool
    missing: List[str] = field(default_factory=list)
    install_hint: str = ""


def check_environment() -> EnvironmentCheckResult:
    missing: List[str] = []
    if shutil.which("java") is None:
        missing.append("java")
    if shutil.which("torchserve") is None:
        missing.append("torchserve")

    hints: List[str] = []
    if "java" in missing:
        hints.append("sudo apt install openjdk-17-jre-headless")
    if "torchserve" in missing:
        hints.append("pip install torchserve torch-model-archiver")

    return EnvironmentCheckResult(
        ok=not missing,
        missing=missing,
        install_hint=" && ".join(hints),
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
pytest tests/test_torchserve_runtime.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add animate/__init__.py animate/torchserve_runtime.py tests/test_torchserve_runtime.py
git commit -m "feat(m9): add torchserve env check (Task 3)"
```

---

## Task 4: `TorchServeRuntime` — 기동/정지 + health check

**Files:**
- Modify: `animate/torchserve_runtime.py` (add class)
- Modify: `tests/test_torchserve_runtime.py` (add tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_torchserve_runtime.py`:
```python
from pathlib import Path
from unittest.mock import MagicMock, patch

from animate.torchserve_runtime import TorchServeRuntime, TorchServeNotReady


def test_runtime_start_spawns_subprocess_and_polls_health(tmp_path: Path) -> None:
    model_store = tmp_path / "model-store"
    model_store.mkdir()
    config_path = tmp_path / "ts_config.properties"
    config_path.write_text("default_workers_per_model=1\n")

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # still running

    responses = [ConnectionError(), ConnectionError(), _ok_response()]

    with patch("animate.torchserve_runtime.subprocess.Popen", return_value=mock_proc) as popen, \
         patch("animate.torchserve_runtime.urlopen", side_effect=responses) as urlopen, \
         patch("animate.torchserve_runtime.time.sleep"):
        rt = TorchServeRuntime(
            model_store=model_store,
            config_path=config_path,
            models=["drawn_humanoid_detector.mar"],
            health_url="http://127.0.0.1:8080/ping",
            poll_interval_sec=0.1,
            health_timeout_sec=5.0,
        )
        rt.start()

    assert popen.called
    args = popen.call_args[0][0]
    assert "torchserve" in args[0]
    assert "--start" in args
    assert "--model-store" in args
    assert str(model_store) in args
    assert urlopen.call_count == 3


def test_runtime_start_raises_if_health_never_ok(tmp_path: Path) -> None:
    model_store = tmp_path / "model-store"
    model_store.mkdir()
    config_path = tmp_path / "ts_config.properties"
    config_path.touch()

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None

    with patch("animate.torchserve_runtime.subprocess.Popen", return_value=mock_proc), \
         patch("animate.torchserve_runtime.urlopen", side_effect=ConnectionError()), \
         patch("animate.torchserve_runtime.time.sleep"):
        rt = TorchServeRuntime(
            model_store=model_store,
            config_path=config_path,
            models=["m.mar"],
            health_url="http://127.0.0.1:8080/ping",
            poll_interval_sec=0.01,
            health_timeout_sec=0.1,
        )
        with pytest.raises(TorchServeNotReady):
            rt.start()


def test_runtime_stop_invokes_torchserve_stop(tmp_path: Path) -> None:
    model_store = tmp_path / "model-store"
    model_store.mkdir()
    config_path = tmp_path / "ts_config.properties"
    config_path.touch()

    mock_proc = MagicMock()

    with patch("animate.torchserve_runtime.subprocess.Popen", return_value=mock_proc), \
         patch("animate.torchserve_runtime.subprocess.run") as run_mock, \
         patch("animate.torchserve_runtime.urlopen", return_value=_ok_response()), \
         patch("animate.torchserve_runtime.time.sleep"):
        rt = TorchServeRuntime(
            model_store=model_store,
            config_path=config_path,
            models=["m.mar"],
            health_url="http://127.0.0.1:8080/ping",
            poll_interval_sec=0.01,
            health_timeout_sec=0.5,
        )
        rt.start()
        rt.stop()

    assert any(
        "--stop" in call.args[0] and "torchserve" in call.args[0][0]
        for call in run_mock.call_args_list
    )


def _ok_response():
    resp = MagicMock()
    resp.read.return_value = b'{"status": "Healthy"}'
    resp.__enter__ = lambda self: resp
    resp.__exit__ = lambda *a: None
    return resp
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_torchserve_runtime.py::test_runtime_start_spawns_subprocess_and_polls_health -v
```
Expected: `ImportError: cannot import name 'TorchServeRuntime'`.

- [ ] **Step 3: Implement `TorchServeRuntime`**

Append to `animate/torchserve_runtime.py`:
```python
import json
import subprocess
import time
from pathlib import Path
from typing import List, Optional
from urllib.error import URLError
from urllib.request import urlopen


class TorchServeNotReady(RuntimeError):
    pass


class TorchServeRuntime:
    def __init__(
        self,
        model_store: Path,
        config_path: Path,
        models: List[str],
        health_url: str = "http://127.0.0.1:8080/ping",
        poll_interval_sec: float = 1.0,
        health_timeout_sec: float = 30.0,
    ) -> None:
        self._model_store = Path(model_store)
        self._config_path = Path(config_path)
        self._models = list(models)
        self._health_url = health_url
        self._poll_interval_sec = poll_interval_sec
        self._health_timeout_sec = health_timeout_sec
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        env = check_environment()
        if not env.ok:
            raise TorchServeNotReady(
                f"missing executables: {env.missing}. install: {env.install_hint}"
            )

        cmd = [
            "torchserve",
            "--start",
            "--model-store", str(self._model_store),
            "--ts-config", str(self._config_path),
            "--models", *self._models,
            "--disable-token-auth",
            "--no-config-snapshots",
        ]
        self._proc = subprocess.Popen(cmd)
        self._wait_for_health()

    def _wait_for_health(self) -> None:
        deadline = time.monotonic() + self._health_timeout_sec
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                with urlopen(self._health_url, timeout=2.0) as resp:
                    body = resp.read().decode()
                    if '"Healthy"' in body or '"status": "Healthy"' in body:
                        return
            except (URLError, ConnectionError, OSError) as e:
                last_err = e
            time.sleep(self._poll_interval_sec)
        raise TorchServeNotReady(
            f"health probe did not pass within {self._health_timeout_sec}s "
            f"(last error: {last_err})"
        )

    def stop(self) -> None:
        subprocess.run(["torchserve", "--stop"], check=False)
        self._proc = None
```

- [ ] **Step 4: Run all tests in the file**

Run:
```bash
pytest tests/test_torchserve_runtime.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add animate/torchserve_runtime.py tests/test_torchserve_runtime.py
git commit -m "feat(m9): add TorchServeRuntime with health-polling lifecycle (Task 4)"
```

---

## Task 5: `animate/animated_drawings_runner.py` — 흰 배경 합성 헬퍼

**Files:**
- Create: `animate/animated_drawings_runner.py`
- Create: `tests/test_animated_drawings_runner.py`

- [ ] **Step 1: Write failing test for composite**

Create `/home/ingon/AR_book/stickerbook/tests/test_animated_drawings_runner.py`:
```python
import numpy as np

from animate.animated_drawings_runner import composite_on_white_bg


def test_composite_places_opaque_pixels_over_white() -> None:
    tex = np.zeros((10, 10, 4), dtype=np.uint8)
    tex[2:5, 2:5, :3] = (0, 0, 255)  # red BGR
    tex[2:5, 2:5, 3] = 255            # opaque

    out = composite_on_white_bg(tex)

    assert out.shape == (10, 10, 3)
    assert out.dtype == np.uint8
    # transparent area is white
    assert tuple(out[0, 0]) == (255, 255, 255)
    # opaque area keeps red
    assert tuple(out[3, 3]) == (0, 0, 255)


def test_composite_blends_semitransparent_pixels_with_white() -> None:
    tex = np.zeros((4, 4, 4), dtype=np.uint8)
    tex[1, 1, :3] = (0, 0, 255)
    tex[1, 1, 3] = 128  # ~50%

    out = composite_on_white_bg(tex)

    px = out[1, 1]
    assert 120 <= int(px[2]) <= 135   # red ~halved blended with white
    assert 120 <= int(px[1]) <= 135   # green: white pulls it up
    assert 120 <= int(px[0]) <= 135


def test_composite_accepts_bgra_float_and_returns_uint8() -> None:
    tex = np.zeros((4, 4, 4), dtype=np.float32)
    tex[..., 3] = 255.0
    out = composite_on_white_bg(tex)
    assert out.dtype == np.uint8
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_animated_drawings_runner.py::test_composite_places_opaque_pixels_over_white -v
```
Expected: `ModuleNotFoundError: No module named 'animate.animated_drawings_runner'`.

- [ ] **Step 3: Implement `composite_on_white_bg`**

Create `/home/ingon/AR_book/stickerbook/animate/animated_drawings_runner.py`:
```python
"""Runs AnimatedDrawings image_to_animation.py and parses the result."""
from __future__ import annotations

import numpy as np


def composite_on_white_bg(texture_bgra: np.ndarray) -> np.ndarray:
    """Alpha-composite an RGBA/BGRA sticker over solid white, returning BGR uint8.

    AnimatedDrawings expects a full-opaque image. Transparent pixels become white.
    """
    tex = np.asarray(texture_bgra, dtype=np.float32)
    if tex.shape[-1] != 4:
        raise ValueError("texture_bgra must have 4 channels (BGRA)")
    rgb = tex[..., :3]
    alpha = tex[..., 3:4] / 255.0
    white = np.full_like(rgb, 255.0)
    out = rgb * alpha + white * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_animated_drawings_runner.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add animate/animated_drawings_runner.py tests/test_animated_drawings_runner.py
git commit -m "feat(m9): composite BGRA sticker onto white bg for AD input (Task 5)"
```

---

## Task 6: `AnimationResult` + `run_animated_drawings` subprocess wrapper

**Files:**
- Modify: `animate/animated_drawings_runner.py` (add dataclass + function)
- Modify: `tests/test_animated_drawings_runner.py` (add tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_animated_drawings_runner.py`:
```python
import subprocess
from pathlib import Path
from unittest.mock import patch

import cv2

from animate.animated_drawings_runner import (
    AnimationResult,
    run_animated_drawings,
)


def test_run_returns_success_when_video_and_cfg_produced(tmp_path: Path) -> None:
    tex = np.zeros((32, 32, 4), dtype=np.uint8)
    tex[..., :3] = 200
    tex[..., 3] = 255
    ad_repo = tmp_path / "ad"
    ad_repo.mkdir()
    (ad_repo / "examples").mkdir()
    (ad_repo / "examples" / "image_to_animation.py").write_text("# placeholder")
    work_dir = tmp_path / "work"

    def fake_run(cmd, *args, **kwargs):
        # Simulate AD producing artifacts inside out dir
        out_dir = Path(cmd[cmd.index("--output") + 1]) if "--output" in cmd else Path(cmd[-1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "video.gif").write_bytes(b"fake-gif")
        (out_dir / "char_cfg.yaml").write_text("skeleton: []\n")
        result = subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")
        return result

    with patch("animate.animated_drawings_runner.subprocess.run", side_effect=fake_run):
        result = run_animated_drawings(
            texture_bgra=tex,
            motion="dab",
            ad_repo_path=ad_repo,
            work_dir=work_dir,
            timeout_sec=5.0,
        )

    assert isinstance(result, AnimationResult)
    assert result.success is True
    assert result.video_path is not None
    assert result.video_path.exists()
    assert result.char_cfg_path is not None
    assert result.error is None
    assert result.duration_sec > 0


def test_run_returns_failure_when_subprocess_nonzero_exit(tmp_path: Path) -> None:
    tex = np.zeros((32, 32, 4), dtype=np.uint8)
    tex[..., 3] = 255
    ad_repo = tmp_path / "ad"
    (ad_repo / "examples").mkdir(parents=True)
    (ad_repo / "examples" / "image_to_animation.py").write_text("#")
    work_dir = tmp_path / "work"

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=2, stdout="", stderr="boom")

    with patch("animate.animated_drawings_runner.subprocess.run", side_effect=fake_run):
        result = run_animated_drawings(
            texture_bgra=tex, motion="dab", ad_repo_path=ad_repo,
            work_dir=work_dir, timeout_sec=5.0,
        )

    assert result.success is False
    assert result.video_path is None
    assert "boom" in (result.error or "") or "exit" in (result.error or "").lower()


def test_run_returns_failure_on_timeout(tmp_path: Path) -> None:
    tex = np.zeros((32, 32, 4), dtype=np.uint8)
    tex[..., 3] = 255
    ad_repo = tmp_path / "ad"
    (ad_repo / "examples").mkdir(parents=True)
    (ad_repo / "examples" / "image_to_animation.py").write_text("#")
    work_dir = tmp_path / "work"

    def raise_timeout(cmd, *args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 1))

    with patch("animate.animated_drawings_runner.subprocess.run", side_effect=raise_timeout):
        result = run_animated_drawings(
            texture_bgra=tex, motion="dab", ad_repo_path=ad_repo,
            work_dir=work_dir, timeout_sec=0.1,
        )

    assert result.success is False
    assert "timeout" in (result.error or "").lower()


def test_run_writes_input_png_and_passes_path_to_subprocess(tmp_path: Path) -> None:
    tex = np.zeros((32, 32, 4), dtype=np.uint8)
    tex[..., :3] = 100
    tex[..., 3] = 255
    ad_repo = tmp_path / "ad"
    (ad_repo / "examples").mkdir(parents=True)
    (ad_repo / "examples" / "image_to_animation.py").write_text("#")
    work_dir = tmp_path / "work"

    captured_cmd: list = []

    def fake_run(cmd, *args, **kwargs):
        captured_cmd.extend(cmd)
        out_dir = Path(cmd[-1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "video.gif").write_bytes(b"g")
        (out_dir / "char_cfg.yaml").write_text("skeleton: []\n")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    with patch("animate.animated_drawings_runner.subprocess.run", side_effect=fake_run):
        run_animated_drawings(
            texture_bgra=tex, motion="dab", ad_repo_path=ad_repo,
            work_dir=work_dir, timeout_sec=5.0,
        )

    input_png_path = work_dir / "input.png"
    assert input_png_path.exists()
    assert str(input_png_path) in captured_cmd
    loaded = cv2.imread(str(input_png_path))
    assert loaded.shape == (32, 32, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_animated_drawings_runner.py -v -k "test_run_"
```
Expected: ImportErrors for `AnimationResult`, `run_animated_drawings`.

- [ ] **Step 3: Implement dataclass + function**

Append to `animate/animated_drawings_runner.py`:
```python
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2


@dataclass
class AnimationResult:
    success: bool
    video_path: Optional[Path]
    char_cfg_path: Optional[Path]
    duration_sec: float
    error: Optional[str]


def run_animated_drawings(
    texture_bgra: np.ndarray,
    motion: str,
    ad_repo_path: Path,
    work_dir: Path,
    timeout_sec: float = 30.0,
) -> AnimationResult:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    input_png = work_dir / "input.png"
    rgb = composite_on_white_bg(texture_bgra)
    cv2.imwrite(str(input_png), rgb)

    script = Path(ad_repo_path) / "examples" / "image_to_animation.py"
    out_dir = work_dir / "out"

    cmd = ["python", str(script), str(input_png), str(out_dir)]
    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        return AnimationResult(
            success=False, video_path=None, char_cfg_path=None,
            duration_sec=time.monotonic() - start,
            error=f"timeout after {timeout_sec}s",
        )
    duration = time.monotonic() - start

    if result.returncode != 0:
        return AnimationResult(
            success=False, video_path=None, char_cfg_path=None,
            duration_sec=duration,
            error=f"exit {result.returncode}: {result.stderr[-500:]}",
        )

    video_gif = out_dir / "video.gif"
    video_mp4 = out_dir / "video.mp4"
    video = video_gif if video_gif.exists() else (video_mp4 if video_mp4.exists() else None)
    cfg = out_dir / "char_cfg.yaml"

    if video is None:
        return AnimationResult(
            success=False, video_path=None, char_cfg_path=None,
            duration_sec=duration,
            error="no video artifact produced",
        )

    return AnimationResult(
        success=True,
        video_path=video,
        char_cfg_path=cfg if cfg.exists() else None,
        duration_sec=duration,
        error=None,
    )
```

Note: `motion` parameter is accepted for future extension (zombie/wave) but not yet passed to AD. For Stage 1 the default motion baked into `image_to_animation.py` (dab) is used. A future task can wire `--motion` if AD CLI supports it.

- [ ] **Step 4: Run tests**

Run:
```bash
pytest tests/test_animated_drawings_runner.py -v
```
Expected: all passed (previous composite tests + 4 new).

- [ ] **Step 5: Commit**

```bash
git add animate/animated_drawings_runner.py tests/test_animated_drawings_runner.py
git commit -m "feat(m9): run AnimatedDrawings via subprocess with result parsing (Task 6)"
```

---

## Task 7: Joint spread sanity check (R13 대응)

**목적:** AD 는 joint 가 중앙에 뭉쳐도 success 를 반환함 (M7.5 여자아이 케이스). success 판정 전에 joint spread 를 검사해 실패로 강등.

**Files:**
- Modify: `animate/animated_drawings_runner.py`
- Modify: `tests/test_animated_drawings_runner.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_animated_drawings_runner.py`:
```python
import yaml

from animate.animated_drawings_runner import (
    joint_spread_ratio,
    JointSpreadError,
)


def _write_char_cfg(path: Path, joints: list) -> None:
    path.write_text(yaml.safe_dump({"skeleton": joints}))


def test_joint_spread_ratio_large_for_spread_skeleton(tmp_path: Path) -> None:
    cfg = tmp_path / "char_cfg.yaml"
    _write_char_cfg(cfg, [
        {"name": "root", "loc": [50, 50]},
        {"name": "left_hand", "loc": [10, 10]},
        {"name": "right_hand", "loc": [90, 90]},
    ])
    assert joint_spread_ratio(cfg, image_size=(100, 100)) > 0.5


def test_joint_spread_ratio_small_for_bunched_skeleton(tmp_path: Path) -> None:
    cfg = tmp_path / "char_cfg.yaml"
    _write_char_cfg(cfg, [
        {"name": "root", "loc": [50, 50]},
        {"name": "left_hand", "loc": [51, 50]},
        {"name": "right_hand", "loc": [50, 51]},
    ])
    assert joint_spread_ratio(cfg, image_size=(100, 100)) < 0.1


def test_run_downgrades_success_to_failure_when_joints_bunched(tmp_path: Path) -> None:
    tex = np.zeros((32, 32, 4), dtype=np.uint8)
    tex[..., 3] = 255
    ad_repo = tmp_path / "ad"
    (ad_repo / "examples").mkdir(parents=True)
    (ad_repo / "examples" / "image_to_animation.py").write_text("#")
    work_dir = tmp_path / "work"

    def fake_run(cmd, *args, **kwargs):
        out_dir = Path(cmd[-1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "video.gif").write_bytes(b"g")
        # bunched joints
        _write_char_cfg(out_dir / "char_cfg.yaml", [
            {"name": "root", "loc": [50, 50]},
            {"name": "left_hand", "loc": [50, 50]},
            {"name": "right_hand", "loc": [50, 50]},
        ])
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    with patch("animate.animated_drawings_runner.subprocess.run", side_effect=fake_run):
        result = run_animated_drawings(
            texture_bgra=tex, motion="dab", ad_repo_path=ad_repo,
            work_dir=work_dir, timeout_sec=5.0,
        )

    assert result.success is False
    assert "bunched" in (result.error or "").lower() or "spread" in (result.error or "").lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_animated_drawings_runner.py -v -k "joint_spread or bunched"
```
Expected: ImportError for `joint_spread_ratio`, `JointSpreadError`.

- [ ] **Step 3: Implement sanity check**

Add to `animate/animated_drawings_runner.py`:
```python
from typing import Tuple

import yaml


class JointSpreadError(RuntimeError):
    pass


MIN_JOINT_SPREAD_RATIO = 0.15  # 이미지 대각선 대비 joint 범위


def joint_spread_ratio(char_cfg_path: Path, image_size: Tuple[int, int]) -> float:
    data = yaml.safe_load(Path(char_cfg_path).read_text())
    joints = data.get("skeleton", [])
    locs = [j.get("loc") for j in joints if "loc" in j]
    if not locs:
        return 0.0
    xs = [p[0] for p in locs]
    ys = [p[1] for p in locs]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    w, h = image_size
    diag = (w * w + h * h) ** 0.5
    return ((dx * dx + dy * dy) ** 0.5) / max(diag, 1.0)
```

Then modify the success branch in `run_animated_drawings` (just before `return AnimationResult(success=True, ...)`):
```python
    # Joint spread sanity check (R13)
    h, w = texture_bgra.shape[:2]
    spread = joint_spread_ratio(cfg, image_size=(w, h)) if cfg.exists() else 0.0
    if spread < MIN_JOINT_SPREAD_RATIO:
        return AnimationResult(
            success=False, video_path=None, char_cfg_path=cfg if cfg.exists() else None,
            duration_sec=duration,
            error=f"joint spread {spread:.3f} below threshold {MIN_JOINT_SPREAD_RATIO} (bunched skeleton)",
        )
```

- [ ] **Step 4: Run tests**

Run:
```bash
pytest tests/test_animated_drawings_runner.py -v
```
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add animate/animated_drawings_runner.py tests/test_animated_drawings_runner.py
git commit -m "feat(m9): reject bunched-joint AD outputs (R13) (Task 7)"
```

---

## Task 8: `AnimationWorker` — single-worker queue

**Files:**
- Create: `animate/animation_worker.py`
- Create: `tests/test_animation_worker.py`

- [ ] **Step 1: Write failing tests**

Create `/home/ingon/AR_book/stickerbook/tests/test_animation_worker.py`:
```python
import time
from pathlib import Path

import numpy as np

from animate.animated_drawings_runner import AnimationResult
from animate.animation_worker import AnimationWorker


def _stub_runner_success(tex, motion, ad_repo_path, work_dir, timeout_sec):
    return AnimationResult(
        success=True, video_path=Path("/dev/null"),
        char_cfg_path=None, duration_sec=0.01, error=None,
    )


def _stub_runner_raises(tex, motion, ad_repo_path, work_dir, timeout_sec):
    raise RuntimeError("boom")


def test_submit_returns_future_resolving_to_animation_result() -> None:
    worker = AnimationWorker(
        runner=_stub_runner_success,
        ad_repo_path=Path("/tmp"),
        work_dir_base=Path("/tmp/wdir"),
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

    def slow_runner(tex, motion, ad_repo_path, work_dir, timeout_sec):
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

    def flaky(tex, motion, ad_repo_path, work_dir, timeout_sec):
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_animation_worker.py -v
```
Expected: `ModuleNotFoundError: No module named 'animate.animation_worker'`.

- [ ] **Step 3: Implement `AnimationWorker`**

Create `/home/ingon/AR_book/stickerbook/animate/animation_worker.py`:
```python
"""Single-worker queue for AnimatedDrawings jobs."""
from __future__ import annotations

import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from animate.animated_drawings_runner import AnimationResult


RunnerFn = Callable[[np.ndarray, str, Path, Path, float], AnimationResult]


class AnimationWorker:
    def __init__(
        self,
        runner: RunnerFn,
        ad_repo_path: Path,
        work_dir_base: Path,
        motion: str = "dab",
        timeout_sec: float = 30.0,
    ) -> None:
        self._runner = runner
        self._ad_repo_path = Path(ad_repo_path)
        self._work_dir_base = Path(work_dir_base)
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
            self._timeout_sec,
        )

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)
```

- [ ] **Step 4: Run tests**

Run:
```bash
pytest tests/test_animation_worker.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add animate/animation_worker.py tests/test_animation_worker.py
git commit -m "feat(m9): single-worker animation queue (Task 8)"
```

---

## Task 9: Sticker animation state 확장

**목적:** `AnchoredSticker` 에 애니메이션 상태 필드를 추가하여 렌더 분기 기반을 마련.

**Files:**
- Modify: `app.py` (AnchoredSticker 클래스)
- Create: `tests/test_anchored_sticker_animation.py`

- [ ] **Step 1: Write failing test for AnimationState enum and extended AnchoredSticker**

Create `/home/ingon/AR_book/stickerbook/tests/test_anchored_sticker_animation.py`:
```python
from pathlib import Path

import numpy as np

from app import AnchoredSticker, AnimationState
from extract.segmenter import StickerAsset
from track.homography_anchor import HomographyAnchor


def _dummy_asset() -> StickerAsset:
    tex = np.zeros((10, 10, 4), dtype=np.uint8)
    tex[..., 3] = 255
    mask = np.full((10, 10), 255, dtype=np.uint8)
    return StickerAsset(texture_bgra=tex, mask_u8=mask, source_region=(0, 0, 10, 10))


def test_anchored_sticker_defaults_to_static_animation_state() -> None:
    item = AnchoredSticker(sticker=_dummy_asset(), anchor=HomographyAnchor())
    assert item.animation_state is AnimationState.STATIC
    assert item.animation_video_path is None
    assert item.animation_started_at is None


def test_animation_state_enum_has_expected_members() -> None:
    names = {s.name for s in AnimationState}
    assert names == {"STATIC", "PREPARING", "ANIMATED", "FAILED"}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_anchored_sticker_animation.py -v
```
Expected: `ImportError: cannot import name 'AnimationState' from 'app'`.

- [ ] **Step 3: Add enum + fields to `app.py`**

In `/home/ingon/AR_book/stickerbook/app.py`, add imports near top:
```python
from typing import Optional
```
(already present). Then add `AnimationState` enum just after `AppAction`:

```python
class AnimationState(Enum):
    STATIC = auto()
    PREPARING = auto()
    ANIMATED = auto()
    FAILED = auto()
```

Modify `AnchoredSticker`:
```python
@dataclass
class AnchoredSticker:
    sticker: StickerAsset
    anchor: HomographyAnchor
    animation_state: AnimationState = AnimationState.STATIC
    animation_video_path: Optional[Path] = None
    animation_started_at: Optional[float] = None
    animation_future: Optional["Future"] = None  # set while PREPARING
```

- [ ] **Step 4: Run test**

Run:
```bash
pytest tests/test_anchored_sticker_animation.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_anchored_sticker_animation.py
git commit -m "feat(m9): AnimationState enum + AnchoredSticker fields (Task 9)"
```

---

## Task 10: `AnimatedStickerRenderer` — 비디오 프레임 루프

**Files:**
- Create: `render/animated_sticker_renderer.py`
- Create: `tests/test_animated_sticker_renderer.py`

- [ ] **Step 1: Write failing tests**

Create `/home/ingon/AR_book/stickerbook/tests/test_animated_sticker_renderer.py`:
```python
from pathlib import Path

import cv2
import numpy as np
import pytest

from render.animated_sticker_renderer import AnimatedStickerRenderer


@pytest.fixture()
def sample_video(tmp_path: Path) -> Path:
    """Write a small 3-frame BGR video for testing."""
    path = tmp_path / "video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (20, 20))
    frames = []
    for i in range(3):
        f = np.zeros((20, 20, 3), dtype=np.uint8)
        f[..., 0] = (i + 1) * 50
        writer.write(f)
        frames.append(f)
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return path


def test_renderer_reads_first_frame_as_bgra(sample_video: Path) -> None:
    r = AnimatedStickerRenderer(video_path=sample_video)
    try:
        bgra = r.next_frame_bgra()
    finally:
        r.release()
    assert bgra.shape == (20, 20, 4)
    assert bgra.dtype == np.uint8


def test_renderer_loops_back_to_zero_after_last_frame(sample_video: Path) -> None:
    r = AnimatedStickerRenderer(video_path=sample_video)
    try:
        seen = [tuple(r.next_frame_bgra()[0, 0, :3]) for _ in range(6)]
    finally:
        r.release()
    # First 3 unique, then wraps: frame 3 should equal frame 0
    assert seen[0] == seen[3]
    assert seen[1] == seen[4]


def test_renderer_applies_chroma_key_to_black_background(sample_video: Path) -> None:
    # Write a video with explicit black bg + non-black fg
    path = sample_video.parent / "ck.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (10, 10))
    f = np.zeros((10, 10, 3), dtype=np.uint8)
    f[3:7, 3:7] = (0, 255, 0)  # green square
    for _ in range(2):
        writer.write(f)
    writer.release()

    r = AnimatedStickerRenderer(video_path=path, chroma_key_threshold=5)
    try:
        bgra = r.next_frame_bgra()
    finally:
        r.release()

    # Background (near black) -> alpha 0
    assert bgra[0, 0, 3] == 0
    # Foreground (green) -> alpha 255
    assert bgra[5, 5, 3] == 255
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_animated_sticker_renderer.py -v
```
Expected: `ModuleNotFoundError: No module named 'render.animated_sticker_renderer'`.

- [ ] **Step 3: Implement renderer**

Create `/home/ingon/AR_book/stickerbook/render/animated_sticker_renderer.py`:
```python
"""Plays back an AnimatedDrawings video as a BGRA frame source for billboard rendering."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class AnimatedStickerRenderer:
    def __init__(
        self,
        video_path: Path,
        chroma_key_threshold: int = 10,
    ) -> None:
        self._video_path = Path(video_path)
        self._cap = cv2.VideoCapture(str(self._video_path))
        if not self._cap.isOpened():
            raise IOError(f"cannot open video: {self._video_path}")
        self._chroma_threshold = int(chroma_key_threshold)

    def next_frame_bgra(self) -> np.ndarray:
        ok, frame_bgr = self._cap.read()
        if not ok or frame_bgr is None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame_bgr = self._cap.read()
            if not ok or frame_bgr is None:
                raise IOError(f"video has no decodable frames: {self._video_path}")
        return self._bgr_to_bgra_chroma(frame_bgr)

    def _bgr_to_bgra_chroma(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        bgra = np.zeros((h, w, 4), dtype=np.uint8)
        bgra[..., :3] = bgr
        # Alpha 0 where all BGR channels below threshold (black chroma key)
        is_bg = np.all(bgr <= self._chroma_threshold, axis=-1)
        bgra[..., 3] = np.where(is_bg, 0, 255).astype(np.uint8)
        return bgra

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
```

Note: If M9.1 concludes that AD outputs have a WHITE background instead of BLACK, invert the chroma logic (`np.all(bgr >= 255 - threshold, axis=-1)`). This is the single known variant risk; swap the condition based on Task 1 findings.

- [ ] **Step 4: Run tests**

Run:
```bash
pytest tests/test_animated_sticker_renderer.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add render/animated_sticker_renderer.py tests/test_animated_sticker_renderer.py
git commit -m "feat(m9): AnimatedStickerRenderer with looping video frame source (Task 10)"
```

---

## Task 11: Billboard render path for animated sticker

**목적:** 기존 `render_sticker_as_billboard` 는 StickerAsset 을 받는다. 동일한 billboard 기하를 사용하되 매 프레임 다른 텍스처를 쓰는 변형을 추가.

**Files:**
- Modify: `render/tilt_renderer.py`
- Modify: `tests/test_tilt_renderer.py`

- [ ] **Step 1: Write failing test**

Append to `/home/ingon/AR_book/stickerbook/tests/test_tilt_renderer.py`:
```python
from render.tilt_renderer import render_bgra_as_billboard


def test_render_bgra_as_billboard_modifies_frame_with_identity_h() -> None:
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    before = frame.copy()
    tex_bgra = np.zeros((80, 80, 4), dtype=np.uint8)
    tex_bgra[..., 2] = 255  # red
    tex_bgra[..., 3] = 255  # opaque

    render_bgra_as_billboard(
        frame=frame,
        texture_bgra=tex_bgra,
        source_region=(250, 200, 80, 80),
        homography=np.eye(3),
        enable_shadow=False,
    )
    assert not np.array_equal(frame, before)


def test_render_bgra_as_billboard_degenerate_homography_no_op() -> None:
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    before = frame.copy()
    tex_bgra = np.zeros((80, 80, 4), dtype=np.uint8)
    tex_bgra[..., 3] = 255

    render_bgra_as_billboard(
        frame=frame, texture_bgra=tex_bgra,
        source_region=(250, 200, 80, 80),
        homography=np.zeros((3, 3)), enable_shadow=False,
    )
    assert np.array_equal(frame, before)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_tilt_renderer.py::test_render_bgra_as_billboard_modifies_frame_with_identity_h -v
```
Expected: `ImportError: cannot import name 'render_bgra_as_billboard'`.

- [ ] **Step 3: Refactor `render_sticker_as_billboard` to delegate**

In `render/tilt_renderer.py`, find `render_sticker_as_billboard`. Extract the core into a new function that takes texture_bgra + source_region directly, and rewrite the StickerAsset variant to call it.

Read current `render_sticker_as_billboard` signature first, then:

```python
def render_bgra_as_billboard(
    frame: np.ndarray,
    texture_bgra: np.ndarray,
    source_region: Tuple[int, int, int, int],
    homography: np.ndarray,
    enable_shadow: bool = True,
    popup_lift_ratio: float = 1.0,
) -> None:
    # (Body copied from existing render_sticker_as_billboard, replacing
    # sticker.texture_bgra / sticker.source_region with the parameters.)
    corners = billboard_corners_2d(
        homography, source_region=source_region, popup_lift_ratio=popup_lift_ratio
    )
    if not np.all(np.isfinite(corners)):
        return
    # ... existing warp + shadow + blend logic ...
```

Then rewrite the original:
```python
def render_sticker_as_billboard(
    frame: np.ndarray,
    sticker: "StickerAsset",
    homography: np.ndarray,
    enable_shadow: bool = True,
    popup_lift_ratio: float = 1.0,
) -> None:
    render_bgra_as_billboard(
        frame=frame,
        texture_bgra=sticker.texture_bgra,
        source_region=sticker.source_region,
        homography=homography,
        enable_shadow=enable_shadow,
        popup_lift_ratio=popup_lift_ratio,
    )
```

- [ ] **Step 4: Run full tilt_renderer test suite**

Run:
```bash
pytest tests/test_tilt_renderer.py -v
```
Expected: all previous tests + 2 new tests pass (13 total if previously 11).

- [ ] **Step 5: Commit**

```bash
git add render/tilt_renderer.py tests/test_tilt_renderer.py
git commit -m "refactor(m9): extract render_bgra_as_billboard from sticker variant (Task 11)"
```

---

## Task 12: `SpinnerOverlay` — PREPARING 시각화

**Files:**
- Create: `render/spinner_overlay.py`
- Create: `tests/test_spinner_overlay.py`

- [ ] **Step 1: Write failing test**

Create `/home/ingon/AR_book/stickerbook/tests/test_spinner_overlay.py`:
```python
import numpy as np

from render.spinner_overlay import draw_spinner


def test_draw_spinner_modifies_frame_in_region() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    before = frame.copy()

    draw_spinner(frame, center=(100, 100), radius=20, phase=0.0)

    assert not np.array_equal(frame, before)
    # Far-away pixels untouched
    assert tuple(frame[0, 0]) == (0, 0, 0)
    assert tuple(frame[199, 199]) == (0, 0, 0)


def test_draw_spinner_different_phase_produces_different_pixels() -> None:
    f1 = np.zeros((200, 200, 3), dtype=np.uint8)
    f2 = np.zeros((200, 200, 3), dtype=np.uint8)
    draw_spinner(f1, center=(100, 100), radius=20, phase=0.0)
    draw_spinner(f2, center=(100, 100), radius=20, phase=1.5)
    assert not np.array_equal(f1, f2)


def test_draw_spinner_does_not_raise_when_center_near_frame_edge() -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    # Should clip instead of raising
    draw_spinner(frame, center=(5, 5), radius=15, phase=0.0)
    draw_spinner(frame, center=(48, 48), radius=15, phase=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_spinner_overlay.py -v
```
Expected: `ModuleNotFoundError: No module named 'render.spinner_overlay'`.

- [ ] **Step 3: Implement spinner**

Create `/home/ingon/AR_book/stickerbook/render/spinner_overlay.py`:
```python
"""Rotating-dots spinner drawn over a sticker region while AD is processing."""
from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np


def draw_spinner(
    frame: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    phase: float,
    num_dots: int = 8,
    dot_radius: int = 3,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    cx, cy = int(center[0]), int(center[1])
    for i in range(num_dots):
        theta = phase + (i * 2.0 * math.pi / num_dots)
        x = int(cx + radius * math.cos(theta))
        y = int(cy + radius * math.sin(theta))
        # Fade per dot by index (gives rotation illusion)
        fade = int(255 * (i + 1) / num_dots)
        c = tuple(int(v * fade / 255) for v in color)
        cv2.circle(frame, (x, y), dot_radius, c, thickness=-1)
```

- [ ] **Step 4: Run tests**

Run:
```bash
pytest tests/test_spinner_overlay.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add render/spinner_overlay.py tests/test_spinner_overlay.py
git commit -m "feat(m9): spinner overlay for PREPARING state (Task 12)"
```

---

## Task 13: App wiring — TorchServe lifecycle, AnimationWorker, state routing

**목적:** 클릭 시 `AnimationWorker.submit`, poll 시 Future 완료 감지하여 상태 전이, 렌더 루프에서 분기.

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`
- Modify: `config.py` (AD_REPO_PATH, ANIMATION_WORK_DIR)

- [ ] **Step 1: Add config paths**

Modify `/home/ingon/AR_book/stickerbook/config.py` — append:
```python
AD_REPO_PATH = Path(os.environ.get(
    "STICKERBOOK_AD_REPO",
    str(Path.home() / "AR_book" / "AnimatedDrawings"),
))
ANIMATION_WORK_DIR = Path(os.environ.get(
    "STICKERBOOK_AD_WORK_DIR",
    "/tmp/stickerbook_ad",
))
TORCHSERVE_CONFIG_PATH = Path(os.environ.get(
    "STICKERBOOK_TS_CONFIG",
    "/tmp/ts_config.properties",
))
TORCHSERVE_MODELS = [
    "drawn_humanoid_detector.mar",
    "drawn_humanoid_pose_estimator.mar",
]
```

Ensure `from pathlib import Path` and `import os` are imported at top.

- [ ] **Step 2: Write failing test for click → PREPARING transition**

Add to `tests/test_app.py` (keep existing tests intact):
```python
from unittest.mock import MagicMock

from animate.animated_drawings_runner import AnimationResult
from app import App, AnimationState


def test_on_click_submits_to_animation_worker_and_sets_preparing(monkeypatch) -> None:
    app = App()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    app._current_frame = frame
    app._segmenter = MagicMock()
    app._executor = MagicMock()
    app._animation_worker = MagicMock()
    fut = MagicMock()
    app._animation_worker.submit.return_value = fut
    # Pretend SAM future resolves immediately via direct short-circuit
    sticker_asset_stub = MagicMock()
    sticker_asset_stub.source_region = (0, 0, 10, 10)
    sticker_asset_stub.texture_bgra = np.zeros((10, 10, 4), dtype=np.uint8)

    # Directly invoke the transition helper (bypass SAM async)
    anchored = app._promote_to_live(sticker_asset_stub, frame)

    assert anchored.animation_state is AnimationState.PREPARING
    app._animation_worker.submit.assert_called_once()
    assert anchored.animation_future is fut


def test_poll_animation_transitions_to_animated_on_success(tmp_path: Path) -> None:
    app = App()
    app._anchored = []
    mock_future = MagicMock()
    mock_future.done.return_value = True
    vid = tmp_path / "video.mp4"
    vid.write_bytes(b"x")
    mock_future.result.return_value = AnimationResult(
        success=True, video_path=vid, char_cfg_path=None,
        duration_sec=0.1, error=None,
    )
    from app import AnchoredSticker
    from track.homography_anchor import HomographyAnchor
    asset = MagicMock()
    asset.source_region = (0, 0, 10, 10)
    asset.texture_bgra = np.zeros((10, 10, 4), dtype=np.uint8)
    item = AnchoredSticker(
        sticker=asset, anchor=HomographyAnchor(),
        animation_state=AnimationState.PREPARING,
        animation_future=mock_future,
    )
    app._anchored.append(item)

    app._poll_animations()

    assert item.animation_state is AnimationState.ANIMATED
    assert item.animation_video_path == vid


def test_poll_animation_transitions_to_failed_on_error() -> None:
    app = App()
    mock_future = MagicMock()
    mock_future.done.return_value = True
    mock_future.result.return_value = AnimationResult(
        success=False, video_path=None, char_cfg_path=None,
        duration_sec=0.1, error="joint spread low",
    )
    from app import AnchoredSticker
    from track.homography_anchor import HomographyAnchor
    asset = MagicMock()
    asset.source_region = (0, 0, 10, 10)
    asset.texture_bgra = np.zeros((10, 10, 4), dtype=np.uint8)
    item = AnchoredSticker(
        sticker=asset, anchor=HomographyAnchor(),
        animation_state=AnimationState.PREPARING,
        animation_future=mock_future,
    )
    app._anchored = [item]

    app._poll_animations()

    assert item.animation_state is AnimationState.FAILED
    assert item.animation_video_path is None
```

- [ ] **Step 3: Run failing tests**

Run:
```bash
pytest tests/test_app.py::test_on_click_submits_to_animation_worker_and_sets_preparing -v
```
Expected: AttributeError / ImportError for `_animation_worker`, `_promote_to_live`, `_poll_animations`.

- [ ] **Step 4: Wire up `App`**

In `/home/ingon/AR_book/stickerbook/app.py`:

a) Add imports near top:
```python
from animate.animated_drawings_runner import AnimationResult, run_animated_drawings
from animate.animation_worker import AnimationWorker
from animate.torchserve_runtime import TorchServeRuntime
from config import (
    AD_REPO_PATH, ANIMATION_WORK_DIR,
    TORCHSERVE_CONFIG_PATH, TORCHSERVE_MODELS,
)
from render.animated_sticker_renderer import AnimatedStickerRenderer
from render.spinner_overlay import draw_spinner
from render.tilt_renderer import render_bgra_as_billboard
```

b) In `App.__init__` append:
```python
        self._torchserve: Optional[TorchServeRuntime] = None
        self._animation_worker: Optional[AnimationWorker] = None
        self._animated_renderers: Dict[int, AnimatedStickerRenderer] = {}  # keyed by id(sticker)
        self._spinner_phase: float = 0.0
```

c) Add helper methods (before `run`):
```python
    def _promote_to_live(
        self, sticker_asset: StickerAsset, ref_frame: np.ndarray
    ) -> AnchoredSticker:
        anchor = HomographyAnchor()
        anchor.initialize(ref_frame, sticker_asset.source_region)
        item = AnchoredSticker(sticker=sticker_asset, anchor=anchor)
        if self._animation_worker is not None:
            item.animation_future = self._animation_worker.submit(sticker_asset.texture_bgra)
            item.animation_state = AnimationState.PREPARING
            item.animation_started_at = perf_counter()
        self._anchored.append(item)
        return item

    def _poll_animations(self) -> None:
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
                print(
                    f"[app] sticker {id(item)} animated "
                    f"({result.duration_sec:.1f}s)"
                )
            else:
                item.animation_state = AnimationState.FAILED
                print(f"[app] animation failed: {result.error}")
            item.animation_future = None
```

d) Modify `_poll_pending` (SAM completion branch) to use `_promote_to_live`:
Replace the block that creates `AnchoredSticker` + appends (lines around current 144–157) with:
```python
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
                        item = self._promote_to_live(sticker, ref_frame)
                        x, y, w, h = sticker.source_region
                        print(
                            f"[app] sticker #{len(self._anchored)} created + anchored "
                            f"at ({x},{y}) size {w}x{h}, animation_state={item.animation_state.name}"
                        )
```
**Note:** `_promote_to_live` already calls `HomographyAnchor().initialize(...)` — deduplicate by having `_poll_pending` do the lost-check first (with a local `_probe_anchor` helper) or passing the pre-initialized anchor into `_promote_to_live`. Simplest: change `_promote_to_live` signature to accept an already-initialized anchor:
```python
    def _promote_to_live(
        self, sticker_asset: StickerAsset, anchor: HomographyAnchor
    ) -> AnchoredSticker:
        item = AnchoredSticker(sticker=sticker_asset, anchor=anchor)
        if self._animation_worker is not None:
            item.animation_future = self._animation_worker.submit(sticker_asset.texture_bgra)
            item.animation_state = AnimationState.PREPARING
            item.animation_started_at = perf_counter()
        self._anchored.append(item)
        return item
```
And the call site becomes:
```python
                    else:
                        item = self._promote_to_live(sticker, anchor)
```

Update the test in Step 2 to reflect this signature (pass anchor explicitly, not ref_frame).

e) Modify `run` to instantiate TorchServeRuntime + AnimationWorker, call them in try/finally:
```python
        # after segmenter/executor setup
        self._torchserve = TorchServeRuntime(
            model_store=AD_REPO_PATH / "torchserve" / "model-store",
            config_path=TORCHSERVE_CONFIG_PATH,
            models=TORCHSERVE_MODELS,
        )
        try:
            self._torchserve.start()
            self._animation_worker = AnimationWorker(
                runner=run_animated_drawings,
                ad_repo_path=AD_REPO_PATH,
                work_dir_base=ANIMATION_WORK_DIR,
            )
        except Exception as e:
            print(f"[app] WARNING: animation unavailable ({e}); stickers will remain STATIC")
            self._torchserve = None
            self._animation_worker = None
```

f) In main loop after `_poll_pending()`, call `self._poll_animations()`.

g) In render loop (where billboards are drawn), replace `render_sticker_as_billboard(display, item.sticker, state.homography)` with:
```python
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
                            )
                        else:
                            render_sticker_as_billboard(display, item.sticker, state.homography)
                    else:
                        render_sticker_as_billboard(display, item.sticker, state.homography)
                        if item.animation_state is AnimationState.PREPARING:
                            # overlay spinner at sticker bbox center
                            x, y, w, h = item.sticker.source_region
                            cx, cy = x + w // 2, y + h // 2
                            self._spinner_phase += 0.1
                            draw_spinner(display, (cx, cy), min(w, h) // 4, self._spinner_phase)
```

h) In `finally`, append cleanup:
```python
            for r in self._animated_renderers.values():
                r.release()
            if self._animation_worker is not None:
                self._animation_worker.shutdown(wait=False)
            if self._torchserve is not None:
                self._torchserve.stop()
```

- [ ] **Step 5: Run the new tests**

Run:
```bash
pytest tests/test_app.py -v -k "animation or preparing or animated or failed"
```
Expected: 3 new tests pass. (Adjust test `_promote_to_live` call to pass anchor as per the signature revision.)

- [ ] **Step 6: Run full test suite**

Run:
```bash
pytest -v
```
Expected: all passing. If any prior test_app tests fail due to `_promote_to_live` signature change, update them in the same commit.

- [ ] **Step 7: Commit**

```bash
git add app.py config.py tests/test_app.py
git commit -m "feat(m9): wire TorchServe + AnimationWorker + renderer routing into App (Task 13)"
```

---

## Task 14: Perf tracker + README + DESIGN status update

**Files:**
- Modify: `app.py` (perf records)
- Modify: `README.md`
- Modify: `docs/DESIGN.md` (milestone status)

- [ ] **Step 1: Add perf recording for animation**

In `app.py` `_poll_animations` where success is determined, after `item.animation_state = AnimationState.ANIMATED`, record:
```python
        perf.record("animation_success_sec", result.duration_sec)
```
For this, `_poll_animations` needs access to the running-loop `perf` — pass it as a parameter:
```python
def _poll_animations(self, perf: "_PerfTracker") -> None:
    ...
    perf.record("animation_success_sec", result.duration_sec)  # on success
    perf.record("animation_failure_sec", result.duration_sec)  # on failure
```

And update call site in `run()`:
```python
                self._poll_animations(perf)
```

Also extend `_PerfTracker.report()` only if needed — current impl iterates self._samples, so new keys appear automatically.

Add `animation_success_sec`, `animation_failure_sec` to the `order` list in `_PerfTracker.report`:
```python
        order = [
            "capture", "poll", "detect", "track_render", "iter",
            "animation_success_sec", "animation_failure_sec",
        ]
```

- [ ] **Step 2: Write failing test for perf reporting**

Add to `tests/test_app.py`:
```python
def test_perf_report_includes_animation_metrics_when_present() -> None:
    from app import _PerfTracker
    pt = _PerfTracker()
    pt.record("animation_success_sec", 8.5)
    pt.record("animation_failure_sec", 3.2)
    report = pt.report()
    assert "animation_success_sec" in report
    assert "animation_failure_sec" in report
```

- [ ] **Step 3: Run test**

Run:
```bash
pytest tests/test_app.py::test_perf_report_includes_animation_metrics_when_present -v
```
Expected: pass (since `_samples` keys auto-appear).

- [ ] **Step 4: Update README.md**

Read current README first:
```bash
cat /home/ingon/AR_book/stickerbook/README.md | head -80
```

Then:
- Remove any Docker / `docker run` / `docker_torchserve` mentions (if present from earlier M7.5 notes).
- Add a **"Setup — AnimatedDrawings (M9)"** section:
  ```markdown
  ### Setup — AnimatedDrawings animation (optional)

  M9 adds dancing stickers via AnimatedDrawings. Requires TorchServe natively (no Docker).

  ```bash
  sudo apt install openjdk-17-jre-headless
  pip install torchserve torch-model-archiver
  # AD repo must be present:
  ls ~/AR_book/AnimatedDrawings/torchserve/model-store/*.mar
  # stickerbook will spawn torchserve on startup and shut it down on exit.
  ```

  Config: override defaults via env vars
  - `STICKERBOOK_AD_REPO=/path/to/AnimatedDrawings`
  - `STICKERBOOK_AD_WORK_DIR=/tmp/stickerbook_ad`
  - `STICKERBOOK_TS_CONFIG=/tmp/ts_config.properties`

  If setup is incomplete, stickerbook logs a warning and runs without animation (stickers remain static — M8.1 behavior).
  ```

- Add one sentence to the Status section:
  ```
  - M9: live AnimatedDrawings integration (dancing stickers).
  ```

- [ ] **Step 5: Update DESIGN.md status**

In `/home/ingon/AR_book/stickerbook/docs/DESIGN.md`, mark M9 sub-milestones as they complete:
```
| M9.1 | AD 출력 포맷 검증 | ... | ✅ |
| M9.2 | TorchServeRuntime + 네이티브 설치 | ... | ✅ |
...
```
(Each sub-task marks its own row in its own commit — if multiple sub-milestones were completed together, update them all here.)

- [ ] **Step 6: Run full test suite one more time**

Run:
```bash
pytest -v
```
Expected: all passing.

- [ ] **Step 7: Commit**

```bash
git add app.py tests/test_app.py README.md docs/DESIGN.md
git commit -m "docs(m9): perf metrics + README native-setup guide + M9 status"
```

---

## Task 15: E2E manual verification

**목적:** 실제 웹캠 + 실제 AD 로 종단간 동작 확인. 자동화된 테스트로는 잡히지 않는 통합 문제를 눈으로 잡는다.

**Files:** (no file changes)

- [ ] **Step 1: Pre-flight checks**

Run:
```bash
cd /home/ingon/AR_book/stickerbook
java -version 2>&1 | head -1
which torchserve
ls ~/AR_book/AnimatedDrawings/torchserve/model-store/*.mar
cat /tmp/ts_config.properties 2>/dev/null || echo "default_workers_per_model=1" > /tmp/ts_config.properties
```
Expected: java ≥ 11, torchserve found, both .mar files present, config exists.

- [ ] **Step 2: Run app in integration mode**

Run:
```bash
python main.py --sam-weights models/mobile_sam.pt
```

Observe startup logs:
- `[TorchServeRuntime] health OK` (or similar)
- No errors

- [ ] **Step 3: E2E scenario**

Perform manually:
1. Hold a sketchbook with a child-drawn humanoid figure in front of webcam.
2. Click the drawing.
3. Observe: detection → segmentation (1–2s) → static billboard pops up → spinner appears over the billboard → 7–12 s later → spinner vanishes, sticker starts dancing (dab motion).
4. Move camera left-right. Animated sticker must stay anchored to the paper.
5. Click a second drawing while the first is still PREPARING. Expect: second enters queue (its spinner shows too).
6. After both animate, press `R`. All stickers reset.
7. Click a non-human drawing (e.g. flower). Expect: static billboard, then spinner, then spinner vanishes (FAILED silently). Static sticker remains.
8. Press `Q`. Verify `[perf]` report includes `animation_success_sec` and `animation_failure_sec`.

- [ ] **Step 4: Record findings**

Create or append to `/home/ingon/AR_book/stickerbook/docs/M9_E2E_VERIFICATION.md`:
```markdown
# M9 E2E Verification — <date>

## Pass/Fail per step
1. ...
2. ...

## Observed latencies
- animation_success_sec p50: ...s / p95: ...s
- animation_failure_sec p50: ...s

## Notes / surprises
- ...
```

- [ ] **Step 5: Commit**

```bash
git add docs/M9_E2E_VERIFICATION.md
git commit -m "docs(m9): E2E verification report"
```

---

## Final checklist before declaring M9 done

- [ ] All 49 pre-M9 tests + all new M9 tests pass (`pytest -v`)
- [ ] Docker is no longer required anywhere (grep repo: `grep -r -i docker . --include='*.md' --include='*.py'` should only show historical / comparative mentions)
- [ ] App runs end-to-end on live webcam: click → dancing sticker (human) or static + log (non-human)
- [ ] `docs/DESIGN.md` M9 sub-milestones all ✅
- [ ] `docs/M9_1_OUTPUT_FORMAT.md` and `docs/M9_E2E_VERIFICATION.md` exist with real findings
- [ ] README has Setup section for native TorchServe; no Docker references

Once done, call advisor() or superpowers:requesting-code-review before merging.
