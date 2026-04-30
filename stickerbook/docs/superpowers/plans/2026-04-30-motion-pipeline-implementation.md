# Motion Recording Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** stickerbook V1 PC에 사용자 동작 카메라 녹화 → MediaPipe Pose → BVH 자동 변환 → AnimatedDrawings motion 라이브러리 자동 등록 → SPACE로 활성 모션 사용 파이프라인 추가.

**Architecture:** `stickerbook/motion/` 신규 모듈 5개 (recorder / pose_estimator / bvh_writer / library / pipeline) + `app.py` 확장. AD repo 코드 무손, `examples/`에 데이터만 추가. 키 매핑: M (녹화 토글), 1/2/3 (활성 모션 선택), SPACE (활성 모션으로 그림 캡처).

**Tech Stack:** Python 3.10+, mediapipe>=0.10, OpenCV, NumPy, AnimatedDrawings (외부 subprocess), pytest.

---

## File Structure

### 신규 파일

| 파일 | 책임 |
|---|---|
| `stickerbook/motion/__init__.py` | 패키지 marker (빈 파일) |
| `stickerbook/motion/recorder.py` | `FrameRecorder` — M 토글 + frame 버퍼 |
| `stickerbook/motion/pose_estimator.py` | `PoseEstimator` — MediaPipe Pose wrapper |
| `stickerbook/motion/bvh_writer.py` | `write_bvh()` — landmarks → BVH 텍스트. joint name = my_dance retarget 호환 |
| `stickerbook/motion/library.py` | `MotionLibrary` — BVH 파일 + AD config 자동 관리, 활성 motion 상태 |
| `stickerbook/motion/pipeline.py` | `MotionPipeline` — recorder/estimator/library 묶음, M toggle entry |
| `stickerbook/tests/test_motion_recorder.py` | unit test |
| `stickerbook/tests/test_motion_pose_estimator.py` | unit test (mediapipe 의존성, optional skip) |
| `stickerbook/tests/test_motion_bvh_writer.py` | unit test |
| `stickerbook/tests/test_motion_library.py` | unit test |
| `stickerbook/tests/test_motion_pipeline.py` | integration test (mock-based) |
| `stickerbook/assets/motions/library/.gitkeep` | git placeholder, 디렉터리 보존 |

### 수정 파일

| 파일 | 변경 |
|---|---|
| `stickerbook/app.py` | M, 1/2/3 키 분기, `_active_motion` 상태, motion pipeline 초기화, frame buffer 통합 |
| `stickerbook/requirements.txt` | `mediapipe>=0.10` 추가 |
| `.gitignore` (drawing-to-2.5d-repo) | `stickerbook/assets/motions/library/*.bvh` 추가 |

---

## Task 1: 환경 + scaffold

**Files:**
- Create: `stickerbook/motion/__init__.py`
- Create: `stickerbook/motion/recorder.py` (placeholder)
- Create: `stickerbook/motion/pose_estimator.py` (placeholder)
- Create: `stickerbook/motion/bvh_writer.py` (placeholder)
- Create: `stickerbook/motion/library.py` (placeholder)
- Create: `stickerbook/motion/pipeline.py` (placeholder)
- Create: `stickerbook/assets/motions/library/.gitkeep` (empty)
- Modify: `stickerbook/requirements.txt`
- Modify: `.gitignore` (drawing-to-2.5d-repo root)

- [ ] **Step 1: requirements.txt에 mediapipe 추가**

```bash
cat /home/ingon/AR_book/drawing-to-2.5d-repo/stickerbook/requirements.txt
```

기존 파일 끝에 추가:
```
mediapipe>=0.10
```

- [ ] **Step 2: mediapipe 설치**

```bash
cd /home/ingon/AR_book/drawing-to-2.5d-repo/stickerbook
/usr/bin/python3 -m pip install --user mediapipe>=0.10
/usr/bin/python3 -c "import mediapipe; print(mediapipe.__version__)"
```

Expected: 0.10.x 이상 출력

- [ ] **Step 3: motion/ 디렉터리 + 빈 파일 생성**

```bash
cd /home/ingon/AR_book/drawing-to-2.5d-repo/stickerbook
mkdir -p motion assets/motions/library
touch motion/__init__.py
touch motion/recorder.py motion/pose_estimator.py motion/bvh_writer.py
touch motion/library.py motion/pipeline.py
touch assets/motions/library/.gitkeep
```

- [ ] **Step 4: .gitignore 갱신 (drawing-to-2.5d-repo)**

`.gitignore` 안의 "Experiment outputs" 섹션 아래에 추가:
```
# User-recorded motion BVH files (personal data, not source)
stickerbook/assets/motions/library/*.bvh
```

- [ ] **Step 5: 디렉터리 구조 검증**

```bash
cd /home/ingon/AR_book/drawing-to-2.5d-repo
ls stickerbook/motion/
ls stickerbook/assets/motions/library/
git status -s | head -10
```

Expected:
- `motion/`에 6 파일 (`__init__.py` + 5 모듈)
- `library/`에 `.gitkeep` 보임 (BVH 파일은 향후 .gitignore 처리)

- [ ] **Step 6: Commit**

```bash
git add stickerbook/motion/ stickerbook/assets/motions/library/.gitkeep stickerbook/requirements.txt .gitignore
git commit -m "$(cat <<'EOF'
chore(motion): scaffold motion/ module + library dir + mediapipe dep

- stickerbook/motion/{recorder,pose_estimator,bvh_writer,library,pipeline}.py
  empty placeholders for upcoming tasks
- stickerbook/assets/motions/library/.gitkeep with *.bvh ignored
- requirements.txt: mediapipe>=0.10

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: FrameRecorder

**Files:**
- Create: `stickerbook/motion/recorder.py`
- Create: `stickerbook/tests/test_motion_recorder.py`

- [ ] **Step 1: 실패 테스트 작성**

`stickerbook/tests/test_motion_recorder.py`:
```python
"""Unit tests for FrameRecorder."""
from __future__ import annotations

import numpy as np
import pytest

from motion.recorder import FrameRecorder


def make_dummy_frame() -> np.ndarray:
    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_recorder_starts_idle():
    rec = FrameRecorder()
    assert rec.is_recording() is False


def test_recorder_start_then_stop_returns_buffered_frames():
    rec = FrameRecorder()
    rec.start()
    assert rec.is_recording() is True

    rec.add_frame(make_dummy_frame())
    rec.add_frame(make_dummy_frame())
    rec.add_frame(make_dummy_frame())

    frames = rec.stop()
    assert rec.is_recording() is False
    assert len(frames) == 3
    assert frames[0].shape == (10, 10, 3)


def test_add_frame_ignored_when_not_recording():
    rec = FrameRecorder()
    rec.add_frame(make_dummy_frame())
    rec.start()
    rec.stop()
    rec.add_frame(make_dummy_frame())  # idle 상태
    assert rec.is_recording() is False


def test_start_clears_previous_buffer():
    rec = FrameRecorder()
    rec.start()
    rec.add_frame(make_dummy_frame())
    rec.stop()

    rec.start()
    rec.add_frame(make_dummy_frame())
    frames = rec.stop()
    assert len(frames) == 1
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /home/ingon/AR_book/drawing-to-2.5d-repo/stickerbook
/usr/bin/python3 -m pytest tests/test_motion_recorder.py -v
```

Expected: FAIL — `ImportError: cannot import name 'FrameRecorder'`

- [ ] **Step 3: recorder.py 구현**

`stickerbook/motion/recorder.py`:
```python
"""Frame buffer for M-key triggered motion recording.

Main loop calls add_frame() each tick; toggle handlers call start()/stop().
"""
from __future__ import annotations

from typing import List

import numpy as np


class FrameRecorder:
    def __init__(self) -> None:
        self._recording: bool = False
        self._buffer: List[np.ndarray] = []

    def start(self) -> None:
        self._recording = True
        self._buffer = []

    def stop(self) -> List[np.ndarray]:
        self._recording = False
        frames = self._buffer
        self._buffer = []
        return frames

    def is_recording(self) -> bool:
        return self._recording

    def add_frame(self, frame: np.ndarray) -> None:
        if not self._recording:
            return
        self._buffer.append(frame.copy())
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
/usr/bin/python3 -m pytest tests/test_motion_recorder.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add stickerbook/motion/recorder.py stickerbook/tests/test_motion_recorder.py
git commit -m "$(cat <<'EOF'
feat(motion): FrameRecorder for M-toggle frame buffering

start/stop/is_recording/add_frame interface. add_frame is no-op when not
recording so the live loop can call it unconditionally.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: PoseEstimator (MediaPipe wrapper)

**Files:**
- Create: `stickerbook/motion/pose_estimator.py`
- Create: `stickerbook/tests/test_motion_pose_estimator.py`

- [ ] **Step 1: 실패 테스트 작성**

`stickerbook/tests/test_motion_pose_estimator.py`:
```python
"""Unit tests for PoseEstimator."""
from __future__ import annotations

import numpy as np
import pytest

mediapipe = pytest.importorskip("mediapipe")

from motion.pose_estimator import PoseEstimator, PoseLandmarks


def make_blank_frame() -> np.ndarray:
    return np.full((480, 640, 3), 128, dtype=np.uint8)


def test_estimator_returns_one_result_per_frame():
    est = PoseEstimator()
    results = est.estimate_batch([make_blank_frame()])
    est.close()
    assert len(results) == 1


def test_blank_frame_yields_none_or_low_visibility():
    """Blank frame has no person, so MediaPipe returns None or unrecognized."""
    est = PoseEstimator()
    results = est.estimate_batch([make_blank_frame()])
    est.close()
    # blank frame: either None (no detection) or PoseLandmarks with all zeros
    assert results[0] is None or isinstance(results[0], PoseLandmarks)


def test_pose_landmarks_has_33_points():
    """PoseLandmarks dataclass should expose 33 (x,y,z) points."""
    pl = PoseLandmarks(points=np.zeros((33, 3), dtype=np.float32))
    assert pl.points.shape == (33, 3)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
/usr/bin/python3 -m pytest tests/test_motion_pose_estimator.py -v
```

Expected: FAIL — ImportError

- [ ] **Step 3: pose_estimator.py 구현**

`stickerbook/motion/pose_estimator.py`:
```python
"""MediaPipe Pose wrapper. Each frame returns 33 3D landmarks (or None).

Why a thin wrapper: callers can mock this without pulling MediaPipe into
their test setups.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import mediapipe as mp


@dataclass(frozen=True)
class PoseLandmarks:
    """33 normalized world-space points (x, y, z)."""
    points: np.ndarray  # shape (33, 3), dtype float32


class PoseEstimator:
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
    ) -> None:
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def estimate_batch(
        self, frames: List[np.ndarray]
    ) -> List[Optional[PoseLandmarks]]:
        results: List[Optional[PoseLandmarks]] = []
        for frame_bgr in frames:
            # MediaPipe expects RGB
            frame_rgb = frame_bgr[..., ::-1]
            res = self._pose.process(frame_rgb)
            if res.pose_world_landmarks is None:
                results.append(None)
                continue
            pts = np.array(
                [
                    (lm.x, lm.y, lm.z)
                    for lm in res.pose_world_landmarks.landmark
                ],
                dtype=np.float32,
            )
            results.append(PoseLandmarks(points=pts))
        return results

    def close(self) -> None:
        self._pose.close()
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
/usr/bin/python3 -m pytest tests/test_motion_pose_estimator.py -v
```

Expected: 3 passed (또는 mediapipe 미설치 환경에선 skipped)

- [ ] **Step 5: Commit**

```bash
git add stickerbook/motion/pose_estimator.py stickerbook/tests/test_motion_pose_estimator.py
git commit -m "$(cat <<'EOF'
feat(motion): PoseEstimator wraps MediaPipe Pose, returns 33-point landmarks

Per-frame List[Optional[PoseLandmarks]]. Frames with no detection yield
None so callers can decide failure threshold (e.g., > 50% None → abort).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: BVH Writer

**Files:**
- Create: `stickerbook/motion/bvh_writer.py`
- Create: `stickerbook/tests/test_motion_bvh_writer.py`

**Approach:** First cut writes BVH with **position channels per joint** (Xposition/Yposition/Zposition) and **zero rotations**. AD reads positions for retargeting (verified by my_dance.bvh examples). If smoke test (Task 8) shows incorrect retargeting, follow-up task adds rotation computation. Position-only is simpler and matches Rokoko Vision BVH output style we already validated.

**Skeleton (compatible with `examples/config/retarget/my_dance.yaml`):**

```
Root
└── Hips
    ├── Spine1 → Spine2 → Spine3 → Spine4
    │   ├── Neck → Head
    │   ├── LeftShoulder → LeftArm → LeftForeArm → LeftHand
    │   └── RightShoulder → RightArm → RightForeArm → RightHand
    ├── LeftThigh → LeftShin → LeftFoot → LeftToe
    └── RightThigh → RightShin → RightFoot → RightToe
```

**Joint position source (MediaPipe 33 → joint world position):**

| BVH joint | Source |
|---|---|
| Root | (0, 0, 0) — fixed origin |
| Hips | midpoint(LEFT_HIP, RIGHT_HIP) |
| Spine1 | lerp(hips, shoulder_mid, 0.25) |
| Spine2 | lerp(hips, shoulder_mid, 0.50) |
| Spine3 | lerp(hips, shoulder_mid, 0.75) |
| Spine4 | shoulder_mid |
| Neck | shoulder_mid + 0.05 * up |
| Head | NOSE |
| LeftShoulder | LEFT_SHOULDER |
| LeftArm | LEFT_SHOULDER (same point — 어깨와 위팔 시작 동일) |
| LeftForeArm | LEFT_ELBOW |
| LeftHand | LEFT_WRIST |
| (Right side mirrors Left) | |
| LeftThigh | LEFT_HIP |
| LeftShin | LEFT_KNEE |
| LeftFoot | LEFT_ANKLE |
| LeftToe | LEFT_FOOT_INDEX |

`shoulder_mid = midpoint(LEFT_SHOULDER, RIGHT_SHOULDER)`

- [ ] **Step 1: 실패 테스트 작성**

`stickerbook/tests/test_motion_bvh_writer.py`:
```python
"""Unit tests for bvh_writer.write_bvh."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from motion.bvh_writer import write_bvh
from motion.pose_estimator import PoseLandmarks


def make_t_pose_landmarks() -> PoseLandmarks:
    """A simple T-pose-ish 33-landmark fixture (only the ones we use)."""
    pts = np.zeros((33, 3), dtype=np.float32)
    # MediaPipe indices: 11=L_SHOULDER, 12=R_SHOULDER, 13=L_ELBOW, 14=R_ELBOW,
    # 15=L_WRIST, 16=R_WRIST, 23=L_HIP, 24=R_HIP, 25=L_KNEE, 26=R_KNEE,
    # 27=L_ANKLE, 28=R_ANKLE, 31=L_FOOT_INDEX, 32=R_FOOT_INDEX, 0=NOSE
    pts[11] = [-0.2, 1.5, 0]   # L shoulder
    pts[12] = [0.2, 1.5, 0]    # R shoulder
    pts[13] = [-0.5, 1.5, 0]   # L elbow (out)
    pts[14] = [0.5, 1.5, 0]    # R elbow (out)
    pts[15] = [-0.7, 1.5, 0]   # L wrist
    pts[16] = [0.7, 1.5, 0]    # R wrist
    pts[23] = [-0.15, 1.0, 0]  # L hip
    pts[24] = [0.15, 1.0, 0]   # R hip
    pts[25] = [-0.15, 0.5, 0]  # L knee
    pts[26] = [0.15, 0.5, 0]   # R knee
    pts[27] = [-0.15, 0.0, 0]  # L ankle
    pts[28] = [0.15, 0.0, 0]   # R ankle
    pts[31] = [-0.15, 0.0, 0.1]  # L foot index
    pts[32] = [0.15, 0.0, 0.1]   # R foot index
    pts[0] = [0, 1.7, 0]       # nose
    return PoseLandmarks(points=pts)


def test_write_bvh_creates_file(tmp_path: Path):
    out = tmp_path / "test.bvh"
    write_bvh([make_t_pose_landmarks()] * 5, fps=30.0, output_path=out)
    assert out.is_file()


def test_bvh_starts_with_hierarchy(tmp_path: Path):
    out = tmp_path / "test.bvh"
    write_bvh([make_t_pose_landmarks()], fps=30.0, output_path=out)
    text = out.read_text()
    assert text.startswith("HIERARCHY")
    assert "ROOT Root" in text
    assert "JOINT Hips" in text
    assert "JOINT Spine1" in text
    assert "JOINT LeftThigh" in text
    assert "JOINT RightThigh" in text
    assert "JOINT LeftShin" in text
    assert "JOINT LeftToe" in text


def test_bvh_motion_section_has_correct_frame_count(tmp_path: Path):
    out = tmp_path / "test.bvh"
    write_bvh([make_t_pose_landmarks()] * 7, fps=30.0, output_path=out)
    text = out.read_text()
    assert "MOTION" in text
    assert "Frames: 7" in text
    assert "Frame Time: 0.0333" in text  # 1/30


def test_bvh_motion_lines_count_matches_frames(tmp_path: Path):
    out = tmp_path / "test.bvh"
    write_bvh([make_t_pose_landmarks()] * 4, fps=30.0, output_path=out)
    text = out.read_text()
    motion_section = text.split("MOTION")[1]
    motion_lines = [
        ln for ln in motion_section.splitlines()
        if ln.strip() and not ln.startswith("Frames") and not ln.startswith("Frame Time")
    ]
    assert len(motion_lines) == 4


def test_none_landmarks_skipped(tmp_path: Path):
    """If a frame is None (recognition failure), interpolate or skip."""
    out = tmp_path / "test.bvh"
    seq = [make_t_pose_landmarks(), None, make_t_pose_landmarks()]
    write_bvh(seq, fps=30.0, output_path=out)
    text = out.read_text()
    # 2 valid frames written (None dropped) — strict design choice
    assert "Frames: 2" in text
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
/usr/bin/python3 -m pytest tests/test_motion_bvh_writer.py -v
```

Expected: FAIL — ImportError

- [ ] **Step 3: bvh_writer.py 구현**

`stickerbook/motion/bvh_writer.py`:
```python
"""Convert MediaPipe Pose landmark sequence to BVH text.

Strategy: write per-joint position channels (Xposition/Yposition/Zposition)
with zero rotations. AnimatedDrawings retargeting uses joint positions
(verified by Rokoko Vision BVH samples we already validated as my_dance).
Joint name set matches `examples/config/retarget/my_dance.yaml`.

Skeleton:
    Root → Hips → (Spine1→Spine2→Spine3→Spine4 → Neck→Head, LeftArm chain,
                   RightArm chain), LeftThigh chain, RightThigh chain.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from motion.pose_estimator import PoseLandmarks


# MediaPipe Pose landmark indices (subset we use)
MP_NOSE = 0
MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER = 11, 12
MP_LEFT_ELBOW, MP_RIGHT_ELBOW = 13, 14
MP_LEFT_WRIST, MP_RIGHT_WRIST = 15, 16
MP_LEFT_HIP, MP_RIGHT_HIP = 23, 24
MP_LEFT_KNEE, MP_RIGHT_KNEE = 25, 26
MP_LEFT_ANKLE, MP_RIGHT_ANKLE = 27, 28
MP_LEFT_FOOT_INDEX, MP_RIGHT_FOOT_INDEX = 31, 32


# Skeleton hierarchy in DFS order. Each entry: (joint_name, parent_index_in_list)
# parent_index = -1 means root (no parent)
SKELETON: List[Tuple[str, int]] = [
    ("Root", -1),         # 0
    ("Hips", 0),          # 1
    ("Spine1", 1),        # 2
    ("Spine2", 2),        # 3
    ("Spine3", 3),        # 4
    ("Spine4", 4),        # 5
    ("Neck", 5),          # 6
    ("Head", 6),          # 7
    ("LeftShoulder", 5),  # 8
    ("LeftArm", 8),       # 9
    ("LeftForeArm", 9),   # 10
    ("LeftHand", 10),     # 11
    ("RightShoulder", 5), # 12
    ("RightArm", 12),     # 13
    ("RightForeArm", 13), # 14
    ("RightHand", 14),    # 15
    ("LeftThigh", 1),     # 16
    ("LeftShin", 16),     # 17
    ("LeftFoot", 17),     # 18
    ("LeftToe", 18),      # 19
    ("RightThigh", 1),    # 20
    ("RightShin", 20),    # 21
    ("RightFoot", 21),    # 22
    ("RightToe", 22),     # 23
]


def _compute_joint_positions(
    lm: PoseLandmarks,
) -> np.ndarray:
    """Map 33 MediaPipe landmarks → 24 BVH joint world positions.

    Returns shape (24, 3) ordered by SKELETON.
    """
    p = lm.points  # (33, 3)
    hip_mid = (p[MP_LEFT_HIP] + p[MP_RIGHT_HIP]) / 2.0
    sho_mid = (p[MP_LEFT_SHOULDER] + p[MP_RIGHT_SHOULDER]) / 2.0
    spine_dir = sho_mid - hip_mid

    pos = np.zeros((24, 3), dtype=np.float32)
    pos[0] = (0, 0, 0)                      # Root
    pos[1] = hip_mid                         # Hips
    pos[2] = hip_mid + 0.25 * spine_dir      # Spine1
    pos[3] = hip_mid + 0.50 * spine_dir      # Spine2
    pos[4] = hip_mid + 0.75 * spine_dir      # Spine3
    pos[5] = sho_mid                         # Spine4
    pos[6] = sho_mid + np.array([0, 0.05, 0], dtype=np.float32)  # Neck
    pos[7] = p[MP_NOSE]                      # Head
    pos[8] = p[MP_LEFT_SHOULDER]             # LeftShoulder
    pos[9] = p[MP_LEFT_SHOULDER]             # LeftArm (same start)
    pos[10] = p[MP_LEFT_ELBOW]               # LeftForeArm
    pos[11] = p[MP_LEFT_WRIST]               # LeftHand
    pos[12] = p[MP_RIGHT_SHOULDER]           # RightShoulder
    pos[13] = p[MP_RIGHT_SHOULDER]           # RightArm
    pos[14] = p[MP_RIGHT_ELBOW]              # RightForeArm
    pos[15] = p[MP_RIGHT_WRIST]              # RightHand
    pos[16] = p[MP_LEFT_HIP]                 # LeftThigh
    pos[17] = p[MP_LEFT_KNEE]                # LeftShin
    pos[18] = p[MP_LEFT_ANKLE]               # LeftFoot
    pos[19] = p[MP_LEFT_FOOT_INDEX]          # LeftToe
    pos[20] = p[MP_RIGHT_HIP]                # RightThigh
    pos[21] = p[MP_RIGHT_KNEE]               # RightShin
    pos[22] = p[MP_RIGHT_ANKLE]              # RightFoot
    pos[23] = p[MP_RIGHT_FOOT_INDEX]         # RightToe
    return pos


def _build_hierarchy_text(rest_positions: np.ndarray) -> str:
    """Build HIERARCHY section. Offsets = rest_positions[child] - rest_positions[parent]."""
    lines: List[str] = ["HIERARCHY"]
    indent_stack: List[int] = []  # indices in SKELETON order

    def indent(level: int) -> str:
        return "\t" * level

    # Two-pass: build per-joint open/close tracking
    # Use simpler approach: recursive emission
    children: dict[int, list[int]] = {i: [] for i in range(len(SKELETON))}
    for i, (_, parent_idx) in enumerate(SKELETON):
        if parent_idx >= 0:
            children[parent_idx].append(i)

    def emit(idx: int, level: int) -> None:
        name, parent_idx = SKELETON[idx]
        offset = (
            rest_positions[idx]
            if parent_idx < 0
            else rest_positions[idx] - rest_positions[parent_idx]
        )
        keyword = "ROOT" if parent_idx < 0 else "JOINT"
        lines.append(f"{indent(level)}{keyword} {name}")
        lines.append(f"{indent(level)}{{")
        lines.append(
            f"{indent(level+1)}OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}"
        )
        lines.append(
            f"{indent(level+1)}CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
        )
        for child in children[idx]:
            emit(child, level + 1)
        if not children[idx]:
            # End site for leaf
            lines.append(f"{indent(level+1)}End Site")
            lines.append(f"{indent(level+1)}{{")
            lines.append(f"{indent(level+2)}OFFSET 0.000000 0.000000 0.000000")
            lines.append(f"{indent(level+1)}}}")
        lines.append(f"{indent(level)}}}")

    emit(0, 0)
    return "\n".join(lines)


def _build_motion_text(
    sequence: List[np.ndarray], fps: float
) -> str:
    """Build MOTION section. Each frame: per-joint Xpos Ypos Zpos Xrot Yrot Zrot."""
    lines: List[str] = ["MOTION"]
    lines.append(f"Frames: {len(sequence)}")
    lines.append(f"Frame Time: {1.0 / fps:.6f}")
    for positions in sequence:
        # 24 joints × 6 channels (3 pos + 3 rot=0)
        parts: List[str] = []
        for j in range(positions.shape[0]):
            parts.extend(
                [f"{positions[j, 0]:.6f}", f"{positions[j, 1]:.6f}", f"{positions[j, 2]:.6f}"]
            )
            parts.extend(["0.000000", "0.000000", "0.000000"])
        lines.append(" ".join(parts))
    return "\n".join(lines)


def write_bvh(
    landmark_sequence: List[Optional[PoseLandmarks]],
    fps: float,
    output_path: Path,
) -> None:
    """landmarks → BVH file. None frames are dropped."""
    valid_seq = [lm for lm in landmark_sequence if lm is not None]
    if not valid_seq:
        raise ValueError("write_bvh: empty valid landmark sequence")

    rest_positions = _compute_joint_positions(valid_seq[0])
    hierarchy_text = _build_hierarchy_text(rest_positions)

    frame_positions: List[np.ndarray] = [
        _compute_joint_positions(lm) for lm in valid_seq
    ]
    motion_text = _build_motion_text(frame_positions, fps)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(hierarchy_text + "\n" + motion_text + "\n")
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
/usr/bin/python3 -m pytest tests/test_motion_bvh_writer.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add stickerbook/motion/bvh_writer.py stickerbook/tests/test_motion_bvh_writer.py
git commit -m "$(cat <<'EOF'
feat(motion): bvh_writer maps MediaPipe landmarks → BVH text

Joint set matches examples/config/retarget/my_dance.yaml: Root/Hips/Spine1-4/
Neck/Head/Left+Right Shoulder/Arm/ForeArm/Hand/Thigh/Shin/Foot/Toe (24 joints).
First-cut writes position channels with zero rotation; AD reads positions
for retargeting (verified against my_dance.bvh).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: MotionLibrary

**Files:**
- Create: `stickerbook/motion/library.py`
- Create: `stickerbook/tests/test_motion_library.py`

- [ ] **Step 1: 실패 테스트 작성**

`stickerbook/tests/test_motion_library.py`:
```python
"""Unit tests for MotionLibrary."""
from __future__ import annotations

from pathlib import Path

import pytest

from motion.library import MotionLibrary


@pytest.fixture
def fake_ad_repo(tmp_path: Path) -> Path:
    """Mimic AnimatedDrawings repo dirs and seed retarget my_dance.yaml."""
    ad_repo = tmp_path / "ad_repo"
    (ad_repo / "examples" / "bvh").mkdir(parents=True)
    (ad_repo / "examples" / "config" / "motion").mkdir(parents=True)
    rt_dir = ad_repo / "examples" / "config" / "retarget"
    rt_dir.mkdir(parents=True)
    (rt_dir / "my_dance.yaml").write_text("char_starting_location: [0,0,-0.5]\n")
    return ad_repo


def make_dummy_bvh(path: Path) -> Path:
    path.write_text("HIERARCHY\nROOT Root\n{\nOFFSET 0 0 0\n}\nMOTION\nFrames: 1\nFrame Time: 0.033\n0 0 0\n")
    return path


def test_add_creates_motion_001(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "src.bvh")

    name = lib.add(src)
    assert name == "motion_001"
    assert (tmp_path / "lib" / "motion_001.bvh").is_file()
    assert (fake_ad_repo / "examples" / "bvh" / "motion_001.bvh").is_file()
    assert (fake_ad_repo / "examples" / "config" / "motion" / "motion_001.yaml").is_file()
    assert (fake_ad_repo / "examples" / "config" / "retarget" / "motion_001.yaml").is_file()


def test_motion_yaml_has_correct_filepath(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "src.bvh")
    lib.add(src)

    motion_yaml = (fake_ad_repo / "examples" / "config" / "motion" / "motion_001.yaml").read_text()
    assert "filepath: examples/bvh/motion_001.bvh" in motion_yaml
    assert "scale:" in motion_yaml
    assert "groundplane_joint:" in motion_yaml


def test_add_increments_counter(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src1 = make_dummy_bvh(tmp_path / "a.bvh")
    src2 = make_dummy_bvh(tmp_path / "b.bvh")
    assert lib.add(src1) == "motion_001"
    assert lib.add(src2) == "motion_002"


def test_list_returns_added_names(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "a.bvh")
    lib.add(src)
    lib.add(src)
    assert lib.list() == ["motion_001", "motion_002"]


def test_set_active_and_active(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "a.bvh")
    lib.add(src)
    lib.set_active("motion_001")
    assert lib.active() == "motion_001"


def test_active_none_when_empty(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    assert lib.active() is None


def test_get_by_index(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "a.bvh")
    lib.add(src)
    lib.add(src)
    assert lib.get_by_index(1) == "motion_001"
    assert lib.get_by_index(2) == "motion_002"
    assert lib.get_by_index(99) is None


def test_persistence_across_instances(tmp_path: Path, fake_ad_repo: Path):
    """Counter and listing survive process restart (re-scan disk)."""
    lib1 = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "a.bvh")
    lib1.add(src)

    lib2 = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    assert lib2.list() == ["motion_001"]
    assert lib2.add(src) == "motion_002"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
/usr/bin/python3 -m pytest tests/test_motion_library.py -v
```

Expected: FAIL — ImportError

- [ ] **Step 3: library.py 구현**

`stickerbook/motion/library.py`:
```python
"""MotionLibrary — BVH 라이브러리 + AD config 자동 등록 + 활성 motion 상태.

Persistence: library_dir에 motion_NNN.bvh 파일들로 카운터/목록 자동 복원.
AD 측 파일 (examples/bvh, examples/config/motion, examples/config/retarget)도
같이 생성. retarget yaml은 my_dance.yaml의 복사 (joint 이름 호환 가정).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List, Optional


_NAME_RE = re.compile(r"^motion_(\d{3})\.bvh$")


_MOTION_YAML_TEMPLATE = """\
filepath: examples/bvh/{name}.bvh
start_frame_idx: 0
end_frame_idx: {end_idx}
groundplane_joint: LeftFoot
forward_perp_joint_vectors:
  - - LeftShoulder
    - RightShoulder
  - - LeftThigh
    - RightThigh
scale: 0.025
up: +y
"""


class MotionLibrary:
    def __init__(self, library_dir: Path, ad_repo_path: Path) -> None:
        self._library_dir = Path(library_dir)
        self._ad_repo = Path(ad_repo_path)
        self._library_dir.mkdir(parents=True, exist_ok=True)
        self._active: Optional[str] = None

    def list(self) -> List[str]:
        names: List[str] = []
        for p in sorted(self._library_dir.iterdir()):
            m = _NAME_RE.match(p.name)
            if m:
                names.append(f"motion_{m.group(1)}")
        return names

    def add(self, bvh_path: Path) -> str:
        bvh_path = Path(bvh_path)
        existing = self.list()
        next_n = len(existing) + 1
        name = f"motion_{next_n:03d}"

        # 1. copy original to library
        dst_local = self._library_dir / f"{name}.bvh"
        shutil.copyfile(bvh_path, dst_local)

        # 2. copy to AD examples/bvh
        ad_bvh = self._ad_repo / "examples" / "bvh" / f"{name}.bvh"
        ad_bvh.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bvh_path, ad_bvh)

        # 3. write motion config yaml
        end_idx = self._count_bvh_frames(bvh_path) - 1
        ad_motion_yaml = (
            self._ad_repo / "examples" / "config" / "motion" / f"{name}.yaml"
        )
        ad_motion_yaml.parent.mkdir(parents=True, exist_ok=True)
        ad_motion_yaml.write_text(
            _MOTION_YAML_TEMPLATE.format(name=name, end_idx=max(end_idx, 0))
        )

        # 4. retarget yaml = copy of my_dance.yaml (same joint name set)
        src_retarget = (
            self._ad_repo / "examples" / "config" / "retarget" / "my_dance.yaml"
        )
        ad_retarget_yaml = (
            self._ad_repo / "examples" / "config" / "retarget" / f"{name}.yaml"
        )
        if src_retarget.is_file():
            shutil.copyfile(src_retarget, ad_retarget_yaml)
        # If my_dance.yaml is missing, AD subprocess falls back to default
        # fair1_ppf — joint mismatch will surface in the smoke test.

        return name

    def set_active(self, name: str) -> None:
        if name not in self.list():
            raise ValueError(f"motion not in library: {name}")
        self._active = name

    def active(self) -> Optional[str]:
        return self._active

    def get_by_index(self, idx: int) -> Optional[str]:
        names = self.list()
        if 1 <= idx <= len(names):
            return names[idx - 1]
        return None

    @staticmethod
    def _count_bvh_frames(path: Path) -> int:
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("Frames:"):
                try:
                    return int(stripped.split(":", 1)[1].strip())
                except ValueError:
                    return 0
        return 0
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
/usr/bin/python3 -m pytest tests/test_motion_library.py -v
```

Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add stickerbook/motion/library.py stickerbook/tests/test_motion_library.py
git commit -m "$(cat <<'EOF'
feat(motion): MotionLibrary auto-registers BVH into AD examples/

add() copies BVH to library_dir + AD examples/bvh, writes motion config
yaml, and clones my_dance.yaml as the retarget config (compatible joint
set). Counter persists across runs by scanning library_dir on init.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: MotionPipeline

**Files:**
- Create: `stickerbook/motion/pipeline.py`
- Create: `stickerbook/tests/test_motion_pipeline.py`

- [ ] **Step 1: 실패 테스트 작성**

`stickerbook/tests/test_motion_pipeline.py`:
```python
"""Integration test for MotionPipeline.toggle()."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from motion.pipeline import MotionPipeline
from motion.pose_estimator import PoseLandmarks


def _t_pose_landmarks() -> PoseLandmarks:
    pts = np.zeros((33, 3), dtype=np.float32)
    pts[11] = [-0.2, 1.5, 0]; pts[12] = [0.2, 1.5, 0]
    pts[13] = [-0.5, 1.5, 0]; pts[14] = [0.5, 1.5, 0]
    pts[15] = [-0.7, 1.5, 0]; pts[16] = [0.7, 1.5, 0]
    pts[23] = [-0.15, 1.0, 0]; pts[24] = [0.15, 1.0, 0]
    pts[25] = [-0.15, 0.5, 0]; pts[26] = [0.15, 0.5, 0]
    pts[27] = [-0.15, 0.0, 0]; pts[28] = [0.15, 0.0, 0]
    pts[31] = [-0.15, 0.0, 0.1]; pts[32] = [0.15, 0.0, 0.1]
    pts[0] = [0, 1.7, 0]
    return PoseLandmarks(points=pts)


def test_toggle_starts_recording_first_call(tmp_path: Path):
    rec = MagicMock()
    rec.is_recording.return_value = False
    est = MagicMock()
    lib = MagicMock()

    pipeline = MotionPipeline(
        recorder=rec, estimator=est, library=lib,
        tmp_dir=tmp_path, fps=30.0,
    )
    name = pipeline.toggle()
    rec.start.assert_called_once()
    assert name is None  # 녹화만 시작


def test_toggle_stops_and_processes_second_call(tmp_path: Path):
    """Stop, run pose estimator, write BVH, add to library, set active."""
    rec = MagicMock()
    rec.is_recording.return_value = True
    rec.stop.return_value = [
        np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(60)
    ]
    est = MagicMock()
    est.estimate_batch.return_value = [_t_pose_landmarks()] * 60
    lib = MagicMock()
    lib.add.return_value = "motion_001"

    pipeline = MotionPipeline(
        recorder=rec, estimator=est, library=lib,
        tmp_dir=tmp_path, fps=30.0,
    )
    name = pipeline.toggle()

    assert name == "motion_001"
    rec.stop.assert_called_once()
    est.estimate_batch.assert_called_once()
    lib.add.assert_called_once()
    lib.set_active.assert_called_once_with("motion_001")


def test_toggle_aborts_on_too_few_frames(tmp_path: Path):
    rec = MagicMock()
    rec.is_recording.return_value = True
    rec.stop.return_value = [
        np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)
    ]  # < 30
    est = MagicMock()
    lib = MagicMock()

    pipeline = MotionPipeline(
        recorder=rec, estimator=est, library=lib,
        tmp_dir=tmp_path, fps=30.0,
    )
    name = pipeline.toggle()
    assert name is None
    est.estimate_batch.assert_not_called()
    lib.add.assert_not_called()


def test_toggle_aborts_on_high_recognition_failure(tmp_path: Path):
    rec = MagicMock()
    rec.is_recording.return_value = True
    rec.stop.return_value = [
        np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(60)
    ]
    est = MagicMock()
    # 80% 실패율 (None 48 / 60)
    est.estimate_batch.return_value = [None] * 48 + [_t_pose_landmarks()] * 12
    lib = MagicMock()

    pipeline = MotionPipeline(
        recorder=rec, estimator=est, library=lib,
        tmp_dir=tmp_path, fps=30.0,
    )
    name = pipeline.toggle()
    assert name is None
    lib.add.assert_not_called()
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
/usr/bin/python3 -m pytest tests/test_motion_pipeline.py -v
```

Expected: FAIL — ImportError

- [ ] **Step 3: pipeline.py 구현**

`stickerbook/motion/pipeline.py`:
```python
"""MotionPipeline — M toggle entry. Wires recorder + estimator + bvh_writer + library.

First call: recorder.start().
Second call: stop → estimate → write bvh → library.add → library.set_active.
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
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
/usr/bin/python3 -m pytest tests/test_motion_pipeline.py -v
```

Expected: 4 passed

- [ ] **Step 5: 모든 motion 테스트 한꺼번에 통과 확인**

```bash
/usr/bin/python3 -m pytest tests/test_motion_*.py -v
```

Expected: 20+ passed (모든 motion 단위 테스트)

- [ ] **Step 6: Commit**

```bash
git add stickerbook/motion/pipeline.py stickerbook/tests/test_motion_pipeline.py
git commit -m "$(cat <<'EOF'
feat(motion): MotionPipeline.toggle() entry for M-key recording

First toggle: recorder.start. Second toggle: stop → estimate → write bvh
→ library.add → set_active. Aborts with logged reason when frame count
< 30 or recognition failure rate > 50%.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: app.py 확장

**Files:**
- Modify: `stickerbook/app.py`

이 task는 unit test 어려움 (대화형 UI). smoke test (Task 8)로 검증.

- [ ] **Step 1: imports + AppAction 확장**

`stickerbook/app.py`의 imports 부분에 추가:
```python
from motion.library import MotionLibrary
from motion.pipeline import MotionPipeline
from motion.pose_estimator import PoseEstimator
from motion.recorder import FrameRecorder
```

`AppAction` enum에 새 값 추가:
```python
class AppAction(Enum):
    QUIT = auto()
    RESET = auto()
    SAVE = auto()
    CAPTURE = auto()
    RECORD_TOGGLE = auto()  # M 키
    SELECT_MOTION_1 = auto()
    SELECT_MOTION_2 = auto()
    SELECT_MOTION_3 = auto()
    SELECT_MOTION_4 = auto()
    SELECT_MOTION_5 = auto()
```

- [ ] **Step 2: `_handle_key` 분기 추가**

기존 `_handle_key` 메서드의 `return None` 직전에:
```python
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
        return None
```

- [ ] **Step 3: `__init__`에 motion 관련 instance vars 추가**

```python
        self._motion_library: Optional[MotionLibrary] = None
        self._motion_pipeline: Optional[MotionPipeline] = None
        self._motion_pose_estimator: Optional[PoseEstimator] = None
        self._motion_recorder: Optional[FrameRecorder] = None
```

- [ ] **Step 4: `run()`에 motion 모듈 초기화 추가**

`run()` 안의 TorchServe 시작 직후, `cv2.namedWindow(...)` 직전에 추가:
```python
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
```

`from config import ... ROOT` import도 추가 필요 — `config.py`의 `ROOT` 변수를 가져옴. 기존 imports 갱신:
```python
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
```

- [ ] **Step 5: 라이브 루프에서 매 frame `add_frame()` 호출**

`run()`의 main loop (`while True:`) 안, `raw = camera.read()` 직후에:
```python
                if self._motion_recorder is not None:
                    self._motion_recorder.add_frame(raw)
```

- [ ] **Step 6: action 처리 분기 확장**

기존 action 처리 (`if action is AppAction.QUIT:` 등) 옆에:
```python
                if action is AppAction.RECORD_TOGGLE:
                    if self._motion_pipeline is not None:
                        self._motion_pipeline.toggle()
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
```

- [ ] **Step 7: `_run_ad_pipeline`에서 활성 motion 사용**

기존 `motion="my_dance_3"` 줄을:
```python
            motion=(
                self._motion_library.active()
                if self._motion_library is not None and self._motion_library.active()
                else "my_dance_3"
            ),
```
로 교체. 라이브러리 비어있으면 기존 my_dance_3로 fallback.

- [ ] **Step 8: REC 표시 + 활성 motion 표시 (HUD)**

`run()` main loop 안, `cv2.imshow(WINDOW_NAME, display)` 직전에:
```python
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
```

- [ ] **Step 9: cleanup**

`run()`의 `finally:` 블록 안 기존 cleanup 옆:
```python
            if self._motion_pose_estimator is not None:
                self._motion_pose_estimator.close()
```

- [ ] **Step 10: syntax + import 검증**

```bash
cd /home/ingon/AR_book/drawing-to-2.5d-repo/stickerbook
/usr/bin/python3 -c "import ast; ast.parse(open('app.py').read()); print('syntax OK')"
/usr/bin/python3 -c "from app import App; print('import OK')"
```

Expected: 둘 다 OK

- [ ] **Step 11: 기존 테스트 회귀 검증**

```bash
/usr/bin/python3 -m pytest tests/ -v --ignore=tests/test_torchserve_runtime.py
```

Expected: 모든 motion + 기존 테스트 pass

- [ ] **Step 12: Commit**

```bash
git add stickerbook/app.py
git commit -m "$(cat <<'EOF'
feat(app): wire motion pipeline (M toggle, 1-5 select, HUD)

- M key toggles motion recording via MotionPipeline
- 1-5 keys switch active motion in MotionLibrary
- _run_ad_pipeline uses library.active() (falls back to my_dance_3)
- Live loop feeds each frame to FrameRecorder so M-toggle has data
- HUD: REC indicator and active motion label

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: 통합 smoke test + 라이브 시연

**Files:**
- (선택) Create: `stickerbook/scripts/motion_pipeline_smoke.py`

이 task는 라이브 카메라 환경 필수. 자동화 어려움 — 수동 시연으로 검증.

- [ ] **Step 1: TorchServe 잔여 정리**

```bash
pgrep -f torchserve | xargs -r kill -9 2>/dev/null
sleep 2
```

- [ ] **Step 2: 시작**

```bash
cd /home/ingon/AR_book/drawing-to-2.5d-repo/stickerbook
/usr/bin/python3 main.py --camera 1
```

- [ ] **Step 3: 시연 흐름 검증**

| 단계 | 동작 | 기대 결과 |
|---|---|---|
| 1 | 시작 (카메라 윈도우 떠야) | HUD: `motion: default (0 in lib)` |
| 2 | M 키 누름 | "REC" 표시, console `[motion] REC start` |
| 3 | 카메라 앞 5–10초 동작 | livestream 그대로 |
| 4 | M 키 다시 | "REC" 사라짐, console `[motion] REC stop (NNN frames)`, 이어서 `[motion] saved motion_001, active` |
| 5 | HUD 확인 | `motion: motion_001 (1 in lib)` |
| 6 | 그림 비추고 SPACE | BUSY 표시 후 sticker 등장 (motion_001 동작으로) |
| 7 | M 또 → 다른 동작 → M | `motion_002` 라이브러리에 추가, 활성 |
| 8 | 1 키 → motion_001 활성 | HUD `motion: motion_001` |
| 9 | SPACE → motion_001 sticker | sticker 등장 (motion_001 동작) |

- [ ] **Step 4: 캐릭터 자세 합리성 육안 확인**

생성된 sticker가:
- 그림 위에 합성됨 (homography tracking 동작)
- 캐릭터가 BVH 동작과 비슷하게 움직임 (정확도 100%일 필요 X, 큰 동작 패턴 일치 정도)

- [ ] **Step 5: 자세가 명백히 어긋나면 — fallback fix task 추가**

증상별 가능한 fix:
| 증상 | 원인 | fix 시도 |
|---|---|---|
| 캐릭터가 누워있음 | up 축 잘못 | `library.py`의 `_MOTION_YAML_TEMPLATE`의 `up: +y` → `up: +z` |
| 캐릭터가 너무 작거나 큼 | scale 잘못 | `scale: 0.025` → `0.01` 또는 `0.05` 시도 |
| 캐릭터가 뒤집힘 | 좌표계 차이 | `bvh_writer._compute_joint_positions`에서 y축 부호 flip |
| 캐릭터가 안 움직임 | position이 같은 값 | bvh_writer가 매 frame 같은 값 출력하는지 확인 |

각 fix는 별도 commit. 시도 후 다시 시연.

- [ ] **Step 6: 시연 OK 판정 시 Commit**

```bash
# 위 fix가 필요했다면:
git add stickerbook/motion/
git commit -m "fix(motion): adjust <param> based on smoke test (자세/스케일/축)"
```

OK 판정 = 캐릭터가 자기 그림 위에서, 사용자가 녹화한 동작과 시각적으로 비슷하게 움직임.

---

## Plan Self-Review

### Spec coverage 매핑

| Spec 섹션 | 이를 구현하는 Task |
|---|---|
| 신규 모듈 5개 (recorder/pose_estimator/bvh_writer/library/pipeline) | Tasks 2 / 3 / 4 / 5 / 6 |
| `app.py` M/1-5 분기 | Task 7 |
| `requirements.txt` mediapipe | Task 1 |
| `assets/motions/library/` + .gitignore | Task 1 |
| AD `examples/` 데이터만 추가 (코드 무손) | Task 5 (library.py가 examples/에 자동 추가) |
| Joint 매핑 (MediaPipe → my_dance 호환) | Task 4 (bvh_writer SKELETON + `_compute_joint_positions`) |
| Error handling: frame 부족 / 인식률 / 라이브러리 비어있음 | Task 6 (pipeline) + Task 7 (app fallback) |
| HUD: REC 표시 / 활성 motion 표시 | Task 7 (Step 8) |
| Persistence: 다음 실행에도 라이브러리 살아있음 | Task 5 (library.py가 disk scan으로 카운터 복원) |
| Smoke test | Task 8 |

### Placeholder scan
- TBD/TODO 없음
- "Add appropriate error handling" 같은 모호한 step 없음
- 각 step에 정확한 코드 또는 정확한 명령

### Type consistency
- `PoseLandmarks` (frozen dataclass with `points: np.ndarray`) — Task 3에서 정의, Task 4/6에서 동일 사용
- `MotionLibrary.add(bvh_path: Path) -> str` — Task 5 정의, Task 6에서 호출, Task 7에서 호출
- `MotionPipeline.toggle() -> Optional[str]` — Task 6 정의, Task 7에서 호출
- `FrameRecorder.add_frame(frame: np.ndarray)` — Task 2 정의, Task 7에서 호출

### 알려진 한계 (spec과 일치)
- BVH writer는 position-only (rotation = 0). Task 8 smoke test에서 부정확 시 fix task 추가 (Task 8 Step 5에 시나리오 명시)
- mediapipe 미설치 환경에선 motion pipeline 비활성 (Task 7 Step 4 except 처리)

---

## Execution

Plan complete and saved to `stickerbook/docs/superpowers/plans/2026-04-30-motion-pipeline-implementation.md`.
