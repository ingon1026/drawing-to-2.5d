# Motion Recording Pipeline — Design

> 작성일: 2026-04-30
> 상태: **Approved (브레인스토밍)**, 다음은 writing-plans
> 목표 일정: 1주

## 개요

V1 stickerbook (`drawing-to-2.5d-repo/stickerbook`)에 사용자 동작을 카메라로 녹화 → MediaPipe Pose → BVH 자동 변환 → AnimatedDrawings 모션 라이브러리에 자동 등록 → 그림 캡처(SPACE) 시 활성 모션으로 사용하는 통합 파이프라인을 추가한다.

**기존 흐름 (수작업)**: 폰 영상 → Rokoko Vision (FBX) → Blender (FBX→BVH) → AD `examples/` 수동 복사 → motion config yaml 수동 작성 → app.py에서 motion 이름 변경.

**새 흐름 (자동화)**: M 키 토글로 위 모든 단계가 한 번에. UX는 키 한 번 추가만.

## 목적 / 입출력 / 제약

| | 내용 |
|---|---|
| Input | 카메라 livestream (PC 카메라, 1280×720 @ 30fps) |
| Output | BVH 파일 + AD motion config yaml (자동 등록), 활성 motion 변경 |
| Constraint | 단일 카메라, 단일 사람 가정. 1주 데모용. PC GPU 환경 |

## 비목표

- 라이브 follow (매 프레임 실시간 따라하기) — 별도 R&D 트랙
- multi-view mocap
- 손가락 모션 (MediaPipe Pose 33 landmark에 손가락 X)
- 매우 정확한 retargeting (졸라맨 데모용 정확도 충분)

## UX

### 키 매핑

| 키 | 동작 |
|---|---|
| **M** | 동작 녹화 토글 (M 시작 → M 종료). 종료 시 자동 변환 + 라이브러리 저장 + 활성 |
| **SPACE** | 그림 캡처, **활성 모션**으로 졸라맨 등장 (기존 기능) |
| **1/2/3/...** | 라이브러리 N번 모션 활성으로 변경 |
| **R** | sticker reset (기존) |
| **S** | save (기존) |
| **Q / ESC** | quit (기존) |

### 시연 흐름

```
1. (라이브러리 비어있음) M → "REC" 표시
2. 카메라 앞에서 5–10초 동작 → M 다시
   → MediaPipe pose 추출 (수 초)
   → BVH 변환 + 라이브러리 저장 + AD config 자동 생성
   → 활성 motion = motion_001
3. 그림 비추고 SPACE → 졸라맨이 motion_001 따라 추는 sticker 등장 (기존 tracking/AR/lost 그대로)
4. M 또 → 라이브러리 motion_002 추가
5. 1 키로 #1 활성, 2 키로 #2 활성
6. 그림 또 비추고 SPACE → 활성 모션으로 sticker
```

라이브러리는 디스크 영구 저장 — 다음 실행에도 살아있음.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  stickerbook/app.py (확장)                       │
│   ├─ M 키          → motion.pipeline.toggle()    │
│   ├─ 1/2/3 키      → motion.library.set_active() │
│   ├─ SPACE 키      → run_animated_drawings(      │
│   │                   motion=library.active())   │
│   └─ 기존 흐름 (HomographyAnchor, AR 합성, lost  │
│      처리, multi-sticker, R/S/Q) 그대로          │
│                                                  │
│  stickerbook/motion/  ← 신규 모듈                │
│   ├─ recorder.py       M 토글 + frame 버퍼       │
│   ├─ pose_estimator.py MediaPipe Pose wrapper    │
│   ├─ bvh_writer.py     landmarks → BVH 텍스트    │
│   ├─ library.py        BVH 파일 + AD config 관리 │
│   └─ pipeline.py       위 4개 묶음 (M 종료 entry)│
│                                                  │
│  AnimatedDrawings/ (upstream, 코드 무손)         │
│   ├─ examples/bvh/motion_001.bvh ← library가 복사│
│   ├─ examples/config/motion/motion_001.yaml ←자동│
│   └─ examples/config/retarget/motion_001.yaml ←symlink to my_dance.yaml │
└─────────────────────────────────────────────────┘
```

## 모듈 책임 / 인터페이스

### `motion/recorder.py`

```python
class FrameRecorder:
    def start() -> None: ...
    def stop() -> List[np.ndarray]: ...    # BGR frames
    def is_recording() -> bool: ...
    def add_frame(frame: np.ndarray) -> None: ...
```

라이브 루프가 매 frame 마다 `if recorder.is_recording(): recorder.add_frame(frame)` 호출.

### `motion/pose_estimator.py`

```python
class PoseEstimator:
    def estimate_batch(frames: List[np.ndarray]) -> List[Optional[PoseLandmarks]]:
        """매 frame의 33 3D landmarks. 인식 실패 frame은 None."""
```

내부적으로 `mediapipe.solutions.pose.Pose` 사용.

### `motion/bvh_writer.py`

```python
def write_bvh(
    landmark_sequence: List[PoseLandmarks],
    fps: float,
    output_path: Path,
) -> None:
    """landmarks → BVH 텍스트. joint 이름은 my_dance retarget config 호환."""
```

#### Joint 매핑 (MediaPipe 33 → BVH)

```
MediaPipe Pose 33                →  BVH (my_dance retarget 호환)
─────────────────────────────────────────────────────────────────
left_hip ↔ right_hip 평균        →  Hips (root)
hip ↔ shoulder midpoint 분할     →  Spine1, Spine2, Spine3, Spine4 (interp)
left_shoulder / right_shoulder   →  LeftShoulder / RightShoulder
left_elbow   / right_elbow       →  LeftForeArm  / RightForeArm
left_wrist   / right_wrist       →  LeftHand     / RightHand
nose, mid_eye                    →  Neck → Head
left_hip   / right_hip           →  LeftThigh    / RightThigh
left_knee  / right_knee          →  LeftShin     / RightShin
left_ankle / right_ankle         →  LeftFoot     / RightFoot
left_foot_index / right_foot_index → LeftToe    / RightToe
```

각 joint의 local rotation은 parent-child 좌표계 cross product로 추정. 표준 IK-style 변환.

### `motion/library.py`

```python
class MotionLibrary:
    def __init__(self, library_dir: Path, ad_repo_path: Path): ...
    def add(self, bvh_path: Path) -> str:
        """라이브러리 등록.
        - assets/motions/library/motion_NNN.bvh 보관
        - AD examples/bvh/motion_NNN.bvh 복사
        - AD examples/config/motion/motion_NNN.yaml 생성
        - AD examples/config/retarget/motion_NNN.yaml symlink (my_dance.yaml로)
        반환: motion 이름 (motion_NNN)."""
    def list(self) -> List[str]: ...
    def set_active(self, name: str) -> None: ...
    def active(self) -> Optional[str]: ...
    def get_by_index(self, idx: int) -> Optional[str]: ...  # 1/2/3 키용
```

NNN = 3자리 zero-pad. 카운터는 disk scan으로 결정 (e.g. 마지막 motion_007이 있으면 다음은 008).

### `motion/pipeline.py`

```python
class MotionPipeline:
    def __init__(self, recorder, estimator, library, fps: float = 30.0): ...
    def toggle(self) -> Optional[str]:
        """녹화 시작 또는 종료. 종료 시 추출+변환+저장+활성 한 번에.
        반환: 새로 추가된 motion 이름 또는 None (실패시)."""
```

## File layout

```
stickerbook/
  motion/                                  ← 신규
    __init__.py
    recorder.py
    pose_estimator.py
    bvh_writer.py
    library.py
    pipeline.py
  assets/
    motions/
      library/                             ← 신규, gitignored
        motion_001.bvh
        motion_002.bvh
        ...

AnimatedDrawings/examples/                 ← upstream, but 우리가 데이터만 추가
  bvh/
    motion_001.bvh                         ← library.py가 위 원본을 복사
  config/motion/
    motion_001.yaml                        ← library.py가 자동 생성
  config/retarget/
    motion_001.yaml                        ← my_dance.yaml의 symlink
```

## Data flow

### M 토글 흐름

```
M start → recorder.start() → frame buffer 비움
        → main loop: 매 frame 마다 recorder.add_frame()

M stop  → frames = recorder.stop()
        → frame 수 < 30: 경고 + abort
        → landmarks_seq = pose_estimator.estimate_batch(frames)
        → 인식률 < 50%: 경고 + abort
        → temp_bvh = .../tmp_motion.bvh
        → bvh_writer.write_bvh(landmarks_seq, fps, temp_bvh)
        → name = library.add(temp_bvh)
        → library.set_active(name)
        → "[motion] saved {name}, active" 출력
```

### SPACE 흐름 (변경 1줄)

```
SPACE → motion_name = library.active() or "dab"  (fallback)
      → run_animated_drawings(motion=motion_name, ...)
      → 나머지 기존 그대로 (HomographyAnchor, AR 합성, lost 처리)
```

### 1/2/3 흐름

```
N 키 → name = library.get_by_index(N)
     → if name: library.set_active(name)
     → 다음 SPACE는 그 motion 사용
```

## Error handling

| 상황 | 처리 |
|---|---|
| 녹화 frame 수 < 30 (1초 미만) | 경고 + 라이브러리 추가 X |
| MediaPipe 인식 실패율 > 50% | 경고 + 추가 X |
| BVH writer 실패 (좌표 NaN 등) | 경고 + 추가 X |
| 1/2/3 라이브러리 범위 밖 | silent ignore |
| SPACE 시 라이브러리 비어있음 | AD default motion (`dab`) fallback |
| MediaPipe 모델 로드 실패 | 시작 시 경고 + M 키 비활성화 |
| AD 호출 실패 (기존 케이스) | 기존 처리 그대로 |

## Testing

- **`bvh_writer.py` 단위 테스트**: 알려진 T-pose landmark sequence 입력 → BVH 첫 frame의 골격 회전 검증 (단순 sanity check)
- **`library.py` 단위 테스트**: add → list → set_active → active → get_by_index 사이클. AD config 파일 생성 검증
- **통합 smoke test**: 더미 BGR frames (예: 검정 frame 30장) → MediaPipe (인식 실패 예상) → 경고 처리 검증
- **시각적 E2E**: 짧은 사용자 동작 → BVH 생성 → my_dance처럼 stickerbook 시연 → 캐릭터 자세 합리성 육안 확인

## Risks / 알려진 한계

1. **MediaPipe 33 landmark → joint rotation 변환 정확도**
   - landmark는 위치만, BVH는 rotation 필요
   - parent-child 좌표계 cross product로 rotation 추정 (표준 IK-like)
   - **첫 시도 자세 어긋날 가능성 60%**. up/scale 조정 또는 rotation 알고리즘 보정으로 fix 예상
2. **단일 카메라 깊이 한계**
   - 측면 동작 (옆 차기 등) 부정확. **정면 동작 권장**
3. **변환 시간**
   - MediaPipe inference: 5초 영상 (150 frames @ 30fps) → CPU 5–10초, GPU 2–3초
   - BVH 작성: <1초
   - **총 5–15초 예상**
4. **라이브러리 영구성 vs git**
   - `assets/motions/library/` = 사용자 개인 데이터 → `.gitignore` 추가
5. **MediaPipe 의존성**
   - 신규 의존성 `mediapipe>=0.10`. `requirements.txt` 추가
   - 모델 weight는 첫 호출 시 자동 다운로드 (~10MB)

## Open questions

- 활성 motion 시각적 표시 방법 (화면 한 구석에 "active: motion_002" 표시?)
- 녹화 길이 상한 (예: 30초 초과 시 자동 trim or 경고?)
- 라이브러리 항목 삭제 UX (필요? `D` 키 + 인덱스?) — 일단 비목표

## 다음 단계

- 본 spec 검토 후 [writing-plans](../../../../) skill로 implementation plan 작성
- 구현 후 my_dance.bvh 같은 기존 BVH로 sanity check 가능 (M 거치지 않고 library.add(my_dance.bvh) 직접 호출)
