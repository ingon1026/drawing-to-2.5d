# 2.5D 증강현실 (AR Stickerbook) — Stage 1 PoC 설계

> 작성일: 2026-04-21
> **Status (2026-04-21)**: Stage 1 구현 완료. M1~M6.1 작동 검증 (실기기 웹캠 테스트 통과).
> 남은 것: M7 (성능 측정·문서 정리), M7.5 (AnimatedDrawings 라운드트립 테스트), M8 (pop-up 스티커 렌더), Stage 2 (Galaxy 테블릿 on-device).

---

## Context

AR_book 워크스페이스에는 이미 여러 하위 프로젝트 (`drawing-2p5d`, `drawing-classifier(-v4)`, `sketch-guide` 등) 가 있으나, 방향이 산만해져 **새 폴더로 리셋** 하고 "**2.5D 증강현실**" 이라는 한 줄 목표에 집중하려 함.

- 참고 레포
  - **메인**: [facebookresearch/AnimatedDrawings](https://github.com/facebookresearch/AnimatedDrawings) (MIT) — 분할·포즈·ARAP 변형·BVH 리타게팅
  - **딱지화 기법**: [tatsuya-ogawa/RakugakiAR](https://github.com/tatsuya-ogawa/RakugakiAR) (Swift/ARKit) + [Qiita 시리즈 3편](https://qiita.com/pic18f14k50/items/0d20f5f544c9010f79e6)
  - **팀장 참고 코드**: `/home/ingon/AR_book/LivingDrawing/` (Python PC PoC, YOLO+MobileSAM+MediaPipe+warpAffine)

이 문서는 **Stage 1 (Python PC PoC)** 의 설계만 다룬다. Stage 2 (Galaxy 테블릿 on-device 포팅 — ARCore + ONNX Runtime Mobile, 네이티브 Android 기반) 는 연결 지점(데이터 계약)만 명시하고 세부 구현은 별도 문서로 미룬다.

---

## 목표 & 범위

### Goal
아이가 스케치북에 그린 그림을 웹캠으로 보여주고, 원하는 그림을 **클릭하면** 그 그림이 **2.5D 딱지로 추출되어** 월드 공간(종이 위)에 **고정된 AR 오버레이** 로 보이는 엔드투엔드 PC Python PoC 를 만든다.

### Input / Output / Constraints

| 구분 | 내용 |
|---|---|
| Input | 일반 웹캠 (30fps, 720p 이상) + 스케치북/공책 페이지 위 손그림 |
| Output (런타임) | 라이브 카메라 뷰 + 월드 고정된 2.5D 딱지 오버레이 |
| Output (자산) | `texture.png`, `mask.png`, `char_cfg.yaml` (AnimatedDrawings 호환) |
| Constraint | Python 3.10+, 일반 노트북 GPU (없어도 CPU 로 동작 가능), 오프라인 |

### Stage 1 (이 문서 범위) = "AR 딱지화만"
1. 카메라 프레임 캡처
2. 매 프레임 그림 후보 감지 (YOLO + 컨투어)
3. 클릭 시 MobileSAM point-prompt 로 누끼
4. Homography 기반 종이 앵커에 딱지 월드 고정
5. warpPerspective + 그림자 + 알파 합성
6. Stage 2 호환 자산 파일 저장

### Stage 2 (후속) — 명시적 비포함
- BVH 애니메이션 (AnimatedDrawings 의 움직임)
- Galaxy 테블릿 on-device 포팅 (ARCore + ONNX Runtime Mobile, 네이티브 Android)
- 다중 아이·다중 스케치북 동시 처리

### 비목표
- 네발짐승 측면 자세 (AnimatedDrawings 불가)
- 종이를 가리거나 프레임 밖에 나갔을 때 완벽한 추적 (homography 한계 수용)
- 100% 자동 분할 (SAM point-prompt 기반 반자동)
- 3D 모델 생성, depth 카메라 사용

---

## 사용자 시나리오 (Stage 1)

```
1. 아이가 스케치북에 캐릭터를 그림
2. 스케치북을 책상 위에 펴고 노트북 웹캠을 향함
3. 화면: 라이브 웹캠 뷰 + 감지된 그림 주변에 얇은 가이드 박스
4. 사용자가 마우스로 그림을 클릭
5. (200ms~1s) 클릭한 그림이 "슉" 하고 누끼되어 2.5D 딱지가 됨
6. 딱지는 원본 그림이 있던 종이 위치에 월드 고정 (= 카메라가 움직여도 같은 종이 위치에 붙어있음)
7. 다른 그림을 클릭하면 딱지가 추가됨 (다중)
8. R 키: 모든 딱지 리셋. S 키: 현재 딱지 자산을 `assets/captures/<timestamp>/` 에 저장
```

---

## 상태머신

```
        ┌───────────────────────────────────────┐
        │                                       │
  ┌─────▼─────┐  click   ┌──────────────┐  SAM done   ┌──────────┐
  │   SCAN    ├─────────▶│  EXTRACTING  ├────────────▶│   LIVE   │
  │ (후보박스) │           │ (백그라운드) │             │(딱지 합성)│
  └───────────┘          └──────────────┘             └────┬─────┘
        ▲                                                  │
        │                     R key                        │
        └──────────────────────────────────────────────────┘
                                click (while LIVE) ─┐
                                                    ▼
                                            (LIVE + EXTRACTING 중첩)
```

- SCAN: YOLO/컨투어 매 프레임 실행, 후보 박스 오버레이
- EXTRACTING: SAM 스레드 실행 중 (UI freeze 방지). 배경에서 다른 프레임 계속 렌더링됨
- LIVE: 생성된 Sticker 들을 homography anchor 로 월드 고정 합성
- LIVE 중에도 새 클릭 허용 → 새 SAM 작업이 병렬로 돌 수 있음 (최대 2개까지)

---

## 아키텍처 — 모듈 분리

```
┌──────────────────────────────────────────────────────────────────┐
│                           app.py                                 │
│         (상태머신 + 이벤트 루프 + 키보드/마우스 핸들러)          │
└──┬────────┬─────────┬────────┬─────────┬────────┬─────────┬──────┘
   │        │         │        │         │        │         │
   ▼        ▼         ▼        ▼         ▼        ▼         ▼
capture/  detect/  extract/  track/   render/  export/   config.py
 (cam)   (후보)    (SAM)    (앵커)   (딱지)  (자산저장)
```

각 모듈의 책임과 경계:

### `capture/camera.py`
- 책임: OpenCV VideoCapture 추상화, 매 프레임 BGR ndarray 제공
- 인터페이스: `Camera.read() -> np.ndarray (H,W,3)`, `Camera.release()`
- 의존: `cv2`

### `detect/candidate_detector.py`
- 책임: 매 프레임에서 "그림 후보 박스" 반환
- 1차: **YOLO26n** (`yolo26n.pt`, COCO 80 클래스 중 person + 주요 동물) 추론
  - 로드: `from ultralytics import YOLO; model = YOLO("yolo26n.pt")`
- 2차 폴백: 흰 배경에서 dark contour 탐지 (findContours + minAreaRect)
- 인터페이스: `CandidateDetector.detect(frame) -> List[CandidateBox]`
- 의존: `ultralytics` (최신), `cv2`

### `extract/segmenter.py`
- 책임: 클릭 좌표 + 프레임 → MobileSAM point-prompt 분할
- **중요**: 백그라운드 스레드에서 실행 (UI freeze 방지)
- 인터페이스: `Segmenter.segment_async(frame, (x,y)) -> Future[StickerAsset]`
- 후처리: 마스크 morphology (opening/closing), contour smoothing
- 의존: `mobile_sam`, `torch`, `cv2`

### `track/world_anchor.py` (추상) + `track/homography_anchor.py` (구현)
- 책임: "월드 위치 고정" 추상화. Stage 2 에선 ARCoreAnchor 구현체로 교체
- 추상 인터페이스:
  ```python
  class WorldAnchor(Protocol):
      def initialize(self, frame, sticker_region: np.ndarray) -> None
      def update(self, frame) -> AnchorState  # homography | None, confidence
      def is_lost(self) -> bool
  ```
- `HomographyAnchor` 구현:
  - 초기화 시 sticker_region 주변 마진에서 ORB/AKAZE keypoints 저장
  - 매 프레임 BFMatcher + ratio test + `cv2.findHomography(RANSAC)`
  - inlier ratio < 임계값 or 매칭 keypoint < N → `is_lost = True`
  - N 프레임 연속 lost → 딱지 숨김 (hard fail)
- 의존: `cv2`

### `render/sticker.py` (데이터 클래스)
```python
@dataclass
class Sticker:
    id: str                        # uuid
    texture: np.ndarray            # RGBA, 누끼된 그림
    mask: np.ndarray               # 이진 마스크
    anchor: WorldAnchor            # homography anchor
    local_offset: np.ndarray       # 앵커 기준 상대 위치 (2D 평면)
    created_at: float
```

### `render/tilt_renderer.py`
- 책임: Sticker + AnchorState → 현재 프레임에 그려질 warped texture
- 처리: homography 로부터 tilt 각도 추출 → warpPerspective → shadow 오프셋 합성 (Gaussian blur + dark alpha)
- 인터페이스: `TiltRenderer.render(sticker, anchor_state) -> np.ndarray (RGBA)`

### `render/compositor.py`
- 책임: 카메라 프레임 + 활성 Sticker 리스트 → 최종 출력 프레임
- 알파 블렌딩, 뎁스 정렬 (created_at 순)
- 인터페이스: `Compositor.compose(frame, stickers) -> np.ndarray (BGR)`

### `export/animated_drawings.py`
- 책임: Stage 2 호환 포맷으로 자산 저장
- 출력:
  ```
  assets/captures/<timestamp>/
    texture.png         # 누끼된 그림 (투명 배경)
    mask.png            # 이진 마스크
    char_cfg.yaml       # AnimatedDrawings 설정 스텁 (skeleton/joints 는 Stage 2 에서 추가)
  ```

### `app.py`
- 책임: 메인 루프, 상태 관리, UI 렌더링, 키/마우스 이벤트
- OpenCV 윈도우 + `setMouseCallback` 으로 클릭 수신
- 키: R(리셋), S(저장), Q(종료)

### `main.py`
- `python main.py --camera 0 --model-dir ./models` 같은 엔트리
- config.py 로부터 설정 로드

---

## 핵심 데이터 타입 (모듈 간 계약)

```python
@dataclass
class CandidateBox:
    x: int; y: int; w: int; h: int
    confidence: float
    source: Literal['yolo', 'contour']

@dataclass
class AnchorState:
    homography: Optional[np.ndarray]   # 3x3, None if lost
    confidence: float                  # 0.0 ~ 1.0
    lost: bool

@dataclass
class StickerAsset:   # Segmenter 출력
    texture_rgba: np.ndarray
    mask_u8: np.ndarray
    source_region: Tuple[int,int,int,int]  # 원본 프레임 내 bbox
```

Stage 2 로 넘어갈 때 이 데이터 클래스들은 **JSON + PNG 로 직렬화** 가능해야 함. WorldAnchor 만 구현체 교체.

---

## 월드 앵커 전략 (Homography 상세)

1. **초기화** (SAM 완료 직후)
   - SAM 이 준 source_region bbox 의 4 코너 저장
   - bbox 를 20~30% 확장한 영역에서 ORB keypoints 추출 (최대 500개)
   - Keypoints + descriptors 저장
2. **매 프레임 업데이트**
   - 현재 프레임에서 같은 ORB 추출
   - BFMatcher + Lowe ratio test (0.75)
   - inlier ≥ 10 + inlier ratio ≥ 0.3 → `findHomography(RANSAC)`
   - 실패 시 직전 homography 유지, `lost_frames++`
   - `lost_frames >= 15` (~0.5s @ 30fps) → `is_lost = True`, 딱지 숨김
3. **재획득**
   - lost 상태에서도 주기적 (5프레임마다) 매칭 재시도
   - 재획득 성공 시 자연스럽게 복귀

트레이드오프 수용:
- 종이 일부 가림 → 일시적 lost, 괜찮음
- 각도 70° 이상 틸트 → lost 빈번, 사용자 교육으로 대응
- 공책 라인 → keypoints 가 라인을 타고 형성되어 오히려 robust 할 수도 (라인은 안정적 feature)

---

## 딱지 시각 효과 (Tilt Render)

1. Homography `H` 로부터 4코너 투영 → 현재 프레임 내 딱지 영역
2. Sticker.texture 를 `warpPerspective(H)` 로 변환
3. 그림자:
   - 변환된 마스크를 (offset_x, offset_y) = (5, 8) 만큼 shift + Gaussian blur (sigma=4) + alpha 0.4 검정
   - 딱지 아래에 먼저 합성
4. 최종 알파 블렌딩으로 프레임에 합성

---

## 폴더/파일 구조 (제안)

```
AR_book/<새폴더>/
├── CLAUDE.md              # 이 프로젝트 전용 규칙
├── README.md
├── requirements.txt
├── config.py              # 전역 설정
├── main.py                # 엔트리
├── app.py                 # 상태머신 + UI 루프
│
├── capture/
│   ├── __init__.py
│   └── camera.py
├── detect/
│   ├── __init__.py
│   └── candidate_detector.py
├── extract/
│   ├── __init__.py
│   └── segmenter.py
├── track/
│   ├── __init__.py
│   ├── world_anchor.py         # Protocol / ABC
│   └── homography_anchor.py    # Stage 1 구현
├── render/
│   ├── __init__.py
│   ├── sticker.py              # 데이터 클래스
│   ├── tilt_renderer.py
│   └── compositor.py
├── export/
│   ├── __init__.py
│   └── animated_drawings.py
│
├── models/
│   ├── yolo26n.pt               # detection (ultralytics 자동 다운로드 or 수동 배치)
│   └── mobile_sam.pt            # symlink or copy from LivingDrawing/
│
├── assets/
│   ├── captures/                # 저장된 딱지 자산
│   ├── samples/                 # 테스트용 샘플 이미지
│   └── debug/                   # 디버그 중간 이미지
│
└── tests/
    ├── test_homography_anchor.py
    ├── test_segmenter.py
    └── fixtures/                # 고정 입력 이미지
```

**새 폴더 이름 후보** (본인 선택):
- `AR_book/stickerbook/` — 짧고 컨셉 직관적
- `AR_book/ar-2p5d/` — 기존 `drawing-2p5d` 와 구분, 의미 명확
- `AR_book/livingbook/` — LivingDrawing 연장선
- 또는 본인 선호 이름

---

## 의존성

```
# requirements.txt
opencv-python>=4.9
ultralytics                # 최신 (YOLO26 지원 버전)
mobile-sam @ git+https://github.com/ChaoningZhang/MobileSAM.git
torch>=2.1
numpy>=1.26
pyyaml>=6.0
```

모델 자산:
- `yolo26n.pt` — detection 용. `YOLO("yolo26n.pt")` 호출 시 ultralytics 가 자동 다운로드. 또는 수동 배치.
- 참고: `yolo26n-cls.pt` (classification 변형) 가 `/home/ingon/AR_book/` 루트에 이미 있으나, 본 파이프라인은 detection 변형(`yolo26n.pt`) 사용
- `mobile_sam.pt` (~39MB) — `LivingDrawing/` 에서 symlink/복사 가능
- AnimatedDrawings 는 Stage 2 에서 별도 설치 (Stage 1 은 포맷 호환만 유지)

---

## Stage 2 포팅 전략 (Android Native + ARCore + ONNX)

| Stage 1 (Python PC) | Stage 2 (Android native on Galaxy) |
|---|---|
| `capture/camera.py` | Android `Camera2` / `CameraX` + ARCore `Frame` |
| `detect/candidate_detector.py` | ONNX-변환된 YOLO26n + `onnxruntime-mobile` (NNAPI/GPU delegate) |
| `extract/segmenter.py` | MobileSAM ONNX + `onnxruntime-mobile` |
| `track/homography_anchor.py` | **ARCore `Anchor` API 로 교체** (WorldAnchor 인터페이스만 일치) |
| `render/tilt_renderer.py` + `M8 billboard` | OpenGL ES / Filament 큐보이드 스프라이트 |
| `render/compositor.py` | ARCore 배경 텍스처 + 전경 draw call |
| `export/animated_drawings.py` | 그대로 재사용 (사전 처리 또는 서버 백엔드) |

**포팅 핵심**: 데이터 계약 (Sticker JSON + PNG) 을 Stage 2 에서 **그대로 로드** 하면 됨. 즉 Stage 1 자산으로 Stage 2 앱을 로컬에서 테스트 가능.

---

## 리스크 & 대응 (M7 업데이트: 실제 관찰 결과 반영)

| # | 리스크 | 실제 관찰 | 대응 결과 |
|---|---|---|---|
| R1 | MobileSAM 이 "아이 그림" 도메인에서 성능 저하 | **경미** — SAM 이 얼굴·상체·연필 스케치 모두 적절히 분할. 가끔 페이지 배경까지 포함하는 경우 있으나 point-prompt 위치 변경으로 해결 | 별도 폴백 미구현 (SAM 성능 충분) |
| R2 | Homography 추적 실패율 | **관찰됨** — 종이 프레임 밖 / 빠른 이동 / 심한 각도에서 lost. 복귀 안됨 → M5.5 로 5프레임 주기 재획득 루프 추가 | ✅ M5.5 재획득 로직으로 해결 |
| R3 | SAM 인퍼런스가 CPU 에서 1초 이상 | **관찰됨** — 합성 프레임 E2E 에서 754ms 측정. 실기기에서도 유사. 클릭 후 체감은 수용 가능 | 비동기 스레드로 UI freeze 방지 (M3) |
| R4 | 공책 라인이 SAM 이나 homography 를 혼란 | **경미** — 공책/사무실 배경에서 모두 동작 확인. 라인이 오히려 ORB feature 로 기여해 tracking 안정화 | 대응 불필요 |
| R5 | AnimatedDrawings char_cfg.yaml 포맷이 Stage 1 에서 완전히 채워지지 않음 | **의도된 상태** — skeleton: [] 로 빈 스텁 저장. Stage 2 에서 pose/rigging 추가 | M7.5 에서 실제 로드 검증 예정 |
| R6 | Python threading 과 GIL | **문제 없음** — PyTorch 인퍼런스는 GIL 밖에서 돌아 UI freeze 없음. `max_workers=1` ThreadPoolExecutor 로 충분 | multiprocessing 전환 불필요 |
| **R7 (신규)** | **SAM 입력이 오버레이 오염** — 초기 구현에서 contour 박스가 그려진 프레임이 SAM 에 전달되어 texture.png 에 파란 박스 박힘 | **발견·수정됨** (M6.1) | `raw` 와 `display` 프레임 분리. `self._current_frame` 은 raw 만 참조, 오버레이는 display 전용 |

---

## 검증 계획

### Unit
- `test_homography_anchor.py`: 고정된 두 이미지 간 homography 계산 정확도
- `test_segmenter.py`: 샘플 클릭 좌표에서 마스크 IoU > 0.7
- `test_candidate_detector.py`: YOLO 실패 + contour 폴백 경로

### Integration
- 녹화된 웹캠 비디오 (`assets/samples/session1.mp4`) + 스크립트 클릭 이벤트 → 전체 파이프라인 실행 + 출력 프레임 저장

### E2E
- 실제 웹캠 + 손그림 → 라이브 데모
- 체크리스트:
  - [ ] 그림 인식 박스가 라이브 표시되는가
  - [ ] 클릭 후 1초 이내에 딱지가 나타나는가
  - [ ] 카메라를 좌우로 움직일 때 딱지가 종이에 붙어있는 것처럼 보이는가
  - [ ] 종이를 프레임 밖으로 내보내면 딱지가 사라지고, 되돌리면 복귀하는가
  - [ ] 다중 딱지가 각자 독립적으로 앵커링되는가
  - [ ] S 키로 저장한 자산이 AnimatedDrawings 로컬 데모에 입력되는가

### Performance

**목표**: SCAN 상태 ≥25fps, LIVE 상태 (딱지 3개) ≥20fps, SAM latency p95 ≤ 1.2s

**실측 (M7, WSL2 Ubuntu 24.04 / CPU, 2026-04-21)** — 딱지 3개, last 120 frames:

| Phase | avg | p95 | peak |
|---|---|---|---|
| capture | 0.5ms | 0.7ms | 1.2ms |
| poll | 0.0ms | 0.0ms | 0.0ms |
| detect (YOLO + contour) | 16.2ms | 19.4ms | 20.2ms |
| track_render (3 stickers) | 25.0ms | 33.6ms | 34.7ms |
| **iter (total)** | **41.9ms (≈24fps)** | 52.0ms (≈19fps) | 54.2ms |

**판정**: SCAN 목표 ✅ 여유. LIVE@3 목표 🟡 경계 (avg 24fps / p95 19fps). SAM p95 별도 측정치 ≈1s ✅.

**병목**: `track_render` 가 딱지 수에 선형 증가 (딱지당 ~8.3ms, 주로 매프레임 ORB 재검출). 5개 이상이면 20fps 밑으로 떨어질 가능성.

**M7.1 최적화 후보 (skip, M7 범위 밖)**: 공유 ORB (프레임당 1회 검출 후 전 anchor 공유), detector 프레임 스킵 (후보 박스는 시각 힌트 목적).

---

## 마일스톤

| M | 내용 | 검증 기준 | Status |
|---|---|---|---|
| M1 | 폴더 스캐폴드 + `capture` + 최소 UI 루프 | 웹캠 라이브 윈도우가 뜨고 Q 로 종료 | ✅ |
| M2 | `detect` (YOLO + 컨투어) + 후보 박스 시각화 | 스케치북에 그린 그림 주변에 박스 표시 | ✅ |
| M2.5 | Otsu → Adaptive threshold + morph closing | 연필 스케치가 다크 배경과 공존해도 감지 | ✅ (R4 관련 튜닝) |
| M3 | `extract` (SAM point-prompt) + 비동기 | 클릭 시 1~2초 뒤 누끼된 texture.png 저장 | ✅ |
| M4 | `render/tilt_renderer` + 정적 합성 | 생성된 딱지가 화면 고정 좌표에 warped 로 표시 | ✅ |
| M5 | `track/homography_anchor` + 월드 고정 | 카메라 이동 시 딱지가 종이 위치에 남아있음 | ✅ |
| M5.5 | 재획득 루프 (5프레임 주기 ORB 재시도) | 종이 다시 들이면 딱지 복귀 | ✅ (DESIGN 원안 누락 버그 수정) |
| M6 | 다중 딱지 + 리셋/저장 + `export` | E2E 체크리스트 통과 | ✅ |
| M6.1 | `raw`/`display` 프레임 분리 (R7 수정) | 저장된 texture.png 에 contour 박스 없음 | ✅ |
| M7 | 성능 측정 + 리스크 문서화 + README 정리 | Stage 2 포팅 준비 완료 | ✅ |
| M7.5 | AnimatedDrawings 로컬 라운드트립 검증 | `texture.png` 입력으로 애니메이션 비디오 생성 | ✅ (TorchServe via Docker, dab GIF 생성 확인) |
| M8 / M8.1 | Popup billboard 렌더 (정지 딱지) | 종이 기울여도 캐릭터 벌떡 서있음 | ✅ |
| M9 | AD 라이브 통합 (춤추는 딱지) | 클릭 → 7–12초 → 종이 위에서 dab 춤 | ⏳ 예정 |

각 마일스톤은 독립 commit / PR 단위. M1~M3 가 **검증의 핵심** (여기서 "잘못된 방향" 이었는지 일찍 알 수 있음).

## 교훈 (M1~M6.1 완료 후 돌아보기)

- **프레임 가변성 주의**: OpenCV 프레임 배열은 `draw_*` 호출 시 in-place 변형된다. ML 입력·시각화 출력을 엄격히 분리 (`raw` / `display`) 하지 않으면 저장된 자산이 오염됨 — M6.1 에서 뒤늦게 발견, R7 로 기록.
- **재획득은 설계 문서에만 있고 구현에서 누락되기 쉬움**: M5 구현 시 DESIGN.md 의 "lost 상태에서도 재시도" 문장이 코드에 안 반영 — 실기기 테스트에서 사용자가 발견 후 M5.5 로 수정. 교훈: **설계 체크리스트 마지막 passthrough 가 필요**.
- **YOLO base model 은 아이 그림에 약함**: COCO 분포 밖이라 `person`/동물 클래스 hit rate 낮음. 현재는 contour + SAM point-prompt 조합으로 커버. 나중에 sketch-domain 파인튜닝 모델 고려.
- **Adaptive threshold > Otsu**: 조명/배경 다양성 높은 실사 환경에선 local threshold 가 필수. Otsu 는 합성 2-modal 시나리오에만 유효.

---

## 선결 TODO (초기 플랜 종료 시점 기록 — 모두 완료됨)

1. ~~`AR_book/claude.md` 정리~~ ✅ 완료
2. ~~새 폴더 이름 확정~~ ✅ `stickerbook`
3. ~~새 폴더 스캐폴드~~ ✅
4. ~~`models/` YOLO + MobileSAM 배치~~ ✅

---

## 검토 시 확인할 포인트 (사용자용)

- 모듈 분리가 너무 많은지 (과도하다면 `capture + detect` 통합, `render + compositor` 통합 고려)
- `export/animated_drawings.py` 는 M6 에서 시작해도 괜찮은지, 아니면 M3 부터 바로 만들지
- 상태머신에서 "LIVE + EXTRACTING 중첩" 을 허용할지 (UX 실험 필요)
- 마커 기반 백업 구현을 미래 옵션으로 남겨둘지 (homography 실패 시 사용자가 ArUco 스티커를 붙일 수 있는 fallback)

---

## M9 — AnimatedDrawings 라이브 통합 (춤추는 딱지)

> 2026-04-24 브레인스토밍 결정. M7.5 의 수동 검증을 라이브 앱에 통합한다.

### 결정 사항 (Q/A 요약)

| 질문 | 결정 |
|---|---|
| 트리거 | 클릭 시 자동으로 AD 파이프라인 시작, 7–12초 스피너 후 춤 |
| AD 호출 | `subprocess.run(["python", "image_to_animation.py", ...])` shell out |
| 런타임 | Docker 없이 **TorchServe 네이티브** (pip + openjdk-17) |
| 실패 처리 | 조용히 정지 딱지 유지, 로그에만 기록 (토스트 없음) |
| 애니메이션 | **dab 고정** (zombie/wave 등은 후속) |
| 큐잉 | 단일 워커 `ThreadPoolExecutor(max_workers=1)` — 다중 클릭 순차 처리 |

### 상태머신 확장

```
SCAN → click → SEGMENTING → LIVE (정지 딱지)
                                ↓ (자동, 백그라운드)
                          Sticker.animation_state:
                              STATIC → PREPARING → ANIMATED  (성공 시)
                                              ↘ FAILED        (실패 시, STATIC 유지)
```

- LIVE 는 앱 전역 상태로 유지. ANIMATING 은 **Sticker 단위 속성**.
- 딱지 3개가 각자 다른 animation_state 를 가질 수 있음.
- 정지 딱지는 M8.1 billboard 그대로 동작. 애니메이션 성공한 딱지만 AnimatedStickerRenderer 로 전환.

### Sticker 데이터 확장

```python
class AnimationState(Enum):
    STATIC      # 처음 생성
    PREPARING   # AD 작업 진행 중
    ANIMATED    # video.mp4 준비됨, 재생 중
    FAILED      # AD 실패, STATIC 유지

@dataclass
class Sticker:
    # 기존 필드 ...
    animation_state: AnimationState = AnimationState.STATIC
    animation_video_path: Optional[Path] = None
    animation_frame_index: int = 0
    animation_started_at: Optional[float] = None
```

### 새 모듈

```
stickerbook/
├── animate/                              # ← 신규
│   ├── __init__.py
│   ├── torchserve_runtime.py             # TorchServe 네이티브 기동/정지
│   ├── animated_drawings_runner.py       # AD 스크립트 shell out + 결과 파싱
│   └── animation_worker.py               # 단일 워커 큐 (Future 반환)
├── render/
│   ├── animated_sticker_renderer.py      # ← 신규: video 프레임 재생 + billboard warp
│   ├── spinner_overlay.py                # ← 신규: PREPARING 시각화
│   └── ...existing files
```

### 실행 흐름 (클릭 한 번)

```
click (x, y)
   │
   ▼
Segmenter (1–2초)                 ──▶ StickerAsset + HomographyAnchor
   │
   ▼ (자동 이어서)
AnimationWorker.submit(sticker)   ──▶ Future[AnimationResult]
   │                                       │
   │                                       ▼
   │      ┌──────────────────────────────────────────────────────┐
   │      │ 1. texture_bgra → 흰 배경 합성 → /tmp/<uuid>/input.png │
   │      │ 2. subprocess.run([python, image_to_animation.py,     │
   │      │    input.png, out_dir, motion_cfg=dab])               │
   │      │ 3. out_dir/video.mp4 존재? → success 판정             │
   │      └──────────────────────────────────────────────────────┘
   │
   ▼
sticker.animation_state = PREPARING  (spinner overlay 표시)
   │
   ▼ Future.done()
   ├── success: animation_state = ANIMATED, video_path 설정
   └── failure: animation_state = FAILED (정지 딱지 유지)
   │
   ▼ (LIVE 합성)
compositor → 각 sticker 에 대해:
   - STATIC / FAILED: 기존 billboard
   - PREPARING: 정지 billboard + spinner
   - ANIMATED: AnimatedStickerRenderer.render_on(frame, sticker, anchor)
```

### TorchServeRuntime 생명주기

`app.py` 의 `__enter__` / `__exit__` 에서 관리:

```python
self._torchserve = TorchServeRuntime(
    models_dir="~/AR_book/AnimatedDrawings/torchserve/model-store",
    config_path="/tmp/ts_config.properties",   # M7.5 검증: default_workers_per_model=1
)
self._torchserve.start()                        # health check 통과까지 blocking (~5초)
# ... 앱 실행 ...
self._torchserve.stop()                         # torchserve --stop
```

첫 실행 시 `java -version` + `which torchserve` 를 사전 체크. 실패 시 친절한 에러:
```
[TorchServeRuntime] java 17+ not found. Install: sudo apt install openjdk-17-jre-headless
[TorchServeRuntime] torchserve not found. Install: pip install torchserve torch-model-archiver
```

### 데이터 계약

```python
# animate/animated_drawings_runner.py
@dataclass
class AnimationResult:
    success: bool
    video_path: Optional[Path]      # video.mp4 (or PNG sequence dir — M9.1 에서 결정)
    char_cfg_path: Optional[Path]   # AD char_cfg.yaml (디버그/Stage 2 참고)
    duration_sec: float              # AD 호출 실측 시간 (perf tracker)
    error: Optional[str]

def run_animated_drawings(
    texture_bgra: np.ndarray,       # StickerAsset.texture_bgra
    motion: str = "dab",
    ad_repo_path: Path,
    work_dir: Path,                  # /tmp/stickerbook_ad/<uuid>/
    timeout_sec: float = 30.0,
) -> AnimationResult: ...
```

### 프레임 재생

`render/animated_sticker_renderer.py`:

```python
class AnimatedStickerRenderer:
    def __init__(self, video_path: Path):
        self._cap = cv2.VideoCapture(str(video_path))
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def next_frame_bgra(self) -> np.ndarray:
        ok, frame_bgr = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame_bgr = self._cap.read()
        return self._bgr_to_bgra_with_chroma_key(frame_bgr)

    def render_on(self, frame, sticker, anchor_state):
        # 기존 billboard_corners_2d + warpPerspective 재사용
        ...
```

### 마일스톤

| M | 내용 | 검증 기준 | 예상 |
|---|---|---|---|
| M9.1 | **출력 포맷 검증** (R8 해소): AD CLI 수동 → video.mp4 vs gif / 알파 채널 / 프레임 추출 방법 결정 | 재생 가능한 프레임 소스 확정 | 0.5일 |
| M9.2 | **TorchServeRuntime** + 네이티브 설치: `animate/torchserve_runtime.py` + Java 체크 + health check | 앱 기동/종료 시 TorchServe 자동. Docker 없이 M7.5 동일 결과 | 1일 |
| M9.3 | **AnimatedDrawingsRunner**: StickerAsset → AnimationResult (흰배경 합성, subprocess, 결과 파싱) | 테스트 입력 → video 생성. 여자아이류 실패 케이스 graceful | 1일 |
| M9.4 | **AnimationWorker** 단일 큐 + `Sticker.animation_state` 통합 | 3번 클릭 → 순차 처리, 큐 상태 로그 | 0.5일 |
| M9.5 | **AnimatedStickerRenderer** + compositor 통합 + SpinnerOverlay | 클릭 → 정지 딱지 → 스피너 → 춤. 카메라 이동해도 billboard 앵커 유지 | 1일 |
| M9.6 | E2E + perf report (`animation_success_rate`, `animation_latency_p50/p95`) + README Docker 언급 제거 | 완료 기준 충족 | 0.5일 |

**총 4.5일 추정**. M9.1 이 가장 큰 리스크 — 결과에 따라 M9.5 프레임 읽기 전략 변경 가능.

### 테스트 전략 (TDD 유지)

```
tests/
├── test_torchserve_runtime.py           (integration, @slow)
│     test_start_stop_lifecycle
│     test_health_check_detects_not_ready
│     test_raises_helpful_error_if_java_missing
├── test_animated_drawings_runner.py     (integration, @slow)
│     test_runs_on_valid_human_sticker_returns_mp4
│     test_returns_failed_result_on_bunched_joints
│     test_handles_subprocess_nonzero_exit
├── test_animation_worker.py             (unit)
│     test_single_worker_processes_jobs_sequentially
│     test_future_resolves_with_animation_result
│     test_worker_survives_job_exception
├── test_animated_sticker_renderer.py    (unit)
│     test_frame_loops_back_to_zero_at_end
│     test_render_on_uses_current_anchor_homography
│     test_bgra_conversion_preserves_alpha_if_available
└── test_app_animation_state_transition.py (unit)
      test_click_transitions_sticker_to_preparing
      test_animation_success_transitions_to_animated
      test_animation_failure_keeps_sticker_static
```

### 리스크 추가

| # | 리스크 | 영향 | 대응 |
|---|---|---|---|
| R8 | AD 출력이 MP4 면 알파 채널 없음 → 배경 검은 사각형 | 렌더 깨짐 | M9.1 에서 검증. PNG 시퀀스 추출로 우회 가능 |
| R9 | TorchServe 네이티브 설치가 conda/venv 혼재 환경에서 꼬임 | 설치 실패 | 명시적 가이드 + health check 에서 친절한 에러 |
| R10 | Java 17 가 WSL 에 없음 | 기동 실패 | `TorchServeRuntime.start()` 가 `java -version` 먼저 체크 |
| R11 | 7–12초 이상 걸림 (저사양 노트북) | 대기 체감 악화 | 30초 timeout 초과 시 FAILED. 스피너에 경과 시간 |
| R12 | 딱지 여러 개 큐잉 시 마지막은 30초+ 대기 | UX 저하 | 스피너에 "N 번째 대기 중" 표시 |
| R13 | joint 가 뭉치면 AD 는 "성공" 반환하지만 애니메이션 이상함 | 품질 저하 | M9.3 에서 joint spread sanity check → 미달 시 FAILED |

### Stage 2 (갤탭) 호환성

- **제거 대상 (갤탭 못 감)**: TorchServe, Java, subprocess, Docker
- **유지 대상**: `AnimationResult`, `AnimationWorker` 인터페이스, AnimatedStickerRenderer (프레임 소스만 교체)
- 포팅 시 내부 구현만 ONNX + onnxruntime-mobile 로 교체, 상위 계약은 유지

### 비포함 (명시)

- zombie/wave/run 등 다른 모션
- 사용자가 애니메이션 선택하는 UI
- 애니메이션 속도 조절
- 춤추는 딱지의 MP4 저장 (현재 S 키는 정지 텍스처만 저장)
- 실시간 포즈 재인식 (한 번 검출된 스켈레톤 고정)

