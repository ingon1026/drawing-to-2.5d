# drawing-to-2.5d

> 종이 위 손그림을 카메라로 촬영하면, 그림만 오려내고
> 깊이(depth)와 표면(normal) 정보를 입혀 **2.5D asset**으로 만들어주는 파이프라인

---

## 뭘 하는 건가?

```
카메라로 종이 비추기
 → 그림마다 실시간 색상 오버레이
 → 원하는 그림 터치/클릭
 → 누끼 + 깊이맵 + 노멀맵 자동 생성
 → 2.5D 뷰어에서 바로 확인
```

Unity/AR에서 평면 그림에 빛과 그림자가 반응하는 **2.5D 효과**를 만들기 위한 asset 생성 도구.

---

## 세그멘테이션 방식

### Contour 기반 (기본 — 실시간)
카메라 프리뷰와 실제 세그멘테이션 모두 contour 방식을 사용.
프리뷰에서 보이는 그림이 그대로 추출 결과가 되므로 실패가 없음.

```
grayscale → erosion(선 두껍게) → adaptive threshold → contour 검출 → mask
```

### magic_touch ML (CLI 전용)
`pipeline.py` CLI에서 좌표 지정 방식으로 사용.
MediaPipe Interactive Segmenter (TFLite 직접 추론).

---

## 사용 모델

| 모델 | 역할 | 사용처 |
|------|------|--------|
| `Intel/dpt-hybrid-midas` | 깊이 추정 (고품질) | PC PyTorch 백엔드 |
| `midas_v21_small.tflite` | 깊이 추정 (경량) | PC TFLite / Android |
| `magic_touch.tflite` | 누끼 세그멘테이션 | CLI(pipeline.py)만 사용 |

---

## 구조

```
Python (PC)
├── config.py           # 설정값
├── normalize.py        # 입력 전처리 (white balance)
├── auto_segment.py     # contour 기반 세그멘테이션 (실시간)
├── segment.py          # magic_touch ML 세그멘테이션 (CLI용)
├── postprocess.py      # 마스크 후처리
├── depth.py            # depth + normal map 생성
├── export.py           # PNG 내보내기
├── live_demo.py        # 라이브 데모 (카메라 → contour → viewer)
├── pipeline.py         # CLI 진입점 (magic_touch 사용)
├── viewer.py           # 2.5D 뷰어 (Pygame, 독립 실행)
└── export_tflite.py    # 모델 변환 도구

Android (Kotlin)
├── MainActivity.kt         # 카메라 + 터치 → contour 추출 → bounce
├── ContourAnalyzer.kt      # contour 감지 + 히트테스트 + mask 생성
├── ContourOverlay.kt       # 실시간 contour 오버레이
├── BounceView.kt           # 2.5D bounce 애니메이션
├── DepthHelper.kt          # MiDaS TFLite depth 추론
└── SegmentationHelper.kt   # magic_touch 래퍼 (현재 미사용)
```

---

## 실행

### 라이브 데모 (PC)
```bash
python3 live_demo.py
```
카메라 → 그림 클릭 → 2~3초 처리 → 2.5D 뷰어

| 키 | 동작 |
|----|------|
| 클릭 | 오브젝트 선택 |
| `SPACE` | 다시 떨어뜨리기 (뷰어) |
| `R` | 카메라로 돌아가기 |
| `Q` | 종료 |

### CLI (PC)
```bash
python3 pipeline.py --input image.jpg --x 0.5 --y 0.4 --debug
```

### Android
Android Studio에서 `android/` 프로젝트 빌드 후 태블릿 설치.

---

## 출력물

| 파일 | 설명 |
|------|------|
| `object.png` | 투명 배경 누끼 (BGRA) |
| `mask.png` | 이진 마스크 |
| `depth.png` | 깊이맵 (0=far, 255=near) |
| `normal.png` | 노멀맵 (RGB) |

---

## 설치

```bash
pip install -r requirements.txt

# RealSense 사용 시
pip install pyrealsense2

# PyTorch depth 백엔드 사용 시
pip install torch transformers
```

---

## 환경

Python 3.12 / Ubuntu 24.04 (WSL2) / CUDA 12.8 / Intel RealSense D455 (선택)
Android: minSdk 26 / CameraX / TFLite / OpenCV
