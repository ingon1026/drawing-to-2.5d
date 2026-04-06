# drawing-to-2.5d

종이에 그린 그림을 카메라로 촬영하면, 그림만 오려내고(누끼) 깊이/표면 정보를 자동 생성하여 **2.5D asset**으로 변환하는 파이프라인.

Unity나 AR 환경에서 평면 그림을 입체적으로 렌더링하기 위한 중간 asset(depth map, normal map)을 생성하는 것이 목표.

---

## 개요

### 이게 뭘 하는 건가?

1. 카메라로 종이를 비추면, 그림마다 **실시간으로 색상 오버레이**가 표시됨
2. 원하는 그림을 **클릭**하면 해당 영역만 깔끔하게 오려냄
3. 오려낸 그림에 **깊이 정보**(depth map)와 **표면 방향 정보**(normal map)를 자동 생성
4. 결과물을 **2.5D 뷰어**에서 통통 튀는 애니메이션으로 바로 확인 가능

### 왜 필요한가?

이 4개의 asset을 Unity/AR에 넣으면, 평면 그림인데 **빛도 받고 그림자도 생기는 2.5D 효과**를 만들 수 있음.
아이가 그린 그림이 종이에서 튀어나와서 살아 움직이는 느낌을 구현하기 위한 첫 단계.

---

## 파이프라인

```
카메라 입력 (Intel RealSense D455 / 일반 웹캠)
  │
  ▼
실시간 오브젝트 프리뷰
  │  Adaptive Threshold + Contour 기반, 30fps
  │  각 오브젝트별 색상 오버레이 표시
  │  마우스 hover 시 해당 영역 하이라이트
  │
  ▼ (클릭)
입력 전처리
  │  Gray-world 화이트밸런스로 조명 편차 보정
  │
  ▼
누끼 추출 (Segmentation)
  │  MediaPipe magic_touch 모델 (TFLite 직접 추론)
  │  클릭 지점 중심 Gaussian heatmap → 모델 입력의 4번째 채널
  │  출력: 픽셀별 foreground confidence → 이진 마스크
  │
  ▼
마스크 후처리
  │  Morphological Close → 선 사이 빈틈 채움
  │  Morphological Open → 노이즈 제거
  │  최대 연결 컴포넌트만 유지
  │  Flood Fill로 내부 구멍 채움
  │  Gaussian Blur + 재이진화로 엣지 스무딩
  │
  ▼
Depth 추정
  │  MiDaS (DPT-hybrid) 모델
  │  단안 이미지에서 상대 깊이값 추정
  │  마스크 영역 내에서 [0, 1] 정규화
  │
  ▼
Normal Map 생성
  │  Depth map에 Sobel gradient 적용
  │  법선 벡터 (-dx, -dy, 1) 정규화 → RGB 인코딩
  │  평면 = (128, 128, 255), 기울기 = 색상 변화
  │
  ▼
Asset 내보내기 + 2.5D 뷰어
```

---

## 사용 모델

| 모델 | 역할 | 상세 |
|------|------|------|
| **magic_touch.tflite** | 터치 기반 누끼 세그멘테이션 | MediaPipe Interactive Segmenter. 입력: [1,512,512,4] (RGB+heatmap). 첫 실행 시 자동 다운로드 (~12MB) |
| **Intel/dpt-hybrid-midas** | 단안 깊이 추정 | DPT + MiDaS hybrid backbone. HuggingFace에서 첫 실행 시 자동 다운로드/캐시 (~400MB) |
| **SAM2.1-hiera-tiny** | 정밀 auto-mask (선택) | Meta Segment Anything 2. 정확하지만 느림 (~1-2s/frame). 기본적으로 contour 방식 사용, 필요 시 전환 가능 |

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| 세그멘테이션 | MediaPipe Interactive Segmenter (TFLite 직접 추론) |
| 깊이 추정 | MiDaS / DPT (PyTorch, HuggingFace Transformers) |
| 실시간 프리뷰 | OpenCV (Adaptive Threshold + Contour Detection) |
| 카메라 입력 | pyrealsense2 (RealSense D455) / OpenCV VideoCapture (웹캠 fallback) |
| 이미지 처리 | OpenCV, NumPy |
| 2.5D 뷰어 UI | Pygame (bounce + sway + squash-stretch 애니메이션) |
| 정밀 auto-mask (선택) | SAM2 (Meta, PyTorch, CUDA) |

---

## 설치

```bash
# 기본 의존성
pip install -r requirements.txt

# RealSense 카메라 사용 시
pip install pyrealsense2

# SAM2 (선택 — 정밀 auto-mask가 필요한 경우만)
pip install sam2
mkdir -p models
wget -P models/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

> `magic_touch.tflite` (세그멘테이션)과 `dpt-hybrid-midas` (깊이 추정) 모델은 **첫 실행 시 자동 다운로드**됩니다.

---

## 실행 방법

### 라이브 데모 (카메라 → 프리뷰 → 클릭 → 2.5D 뷰어)

```bash
python3 live_demo.py
```

1. 카메라 화면이 뜨면 종이를 비춤
2. 그림마다 색상 오버레이가 실시간으로 표시됨
3. 원하는 그림 위에 마우스를 올리면 하이라이트
4. **클릭** → 2~3초 처리 → 2.5D 뷰어에서 결과 확인

**카메라 화면 조작:**
| 키 | 동작 |
|----|------|
| 마우스 클릭 | 오브젝트 선택 → 처리 시작 |
| `Q` / `ESC` | 종료 |

**뷰어 조작:**
| 키 | 동작 |
|----|------|
| `SPACE` | 다시 떨어뜨리기 (bounce) |
| `R` | 카메라로 돌아가기 (retake) |
| `Q` / `ESC` | 종료 |

### CLI 파이프라인 (저장된 이미지 기반)

```bash
python3 pipeline.py --input image.jpg --x 0.5 --y 0.4 --debug
```

| 인자 | 설명 |
|------|------|
| `--input`, `-i` | 입력 이미지 경로 (필수) |
| `--x`, `--y` | 정규화된 터치 좌표 0~1 (필수, 좌상단이 0,0) |
| `--output`, `-o` | 출력 디렉토리 (기본: `output/`) |
| `--debug`, `-d` | 디버그 오버레이 이미지 추가 생성 |
| `--normal-strength` | 노멀맵 강도 조절 (기본: 1.0) |

### 단독 뷰어 (기존 asset 확인용)

```bash
python3 viewer.py --image output/object.png --shadow
```

---

## 출력물

| 파일 | 설명 | Unity/AR 용도 |
|------|------|---------------|
| `object.png` | 투명 배경 누끼 (BGRA) | 메인 텍스처 (Sprite Renderer) |
| `mask.png` | 이진 마스크 (흰=그림, 검=배경) | 충돌 영역, 이펙트 범위 |
| `depth.png` | 깊이맵 (밝을수록 가까움) | Parallax offset, mesh displacement |
| `normal.png` | 노멀맵 (RGB, 중립=(128,128,255)) | Lit Shader의 Normal Map 슬롯 |

---

## 프로젝트 구조

```
├── config.py           # 전체 설정값 (모델 경로, 임계값, 커널 크기 등)
├── normalize.py        # 입력 전처리 (gray-world 화이트밸런스, shadow removal)
├── segment.py          # 누끼 추출 (TFLite 직접 추론, GPU 의존성 없음)
├── postprocess.py      # 마스크 후처리 (모폴로지, 최대 컴포넌트, hole fill)
├── depth.py            # MiDaS depth 추정 + Sobel 기반 normal map 생성
├── export.py           # mask/object/depth/normal PNG 내보내기
├── auto_segment.py     # 실시간 프리뷰 (contour 기반 + SAM2 백업)
├── pipeline.py         # CLI 파이프라인 진입점
├── live_demo.py        # 카메라 → 실시간 프리뷰 → 처리 → 뷰어 통합
├── viewer.py           # Pygame 기반 2.5D 뷰어 (bounce + squash 애니메이션)
└── requirements.txt    # 의존성
```

---

## 해결한 기술적 문제

| 문제 | 원인 | 해결 |
|------|------|------|
| MediaPipe segfault | Tasks API가 내부적으로 EGL/OpenGL 사용, WSL2에서 GPU 접근 실패 | TFLite Interpreter로 직접 추론, GPU 의존성 완전 우회 |
| RealSense RGB 안 잡힘 | V4L2로 열면 depth/IR 스트림이 잡힘 | `pyrealsense2` SDK로 RGB 스트림 직접 지정 |
| SAM2 실시간 불가 | 프레임당 1~2초 소요 | Adaptive Threshold + Contour 기반 고전 CV로 대체 (30fps 실시간) |

---

## 환경

- Python 3.12
- Ubuntu 24.04 (WSL2)
- CUDA 12.8 (SAM2 사용 시)
- Intel RealSense D455 (선택, 일반 웹캠도 가능)
