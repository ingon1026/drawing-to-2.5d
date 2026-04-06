# ✏️ drawing-to-2.5d

> 종이 위 손그림을 카메라로 촬영하면, 그림만 쏙 오려내고  
> 깊이(depth)와 표면(normal) 정보를 입혀 **2.5D asset**으로 만들어주는 파이프라인

---

## 🎯 뭘 하는 건가?

```
📷 카메라로 종이 비추기
 → 🎨 그림마다 실시간 색상 오버레이
 → 👆 원하는 그림 클릭
 → ✂️ 누끼 + 🗺️ 깊이맵 + 💡 노멀맵 자동 생성
 → 🎮 2.5D 뷰어에서 바로 확인
```

Unity/AR에서 평면 그림에 빛과 그림자가 반응하는 **2.5D 효과**를 만들기 위한 asset 생성 도구입니다.

---

## 🧠 사용 모델

| 모델 | 역할 | 비고 |
|------|------|------|
| `magic_touch.tflite` | 누끼 세그멘테이션 | MediaPipe, 자동 다운로드 |
| `Intel/dpt-hybrid-midas` | 깊이 추정 | MiDaS, 자동 다운로드 |
| `SAM2.1-hiera-tiny` | 정밀 auto-mask | 선택, 기본은 contour 방식 |

## 🛠️ 기술 스택

| 분류 | 기술 |
|------|------|
| 세그멘테이션 | MediaPipe (TFLite 직접 추론) |
| 깊이 추정 | MiDaS / DPT (PyTorch) |
| 실시간 프리뷰 | OpenCV — Adaptive Threshold + Contour |
| 카메라 | pyrealsense2 / OpenCV (웹캠 fallback) |
| 뷰어 UI | Pygame — bounce + squash 애니메이션 |

---

## 📦 설치

```bash
pip install -r requirements.txt

# RealSense 사용 시
pip install pyrealsense2

# SAM2 (선택)
pip install sam2
mkdir -p models
wget -P models/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

> 💡 세그멘테이션/깊이 모델은 첫 실행 시 자동 다운로드됩니다.

---

## 🚀 실행

### 라이브 데모
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

### CLI
```bash
python3 pipeline.py --input image.jpg --x 0.5 --y 0.4 --debug
```

---

## 📤 출력물

| 파일 | 설명 |
|------|------|
| `object.png` | 투명 배경 누끼 (BGRA) |
| `mask.png` | 이진 마스크 |
| `depth.png` | 깊이맵 |
| `normal.png` | 노멀맵 (RGB) |

---

## 📁 구조

```
├── config.py           # 설정값
├── normalize.py        # 입력 전처리
├── segment.py          # 누끼 추출 (TFLite)
├── postprocess.py      # 마스크 후처리
├── depth.py            # depth + normal map
├── export.py           # PNG 내보내기
├── auto_segment.py     # 실시간 프리뷰
├── pipeline.py         # CLI 진입점
├── live_demo.py        # 라이브 데모
├── viewer.py           # 2.5D 뷰어 (Pygame)
└── requirements.txt    # 의존성
```

---

## 💻 환경

Python 3.12 · Ubuntu 24.04 (WSL2) · CUDA 12.8 · Intel RealSense D455 (선택)
