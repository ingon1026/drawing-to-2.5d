# 📱 Drawing 2.5D — Android App

> 종이 위 손그림을 갤탭 카메라로 비추고 터치하면,  
> 그림만 쏙 오려내서 **같은 화면에서 통통 튀는** 2.5D 데모 앱

---

## 🎯 동작 흐름

하나의 화면에서 전부 처리:

```
📷 카메라 프리뷰 (전체화면)
 + 🎨 contour 오버레이 (실시간, OpenCV)
 → 👆 그림 터치
 → ✂️ magic_touch.tflite로 누끼 추출
 → 🎮 같은 화면에서 통통 튀기 애니메이션
 → 👆 다시 터치하면 카메라로 복귀
```

가로/세로 모드 자동 대응.

---

## 🧠 사용 모델

| 모델 | 역할 | 크기 |
|------|------|------|
| `magic_touch.tflite` | 터치 기반 누끼 세그멘테이션 | ~6MB |
| `midas_v21_small.tflite` | 깊이 추정 | ~63MB |

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|------|------|
| 언어 | Kotlin |
| 카메라 | CameraX |
| 세그멘테이션 | TFLite (magic_touch) |
| 깊이 추정 | TFLite (MiDaS v2.1) |
| 실시간 프리뷰 | OpenCV Android — Adaptive Threshold + Contour |
| 애니메이션 | Canvas 2D — bounce + sway + squash-stretch |

---

## 📦 설치

### 1. Android Studio에서 이 폴더 열기
```
File → Open → android/
```

### 2. 모델 파일 복사
`app/src/main/assets/`에 두 파일 필요:

```bash
# PC에서 먼저 Python 파이프라인을 한 번 실행하면 models/에 자동 다운로드됨
cp models/magic_touch.tflite android/app/src/main/assets/

# MiDaS TFLite 직접 다운로드
wget -O android/app/src/main/assets/midas_v21_small.tflite \
  https://github.com/isl-org/MiDaS/releases/download/v2_1/model_opt.tflite
```

### 3. Gradle Sync → Build → 갤탭에 설치

---

## 📁 구조

```
android/
├── app/src/main/
│   ├── java/com/drawing25d/
│   │   ├── MainActivity.kt        # 단일 Activity, 전체 흐름 관리
│   │   ├── SegmentationHelper.kt  # magic_touch TFLite 래퍼
│   │   ├── DepthHelper.kt         # MiDaS TFLite 래퍼
│   │   ├── ContourAnalyzer.kt     # 실시간 contour 감지 (OpenCV)
│   │   ├── ContourOverlay.kt      # contour 색상 오버레이 뷰
│   │   └── BounceView.kt          # 통통 튀기 Canvas 애니메이션
│   ├── res/layout/activity_main.xml
│   ├── AndroidManifest.xml
│   └── assets/                    # tflite 모델 (git 제외, 수동 복사)
├── build.gradle.kts
├── settings.gradle.kts
└── gradle.properties
```

---

## ⚙️ 누끼 추출 방식

- **프리뷰**: contour 기반 (가볍고 빠름, 시각적 가이드 전용)
- **실제 누끼**: `magic_touch.tflite`로 터치 좌표 기반 세그멘테이션
- foreground > 30% 시 자동 거부 (종이 전체가 따지는 것 방지)
- 누끼 후 투명 영역 자동 크롭

---

## 💻 환경

- Kotlin · CameraX · TFLite · OpenCV Android
- minSdk 26 · targetSdk 35
- 갤럭시 탭 테스트
