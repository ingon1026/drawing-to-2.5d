# stickerbook

**2.5D 증강현실 (AR Stickerbook)** — Stage 1 Python PC PoC.

아이가 스케치북에 그린 그림을 웹캠으로 보여주고, 원하는 그림을 클릭하면 그 그림이 2.5D 딱지로 추출되어 월드 공간(종이 위)에 고정된 AR 오버레이로 보이는 엔드투엔드 데모.

Stage 2 (Galaxy 테블릿 on-device 포팅, ARCore + ONNX Runtime Mobile) 는 별도 후속 작업이며 이 폴더는 Stage 2 호환 자산 (`texture.png` / `mask.png` / `char_cfg.yaml`) 을 내보내는 것까지 + pop-up 렌더링(M8) 까지 다룬다.

---

## References

- **메인 리깅/애니메이션**: [facebookresearch/AnimatedDrawings](https://github.com/facebookresearch/AnimatedDrawings) (MIT)
- **2.5D 딱지화 기법**: [tatsuya-ogawa/RakugakiAR](https://github.com/tatsuya-ogawa/RakugakiAR) (Swift/ARKit) + [Qiita 시리즈](https://qiita.com/pic18f14k50/items/0d20f5f544c9010f79e6)
- **형제 PoC (모델 재사용)**: `../LivingDrawing/`

자세한 설계·리스크·마일스톤 상태는 [`docs/DESIGN.md`](docs/DESIGN.md) 참조.

---

## Quickstart

### 1. 환경 준비

```bash
# (선택) 가상환경
python3 -m venv .venv
source .venv/bin/activate

# 의존성
pip install -r requirements.txt
```

### 2. 모델 준비

```bash
# MobileSAM (~39MB): LivingDrawing 폴더에서 심볼릭 링크
ln -sf /home/ingon/AR_book/LivingDrawing/mobile_sam.pt models/mobile_sam.pt

# YOLO26n (~5MB): 최초 실행 시 ultralytics 가 자동 다운로드
#   또는 수동: python3 -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
```

### 3. 실행

```bash
python3 main.py --camera 1
```

> **WSL2 노트**: `/dev/video0` 이 Orbbec/Realsense 센서 장치로 오인되는 경우가 있어 `--camera 1` 이 실제 웹캠일 수 있음. 작동 안 하면 `--camera 0`, `--camera 2` 순으로 시도.

### 4. 조작법

| 입력 | 동작 |
|---|---|
| **마우스 좌클릭** | 클릭 좌표의 그림을 MobileSAM 으로 누끼 → 2.5D 딱지 생성 + 종이 추적 시작 |
| **S 키** | 현재 생성된 모든 딱지를 `assets/captures/<timestamp>/sticker_NN/` 로 저장 (PNG + YAML) |
| **R 키** | 모든 딱지 리셋 |
| **Q / Esc** | 종료. 콘솔에 성능 측정 summary 출력 |

---

## 기대 동작

1. 웹캠 윈도우 (`stickerbook`) 가 뜨고 **파란 박스** (contour 기반 그림 후보) / **초록 박스** (YOLO26 person/animal) 가 실시간 표시
2. 아이 그림 클릭 → 약 0.7~1초 후 해당 그림만 투명 배경으로 누끼된 "2.5D 딱지" 가 같은 자리에 합성됨 (그림자 효과 포함)
3. 종이를 이동·회전·기울이면 딱지가 Homography 추적으로 종이를 따라 움직임
4. 종이가 프레임 밖으로 나가면 ~0.5초 뒤 딱지 일시 숨김. 다시 들이면 자동 재획득
5. S 키 → 디스크에 AnimatedDrawings 호환 에셋 저장

---

## 현재 구현 상태 (2026-04-21)

Stage 1 완료. 세부는 [`docs/DESIGN.md`](docs/DESIGN.md#마일스톤) 참조.

| Milestone | 상태 |
|---|---|
| M0 스캐폴드 | ✅ |
| M1 카메라 + UI 루프 | ✅ |
| M2 후보 감지 (YOLO + contour) | ✅ |
| M2.5 Adaptive threshold 튜닝 | ✅ |
| M3 SAM 누끼 + 비동기 | ✅ |
| M4 정적 딱지 렌더 | ✅ |
| M5 Homography 월드 고정 | ✅ |
| M5.5 재획득 루프 | ✅ |
| M6 다중 딱지 + 리셋/저장 | ✅ |
| M6.1 raw/display 프레임 분리 | ✅ |
| M7 성능 측정 + 문서 정리 | 🟡 진행중 |
| M7.5 AnimatedDrawings 라운드트립 | ⏳ 예정 |
| Stage 2 (Android ARCore + ONNX) | ⏳ 별도 프로젝트 |

---

## 테스트

```bash
python3 -m pytest tests/
```

현재 **44 tests passing** (M1~M6.1 커버리지).

---

## Folder layout

```
stickerbook/
├── capture/     # 웹캠 추상 (cv2.VideoCapture 래퍼)
├── detect/      # YOLO26n + contour 기반 그림 후보 감지
├── extract/     # MobileSAM point-prompt 누끼 (ultralytics SAM)
├── track/       # WorldAnchor Protocol + HomographyAnchor 구현 (ORB + RANSAC)
├── render/      # overlay (박스) + tilt_renderer (warpPerspective + 그림자)
├── export/      # AnimatedDrawings 호환 자산 저장 (PNG + YAML)
├── models/      # yolo26n.pt, mobile_sam.pt (가중치)
├── assets/      # captures/ (저장된 딱지), samples/, debug/
├── tests/       # pytest 단위 + mock 기반 통합
└── docs/
    └── DESIGN.md
```

---

## Troubleshooting

| 증상 | 원인 | 해결 |
|---|---|---|
| `CameraError: failed to open camera at index 0` | WSL2 가 Orbbec 장치를 index 0 으로 노출 | `--camera 1` (또는 2~5) |
| 첫 클릭 시 5~10초 지연 | MobileSAM 최초 로드 (torch 모델 로딩) | 정상. 이후 클릭은 ~0.7~1s |
| 딱지가 뜨는데 종이 따라 안 움직임 | 종이/배경 대비 약해서 ORB feature 부족 | 종이 표면에 라인·텍스트가 많을수록 잘 동작. 조명 개선 |
| 종이 기울이면 딱지 사라짐 | 각도 70°+ 는 lost 임계 | DESIGN R2 수용. 평면 유지 권장 |
| `texture.png` 에 파란/초록 박스 박힘 | M6.1 이전 버그 (오버레이 오염) | 현 코드에서 수정됨 (R7) |

---

## License

Internal R&D. 재사용하는 외부 레포의 라이선스 (AnimatedDrawings: MIT, MobileSAM, Ultralytics) 는 각 원 레포 참조.
