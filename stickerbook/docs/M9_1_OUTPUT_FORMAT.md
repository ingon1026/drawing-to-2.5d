# M9.1 — AnimatedDrawings 출력 포맷 검증 결과

검증일: 2026-04-24
검증 스크립트: `animated_drawings.render.start()` 를 직접 호출 (AD default motion = `dab.yaml`, retarget = `fair1_ppf.yaml`)
입력 캐릭터: `AnimatedDrawings/examples/characters/char1/` (번들 샘플, `char_cfg.yaml` + `texture.png` + `mask.png` 사전 생성 완료)
렌더 환경: `PYOPENGL_PLATFORM=glx`, animated_drawings conda env (WSL)

## 무엇을 검증했는가

AD 렌더러가 출력하는 비디오(GIF/MP4) 파일의 포맷, 해상도, FPS, 배경색, 투명 채널 여부를 cv2.VideoCapture 및 Pillow 로 실측하여 `AnimatedStickerRenderer` 의 재생 전략 선택에 필요한 근거를 확보한다.

### 중요: TorchServe 단계는 미실행

- 본 검증은 **렌더 단계만** 실행함. `image_to_annotations.py` 가 요구하는 `drawn_humanoid_detector.mar` / `drawn_humanoid_pose_estimator.mar` 가 이 호스트에 존재하지 않아 (`/home/ingon/AR_book/AnimatedDrawings/torchserve/model-store/` 자체가 없음) TorchServe 단계는 수행 불가능.
- 렌더 단계는 TorchServe 와 독립적이므로 출력 포맷 결론에는 영향 없음 (AD 파이프라인의 rig 소스만 다를 뿐, 렌더링·인코딩 경로는 동일).
- 이후 M9 태스크 (사용자 입력 이미지 → 애니메이션) 를 위해서는 `.mar` 파일을 별도 확보해야 함 — 현재 블로커로 기록.

## 출력물 관찰

출력 경로: `/tmp/m9_1_out/`

| 파일 | 존재 | 크기 | 해상도 | FPS | 프레임 수 | 채널 | 배경 색 |
|---|---|---|---|---|---|---|---|
| video.gif | O | 4,170,921 B (~4.0 MB) | 500x500 | 33.33 | 339 | palette + 1-bit alpha (PIL: mode `P`, transparency index 0) | RGB (255,255,255), alpha 0 |
| video.mp4 (codec `avc1`) | **X** | 0 | - | - | - | - | - (인코더 없음) |
| video.mp4 (codec `mp4v`) | O | 2,833,350 B (~2.7 MB) | 500x500 | 30.00 | 339 | BGR 3ch (alpha drop됨) | (250, 253, 251) [약한 그린 틴트, mp4v 압축 부산물] |

FPS 주석: GIF 의 33.33 은 `GIFWriter` 가 `delta_t*1000` → frame duration (ms) 을 저장한 결과 (`delta_t` = BVH `frame_time` = 0.0333... → duration 30ms). MP4 의 30 은 `MP4Writer` 가 `round(1/delta_t) = 30` 으로 정수화한 결과. 두 파일의 실제 타이밍 의도는 동일 (약 33.3fps) 이나 MP4 쪽은 정수 FPS 로 살짝 느리게 재생됨.

## 배경색 안정성

전체 영상에 걸쳐 프레임 0, n/4, n/2, 3n/4, n-1 각각의 edge ring (상하좌우 20px 테두리) 을 샘플링:

- GIF: 전 프레임 모두 edge ring min = max = (255,255,255). **완전한 순백, 압축 노이즈 0.**
- MP4 (mp4v): 전 프레임 모두 edge ring min = max = (250,253,251). 동일값 baked-in, 노이즈는 없으나 순백이 아님.

전체 프레임 내 threshold 240 / 200 통과 픽셀 수 비교:

- GIF: all-ch≥240 → 211,065 / 250,000 픽셀 | all-ch≥200 → 211,065 / 250,000 픽셀 (완전 동일 → 200–240 구간에 캐릭터 픽셀 0개, 240 을 임계값으로 써도 false positive 없음)
- MP4: all-ch≥240 → 210,975 / 250,000 | all-ch≥200 → 211,065 / 250,000 (차이 90px, 압축 블리딩)

## cv2.VideoCapture 호환성 (시스템 python3 + opencv-python 4.13)

- GIF: cv2.VideoCapture 로 **정상 오픈**. 339 프레임 순차 읽기 OK. 단, **알파 채널은 드롭**되어 BGR 3채널 + 백색(255,255,255) 배경으로 디코드됨 (GIF palette transparency → 파일 background 색으로 채워짐).
- MP4 (mp4v): cv2.VideoCapture 로 **정상 오픈**, 339 프레임 읽기 OK. 원래부터 BGR 3ch.
- MP4 (avc1 = H.264): 이 환경의 cv2 FFmpeg 빌드에 H.264 인코더가 없음. 렌더 시작 시 다음 에러 발생 후 파일 생성되지 않음:
  ```
  VIDEOIO/FFMPEG: Failed to initialize VideoWriter
  Could not find encoder for codec_id=27, error: Encoder not found
  ```
  → AD 에 `OUTPUT_VIDEO_CODEC: avc1` 을 쓰지 말 것. 현 환경 기본값은 `mp4v` 로 설정 필요.

## 렌더러 내부 구조 (주요 근거 요약)

`animated_drawings/controller/video_render_controller.py` 분석:

- 렌더 루프는 GL framebuffer 에서 `GL_BGRA / GL_UNSIGNED_BYTE` 로 RGBA 픽셀을 읽음 (clear color = `[1.0, 1.0, 1.0, 0.0]`, 즉 RGB=white, alpha=0).
- `GIFWriter` → `cv2.cvtColor(..., BGRA2RGBA)` → Pillow 로 RGBA 프레임 저장 → **진짜 알파 보존**.
- `MP4Writer` → `frame[:, :, :3]` 슬라이싱으로 **알파를 명시적으로 버림** → BGR 3ch 만 cv2.VideoWriter 에 전달. 알파=0 이었던 배경 픽셀은 RGB (255,255,255) 가 그대로 남아 MP4 에 박힘 (mp4v 인코더 특성상 약간 디스퍼시브하게 변형되어 250,253,251).

즉, "AD 는 원천적으로 RGBA 렌더를 수행하지만, MP4 경로는 알파를 버리고 내보낸다" — 이는 이후 스티커 합성 때 크로마키 처리가 필수인 이유.

## 후보 재생 전략 비교

(A/B/C 이름은 플랜 명시가 아니라 본 조사에서 정의한 것)

- **후보 A — GIF + Pillow RGBA**: Pillow 로 GIF 프레임을 RGBA 로 직접 디코드. 진짜 1-bit 알파를 그대로 사용 (크로마키 불필요). 단점: Pillow 는 cv2 보다 느리고 전체 프레임을 메모리에 올려야 함, 실시간 프레임 스트리밍이 불편, 기존 `AnimatedStickerRenderer._bgr_to_bgra_chroma` 인터페이스와 불일치.
- **후보 B — MP4 + cv2 + chroma-key**: cv2.VideoCapture 로 빠르게 BGR 프레임 로딩, 임계값(주변 250 근처) 으로 알파 생성. 단점: mp4v 압축으로 BG 가 순백이 아니고 (250,253,251), 임계값 선택이 민감해짐. 또한 `avc1` 은 현재 환경에서 인코딩 불가, 환경마다 인코더 lotery 가 발생할 위험.
- **후보 C — GIF + cv2 + chroma-key**: cv2.VideoCapture 로 GIF 를 BGR 3ch 로 읽음 (cv2 가 알파 자동 드롭 후 BG=255 로 채움). 전 프레임 BG 가 정확히 (255,255,255) 이며 압축 노이즈 0 → 매우 관대한 chroma threshold 로 충분. 기존 `_bgr_to_bgra_chroma` 인터페이스와 정확히 일치. 단점: GIF 는 파일 크기가 MP4 의 약 1.5배, 정수 FPS 가 아니어서 타이밍 재구성 시 `frame_duration = 30ms` 를 수동으로 지정해야 함.

## 재생 전략 결정

**선택:** 후보 C — GIF + cv2.VideoCapture + chroma-key against white.

**이유:**
1. GIF 배경색이 전 프레임에 걸쳐 정확히 (255,255,255) 로 일정 (실측). MP4 (250,253,251) 는 압축 부산물로 인해 순수 하얀색이 아니며 프레임 간 편차가 없더라도 미세 틴트가 유지됨.
2. 임계값 240 과 200 이 같은 결과를 내는 것이 실증적으로 확인됨 → 캐릭터 픽셀이 200~240 구간에 단 1픽셀도 존재하지 않음 → 240 수준의 관대한 임계값을 써도 false positive 위험 0 (char1 기준).
3. 현 환경에서 `avc1` 인코더가 없어 MP4 경로는 `mp4v` 강제. 이 선택은 코덱 의존성을 외부 환경에 노출시키므로 피하는 것이 안전.
4. 기존 `AnimatedStickerRenderer._bgr_to_bgra_chroma` 의 입력이 BGR 3ch 이므로 후보 C 가 인터페이스 일치. Pillow-RGBA (후보 A) 는 추가 디코더 의존성과 메모리 스파이크 발생.
5. cv2.VideoCapture 는 GIF 를 정상 오픈·스트리밍 가능함을 실측 확인.

## M9 후속 태스크에 반영할 사항

- **`AnimatedStickerRenderer._bgr_to_bgra_chroma` chroma threshold: `240`**
  - 판단 규칙: **"BGR 모든 채널이 240 이상이면 투명 (alpha=0), 그 외는 불투명 (alpha=255)"**
  - **주의 — 플랜 원문 표현 교정 필요**: 플랜 Step 5 예시에는 "BGR 모든 채널이 N 이하면 투명" 이라 적혀 있으나, 이는 검은 배경을 가정한 표현임. 실제 AD 출력 배경은 **흰색 (255,255,255)** 이므로 방향은 반대 (`≥ N` 이 투명). 구현 시 `>=` 로 작성해야 함.
  - margin 근거: BG = 255 이고 캐릭터 픽셀 중 all-ch≥200 인 것이 0개로 실측됨 → 240 은 15의 여유 margin 을 두면서 false positive 가 발생하지 않는 안전 임계값. 순백만 정확히 걸러내고 싶다면 254 로 좁혀도 동일 결과.
- **프레임 루프 범위**: 0 ~ 338 (총 339 프레임).
- **프레임 재생 주기**: GIF info `duration = 30 ms/frame` → 약 33.33 fps. `AnimatedStickerRenderer` 가 영상 시간을 직접 제어할 경우 `frame_period_ms = 30` 으로 고정하면 AD 의 원의도와 일치.
- **입력 경로**: AD 가 `video_render` 모드에서 `OUTPUT_VIDEO_PATH` 확장자로 포맷을 결정 (`.gif` → GIFWriter, `.mp4` → MP4Writer). 스티커북에서는 **`.gif` 확장자를 강제**해서 일관성 확보.
- **코덱 주의 (혹시 MP4 가 필요한 경우)**: 현재 환경 cv2 빌드는 `avc1` (H.264) 인코더 미포함. `mp4v` 만 사용 가능. AD `OUTPUT_VIDEO_CODEC` 기본값이 `avc1` 이므로 MP4 경로를 채택하려면 반드시 `mp4v` 로 덮어써야 함.
- **[블로커] TorchServe 모델 파일 미확보**: `drawn_humanoid_detector.mar`, `drawn_humanoid_pose_estimator.mar` 가 이 호스트에 없음. 향후 사용자 그림 → AD 파이프라인 엔드-투-엔드 검증 태스크 전에 이 자산들을 다운로드/빌드해야 함 (M9 후속 태스크의 별도 사전조건).
- **GIF 알파 보존 여지**: 현재 채택안은 cv2 가 알파를 버린 뒤 크로마키로 재구성하는 경로. 만약 이후에 anti-aliased 가장자리를 매끄럽게 살리고 싶다면 Pillow-RGBA (후보 A) 로 전환 가능하나, 현재 GIF 알파는 1-bit (이분) 이므로 실이득은 제한적임.
