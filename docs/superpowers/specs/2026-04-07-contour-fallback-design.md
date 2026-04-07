# Contour Fallback + Qiita-style Preprocessing

## Context

drawing-2p5d 라이브 데모에서 일부 그림이 애니메이션으로 변환되지 않는 버그가 있다.
카메라 프리뷰(contour 기반)에서는 그림이 잘 감지되지만, 클릭 후 magic_touch ML 모델이
특정 그림에서 낮은 foreground 비율(예: 2.7%)을 반환하여 사실상 실패한다.

Qiita의 らくがきAR 재현 기사에서 contour를 최종 세그멘테이션으로 직접 사용하는 방식을
참고하여, ML 실패 시 contour 마스크를 fallback으로 사용하는 전략을 적용한다.

## Approach: Contour Fallback

magic_touch를 기본으로 유지하되, ML foreground가 임계값 미만이면
카메라 프리뷰에서 이미 감지된 contour 마스크를 대신 사용한다.

## Changes

### 1. auto_segment.py — 전처리 개선 + 유틸 함수

**generate_masks_contour()** 전처리를 Qiita 방식으로 변경:
- 현재: `adaptiveThreshold` 단일 처리
- 변경: `erode` (얇은 선 두껍게) → `threshold` (펜 선만 추출)

**contour_mask_to_uint8()** 함수 추가:
- contour 마스크(bool)를 pipeline용 uint8(0/255)로 변환

### 2. live_demo.py — 핵심 흐름 변경

**camera_phase()**:
- 반환값에 클릭한 segment의 contour 마스크 추가
- `(frame, norm_x, norm_y)` → `(frame, norm_x, norm_y, contour_mask)`

**run_pipeline()**:
- `contour_mask` 파라미터 추가
- magic_touch foreground < `ML_FOREGROUND_MIN`(5%) 이면 contour 마스크로 fallback
- fallback 사용 시 로그 출력: `"ML foreground too low, using contour mask fallback"`
- contour도 없고 fg < 0.5% 이면 기존대로 `return None`

**main()**:
- `camera_phase()` 반환값 언패킹을 4개로 변경
- `run_pipeline()`에 contour_mask 전달

### 3. config.py — fallback threshold

`ML_FOREGROUND_MIN = 5.0` 추가

### 변경하지 않는 파일

- segment.py: magic_touch 로직 유지
- postprocess.py: contour/ML 마스크 모두 동일 후처리
- depth.py, export.py, viewer.py, pipeline.py: 변경 불필요

## Flow (after change)

```
카메라 프리뷰 (contour 기반, 개선된 전처리)
  → 클릭 → 프레임 + 좌표 + contour 마스크 전달
  → magic_touch 세그멘테이션
  → fg >= 5%? → ML 마스크 사용
             → contour 마스크 fallback
  → postprocess → depth → normal → export → viewer
```

## Verification

1. 기존 성공 케이스 (fg 51.2%): ML 마스크 사용, 품질 동일 확인
2. 기존 실패 케이스 (fg 2.7%): contour fallback 사용, viewer까지 도달 확인
3. 연한 연필 그림: 개선된 전처리로 contour 감지 향상 확인
4. 터미널 로그에서 fallback 사용 여부 확인 가능
