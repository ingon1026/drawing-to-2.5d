# Sketch Guide

## 개요

"돼지를 그려봐" 같은 미션을 제시하고, 사용자가 그리는 동안 AI가 나머지를 어떻게 그리면 되는지 실시간 애니메이션으로 보여주는 드로잉 가이드 시스템.

## 핵심 기술

- **Sketch RNN** (Google Magenta) — Quick Draw 데이터셋(수백만 장)으로 학습된 드로잉 예측 모델
- **RNN(LSTM) 기반 시퀀스 모델** — 사용자의 스트로크 시퀀스를 인코딩하고, 다음 획을 확률적으로 샘플링
- **HTML5 Canvas + PointerEvents** — 터치/스타일러스 드로잉 엔진

## 동작 방식

```
사용자가 획을 그림
    ↓
펜을 놓으면 400ms 대기 (추가 획 기다림)
    ↓
대기 끝 → 라인 단순화(RDP) → stroke-5 변환 → 모델 인코딩 (1회)
    ↓
AI가 한 획씩 실시간 애니메이션으로 나머지를 그려줌
    ↓
완성되면 다른 색으로 또 다른 변형 제시 (최대 3번)
    ↓
그림이 충분하면 "다 그린 것 같아요?" 사이드 알림
    ↓
사용자가 중간에 터치하면 AI 즉시 중단 → 새 사이클
```

## 실행 방법

### PC
```bash
cd /home/ingon/AR_book/sketch-guide
./run.sh
```
Chrome 앱 모드로 단독 창 실행 (주소창/탭 없음)

### 갤럭시 탭
바탕화면의 `SketchGuide.apk`를 태블릿에 복사 → 설치 → 홈화면에서 앱 실행

### 서버/WiFi 불필요 — 모델이 앱 안에 내장 (완전 오프라인)

## 프로젝트 구조

```
sketch-guide/
├── standalone.html          ← 핵심. 모든 것이 인라인된 단일 파일 (39MB)
├── run.sh                   ← PC 단독 실행 스크립트
├── scripts/
│   └── build-standalone.js  ← standalone.html 빌드 스크립트
├── assets/
│   ├── models/              ← Sketch RNN 사전학습 모델 (JSON)
│   └── web/lib/             ← numjs.js, sketch_rnn.js (원본 라이브러리)
├── src/                     ← Expo React Native 코드 (미사용, 참조용)
└── SUMMARY.md               ← 이 파일

sketch-guide-apk/            ← Capacitor APK 빌드용 프로젝트
├── www/index.html            ← standalone.html 복사본
└── android/                  ← Android 프로젝트 (Gradle 빌드)
```

## 지원 카테고리

pig, cat, dog, bird, flower (5종)

## 기술 상세

| 항목 | 내용 |
|------|------|
| 모델 | Sketch RNN (LSTM VAE), Google Magenta |
| 모델 크기 | large 12MB / small 2.9MB (카테고리당) |
| 학습 데이터 | Google Quick Draw 데이터셋 |
| 스트로크 형식 | stroke-5: [dx, dy, pen_down, pen_up, pen_end] |
| 라인 단순화 | Ramer-Douglas-Peucker (RDP) 알고리즘 |
| Temperature | 0.25 (낮을수록 일관적, 높을수록 다양) |
| 완성 감지 | AI 예측이 15스트로크 미만으로 2회 연속 끝나면 완성 판정 |
| 빌드 도구 | Node.js 빌드 스크립트 (인라인 번들링) |
| APK 빌드 | Capacitor + Android Gradle |

## 완료된 작업

1. Magenta Sketch RNN 데모 분석 및 모바일 구현 가능성 검토
2. HTML5 Canvas 드로잉 엔진 구현 (터치/스타일러스 지원)
3. Sketch RNN 모델 연동 — 사용자 스트로크 인코딩 → 다음 획 예측
4. 상태 머신 기반 턴 관리 (USER_DRAWING → WAITING → AI_ANIMATING)
5. AI 실시간 애니메이션 (프레임 단위 1포인트씩 렌더링)
6. 자동 변형 루프 (최대 3회, 다른 색상으로 반복)
7. 완성 감지 + 사이드 토스트 알림
8. JS + 모델 인라인 번들링 → standalone.html (완전 오프라인)
9. large 모델 교체 (그림 품질 개선)
10. PC 단독 실행 (Chrome --app 모드)
11. 갤럭시 탭 APK 빌드 (Capacitor)

## GitHub

https://github.com/ingon1026/drawing-to-2.5d/tree/sketch-guide/sketch-guide

## 남은 과제

- 태블릿 실테스트 (터치/스타일러스 검증)
- 카테고리 확장 (5개 → 50~100개)
- 커스텀 모델 학습 (자체 데이터셋)
- 그림 완성도 평가/스코어링 고도화
- 레퍼런스 드로잉 DB 저장/관리
