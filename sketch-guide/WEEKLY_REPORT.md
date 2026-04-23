# Sketch Guide 개발 노트 (2026.04.13)

## 목표

Google Magenta Sketch RNN 기반 실시간 드로잉 가이드 시스템. 갤럭시 탭에서 오프라인 동작.

## 접근

- 원본 데모(predict.js) 분석 → 인코딩 1회 + 프레임 단위 애니메이션 구조 적용
- 오프라인화: JS 라이브러리 + 모델 JSON → HTML 인라인 번들링 (standalone.html, 39MB)
- PC: Chrome --app 모드로 단독 창 실행
- 태블릿: Capacitor → APK 빌드

## 현재 상태

- 프로토타입 완성, PC 동작 확인
- APK 빌드 완료, 태블릿 실테스트 미진행
- 카테고리 5종 (pig, cat, dog, bird, flower)
- GitHub: https://github.com/ingon1026/drawing-to-2.5d/tree/sketch-guide/sketch-guide

## 다음

- 갤럭시 탭 실테스트
- 카테고리 확장 검토
- 커스텀 모델 학습 가능성 조사
