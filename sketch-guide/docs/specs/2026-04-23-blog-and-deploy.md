# Sketch Guide — 사내 블로그 글 + 배포 설계

- 작성일: 2026-04-23
- 작성자: ingon (with Claude)
- 대상: 본인 + 후속 작업자
- 산출물: 사내 기술 블로그 글 1편 + Netlify 배포 URL 1개

---

## 1. 목표

- (A) Sketch Guide 프로젝트를 사내에 소개하는 한국어 기술 블로그 글 1편을 작성·게시한다. 톤은 팀장님의 "Animated Drawings" 소개 글과 동일.
- (B) Sketch Guide를 자기 PC가 아닌 외부 정적 호스팅에 올려, 사내 누구나 URL로 접속해 사용할 수 있게 한다. 본인 PC를 켜놓을 필요 없음.

## 2. 결정 사항

### 2-1. 독자 / 접근 통제

- 블로그·배포 모두 **K3I 사내 전용** (외부 공개 X)
- 접근 통제 수준: **URL만 알면 누구나** 접속 가능 (검색엔진엔 `noindex`로 차단)
- 인증, 비밀번호, VPN 모두 사용하지 않음 (단순함 우선)

### 2-2. 글 게시 플랫폼

- 팀장님이 사용한 동일 플랫폼 (Notion 또는 Confluence — 본인 확인)
- 원고는 Markdown으로 작성 → 해당 플랫폼에 붙여넣기

### 2-3. 글 톤·깊이

- 비개발자 다수 → LSTM, RNN, VAE, stroke-5, RDP 등 기술 용어 등장 X
- 팀장님 예시 톤 매칭 + "직접 구현한 부분(오프라인화)" **한 단락만** 추가
- 분량: 한국어 기준 ~2000자

### 2-4. 글 구조 (소제목 4개, 안 B)

```
[리드 단락 - 소제목 없음]
  → "돼지를 그려봐" 한 줄 소개 + AR BOOK 맥락

## 1. 프로젝트 개요
   - 무엇이고 왜 만들었나

## 2. 어떻게 작동하는가
   - STEP 1~4 워크플로우 (소제목 아닌 본문 강조 처리)

## 3. 이 도구가 특별한 이유와 활용처
   - Magenta Sketch RNN 소개
   - 우리가 한 일: 오프라인화 (한 단락)
   - 활용 아이디어 (AR BOOK 콘텐츠, 미술 교육 등)

## 4. 써보기 (사용 팁 + 접속 안내)
   - 좋은 결과를 위한 체크리스트
   - PC 접속 URL + 갤럭시 탭 APK 안내
   - 짧은 마무리 한 줄
```

### 2-5. 배포 플랫폼

- **Netlify Starter (무료)** 확정
  - 상업 이용 허용
  - 40MB 단일 HTML 파일 업로드 가능
  - 드래그·드롭 한 번으로 배포
  - 본인 PC 꺼져도 URL 살아있음
- 비교 후 제외:
  - Vercel Hobby — 약관상 비상업 한정 (회사 R&D 도구로는 부적합)
  - Cloudflare Pages — 25MB/파일 제한, 모델 분리 작업 필요
  - GitHub Pages — 무료 플랜에선 repo public 필수

### 2-6. 배포 자료 구성 (`deploy/` 폴더)

```
deploy/
├── index.html         (standalone.html 복사 → 메타 태그 정비된 버전)
├── SketchGuide.apk    (태블릿용 — 같은 사이트에서 다운로드)
└── robots.txt         (User-agent: *  Disallow: /)
```

- APK 다운로드 URL 패턴: `https://k3idrawing.netlify.app/SketchGuide.apk`
- 사이트 메인 URL: `https://k3idrawing.netlify.app` (사이트 이름은 Netlify 가입 후 변경 가능)
- 별도 클라우드 스토리지(Google Drive 등) 사용하지 않음

### 2-7. `standalone.html` 메타 태그 정비

- 수정 위치: **빌드 스크립트** [`scripts/build-standalone.js`](../../scripts/build-standalone.js) — 재현성 확보 목적
- 추가 항목:
  - `<meta name="robots" content="noindex, nofollow">`
  - `<title>Sketch Guide — K3I</title>`
  - `<meta name="description" content="AI와 함께 그리는 드로잉 가이드 — K3I 사내 도구">`
- 수정 후 빌드 1회 재실행

### 2-8. 시각 자료

- 본인이 캡처/녹화 완료
  - 커버 이미지 1장
  - 사용 흐름 GIF 1개 (10~15초, 사용자 한 획 → AI 자동 완성 장면)
- 원고에는 placeholder로 위치만 표시: `[커버 이미지]`, `[GIF: 사용 흐름]`

## 3. 작업 순서

| # | 담당 | 작업 | 산출물 |
|---|---|---|---|
| 1 | Claude | 빌드 스크립트에 메타 태그 3개 추가 | `scripts/build-standalone.js` 수정 |
| 2 | Claude | 빌드 1회 재실행 → `standalone.html` 갱신 | 메타 태그 들어간 40MB HTML |
| 3 | Claude | `deploy/` 폴더 생성 + `index.html`, `robots.txt` 배치 | 업로드 직전 폴더 |
| 4 | 사용자 | `SketchGuide.apk`를 `deploy/`에 복사 | APK 포함 폴더 |
| 5 | Claude | 블로그 원고 Markdown 작성 | `.md` 파일 (Section 2-4 구조) |
| 6 | 사용자 | Netlify에 가입/로그인 → `deploy/` 드래그·드롭 → 사이트 이름 변경 | 최종 URL 확보 |
| 7 | 사용자 | 원고의 `[배포 URL]` placeholder를 실제 URL로 교체, GIF/커버 끼워넣어 팀장님 플랫폼에 게시 | 사내 공유 |

## 4. 비고

- **라이선스:** Magenta는 Apache 2.0이라 재배포 문제 없음. 블로그 본문에 Google Magenta 크레딧 1줄 명시 필수.
- **카테고리 구성 (최종):** `pig` 단일. Quick Draw 데이터로 학습된 SketchRNN 모델. `assets/models/`에 25종 모델 자산 보유, `categories` 배열 수정 + 재빌드만으로 확장 가능.
- **K3I 로고 트레이스 시도 (미채택):** 초기 구현에서 사내 로고를 정적 트레이스 카테고리로 추가 시도했으나, 손그림과의 정렬·커버리지 감지 품질이 회사 시연용 수준에 못 미쳐 제거. 향후 필요 시 Magenta 학습 방식(stroke 데이터 수집 + 재학습) 또는 배경 트레이싱 오버레이 방식으로 재접근 예정.
- **파일 크기 변화:** 5종(39 MB) → pig 단일(12 MB). 27 MB 절감 → Netlify 업로드 빨라지고 모바일 다운로드 부담 감소.
- **배포 갱신:** standalone.html 재빌드 후 Netlify에 다시 드래그·드롭 (같은 사이트로 업로드하면 같은 URL 유지).
