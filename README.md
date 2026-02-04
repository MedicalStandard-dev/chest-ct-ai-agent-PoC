# Medical AI PACS - CT Chest AI 판독 보조 시스템

> **Upstage Solar Pro 3 + MONAI 기반 의료 영상 AI 판독 보조 시스템**

CT Chest 영상에서 폐결절을 자동 검출하고, 구조화된 리포트를 생성하는 제품형 AI 시스템입니다.

## 🎯 핵심 설계 원칙

| 원칙 | 설명 |
|------|------|
| **Evidence-first** | 모든 소견은 Vision 모델 출력(heatmap)에 기반 |
| **Table-first** | 숫자/측정값은 표로만 제공, 문장에 포함 금지 |
| **Validator-gated** | LLM 출력은 반드시 검증 후 사용 |
| **Fail-closed** | 검증 실패 시 템플릿 기반 안전 출력 |

## ✨ Upstage Solar 활용 기능

### 1. Solar Pro 3 - Report Rewriter
- **Narrative-only 모드**: 표/수치는 LLM에 전달하지 않음
- QUALITY, NOTES 등 비표 텍스트만 다듬음
- 표 변조 원천 차단

### 2. Solar Pro 3 - Groundedness Check
- 생성된 리포트가 AI 분석 결과에 근거하는지 검증
- 환각(hallucination) 방지

### 3. Solar Pro 3 - Physician Q&A
- 의사가 AI 분석 결과에 대해 질문
- 과거 검사 기록(Prior)과 비교 질문 지원
- 한국어 답변

### 4. Solar Embedding - RAG System
- 과거 리포트 의미 기반 검색
- Patient isolation 적용 (환자별 데이터 격리)
- Prior Comparison 지원

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                         CT Volume Input                          │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Vision Pipeline (MONAI)                       │
│  ┌─────────────────┐    ┌─────────────────────────────────────┐ │
│  │ Lung Segmentation│    │     Nodule Detection (Heatmap)     │ │
│  │    (DynUNet)    │    │           (UNet)                    │ │
│  └─────────────────┘    └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Post-Processing Pipeline                      │
│  ┌───────────────┐  ┌────────────────┐  ┌────────────────────┐ │
│  │ Peak Detection │→│ Component      │→│ Measurements       │ │
│  │ (3D NMS)       │  │ Extraction     │  │ (diameter, volume) │ │
│  └───────────────┘  └────────────────┘  └────────────────────┘ │
│                              │                                   │
│  ┌───────────────┐  ┌────────────────┐  ┌────────────────────┐ │
│  │ Evidence Gen  │←│ Location       │←│ Threshold Policy   │ │
│  │ (slice range) │  │ (lung mask)    │  │ (finding/limit)    │ │
│  └───────────────┘  └────────────────┘  └────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG System (Solar Embedding)                  │
│  ┌─────────────────┐    ┌─────────────────────────────────────┐ │
│  │ ChromaDB        │    │     Prior Comparison                │ │
│  │ (Patient-isolated)│   │     (Lesion Tracking)              │ │
│  └─────────────────┘    └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                Report Generation (Table-first)                   │
│  ┌─────────────────┐    ┌─────────────────────────────────────┐ │
│  │ Template Builder │    │     Solar Pro 3 Rewriter           │ │
│  │ (Tables fixed)  │    │     (Narrative-only)                │ │
│  └─────────────────┘    └─────────────────────────────────────┘ │
│                              │                                   │
│  ┌─────────────────┐    ┌─────────────────────────────────────┐ │
│  │ Validator       │    │     Groundedness Check              │ │
│  │ (Safety gate)   │    │     (Solar Pro 3)                   │ │
│  └─────────────────┘    └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Output                                   │
│  • Structured AI Result (JSON)                                  │
│  • Draft Report (Tables + Narrative)                            │
│  • Prior Comparison Table                                        │
│  • Evidence (slice ranges, masks)                                │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 생성되는 리포트 구조

```
DRAFT — Requires physician confirmation
CT Chest | 2026-02-03 | LIDC-IDRI-0001
--------------------------------------------------

QUALITY / LIMITATIONS
- Small nodules (<3 mm) may be underestimated due to resolution limits.
- Several candidates detected (3); false positives possible.
- AI findings are candidates only and require physician confirmation.

KEY FLAGS
- Nodule candidates: 3 (NEW: 0)
- High-confidence findings: 3
- Scan limitation: Yes

FINDINGS - TABLE
Type | Location | Status | Confidence | Evidence
--- | --- | --- | --- | ---
Nodule candidate | RUL | Present | 0.98 | CAND_001
...

MEASUREMENTS - TABLE
Lesion ID | Location | Diameter (mm) | Volume (mm³) | Confidence
--- | --- | --- | --- | ---
CAND_001 | RUL | 7.1 | 185.0 | 0.98
...

PRIOR COMPARISON - TABLE
Lesion ID | Prior Date | Change | Prior Size | Current Size
--- | --- | --- | --- | ---
CAND_001 | 2025-08-03 | INCREASED | 5.2 mm | 7.1 mm
...

NOTES
- This draft is generated from AI outputs.
- Requires physician confirmation.

AUDIT
- model_version: nodule-det-v1.0
- pipeline_version: 1.0.0
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
SOLAR_API_KEY=your_openrouter_api_key
SOLAR_API_ENDPOINT=https://openrouter.ai/api/v1
SOLAR_MODEL=upstage/solar-pro-3:free
SOLAR_EMBEDDING_MODEL=upstage/solar-embedding-1-large
USE_MOCK_SOLAR=false
USE_MOCK_EMBEDDING=false
```

### 3. 서버 실행

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. UI 접속

```
http://localhost:8000/ui
```

## 📁 프로젝트 구조

```
medical-ai-pacs/
├── api/
│   ├── main.py              # FastAPI 앱
│   └── schemas.py           # Pydantic 스키마
├── monai_pipeline/
│   ├── nodule_detection.py  # 결절 검출 모델
│   ├── lung_segmentation.py # 폐 분할 모델
│   ├── candidate_processor.py # 후보 처리
│   ├── evidence_generator.py  # Evidence 생성
│   ├── tracking_engine.py   # 병변 추적
│   └── production_pipeline.py # 통합 파이프라인
├── solar_integration/
│   ├── embeddings.py        # Solar Embedding
│   ├── rag_system.py        # RAG + ChromaDB
│   ├── rewriter.py          # Narrative Rewriter
│   ├── validator.py         # 안전 검증기
│   ├── templates.py         # 리포트 템플릿
│   ├── report_generator.py  # 리포트 생성기
│   ├── solar_features.py    # Q&A, Groundedness 등
│   └── sample_data.py       # 샘플 Prior 데이터
├── config/
│   └── settings.py          # 환경 설정
├── static/
│   └── index.html           # 대시보드 UI
├── models/
│   ├── nodule_det/          # 학습된 결절 검출 모델
│   └── lung_seg/            # 학습된 폐 분할 모델
├── data/
│   └── processed/lidc/      # 전처리된 LIDC 데이터
└── docs/
    └── PROJECT_PLAN.md      # 상세 기획서
```

## 🔌 API 엔드포인트

| Endpoint | Method | 설명 |
|----------|--------|------|
| `/ui` | GET | 대시보드 UI |
| `/health` | GET | 시스템 상태 |
| `/api/v1/analyze` | POST | CT 분석 요청 |
| `/api/v1/qa` | POST | 의사 Q&A |
| `/api/v1/groundedness-check` | POST | 환각 검증 |
| `/api/v1/patient/{id}/history` | GET | 환자 이력 조회 |
| `/api/v1/lidc-cases` | GET | 사용 가능한 LIDC 케이스 |

## 🧪 데이터셋

- **LIDC-IDRI**: 폐결절 검출 학습/테스트
- **MSD Task06_Lung**: 폐 분할 학습

## 📈 모델 성능

| 모델 | 데이터셋 | 메트릭 |
|------|----------|--------|
| Nodule Detection | LIDC-IDRI | Best Loss: 0.23 |
| Lung Segmentation | MSD Lung | Best Dice: 0.23 (개선 필요) |

## ⚠️ 제한사항

1. **Lung Segmentation**: 현재 Dice 점수가 낮아 location 판정이 부정확할 수 있음
2. **Small Nodules**: 3mm 미만 결절은 검출 정확도 저하
3. **의사 확인 필수**: 모든 AI 소견은 의사 확인 후 사용

## 🔒 안전 설계

- **Validator-gated**: 금지 표현 자동 차단
- **Fail-closed**: 검증 실패 시 템플릿 폴백
- **Table-protected**: LLM이 수치/표를 변경 불가
- **Audit log**: 모든 처리 과정 기록

## 📜 라이선스

이 프로젝트는 연구/교육 목적으로 개발되었습니다.

---

**Developed for Upstage Solar API Demonstration**
