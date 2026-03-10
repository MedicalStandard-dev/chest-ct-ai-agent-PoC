# Medical AI PACS — CT Chest AI 판독 보조 시스템

> **Upstage Solar Pro 3 + MONAI 기반 의료 영상 AI 판독 보조 시스템**

CT Chest 영상에서 폐결절을 자동 검출하고, 구조화된 판독 보조 리포트를 생성하는 제품형 AI 시스템입니다.

---

## 핵심 설계 원칙

| 원칙 | 설명 |
|------|------|
| **Evidence-first** | 모든 소견은 Vision 모델 출력(heatmap/검출)에 기반 |
| **Table-first** | 숫자/측정값은 표로만 제공, 문장 삽입 금지 |
| **Validator-gated** | LLM 출력은 반드시 검증 통과 후 사용 |
| **Fail-closed** | 검증 실패 시 템플릿 기반 안전 출력으로 폴백 |

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| Vision AI | MONAI (RetinaNet, DynUNet, UNet) |
| LLM / Report | Upstage Solar Pro 3 (via OpenRouter) |
| Embedding / RAG | Upstage Solar Embedding + ChromaDB |
| API Server | FastAPI + Uvicorn |
| 학습 데이터 | LIDC-IDRI, MSD Task06_Lung |
| GPU | NVIDIA RTX 3070 Ti (8GB VRAM) |

---

## 시스템 아키텍처

```
CT DICOM (zip 또는 디렉터리)
    │
    ▼
Preprocess
  └─ HU clip (-1000 ~ 400)
  └─ Spacing 1mm isotropic resampling
  └─ Orientation 정규화
    │
    ▼
┌──────────────────────────────────────────────┐
│              Vision Pipeline (MONAI)          │
│                                              │
│  ┌───────────────────┐  ┌──────────────────┐ │
│  │  Lung Segmentation│  │ Nodule Detection │ │
│  │  (DynUNet)        │  │ 1순위: LUNA16    │ │
│  │                   │  │   RetinaNet      │ │
│  │                   │  │ 2순위: UNet      │ │
│  │                   │  │   Heatmap        │ │
│  └───────────────────┘  └──────────────────┘ │
└──────────────────────────────────────────────┘
    │
    ▼
Post-Processing
  └─ 3D Peak Detection (NMS)
  └─ Component Extraction → Diameter / Volume 측정
  └─ 폐엽 위치 판정 (RUL / RML / RLL / LUL / LLL / EXTRA)
  └─ Evidence 생성 (slice_range, mask_path, contour_points)
    │
    ▼
┌──────────────────────────────────────────────┐
│          RAG System (Solar Embedding)         │
│  ChromaDB (patient-isolated)                 │
│  → Prior 검색 → 병변 추적 (NEW/STABLE/±)    │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│             Report Generation                 │
│  Template Builder (표/수치 고정)             │
│  → Solar Pro 3 Rewriter (narrative-only)     │
│  → Validator (안전 게이트)                   │
│  → Groundedness Check (Solar Pro 3)          │
└──────────────────────────────────────────────┘
    │
    ▼
Draft Report (Markdown) + Structured AI Result (JSON)
```

---

## Upstage Solar API 활용

| 기능 | 모델 | 설명 |
|------|------|------|
| Report Rewriting | Solar Pro 3 (OpenRouter) | Narrative 텍스트만 다듬기 (표/수치 미전달) |
| Groundedness Check | Solar Pro 3 (OpenRouter) | 환각(hallucination) 방지 검증 |
| Physician Q&A | Solar Pro 3 (OpenRouter) | 의사 질의응답 (한국어 지원) |
| RAG Embedding | Solar Embedding (Upstage) | 과거 리포트 의미 기반 벡터 검색 |

---

## 프로젝트 구조

```
medical-ai-pacs/
├── api/
│   ├── main.py                      # FastAPI 앱 + 엔드포인트
│   └── schemas.py                   # Pydantic 요청/응답 스키마
│
├── monai_pipeline/
│   ├── production_pipeline.py       # 통합 파이프라인 오케스트레이터
│   ├── luna16_detector.py           # LUNA16 RetinaNet 검출기 (주 검출기)
│   ├── nodule_detection.py          # UNet Heatmap 검출기 (대체)
│   ├── lung_segmentation.py         # DynUNet 폐 분할
│   ├── candidate_processor.py       # 후보 처리 + 임계값 정책
│   ├── evidence_generator.py        # Evidence 생성 (slice_range, mask)
│   ├── findings_classifier.py       # Rule-based 소견 분류
│   ├── tracking_engine.py           # 병변 추적 (Prior 비교)
│   ├── calibration.py               # 임계값 관리 + 확률 보정
│   └── data_processing/
│       ├── lidc_preprocessor.py     # LIDC-IDRI 전처리
│       └── msd_loader.py            # MSD Task06_Lung 로더
│
├── solar_integration/
│   ├── report_generator.py          # 리포트 생성 총괄 (fail-closed)
│   ├── templates.py                 # 템플릿 기반 리포트/표 생성
│   ├── rewriter.py                  # Solar Narrative Rewriter
│   ├── validator.py                 # 안전 검증기 (금지 표현/수치 변조 차단)
│   ├── solar_features.py            # Q&A, Groundedness Check
│   ├── embeddings.py                # Solar Embedding 연동
│   ├── rag_system.py                # RAG + ChromaDB (patient-isolated)
│   ├── tracking.py                  # Solar 기반 Prior 추적 보조
│   └── sample_data.py               # 샘플 Prior 시드 데이터
│
├── config/
│   └── settings.py                  # .env 기반 설정
│
├── utils/
│   └── logger.py                    # 중앙 로거
│
├── static/
│   └── index.html                   # 대시보드 UI (6개 탭)
│
├── scripts/
│   ├── train_lung_segmentation.py   # 폐 분할 모델 학습
│   ├── train_nodule_detection.py    # 결절 검출 모델 학습
│   ├── preprocess_lidc.py           # LIDC-IDRI 전처리 실행
│   ├── generate_synthetic_priors.py # Synthetic Prior 생성
│   ├── test_pipeline.py             # 파이프라인 통합 테스트
│   └── run_full_pipeline.py         # 전체 파이프라인 실행
│
├── models/
│   ├── nodule_det/                  # Custom UNet heatmap 가중치
│   │   └── best_nodule_det_model.pth
│   ├── lung_seg/                    # Custom DynUNet 가중치
│   │   └── best_lung_seg_model.pth
│   └── luna16_retinanet/            # MONAI LUNA16 pretrained RetinaNet (주 검출기)
│       └── lung_nodule_ct_detection/
│
├── data/
│   ├── LIDC-IDRI/                   # LIDC 원본 DICOM + XML
│   ├── LIDC-preprocessed-v2/        # LIDC 전처리 결과 (현행)
│   ├── auxneg-preprocessed/         # 보조 음성 샘플 (학습용)
│   ├── Task06_Lung/                 # MSD 폐 분할 학습 데이터
│   ├── lunit_clean/                 # Lunit 외부 검증 데이터
│   ├── dicom_storage/               # API 업로드 임시 저장소 (자동 생성)
│   ├── dicom_output/                # 출력 저장소 (자동 생성)
│   └── chroma_db/                   # ChromaDB 벡터 DB (런타임 생성)
│
├── tests/
│   └── test_integration.py          # 통합 테스트 (mock 모드)
│
├── outputs/                         # 파이프라인 실행 결과
├── logs/                            # 애플리케이션 로그
└── docs/
    └── PROJECT_PLAN.md              # 상세 설계 문서
```

---

## 빠른 시작

### 1. 환경 설정

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 환경 변수 설정 (`.env`)

```env
# Upstage Solar Embedding API
UPSTAGE_API_KEY=your_upstage_api_key

# Solar Pro 3 via OpenRouter (Report 생성)
SOLAR_API_KEY=your_openrouter_api_key
SOLAR_API_ENDPOINT=https://openrouter.ai/api/v1
SOLAR_MODEL=upstage/solar-pro-3:free

# Mock 모드 제어 (API 키 없을 때 개발용)
USE_MOCK_SOLAR=false           # Solar LLM mock 여부
USE_MOCK_EMBEDDING=false       # Embedding mock 여부
USE_MOCK_VISION=true           # Vision 모델 mock 여부 (모델 없을 때 true)
```

> **API 키 발급**
> - Upstage API: https://console.upstage.ai/
> - OpenRouter: https://openrouter.ai/

### 3. 서버 실행

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 대시보드 접속

```
http://localhost:8000/ui
```

---

## API 엔드포인트

| Endpoint | Method | 설명 |
|----------|--------|------|
| `/ui` | GET | 대시보드 UI |
| `/health` | GET | 시스템 상태 확인 |
| `/api/v1/analyze` | POST | CT 분석 + 리포트 생성 |
| `/api/v1/generate-report` | POST | 기존 AI 결과로 리포트 재생성 |
| `/api/v1/qa` | POST | 의사 Q&A (Solar Pro 3) |
| `/api/v1/groundedness-check` | POST | 환각 검증 (Solar Pro 3) |
| `/api/v1/patient/{id}/history` | GET | 환자 이력 조회 (RAG) |
| `/api/v1/semantic-search` | POST | 의미 기반 검색 (Solar Embedding) |
| `/api/v1/lidc-cases` | GET | 사용 가능한 LIDC 케이스 목록 |

### 분석 요청 예시

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "study_uid": "LIDC-IDRI-0001",
    "patient_id": "LIDC-IDRI-0001",
    "include_report": true,
    "include_prior_comparison": true
  }'
```

### Q&A 요청 예시

```bash
curl -X POST http://localhost:8000/api/v1/qa \
  -H "Content-Type: application/json" \
  -d '{
    "question": "과거와 비교해서 결절 크기가 얼마나 변했나요?",
    "patient_id": "LIDC-IDRI-0001",
    "ai_results": {}
  }'
```

---

## 모델 현황

| 모델 | 아키텍처 | 학습 데이터 | 상태 |
|------|----------|------------|------|
| Nodule Detection (주) | LUNA16 RetinaNet (MONAI pretrained) | LUNA16 | 사용 가능 (mAP ~0.852) |
| Nodule Detection (보조) | MONAI UNet Heatmap | LIDC-IDRI | 학습됨 (성능 개선 필요) |
| Lung Segmentation | MONAI DynUNet | MSD Task06_Lung | 학습됨 (Dice ~0.23, 개선 필요) |

### LUNA16 RetinaNet 활성화 방법

```bash
# MONAI Model Zoo에서 다운로드
python -c "
from monai.bundle import download
download('lung_nodule_ct_detection', bundle_dir='models/luna16_retinanet')
"
```

`models/luna16_retinanet/lung_nodule_ct_detection/` 경로가 존재하면 자동으로 활성화됩니다.

---

## 데이터셋

| 데이터셋 | 출처 | 용도 |
|----------|------|------|
| LIDC-IDRI | [Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) | 폐결절 검출 학습/검증 (~200 케이스) |
| MSD Task06_Lung | [Medical Segmentation Decathlon](https://medicaldecathlon.com/) | 폐 분할 학습 (64 케이스) |
| auxneg (보조 음성) | 내부 생성 | 거짓 양성 억제 학습 |
| lunit_clean | 외부 | 외부 검증 |

---

## 리포트 출력 예시

```
DRAFT — Requires physician confirmation
CT Chest | 2026-02-24 | LIDC-IDRI-0001
--------------------------------------------------

QUALITY / LIMITATIONS
- Small nodules (<3 mm) may be underestimated due to resolution limits.
- Several candidates detected; false positives possible.
- AI findings are candidates only and require physician confirmation.

KEY FLAGS
- Nodule candidates: 3
- High-confidence findings: 3

FINDINGS - TABLE
Type             | Location | Status  | Confidence | Evidence
---------------- | -------- | ------- | ---------- | --------
Nodule candidate | RUL      | Present | 0.98       | CAND_001
Nodule candidate | LUL      | Present | 0.87       | CAND_002

MEASUREMENTS - TABLE
Lesion ID | Location | Diameter (mm) | Volume (mm³) | Confidence
--------- | -------- | ------------- | ------------ | ----------
CAND_001  | RUL      | 7.1           | 185.0        | 0.98
CAND_002  | LUL      | 4.3           | 41.6         | 0.87

PRIOR COMPARISON - TABLE
Lesion ID | Prior Date | Change    | Prior Size | Current Size
--------- | ---------- | --------- | ---------- | ------------
CAND_001  | 2025-08-03 | INCREASED | 5.2 mm     | 7.1 mm
CAND_002  | 2025-08-03 | STABLE    | 4.2 mm     | 4.3 mm

AUDIT
- model_version: luna16-retinanet-v1 / nodule-det-v1.0
- pipeline_version: 1.0.0
```

---

## 안전 설계

### Validator 규칙

| 규칙 | 내용 |
|------|------|
| **forbidden_expression** | "diagnosed with", "confirmed malignancy", "prescribe", "recommend surgery" 등 진단 단정/처방 표현 자동 차단 |
| **numeric_preservation** | LLM이 수치를 변경하는 것 차단 |
| **location_preservation** | 위치 정보 변경 차단 |
| **table_integrity** | 표 블록은 byte-identical로 보존 |

### Fail-closed 정책

```python
if not validation.passed:
    return fallback_template_report()   # LLM 없이 안전한 템플릿 출력
```

### QUALITY/LIMITATIONS 자동 생성 조건

| 조건 | 출력 문구 |
|------|----------|
| spacing > 1.0mm | "Small nodules may be underestimated" |
| spacing ≥ 1.5mm | "Small nodules may be missed" |
| coverage < 95% | "Lung coverage adequate but not complete" |
| candidates > 5 | "Several candidates; false positives possible" |
| **항상** | "AI findings are candidates only and require physician confirmation." |

---

## 처리 성능 (RTX 3070 Ti 기준)

| 단계 | 소요 시간 |
|------|-----------|
| Volume Loading | ~2s |
| Nodule Detection (LUNA16 RetinaNet) | ~3–5s |
| Lung Segmentation | ~20s |
| Post-processing | ~5s |
| Report Generation (Solar API 포함) | ~10s |
| **총합** | **~40–45s** |

---

## 학습 실행

### 폐 분할 모델 학습

```bash
python scripts/train_lung_segmentation.py \
  --data_dir data/Task06_Lung \
  --output_dir models/lung_seg \
  --epochs 100
```

### 결절 검출 모델 학습

```bash
python scripts/train_nodule_detection.py \
  --data_dir data/LIDC-preprocessed-v2 \
  --output_dir models/nodule_det \
  --epochs 100
```

### LIDC-IDRI 전처리

```bash
python scripts/preprocess_lidc.py \
  --input_dir data/LIDC-IDRI \
  --output_dir data/LIDC-preprocessed-v2
```

### 전체 모델 일괄 학습

```bash
bash scripts/train_all.sh
```

### 파이프라인 테스트

```bash
python scripts/test_pipeline.py
```

---

## 제한사항

1. **Lung Segmentation**: Best Dice ~0.23으로 병변 위치 판정(폐엽) 부정확 → 일부 결절이 EXTRA로 표시될 수 있음
2. **Small Nodules**: 3mm 미만 검출 정확도 저하
3. **Prior Comparison**: RAG에 등록된 prior가 없는 경우 상세 비교 불가
4. **의사 확인 필수**: 모든 AI 소견은 DRAFT이며 반드시 의사 확인 후 사용
5. **`USE_MOCK_VISION=true`**: 실제 모델 미설치 시 mock heatmap으로 동작 (결과는 테스트용)

---

## 향후 개선 계획

### Phase 2
- [ ] Lung Segmentation 모델 재학습 (목표 Dice: 0.7+)
- [ ] Findings Classifier 양/악성 분류 실구현
- [ ] DICOM SR (Structured Report) 출력 지원

### Phase 3
- [ ] FHIR / HL7 통합
- [ ] DICOM Viewer 연동 (Evidence 하이라이트)
- [ ] 다기관 외부 검증

---

## 참고 자료

- [LIDC-IDRI Dataset](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
- [Medical Segmentation Decathlon](https://medicaldecathlon.com/)
- [MONAI Documentation](https://docs.monai.io/)
- [MONAI Model Zoo — Lung Nodule CT Detection](https://monai.io/model-zoo.html)
- [Upstage Solar API](https://www.upstage.ai/)
- [OpenRouter](https://openrouter.ai/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

---

---

## 프로젝트 위치 (R&D 로드맵)

본 프로젝트는 **㈜메디칼스탠다드 차세대 PACS 플랫폼 R&D** (RS-2024-00470062) 중 **1차년도 AI 오케스트레이터 에이전트 개발**의 선행 PoC입니다.

```
2025 (완료)           2026 (1차년도, 현재)            2027~2028
─────────────         ────────────────────────        ──────────────────
PACS 기반             AI 오케스트레이터 에이전트       Cloud 배포·해외진출
• PPWeb 뷰어           (LangGraph 멀티에이전트)         • Kubernetes 배포
• AI Gateway           • Agent 1: 멀티모달 DICOM 분석  • Web API 원격판독
• AI SaMD 연동 PoC    • Agent 2: AI SaMD 동적 라우팅  • FHIR/HL7
• Rule-based 분석      • Agent 3: 다중 AI 앙상블       • 해외 AI SaMD 연동
                       • Agent 4: RAG 판독문 생성
                              ↑
                  이 프로젝트: Agent 4 PoC
                  (Solar Embedding + Solar Pro 3
                   기반 RAG 리포트 파이프라인 검증)
```

최종 목표 아키텍처(LangGraph StateGraph + 멀티모달 VLM + 루닛/뷰노 등 외부 AI SaMD 연동)로 통합되기 전, **RAG + 리포트 생성 파이프라인**을 먼저 검증하기 위한 독립형 데모입니다.

**Developed as part of 메디칼스탠다드 R&D / Upstage Solar API PoC**
