# Medical AI PACS - 상세 기획서

> **Version**: 2.0  
> **Last Updated**: 2026-02-03  
> **Status**: Production Ready (Demo)

---

## 1. 프로젝트 개요

### 1.1 목표
CT Chest 영상에서 폐결절을 자동 검출하고, **Upstage Solar API**를 활용하여 구조화된 판독 보조 리포트를 생성하는 제품형 시스템 구축

### 1.2 핵심 설계 원칙

| 원칙 | 설명 | 구현 상태 |
|------|------|----------|
| **Evidence-first** | 모든 소견은 Vision 모델 출력에 기반 | ✅ 완료 |
| **Table-first** | 숫자/측정값은 표로만 제공 | ✅ 완료 |
| **Validator-gated** | LLM 출력은 반드시 검증 후 사용 | ✅ 완료 |
| **Fail-closed** | 검증 실패 시 템플릿 기반 안전 출력 | ✅ 완료 |

### 1.3 Upstage Solar 활용 현황

| 기능 | API | 설명 | 상태 |
|------|-----|------|------|
| Report Rewriting | Solar Pro 3 | Narrative-only 다듬기 | ✅ 완료 |
| Groundedness Check | Solar Pro 3 | 환각 검증 | ✅ 완료 |
| Physician Q&A | Solar Pro 3 | 의사 질의응답 (한국어) | ✅ 완료 |
| RAG Embedding | Solar Embedding | 과거 리포트 검색 | ✅ 완료 |

---

## 2. 시스템 아키텍처

### 2.1 전체 파이프라인

```
CT Volume
    │
    ▼
┌─────────────────────────────────────────┐
│         Vision Pipeline (MONAI)          │
│  ┌─────────────┐    ┌─────────────────┐ │
│  │ Lung Seg    │    │ Nodule Det      │ │
│  │ (DynUNet)   │    │ (UNet Heatmap)  │ │
│  └─────────────┘    └─────────────────┘ │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│         Post-Processing Pipeline         │
│  Peak Detection → Component Extraction   │
│  → Measurements → Location → Evidence    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│         RAG System (Solar Embedding)     │
│  ChromaDB → Prior Retrieval → Tracking   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│         Report Generation                │
│  Template → Solar Rewrite → Validate     │
└─────────────────────────────────────────┘
    │
    ▼
Draft Report + Structured AI Result
```

### 2.2 모듈 구조

```
medical-ai-pacs/
├── api/                    # FastAPI 앱
│   ├── main.py            # 엔드포인트 정의
│   └── schemas.py         # Pydantic 스키마
│
├── monai_pipeline/         # Vision 모델
│   ├── nodule_detection.py    # 결절 검출
│   ├── lung_segmentation.py   # 폐 분할
│   ├── candidate_processor.py # 후보 처리
│   ├── evidence_generator.py  # Evidence 생성
│   ├── tracking_engine.py     # 병변 추적
│   └── production_pipeline.py # 통합 파이프라인
│
├── solar_integration/      # Solar API 연동
│   ├── embeddings.py      # Solar Embedding
│   ├── rag_system.py      # RAG + ChromaDB
│   ├── rewriter.py        # Narrative Rewriter
│   ├── validator.py       # 안전 검증기
│   ├── templates.py       # 리포트 템플릿
│   ├── report_generator.py # 리포트 생성기
│   ├── solar_features.py  # Q&A, Groundedness
│   └── sample_data.py     # 샘플 데이터
│
├── config/                 # 설정
│   └── settings.py
│
├── static/                 # UI
│   └── index.html
│
└── models/                 # 학습된 모델
    ├── nodule_det/
    └── lung_seg/
```

---

## 3. 구현 상세

### 3.1 Vision Pipeline

#### 3.1.1 Nodule Detection
- **아키텍처**: MONAI UNet
- **입력**: CT Volume (1mm isotropic)
- **출력**: 3D Heatmap (0~1)
- **학습 데이터**: LIDC-IDRI (49 cases)
- **성능**: Best Loss 0.23

#### 3.1.2 Lung Segmentation
- **아키텍처**: MONAI DynUNet
- **입력**: CT Volume
- **출력**: Binary Lung Mask
- **학습 데이터**: MSD Task06_Lung
- **성능**: Best Dice 0.23 (개선 필요)

### 3.2 Post-Processing Pipeline

#### 3.2.1 Candidate Processor
```python
ThresholdPolicy:
    peak_threshold: 0.97        # Peak 검출 임계값
    min_diameter_mm: 3.0        # 최소 직경
    max_diameter_mm: 30.0       # 최대 직경
    min_voxel_count: 20         # 최소 복셀 수
    search_radius_mm: 15.0      # 검색 반경
    adaptive_threshold_ratio: 0.5
    high_confidence_threshold: 0.9
```

#### 3.2.2 측정값 계산
- **Diameter**: Bounding box + Spherical volume 가중 평균
- **Volume**: Component voxel count × spacing³
- **Location**: Lung mask 기반 폐엽 판정

#### 3.2.3 Evidence Generation
- slice_range: 결절이 존재하는 z 범위
- mask_path: Component mask 파일
- contour_points: 윤곽점 좌표

### 3.3 RAG System

#### 3.3.1 Solar Embedding
- **모델**: solar-embedding-1-large (1024차원)
- **저장소**: ChromaDB
- **격리**: Patient ID 기반 collection 분리

#### 3.3.2 Prior Comparison
```python
# 과거 검사 검색
prior = rag_system.retrieve_most_recent_prior(patient_id, current_date)

# 병변 추적
tracking_result = tracker.compare_studies(current, prior)
# → NEW, STABLE, INCREASED, DECREASED
```

### 3.4 Report Generation

#### 3.4.1 Template Builder
```
QUALITY / LIMITATIONS
- Resolution-based: spacing > 1.0mm → small nodule 경고
- Coverage-based: coverage < 95% → incomplete 경고
- Candidate-based: >5개 → false positive 경고
- 항상: "AI findings are candidates only..."

FINDINGS - TABLE
MEASUREMENTS - TABLE
PRIOR COMPARISON - TABLE
KEY FLAGS
NOTES
AUDIT
```

#### 3.4.2 Solar Pro 3 Rewriter
```python
# Narrative-only 모드
# 표/수치는 LLM에 전달하지 않음
narratives = extractor.extract(report)
# → QUALITY, NOTES, HEADER만 추출

rewritten = await solar.rewrite(narratives)
# → 문장만 다듬음, 수치 변경 불가
```

#### 3.4.3 Validator
```python
# 금지 표현 검사
FORBIDDEN_PATTERNS = [
    "diagnosed with",
    "confirmed malignancy",
    "prescribe",
    "recommend surgery",
    ...
]

# 수치 보존 검사
# LLM 출력에서 수치가 변경되면 차단
```

### 3.5 Solar Features

#### 3.5.1 Groundedness Check
```python
# 리포트 내용이 AI 결과에 근거하는지 검증
result = await solar.check_groundedness(
    claim=report_text,
    context=ai_results
)
# → is_grounded: True/False
```

#### 3.5.2 Physician Q&A
```python
# 의사 질문에 한국어로 답변
# 과거 검사 기록도 컨텍스트에 포함
result = await solar.answer_question(
    question="과거와 비교해서 크기가 얼마나 커졌나요?",
    ai_results=current_results,
    prior_data=prior_data
)
```

---

## 4. API 명세

### 4.1 CT 분석

```http
POST /api/v1/analyze
Content-Type: application/json

{
    "study_uid": "LIDC-IDRI-0001",
    "series_uid": "1.3.6.1...",
    "patient_id": "LIDC-IDRI-0001",
    "include_report": true,
    "include_prior_comparison": true
}
```

**응답:**
```json
{
    "request_id": "uuid",
    "status": "completed",
    "structured_ai_result": {
        "study_uid": "...",
        "nodules": [...],
        "quality": {...},
        "findings": {...}
    },
    "draft_report": {
        "rendered_report": "...",
        "tables": {...},
        "validation_passed": true
    },
    "groundedness": {
        "is_grounded": true,
        "confidence": 0.95
    },
    "solar_features_used": [
        "Report Generation (Solar Pro 3)",
        "Groundedness Check (Solar Pro 3)"
    ]
}
```

### 4.2 Physician Q&A

```http
POST /api/v1/qa
Content-Type: application/json

{
    "question": "이 결절이 왜 의심되나요?",
    "patient_id": "LIDC-IDRI-0001",
    "ai_results": {...}
}
```

**응답:**
```json
{
    "question": "이 결절이 왜 의심되나요?",
    "answer": "AI가 결절을 의심하게 된 이유는 다음과 같습니다...",
    "sources": ["Nodule Detection", "AI Confidence Scores"],
    "confidence": 0.85
}
```

### 4.3 환자 이력 조회

```http
GET /api/v1/patient/{patient_id}/history?max_results=5
```

---

## 5. 데이터 스키마

### 5.1 StructuredAIResult
```python
class StructuredAIResult(BaseModel):
    study_uid: str
    series_uid: str
    acquisition_datetime: datetime
    quality: QualityMetrics
    lung_volume_ml: float
    nodules: List[NoduleCandidate]
    findings: StructuredFindings
    versioning: ModelVersioning
    processing_time_seconds: float
```

### 5.2 NoduleCandidate
```python
class NoduleCandidate(BaseModel):
    id: str
    center_zyx: tuple[float, float, float]
    bbox_zyx: tuple[int, int, int, int, int, int]
    diameter_mm: float
    volume_mm3: float
    confidence: float  # 0.0 ~ 1.0
    evidence: VisionEvidence
    location_code: str  # RUL, RML, RLL, LUL, LLL, EXTRA
```

### 5.3 DraftReport
```python
class DraftReport(BaseModel):
    sections: ReportSections
    tables: ReportTables
    audit: AuditInfo
    rendered_report: str
    validation_passed: bool
    validation_warnings: List[str]
```

---

## 6. UI Dashboard

### 6.1 기능

| 탭 | 설명 |
|----|------|
| FINDINGS | 검출된 소견 테이블 |
| MEASUREMENTS | 측정값 테이블 |
| PRIOR | 과거 검사 비교 (RAG) |
| REPORT | AI 생성 리포트 |
| Q&A | 의사 질의응답 |
| JSON | Raw 결과 |

### 6.2 Solar Features 표시

- Solar Pro 3 사용 현황
- Groundedness 결과 (PASS/FAIL)
- Q&A 인터페이스

---

## 7. 안전 설계

### 7.1 Validator Rules

| 규칙 | 설명 |
|------|------|
| forbidden_expression | 금지 표현 차단 |
| numeric_preservation | 수치 변경 차단 |
| location_preservation | 위치 정보 변경 차단 |
| hallucination_check | 근거 없는 내용 차단 |

### 7.2 Fail-closed Policy

```python
if not validation.passed:
    return fallback_template_report()  # 안전한 템플릿 사용
```

### 7.3 QUALITY/LIMITATIONS 자동 생성

| 조건 | 출력 |
|------|------|
| spacing > 1.0mm | "Small nodules may be underestimated" |
| spacing ≥ 1.5mm | "Small nodules may be missed" |
| coverage < 95% | "Lung coverage adequate but not complete" |
| candidates > 5 | "Several candidates; false positives possible" |
| candidates > 10 | "Multiple candidates; high FP rate" |
| **항상** | "AI findings are candidates only..." |

---

## 8. 성능 및 제한사항

### 8.1 처리 시간

| 단계 | 시간 |
|------|------|
| Volume Loading | ~2s |
| Nodule Detection | ~3s |
| Lung Segmentation | ~20s |
| Post-processing | ~5s |
| Report Generation | ~10s |
| **Total** | ~40s |

### 8.2 알려진 제한사항

1. **Lung Segmentation**: Dice 0.23으로 location 판정 부정확
2. **Small Nodules**: 3mm 미만 검출 정확도 저하
3. **Location "EXTRA"**: Lung mask 외부로 판정되는 경우 있음
4. **Prior Comparison**: AI result가 저장되지 않은 prior는 상세 비교 불가

---

## 9. 향후 개선 계획

### Phase 2 (예정)
- [ ] Lung Segmentation 모델 재학습
- [ ] Findings Classifier (양/악성 분류)
- [ ] DICOM SR 출력 지원
- [ ] Multi-GPU 추론

### Phase 3 (예정)
- [ ] FHIR/HL7 통합
- [ ] Viewer 연동 (Evidence 하이라이트)
- [ ] 다기관 검증

---

## 10. 참고 자료

- [LIDC-IDRI Dataset](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
- [MONAI Documentation](https://docs.monai.io/)
- [Upstage Solar API](https://www.upstage.ai/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

---

**End of Document**
