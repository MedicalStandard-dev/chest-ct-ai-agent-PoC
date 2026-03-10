# Medical AI PACS — 상세 기획서

> **Version**: 2.1
> **Last Updated**: 2026-02-24
> **Status**: Demo / PoC (Production-oriented)

---

## 1. 프로젝트 개요

### 1.1 목표
CT Chest 영상에서 폐결절을 자동 검출하고, **Upstage Solar API**를 활용하여 구조화된 판독 보조 리포트를 생성하는 제품형 시스템 구축.

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
| Report Rewriting | Solar Pro 3 (OpenRouter) | Narrative-only 다듬기 | ✅ 완료 |
| Groundedness Check | Solar Pro 3 (OpenRouter) | 환각 검증 | ✅ 완료 |
| Physician Q&A | Solar Pro 3 (OpenRouter) | 의사 질의응답 (한국어) | ✅ 완료 |
| RAG Embedding | Solar Embedding (Upstage API) | 과거 리포트 검색 | ✅ 완료 |

---

## 2. 시스템 아키텍처

### 2.1 전체 파이프라인

```
CT Volume
    │
    ▼
Preprocess (HU clip, spacing 1mm iso, orientation)
    │
    ▼
┌──────────────────────────────────────────┐
│         Vision Pipeline (MONAI)           │
│  Lung Segmentation (DynUNet)             │
│  Nodule Detection:                        │
│    1순위: LUNA16 RetinaNet (pretrained)  │
│    2순위: UNet Heatmap (custom trained)  │
└──────────────────────────────────────────┘
    │
    ▼
Post-Processing
  Peak Detection → Component Extraction
  → Measurements → Location (lung mask)
  → Evidence Generation
    │
    ▼
Prior Tracking (있는 경우)
    │
    ▼
┌──────────────────────────────────────────┐
│         RAG System (Solar Embedding)      │
│  ChromaDB (patient-isolated)             │
│  → Prior Retrieval → Lesion Tracking     │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│         Report Generation                 │
│  Template Builder (Tables fixed)         │
│  → Solar Pro 3 Rewriter (narrative-only) │
│  → Validator (safety gate)               │
│  → Groundedness Check                    │
└──────────────────────────────────────────┘
    │
    ▼
Draft Report + Structured AI Result (JSON)
```

### 2.2 모듈 구조

```
medical-ai-pacs/
├── api/                        # FastAPI 앱
│   ├── main.py
│   └── schemas.py
│
├── monai_pipeline/             # Vision 모델
│   ├── production_pipeline.py  # 오케스트레이터
│   ├── luna16_detector.py      # LUNA16 RetinaNet (주 검출기)
│   ├── nodule_detection.py     # UNet heatmap (보조)
│   ├── lung_segmentation.py    # DynUNet 폐 분할
│   ├── candidate_processor.py  # 후보 처리 + 임계값 정책
│   ├── evidence_generator.py   # Evidence 생성
│   ├── findings_classifier.py  # Rule-based 소견 분류
│   ├── tracking_engine.py      # 병변 추적
│   ├── calibration.py          # 임계값 관리
│   └── data_processing/
│       ├── lidc_preprocessor.py
│       └── msd_loader.py
│
├── solar_integration/          # Solar API 연동
│   ├── report_generator.py     # 리포트 생성 총괄
│   ├── templates.py            # 템플릿
│   ├── rewriter.py             # Narrative Rewriter
│   ├── validator.py            # 안전 검증기
│   ├── solar_features.py       # Q&A, Groundedness
│   ├── embeddings.py           # Solar Embedding
│   ├── rag_system.py           # RAG + ChromaDB
│   ├── tracking.py             # Solar 기반 Prior 추적 보조
│   └── sample_data.py          # 샘플 데이터
│
├── config/settings.py
├── utils/logger.py
├── static/index.html
└── models/
    ├── nodule_det/              # Custom UNet heatmap
    ├── lung_seg/                # Custom DynUNet
    └── luna16_retinanet/        # MONAI LUNA16 pretrained (주 검출기)
```

---

## 3. 구현 상세

### 3.1 Vision Pipeline

#### 3.1.1 Nodule Detection — LUNA16 RetinaNet (주 검출기)
- **아키텍처**: ResNet50 + FPN + RetinaNet (3D), MONAI pretrained
- **학습**: LUNA16 (mAP ~0.852 @ fold0)
- **모델 위치**: `models/luna16_retinanet/lung_nodule_ct_detection/`
- **사용 조건**: `luna16_bundle_dir` 경로 존재 시 자동 활성화

#### 3.1.2 Nodule Detection — UNet Heatmap (보조/대체)
- **아키텍처**: MONAI UNet
- **입력**: CT Volume (1mm isotropic), **출력**: 3D Heatmap (0~1)
- **학습 데이터**: LIDC-IDRI + auxneg 보조 음성 샘플
- **모델 위치**: `models/nodule_det/best_nodule_det_model.pth`
- **한계**: 현재 성능 개선 필요

#### 3.1.3 Lung Segmentation
- **아키텍처**: MONAI DynUNet
- **학습 데이터**: MSD Task06_Lung
- **모델 위치**: `models/lung_seg/best_lung_seg_model.pth`
- **한계**: Best Dice ~0.23으로 위치 판정 부정확 (개선 필요)

### 3.2 Post-Processing Pipeline

#### Candidate Processor 임계값 정책
```python
ThresholdPolicy:
    peak_threshold: 0.97           # Peak 검출 임계값
    min_diameter_mm: 3.0           # 최소 직경
    max_diameter_mm: 30.0          # 최대 직경
    min_voxel_count: 20            # 최소 복셀 수
    search_radius_mm: 15.0         # 검색 반경
    high_confidence_threshold: 0.9
    nodule_detection_threshold: 0.15  # 하한
    nodule_reporting_threshold: 0.75  # 리포트 포함 기준
```

#### 측정값 계산
- **Diameter**: Bounding box + Spherical volume 가중 평균
- **Volume**: Component voxel count × spacing³
- **Location**: Lung mask 기반 폐엽 판정 (RUL/RML/RLL/LUL/LLL/EXTRA)

#### Evidence Generation
- `slice_range`: 결절이 존재하는 z 슬라이스 범위
- `mask_path`: Component mask 파일
- `contour_points`: 윤곽점 좌표

### 3.3 RAG System

#### Solar Embedding
- **모델**: solar-embedding-1-large (Upstage API 직접)
- **저장소**: ChromaDB (persistent)
- **격리**: Patient ID 기반 collection 분리

#### Prior Comparison
```python
# 과거 검사 검색 → 병변 추적
prior = rag_system.retrieve_most_recent_prior(patient_id, current_date)
tracking_result = tracker.compare_studies(current, prior)
# → NEW, STABLE, INCREASED, DECREASED
```

### 3.4 Report Generation

#### Template Builder
```
QUALITY / LIMITATIONS  ← 자동 생성 (spacing/coverage/candidate 수 기반)
FINDINGS - TABLE       ← Vision 결과 기반 (LLM 미전달)
MEASUREMENTS - TABLE   ← Vision 결과 기반 (LLM 미전달)
PRIOR COMPARISON       ← RAG + tracking 기반
KEY FLAGS              ← 자동 집계
NOTES                  ← Solar Rewriter 대상
AUDIT                  ← 자동 생성
```

#### Solar Pro 3 Rewriter (narrative-only)
```python
# QUALITY, NOTES, HEADER 텍스트만 추출 후 Solar 전달
# 표/수치는 전혀 전달하지 않음
narratives = extractor.extract(report)
rewritten = await solar.rewrite(narratives)
```

#### Validator
```python
FORBIDDEN_PATTERNS = [
    "diagnosed with", "confirmed malignancy",
    "prescribe", "recommend surgery", ...
]
# 금지 표현 차단 + 수치 보존 + 위치 정보 보존 + 표 무결성
```

---

## 4. API 명세

### POST /api/v1/analyze
```json
// 요청
{
    "study_uid": "LIDC-IDRI-0001",
    "patient_id": "LIDC-IDRI-0001",
    "include_report": true,
    "include_prior_comparison": true
}

// 응답
{
    "request_id": "uuid",
    "status": "completed",
    "structured_ai_result": { "nodules": [...], "quality": {...} },
    "draft_report": { "rendered_report": "...", "validation_passed": true },
    "groundedness": { "is_grounded": true },
    "solar_features_used": ["Report Generation (Solar Pro 3)", ...]
}
```

### POST /api/v1/qa
```json
// 요청
{ "question": "과거와 비교해서 크기가 얼마나 커졌나요?", "patient_id": "...", "ai_results": {...} }

// 응답
{ "answer": "AI가 분석한 결과...", "confidence": 0.85 }
```

---

## 5. 데이터 스키마 (핵심)

```python
class StructuredAIResult(BaseModel):
    study_uid: str
    nodules: List[NoduleCandidate]       # finding 기준 통과 후보
    low_confidence_nodules: List[...]    # 기준 미달 후보 (hidden)
    quality: QualityMetrics
    findings: StructuredFindings
    versioning: ModelVersioning

class NoduleCandidate(BaseModel):
    id: str                             # CAND_001 등
    center_zyx: tuple[float, float, float]
    bbox_zyx: tuple[int, ...]
    diameter_mm: float
    volume_mm3: float
    confidence: float
    evidence: VisionEvidence            # slice_range, mask_path
    location_code: str                  # RUL/RML/RLL/LUL/LLL/EXTRA
```

---

## 6. UI Dashboard

| 탭 | 설명 |
|----|------|
| FINDINGS | 검출된 소견 테이블 |
| MEASUREMENTS | 측정값 테이블 |
| PRIOR | 과거 검사 비교 (RAG) |
| REPORT | AI 생성 리포트 |
| Q&A | 의사 질의응답 |
| JSON | Raw 결과 |

---

## 7. 안전 설계

### Validator Rules

| 규칙 | 설명 |
|------|------|
| forbidden_expression | 진단 단정/처방 등 금지 표현 차단 |
| numeric_preservation | 수치 변경 차단 |
| location_preservation | 위치 정보 변경 차단 |
| table_integrity | 표 블록 byte-identical 보존 |

### Fail-closed Policy
```python
if not validation.passed:
    return fallback_template_report()  # 안전한 템플릿 사용
```

### QUALITY/LIMITATIONS 자동 생성 조건

| 조건 | 출력 문구 |
|------|----------|
| spacing > 1.0mm | "Small nodules may be underestimated" |
| spacing ≥ 1.5mm | "Small nodules may be missed" |
| coverage < 95% | "Lung coverage adequate but not complete" |
| candidates > 5 | "Several candidates; false positives possible" |
| **항상** | "AI findings are candidates only..." |

---

## 8. 성능 및 제한사항

### 처리 시간 (RTX 3070 Ti 기준)

| 단계 | 시간 |
|------|------|
| Volume Loading | ~2s |
| Nodule Detection (LUNA16) | ~3–5s |
| Lung Segmentation | ~20s |
| Post-processing | ~5s |
| Report Generation | ~10s |
| **총합** | ~40–45s |

### 알려진 제한사항

1. **Lung Segmentation**: Dice ~0.23으로 위치 판정 부정확 → 일부 결절이 EXTRA로 표시될 수 있음
2. **Small Nodules**: 3mm 미만 검출 정확도 저하
3. **Prior Comparison**: AI result가 저장되지 않은 prior는 상세 비교 불가
4. **USE_MOCK_VISION=true**: 실제 모델 없을 때 mock heatmap으로 동작

---

## 9. 향후 개선 계획

### Phase 2
- [ ] Lung Segmentation 모델 재학습 (Dice 목표: 0.7+)
- [ ] Findings Classifier (양/악성 분류) 실구현
- [ ] DICOM SR 출력 지원

### Phase 3
- [ ] FHIR/HL7 통합
- [ ] Viewer 연동 (Evidence 하이라이트)
- [ ] 다기관 검증

---

## 10. 참고 자료

- [LIDC-IDRI Dataset](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
- [MONAI Documentation](https://docs.monai.io/)
- [MONAI Model Zoo — Lung Nodule CT Detection](https://monai.io/model-zoo.html)
- [Upstage Solar API](https://www.upstage.ai/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

---

**End of Document**
