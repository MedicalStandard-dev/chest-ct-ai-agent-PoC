# chest-ct-ai-agent-PoC

> CT Chest 영상에서 폐결절을 자동 검출하고, 구조화된 판독 보조 리포트를 생성하는 AI 에이전트 PoC

**MONAI + Upstage Solar Pro 3 기반 의료 영상 AI 판독 보조 시스템**

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| Vision AI | MONAI (LUNA16 RetinaNet, DynUNet) |
| LLM / Report | Upstage Solar Pro 3 (via OpenRouter) |
| Embedding / RAG | Upstage Solar Embedding + ChromaDB |
| API Server | FastAPI + Uvicorn |
| 학습 데이터 | LIDC-IDRI, MSD Task06_Lung |

---

## 시스템 아키텍처

```
CT DICOM
    ↓
Preprocess (HU clip / Spacing 1mm / Orientation)
    ↓
Lung Segmentation (DynUNet)
    ↓
Nodule Detection (LUNA16 RetinaNet → UNet Heatmap fallback)
    ↓
Post-processing (NMS / Diameter·Volume 측정 / 폐엽 위치 판정 / Evidence 생성)
    ↓
RAG (Solar Embedding + ChromaDB, patient-isolated) → Prior 비교
    ↓
Report (Template → Solar Pro 3 Rewriter → Validator)
    ↓
Draft Report (Markdown) + Structured AI Result (JSON)
```

---

## 핵심 설계 원칙

| 원칙 | 설명 |
|------|------|
| **Evidence-first** | 모든 소견은 Vision 모델 출력(heatmap/검출)에 기반 |
| **Table-first** | 숫자/측정값은 표로만 제공, LLM 문장 삽입 금지 |
| **Fail-closed** | Validator 실패 시 LLM 없이 템플릿 리포트로 폴백 |
| **LLM 역할 제한** | Narrative 텍스트 다듬기만 허용, 진단·수치·소견 변경 불가 |

---

## 빠른 시작

```bash
# 1. 환경 설정
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. .env 설정
UPSTAGE_API_KEY=your_upstage_api_key
SOLAR_API_KEY=your_openrouter_api_key
SOLAR_API_ENDPOINT=https://openrouter.ai/api/v1
SOLAR_MODEL=upstage/solar-pro-3:free
USE_MOCK_VISION=true   # 모델 미설치 시

# 3. 서버 실행
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 4. 대시보드
http://localhost:8000/ui
```

> API 키 발급: [Upstage Console](https://console.upstage.ai/) · [OpenRouter](https://openrouter.ai/)

---

## API 엔드포인트

| Endpoint | Method | 설명 |
|----------|--------|------|
| `/ui` | GET | 대시보드 UI |
| `/health` | GET | 시스템 상태 |
| `/api/v1/analyze` | POST | CT 분석 + 리포트 생성 |
| `/api/v1/generate-report` | POST | 기존 AI 결과로 리포트 재생성 |
| `/api/v1/qa` | POST | 의사 Q&A (Solar Pro 3) |
| `/api/v1/groundedness-check` | POST | 환각 검증 |
| `/api/v1/patient/{id}/history` | GET | 환자 이력 조회 |
| `/api/v1/semantic-search` | POST | 의미 기반 검색 |

---

## 모델 현황

| 모델 | 아키텍처 | 상태 |
|------|----------|------|
| Nodule Detection (주) | LUNA16 RetinaNet (MONAI pretrained) | ✅ 사용 가능 (mAP ~0.852) |
| Nodule Detection (보조) | MONAI UNet Heatmap | ⚠️ 학습됨 (성능 개선 필요) |
| Lung Segmentation | MONAI DynUNet | ⚠️ 학습됨 (Dice ~0.23, 개선 필요) |

```bash
# LUNA16 RetinaNet 다운로드
python -c "from monai.bundle import download; download('lung_nodule_ct_detection', bundle_dir='models/luna16_retinanet')"
```

---

## 프로젝트 구조

```
├── api/                    # FastAPI 앱 + Pydantic 스키마
├── monai_pipeline/         # Vision 파이프라인 (검출·분할·후처리)
├── solar_integration/      # RAG, 리포트 생성, Validator
├── config/settings.py      # 중앙 설정 (.env 기반)
├── scripts/                # 학습·전처리·테스트 스크립트
├── static/index.html       # 대시보드 UI
└── models/                 # 모델 가중치 (git 미추적)
```

---

## 제한사항

- Lung Segmentation Dice ~0.23 → 폐엽 위치 판정 부정확 (일부 결절 EXTRA 표시)
- 3mm 미만 소결절 검출 정확도 저하
- Prior RAG 미등록 시 이전 비교 불가
- **모든 AI 소견은 DRAFT이며 반드시 의사 확인 필요**

---

## 향후 계획

- [ ] Lung Segmentation 재학습 (목표 Dice 0.7+)
- [ ] Findings Classifier 악성도 분류 실구현
- [ ] DICOM SR 출력 지원
- [ ] FHIR / HL7 통합

---

## 프로젝트 위치

본 프로젝트는 **㈜메디칼스탠다드 차세대 PACS 플랫폼 R&D** (RS-2024-00470062) 1차년도의 선행 PoC입니다.
RAG 기반 리포트 생성 파이프라인(Agent 4)을 검증하기 위한 독립형 데모입니다.

---

*Developed as part of 메디칼스탠다드 R&D / Upstage Solar API PoC*
