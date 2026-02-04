# Medical AI PACS 프로젝트 이해 보고서

작성일: 2026-02-04  
대상 경로: `/home/jiwoonkim/jiwoon/medical-ai-pacs`

## 1) 프로젝트 한 줄 요약
이 프로젝트는 **CT 흉부 영상에서 폐결절 후보를 탐지하고(MONAI), 근거 기반 표 중심 리포트를 생성하며(Solar 연동), 검증 게이트로 안전성을 보강**하는 데모/프로덕션 지향 PACS AI 시스템입니다.

## 2) 핵심 목표와 설계 원칙
코드/문서에서 반복적으로 확인되는 핵심 원칙은 아래 4가지입니다.

1. **Evidence-first**: 모든 소견은 Vision 결과(heatmap/evidence)에서 출발
2. **Table-first**: 수치/측정값은 표에서 관리
3. **Validator-gated**: LLM 출력은 검증 통과 시에만 반영
4. **Fail-closed**: 위험 시 템플릿 기반 안전 출력으로 폴백

즉, LLM은 “의료 판단 생성기”가 아니라, **표 밖 내러티브 문장 다듬기 도구**로 제한되어 있습니다.

## 3) 현재 코드 기준 아키텍처 이해

### A. API 계층 (`api/main.py`)
- FastAPI 기반
- 주요 엔드포인트:
  - `/api/v1/analyze`: 분석 + (옵션) 리포트 생성
  - `/api/v1/generate-report`: 기존 AI 결과로 리포트 생성
  - `/api/v1/semantic-search`, `/api/v1/patient/{id}/history`: RAG
  - `/api/v1/qa`, `/api/v1/translate`, `/api/v1/prior-summary`, `/api/v1/groundedness-check`: Solar 기능
- 서버 시작 시 RAG/Report Generator/Threshold/Calibrator 초기화 및 샘플 prior seed 시도

### B. Vision 파이프라인 (`monai_pipeline/`)
- `production_pipeline.py`가 중심 오케스트레이터
- 흐름:
  1) Heatmap 생성(모델 추론 또는 mock)
  2) Candidate 처리(peak/component/measurement/status)
  3) Evidence 생성(mask/slice range)
  4) Prior tracking(옵션)
  5) Findings/Measurements/Prior 테이블 생성
- `candidate_processor.py`의 임계값 정책으로 finding/limitation/hidden 분류

### C. Report 생성 (`solar_integration/`)
- `templates.py`: 템플릿 기반 report/table/audit 생성
- `rewriter.py`: narrative-only 추출 후 Solar 호출(표 보호)
- `validator.py`: 금지표현/수치변경/위치변경/표무결성 검증
- `report_generator.py`: 전체 생성 파이프라인 통합 및 fail-closed 적용

### D. RAG 계층 (`solar_integration/rag_system.py`)
- ChromaDB persistent 저장
- 환자 단위 격리(patient isolation) 필수
- 리포트/AI 결과 저장, 환자 이력 조회, semantic search, prior 비교 지원

## 4) 구현 상태 요약 (코드 관찰 기준)

### 구현 강점
- 엔드투엔드 흐름(분석→리포트→검증→RAG)이 코드로 연결되어 있음
- 안전장치(Validator + fallback) 구조가 명확함
- 모듈 분리가 좋고, 책임 경계가 비교적 선명함
- 실제 데이터 경로(`data/processed/lidc`)와 출력 경로(`outputs/`)가 이미 운용 중

### 현재 주의/제한 포인트
- `USE_MOCK_VISION` 기본값이 true여서 환경에 따라 mock 경로로 동작 가능
- Findings 분류기는 현재 `MockFindingsClassifier`가 기본 사용
- 문서상 “Production Ready” 표현이 있으나, 일부 모듈은 placeholder/TODO 흔적 존재
- 리포지토리에 대용량 데이터/venv가 포함되어 파일 수가 매우 큼(운영/협업 시 관리 이슈 가능)

## 5) 프로젝트 구조 핵심
- `api/`: FastAPI 앱/스키마
- `monai_pipeline/`: 영상 분석 및 후처리
- `solar_integration/`: LLM/RAG/검증/리포트
- `config/settings.py`: .env 기반 실행 설정
- `scripts/`: 학습/파이프라인 테스트 스크립트
- `tests/test_integration.py`: mock 중심 통합 테스트

## 6) 실행 및 운영 관점 요약
- 실행: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- UI: `/ui`
- 설정 핵심: `.env`의 `SOLAR_API_KEY`, `USE_MOCK_SOLAR`, `USE_MOCK_EMBEDDING`, `USE_MOCK_VISION`
- 실제 운영 전 체크 포인트:
  1) mock 플래그 의도 확인
  2) 모델 파일 존재 확인(`models/nodule_det`, `models/lung_seg`)
  3) RAG 저장소(`data/chroma_db`) 백업/정리 정책 수립

## 7) 결론
현재 프로젝트는 **“의료 영상 후보 검출 + 안전한 리포트 생성”이라는 제품 컨셉이 코드 레벨에서 잘 구현된 상태**입니다.  
특히 **표 보호 + 검증 게이트 + 폴백** 구조가 안정성 측면에서 강점입니다.  
다만, 실제 임상/상용 전개를 위해서는 mock 경로 정리, 일부 placeholder 모듈 실구현, 대용량 리포지토리 정리 전략이 다음 우선 과제로 보입니다.

---

## 부록: 이번 분석 범위
본 보고서는 아래 주요 파일을 직접 확인해 작성했습니다.
- `README.md`, `docs/PROJECT_PLAN.md`
- `api/main.py`, `api/schemas.py`
- `config/settings.py`
- `monai_pipeline/production_pipeline.py`, `monai_pipeline/candidate_processor.py`, `monai_pipeline/evidence_generator.py`, `monai_pipeline/tracking_engine.py`, `monai_pipeline/findings_classifier.py`, `monai_pipeline/calibration.py`
- `solar_integration/report_generator.py`, `solar_integration/templates.py`, `solar_integration/rewriter.py`, `solar_integration/validator.py`, `solar_integration/rag_system.py`, `solar_integration/solar_features.py`, `solar_integration/embeddings.py`
- `tests/test_integration.py`, `scripts/test_pipeline.py`, `requirements.txt`, `static/index.html`
