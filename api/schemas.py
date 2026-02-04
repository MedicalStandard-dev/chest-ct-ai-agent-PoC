# api/schemas.py
"""
Pydantic schemas for PACS AI Agent
Evidence-first 구조: 모든 임상 문장은 Vision evidence에 연결
"""
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime


# ============================================================================
# Evidence Structures (Vision 출력)
# ============================================================================

class VisionEvidence(BaseModel):
    """Vision 모델 출력의 근거 정보"""
    series_uid: str
    instance_uids: List[str]
    slice_range: tuple[int, int]
    mask_path: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class QualityMetrics(BaseModel):
    """영상 품질 메트릭"""
    slice_thickness_mm: float
    coverage_score: float = Field(ge=0.0, le=1.0)
    artifact_score: float = Field(ge=0.0, le=1.0)  # 0=clean, 1=severe
    
    def is_adequate_for_nodules(self) -> bool:
        """Small nodule detection 가능 여부"""
        return self.slice_thickness_mm < 5.0 and self.coverage_score >= 0.85


class NoduleCandidate(BaseModel):
    """폐결절 후보 (Vision 출력)"""
    id: str
    center_zyx: tuple[float, float, float]
    bbox_zyx: tuple[int, int, int, int, int, int]
    diameter_mm: float
    volume_mm3: float
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: VisionEvidence
    
    # 추가 특성 (optional)
    location_code: Optional[str] = None  # RUL, RML, RLL, LUL, LLL
    characteristics: Optional[Dict[str, Any]] = None


class FindingLabel(BaseModel):
    """멀티라벨 finding 출력"""
    label: Literal["present", "absent", "uncertain"]
    probability: float = Field(ge=0.0, le=1.0)
    evidence: Optional[VisionEvidence] = None


class StructuredFindings(BaseModel):
    """구조화된 findings (멀티라벨)"""
    pleural_effusion: FindingLabel
    pneumothorax: FindingLabel
    consolidation: FindingLabel
    atelectasis: FindingLabel
    emphysema: FindingLabel


class ModelVersioning(BaseModel):
    """모델 버전 및 threshold 정보"""
    model_version: str
    pipeline_version: str
    thresholds: Dict[str, float]
    calibration_date: Optional[str] = None


class StructuredAIResult(BaseModel):
    """
    Vision 모델의 완전한 구조화 출력
    - 임상 문장 생성 전 source-of-truth
    - LLM 입력 금지 (template만 참조)
    """
    study_uid: str
    series_uid: str
    acquisition_datetime: datetime
    
    # Quality
    quality: QualityMetrics
    
    # Lung segmentation
    lung_volume_ml: float
    lung_mask_path: Optional[str] = None
    
    # Nodules
    nodules: List[NoduleCandidate]
    low_confidence_nodules: List[NoduleCandidate] = []  # limitations only
    
    # Multi-label findings
    findings: StructuredFindings
    
    # Versioning
    versioning: ModelVersioning
    
    # Processing metadata
    processing_time_seconds: float
    warnings: List[str] = []


# ============================================================================
# Report Structures
# ============================================================================

class ReportSection(BaseModel):
    """리포트 섹션 (template 기반)"""
    title: str
    content: str
    evidence_ids: List[str] = []  # 근거 nodule/finding IDs
    confidence_level: Optional[str] = None


class FindingsTableRow(BaseModel):
    """FINDINGS 테이블 행"""
    type: str
    location: str
    status: str
    confidence: float
    evidence: str


class MeasurementsTableRow(BaseModel):
    """MEASUREMENTS 테이블 행"""
    lesion_id: str
    location: str
    diameter_mm: float
    volume_mm3: float
    confidence: float
    evidence: str


class PriorComparisonRow(BaseModel):
    """PRIOR COMPARISON 테이블 행"""
    lesion_id: str
    prior_date: str
    change: str
    prior_size: str
    current_size: str
    evidence: str


class ReportTables(BaseModel):
    """Report tables"""
    findings: List[FindingsTableRow] = []
    measurements: List[MeasurementsTableRow] = []
    prior_comparison: List[PriorComparisonRow] = []


class KeyFlags(BaseModel):
    """의사 요약용 KEY FLAGS"""
    nodule_candidates: int = 0
    new_nodules: int = 0
    high_confidence_findings: int = 0
    scan_limitation: bool = False


class AuditInfo(BaseModel):
    """Audit metadata"""
    model_version: str
    pipeline_version: str
    solar_prompt_version: str
    priors_used: int = 0
    request_id: Optional[str] = None
    sources_used: List[str] = []
    validator_result: Optional[str] = None
    processing_time_ms: Optional[float] = None
    llm_mode: Optional[str] = None
    llm_fallback_used: bool = False


class DraftReport(BaseModel):
    """생성된 판독문 초안"""
    study_uid: str
    patient_id: Optional[str] = None
    
    # Report sections
    technique: ReportSection
    comparison: ReportSection
    findings: ReportSection
    impression: ReportSection
    measurements: ReportSection
    limitations: ReportSection

    # Table-first payload
    tables: Optional[ReportTables] = None
    key_flags: Optional[KeyFlags] = None
    limitations_text: List[str] = []
    notes: List[str] = []
    audit: Optional[AuditInfo] = None
    rendered_report: Optional[str] = None
    
    # Metadata
    generated_at: datetime
    generator_version: str
    validation_passed: bool
    validation_warnings: List[str] = []
    
    # Source
    based_on_ai_result: bool = True
    prior_comparison_included: bool = False


# ============================================================================
# API Request/Response
# ============================================================================

class AnalyzeRequest(BaseModel):
    """CT 분석 요청"""
    patient_id: Optional[str] = None
    study_uid: str
    series_uid: Optional[str] = None
    
    # Options
    include_report: bool = False
    include_prior_comparison: bool = False
    include_vector_db_save: bool = False
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    """CT 분석 응답"""
    request_id: str
    status: Literal["completed", "failed", "partial"]
    
    # Core output
    structured_ai_result: StructuredAIResult
    
    # Optional outputs
    draft_report: Optional[DraftReport] = None
    prior_comparison: Optional[Dict[str, Any]] = None
    
    # Korean translation (Solar Pro 3)
    korean_report: Optional[str] = None
    
    # Groundedness check result (Solar Pro 3)
    groundedness: Optional[Dict[str, Any]] = None
    
    # Solar contributions
    solar_features_used: List[str] = []
    
    # Metadata
    processing_summary: Dict[str, Any]


class GenerateReportRequest(BaseModel):
    """리포트 생성 요청 (AI 결과로부터)"""
    structured_ai_result: StructuredAIResult
    patient_id: Optional[str] = None
    include_prior_comparison: bool = False
    template_style: str = "standard"


class GenerateReportResponse(BaseModel):
    """리포트 생성 응답"""
    draft_report: DraftReport
    validation_status: Dict[str, Any]


class VectorDbSaveRequest(BaseModel):
    """Vector DB 저장 요청"""
    structured_ai_result: StructuredAIResult
    draft_report: Optional[DraftReport] = None
    patient_id: Optional[str] = None


class VectorDbSaveResponse(BaseModel):
    """Vector DB 저장 응답"""
    saved: bool
    vector_db_doc_id: Optional[str] = None
    updated: bool = False
    message: str = ""


class PatientHistoryResponse(BaseModel):
    """환자 이력 조회 응답"""
    patient_id: str
    num_studies: int
    studies: List[Dict[str, Any]]


class SemanticSearchRequest(BaseModel):
    """의미 검색 요청"""
    query_text: str
    patient_id: str  # 필수: patient isolation
    n_results: int = Field(default=5, ge=1, le=20)
    modality_filter: Optional[str] = None


class SemanticSearchResponse(BaseModel):
    """의미 검색 응답"""
    query: str
    patient_id: str
    results: List[Dict[str, Any]]
    retriever_info: Dict[str, str]  # embedding model, etc.


# ============================================================================
# Validation Results
# ============================================================================

class ValidationResult(BaseModel):
    """Validator 검증 결과"""
    passed: bool
    errors: List[str] = []
    warnings: List[str] = []
    blocked_content: List[str] = []
    
    def is_safe(self) -> bool:
        """안전 게이트 통과 여부"""
        return self.passed and len(self.errors) == 0
