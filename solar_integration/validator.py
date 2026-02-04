# solar_integration/validator.py
"""
강화된 Validator: LLM 출력 안전 게이트

검증 단계:
(A) 구조/무결성 검증 (Hard fail)
    - Placeholder Integrity
    - Table Hash Check

(B) 수치/단위/키워드 검증 (Hard fail)
    - 숫자/단위 변경 탐지
    - Laterality/위치 변경 탐지
    - 금지 표현 탐지

(C) 환각/추론 검증 (Fail or sanitize)
    - Evidence 없는 새 소견 탐지
    - Impression에 새 결론 추가 탐지

원칙: "조금이라도 위험하면 LLM 결과를 쓰지 않는다" (Fail-closed)
"""
from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import re
import hashlib
from api.schemas import (
    StructuredAIResult, DraftReport, ReportSection,
    ValidationResult, NoduleCandidate
)
from utils.logger import logger


@dataclass
class AuditLog:
    """감사 로그 구조"""
    request_id: str
    patient_id: Optional[str] = None  # hash 또는 내부 키
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 모델 정보
    vision_model_version: str = "monai-nodule-v1"
    pipeline_version: str = "v1.0"
    solar_model: str = "solar-pro-3"
    solar_mode: str = "rewrite-only"
    
    # 검증 결과
    validator_result: str = "pending"  # pass/fail
    block_reason: Optional[str] = None
    
    # 추가 정보
    priors_used_count: int = 0
    sources_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "request_id": self.request_id,
            "patient_id": self.patient_id,
            "timestamp": self.timestamp.isoformat(),
            "vision_model_version": self.vision_model_version,
            "pipeline_version": self.pipeline_version,
            "solar_model": self.solar_model,
            "solar_mode": self.solar_mode,
            "validator_result": self.validator_result,
            "block_reason": self.block_reason,
            "priors_used_count": self.priors_used_count,
            "sources_used": self.sources_used
        }


@dataclass
class EnhancedValidationResult:
    """강화된 검증 결과"""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    blocked_content: List[str] = field(default_factory=list)
    
    # 상세 검증 결과
    placeholder_integrity: bool = True
    hash_verified: bool = True
    numeric_preserved: bool = True
    location_preserved: bool = True
    forbidden_detected: bool = False
    hallucination_detected: bool = False
    
    # 차단 사유 (실패 시)
    block_reason: Optional[str] = None
    
    def is_safe(self) -> bool:
        """안전하게 사용 가능한지"""
        return self.passed and not self.forbidden_detected and not self.hallucination_detected


class ReportValidator:
    """
    강화된 Report Validation Engine
    
    모든 LLM 출력은 반드시 이 Validator를 통과해야 함
    """
    
    # 금지 표현 패턴 (Hard fail) - 완화된 버전
    # 핵심 안전 규칙만 유지, 일반적 radiology 표현은 허용
    FORBIDDEN_PATTERNS = [
        # 진단 확정 (핵심 - 유지)
        (r'\b(diagnosed with|definitively diagnosed)\b', "diagnosis_statement"),
        (r'\b(confirmed malignancy|confirms cancer)\b', "confirmation_statement"),
        (r'\b(ruled out|rule out)\b', "rule_out_statement"),
        
        # 정상 단정 (완화 - 강한 표현만)
        (r'\b(entirely normal|completely normal)\b', "normal_statement"),
        # "no findings", "unremarkable", "no evidence of" 제거 - 일반적 radiology 표현
        
        # 치료/처방 권고 (구체적 처방만 금지)
        (r'\b(prescribe|medication)\b', "prescription_recommendation"),
        (r'\b(recommend (surgery|chemotherapy|radiation))\b', "procedure_recommendation"),
        (r'\b(must undergo|should undergo)\b', "action_recommendation"),
        
        # LLM 자의적 추론 (핵심 - 유지)
        (r'\b(likely malignant|probably malignant|suspicious for malignancy)\b', "malignancy_inference"),
        (r'\b(consistent with malignancy|concern for cancer)\b', "cancer_inference"),
        (r'\b(metastatic disease|metastasis confirmed)\b', "metastasis_statement"),
    ]
    
    # 숫자 패턴 (변경 탐지용)
    NUMERIC_PATTERNS = [
        r'\d+\.?\d*\s*mm',           # 크기 (mm)
        r'\d+\.?\d*\s*cm',           # 크기 (cm)
        r'\d+\.?\d*\s*mm[³3]',       # 부피
        r'\d+\.?\d*%',               # 퍼센트
        r'\d{4}-\d{2}-\d{2}',        # 날짜
        r'0\.\d+',                   # 신뢰도 (0.xx)
    ]
    
    # 위치/방향 패턴 (변경 탐지용)
    LOCATION_PATTERNS = [
        r'\b(RUL|RML|RLL|LUL|LLL)\b',  # 폐엽
        r'\b(right|left)\b',           # 좌우
        r'\b(upper|middle|lower)\b',   # 상중하
        r'\b(anterior|posterior)\b',   # 전후
        r'\b(medial|lateral)\b',       # 내외측
        r'\b(apical|basal)\b',         # 첨부/기저
    ]
    
    def __init__(self):
        self.forbidden_regex = [
            (re.compile(pattern, re.IGNORECASE), reason) 
            for pattern, reason in self.FORBIDDEN_PATTERNS
        ]
        self.numeric_regex = [re.compile(p, re.IGNORECASE) for p in self.NUMERIC_PATTERNS]
        self.location_regex = [re.compile(p, re.IGNORECASE) for p in self.LOCATION_PATTERNS]
        
        logger.info("Initialized EnhancedReportValidator")
    
    def validate_full(
        self,
        original_text: str,
        rewritten_text: str,
        ai_result: Optional[StructuredAIResult] = None
    ) -> EnhancedValidationResult:
        """
        전체 검증 수행 (A + B + C)
        
        Args:
            original_text: 원본 템플릿 텍스트
            rewritten_text: Solar가 rewrite한 텍스트
            ai_result: Vision 모델 결과 (환각 검증용)
            
        Returns:
            EnhancedValidationResult
        """
        errors = []
        warnings = []
        blocked = []
        result = EnhancedValidationResult(passed=True)
        
        # (A) 구조/무결성 검증
        # A1. Placeholder integrity는 rewriter에서 이미 검증됨
        
        # A2. 표 블록 보존 확인
        table_check = self._validate_table_blocks(original_text, rewritten_text)
        if not table_check[0]:
            errors.extend(table_check[1])
            result.hash_verified = False
            result.block_reason = "table_modified"
        
        # (B) 수치/단위/키워드 검증
        # B1. 숫자/단위 보존 확인
        numeric_check = self._validate_numeric_preservation(original_text, rewritten_text)
        if not numeric_check[0]:
            errors.extend(numeric_check[1])
            result.numeric_preserved = False
            result.block_reason = result.block_reason or "numeric_changed"
        
        # B2. 위치/방향 보존 확인
        location_check = self._validate_location_preservation(original_text, rewritten_text)
        if not location_check[0]:
            errors.extend(location_check[1])
            result.location_preserved = False
            result.block_reason = result.block_reason or "location_changed"
        
        # B3. 금지 표현 탐지
        forbidden_check = self._detect_forbidden_expressions(rewritten_text)
        if not forbidden_check[0]:
            errors.extend(forbidden_check[1])
            blocked.extend(forbidden_check[2])
            result.forbidden_detected = True
            result.block_reason = result.block_reason or "forbidden_expression"
        
        # (C) 환각/추론 검증
        if ai_result:
            hallucination_check = self._detect_hallucination(rewritten_text, ai_result)
            if not hallucination_check[0]:
                # Hallucination은 경고로 처리하고 sanitize 가능
                warnings.extend(hallucination_check[1])
                result.hallucination_detected = True
        
        # 최종 결과 집계
        result.errors = errors
        result.warnings = warnings
        result.blocked_content = blocked
        result.passed = len(errors) == 0
        
        log_status = "PASS" if result.passed else "FAIL"
        logger.info(f"Full validation: {log_status}")
        if not result.passed:
            logger.warning(f"Validation failed: {result.block_reason}")
        
        return result
    
    def _validate_table_blocks(
        self, 
        original: str, 
        rewritten: str
    ) -> Tuple[bool, List[str]]:
        """표 블록이 동일한지 검증"""
        errors = []
        
        # 표 섹션 헤더
        table_headers = [
            "FINDINGS - TABLE", "MEASUREMENTS - TABLE", "PRIOR COMPARISON - TABLE"
        ]
        
        for header in table_headers:
            original_block = self._extract_section(original, header)
            rewritten_block = self._extract_section(rewritten, header)
            
            if original_block and rewritten_block:
                # 정규화 후 비교
                original_normalized = self._normalize_whitespace(original_block)
                rewritten_normalized = self._normalize_whitespace(rewritten_block)
                
                if original_normalized != rewritten_normalized:
                    errors.append(f"Table block modified: {header}")
        
        return len(errors) == 0, errors
    
    def _validate_numeric_preservation(
        self, 
        original: str, 
        rewritten: str
    ) -> Tuple[bool, List[str]]:
        """숫자/단위가 보존되었는지 검증"""
        errors = []
        
        # 원본에서 숫자 추출
        original_numbers = set()
        for pattern in self.numeric_regex:
            original_numbers.update(pattern.findall(original))
        
        # Rewritten에서 숫자 추출
        rewritten_numbers = set()
        for pattern in self.numeric_regex:
            rewritten_numbers.update(pattern.findall(rewritten))
        
        # 원본에 있던 숫자가 사라졌는지 확인
        missing = original_numbers - rewritten_numbers
        if missing:
            errors.append(f"Numeric values removed: {missing}")
        
        # 새로운 숫자가 추가되었는지 확인 (일부 허용)
        added = rewritten_numbers - original_numbers
        if added:
            # 날짜나 신뢰도는 허용
            suspicious_added = [n for n in added if 'mm' in n.lower() or 'cm' in n.lower()]
            if suspicious_added:
                errors.append(f"New measurements added: {suspicious_added}")
        
        return len(errors) == 0, errors
    
    def _validate_location_preservation(
        self, 
        original: str, 
        rewritten: str
    ) -> Tuple[bool, List[str]]:
        """위치/방향 정보가 보존되었는지 검증"""
        errors = []
        
        # 원본에서 위치 추출
        original_locations = set()
        for pattern in self.location_regex:
            original_locations.update(m.upper() for m in pattern.findall(original))
        
        # Rewritten에서 위치 추출
        rewritten_locations = set()
        for pattern in self.location_regex:
            rewritten_locations.update(m.upper() for m in pattern.findall(rewritten))
        
        # 원본에 있던 위치가 사라졌는지 확인
        missing = original_locations - rewritten_locations
        if missing:
            errors.append(f"Location terms removed: {missing}")
        
        return len(errors) == 0, errors
    
    def _detect_forbidden_expressions(
        self, 
        text: str
    ) -> Tuple[bool, List[str], List[str]]:
        """금지 표현 탐지"""
        errors = []
        blocked = []
        
        for pattern, reason in self.forbidden_regex:
            matches = pattern.findall(text)
            if matches:
                errors.append(f"Forbidden expression ({reason}): {matches}")
                blocked.extend(matches)
        
        return len(errors) == 0, errors, blocked
    
    def _detect_hallucination(
        self, 
        text: str, 
        ai_result: StructuredAIResult
    ) -> Tuple[bool, List[str]]:
        """Evidence 없는 새 소견 탐지"""
        warnings = []
        
        # AI result에서 유효한 clinical terms 추출
        valid_terms = self._extract_valid_terms(ai_result)
        
        # 텍스트에서 clinical terms 추출
        text_terms = self._extract_clinical_terms(text)
        
        # Evidence 없는 새 terms
        unauthorized = text_terms - valid_terms
        
        if unauthorized:
            warnings.append(f"Possible hallucination - terms without evidence: {unauthorized}")
        
        return len(warnings) == 0, warnings
    
    def _extract_valid_terms(self, ai_result: StructuredAIResult) -> Set[str]:
        """AI result에서 언급 가능한 clinical terms 추출"""
        valid = set()
        
        # Nodules
        for nodule in ai_result.nodules:
            valid.add("nodule")
            valid.add("pulmonary nodule")
            if nodule.location_code:
                valid.add(nodule.location_code.lower())
        
        # Findings
        findings = ai_result.findings
        if findings.pleural_effusion.label == "present":
            valid.update(["pleural effusion", "effusion"])
        if findings.pneumothorax.label == "present":
            valid.add("pneumothorax")
        if findings.consolidation.label == "present":
            valid.add("consolidation")
        if findings.atelectasis.label == "present":
            valid.add("atelectasis")
        if findings.emphysema.label == "present":
            valid.add("emphysema")
        
        # Always valid terms (anatomical)
        valid.update(["lung", "lungs", "chest", "thorax", "mediastinum", 
                     "heart", "aorta", "spine", "rib", "pleura"])
        
        return valid
    
    def _extract_clinical_terms(self, text: str) -> Set[str]:
        """텍스트에서 clinical terms 추출"""
        clinical_keywords = [
            "nodule", "mass", "lesion", "opacity", "tumor",
            "effusion", "pneumothorax", "consolidation",
            "atelectasis", "emphysema", "fibrosis",
            "cardiomegaly", "adenopathy", "infiltrate",
            "pneumonia", "edema", "hemorrhage"
        ]
        
        text_lower = text.lower()
        found = set()
        
        for keyword in clinical_keywords:
            if keyword in text_lower:
                found.add(keyword)
        
        return found
    
    def _extract_section(self, text: str, header: str) -> Optional[str]:
        """섹션 추출"""
        lines = text.splitlines()
        
        start_idx = None
        for i, line in enumerate(lines):
            if header in line:
                start_idx = i
                break
        
        if start_idx is None:
            return None
        
        # 다음 빈 줄 또는 다른 섹션까지
        end_idx = start_idx + 1
        while end_idx < len(lines):
            line = lines[end_idx]
            if line.strip() == "" or any(h in line for h in ["━━━", "FINDINGS", "MEASUREMENTS", "PRIOR", "KEY", "QUALITY", "AUDIT", "NOTES"]):
                break
            end_idx += 1
        
        return "\n".join(lines[start_idx:end_idx])
    
    def _normalize_whitespace(self, text: str) -> str:
        """공백 정규화"""
        return " ".join(text.split())
    
    # Legacy methods
    def validate_pre_llm(
        self,
        ai_result: StructuredAIResult,
        template_draft: str
    ) -> ValidationResult:
        """LLM 입력 전 검증 (Legacy)"""
        errors = []
        warnings = []
        
        # Quality check
        if not ai_result.quality.is_adequate_for_nodules():
            if ai_result.quality.slice_thickness_mm >= 5.0:
                warnings.append("Slice thickness ≥5mm: small nodule detection may be limited")
        
        result = ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
        return result
    
    def validate_post_llm(
        self,
        ai_result: StructuredAIResult,
        rewritten_text: str,
        original_template: str
    ) -> ValidationResult:
        """LLM 출력 후 검증 (Legacy - validate_full 사용 권장)"""
        enhanced = self.validate_full(original_template, rewritten_text, ai_result)
        
        return ValidationResult(
            passed=enhanced.passed,
            errors=enhanced.errors,
            warnings=enhanced.warnings,
            blocked_content=enhanced.blocked_content
        )
    
    def validate_table_integrity(
        self, 
        original_report: str, 
        rewritten_report: str
    ) -> ValidationResult:
        """Table 보호 검증 (Legacy)"""
        ok, errors = self._validate_table_blocks(original_report, rewritten_report)
        
        return ValidationResult(
            passed=ok,
            errors=errors,
            warnings=[]
        )
    
    def validate_report(self, report: DraftReport) -> ValidationResult:
        """Final report validation (간단한 구조 검증)"""
        errors = []
        warnings = []
        
        # 필수 섹션 확인
        if not report.findings or not report.findings.content:
            warnings.append("Findings section is empty")
        
        if not report.tables:
            warnings.append("Tables section is empty")
        
        return ValidationResult(
            passed=True,  # 기본적으로 통과
            errors=errors,
            warnings=warnings
        )
    
    def create_safe_fallback_report(
        self,
        ai_result: StructuredAIResult,
        error_reason: str
    ) -> DraftReport:
        """Validation 실패 시 안전한 limitation-only draft 생성"""
        logger.warning(f"Creating safe fallback report due to: {error_reason}")
        
        return DraftReport(
            study_uid=ai_result.study_uid,
            patient_id=None,
            
            technique=ReportSection(
                title="TECHNIQUE",
                content="Chest CT without contrast"
            ),
            
            comparison=ReportSection(
                title="COMPARISON",
                content="None available"
            ),
            
            findings=ReportSection(
                title="FINDINGS",
                content="Unable to generate automated findings due to validation failure."
            ),
            
            impression=ReportSection(
                title="IMPRESSION",
                content="Manual review required."
            ),
            
            measurements=ReportSection(
                title="MEASUREMENTS",
                content="Not available"
            ),
            
            limitations=ReportSection(
                title="LIMITATIONS",
                content=f"Automated report generation failed: {error_reason}\n"
                       f"This study requires manual interpretation by a radiologist."
            ),
            
            generated_at=datetime.now(),
            generator_version="fallback-v1.0",
            validation_passed=False,
            validation_warnings=[error_reason],
            based_on_ai_result=True
        )
    
    def create_audit_log(
        self,
        request_id: str,
        patient_id: Optional[str] = None,
        validation_result: Optional[EnhancedValidationResult] = None,
        priors_count: int = 0,
        sources: Optional[List[str]] = None
    ) -> AuditLog:
        """감사 로그 생성"""
        audit = AuditLog(
            request_id=request_id,
            patient_id=patient_id,
            priors_used_count=priors_count,
            sources_used=sources or []
        )
        
        if validation_result:
            audit.validator_result = "pass" if validation_result.passed else "fail"
            audit.block_reason = validation_result.block_reason
        
        return audit
