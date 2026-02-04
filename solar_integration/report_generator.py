# solar_integration/report_generator.py
"""
통합 Report Generator (Table-Protected 방식)

Flow:
1. Template generation (Vision evidence + Table-first)
2. Pre-LLM validation
3. Table-protected LLM rewrite (Placeholder 방식)
4. Full validation (구조/수치/환각 검증)
5. Safe fallback on failure (Fail-closed)

원칙:
- 임상 사실(소견/측정/비교)은 시스템이 결정
- Solar는 표 밖 텍스트만 문서화 수행
- 조금이라도 위험하면 LLM 결과를 쓰지 않음
"""
from typing import Optional, Dict, List
import uuid
from api.schemas import StructuredAIResult, DraftReport
from solar_integration.templates import TemplateReportBuilder
from solar_integration.validator import ReportValidator, AuditLog
from solar_integration.rewriter import SolarProRewriter
from solar_integration.rag_system import MedicalRAGSystem
from utils.logger import logger


class ProductionReportGenerator:
    """
    제품형 리포트 생성기 (Table-Protected)
    
    특징:
    - Placeholder 방식으로 표 보호
    - 강화된 검증 (구조/수치/환각)
    - Fail-closed 원칙
    - 감사 로그 생성
    """
    
    def __init__(
        self,
        use_mock_solar: bool = False,
        rag_system: Optional[MedicalRAGSystem] = None
    ):
        """
        Args:
            use_mock_solar: Mock Solar Pro 3 사용 (passthrough)
            rag_system: RAG 시스템 (prior comparison용)
        """
        self.template_builder = TemplateReportBuilder()
        self.validator = ReportValidator()
        self.rewriter = SolarProRewriter(use_mock=use_mock_solar)
        self.rag_system = rag_system
        
        self.use_llm_rewrite = not use_mock_solar
        
        logger.info(
            f"Initialized ProductionReportGenerator "
            f"(LLM rewrite: {'enabled' if self.use_llm_rewrite else 'disabled'}, "
            f"mode: table-protected)"
        )

    def _format_key_flags_text(self, key_flags) -> str:
        if not key_flags:
            return "Nodule candidates: 0 (NEW: 0)\nHigh-confidence findings: 0\nScan limitation: No"
        return (
            f"Nodule candidates: {key_flags.nodule_candidates} (NEW: {key_flags.new_nodules})\n"
            f"High-confidence findings: {key_flags.high_confidence_findings}\n"
            f"Scan limitation: {'Yes' if key_flags.scan_limitation else 'No'}"
        )

    def _format_findings_table_text(self, rows) -> str:
        if not rows:
            return "No findings."
        headers = ["Type", "Location", "Status", "Confidence", "Evidence"]
        lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
        for r in rows:
            lines.append(
                " | ".join([
                    r.type,
                    r.location,
                    r.status,
                    f"{r.confidence:.2f}",
                    r.evidence
                ])
            )
        return "\n".join(lines)

    def _format_prior_table_text(self, rows) -> str:
        if not rows:
            return "No prior comparison available."
        headers = ["Lesion ID", "Prior Date", "Change", "Prior Size", "Current Size", "Evidence"]
        lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
        for r in rows:
            lines.append(
                " | ".join([
                    r.lesion_id,
                    r.prior_date,
                    r.change,
                    r.prior_size,
                    r.current_size,
                    r.evidence
                ])
            )
        return "\n".join(lines)

    def _parse_summary_lines(self, summary_text: str) -> List[str]:
        if not summary_text:
            return []
        lines = [line.strip() for line in summary_text.splitlines() if line.strip()]
        cleaned = []
        for line in lines:
            if line.upper().startswith("SUMMARY"):
                continue
            if not line.startswith("-"):
                line = f"- {line}"
            cleaned.append(line)
        return cleaned[:3]

    def _replace_ai_summary_section(self, report_text: str, summary_lines: List[str]) -> str:
        lines = report_text.splitlines()
        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "AI SUMMARY":
                start_idx = i
                continue
            if start_idx is not None and line.strip() == "FINDINGS - TABLE":
                end_idx = i
                break
        if start_idx is None or end_idx is None:
            return report_text
        new_block = ["AI SUMMARY"] + summary_lines + [""]
        lines[start_idx:end_idx] = new_block
        return "\n".join(lines)
    
    async def generate_report(
        self,
        ai_result: StructuredAIResult,
        patient_id: Optional[str] = None,
        include_prior_comparison: bool = False,
        enable_llm_rewrite: bool = True,
        enable_llm_findings: Optional[bool] = None,
        request_id: Optional[str] = None
    ) -> DraftReport:
        """
        Complete report generation pipeline (Table-Protected)
        
        Args:
            ai_result: Vision model의 구조화된 출력
            patient_id: 환자 ID (prior comparison 시 필수)
            include_prior_comparison: Prior와 비교 포함 여부
            enable_llm_rewrite: LLM rewrite 활성화 여부
            request_id: 요청 ID (감사 로그용)
        
        Returns:
            DraftReport (validated)
        """
        # 요청 ID 생성
        request_id = request_id or str(uuid.uuid4())[:8]
        logger.info(f"[{request_id}] Generating report for study {ai_result.study_uid}")
        
        # 감사 로그 초기화
        priors_count = 0
        sources_used = []
        
        # Step 1: Prior comparison (if requested)
        prior_text = None
        prior_data = None
        if include_prior_comparison and patient_id and self.rag_system:
            try:
                comparison = await self.rag_system.compare_with_prior(
                    ai_result,
                    patient_id,
                    ai_result.acquisition_datetime.strftime("%Y%m%d")
                )
                if comparison:
                    prior_text = comparison.get("comparison_text")
                    prior_data = comparison.get("prior_data")  # RAG에서 prior 데이터
                    priors_count = comparison.get("priors_count", 1)
                    sources_used = comparison.get("sources", [])
                    logger.info(f"[{request_id}] Prior comparison included")
            except Exception as e:
                logger.error(f"[{request_id}] Prior comparison failed: {e}")
        
        # Step 2: Template generation (Table-first)
        try:
            template_report = self.template_builder.build_report(
                ai_result,
                patient_id=patient_id,
                prior_text=prior_text,
                prior_data=prior_data
            )
            logger.info(f"[{request_id}] Template report built (table-first)")
        except Exception as e:
            logger.error(f"[{request_id}] Template generation failed: {e}")
            return self._create_fallback_with_audit(
                ai_result, request_id, patient_id,
                f"Template generation error: {e}",
                priors_count, sources_used
            )

        # Step 2.5: AI SUMMARY (Evidence-bound Summarizer)
        if self.use_llm_rewrite and template_report.rendered_report:
            try:
                key_flags_text = self._format_key_flags_text(template_report.key_flags)
                findings_table_text = self._format_findings_table_text(
                    template_report.tables.findings if template_report.tables else []
                )
                prior_table_text = self._format_prior_table_text(
                    template_report.tables.prior_comparison if template_report.tables else []
                )
                summary_response = await self.rewriter.generate_ai_summary(
                    key_flags_text=key_flags_text,
                    findings_table_text=findings_table_text,
                    prior_table_text=prior_table_text
                )
                summary_lines = self._parse_summary_lines(summary_response)
                if summary_lines:
                    template_report.rendered_report = self._replace_ai_summary_section(
                        template_report.rendered_report,
                        summary_lines
                    )
                    logger.info(f"[{request_id}] AI SUMMARY applied ({len(summary_lines)} lines)")
                else:
                    logger.warning(f"[{request_id}] AI SUMMARY empty, using fallback")
            except Exception as e:
                logger.warning(f"[{request_id}] AI SUMMARY generation failed: {e}")
        
        # Step 3: Pre-LLM validation
        pre_validation = self.validator.validate_pre_llm(
            ai_result,
            template_report.findings.content + "\n" + template_report.impression.content
        )
        
        if not pre_validation.is_safe():
            logger.warning(f"[{request_id}] Pre-LLM validation failed: {pre_validation.errors}")
            return self._create_fallback_with_audit(
                ai_result, request_id, patient_id,
                f"Pre-validation failed: {'; '.join(pre_validation.errors)}",
                priors_count, sources_used
            )
        
        # Step 4: Narrative-only LLM rewrite (Solar는 비표 텍스트만 봄)
        final_report = template_report
        rewrite_audit = {"used": False, "fallback": False, "mode": "none"}
        
        if enable_llm_rewrite and self.use_llm_rewrite and final_report.rendered_report:
            try:
                logger.info(f"[{request_id}] Applying narrative-only LLM rewrite...")
                logger.info(f"[{request_id}] Solar will only see: QUALITY, NOTES sections (no tables/measurements)")
                
                # Narrative-only 방식으로 rewrite (기본)
                rewritten_report, rewrite_audit = await self.rewriter.rewrite_narrative_only(
                    template_report.rendered_report
                )
                rewrite_audit["used"] = True
                rewrite_audit["mode"] = "narrative-only"
                
                if rewrite_audit.get("validation_passed", False):
                    # Step 5: Full validation (추가 안전 검증)
                    full_validation = self.validator.validate_full(
                        template_report.rendered_report,
                        rewritten_report,
                        ai_result
                    )
                    
                    if full_validation.is_safe():
                        final_report.rendered_report = rewritten_report
                        logger.info(
                            f"[{request_id}] Narrative rewrite applied - "
                            f"Solar processed {rewrite_audit.get('segments_extracted', 0)} segments"
                        )
                    else:
                        logger.warning(
                            f"[{request_id}] Full validation failed: {full_validation.block_reason}"
                        )
                        final_report.validation_warnings.append(
                            f"LLM rewrite rejected ({full_validation.block_reason})"
                        )
                        rewrite_audit["fallback"] = True
                else:
                    # 검증 실패 - 이미 fallback 사용됨
                    logger.warning(f"[{request_id}] Narrative rewrite validation failed, using template")
                    final_report.validation_warnings.extend(rewrite_audit.get("errors", []))
                    
            except Exception as e:
                logger.error(f"[{request_id}] LLM rewrite failed: {e}, using template")
                final_report.validation_warnings.append(f"LLM rewrite error: {e}")
                rewrite_audit["fallback"] = True
        else:
            logger.info(f"[{request_id}] Using template-only report (no LLM rewrite)")
        
        # Step 6: Final report validation
        final_validation = self.validator.validate_report(final_report)
        final_report.validation_passed = final_validation.passed
        final_report.validation_warnings.extend(final_validation.warnings)
        
        if not final_validation.passed:
            logger.error(f"[{request_id}] Final validation failed: {final_validation.errors}")
            return self._create_fallback_with_audit(
                ai_result, request_id, patient_id,
                f"Final validation failed: {'; '.join(final_validation.errors)}",
                priors_count, sources_used
            )
        
        # Step 7: 감사 로그 첨부
        audit_log = self.validator.create_audit_log(
            request_id=request_id,
            patient_id=patient_id,
            validation_result=None,  # 성공
            priors_count=priors_count,
            sources=sources_used
        )
        audit_log.validator_result = "pass"
        
        # Audit 정보를 report에 추가
        if final_report.audit:
            final_report.audit.request_id = request_id
            final_report.audit.llm_mode = "table-protected-rewrite"
            final_report.audit.llm_fallback_used = rewrite_audit.get("fallback", False)
        
        logger.info(f"[{request_id}] Report generation completed successfully")
        return final_report
    
    def _create_fallback_with_audit(
        self,
        ai_result: StructuredAIResult,
        request_id: str,
        patient_id: Optional[str],
        error_reason: str,
        priors_count: int,
        sources: list
    ) -> DraftReport:
        """감사 로그가 포함된 fallback report 생성"""
        fallback = self.validator.create_safe_fallback_report(ai_result, error_reason)
        
        # 감사 로그 생성 및 첨부
        audit_log = self.validator.create_audit_log(
            request_id=request_id,
            patient_id=patient_id,
            priors_count=priors_count,
            sources=sources
        )
        audit_log.validator_result = "fail"
        audit_log.block_reason = error_reason
        
        logger.warning(f"[{request_id}] Fallback report created: {error_reason}")
        
        return fallback
    
    def get_generator_info(self) -> dict:
        """Generator 정보 반환"""
        return {
            "template_builder": "table-first-v2.0",
            "validator": "enhanced-safety-gate-v2.0",
            "rewriter": self.rewriter.get_rewriter_info(),
            "rag_enabled": self.rag_system is not None,
            "mode": "table-protected",
            "fail_policy": "fail-closed"
        }
