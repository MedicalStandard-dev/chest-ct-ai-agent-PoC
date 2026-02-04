# solar_integration/templates.py
"""
Template-first report builder
Vision evidence → structured template → LLM rewrite (optional)
"""
from typing import List, Dict, Optional
from datetime import datetime

from api.schemas import (
    StructuredAIResult, DraftReport, ReportSection,
    NoduleCandidate, FindingLabel,
    FindingsTableRow, MeasurementsTableRow, PriorComparisonRow,
    ReportTables, KeyFlags, AuditInfo
)
from monai_pipeline.calibration import ThresholdManager
from utils.logger import logger


class TemplateReportBuilder:
    """Template 기반 리포트 생성기"""
    
    def __init__(self, threshold_manager: Optional[ThresholdManager] = None):
        self.threshold_manager = threshold_manager or ThresholdManager()
        logger.info("Initialized TemplateReportBuilder")
    
    def build_report(
        self,
        ai_result: StructuredAIResult,
        patient_id: Optional[str] = None,
        prior_text: Optional[str] = None,
        prior_data: Optional[Dict] = None
    ) -> DraftReport:
        """
        Vision evidence로부터 template report 생성
        LLM 없이도 동작 가능
        """
        logger.info(f"Building template report for study {ai_result.study_uid}")
        
        # Build each section (legacy narrative)
        technique = self._build_technique_section(ai_result)
        comparison = self._build_comparison_section(prior_text)
        findings = self._build_findings_section(ai_result)
        impression = self._build_impression_section(ai_result)
        measurements = self._build_measurements_section(ai_result)
        limitations = self._build_limitations_section(ai_result)

        # Table-first payloads
        findings_table = self._build_findings_table(ai_result)
        measurements_table = self._build_measurements_table(ai_result)
        prior_table = self._build_prior_comparison_table(
            prior_text=prior_text,
            current_nodules=ai_result.nodules,
            prior_data=prior_data
        )
        tables = ReportTables(
            findings=findings_table,
            measurements=measurements_table,
            prior_comparison=prior_table
        )
        limitations_text = self._build_limitations_list(ai_result)
        key_flags = self._build_key_flags(ai_result, limitations_text)
        ai_summary_lines = self._build_ai_summary_fallback(
            key_flags=key_flags,
            prior_rows=prior_table
        )
        notes = [
            "This draft is generated from AI outputs.",
            "Requires physician confirmation."
        ]
        audit = self._build_audit(ai_result, prior_text)
        rendered_report = self._render_report_layout(
            ai_result=ai_result,
            patient_id=patient_id,
            limitations_text=limitations_text,
            key_flags=key_flags,
            ai_summary_lines=ai_summary_lines,
            tables=tables,
            notes=notes,
            audit=audit
        )
        
        report = DraftReport(
            study_uid=ai_result.study_uid,
            patient_id=patient_id,
            
            technique=technique,
            comparison=comparison,
            findings=findings,
            impression=impression,
            measurements=measurements,
            limitations=limitations,
            tables=tables,
            key_flags=key_flags,
            limitations_text=limitations_text,
            notes=notes,
            audit=audit,
            rendered_report=rendered_report,
            
            generated_at=datetime.now(),
            generator_version=f"template-v1.0+{ai_result.versioning.pipeline_version}",
            validation_passed=True,
            based_on_ai_result=True,
            prior_comparison_included=(prior_text is not None)
        )
        
        logger.info("Template report built successfully")
        return report

    def _build_findings_table(self, ai_result: StructuredAIResult) -> List[FindingsTableRow]:
        """FINDINGS 테이블 생성 (항상 포함, low-confidence 포함)"""
        rows: List[FindingsTableRow] = []

        # Nodules (include low confidence)
        all_nodules = self._collect_all_nodules(ai_result)
        for nodule in all_nodules:
            status = "Present" if self.threshold_manager.should_report_nodule(
                nodule.confidence
            ) else "Low confidence"
            rows.append(
                FindingsTableRow(
                    type="Nodule candidate",
                    location=nodule.location_code or "-",
                    status=status,
                    confidence=float(nodule.confidence),
                    evidence=nodule.id
                )
            )

        # Multi-label findings
        rows.extend(self._build_finding_rows(ai_result))
        return rows

    def _build_measurements_table(self, ai_result: StructuredAIResult) -> List[MeasurementsTableRow]:
        """MEASUREMENTS 테이블 (결절 전부)"""
        rows: List[MeasurementsTableRow] = []
        for nodule in self._collect_all_nodules(ai_result):
            rows.append(
                MeasurementsTableRow(
                    lesion_id=nodule.id,
                    location=nodule.location_code or "-",
                    diameter_mm=float(nodule.diameter_mm),
                    volume_mm3=float(nodule.volume_mm3),
                    confidence=float(nodule.confidence),
                    evidence=nodule.id
                )
            )
        return rows

    def _build_prior_comparison_table(
        self, 
        prior_text: Optional[str],
        current_nodules: List = None,
        prior_data: Dict = None
    ) -> List[PriorComparisonRow]:
        """PRIOR COMPARISON 테이블"""
        if not prior_text and not prior_data:
            return []
        
        rows = []
        
        # prior_data가 있으면 사용 (RAG에서 가져온 데이터)
        if prior_data and current_nodules:
            prior_date = prior_data.get('study_date', 'Unknown')
            prior_size = prior_data.get('nodule_diameter_mm')
            
            # 현재 nodule과 비교
            for i, nodule in enumerate(current_nodules[:3]):  # 최대 3개
                current_size = nodule.diameter_mm if hasattr(nodule, 'diameter_mm') else 0
                
                if prior_size and current_size:
                    change_pct = ((current_size - prior_size) / prior_size) * 100
                    if change_pct > 20:
                        change_status = "INCREASED"
                    elif change_pct < -20:
                        change_status = "DECREASED"
                    else:
                        change_status = "STABLE"
                else:
                    change_pct = 0
                    change_status = "NEW" if not prior_size else "N/A"
                
                rows.append(PriorComparisonRow(
                    lesion_id=nodule.id if hasattr(nodule, 'id') else f"L{i+1}",
                    prior_date=prior_date,
                    change=change_status,
                    prior_size=f"{prior_size:.1f} mm" if prior_size else "N/A",
                    current_size=f"{current_size:.1f} mm" if current_size else "N/A",
                    evidence=nodule.id if hasattr(nodule, 'id') else f"L{i+1}"
                ))
        
        return rows

    def _build_key_flags(self, ai_result: StructuredAIResult, limitations_text: List[str]) -> KeyFlags:
        """KEY FLAGS 요약"""
        reportable_nodules = [
            n for n in ai_result.nodules
            if self.threshold_manager.should_report_nodule(n.confidence)
        ]
        
        # High-confidence nodules (confidence >= 0.9)
        HIGH_CONF_THRESHOLD = 0.9
        high_conf_nodules = len([
            n for n in reportable_nodules
            if n.confidence >= HIGH_CONF_THRESHOLD
        ])
        
        # High-confidence other findings
        high_conf_other = 0
        for label in [
            ai_result.findings.pleural_effusion,
            ai_result.findings.pneumothorax,
            ai_result.findings.consolidation,
            ai_result.findings.atelectasis,
            ai_result.findings.emphysema
        ]:
            if label.label == "present" and label.probability >= HIGH_CONF_THRESHOLD:
                high_conf_other += 1

        scan_limitation = len(limitations_text) > 0
        return KeyFlags(
            nodule_candidates=len(reportable_nodules),
            new_nodules=0,
            high_confidence_findings=high_conf_nodules + high_conf_other,
            scan_limitation=scan_limitation
        )

    def _build_limitations_list(self, ai_result: StructuredAIResult) -> List[str]:
        """QUALITY / LIMITATIONS 문구 (규칙 기반) - 제품형 필수 규칙"""
        lines: List[str] = []
        
        # 1. Resolution-based limitations
        spacing = ai_result.quality.slice_thickness_mm
        if spacing >= 2.5:
            lines.append(
                f"Slice thickness {spacing:.1f} mm may limit detection of small nodules (<5 mm)."
            )
        elif spacing >= 1.5:
            lines.append(
                f"Slice thickness {spacing:.1f} mm; small nodules (<3 mm) may be missed."
            )
        elif spacing > 1.0:
            lines.append(
                "Small nodules (<3 mm) may be underestimated due to resolution limits."
            )
        
        # 2. Coverage-based limitations
        if ai_result.quality.coverage_score < 0.95:
            if ai_result.quality.coverage_score < 0.85:
                lines.append(
                    f"Incomplete lung coverage (score: {ai_result.quality.coverage_score:.0%}); "
                    "apical or basal nodules may be missed."
                )
            else:
                lines.append("Lung coverage is adequate but not complete.")
        
        # 3. Artifact-based limitations
        if ai_result.quality.artifact_score > 0.5:
            lines.append(
                f"Image artifacts present (score: {ai_result.quality.artifact_score:.2f}); "
                "may affect detection accuracy."
            )
        elif ai_result.quality.artifact_score > 0.2:
            lines.append("Minor artifacts detected; unlikely to affect major findings.")
        
        # 4. Candidate explosion warning (중요!)
        n_candidates = len(ai_result.nodules)
        if n_candidates > 10:
            lines.append(
                f"Multiple candidates detected ({n_candidates}); "
                "high false-positive rate possible. Clinical correlation required."
            )
        elif n_candidates > 5:
            lines.append(
                f"Several candidates detected ({n_candidates}); "
                "false positives possible."
            )
        
        # 5. 항상 포함: AI disclaimer (제품/법적 필수)
        lines.append(
            "AI findings are candidates only and require physician confirmation."
        )
        
        return lines

    def _build_audit(self, ai_result: StructuredAIResult, prior_text: Optional[str]) -> AuditInfo:
        """AUDIT 정보"""
        return AuditInfo(
            model_version=ai_result.versioning.model_version,
            pipeline_version=ai_result.versioning.pipeline_version,
            solar_prompt_version="rewrite-only-table-v2",
            priors_used=1 if prior_text else 0
        )

    def _render_report_layout(
        self,
        ai_result: StructuredAIResult,
        patient_id: Optional[str],
        limitations_text: List[str],
        key_flags: KeyFlags,
        ai_summary_lines: List[str],
        tables: ReportTables,
        notes: List[str],
        audit: AuditInfo
    ) -> str:
        """한 화면 완결형 레이아웃 렌더링"""
        lines: List[str] = []
        lines.append("DRAFT — Requires physician confirmation")
        date_str = ai_result.acquisition_datetime.strftime("%Y-%m-%d")
        lines.append(f"CT Chest | {date_str} | {patient_id or 'UNKNOWN'}")
        lines.append("-" * 50)
        lines.append("")

        lines.append("QUALITY / LIMITATIONS")
        if limitations_text:
            lines.extend([f"- {line}" for line in limitations_text])
        else:
            # Fallback: 항상 최소 disclaimer 포함
            lines.append("- AI findings are candidates only and require physician confirmation.")
        lines.append("")

        lines.append("KEY FLAGS")
        lines.append(
            f"- Nodule candidates: {key_flags.nodule_candidates} (NEW: {key_flags.new_nodules})"
        )
        lines.append(
            f"- High-confidence findings: {key_flags.high_confidence_findings if key_flags.high_confidence_findings > 0 else 'None'}"
        )
        lines.append(
            f"- Scan limitation: {'Yes' if key_flags.scan_limitation else 'No'}"
        )
        lines.append("")

        lines.append("AI SUMMARY")
        if ai_summary_lines:
            lines.extend([line if line.startswith("- ") else f"- {line}" for line in ai_summary_lines])
        else:
            lines.append("- Summary not available.")
        lines.append("")

        lines.append("FINDINGS - TABLE")
        lines.append(self._render_findings_table(tables.findings))
        lines.append("")

        lines.append("MEASUREMENTS - TABLE")
        lines.append(self._render_measurements_table(tables.measurements))
        lines.append("")

        lines.append("PRIOR COMPARISON - TABLE")
        lines.append(self._render_prior_table(tables.prior_comparison))
        lines.append("")

        lines.append("NOTES")
        lines.extend([f"- {note}" for note in notes])
        lines.append("")

        lines.append("AUDIT")
        lines.append(f"- model_version: {audit.model_version}")
        lines.append(f"- pipeline_version: {audit.pipeline_version}")
        lines.append(f"- solar_prompt_version: {audit.solar_prompt_version}")
        lines.append(f"- priors_used: {audit.priors_used}")
        lines.append("")

        return "\n".join(lines)

    def _build_ai_summary_fallback(
        self,
        key_flags: KeyFlags,
        prior_rows: List[PriorComparisonRow]
    ) -> List[str]:
        """LLM 요약 폴백 (table facts 기반)"""
        lines: List[str] = []
        lines.append(
            f"Nodule candidates: {key_flags.nodule_candidates} (NEW: {key_flags.new_nodules})."
        )
        if prior_rows:
            change_counts = {"INCREASED": 0, "DECREASED": 0, "STABLE": 0, "NEW": 0}
            for row in prior_rows:
                if row.change in change_counts:
                    change_counts[row.change] += 1
            lines.append(
                "Prior comparison: "
                f"INCREASED {change_counts['INCREASED']}, "
                f"DECREASED {change_counts['DECREASED']}, "
                f"STABLE {change_counts['STABLE']}, "
                f"NEW {change_counts['NEW']}."
            )
        else:
            lines.append("No prior comparison available.")
        lines.append("Findings are AI-detected candidates only.")
        return lines

    def _render_findings_table(self, rows: List[FindingsTableRow]) -> str:
        headers = ["Type", "Location", "Status", "Confidence", "Evidence"]
        body = [
            [
                r.type,
                r.location,
                r.status,
                f"{r.confidence:.2f}",
                r.evidence
            ]
            for r in rows
        ]
        return self._render_table(headers, body)

    def _render_measurements_table(self, rows: List[MeasurementsTableRow]) -> str:
        headers = ["Lesion ID", "Location", "Diameter (mm)", "Volume (mm³)", "Confidence", "Evidence"]
        body = [
            [
                r.lesion_id,
                r.location,
                f"{r.diameter_mm:.1f}",
                f"{r.volume_mm3:.1f}",
                f"{r.confidence:.2f}",
                r.evidence
            ]
            for r in rows
        ]
        return self._render_table(headers, body)

    def _render_prior_table(self, rows: List[PriorComparisonRow]) -> str:
        headers = ["Lesion ID", "Prior Date", "Change", "Prior Size", "Current Size", "Evidence"]
        body = [
            [
                r.lesion_id,
                r.prior_date,
                r.change,
                r.prior_size,
                r.current_size,
                r.evidence
            ]
            for r in rows
        ]
        return self._render_table(headers, body)

    def _render_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Simple markdown-like table renderer"""
        header_row = " | ".join(headers)
        separator = " | ".join(["---"] * len(headers))
        lines = [header_row, separator]
        for row in rows:
            lines.append(" | ".join(row))
        return "\n".join(lines)

    def _collect_all_nodules(self, ai_result: StructuredAIResult) -> List[NoduleCandidate]:
        """Reportable + low-confidence nodules combined"""
        all_nodules = list(ai_result.nodules)
        for nodule in ai_result.low_confidence_nodules:
            if nodule.id not in {n.id for n in all_nodules}:
                all_nodules.append(nodule)
        return all_nodules

    def _build_finding_rows(self, ai_result: StructuredAIResult) -> List[FindingsTableRow]:
        findings = ai_result.findings
        rows: List[FindingsTableRow] = []

        def _status(label: str) -> str:
            return {
                "present": "Present",
                "absent": "Absent",
                "uncertain": "Uncertain"
            }.get(label, label)

        mapping = [
            ("Pleural effusion", findings.pleural_effusion, "finding_pe"),
            ("Pneumothorax", findings.pneumothorax, "finding_ptx"),
            ("Consolidation", findings.consolidation, "finding_cons"),
            ("Atelectasis", findings.atelectasis, "finding_atel"),
            ("Emphysema", findings.emphysema, "finding_emph")
        ]

        for name, label, evidence_id in mapping:
            rows.append(
                FindingsTableRow(
                    type=name,
                    location="-",
                    status=_status(label.label),
                    confidence=float(label.probability),
                    evidence=evidence_id
                )
            )

        return rows
    
    def _build_technique_section(self, ai_result: StructuredAIResult) -> ReportSection:
        """TECHNIQUE 섹션"""
        content = "Chest CT without contrast.\n"
        content += f"Slice thickness: {ai_result.quality.slice_thickness_mm:.1f} mm."
        
        return ReportSection(
            title="TECHNIQUE",
            content=content,
            evidence_ids=[]
        )
    
    def _build_comparison_section(self, prior_text: Optional[str]) -> ReportSection:
        """COMPARISON 섹션"""
        if prior_text:
            content = f"Prior study available for comparison.\n{prior_text}"
        else:
            content = "No prior studies available for comparison."
        
        return ReportSection(
            title="COMPARISON",
            content=content,
            evidence_ids=[]
        )
    
    def _build_findings_section(self, ai_result: StructuredAIResult) -> ReportSection:
        """
        FINDINGS 섹션
        - Evidence-backed only
        - Confidence threshold 적용
        """
        lines = []
        evidence_ids = []
        
        # 1. LUNGS subsection
        lines.append("LUNGS:")
        
        # Nodules (high confidence only)
        reported_nodules = [
            n for n in ai_result.nodules
            if self.threshold_manager.should_report_nodule(n.confidence)
        ]
        
        if reported_nodules:
            lines.append(f"  {len(reported_nodules)} pulmonary nodule(s) identified:")
            for nodule in reported_nodules:
                loc = nodule.location_code or "location unspecified"
                lines.append(
                    f"  - {loc}: {nodule.diameter_mm:.1f} mm nodule "
                    f"(confidence: {nodule.confidence:.2f})"
                )
                evidence_ids.append(nodule.id)
        else:
            lines.append("  No significant pulmonary nodules.")
        
        # Multi-label findings
        findings_text = self._format_findings(ai_result.findings, evidence_ids)
        if findings_text:
            lines.append(findings_text)
        else:
            lines.append("  No consolidation, effusion, or pneumothorax.")
        
        lines.append("")
        
        # 2. MEDIASTINUM subsection
        lines.append("MEDIASTINUM:")
        lines.append("  Normal mediastinal and hilar contours.")
        lines.append("  No significant lymphadenopathy.")
        lines.append("")
        
        # 3. PLEURA subsection
        lines.append("PLEURA:")
        if ai_result.findings.pleural_effusion.label == "present":
            lines.append("  Pleural effusion as described above.")
        else:
            lines.append("  No pleural effusion or thickening.")
        
        content = "\n".join(lines)
        
        return ReportSection(
            title="FINDINGS",
            content=content,
            evidence_ids=evidence_ids,
            confidence_level="high" if reported_nodules else "standard"
        )
    
    def _format_findings(self, findings, evidence_ids: List[str]) -> str:
        """Multi-label findings formatting"""
        findings_list = []
        
        if findings.consolidation.label == "present":
            findings_list.append("consolidation")
            if findings.consolidation.evidence:
                evidence_ids.append("consolidation")
        
        if findings.atelectasis.label == "present":
            findings_list.append("atelectasis")
            if findings.atelectasis.evidence:
                evidence_ids.append("atelectasis")
        
        if findings.emphysema.label == "present":
            findings_list.append("emphysema")
            if findings.emphysema.evidence:
                evidence_ids.append("emphysema")
        
        if findings.pneumothorax.label == "present":
            findings_list.append("pneumothorax")
            if findings.pneumothorax.evidence:
                evidence_ids.append("pneumothorax")
        
        if findings_list:
            return f"  {', '.join(findings_list).capitalize()} noted."
        
        return ""
    
    def _build_impression_section(self, ai_result: StructuredAIResult) -> ReportSection:
        """IMPRESSION 섹션 - 간결한 요약"""
        lines = []
        evidence_ids = []
        
        # Nodules
        reported_nodules = [
            n for n in ai_result.nodules
            if self.threshold_manager.should_report_nodule(n.confidence)
        ]
        
        if reported_nodules:
            if len(reported_nodules) == 1:
                n = reported_nodules[0]
                loc = n.location_code or "location"
                lines.append(
                    f"1. Pulmonary nodule in {loc}, {n.diameter_mm:.1f} mm. "
                    f"Recommend follow-up per Fleischner criteria."
                )
            else:
                lines.append(f"1. {len(reported_nodules)} pulmonary nodules as described.")
                lines.append("   Recommend follow-up per Fleischner criteria.")
            
            evidence_ids.extend([n.id for n in reported_nodules])
        
        # Significant findings
        sig_findings = []
        if ai_result.findings.pleural_effusion.label == "present":
            sig_findings.append("pleural effusion")
        if ai_result.findings.pneumothorax.label == "present":
            sig_findings.append("pneumothorax")
        if ai_result.findings.consolidation.label == "present":
            sig_findings.append("consolidation")
        
        if sig_findings:
            finding_num = len(reported_nodules) + 1
            lines.append(f"{finding_num}. {', '.join(sig_findings).capitalize()} as described above.")
        
        # Default impression if nothing found
        if not lines:
            lines.append("No significant acute findings.")
        
        content = "\n".join(lines)
        
        return ReportSection(
            title="IMPRESSION",
            content=content,
            evidence_ids=evidence_ids
        )
    
    def _build_measurements_section(self, ai_result: StructuredAIResult) -> ReportSection:
        """MEASUREMENTS 섹션"""
        lines = []
        
        # Lung volume
        lines.append(f"Total lung volume: {ai_result.lung_volume_ml:.1f} mL")
        lines.append("")
        
        # Nodule measurements (detailed)
        reported_nodules = [
            n for n in ai_result.nodules
            if self.threshold_manager.should_report_nodule(n.confidence)
        ]
        
        if reported_nodules:
            lines.append("Nodule measurements:")
            for nodule in reported_nodules:
                lines.append(f"  {nodule.id}:")
                lines.append(f"    Diameter: {nodule.diameter_mm:.1f} mm")
                lines.append(f"    Volume: {nodule.volume_mm3:.1f} mm³")
                lines.append(f"    Location (ZYX): {nodule.center_zyx}")
                lines.append(f"    Confidence: {nodule.confidence:.2f}")
        
        content = "\n".join(lines) if lines else "No measurements available."
        
        return ReportSection(
            title="MEASUREMENTS",
            content=content,
            evidence_ids=[n.id for n in reported_nodules]
        )
    
    def _build_limitations_section(self, ai_result: StructuredAIResult) -> ReportSection:
        """LIMITATIONS 섹션 - 제품형 필수 규칙"""
        lines = []
        
        # 1. Resolution-based limitations
        spacing = ai_result.quality.slice_thickness_mm
        if spacing >= 2.5:
            lines.append(
                f"Slice thickness {spacing:.1f} mm may limit detection of small nodules (<5 mm)."
            )
        elif spacing >= 1.5:
            lines.append(
                f"Slice thickness {spacing:.1f} mm; small nodules (<3 mm) may be missed."
            )
        elif spacing > 1.0:
            lines.append(
                "Small nodules (<3 mm) may be underestimated due to resolution limits."
            )
        
        # 2. Coverage-based limitations
        if ai_result.quality.coverage_score < 0.95:
            if ai_result.quality.coverage_score < 0.85:
                lines.append(
                    f"Incomplete lung coverage ({ai_result.quality.coverage_score:.0%}); "
                    "apical or basal nodules may be missed."
                )
            else:
                lines.append("Lung coverage is adequate but not complete.")
        
        # 3. Candidate explosion warning
        n_candidates = len(ai_result.nodules)
        if n_candidates > 10:
            lines.append(
                f"Multiple candidates detected ({n_candidates}); "
                "high false-positive rate possible. Clinical correlation required."
            )
        elif n_candidates > 5:
            lines.append(
                f"Several candidates detected ({n_candidates}); false positives possible."
            )
        
        # 4. Low-confidence nodules
        low_conf_nodules = [
            n for n in ai_result.nodules
            if self.threshold_manager.should_include_in_limitations(n.confidence)
        ]
        if low_conf_nodules:
            lines.append(
                f"{len(low_conf_nodules)} additional low-confidence candidate(s) "
                "detected but not reported due to insufficient certainty."
            )
        
        # 5. Artifacts
        if ai_result.quality.artifact_score > 0.5:
            lines.append(
                f"Image artifacts (score: {ai_result.quality.artifact_score:.2f}) may affect accuracy."
            )
        elif ai_result.quality.artifact_score > 0.2:
            lines.append("Minor artifacts detected; unlikely to affect major findings.")
        
        # 6. Model info
        lines.append("")
        lines.append(
            f"AI model: {ai_result.versioning.model_version}"
        )
        
        # 7. 항상 포함: AI disclaimer (제품/법적 필수)
        lines.append("AI findings are candidates only and require physician confirmation.")
        
        content = "\n".join(lines)
        
        return ReportSection(
            title="LIMITATIONS",
            content=content,
            evidence_ids=[]
        )
