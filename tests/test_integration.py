# tests/test_integration.py
"""
Integration test for PACS AI Agent
Mock mode end-to-end test
"""
import pytest
import asyncio
from datetime import datetime

from api.schemas import (
    StructuredAIResult, QualityMetrics, NoduleCandidate,
    VisionEvidence, StructuredFindings, FindingLabel,
    ModelVersioning
)
from solar_integration.report_generator import ProductionReportGenerator
from solar_integration.rag_system import MedicalRAGSystem
from solar_integration.validator import ReportValidator
from solar_integration.templates import TemplateReportBuilder
from monai_pipeline.calibration import ThresholdManager
from utils.logger import logger


@pytest.fixture
def mock_ai_result():
    """Mock AI result fixture"""
    return StructuredAIResult(
        study_uid="1.2.3.4.5.test",
        series_uid="1.2.3.4.5.test.series",
        acquisition_datetime=datetime.now(),
        
        quality=QualityMetrics(
            slice_thickness_mm=2.5,
            coverage_score=0.95,
            artifact_score=0.1
        ),
        
        lung_volume_ml=4850.5,
        
        nodules=[
            NoduleCandidate(
                id="N1",
                center_zyx=(50, 100, 120),
                bbox_zyx=(45, 95, 115, 55, 105, 125),
                diameter_mm=5.2,
                volume_mm3=73.6,
                confidence=0.89,
                evidence=VisionEvidence(
                    series_uid="1.2.3.4.5.test.series",
                    instance_uids=["instance.1"],
                    slice_range=(45, 55),
                    confidence=0.89
                ),
                location_code="RUL"
            )
        ],
        
        findings=StructuredFindings(
            pleural_effusion=FindingLabel(label="absent", probability=0.1),
            pneumothorax=FindingLabel(label="absent", probability=0.05),
            consolidation=FindingLabel(label="absent", probability=0.15),
            atelectasis=FindingLabel(label="absent", probability=0.2),
            emphysema=FindingLabel(label="uncertain", probability=0.5)
        ),
        
        versioning=ModelVersioning(
            model_version="test-v1.0",
            pipeline_version="1.0.0",
            thresholds={"nodule_reporting": 0.75}
        ),
        
        processing_time_seconds=1.5
    )


@pytest.mark.asyncio
async def test_template_generation(mock_ai_result):
    """Test template generation without LLM"""
    logger.info("\n" + "="*80)
    logger.info("TEST: Template Generation")
    logger.info("="*80)
    
    builder = TemplateReportBuilder()
    report = builder.build_report(mock_ai_result)
    
    assert report is not None
    assert report.findings.content != ""
    assert report.impression.content != ""
    assert "nodule" in report.findings.content.lower()
    assert len(report.findings.evidence_ids) > 0
    
    logger.info("✓ Template generation successful")
    logger.info(f"  Findings: {report.findings.content[:100]}...")
    logger.info(f"  Evidence IDs: {report.findings.evidence_ids}")


@pytest.mark.asyncio
async def test_validation(mock_ai_result):
    """Test validator"""
    logger.info("\n" + "="*80)
    logger.info("TEST: Validator")
    logger.info("="*80)
    
    validator = ReportValidator()
    builder = TemplateReportBuilder()
    
    # Generate template
    report = builder.build_report(mock_ai_result)
    
    # Pre-LLM validation
    pre_validation = validator.validate_pre_llm(
        mock_ai_result,
        report.findings.content
    )
    
    assert pre_validation.passed or len(pre_validation.errors) == 0
    logger.info(f"✓ Pre-LLM validation: {pre_validation.passed}")
    
    # Post-LLM validation (with forbidden patterns)
    bad_text = "This study confirms diagnosis of malignancy. Patient should undergo surgery."
    
    post_validation = validator.validate_post_llm(
        mock_ai_result,
        bad_text,
        report.findings.content
    )
    
    assert not post_validation.passed
    assert len(post_validation.blocked_content) > 0
    logger.info(f"✓ Post-LLM validation correctly blocked: {post_validation.blocked_content}")


@pytest.mark.asyncio
async def test_mock_report_generation(mock_ai_result):
    """Test full report generation in mock mode"""
    logger.info("\n" + "="*80)
    logger.info("TEST: Mock Report Generation")
    logger.info("="*80)
    
    # Use mock mode (no API calls)
    generator = ProductionReportGenerator(use_mock_solar=True)
    
    report = await generator.generate_report(
        ai_result=mock_ai_result,
        patient_id="TEST_PATIENT_001",
        enable_llm_rewrite=False  # Skip LLM in test
    )
    
    assert report is not None
    assert report.validation_passed
    assert report.findings.content != ""
    assert report.impression.content != ""
    
    logger.info("✓ Mock report generation successful")
    logger.info(f"  Study UID: {report.study_uid}")
    logger.info(f"  Validation: {report.validation_passed}")
    logger.info(f"  Warnings: {len(report.validation_warnings)}")


@pytest.mark.asyncio
async def test_rag_system():
    """Test RAG system with mock embedding"""
    logger.info("\n" + "="*80)
    logger.info("TEST: RAG System (Mock Embedding)")
    logger.info("="*80)
    
    # Use mock embedding
    rag = MedicalRAGSystem(use_mock_embedding=True)
    
    # Create test AI result
    from api.schemas import StructuredAIResult, QualityMetrics, StructuredFindings, FindingLabel, ModelVersioning
    
    test_result = StructuredAIResult(
        study_uid="1.2.3.test.rag",
        series_uid="1.2.3.test.rag.series",
        acquisition_datetime=datetime.now(),
        quality=QualityMetrics(
            slice_thickness_mm=2.5,
            coverage_score=0.95,
            artifact_score=0.1
        ),
        lung_volume_ml=5000.0,
        nodules=[],
        findings=StructuredFindings(
            pleural_effusion=FindingLabel(label="absent", probability=0.1),
            pneumothorax=FindingLabel(label="absent", probability=0.05),
            consolidation=FindingLabel(label="absent", probability=0.15),
            atelectasis=FindingLabel(label="absent", probability=0.2),
            emphysema=FindingLabel(label="absent", probability=0.3)
        ),
        versioning=ModelVersioning(
            model_version="test-v1.0",
            pipeline_version="1.0.0",
            thresholds={}
        ),
        processing_time_seconds=1.0
    )
    
    # Store report
    await rag.store_report(
        patient_id="TEST_PT_RAG",
        study_uid="1.2.3.test.rag",
        study_date="20260101",
        report_text="Normal chest CT. No significant findings.",
        ai_result=test_result
    )
    
    logger.info("✓ Report stored")
    
    # Retrieve history
    history = rag.retrieve_patient_history("TEST_PT_RAG")
    assert len(history) > 0
    logger.info(f"✓ Retrieved {len(history)} studies")
    
    # Semantic search
    results = await rag.semantic_search(
        query_text="pulmonary nodule",
        patient_id="TEST_PT_RAG",
        n_results=1
    )
    logger.info(f"✓ Semantic search returned {len(results)} results")


@pytest.mark.asyncio
async def test_end_to_end(mock_ai_result):
    """End-to-end test: AI result → Report with RAG"""
    logger.info("\n" + "="*80)
    logger.info("TEST: End-to-End Pipeline")
    logger.info("="*80)
    
    # Initialize components (mock mode)
    rag = MedicalRAGSystem(use_mock_embedding=True)
    generator = ProductionReportGenerator(
        use_mock_solar=True,
        rag_system=rag
    )
    
    # Store a prior study first
    prior_result = StructuredAIResult(
        study_uid="1.2.3.prior",
        series_uid="1.2.3.prior.series",
        acquisition_datetime=datetime(2025, 1, 1),
        quality=QualityMetrics(
            slice_thickness_mm=2.5,
            coverage_score=0.95,
            artifact_score=0.1
        ),
        lung_volume_ml=4800.0,
        nodules=[
            NoduleCandidate(
                id="N1_prior",
                center_zyx=(50, 100, 120),
                bbox_zyx=(45, 95, 115, 55, 105, 125),
                diameter_mm=5.0,
                volume_mm3=65.0,
                confidence=0.85,
                evidence=VisionEvidence(
                    series_uid="1.2.3.prior.series",
                    instance_uids=["instance.prior.1"],
                    slice_range=(45, 55),
                    confidence=0.85
                ),
                location_code="RUL"
            )
        ],
        findings=StructuredFindings(
            pleural_effusion=FindingLabel(label="absent", probability=0.1),
            pneumothorax=FindingLabel(label="absent", probability=0.05),
            consolidation=FindingLabel(label="absent", probability=0.15),
            atelectasis=FindingLabel(label="absent", probability=0.2),
            emphysema=FindingLabel(label="absent", probability=0.3)
        ),
        versioning=ModelVersioning(
            model_version="test-v1.0",
            pipeline_version="1.0.0",
            thresholds={}
        ),
        processing_time_seconds=1.0
    )
    
    await rag.store_report(
        patient_id="TEST_E2E_PATIENT",
        study_uid="1.2.3.prior",
        study_date="20250101",
        report_text="Prior study with small RUL nodule.",
        ai_result=prior_result
    )
    
    logger.info("✓ Prior study stored")
    
    # Generate report for current study (with prior comparison)
    mock_ai_result.acquisition_datetime = datetime(2026, 1, 1)
    
    report = await generator.generate_report(
        ai_result=mock_ai_result,
        patient_id="TEST_E2E_PATIENT",
        include_prior_comparison=True,
        enable_llm_rewrite=False
    )
    
    assert report is not None
    assert report.validation_passed
    logger.info("✓ Report generated with prior comparison")
    logger.info(f"  Prior comparison included: {report.prior_comparison_included}")
    
    logger.info("\n" + "="*80)
    logger.info("ALL TESTS PASSED")
    logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(test_end_to_end(pytest.fixture(mock_ai_result)()))
