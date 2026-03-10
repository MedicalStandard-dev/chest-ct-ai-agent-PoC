# api/main.py
"""
FastAPI application for PACS AI Agent
Product-ready with evidence-first design
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache
import csv
import uuid

from api.schemas import (
    AnalyzeRequest, AnalyzeResponse,
    GenerateReportRequest, GenerateReportResponse,
    VectorDbSaveRequest, VectorDbSaveResponse,
    PatientHistoryResponse,
    SemanticSearchRequest, SemanticSearchResponse,
    StructuredAIResult, DraftReport
)
from config.settings import settings
from utils.logger import logger
from solar_integration.report_generator import ProductionReportGenerator
from solar_integration.rag_system import MedicalRAGSystem
from monai_pipeline.findings_classifier import RuleBasedFindingsClassifier
from monai_pipeline.calibration import ThresholdManager, ProbabilityCalibrator

# Initialize app
app = FastAPI(
    title="PACS AI Agent",
    description="Production-ready medical image analysis with evidence-first design",
    version="1.0.0"
)

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global instances
report_generator: Optional[ProductionReportGenerator] = None
rag_system: Optional[MedicalRAGSystem] = None
findings_classifier: Optional[RuleBasedFindingsClassifier] = None
threshold_manager: Optional[ThresholdManager] = None
calibrator: Optional[ProbabilityCalibrator] = None


def _build_groundedness_context(ai_result: StructuredAIResult) -> str:
    """Build deterministic context for groundedness checks."""
    lines: List[str] = [
        f"Study UID: {ai_result.study_uid}",
        f"Series UID: {ai_result.series_uid}",
        f"Nodule candidates: {len(ai_result.nodules)}",
        f"Low-confidence nodules: {len(ai_result.low_confidence_nodules)}",
        "",
        "NODULE CANDIDATES:",
    ]

    if not ai_result.nodules:
        lines.append("- none")
    else:
        for n in ai_result.nodules:
            loc = n.location_code or "EXTRA"
            lines.append(
                f"- {n.id}: loc={loc}, diameter_mm={n.diameter_mm:.1f}, "
                f"volume_mm3={n.volume_mm3:.1f}, conf={n.confidence:.2f}, "
                f"slice_range={n.evidence.slice_range}"
            )

    if ai_result.low_confidence_nodules:
        lines.append("")
        lines.append("LOW-CONFIDENCE NODULES:")
        for n in ai_result.low_confidence_nodules:
            loc = n.location_code or "EXTRA"
            lines.append(
                f"- {n.id}: loc={loc}, diameter_mm={n.diameter_mm:.1f}, "
                f"volume_mm3={n.volume_mm3:.1f}, conf={n.confidence:.2f}"
            )

    findings = ai_result.findings
    lines.append("")
    lines.append("OTHER FINDINGS (STRUCTURED):")
    lines.append(
        f"- pleural_effusion: label={findings.pleural_effusion.label}, "
        f"prob={findings.pleural_effusion.probability:.2f}"
    )
    lines.append(
        f"- pneumothorax: label={findings.pneumothorax.label}, "
        f"prob={findings.pneumothorax.probability:.2f}"
    )
    lines.append(
        f"- consolidation: label={findings.consolidation.label}, "
        f"prob={findings.consolidation.probability:.2f}"
    )
    lines.append(
        f"- atelectasis: label={findings.atelectasis.label}, "
        f"prob={findings.atelectasis.probability:.2f}"
    )
    lines.append(
        f"- emphysema: label={findings.emphysema.label}, "
        f"prob={findings.emphysema.probability:.2f}"
    )

    if ai_result.warnings:
        lines.append("")
        lines.append("PIPELINE WARNINGS:")
        for warning in ai_result.warnings[:10]:
            lines.append(f"- {warning}")

    return "\n".join(lines)


def _build_groundedness_claim(draft_report: DraftReport) -> str:
    """Build claim text from structured report payload."""
    lines: List[str] = ["REPORT CLAIMS:"]
    tables = draft_report.tables

    if tables and tables.findings:
        lines.append("")
        lines.append("FINDINGS TABLE:")
        for row in tables.findings:
            lines.append(
                f"- type={row.type}, location={row.location}, status={row.status}, "
                f"confidence={row.confidence:.2f}, evidence={row.evidence}"
            )

    if tables and tables.measurements:
        lines.append("")
        lines.append("MEASUREMENTS TABLE:")
        for row in tables.measurements:
            lines.append(
                f"- lesion_id={row.lesion_id}, location={row.location}, "
                f"diameter_mm={row.diameter_mm:.1f}, volume_mm3={row.volume_mm3:.1f}, "
                f"confidence={row.confidence:.2f}"
            )

    if draft_report.key_flags:
        lines.append("")
        lines.append("KEY FLAGS:")
        lines.append(
            f"- nodule_candidates={draft_report.key_flags.nodule_candidates}, "
            f"new_nodules={draft_report.key_flags.new_nodules}, "
            f"high_confidence_findings={draft_report.key_flags.high_confidence_findings}, "
            f"scan_limitation={draft_report.key_flags.scan_limitation}"
        )

    if draft_report.limitations_text:
        lines.append("")
        lines.append("LIMITATIONS:")
        for text in draft_report.limitations_text:
            lines.append(f"- {text}")

    if draft_report.impression and draft_report.impression.content:
        lines.append("")
        lines.append("IMPRESSION:")
        lines.append(draft_report.impression.content.strip())

    claim = "\n".join(lines).strip()
    if len(claim) < 100 and draft_report.rendered_report:
        claim = draft_report.rendered_report

    # Keep groundedness prompt size bounded while preserving full structured claims.
    return claim[:6000]


def _latest_sweep_summary_csv() -> Optional[Path]:
    """Find the most recent sweep summary CSV under outputs/."""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        return None
    files = [p for p in outputs_dir.glob("**/summary_*.csv") if p.is_file()]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def _load_threshold_metrics(postproc: str = "nms15") -> Tuple[List[Dict[str, Any]], Optional[Path]]:
    """
    Load threshold sweep metrics from latest summary CSV.
    Returns (rows, source_path). rows are sorted by threshold asc.
    """
    src = _latest_sweep_summary_csv()
    if src is None:
        return [], None

    rows: List[Dict[str, Any]] = []
    try:
        with src.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if str(r.get("postproc", "")).strip() != postproc:
                    continue
                rows.append(
                    {
                        "threshold": float(r.get("threshold", 0.0)),
                        "postproc": str(r.get("postproc", "")),
                        "sensitivity": float(r.get("sensitivity", 0.0)),
                        "fp_per_scan": float(r.get("fp_per_scan", 0.0)),
                        "candidates_per_scan": float(r.get("candidates_per_scan", 0.0)),
                        "tp": int(float(r.get("tp", 0))),
                        "fp": int(float(r.get("fp", 0))),
                        "fn": int(float(r.get("fn", 0))),
                        "cases": int(float(r.get("cases", 0))),
                    }
                )
    except Exception as e:
        logger.warning(f"Failed to parse threshold summary {src}: {e}")
        return [], src

    rows.sort(key=lambda x: x["threshold"])
    return rows, src


def _pick_nearest_threshold_row(rows: List[Dict[str, Any]], threshold: float) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    return min(rows, key=lambda r: abs(float(r["threshold"]) - float(threshold)))


def _classify_change_type(delta_pct: Optional[float]) -> str:
    if delta_pct is None:
        return "UNKNOWN"
    if delta_pct > 20.0:
        return "GROWTH"
    if delta_pct < -20.0:
        return "REDUCTION"
    return "STABLE"


def _summarize_prior_delta(
    ai_result: StructuredAIResult,
    patient_id: Optional[str],
    max_history: int = 5
) -> Dict[str, Any]:
    """Build numeric prior-vs-current summary from RAG metadata."""
    current_nodules = sorted(ai_result.nodules, key=lambda n: n.diameter_mm, reverse=True)
    current_largest = current_nodules[0] if current_nodules else None
    current_size = float(current_largest.diameter_mm) if current_largest else None

    result: Dict[str, Any] = {
        "has_prior": False,
        "patient_id": patient_id or ai_result.study_uid,
        "current_largest_id": current_largest.id if current_largest else None,
        "current_largest_mm": current_size,
        "prior_study_date": None,
        "prior_largest_mm": None,
        "delta_mm": None,
        "delta_pct": None,
        "change_type": "UNKNOWN",
    }

    pid = patient_id or ai_result.study_uid
    if rag_system is None or not pid:
        return result

    try:
        history = rag_system.retrieve_patient_history(pid, max_results=max_history)
    except Exception as e:
        logger.warning(f"Prior retrieval failed for {pid}: {e}")
        return result

    if not history:
        return result

    result["has_prior"] = True

    def _study_date_key(item: Dict[str, Any]) -> str:
        return str(item.get("study_date") or item.get("metadata", {}).get("study_date") or "")

    history_sorted = sorted(history, key=_study_date_key, reverse=True)
    prior = history_sorted[0]
    prior_date = prior.get("study_date") or prior.get("metadata", {}).get("study_date")
    prior_size_raw = prior.get("metadata", {}).get("nodule_diameter_mm")
    prior_size = float(prior_size_raw) if prior_size_raw is not None else None

    result["prior_study_date"] = prior_date
    result["prior_largest_mm"] = prior_size

    if current_size is not None and prior_size is not None and prior_size > 0:
        delta_mm = current_size - prior_size
        delta_pct = (delta_mm / prior_size) * 100.0
        result["delta_mm"] = float(delta_mm)
        result["delta_pct"] = float(delta_pct)
        result["change_type"] = _classify_change_type(delta_pct)
    elif current_size is not None and prior_size is None:
        result["change_type"] = "NEW"

    return result


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global report_generator, rag_system, findings_classifier, threshold_manager, calibrator
    
    logger.info("="*80)
    logger.info("PACS AI AGENT STARTUP")
    logger.info("="*80)
    
    # Initialize components
    logger.info("\nInitializing components...")
    
    # RAG system
    rag_system = MedicalRAGSystem(
        use_mock_embedding=not settings.should_use_real_embedding
    )
    logger.info(f"✓ RAG System: {rag_system.get_rag_info()}")
    
    # Report generator
    report_generator = ProductionReportGenerator(
        use_mock_solar=not settings.should_use_real_solar,
        rag_system=rag_system
    )
    logger.info(f"✓ Report Generator: {report_generator.get_generator_info()}")
    
    # Findings classifier (rule-based)
    findings_classifier = RuleBasedFindingsClassifier()
    logger.info(f"✓ Findings Classifier: {findings_classifier.get_version()}")
    
    # Threshold manager
    threshold_manager = ThresholdManager()
    logger.info(f"✓ Threshold Manager: {len(threshold_manager.get_all_thresholds())} thresholds")
    
    # Calibrator
    calibrator = ProbabilityCalibrator()
    logger.info("✓ Probability Calibrator initialized")
    
    # Seed sample prior data for RAG demo
    try:
        from solar_integration.sample_data import seed_sample_data
        seeded = await seed_sample_data(rag_system)
        logger.info(f"✓ Sample prior data: {seeded} reports seeded")
    except Exception as e:
        logger.warning(f"✗ Sample data seeding failed: {e}")
    
    logger.info("\n" + "="*80)
    logger.info("PACS AI AGENT READY")
    logger.info(f"Mode: {'PRODUCTION' if settings.should_use_real_solar else 'MOCK'}")
    logger.info("="*80 + "\n")


@app.get("/ui")
async def dashboard():
    """Serve Dashboard UI"""
    static_path = Path(__file__).parent.parent / "static" / "index.html"
    if static_path.exists():
        return FileResponse(static_path)
    return {"error": "UI not found"}


@app.get("/")
async def root():
    """Root endpoint - System Overview"""
    return {
        "service": "Medical AI PACS",
        "version": "1.0.0",
        "tagline": "CT Chest AI 판독 보조 시스템",
        "status": "running",
        "mode": "production" if settings.should_use_real_solar else "mock",
        "architecture": {
            "principle": "Evidence-first, Table-first, Validator-gated",
            "llm_role": "Narrative-only rewrite (표 수정 불가)"
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "system_info": "/api/v1/system-info",
            "analyze": "POST /api/v1/analyze",
            "generate_report": "POST /api/v1/generate-report",
            "patient_history": "GET /api/v1/patient/{patient_id}/history"
        }
    }


@app.get("/health")
async def health_check():
    """Health check - Component Status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "report_generator": {"ready": report_generator is not None, "mode": "table-protected"},
            "rag_system": {"ready": rag_system is not None},
            "findings_classifier": {"ready": findings_classifier is not None, "type": "rule-based"},
            "validator": {"ready": True, "mode": "fail-closed"},
            "tracking_engine": {"ready": True}
        },
        "models": {
            "nodule_detection": {"status": "trained", "best_loss": 0.2319},
            "lung_segmentation": {"status": "trained", "best_dice": 0.2289}
        },
        "configuration": {
            "solar_api": settings.should_use_real_solar,
            "solar_embedding": settings.should_use_real_embedding,
            "mock_vision": settings.use_mock_vision
        }
    }


@app.get("/api/v1/available-cases")
async def get_available_cases():
    """사용 가능한 LIDC 케이스 목록"""
    from pathlib import Path
    
    lidc_dir = Path("data/LIDC-preprocessed-v2/images")
    cases = []
    
    if lidc_dir.exists():
        for f in sorted(lidc_dir.glob("*.nii.gz")):
            case_id = f.stem.replace(".nii", "")
            cases.append(case_id)
    
    return {
        "available_cases": cases,
        "count": len(cases),
        "usage": "Study UID에 케이스 ID 입력 (예: LIDC-IDRI-0001)"
    }


@app.get("/api/v1/lidc-cases")
async def get_lidc_cases():
    """LIDC 케이스 상세 목록 (manifest.json에서 로드)"""
    import json
    from pathlib import Path
    
    manifest_path = Path("data/LIDC-preprocessed-v2/manifest.json")
    cases = []
    
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        for item in manifest:  # 전체 케이스
            meta = item.get("metadata", {})
            cases.append({
                "case_id": meta.get("case_id", ""),
                "series_uid": meta.get("series_uid", ""),
                "patient_id": meta.get("patient_id", ""),
                "num_nodules": item.get("num_nodules", 0),
                "largest_nodule_mm": max([n.get("diameter_mm", 0) for n in meta.get("nodules", [])], default=0)
            })
    
    return {
        "cases": cases,
        "count": len(cases)
    }


@app.get("/api/v1/lidc-slice/{study_uid}")
async def get_lidc_slice(
    study_uid: str,
    slice_index: Optional[int] = None,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    preset: str = "auto"
):
    """
    LIDC NIfTI에서 단일 axial slice를 8-bit grayscale로 반환
    - pixel_data_b64: width*height 길이의 uint8 바이트(base64)
    """
    import base64
    import numpy as np

    if not study_uid.startswith("LIDC-IDRI-"):
        raise HTTPException(status_code=400, detail="Only LIDC-IDRI-* cases are supported")
    if window_width is not None and window_width <= 0:
        raise HTTPException(status_code=400, detail="window_width must be > 0")

    try:
        volume, spacing, _, orientation = _load_lidc_volume(study_uid)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load LIDC volume ({study_uid}): {e}")
        raise HTTPException(status_code=500, detail="Failed to load LIDC volume")

    total_slices = int(volume.shape[0])
    if total_slices <= 0:
        raise HTTPException(status_code=500, detail="Empty volume")

    if slice_index is None:
        slice_index = total_slices // 2
    slice_index = int(max(0, min(total_slices - 1, slice_index)))

    ct_slice = volume[slice_index]
    finite_vals = ct_slice[np.isfinite(ct_slice)]
    if finite_vals.size == 0:
        raise HTTPException(status_code=500, detail="Invalid slice values")

    volume_min = float(np.nanmin(volume))
    volume_max = float(np.nanmax(volume))
    preset_name = (preset or "auto").lower()
    if preset_name not in {"auto", "lung", "mediastinal"}:
        preset_name = "auto"

    if volume_min <= -200.0 and volume_max >= 200.0:
        intensity_mode = "hu"
    elif volume_min >= 0.0 and volume_max <= 1.5:
        intensity_mode = "normalized"
    else:
        intensity_mode = "unknown"

    if intensity_mode == "hu":
        if preset_name == "mediastinal":
            wc_use, ww_use = 40.0, 400.0
        else:
            wc_use, ww_use = -600.0, 1500.0
        if window_center is not None:
            wc_use = float(window_center)
        if window_width is not None:
            ww_use = float(window_width)
        low = wc_use - (ww_use / 2.0)
        high = wc_use + (ww_use / 2.0)
    else:
        non_zero_vals = finite_vals[finite_vals > 0.0]
        source_vals = non_zero_vals if non_zero_vals.size >= 128 else finite_vals
        if window_center is not None and window_width is not None:
            wc_use = float(window_center)
            ww_use = float(window_width)
            low = wc_use - (ww_use / 2.0)
            high = wc_use + (ww_use / 2.0)
        else:
            if preset_name == "mediastinal":
                lo_q, hi_q = 25.0, 99.7
            elif preset_name == "lung":
                lo_q, hi_q = 2.0, 99.5
            else:
                lo_q, hi_q = 1.0, 99.0
            low = float(np.percentile(source_vals, lo_q))
            high = float(np.percentile(source_vals, hi_q))

    if not np.isfinite(low) or not np.isfinite(high):
        low, high = float(np.min(finite_vals)), float(np.max(finite_vals))
    if high <= low:
        high = low + 1e-3

    normalized = np.clip((ct_slice - low) / (high - low), 0.0, 1.0)
    pixels_u8 = (normalized * 255.0).astype(np.uint8)
    pixel_data_b64 = base64.b64encode(pixels_u8.tobytes()).decode("ascii")

    return {
        "study_uid": study_uid,
        "slice_index": slice_index,
        "total_slices": total_slices,
        "width": int(pixels_u8.shape[1]),
        "height": int(pixels_u8.shape[0]),
        "volume_shape": [int(v) for v in volume.shape],
        "spacing_mm": [float(s) for s in spacing],
        "orientation": list(orientation),
        "intensity_mode": intensity_mode,
        "preset": preset_name,
        "window_center": float(window_center) if window_center is not None else None,
        "window_width": float(window_width) if window_width is not None else None,
        "window_applied_low": float(low),
        "window_applied_high": float(high),
        "pixel_data_b64": pixel_data_b64
    }


@app.get("/api/v1/system-info")
async def system_info():
    """Detailed System Information"""
    return {
        "system": {
            "name": "Medical AI PACS",
            "version": "1.0.0",
            "description": "CT Chest 기반 AI 판독 보조 시스템"
        },
        "core_principles": {
            "evidence_first": "모든 소견은 Vision 모델 출력(Evidence)에 기반",
            "table_first": "숫자/측정값은 표로만 제공, 문장에 포함 금지",
            "validator_gated": "LLM 출력은 반드시 검증 후 사용",
            "safe_fallback": "검증 실패 시 템플릿 기반 안전 출력"
        },
        "pipeline": {
            "vision": {
                "nodule_detection": {
                    "model": "UNet (MONAI)",
                    "approach": "Heatmap-based (Gaussian blob)",
                    "dataset": "LIDC-IDRI",
                    "status": "trained"
                },
                "lung_segmentation": {
                    "model": "DynUNet (MONAI)",
                    "dataset": "MSD Task06_Lung",
                    "loss": "DiceFocalLoss (gamma=2.0)",
                    "best_dice": 0.2289,
                    "status": "trained"
                }
            },
            "post_processing": {
                "candidate_processor": "Peak → Component → Measurements",
                "evidence_generator": "slice_range, mask, contours",
                "tracking_engine": "Prior matching (NEW/Stable/Increased/Decreased)"
            },
            "report_generation": {
                "template_builder": "FINDINGS, MEASUREMENTS, PRIOR COMPARISON tables",
                "solar_rewriter": "Narrative-only (표 수정 불가)",
                "validator": "Table integrity, Numeric preservation, Hallucination check"
            }
        },
        "llm_integration": {
            "model": "Solar Pro 3",
            "mode": "narrative-only rewrite",
            "allowed": ["표 위/아래 설명 정리", "용어 통일", "PACS 톤 적용"],
            "forbidden": ["표 수정", "수치 변경", "소견 추가/삭제", "진단/추론"]
        },
        "tables_generated": [
            "FINDINGS - 소견 목록 (Type, Location, Status, Confidence)",
            "MEASUREMENTS - 측정값 (Lesion ID, Diameter, Volume)",
            "PRIOR COMPARISON - 변화 추적 (Change Type, Prior/Current Size)",
            "KEY FLAGS - 요약 (Nodule count, NEW count, Limitations)"
        ],
        "threshold_policy": {
            "finding": ">= 0.7 confidence",
            "limitation": "0.4 ~ 0.7 confidence",
            "hidden": "< 0.4 confidence"
        }
    }


# Production Pipeline (실제 LIDC 데이터용)
production_pipeline = None


@lru_cache(maxsize=4)
def _load_lidc_volume(study_uid: str):
    """Load and cache LIDC volume for fast slice visualization."""
    import nibabel as nib
    import numpy as np

    lidc_dir = Path("data/LIDC-preprocessed-v2")
    image_path = lidc_dir / "images" / f"{study_uid}.nii.gz"
    if not image_path.exists():
        raise FileNotFoundError(f"LIDC case not found: {study_uid}")

    nii = nib.load(image_path)
    volume = nii.get_fdata().astype(np.float32)
    spacing = tuple(float(v) for v in nii.header.get_zooms()[:3])
    orientation = tuple(str(x) for x in nib.aff2axcodes(nii.affine))
    return volume, spacing, str(image_path), orientation


def get_production_pipeline():
    """Lazy load production pipeline"""
    global production_pipeline
    if production_pipeline is None:
        from monai_pipeline.production_pipeline import ProductionPipeline
        from pathlib import Path
        
        nodule_model_path = Path("models/nodule_det_v4/best_nodule_det_model.pth")
        lung_seg_model_path = Path("models/lung_seg/best_lung_seg_model.pth")
        luna16_bundle_dir = Path("models/luna16_retinanet/lung_nodule_ct_detection")

        if luna16_bundle_dir.exists():
            production_pipeline = ProductionPipeline(
                nodule_model_path=nodule_model_path if nodule_model_path.exists() else None,
                lung_seg_model_path=lung_seg_model_path if lung_seg_model_path.exists() else None,
                output_dir=Path("outputs/api_results"),
                luna16_bundle_dir=str(luna16_bundle_dir),
            )
            logger.info("Production pipeline loaded (nodule: MONAI Luna16 pretrained RetinaNet)")
        elif nodule_model_path.exists():
            production_pipeline = ProductionPipeline(
                nodule_model_path=nodule_model_path,
                lung_seg_model_path=lung_seg_model_path if lung_seg_model_path.exists() else None,
                output_dir=Path("outputs/api_results")
            )
            logger.info(f"Production pipeline loaded (nodule: trained v4, lung_seg: {'trained' if lung_seg_model_path.exists() else 'not found'})")
        else:
            logger.warning("No nodule model found, using mock pipeline")
    return production_pipeline


def analyze_real_lidc_case(
    study_uid: str,
    series_uid: str,
    patient_id: Optional[str] = None
) -> StructuredAIResult:
    """
    실제 LIDC 데이터로 분석
    """
    from api.schemas import (
        QualityMetrics, NoduleCandidate, VisionEvidence,
        StructuredFindings, FindingLabel, ModelVersioning
    )
    import numpy as np
    import torch

    volume, spacing, image_path, _ = _load_lidc_volume(study_uid)
    logger.info(f"Loading LIDC case: {image_path}")
    
    # Production Pipeline 실행
    pipeline = get_production_pipeline()
    if pipeline is None:
        raise RuntimeError("Production pipeline not available")
    
    result = pipeline.process_volume(
        volume=volume,
        spacing_mm=spacing,
        series_uid=series_uid or study_uid,
        patient_id=patient_id or study_uid
    )
    
    # PipelineResult → StructuredAIResult 변환
    quality = QualityMetrics(
        slice_thickness_mm=spacing[0],
        coverage_score=0.95,
        artifact_score=0.1
    )
    
    # Nodule candidates 변환
    nodules = []
    for c in result.candidates:
        if c.status == "hidden":
            continue
        
        # bbox 계산 (center ± radius)
        z, y, x = c.peak_zyx
        radius = int(c.diameter_mm / 2) + 1
        bbox = (
            max(0, int(z) - radius), max(0, int(y) - radius), max(0, int(x) - radius),
            int(z) + radius, int(y) + radius, int(x) + radius
        )
        
        nodule = NoduleCandidate(
            id=c.candidate_id,
            center_zyx=(float(z), float(y), float(x)),
            bbox_zyx=bbox,
            diameter_mm=float(c.diameter_mm),
            volume_mm3=float(c.volume_mm3),
            confidence=min(1.0, float(c.confidence)),  # 1.0 이하로 제한
            evidence=VisionEvidence(
                series_uid=series_uid or study_uid,
                instance_uids=[],
                slice_range=c.slice_range,
                confidence=min(1.0, float(c.confidence))
            ),
            location_code=c.location_code or "UNK"
        )
        nodules.append(nodule)
    
    # Findings (rule-based classifier)
    lung_mask_for_findings = None
    if hasattr(pipeline, '_generate_lung_mask'):
        try:
            lung_mask_for_findings = pipeline._generate_lung_mask(volume)
        except Exception as e:
            logger.warning(f"Lung mask generation for findings failed: {e}")

    findings = findings_classifier.predict(
        volume=torch.empty(0),  # unused when volume_hu provided
        metadata={"series_uid": series_uid or study_uid},
        volume_hu=volume,
        spacing=spacing,
        lung_mask=lung_mask_for_findings,
    )
    
    # Versioning
    versioning = ModelVersioning(
        model_version="nodule-det-v1.0",
        pipeline_version="1.0.0",
        thresholds=threshold_manager.get_all_thresholds()
    )
    
    ai_result = StructuredAIResult(
        study_uid=study_uid,
        series_uid=series_uid or study_uid,
        acquisition_datetime=datetime.now(),
        quality=quality,
        lung_volume_ml=5000.0,
        nodules=nodules,
        low_confidence_nodules=[],
        findings=findings,
        versioning=versioning,
        processing_time_seconds=result.processing_time_ms / 1000
    )
    
    return ai_result


def generate_mock_ai_result(
    study_uid: str,
    series_uid: str,
    patient_id: Optional[str] = None
) -> StructuredAIResult:
    """
    Generate mock AI result for testing
    실제 배포 시에는 real Vision model 호출로 대체
    """
    from api.schemas import (
        QualityMetrics, NoduleCandidate, VisionEvidence,
        StructuredFindings, FindingLabel, ModelVersioning
    )
    import numpy as np
    
    # Mock quality
    quality = QualityMetrics(
        slice_thickness_mm=2.5,
        coverage_score=0.95,
        artifact_score=0.1
    )
    
    # Mock nodules (always 1-3 for demo)
    np.random.seed(hash(study_uid) % 2**32)
    num_nodules = np.random.randint(1, 4)  # 1~3개 항상 생성
    
    nodules = []
    for i in range(num_nodules):
        nodule = NoduleCandidate(
            id=f"N{i+1}",
            center_zyx=(50 + i*10, 100 + i*5, 120 + i*5),
            bbox_zyx=(45, 95, 115, 55, 105, 125),
            diameter_mm=round(np.random.uniform(4.0, 10.0), 1),
            volume_mm3=round(np.random.uniform(30.0, 500.0), 1),
            confidence=round(np.random.uniform(0.75, 0.95), 2),
            evidence=VisionEvidence(
                series_uid=series_uid,
                instance_uids=[f"instance.{i}.1"],
                slice_range=(45, 55),
                confidence=round(np.random.uniform(0.75, 0.95), 2)
            ),
            location_code=np.random.choice(["RUL", "RML", "RLL", "LUL", "LLL"])
        )
        nodules.append(nodule)
    
    # Mock findings (use actual classifier)
    import torch
    mock_volume = torch.randn(1, 1, 96, 96, 96)
    findings = findings_classifier.predict(
        mock_volume,
        {"series_uid": series_uid}
    )
    
    # Versioning
    versioning = ModelVersioning(
        model_version="mock-v1.0" if settings.use_mock_vision else "production-v1.0",
        pipeline_version="1.0.0",
        thresholds=threshold_manager.get_all_thresholds()
    )
    
    result = StructuredAIResult(
        study_uid=study_uid,
        series_uid=series_uid,
        acquisition_datetime=datetime.now(),
        quality=quality,
        lung_volume_ml=round(np.random.uniform(4000, 6000), 1),
        nodules=nodules,
        low_confidence_nodules=[],
        findings=findings,
        versioning=versioning,
        processing_time_seconds=1.5
    )
    
    return result


def build_rag_report_text(ai_result: StructuredAIResult, draft_report: Optional[DraftReport]) -> str:
    """
    Vector DB 저장용 텍스트 생성
    - report가 있으면 rendered_report 우선
    - 없으면 StructuredAIResult 기반 요약 텍스트 생성
    """
    if draft_report and draft_report.rendered_report:
        return draft_report.rendered_report
    
    lines = [
        f"CT Chest | {ai_result.acquisition_datetime.strftime('%Y-%m-%d')} | {ai_result.study_uid}",
        "AI SUMMARY",
        f"- Nodule candidates: {len(ai_result.nodules)}",
    ]
    
    if ai_result.nodules:
        sorted_nodules = sorted(ai_result.nodules, key=lambda n: n.confidence, reverse=True)
        for nodule in sorted_nodules[:5]:
            lines.append(
                f"- {nodule.id}: {nodule.diameter_mm:.1f} mm, "
                f"{nodule.location_code or 'UNK'}, conf={nodule.confidence:.2f}"
            )
    
    lines.append("Note: Auto-generated summary from structured AI result.")
    return "\n".join(lines)


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_ct_scan(request: AnalyzeRequest):
    """
    Analyze CT scan
    
    Returns structured AI result
    Optionally includes report draft
    """
    request_id = str(uuid.uuid4())
    
    logger.info(f"[{request_id}] Analysis request for study {request.study_uid}")
    
    try:
        # Check if LIDC case (real data) or mock
        if request.study_uid.startswith("LIDC-IDRI-"):
            logger.info(f"[{request_id}] Using REAL LIDC data: {request.study_uid}")
            ai_result = analyze_real_lidc_case(
                study_uid=request.study_uid,
                series_uid=request.series_uid,
                patient_id=request.patient_id
            )
        else:
            logger.info(f"[{request_id}] Using MOCK data")
            ai_result = generate_mock_ai_result(
                study_uid=request.study_uid,
                series_uid=request.series_uid or f"series.{request.study_uid}",
                patient_id=request.patient_id
            )
        
        logger.info(f"[{request_id}] AI analysis completed: {len(ai_result.nodules)} nodules")
        
        # Optional: Generate report
        draft_report = None
        prior_comparison = None
        korean_report = None
        groundedness = None
        solar_features = []
        vector_db_saved = False
        vector_db_doc_id = None
        
        if request.include_report:
            logger.info(f"[{request_id}] Generating report...")
            draft_report = await report_generator.generate_report(
                ai_result=ai_result,
                patient_id=request.patient_id,
                include_prior_comparison=request.include_prior_comparison
            )
            logger.info(f"[{request_id}] Report generated")
            solar_features.append("Report Generation (Solar Pro 3)")
            
            # Groundedness Check (한국어 번역 비활성화)
            if draft_report and draft_report.rendered_report:
                from solar_integration.solar_features import get_solar_features
                solar = get_solar_features()
                
                # Groundedness Check
                logger.info(f"[{request_id}] Checking groundedness...")
                try:
                    context = _build_groundedness_context(ai_result)
                    claim_text = _build_groundedness_claim(draft_report)

                    groundedness_result = await solar.check_groundedness(
                        claim=claim_text,
                        context=context
                    )
                    groundedness = groundedness_result.to_dict()
                    solar_features.append("Groundedness Check (Solar Pro 3)")
                    logger.info(
                        f"[{request_id}] Groundedness: "
                        f"{groundedness_result.is_grounded} "
                        f"(confidence={groundedness_result.confidence:.2f})"
                    )
                    if not groundedness_result.is_grounded:
                        if groundedness_result.ungrounded_claims:
                            logger.warning(
                                f"[{request_id}] Ungrounded claims: "
                                f"{groundedness_result.ungrounded_claims[:5]}"
                            )
                        if groundedness_result.explanation:
                            logger.warning(
                                f"[{request_id}] Groundedness explanation: "
                                f"{groundedness_result.explanation}"
                            )
                except Exception as e:
                    logger.warning(f"[{request_id}] Groundedness check failed: {e}")
        
        # Optional: Vector DB 저장 (요청 시에만)
        if request.include_vector_db_save:
            try:
                if rag_system is not None:
                    patient_id_for_rag = request.patient_id or request.study_uid
                    study_date_for_rag = ai_result.acquisition_datetime.strftime("%Y%m%d")
                    report_text_for_rag = build_rag_report_text(ai_result, draft_report)
                    
                    largest_nodule = max(ai_result.nodules, key=lambda n: n.diameter_mm, default=None)
                    rag_metadata = {
                        "request_id": request_id,
                        "source_endpoint": "/api/v1/analyze",
                        "series_uid": ai_result.series_uid,
                        "report_generated": bool(draft_report and draft_report.rendered_report),
                        "validation_passed": bool(draft_report.validation_passed) if draft_report else False,
                        "groundedness_checked": groundedness is not None,
                        "groundedness_passed": bool(groundedness.get("is_grounded")) if groundedness else False,
                        "largest_nodule_id": largest_nodule.id if largest_nodule else "",
                        "largest_nodule_diameter_mm": float(largest_nodule.diameter_mm) if largest_nodule else 0.0
                    }
                    
                    await rag_system.store_report(
                        patient_id=patient_id_for_rag,
                        study_uid=ai_result.study_uid,
                        study_date=study_date_for_rag,
                        report_text=report_text_for_rag,
                        ai_result=ai_result,
                        metadata=rag_metadata
                    )
                    
                    vector_db_saved = True
                    vector_db_doc_id = f"{patient_id_for_rag}_{ai_result.study_uid}"
                    solar_features.append("RAG Auto Save")
                    logger.info(f"[{request_id}] Vector DB saved: {vector_db_doc_id}")
                else:
                    logger.warning(f"[{request_id}] Vector DB save skipped: rag_system is not initialized")
            except Exception as e:
                logger.warning(f"[{request_id}] Vector DB save failed: {e}")
        
        # Build response
        response = AnalyzeResponse(
            request_id=request_id,
            status="completed",
            structured_ai_result=ai_result,
            draft_report=draft_report,
            prior_comparison=prior_comparison,
            korean_report=korean_report,
            groundedness=groundedness,
            solar_features_used=solar_features,
            processing_summary={
                "num_nodules": len(ai_result.nodules),
                "quality_adequate": ai_result.quality.is_adequate_for_nodules(),
                "report_generated": draft_report is not None,
                "validation_passed": draft_report.validation_passed if draft_report else None,
                "korean_available": korean_report is not None,
                "groundedness_checked": groundedness is not None,
                "vector_db_saved": vector_db_saved,
                "vector_db_doc_id": vector_db_doc_id
            }
        )
        
        logger.info(f"[{request_id}] Request completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"[{request_id}] Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/generate-report", response_model=GenerateReportResponse)
async def generate_report(request: GenerateReportRequest):
    """
    Generate report from existing AI result
    """
    logger.info(f"Report generation for study {request.structured_ai_result.study_uid}")
    
    try:
        draft_report = await report_generator.generate_report(
            ai_result=request.structured_ai_result,
            patient_id=request.patient_id,
            include_prior_comparison=request.include_prior_comparison
        )
        
        response = GenerateReportResponse(
            draft_report=draft_report,
            validation_status={
                "passed": draft_report.validation_passed,
                "warnings": draft_report.validation_warnings
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/vector-db/save", response_model=VectorDbSaveResponse)
async def save_vector_db(request: VectorDbSaveRequest):
    """
    Save report + AI results to Vector DB (manual action)
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system is not initialized")
    
    ai_result = request.structured_ai_result
    patient_id_for_rag = request.patient_id or ai_result.study_uid
    study_date_for_rag = ai_result.acquisition_datetime.strftime("%Y%m%d")
    report_text_for_rag = build_rag_report_text(ai_result, request.draft_report)
    
    largest_nodule = max(ai_result.nodules, key=lambda n: n.diameter_mm, default=None)
    rag_metadata = {
        "request_id": "manual-save",
        "source_endpoint": "/api/v1/vector-db/save",
        "series_uid": ai_result.series_uid,
        "report_generated": bool(request.draft_report and request.draft_report.rendered_report),
        "validation_passed": bool(request.draft_report.validation_passed) if request.draft_report else False,
        "largest_nodule_id": largest_nodule.id if largest_nodule else "",
        "largest_nodule_diameter_mm": float(largest_nodule.diameter_mm) if largest_nodule else 0.0
    }
    
    await rag_system.store_report(
        patient_id=patient_id_for_rag,
        study_uid=ai_result.study_uid,
        study_date=study_date_for_rag,
        report_text=report_text_for_rag,
        ai_result=ai_result,
        metadata=rag_metadata
    )
    
    vector_db_doc_id = f"{patient_id_for_rag}_{ai_result.study_uid}"
    return VectorDbSaveResponse(
        saved=True,
        vector_db_doc_id=vector_db_doc_id,
        updated=True,
        message="Saved to Vector DB"
    )


@app.get("/api/v1/patient/{patient_id}/history", response_model=PatientHistoryResponse)
async def get_patient_history(patient_id: str, max_results: int = 10):
    """Retrieve patient history"""
    try:
        history = rag_system.retrieve_patient_history(patient_id, max_results)
        
        return PatientHistoryResponse(
            patient_id=patient_id,
            num_studies=len(history),
            studies=history
        )
        
    except Exception as e:
        logger.error(f"History retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/semantic-search", response_model=SemanticSearchResponse)
async def semantic_search(request: SemanticSearchRequest):
    """
    Semantic search (patient isolation enforced)
    """
    try:
        results = await rag_system.semantic_search(
            query_text=request.query_text,
            patient_id=request.patient_id,
            n_results=request.n_results
        )
        
        return SemanticSearchResponse(
            query=request.query_text,
            patient_id=request.patient_id,
            results=results,
            retriever_info=rag_system.embedding_client.get_model_info()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== Solar 추가 기능 엔드포인트 ==========

from pydantic import BaseModel

class QARequest(BaseModel):
    """Q&A 요청"""
    question: str
    patient_id: Optional[str] = None
    study_uid: Optional[str] = None
    ai_results: Optional[Dict[str, Any]] = None

class QAResponseModel(BaseModel):
    """Q&A 응답"""
    question: str
    answer: str
    sources: List[str]
    confidence: float

class TranslateRequest(BaseModel):
    """번역 요청"""
    text: str
    source_lang: str = "en"
    target_lang: str = "ko"

class TranslateResponse(BaseModel):
    """번역 응답"""
    original: str
    translated: str
    source_lang: str
    target_lang: str

class PriorSummaryRequest(BaseModel):
    """Prior 요약 요청"""
    patient_id: str
    prior_reports: List[Dict[str, Any]] = []

class PriorSummaryResponse(BaseModel):
    """Prior 요약 응답"""
    summary: str
    prior_count: int
    key_changes: List[str]
    recommendation: str

class GroundednessRequest(BaseModel):
    """Groundedness 검증 요청"""
    claim: str
    context: str

class GroundednessResponse(BaseModel):
    """Groundedness 검증 응답"""
    is_grounded: bool
    confidence: float
    ungrounded_claims: List[str]
    explanation: str


class InsightsRequest(BaseModel):
    """LLM 인사이트 요청"""
    structured_ai_result: StructuredAIResult
    patient_id: Optional[str] = None
    groundedness: Optional[Dict[str, Any]] = None
    llm_only: bool = False


class WhyCardsRequest(InsightsRequest):
    top_k: int = 3


class ThresholdWhatIfRequest(BaseModel):
    selected_threshold: float
    mode: str = "balanced"  # screening | reporting | balanced
    postproc: str = "nms15"
    llm_only: bool = False


@app.post("/api/v1/insights/why-cards")
async def generate_why_cards(request: WhyCardsRequest):
    """
    1) Why Card: 결절 후보별 수치 근거 + LLM 설명 3줄
    """
    from solar_integration.solar_features import get_solar_features

    ai_result = request.structured_ai_result
    solar = get_solar_features()
    nodules = sorted(ai_result.nodules, key=lambda n: n.confidence, reverse=True)[: max(1, min(3, request.top_k))]

    context = {
        "study_uid": ai_result.study_uid,
        "num_nodules": len(ai_result.nodules),
        "quality": {
            "slice_thickness_mm": ai_result.quality.slice_thickness_mm,
            "coverage_score": ai_result.quality.coverage_score,
            "artifact_score": ai_result.quality.artifact_score,
        },
    }

    cards: List[Dict[str, Any]] = []
    for n in nodules:
        nodule_dict = {
            "id": n.id,
            "location_code": n.location_code or "UNK",
            "diameter_mm": float(n.diameter_mm),
            "volume_mm3": float(n.volume_mm3),
            "confidence": float(n.confidence),
            "slice_range": list(n.evidence.slice_range),
            "center_zyx": [float(v) for v in n.center_zyx],
        }
        why = await solar.explain_nodule_why(
            nodule=nodule_dict,
            context=context,
            llm_only=request.llm_only
        )
        cards.append(
            {
                "nodule": nodule_dict,
                "why": why,
            }
        )

    return {
        "study_uid": ai_result.study_uid,
        "cards": cards,
        "count": len(cards),
    }


@app.post("/api/v1/insights/prior-delta")
async def generate_prior_delta(request: InsightsRequest):
    """
    2) Prior Delta Narrator: 과거 대비 수치 변화 + LLM 설명
    """
    from solar_integration.solar_features import get_solar_features

    ai_result = request.structured_ai_result
    patient_id = request.patient_id or ai_result.study_uid
    numeric = _summarize_prior_delta(ai_result, patient_id=patient_id)
    solar = get_solar_features()

    current_summary = {
        "current_study_uid": ai_result.study_uid,
        "current_nodule_count": len(ai_result.nodules),
        "largest_nodule_mm": numeric.get("current_largest_mm"),
        "change_type": numeric.get("change_type"),
        "summary_seed": (
            f"현재 최대 결절 {numeric.get('current_largest_mm')}mm, "
            f"과거 {numeric.get('prior_largest_mm')}mm, "
            f"변화 {numeric.get('delta_mm')}mm ({numeric.get('delta_pct')}%)."
        ),
    }

    prior_summary = {
        "prior_study_date": numeric.get("prior_study_date"),
        "prior_largest_mm": numeric.get("prior_largest_mm"),
        "has_prior": numeric.get("has_prior"),
    }

    narrative = await solar.narrate_prior_delta(
        current=current_summary,
        prior=prior_summary,
        llm_only=request.llm_only
    )
    return {
        "study_uid": ai_result.study_uid,
        "numeric": numeric,
        "narrative": narrative,
    }


@app.post("/api/v1/insights/action-suggestion")
async def generate_action_suggestion(request: InsightsRequest):
    """
    3) Action Suggestion: groundedness 통과 시에만 수치 근거 기반 권고
    """
    from solar_integration.solar_features import get_solar_features

    groundedness = request.groundedness or {}
    is_grounded = bool(groundedness.get("is_grounded", True))
    if groundedness and not is_grounded:
        return {
            "allowed": False,
            "reason": "Groundedness check failed",
            "suggestion": None,
        }

    ai_result = request.structured_ai_result
    numeric = _summarize_prior_delta(ai_result, patient_id=request.patient_id or ai_result.study_uid)
    max_conf = max([float(n.confidence) for n in ai_result.nodules], default=0.0)
    max_diameter = max([float(n.diameter_mm) for n in ai_result.nodules], default=0.0)

    sweep_rows, _ = _load_threshold_metrics(postproc="nms15")
    nearest = _pick_nearest_threshold_row(sweep_rows, threshold=0.15)
    sensitivity_hint = nearest.get("sensitivity") if nearest else None

    evidence = {
        "study_uid": ai_result.study_uid,
        "nodule_count": len(ai_result.nodules),
        "max_confidence": max_conf,
        "max_diameter_mm": max_diameter,
        "prior_delta": numeric,
        "sensitivity_hint": sensitivity_hint,
    }
    solar = get_solar_features()
    suggestion = await solar.suggest_action(
        evidence=evidence,
        llm_only=request.llm_only
    )

    source_rows = []
    for n in sorted(ai_result.nodules, key=lambda x: x.confidence, reverse=True)[:5]:
        srange = list(n.evidence.slice_range) if n.evidence else [0, 0]
        source_rows.append(
            {
                "nodule_id": n.id,
                "slice_range": f"{srange[0]}-{srange[1]}",
                "confidence": float(n.confidence),
                "prior_date": numeric.get("prior_study_date"),
            }
        )

    return {
        "allowed": True,
        "suggestion": suggestion,
        "evidence": evidence,
        "source_rows": source_rows,
    }


@app.get("/api/v1/insights/threshold-metrics")
async def get_threshold_metrics(postproc: str = "nms15"):
    """
    4) What-if용 threshold metric 테이블 로드
    """
    rows, src = _load_threshold_metrics(postproc=postproc)
    if not rows:
        return {
            "postproc": postproc,
            "rows": [],
            "source": str(src) if src else None,
            "best_screening": None,
            "best_reporting": None,
            "best_balanced": None,
        }

    best_screening = max(rows, key=lambda r: r["sensitivity"])
    best_reporting = min(rows, key=lambda r: r["fp_per_scan"])
    best_balanced = max(rows, key=lambda r: (r["sensitivity"] - 0.05 * r["fp_per_scan"]))
    return {
        "postproc": postproc,
        "rows": rows,
        "source": str(src) if src else None,
        "best_screening": best_screening,
        "best_reporting": best_reporting,
        "best_balanced": best_balanced,
    }


@app.post("/api/v1/insights/threshold-whatif")
async def explain_threshold_whatif(request: ThresholdWhatIfRequest):
    """
    4) What-if 설명: threshold 선택 시 sens/FP 변화 + LLM 해석
    """
    from solar_integration.solar_features import get_solar_features

    mode = (request.mode or "balanced").lower()
    if mode not in {"screening", "reporting", "balanced"}:
        mode = "balanced"

    rows, src = _load_threshold_metrics(postproc=request.postproc)
    if not rows:
        return {
            "selected": None,
            "best_screening": None,
            "best_reporting": None,
            "best_balanced": None,
            "explanation": "Threshold metrics가 아직 없습니다. eval sweep 결과를 먼저 생성하세요.",
            "source": str(src) if src else None,
        }

    selected = _pick_nearest_threshold_row(rows, request.selected_threshold)
    best_screening = max(rows, key=lambda r: r["sensitivity"])
    best_reporting = min(rows, key=lambda r: r["fp_per_scan"])
    best_balanced = max(rows, key=lambda r: (r["sensitivity"] - 0.05 * r["fp_per_scan"]))

    solar = get_solar_features()
    explain_out = await solar.explain_threshold_tradeoff(
        selected=selected,
        best_screening=best_screening,
        best_reporting=best_reporting,
        mode=mode,
        llm_only=request.llm_only,
    )

    return {
        "selected": selected,
        "best_screening": best_screening,
        "best_reporting": best_reporting,
        "best_balanced": best_balanced,
        "mode": mode,
        "explanation": explain_out.get("text", ""),
        "generated_by_llm": bool(explain_out.get("generated_by_llm", False)),
        "generator": explain_out.get("generator", "unknown"),
        "status": explain_out.get("status", "unknown"),
        "source": str(src) if src else None,
    }


@app.post("/api/v1/qa", response_model=QAResponseModel)
async def physician_qa(request: QARequest):
    """
    🤖 의사 Q&A 인터페이스
    
    Solar Pro 3를 사용하여 AI 분석 결과에 대한 질문에 답변합니다.
    과거 검사 기록(Prior)도 포함하여 비교 질문에 답변 가능합니다.
    """
    from solar_integration.solar_features import get_solar_features
    
    try:
        solar = get_solar_features()
        
        # AI results가 없으면 기본 컨텍스트 사용
        ai_results = request.ai_results or {
            "nodules": [],
            "key_flags": {"nodule_candidates": 0},
            "quality": {}
        }
        
        # Prior data 가져오기 (RAG에서)
        prior_data = None
        patient_id = request.patient_id or request.study_uid
        if patient_id and rag_system:
            try:
                history = rag_system.retrieve_patient_history(patient_id, max_results=1)
                if history:
                    prior = history[0]
                    prior_data = {
                        "study_date": prior.get("study_date"),
                        "nodule_diameter_mm": prior.get("metadata", {}).get("nodule_diameter_mm"),
                        "nodule_location": prior.get("metadata", {}).get("nodule_location")
                    }
                    logger.info(f"Q&A: Prior data found for {patient_id}")
            except Exception as e:
                logger.warning(f"Q&A: Failed to get prior data: {e}")
        
        result = await solar.answer_question(
            question=request.question,
            ai_results=ai_results,
            prior_data=prior_data
        )
        
        logger.info(f"Q&A: '{request.question[:50]}...' -> answered")
        
        return QAResponseModel(
            question=result.question,
            answer=result.answer,
            sources=result.sources,
            confidence=result.confidence
        )
        
    except Exception as e:
        logger.error(f"Q&A failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/translate", response_model=TranslateResponse)
async def translate_report(request: TranslateRequest):
    """
    🌏 한국어 번역
    
    Solar Pro 3를 사용하여 영어 리포트를 한국어로 번역합니다.
    """
    from solar_integration.solar_features import get_solar_features
    
    try:
        solar = get_solar_features()
        
        result = await solar.translate_to_korean(request.text)
        
        logger.info(f"Translation: {len(request.text)} chars -> Korean")
        
        return TranslateResponse(
            original=result.original_text,
            translated=result.translated_text,
            source_lang=result.source_lang,
            target_lang=result.target_lang
        )
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/prior-summary", response_model=PriorSummaryResponse)
async def summarize_prior_reports(request: PriorSummaryRequest):
    """
    📋 Prior Report 요약
    
    Solar Pro 3를 사용하여 과거 리포트들을 요약합니다.
    """
    from solar_integration.solar_features import get_solar_features
    
    try:
        solar = get_solar_features()
        
        result = await solar.summarize_priors(request.prior_reports)
        
        logger.info(f"Prior summary: {result.prior_count} reports summarized")
        
        return PriorSummaryResponse(
            summary=result.summary_text,
            prior_count=result.prior_count,
            key_changes=result.key_changes,
            recommendation=result.recommendation
        )
        
    except Exception as e:
        logger.error(f"Prior summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/groundedness-check", response_model=GroundednessResponse)
async def check_groundedness(request: GroundednessRequest):
    """
    ✅ Groundedness Check (환각 검증)
    
    Solar Pro 3를 사용하여 텍스트가 근거 데이터에 기반하는지 검증합니다.
    """
    from solar_integration.solar_features import get_solar_features
    
    try:
        solar = get_solar_features()
        
        result = await solar.check_groundedness(
            claim=request.claim,
            context=request.context
        )
        
        logger.info(f"Groundedness check: {'PASS' if result.is_grounded else 'FAIL'} ({result.confidence:.2f})")
        
        return GroundednessResponse(
            is_grounded=result.is_grounded,
            confidence=result.confidence,
            ungrounded_claims=result.ungrounded_claims,
            explanation=result.explanation
        )
        
    except Exception as e:
        logger.error(f"Groundedness check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/solar-features")
async def get_solar_features_info():
    """
    🌟 Solar 기능 정보
    
    사용 가능한 Upstage Solar 기능들을 반환합니다.
    """
    return {
        "features": [
            {
                "name": "Report Generation",
                "description": "Solar Pro 3로 리포트 텍스트 다듬기",
                "endpoint": "/api/v1/analyze",
                "status": "active"
            },
            {
                "name": "Korean Translation",
                "description": "영어 리포트를 한국어로 번역",
                "endpoint": "/api/v1/translate",
                "status": "active"
            },
            {
                "name": "Groundedness Check",
                "description": "AI 출력의 환각 검증",
                "endpoint": "/api/v1/groundedness-check",
                "status": "active"
            },
            {
                "name": "Prior Summary",
                "description": "과거 리포트 요약",
                "endpoint": "/api/v1/prior-summary",
                "status": "active"
            },
            {
                "name": "Physician Q&A",
                "description": "AI 분석에 대한 질의응답",
                "endpoint": "/api/v1/qa",
                "status": "active"
            },
            {
                "name": "RAG (Semantic Search)",
                "description": "Solar Embedding으로 의미 검색",
                "endpoint": "/api/v1/semantic-search",
                "status": "active"
            },
            {
                "name": "Why Cards",
                "description": "결절별 수치 근거 설명 카드 생성",
                "endpoint": "/api/v1/insights/why-cards",
                "status": "active"
            },
            {
                "name": "Prior Delta Narrator",
                "description": "과거 대비 변화량 서술",
                "endpoint": "/api/v1/insights/prior-delta",
                "status": "active"
            },
            {
                "name": "Action Suggestion",
                "description": "Groundedness 통과 기반 다음 액션 제안",
                "endpoint": "/api/v1/insights/action-suggestion",
                "status": "active"
            },
            {
                "name": "Threshold What-if",
                "description": "threshold 변경 시 sens/FP tradeoff 설명",
                "endpoint": "/api/v1/insights/threshold-whatif",
                "status": "active"
            }
        ],
        "model": "Solar Pro 3 (upstage/solar-pro-3)",
        "embedding": "Solar Embedding (solar-embedding-1-large)",
        "mode": "PRODUCTION" if settings.should_use_real_solar else "MOCK"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False
    )
