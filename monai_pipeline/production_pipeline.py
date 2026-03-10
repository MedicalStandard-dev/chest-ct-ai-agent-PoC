# monai_pipeline/production_pipeline.py
"""
Production Pipeline: Heatmap → 제품 출력 전체 파이프라인

전체 흐름:
1. Model Inference (Heatmap 생성)
2. Peak → Candidate 생성
3. Component 분리 + Measurements
4. Evidence 생성
5. Prior Tracking (있는 경우)
6. Table 데이터 생성
7. Report Generator로 전달

핵심:
- heatmap 모델은 '후보 생성기'
- 제품은 후보를 '증거·표·비교·로그'로 변환
"""
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

from utils.logger import logger


@dataclass
class PipelineResult:
    """파이프라인 최종 결과"""
    # 기본 정보
    series_uid: str
    patient_id: Optional[str] = None
    study_date: Optional[str] = None
    
    # 후보 결과
    total_candidates: int = 0
    findings_count: int = 0
    limitations_count: int = 0
    
    # 테이블 데이터
    findings_table: List[Dict] = field(default_factory=list)
    measurements_table: List[Dict] = field(default_factory=list)
    prior_comparison_table: List[Dict] = field(default_factory=list)
    
    # Key Flags
    key_flags: Dict = field(default_factory=dict)
    
    # Evidence
    evidences: List[Dict] = field(default_factory=list)
    
    # Quality/Limitations
    quality_info: Dict = field(default_factory=dict)
    
    # 처리 시간
    processing_time_ms: float = 0.0
    
    # 원본 데이터 (내부용)
    candidates: List[Any] = field(default_factory=list)
    tracking_matches: List[Any] = field(default_factory=list)
    _structured_findings: Any = field(default=None, repr=False)
    
    def to_structured_result(self) -> Dict:
        """StructuredAIResult 형식으로 변환"""
        return {
            "study_uid": self.series_uid,
            "series_uid": self.series_uid,
            "patient_id": self.patient_id,
            "acquisition_datetime": self.study_date,
            "nodules": self._candidates_to_nodules(),
            "findings": self._create_findings_summary(),
            "quality": self.quality_info,
            "tables": {
                "findings": self.findings_table,
                "measurements": self.measurements_table,
                "prior_comparison": self.prior_comparison_table
            },
            "key_flags": self.key_flags,
            "evidences": self.evidences,
            "processing_time_ms": self.processing_time_ms
        }
    
    def _candidates_to_nodules(self) -> List[Dict]:
        """Candidate를 NoduleCandidate 형식으로 변환"""
        nodules = []
        for c in self.candidates:
            if hasattr(c, '_raw') and hasattr(c._raw, 'model_dump'):
                nodules.append(c._raw.model_dump())
            elif hasattr(c, 'to_dict'):
                nodules.append(c.to_dict())
            elif isinstance(c, dict):
                nodules.append(c)
        return nodules
    
    def _create_findings_summary(self) -> Dict:
        """Findings 요약 생성 (rule-based classifier 사용)"""
        base = {
            "nodule_candidates": self.findings_count,
        }
        if hasattr(self, '_structured_findings') and self._structured_findings is not None:
            sf = self._structured_findings
            for name in ["pleural_effusion", "pneumothorax", "consolidation",
                         "atelectasis", "emphysema"]:
                fl = getattr(sf, name)
                base[name] = {"label": fl.label, "probability": float(fl.probability)}
        else:
            for name in ["pleural_effusion", "pneumothorax", "consolidation"]:
                base[name] = {"label": "absent", "probability": 0.0}
        return base


class ProductionPipeline:
    """
    제품형 파이프라인
    
    Nodule Detection 모델 + 후처리 + 테이블 생성
    """
    
    def __init__(
        self,
        nodule_model_path: Optional[Path] = None,
        lung_seg_model_path: Optional[Path] = None,
        device: str = "cuda",
        output_dir: Optional[Path] = None,
        nodule_detection_threshold: float = 0.15,
        nodule_max_diameter_mm: float = 50.0,
        nodule_max_volume_mm3: float = 65000.0,
        nodule_restrict_to_lung_mask: bool = True,
        nodule_min_lung_overlap_ratio: float = 0.3,
        luna16_bundle_dir: Optional[str] = None,
    ):
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else None

        # 모델 로드 (lazy)
        self._nodule_inference = None
        self._lung_seg_inference = None
        self._luna16_detector = None
        self.nodule_model_path = nodule_model_path
        self.lung_seg_model_path = lung_seg_model_path
        self.luna16_bundle_dir = luna16_bundle_dir
        self.nodule_detection_threshold = nodule_detection_threshold
        self.nodule_max_diameter_mm = nodule_max_diameter_mm
        self.nodule_max_volume_mm3 = nodule_max_volume_mm3
        self.nodule_restrict_to_lung_mask = nodule_restrict_to_lung_mask
        self.nodule_min_lung_overlap_ratio = nodule_min_lung_overlap_ratio
        
        # 프로세서 초기화
        from monai_pipeline.candidate_processor import CandidateProcessor, ThresholdPolicy
        from monai_pipeline.evidence_generator import EvidenceGenerator
        from monai_pipeline.tracking_engine import LesionTrackingEngine
        
        self.candidate_processor = CandidateProcessor(
            policy=ThresholdPolicy(),
            output_dir=self.output_dir
        )
        
        self.evidence_generator = EvidenceGenerator(
            output_dir=self.output_dir / "evidence" if self.output_dir else None,
            save_masks=True
        )
        
        self.tracking_engine = LesionTrackingEngine()
        
        logger.info("ProductionPipeline initialized")
    
    @property
    def nodule_inference(self):
        """Lazy load nodule detection model"""
        if self._nodule_inference is None and self.nodule_model_path:
            from monai_pipeline.nodule_detection import NoduleDetectionInference
            self._nodule_inference = NoduleDetectionInference(
                model_path=self.nodule_model_path,
                model_type="unet",
                device=self.device,
                roi_size=(128, 128, 128),
                detection_threshold=self.nodule_detection_threshold,
                max_diameter_mm=self.nodule_max_diameter_mm,
                max_volume_mm3=self.nodule_max_volume_mm3,
                restrict_to_lung_mask=self.nodule_restrict_to_lung_mask,
                min_lung_overlap_ratio=self.nodule_min_lung_overlap_ratio,
                lung_seg_model_path=self.lung_seg_model_path,
            )
        return self._nodule_inference
    
    @property
    def luna16_detector(self):
        """Lazy load MONAI LUNA16 pretrained RetinaNet"""
        if self._luna16_detector is None and self.luna16_bundle_dir:
            try:
                from monai_pipeline.luna16_detector import Luna16Detector
                self._luna16_detector = Luna16Detector(
                    bundle_dir=self.luna16_bundle_dir,
                    device=self.device,
                    score_thresh=0.02,
                    amp=True,
                )
                logger.info("Luna16Detector loaded (MONAI pretrained RetinaNet)")
            except Exception as e:
                logger.warning(f"Failed to load Luna16Detector: {e}")
                self._luna16_detector = None
        return self._luna16_detector

    @property
    def lung_seg_inference(self):
        """Lazy load lung segmentation model"""
        if self._lung_seg_inference is None and self.lung_seg_model_path:
            try:
                from monai_pipeline.lung_segmentation import LungSegmentationInference
                self._lung_seg_inference = LungSegmentationInference(
                    model_path=self.lung_seg_model_path
                )
                logger.info("Lung segmentation model loaded")
            except Exception as e:
                logger.warning(f"Failed to load lung segmentation model: {e}")
                self._lung_seg_inference = None
        return self._lung_seg_inference
    
    def process_volume(
        self,
        volume: np.ndarray,
        spacing_mm: Tuple[float, float, float],
        series_uid: str,
        patient_id: Optional[str] = None,
        study_date: Optional[str] = None,
        prior_lesions: Optional[List[Dict]] = None,
        lung_mask: Optional[np.ndarray] = None,
        quality_info: Optional[Dict] = None
    ) -> PipelineResult:
        """
        전체 파이프라인 실행
        
        Args:
            volume: (D, H, W) CT volume (HU normalized)
            spacing_mm: voxel spacing
            series_uid: DICOM series UID
            patient_id: Patient ID
            study_date: Study date (YYYY-MM-DD)
            prior_lesions: 이전 검사 병변 리스트
            lung_mask: Optional lung segmentation mask
            quality_info: Quality assessment info
            
        Returns:
            PipelineResult
        """
        import time
        start_time = time.time()
        
        logger.info(f"Processing volume: {volume.shape}, spacing={spacing_mm}")
        
        result = PipelineResult(
            series_uid=series_uid,
            patient_id=patient_id,
            study_date=study_date or datetime.now().strftime("%Y-%m-%d"),
            quality_info=quality_info or {}
        )
        
        # 1. Lung mask (heuristic)
        if lung_mask is None:
            try:
                lung_mask = self._generate_lung_mask(volume)
                logger.info(f"Lung mask generated (heuristic): {lung_mask.sum()} voxels")
            except Exception as e:
                logger.warning(f"Lung mask generation failed: {e}")
                lung_mask = None

        # 2. Candidate 추출: Luna16Detector 우선, fallback → heatmap
        if self.luna16_detector is not None:
            raw_candidates = self.luna16_detector.detect(
                volume_zyx=volume,
                spacing_mm=spacing_mm,
                series_uid=series_uid,
                lung_mask=lung_mask,
            )
            logger.info(f"Extracted {len(raw_candidates)} candidates via Luna16Detector (MONAI pretrained)")
        else:
            heatmap = self._generate_heatmap(volume)
            logger.info(f"Heatmap generated: shape={heatmap.shape}, max={heatmap.max():.3f}")
            raw_candidates = self.nodule_inference.extract_candidates(
                heatmap=heatmap,
                spacing_mm=spacing_mm,
                series_uid=series_uid,
                lung_mask=lung_mask,
            )
            logger.info(f"Extracted {len(raw_candidates)} candidates via extract_candidates")

        # NoduleCandidate(pydantic) → 하류 호환 SimpleNamespace 래핑
        candidates = []
        for c in raw_candidates:
            from types import SimpleNamespace
            w = SimpleNamespace(
                candidate_id=c.id,
                status="finding" if c.confidence >= 0.25 else "limitation",
                peak_zyx=c.center_zyx,
                confidence=c.confidence,
                diameter_mm=c.diameter_mm,
                volume_mm3=c.volume_mm3,
                slice_range=c.evidence.slice_range if c.evidence else (0, 0),
                location_code=c.location_code or "UNK",
                center_zyx=c.center_zyx,
                bbox_zyx=c.bbox_zyx,
                evidence=c.evidence,
                _raw=c,
            )
            candidates.append(w)

        result.candidates = candidates
        result.total_candidates = len(candidates)
        result.findings_count = len([c for c in candidates if c.status == "finding"])
        result.limitations_count = len([c for c in candidates if c.status == "limitation"])

        # 3. Evidence (NoduleCandidate에 이미 포함)
        result.evidences = [
            c.evidence.model_dump() if hasattr(c.evidence, 'model_dump') else {}
            for c in candidates if c.evidence
        ]
        
        # 4. Prior Tracking (있는 경우)
        if prior_lesions:
            result.tracking_matches = self._perform_tracking(candidates, prior_lesions)
            result.prior_comparison_table = self._build_prior_table(result.tracking_matches)
        
        # 5. 테이블 생성
        result.findings_table = self._build_findings_table(candidates)
        result.measurements_table = self._build_measurements_table(candidates)
        
        # 6. Key Flags 생성
        result.key_flags = self._build_key_flags(candidates, result.tracking_matches)

        # 7. Findings classification (rule-based)
        try:
            from monai_pipeline.findings_classifier import RuleBasedFindingsClassifier
            _findings_clf = RuleBasedFindingsClassifier()
            result._structured_findings = _findings_clf.predict(
                volume=torch.empty(0),
                metadata={"series_uid": series_uid},
                volume_hu=volume,
                spacing=spacing_mm,
                lung_mask=lung_mask,
            )
            logger.info(f"Findings classification done: {_findings_clf.get_version()}")
        except Exception as e:
            logger.warning(f"Findings classification failed: {e}")
            result._structured_findings = None

        # 처리 시간
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Pipeline complete: {result.findings_count} findings, "
                   f"{result.limitations_count} limitations, "
                   f"{result.processing_time_ms:.0f}ms")
        
        return result
    
    def _generate_heatmap(self, volume: np.ndarray) -> np.ndarray:
        """Heatmap 생성 (모델 추론 또는 mock)"""
        if self.nodule_inference is None:
            # Mock: 랜덤 heatmap (테스트용)
            logger.warning("No nodule model loaded, using mock heatmap")
            heatmap = np.random.rand(*volume.shape) * 0.3
            # Add some fake peaks
            for _ in range(np.random.randint(1, 5)):
                z = np.random.randint(volume.shape[0] // 4, 3 * volume.shape[0] // 4)
                y = np.random.randint(volume.shape[1] // 4, 3 * volume.shape[1] // 4)
                x = np.random.randint(volume.shape[2] // 4, 3 * volume.shape[2] // 4)
                heatmap[max(0,z-5):z+5, max(0,y-5):y+5, max(0,x-5):x+5] = np.random.uniform(0.5, 0.95)
            return heatmap
        
        # Real inference
        volume_norm = self._normalize_ct_volume(volume)
        volume_tensor = torch.from_numpy(volume_norm).unsqueeze(0).unsqueeze(0).float()
        heatmap = self.nodule_inference.predict_heatmap(volume_tensor)
        return heatmap

    def _normalize_ct_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalize CT volume to [0,1] consistently.
        - If already [0,1], clamp only.
        - Else apply HU clip(-1000, 400) and scale.
        """
        vol = np.asarray(volume, dtype=np.float32)
        v_min = float(np.min(vol))
        v_max = float(np.max(vol))
        if v_min >= -1e-3 and v_max <= 1.0 + 1e-3:
            return np.clip(vol, 0.0, 1.0)
        vol = np.clip(vol, -1000.0, 400.0)
        return (vol + 1000.0) / 1400.0
    
    def _generate_lung_mask(self, volume: np.ndarray) -> np.ndarray:
        """Lung mask 생성 (heuristic 방식 — 학습 시와 동일)"""
        from scipy.ndimage import binary_fill_holes
        from scipy.ndimage import label as scipy_label

        vol = self._normalize_ct_volume(volume)
        body_mask = vol > 0.03
        lung_like = vol < 0.65
        mask = np.logical_and(body_mask, lung_like)
        mask = binary_fill_holes(mask)

        labeled, n = scipy_label(mask.astype(np.uint8))
        if n > 0:
            comp_sizes = np.bincount(labeled.ravel())
            comp_sizes[0] = 0
            keep = np.argsort(comp_sizes)[-4:]
            keep = set(int(k) for k in keep if k > 0 and comp_sizes[k] >= 128)
            if keep:
                mask = np.isin(labeled, list(keep))
        return mask.astype(np.uint8)
    
    def _perform_tracking(
        self,
        candidates: List,
        prior_lesions: List[Dict]
    ) -> List:
        """Prior 매칭 수행"""
        from monai_pipeline.tracking_engine import PriorLesion
        
        # Convert prior_lesions to PriorLesion objects
        priors = []
        for pl in prior_lesions:
            priors.append(PriorLesion(
                lesion_id=pl.get("lesion_id", "unknown"),
                center_mm=tuple(pl.get("center_mm", (0, 0, 0))),
                diameter_mm=pl.get("diameter_mm", 0.0),
                volume_mm3=pl.get("volume_mm3", 0.0),
                study_date=pl.get("study_date", "unknown")
            ))
        
        # Convert candidates to dict format
        current = []
        for c in candidates:
            if c.status != "hidden":
                current.append({
                    "id": c.candidate_id,
                    "center_mm": c.center_zyx,
                    "diameter_mm": c.diameter_mm,
                    "volume_mm3": c.volume_mm3
                })
        
        matches = self.tracking_engine.track(current, priors)
        return matches
    
    def _build_findings_table(self, candidates: List) -> List[Dict]:
        """FINDINGS TABLE 생성"""
        rows = []
        
        for c in candidates:
            if c.status == "hidden":
                continue
            
            rows.append({
                "type": "Nodule candidate",
                "location": c.location_code,
                "status": "Present" if c.status == "finding" else "Low confidence",
                "confidence": round(c.confidence, 2),
                "evidence_id": c.candidate_id
            })
        
        return rows
    
    def _build_measurements_table(self, candidates: List) -> List[Dict]:
        """MEASUREMENTS TABLE 생성"""
        rows = []
        
        for c in candidates:
            if c.status == "hidden":
                continue
            
            rows.append({
                "lesion_id": c.candidate_id,
                "location": c.location_code,
                "diameter_mm": round(c.diameter_mm, 1),
                "volume_mm3": round(c.volume_mm3, 1),
                "confidence": round(c.confidence, 2),
                "evidence_id": c.candidate_id
            })
        
        return rows
    
    def _build_prior_table(self, matches: List) -> List[Dict]:
        """PRIOR COMPARISON TABLE 생성"""
        rows = []
        
        for m in matches:
            if hasattr(m, 'to_table_row'):
                rows.append(m.to_table_row())
        
        return rows
    
    def _build_key_flags(self, candidates: List, matches: List) -> Dict:
        """KEY FLAGS 생성"""
        findings = [c for c in candidates if c.status == "finding"]
        limitations = [c for c in candidates if c.status == "limitation"]
        
        new_count = sum(1 for m in matches if hasattr(m, 'change_type') and m.change_type.value == "NEW")
        
        high_conf_thresh = 0.35
        
        return {
            "nodule_candidates": len(findings) + len(limitations),
            "new_nodules": new_count,
            "high_confidence_findings": len([c for c in findings if c.confidence >= high_conf_thresh]),
            "has_limitations": len(limitations) > 0
        }


def create_pipeline(
    nodule_model_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> ProductionPipeline:
    """파이프라인 생성 헬퍼"""
    return ProductionPipeline(
        nodule_model_path=Path(nodule_model_path) if nodule_model_path else None,
        output_dir=Path(output_dir) if output_dir else None
    )
