# monai_pipeline/evidence_generator.py
"""
Evidence Generator: 데모 임팩트 핵심

"클릭 → slice로 점프" 기능의 핵심

생성 항목:
- series_uid: DICOM 시리즈 ID
- instance_uids: 해당 slice들의 SOP Instance UID
- slice_range: 결절이 존재하는 z 범위
- overlay: viewer용 마스크/윤곽선
- thumbnail: 대표 slice 이미지
"""
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import json

from utils.logger import logger


@dataclass
class VisionEvidenceData:
    """Vision Evidence 데이터 (api/schemas.py의 VisionEvidence와 호환)"""
    evidence_id: str
    series_uid: str
    
    # Slice 정보
    slice_range: Tuple[int, int]  # (start_z, end_z)
    center_slice: int  # 대표 slice
    instance_uids: List[str] = field(default_factory=list)
    
    # Overlay 정보
    mask_path: Optional[str] = None
    contour_points: Optional[List[List[Tuple[int, int]]]] = None  # slice별 윤곽점
    
    # 위치 정보
    center_voxel: Tuple[int, int, int] = (0, 0, 0)
    bbox_voxel: Optional[Tuple[slice, slice, slice]] = None
    
    # 메타
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "evidence_id": self.evidence_id,
            "series_uid": self.series_uid,
            "slice_range": self.slice_range,
            "center_slice": self.center_slice,
            "instance_uids": self.instance_uids,
            "mask_path": self.mask_path,
            "center_voxel": self.center_voxel,
            "confidence": round(self.confidence, 3)
        }
    
    def to_schema(self) -> Dict:
        """api/schemas.py VisionEvidence 형식으로 변환"""
        return {
            "series_uid": self.series_uid,
            "instance_uids": self.instance_uids,
            "slice_range": self.slice_range,
            "confidence": self.confidence,
            "mask_path": self.mask_path
        }


class EvidenceGenerator:
    """
    Vision Evidence 생성기
    
    CandidateComponent에서 viewer 연동용 Evidence 데이터 생성
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        save_masks: bool = True,
        generate_contours: bool = True
    ):
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_masks = save_masks
        self.generate_contours = generate_contours
        
        logger.info("EvidenceGenerator initialized")
    
    def generate_from_candidate(
        self,
        candidate: "CandidateComponent",
        series_uid: str,
        instance_uid_map: Optional[Dict[int, str]] = None
    ) -> VisionEvidenceData:
        """
        CandidateComponent에서 Evidence 생성
        
        Args:
            candidate: CandidateComponent 객체
            series_uid: DICOM Series UID
            instance_uid_map: {slice_index: instance_uid} 매핑
            
        Returns:
            VisionEvidenceData
        """
        evidence_id = f"ev_{candidate.candidate_id}"
        
        # Slice range
        slice_range = candidate.slice_range
        center_slice = (slice_range[0] + slice_range[1]) // 2
        
        # Instance UIDs
        instance_uids = []
        if instance_uid_map:
            for z in range(slice_range[0], slice_range[1] + 1):
                if z in instance_uid_map:
                    instance_uids.append(instance_uid_map[z])
        
        # Contours (optional)
        contour_points = None
        if self.generate_contours and candidate.component_mask is not None:
            contour_points = self._extract_contours(candidate.component_mask, slice_range)
        
        # Save mask
        mask_path = None
        if self.save_masks and self.output_dir and candidate.component_mask is not None:
            mask_path = self._save_mask(
                candidate.component_mask, 
                evidence_id,
                self.output_dir
            )
        
        evidence = VisionEvidenceData(
            evidence_id=evidence_id,
            series_uid=series_uid,
            slice_range=slice_range,
            center_slice=center_slice,
            instance_uids=instance_uids,
            mask_path=mask_path,
            contour_points=contour_points,
            center_voxel=candidate.peak_zyx,
            confidence=candidate.confidence
        )
        
        return evidence
    
    def generate_batch(
        self,
        candidates: List["CandidateComponent"],
        series_uid: str,
        instance_uid_map: Optional[Dict[int, str]] = None
    ) -> List[VisionEvidenceData]:
        """
        여러 CandidateComponent에서 Evidence 일괄 생성
        """
        evidences = []
        
        for candidate in candidates:
            if candidate.status == "hidden":
                continue
            
            evidence = self.generate_from_candidate(
                candidate=candidate,
                series_uid=series_uid,
                instance_uid_map=instance_uid_map
            )
            evidences.append(evidence)
        
        logger.info(f"Generated {len(evidences)} evidence objects")
        return evidences
    
    def _extract_contours(
        self,
        mask: np.ndarray,
        slice_range: Tuple[int, int]
    ) -> List[List[Tuple[int, int]]]:
        """
        마스크에서 slice별 윤곽선 추출
        
        Returns:
            List of contour points per slice
        """
        try:
            from skimage import measure
        except ImportError:
            logger.warning("skimage not available, skipping contour extraction")
            return None
        
        contours_per_slice = []
        
        for z in range(slice_range[0], slice_range[1] + 1):
            if z >= mask.shape[0]:
                continue
            
            slice_mask = mask[z]
            
            if not np.any(slice_mask):
                contours_per_slice.append([])
                continue
            
            # Find contours
            contours = measure.find_contours(slice_mask, 0.5)
            
            # Convert to list of tuples
            slice_contours = []
            for contour in contours:
                points = [(int(p[1]), int(p[0])) for p in contour]  # (x, y)
                slice_contours.extend(points)
            
            contours_per_slice.append(slice_contours)
        
        return contours_per_slice
    
    def _save_mask(
        self,
        mask: np.ndarray,
        evidence_id: str,
        output_dir: Path
    ) -> str:
        """마스크를 NIfTI로 저장"""
        try:
            import nibabel as nib
        except ImportError:
            logger.warning("nibabel not available, skipping mask save")
            return None
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mask_path = output_dir / f"{evidence_id}_mask.nii.gz"
        nib.save(
            nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)),
            mask_path
        )
        
        return str(mask_path)
    
    def create_evidence_manifest(
        self,
        evidences: List[VisionEvidenceData],
        output_path: Path
    ):
        """Evidence manifest 저장"""
        manifest = {
            "total": len(evidences),
            "evidences": [e.to_dict() for e in evidences]
        }
        
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Evidence manifest saved to {output_path}")


def create_instance_uid_map_from_dicom(dicom_dir: Path) -> Dict[int, str]:
    """
    DICOM 디렉토리에서 slice index → instance UID 매핑 생성
    """
    try:
        import pydicom
    except ImportError:
        logger.warning("pydicom not available")
        return {}
    
    uid_map = {}
    
    dicom_files = list(dicom_dir.glob("*.dcm"))
    if not dicom_files:
        dicom_files = [f for f in dicom_dir.iterdir() if f.is_file()]
    
    slices = []
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            z_pos = float(ds.ImagePositionPatient[2])
            instance_uid = str(ds.SOPInstanceUID)
            slices.append((z_pos, instance_uid))
        except Exception:
            continue
    
    # Sort by Z position
    slices.sort(key=lambda x: x[0])
    
    # Create mapping (slice index → instance UID)
    for idx, (_, uid) in enumerate(slices):
        uid_map[idx] = uid
    
    return uid_map
