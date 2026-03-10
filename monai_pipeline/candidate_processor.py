# monai_pipeline/candidate_processor.py
"""
Candidate Processor: Heatmap → 제품 출력 변환 파이프라인

로드맵:
1. Peak → Candidate 생성 (threshold + local maxima)
2. Peak → Component 분리 (connected components)
3. Measurements 계산 (diameter, volume)
4. Evidence 생성 (slice_range, mask)
5. Threshold 정책 적용 (findings vs limitations)

핵심 원칙:
- heatmap 모델은 '후보 생성기'일 뿐
- 제품은 그 후보를 '증거·표·비교·로그'로 바꾸는 시스템
"""
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import (
    label as scipy_label,
    center_of_mass,
    find_objects,
    maximum_filter,
    binary_dilation,
    generate_binary_structure
)
from pathlib import Path
import json

from utils.logger import logger


@dataclass
class CandidateComponent:
    """후보 결절의 Component 정보"""
    candidate_id: str
    
    # Peak 정보
    peak_zyx: Tuple[float, float, float]  # voxel coordinates
    peak_value: float  # heatmap value at peak
    
    # Component 정보
    component_mask: Optional[np.ndarray] = None  # binary mask
    voxel_count: int = 0
    bbox: Optional[Tuple[slice, slice, slice]] = None  # bounding box
    
    # Measurements (spacing 적용 후)
    center_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    diameter_mm: float = 0.0
    volume_mm3: float = 0.0
    
    # Evidence
    slice_range: Tuple[int, int] = (0, 0)  # z slice range
    
    # Confidence & Status
    confidence: float = 0.0
    status: str = "candidate"  # candidate, finding, limitation, hidden
    
    # Location code (lung region)
    location_code: str = "UNK"  # RUL, RML, RLL, LUL, LLL, etc.
    
    def to_dict(self) -> Dict:
        return {
            "candidate_id": self.candidate_id,
            "peak_zyx": self.peak_zyx,
            "peak_value": self.peak_value,
            "voxel_count": self.voxel_count,
            "center_mm": self.center_mm,
            "diameter_mm": round(self.diameter_mm, 2),
            "volume_mm3": round(self.volume_mm3, 1),
            "slice_range": self.slice_range,
            "confidence": round(self.confidence, 3),
            "status": self.status,
            "location_code": self.location_code
        }


@dataclass
class ThresholdPolicy:
    """후보 분류 임계값 정책"""
    # Heatmap peak threshold (높을수록 엄격)
    peak_threshold: float = 0.15  # v4 model heatmap activation range에 맞춤

    # Confidence thresholds for status
    finding_threshold: float = 0.25     # >= 0.25 → Findings 표
    limitation_threshold: float = 0.10  # 0.10 ~ 0.25 → Limitations에만
    hidden_threshold: float = 0.10      # < 0.10 → 숨김
    
    # Size filters
    min_diameter_mm: float = 3.0
    max_diameter_mm: float = 30.0
    min_voxel_count: int = 20  # 최소 voxel 수 (너무 작은 것 필터)
    
    # Component extraction (개선됨)
    search_radius_mm: float = 15.0      # peak 주변 검색 반경 (이름 변경)
    adaptive_threshold_ratio: float = 0.5  # 0.3 → 0.5 (더 tight한 component)
    
    # High-confidence threshold (Key Flags용)
    high_confidence_threshold: float = 0.35  # v4 model 기준 높은 신뢰도
    
    def classify(self, confidence: float, diameter_mm: float, voxel_count: int = 100) -> str:
        """후보 분류"""
        # Voxel count filter (너무 작으면 노이즈)
        if voxel_count < self.min_voxel_count:
            return "hidden"
        
        # Size filter
        if diameter_mm < self.min_diameter_mm:
            return "hidden"
        if diameter_mm > self.max_diameter_mm:
            return "limitation"  # 너무 큰 것은 mass로 별도 처리 필요
        
        # Confidence-based classification
        if confidence >= self.finding_threshold:
            return "finding"
        elif confidence >= self.limitation_threshold:
            return "limitation"
        else:
            return "hidden"


class CandidateProcessor:
    """
    Heatmap → Candidate → Component → Measurements → Evidence
    
    전체 파이프라인 관리
    """
    
    def __init__(
        self,
        policy: Optional[ThresholdPolicy] = None,
        output_dir: Optional[Path] = None
    ):
        self.policy = policy or ThresholdPolicy()
        self.output_dir = Path(output_dir) if output_dir else None
        
        logger.info(f"CandidateProcessor initialized with policy: "
                   f"peak_thresh={self.policy.peak_threshold}, "
                   f"finding_thresh={self.policy.finding_threshold}")
    
    def process(
        self,
        heatmap: np.ndarray,
        spacing_mm: Tuple[float, float, float],
        series_uid: str,
        lung_mask: Optional[np.ndarray] = None
    ) -> List[CandidateComponent]:
        """
        전체 파이프라인 실행
        
        Args:
            heatmap: (D, H, W) heatmap array (0~1)
            spacing_mm: (z, y, x) spacing in mm
            series_uid: DICOM series UID
            lung_mask: Optional lung segmentation mask
            
        Returns:
            List of CandidateComponent objects
        """
        logger.info(f"Processing heatmap: shape={heatmap.shape}, spacing={spacing_mm}")
        
        # 1. Peak detection (local maxima)
        peaks = self._detect_peaks(heatmap)
        logger.info(f"Detected {len(peaks)} peaks above threshold {self.policy.peak_threshold}")
        
        if len(peaks) == 0:
            return []
        
        # 2. Component extraction for each peak
        candidates = []
        for i, (z, y, x, peak_val) in enumerate(peaks):
            candidate_id = f"CAND_{series_uid[-8:]}_{i+1:03d}"
            
            # Extract component
            component = self._extract_component(
                heatmap=heatmap,
                peak_zyx=(z, y, x),
                peak_value=peak_val,
                candidate_id=candidate_id,
                spacing_mm=spacing_mm
            )
            
            if component is not None:
                # 3. Calculate measurements
                self._calculate_measurements(component, spacing_mm)
                
                # 4. Generate evidence
                self._generate_evidence(component)
                
                # 5. Apply threshold policy
                component.status = self.policy.classify(
                    component.confidence, 
                    component.diameter_mm,
                    component.voxel_count
                )
                
                # 6. Determine location (lung mask optional)
                component.location_code = self._determine_location(
                    component.peak_zyx, 
                    heatmap.shape,
                    lung_mask
                )
                
                candidates.append(component)
        
        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        # Filter hidden candidates
        visible = [c for c in candidates if c.status != "hidden"]
        logger.info(f"Final candidates: {len(visible)} visible, "
                   f"{len(candidates) - len(visible)} hidden")
        
        return candidates
    
    def _detect_peaks(
        self,
        heatmap: np.ndarray,
        min_distance: int = 10  # 5 → 10 (peak 간 최소 거리 증가)
    ) -> List[Tuple[int, int, int, float]]:
        """
        Local maxima 검출 (3D NMS)
        
        Returns:
            List of (z, y, x, value) tuples
        """
        # Threshold
        binary = heatmap >= self.policy.peak_threshold
        
        if not np.any(binary):
            return []
        
        # Local maxima detection using maximum filter
        # Size determines minimum distance between peaks
        size = 2 * min_distance + 1
        local_max = maximum_filter(heatmap, size=size)
        
        # Peak = local max AND above threshold
        peaks_mask = (heatmap == local_max) & binary
        
        # Get peak coordinates
        peak_coords = np.where(peaks_mask)
        peaks = []
        
        for z, y, x in zip(*peak_coords):
            val = heatmap[z, y, x]
            peaks.append((int(z), int(y), int(x), float(val)))
        
        # Sort by value (highest first)
        peaks.sort(key=lambda p: p[3], reverse=True)
        
        return peaks
    
    def _extract_component(
        self,
        heatmap: np.ndarray,
        peak_zyx: Tuple[int, int, int],
        peak_value: float,
        candidate_id: str,
        spacing_mm: Tuple[float, float, float]
    ) -> Optional[CandidateComponent]:
        """
        Peak에서 flood-fill 방식으로 실제 component 추출 (개선됨)
        
        1. Peak에서 시작해서 adaptive threshold 이상인 연결된 영역 찾기
        2. 검색 영역은 제한하되, component 크기는 실제 연결된 영역 기반
        """
        z, y, x = peak_zyx
        
        # Adaptive threshold: peak 값의 일정 비율
        adaptive_thresh = peak_value * self.policy.adaptive_threshold_ratio
        
        # 검색 영역 제한 (메모리/성능)
        search_radius_voxels = [
            int(self.policy.search_radius_mm / s) 
            for s in spacing_mm
        ]
        
        # Bounds for search area
        z_min = max(0, z - search_radius_voxels[0])
        z_max = min(heatmap.shape[0], z + search_radius_voxels[0] + 1)
        y_min = max(0, y - search_radius_voxels[1])
        y_max = min(heatmap.shape[1], y + search_radius_voxels[1] + 1)
        x_min = max(0, x - search_radius_voxels[2])
        x_max = min(heatmap.shape[2], x + search_radius_voxels[2] + 1)
        
        # Extract local region
        local_heatmap = heatmap[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Apply adaptive threshold
        local_binary = local_heatmap >= adaptive_thresh
        
        if not np.any(local_binary):
            return None
        
        # Connected components (26-connectivity for 3D)
        struct = generate_binary_structure(3, 2)
        labeled, num_features = scipy_label(local_binary, structure=struct)
        
        if num_features == 0:
            return None
        
        # Find component containing the peak
        local_peak = (z - z_min, y - y_min, x - x_min)
        
        # Check if peak is within local region
        if not (0 <= local_peak[0] < local_binary.shape[0] and
                0 <= local_peak[1] < local_binary.shape[1] and
                0 <= local_peak[2] < local_binary.shape[2]):
            return None
        
        peak_label = labeled[local_peak]
        
        if peak_label == 0:
            # Peak not in any component - dilate once
            local_binary_dilated = binary_dilation(local_binary, iterations=1)
            labeled, _ = scipy_label(local_binary_dilated, structure=struct)
            peak_label = labeled[local_peak]
        
        if peak_label == 0:
            return None
        
        # Extract ONLY the component containing the peak (실제 크기!)
        component_mask_local = (labeled == peak_label).astype(np.uint8)
        voxel_count = int(np.sum(component_mask_local))
        
        # 너무 작은 component 필터링 (노이즈)
        if voxel_count < self.policy.min_voxel_count:
            return None
        
        # Create full-size mask
        full_mask = np.zeros(heatmap.shape, dtype=np.uint8)
        full_mask[z_min:z_max, y_min:y_max, x_min:x_max] = component_mask_local
        
        # Get bounding box of the actual component
        slices = find_objects(full_mask)[0] if np.any(full_mask) else None
        
        # Create candidate
        candidate = CandidateComponent(
            candidate_id=candidate_id,
            peak_zyx=peak_zyx,
            peak_value=peak_value,
            component_mask=full_mask,
            voxel_count=voxel_count,
            bbox=slices,
            confidence=peak_value
        )
        
        return candidate
    
    def _calculate_measurements(
        self,
        candidate: CandidateComponent,
        spacing_mm: Tuple[float, float, float]
    ):
        """
        Component에서 measurements 계산 (개선됨)
        
        - diameter_mm: 실제 component의 최대 직경 (bounding box + volume 검증)
        - volume_mm3: voxel count × voxel volume
        - center_mm: center of mass in mm
        """
        if candidate.component_mask is None or candidate.voxel_count == 0:
            return
        
        mask = candidate.component_mask
        
        # Volume (mm³)
        voxel_volume = spacing_mm[0] * spacing_mm[1] * spacing_mm[2]
        candidate.volume_mm3 = candidate.voxel_count * voxel_volume
        
        # Center of mass
        coords = np.where(mask > 0)
        if len(coords[0]) > 0:
            center_voxel = (
                np.mean(coords[0]),
                np.mean(coords[1]),
                np.mean(coords[2])
            )
            candidate.center_mm = (
                center_voxel[0] * spacing_mm[0],
                center_voxel[1] * spacing_mm[1],
                center_voxel[2] * spacing_mm[2]
            )
        
        # Diameter 계산: Bounding box + Volume 기반 cross-validation
        diameter_bbox = 0.0
        diameter_volume = 0.0
        
        # 방법 1: Bounding box 기반 (실제 extent)
        if candidate.bbox:
            z_slice, y_slice, x_slice = candidate.bbox
            
            extent_z = (z_slice.stop - z_slice.start) * spacing_mm[0]
            extent_y = (y_slice.stop - y_slice.start) * spacing_mm[1]
            extent_x = (x_slice.stop - x_slice.start) * spacing_mm[2]
            
            # 최대 extent가 직경 (보수적 추정)
            diameter_bbox = max(extent_z, extent_y, extent_x)
        
        # 방법 2: Volume 기반 (구 가정) - equivalent spherical diameter
        # V = (4/3) * pi * r³ → d = 2 * (3V / 4pi)^(1/3)
        if candidate.volume_mm3 > 0:
            r = (3 * candidate.volume_mm3 / (4 * np.pi)) ** (1/3)
            diameter_volume = 2 * r
        
        # 두 방법의 평균 또는 선택
        # bounding box는 과대추정, volume 기반은 구형 가정
        # 실제 결절은 비구형이므로 두 값의 조화 사용
        if diameter_bbox > 0 and diameter_volume > 0:
            # 작은 값 쪽으로 가중치 (보수적)
            candidate.diameter_mm = (diameter_bbox + 2 * diameter_volume) / 3
        elif diameter_bbox > 0:
            candidate.diameter_mm = diameter_bbox
        else:
            candidate.diameter_mm = diameter_volume
    
    def _generate_evidence(self, candidate: CandidateComponent):
        """
        Evidence 정보 생성 (slice range, etc.)
        """
        if candidate.component_mask is None:
            return
        
        # Slice range (z axis)
        z_coords = np.where(np.any(candidate.component_mask, axis=(1, 2)))[0]
        
        if len(z_coords) > 0:
            candidate.slice_range = (int(z_coords.min()), int(z_coords.max()))
    
    def _determine_location(
        self,
        peak_zyx: Tuple[int, int, int],
        volume_shape: Tuple[int, int, int],
        lung_mask: Optional[np.ndarray] = None
    ) -> str:
        """
        결절 위치 결정 (폐엽 기반, 개선됨)
        
        CT 좌표계 기준:
        - X axis: Right → Left (환자 기준)
        - Y axis: Anterior → Posterior  
        - Z axis: Inferior → Superior (feet to head)
        
        Args:
            peak_zyx: (z, y, x) voxel coordinates
            volume_shape: (D, H, W) shape of the volume
            lung_mask: Optional lung segmentation mask
            
        Returns:
            Location code: RUL, RML, RLL, LUL, LLL, or combinations
        """
        z, y, x = peak_zyx
        D, H, W = volume_shape
        
        # Lung mask가 있으면 해당 위치가 폐 내부인지 확인
        if lung_mask is not None:
            if not lung_mask[int(z), int(y), int(x)]:
                # 폐 외부 - extrapulmonary
                return "EXTRA"
        
        # 좌우 결정 (X축 기준)
        # CT에서 환자 오른쪽이 이미지 왼쪽 (x가 작은 쪽)
        x_ratio = x / W
        side = "R" if x_ratio < 0.5 else "L"
        
        # 상하 결정 (Z축 기준) - CT는 보통 feet-first
        # z가 작을수록 하부 (inferior), z가 클수록 상부 (superior)
        z_ratio = z / D
        
        # 전후 결정 (Y축 기준) - anterior vs posterior
        y_ratio = y / H
        
        # 폐엽 결정 로직 (해부학적 근사)
        # - 상엽(UL): 상부 40%
        # - 중엽(ML): 중부 30% (우측만)
        # - 하엽(LL): 하부 30%
        
        if z_ratio > 0.6:
            # 상부 - Upper Lobe
            lobe = "UL"
        elif z_ratio < 0.3:
            # 하부 - Lower Lobe
            lobe = "LL"
        else:
            # 중간부
            if side == "R":
                # 우측 중엽 vs 하엽
                # 전방(anterior)이면 중엽, 후방(posterior)이면 하엽
                if y_ratio < 0.5:
                    lobe = "ML"  # Right Middle Lobe
                else:
                    lobe = "LL"  # Right Lower Lobe
            else:
                # 좌측에는 중엽 없음 - Lingula는 상엽의 일부
                if y_ratio < 0.4:
                    lobe = "UL"  # Left Upper Lobe (including Lingula)
                else:
                    lobe = "LL"  # Left Lower Lobe
        
        return f"{side}{lobe}"
    
    def save_candidates(
        self,
        candidates: List[CandidateComponent],
        output_path: Path,
        save_masks: bool = False
    ):
        """
        후보 결과 저장
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON summary
        summary = {
            "total_candidates": len(candidates),
            "findings": len([c for c in candidates if c.status == "finding"]),
            "limitations": len([c for c in candidates if c.status == "limitation"]),
            "hidden": len([c for c in candidates if c.status == "hidden"]),
            "candidates": [c.to_dict() for c in candidates]
        }
        
        with open(output_path / "candidates.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save masks if requested
        if save_masks:
            import nibabel as nib
            
            for c in candidates:
                if c.component_mask is not None and c.status != "hidden":
                    mask_path = output_path / f"{c.candidate_id}_mask.nii.gz"
                    nib.save(
                        nib.Nifti1Image(c.component_mask.astype(np.uint8), np.eye(4)),
                        mask_path
                    )
        
        logger.info(f"Saved candidates to {output_path}")


def create_processor_from_config(config: Dict) -> CandidateProcessor:
    """설정에서 Processor 생성"""
    policy = ThresholdPolicy(
        peak_threshold=config.get("peak_threshold", 0.15),
        finding_threshold=config.get("finding_threshold", 0.25),
        limitation_threshold=config.get("limitation_threshold", 0.10),
        min_diameter_mm=config.get("min_diameter_mm", 3.0),
        max_diameter_mm=config.get("max_diameter_mm", 30.0),
        min_voxel_count=config.get("min_voxel_count", 20),
        search_radius_mm=config.get("search_radius_mm", 15.0),
        adaptive_threshold_ratio=config.get("adaptive_threshold_ratio", 0.5),
        high_confidence_threshold=config.get("high_confidence_threshold", 0.35)
    )
    
    return CandidateProcessor(policy=policy)
