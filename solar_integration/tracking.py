# solar_integration/tracking.py
"""
Lesion tracking for prior comparison
규칙 기반 매칭 우선, semantic search 보조
"""
from typing import List, Dict, Optional, Tuple
import numpy as np
from api.schemas import NoduleCandidate, StructuredAIResult
from utils.logger import logger


class LesionTracker:
    """병변 추적 (prior study와 비교)"""
    
    def __init__(
        self,
        distance_threshold_mm: float = 10.0,
        size_change_threshold_pct: float = 20.0
    ):
        """
        Args:
            distance_threshold_mm: 동일 병변으로 간주할 최대 거리 (mm)
            size_change_threshold_pct: 크기 변화 임계값 (%)
        """
        self.distance_threshold_mm = distance_threshold_mm
        self.size_change_threshold_pct = size_change_threshold_pct
        logger.info("Initialized LesionTracker")
    
    def compare_studies(
        self,
        current_result: StructuredAIResult,
        prior_result: StructuredAIResult,
        spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.5)
    ) -> Dict[str, List[Dict]]:
        """
        두 검사의 결절 비교
        
        Returns:
            {
                "new": [...],
                "stable": [...],
                "growing": [...],
                "shrinking": [...],
                "resolved": [...]
            }
        """
        logger.info(
            f"Comparing studies: current={len(current_result.nodules)} nodules, "
            f"prior={len(prior_result.nodules)} nodules"
        )
        
        comparison = {
            "new": [],
            "stable": [],
            "growing": [],
            "shrinking": [],
            "resolved": []
        }
        
        # Current nodules 처리
        matched_prior_ids = set()
        
        for current_nodule in current_result.nodules:
            # Find matching prior nodule
            match = self._find_matching_nodule(
                current_nodule,
                prior_result.nodules,
                spacing_mm
            )
            
            if match:
                prior_nodule, distance = match
                matched_prior_ids.add(prior_nodule.id)
                
                # 크기 변화 계산
                size_change_pct = (
                    (current_nodule.diameter_mm - prior_nodule.diameter_mm) /
                    prior_nodule.diameter_mm * 100
                )
                
                comparison_entry = {
                    "current": self._nodule_to_dict(current_nodule),
                    "prior": self._nodule_to_dict(prior_nodule),
                    "distance_mm": float(distance),
                    "size_change_mm": current_nodule.diameter_mm - prior_nodule.diameter_mm,
                    "size_change_pct": size_change_pct
                }
                
                # Categorize
                if abs(size_change_pct) < self.size_change_threshold_pct:
                    comparison["stable"].append(comparison_entry)
                elif size_change_pct > 0:
                    comparison["growing"].append(comparison_entry)
                else:
                    comparison["shrinking"].append(comparison_entry)
            else:
                # New nodule
                comparison["new"].append(self._nodule_to_dict(current_nodule))
        
        # Resolved nodules (prior에만 있고 current에 없음)
        for prior_nodule in prior_result.nodules:
            if prior_nodule.id not in matched_prior_ids:
                comparison["resolved"].append(self._nodule_to_dict(prior_nodule))
        
        logger.info(
            f"Comparison results: new={len(comparison['new'])}, "
            f"stable={len(comparison['stable'])}, "
            f"growing={len(comparison['growing'])}, "
            f"shrinking={len(comparison['shrinking'])}, "
            f"resolved={len(comparison['resolved'])}"
        )
        
        return comparison
    
    def _find_matching_nodule(
        self,
        target: NoduleCandidate,
        candidates: List[NoduleCandidate],
        spacing: Tuple[float, float, float]
    ) -> Optional[Tuple[NoduleCandidate, float]]:
        """
        Target nodule과 매칭되는 candidate 찾기
        
        Returns:
            (matched_nodule, distance_mm) or None
        """
        best_match = None
        best_distance = float('inf')
        
        for candidate in candidates:
            distance = self._calculate_distance(
                target.center_zyx,
                candidate.center_zyx,
                spacing
            )
            
            if distance < self.distance_threshold_mm and distance < best_distance:
                best_match = candidate
                best_distance = distance
        
        if best_match:
            return (best_match, best_distance)
        
        return None
    
    def _calculate_distance(
        self,
        pos1: Tuple[float, float, float],
        pos2: Tuple[float, float, float],
        spacing: Tuple[float, float, float]
    ) -> float:
        """3D Euclidean distance (mm)"""
        z1, y1, x1 = pos1
        z2, y2, x2 = pos2
        sz, sy, sx = spacing
        
        dist = np.sqrt(
            ((z1 - z2) * sz) ** 2 +
            ((y1 - y2) * sy) ** 2 +
            ((x1 - x2) * sx) ** 2
        )
        
        return float(dist)
    
    def _nodule_to_dict(self, nodule: NoduleCandidate) -> Dict:
        """Nodule을 dict로 변환 (serialization)"""
        return {
            "id": nodule.id,
            "location": nodule.location_code or "unspecified",
            "center_zyx": nodule.center_zyx,
            "diameter_mm": nodule.diameter_mm,
            "volume_mm3": nodule.volume_mm3,
            "confidence": nodule.confidence
        }
    
    def generate_comparison_text(self, comparison: Dict[str, List]) -> str:
        """
        Comparison 결과를 텍스트로 변환
        Template에 삽입할 용도
        """
        lines = []
        
        # New nodules
        if comparison["new"]:
            lines.append(f"{len(comparison['new'])} new nodule(s):")
            for nodule in comparison["new"]:
                lines.append(
                    f"  - {nodule['location']}: {nodule['diameter_mm']:.1f} mm"
                )
        
        # Growing nodules
        if comparison["growing"]:
            lines.append(f"{len(comparison['growing'])} growing nodule(s):")
            for entry in comparison["growing"]:
                curr = entry["current"]
                prior = entry["prior"]
                change = entry["size_change_mm"]
                lines.append(
                    f"  - {curr['location']}: "
                    f"{prior['diameter_mm']:.1f} → {curr['diameter_mm']:.1f} mm "
                    f"(+{change:.1f} mm, {entry['size_change_pct']:.1f}%)"
                )
        
        # Stable nodules
        if comparison["stable"]:
            lines.append(f"{len(comparison['stable'])} stable nodule(s)")
        
        # Shrinking nodules
        if comparison["shrinking"]:
            lines.append(f"{len(comparison['shrinking'])} shrinking nodule(s):")
            for entry in comparison["shrinking"]:
                curr = entry["current"]
                prior = entry["prior"]
                change = entry["size_change_mm"]
                lines.append(
                    f"  - {curr['location']}: "
                    f"{prior['diameter_mm']:.1f} → {curr['diameter_mm']:.1f} mm "
                    f"({change:.1f} mm, {entry['size_change_pct']:.1f}%)"
                )
        
        # Resolved nodules
        if comparison["resolved"]:
            lines.append(f"{len(comparison['resolved'])} resolved nodule(s)")
        
        if not lines:
            return "No significant interval changes."
        
        return "\n".join(lines)
    
    def get_significant_changes(self, comparison: Dict[str, List]) -> List[str]:
        """
        임상적으로 유의미한 변화만 추출
        Impression에 포함할 용도
        """
        significant = []
        
        # New nodules (항상 유의미)
        if comparison["new"]:
            significant.append(f"{len(comparison['new'])} new nodule(s)")
        
        # Growing nodules (>20% 증가)
        growing = [
            e for e in comparison["growing"]
            if e["size_change_pct"] > self.size_change_threshold_pct
        ]
        if growing:
            significant.append(f"{len(growing)} growing nodule(s)")
        
        # Resolved nodules
        if comparison["resolved"]:
            significant.append(f"{len(comparison['resolved'])} resolved nodule(s)")
        
        return significant
