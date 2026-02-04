# monai_pipeline/tracking_engine.py
"""
Lesion Tracking Engine: Prior Comparison 전용

RAG보다 먼저 실행됨!

원리:
1. Center distance 기반 매칭
2. Diameter/Volume overlap 확인
3. 변화 계산 (INCREASED, STABLE, DECREASED, NEW, RESOLVED)

학습 없음 - 순수 규칙 기반
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from enum import Enum

from utils.logger import logger


class ChangeType(str, Enum):
    """변화 유형"""
    NEW = "NEW"           # 이전에 없음
    STABLE = "Stable"     # 변화 없음 (±20%)
    INCREASED = "Increased"  # 20% 이상 증가
    DECREASED = "Decreased"  # 20% 이상 감소
    RESOLVED = "Resolved"    # 이전에 있었으나 현재 없음


@dataclass
class LesionMatch:
    """병변 매칭 결과"""
    current_id: str
    prior_id: Optional[str] = None
    
    # 현재 측정값
    current_center_mm: Tuple[float, float, float] = (0, 0, 0)
    current_diameter_mm: float = 0.0
    current_volume_mm3: float = 0.0
    
    # 이전 측정값
    prior_center_mm: Optional[Tuple[float, float, float]] = None
    prior_diameter_mm: Optional[float] = None
    prior_volume_mm3: Optional[float] = None
    prior_date: Optional[str] = None
    
    # 변화 분석
    change_type: ChangeType = ChangeType.NEW
    diameter_change_pct: float = 0.0
    volume_change_pct: float = 0.0
    distance_mm: float = 0.0
    
    # 매칭 신뢰도
    match_confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "current_id": self.current_id,
            "prior_id": self.prior_id,
            "current_diameter_mm": round(self.current_diameter_mm, 2),
            "prior_diameter_mm": round(self.prior_diameter_mm, 2) if self.prior_diameter_mm else None,
            "current_volume_mm3": round(self.current_volume_mm3, 1),
            "prior_volume_mm3": round(self.prior_volume_mm3, 1) if self.prior_volume_mm3 else None,
            "prior_date": self.prior_date,
            "change_type": self.change_type.value,
            "diameter_change_pct": round(self.diameter_change_pct, 1),
            "volume_change_pct": round(self.volume_change_pct, 1),
            "distance_mm": round(self.distance_mm, 2),
            "match_confidence": round(self.match_confidence, 3)
        }
    
    def to_table_row(self) -> Dict:
        """Prior Comparison Table용 row"""
        return {
            "lesion_id": self.current_id,
            "prior_date": self.prior_date or "-",
            "change_type": self.change_type.value,
            "prior_size_mm": f"{self.prior_diameter_mm:.1f}" if self.prior_diameter_mm else "-",
            "current_size_mm": f"{self.current_diameter_mm:.1f}",
            "change_pct": f"{self.diameter_change_pct:+.0f}%" if self.prior_diameter_mm else "-"
        }


@dataclass 
class PriorLesion:
    """이전 검사의 병변 정보"""
    lesion_id: str
    center_mm: Tuple[float, float, float]
    diameter_mm: float
    volume_mm3: float
    study_date: str
    confidence: float = 0.0
    matched: bool = False  # 매칭 여부 추적


@dataclass
class TrackingPolicy:
    """매칭 정책"""
    # 거리 임계값 (mm)
    max_distance_mm: float = 15.0
    
    # 크기 변화 임계값 (%)
    stable_threshold_pct: float = 20.0  # ±20% 이내면 Stable
    
    # 매칭 신뢰도 가중치
    distance_weight: float = 0.6
    size_weight: float = 0.4
    
    # 최소 매칭 신뢰도
    min_match_confidence: float = 0.5


class LesionTrackingEngine:
    """
    병변 추적 엔진
    
    현재 검사의 병변과 이전 검사의 병변을 매칭하고
    변화를 분석합니다.
    """
    
    def __init__(self, policy: Optional[TrackingPolicy] = None):
        self.policy = policy or TrackingPolicy()
        logger.info("LesionTrackingEngine initialized")
    
    def track(
        self,
        current_lesions: List[Dict],
        prior_lesions: List[PriorLesion]
    ) -> List[LesionMatch]:
        """
        현재/이전 병변 매칭 및 변화 분석
        
        Args:
            current_lesions: 현재 검사 병변 리스트
                [{"id": str, "center_mm": tuple, "diameter_mm": float, "volume_mm3": float}, ...]
            prior_lesions: 이전 검사 병변 리스트
            
        Returns:
            List of LesionMatch objects
        """
        logger.info(f"Tracking: {len(current_lesions)} current, {len(prior_lesions)} prior")
        
        matches = []
        
        # Reset matched flags
        for pl in prior_lesions:
            pl.matched = False
        
        # 1. 각 현재 병변에 대해 최적 매칭 찾기
        for curr in current_lesions:
            best_match = self._find_best_match(curr, prior_lesions)
            matches.append(best_match)
        
        # 2. 매칭되지 않은 이전 병변 = RESOLVED
        for pl in prior_lesions:
            if not pl.matched:
                resolved = LesionMatch(
                    current_id=f"RESOLVED_{pl.lesion_id}",
                    prior_id=pl.lesion_id,
                    prior_center_mm=pl.center_mm,
                    prior_diameter_mm=pl.diameter_mm,
                    prior_volume_mm3=pl.volume_mm3,
                    prior_date=pl.study_date,
                    change_type=ChangeType.RESOLVED
                )
                matches.append(resolved)
        
        # Sort by change type priority: INCREASED > NEW > STABLE > DECREASED > RESOLVED
        priority = {
            ChangeType.INCREASED: 0,
            ChangeType.NEW: 1,
            ChangeType.STABLE: 2,
            ChangeType.DECREASED: 3,
            ChangeType.RESOLVED: 4
        }
        matches.sort(key=lambda m: priority.get(m.change_type, 5))
        
        logger.info(f"Tracking complete: "
                   f"{sum(1 for m in matches if m.change_type == ChangeType.NEW)} NEW, "
                   f"{sum(1 for m in matches if m.change_type == ChangeType.STABLE)} Stable, "
                   f"{sum(1 for m in matches if m.change_type == ChangeType.INCREASED)} Increased, "
                   f"{sum(1 for m in matches if m.change_type == ChangeType.DECREASED)} Decreased, "
                   f"{sum(1 for m in matches if m.change_type == ChangeType.RESOLVED)} Resolved")
        
        return matches
    
    def _find_best_match(
        self,
        current: Dict,
        prior_lesions: List[PriorLesion]
    ) -> LesionMatch:
        """
        현재 병변에 대한 최적 매칭 찾기
        """
        curr_id = current["id"]
        curr_center = current["center_mm"]
        curr_diameter = current["diameter_mm"]
        curr_volume = current["volume_mm3"]
        
        best_prior = None
        best_confidence = 0.0
        best_distance = float('inf')
        
        for pl in prior_lesions:
            if pl.matched:
                continue
            
            # 거리 계산
            distance = self._calculate_distance(curr_center, pl.center_mm)
            
            if distance > self.policy.max_distance_mm:
                continue
            
            # 매칭 신뢰도 계산
            confidence = self._calculate_match_confidence(
                distance=distance,
                curr_diameter=curr_diameter,
                prior_diameter=pl.diameter_mm
            )
            
            if confidence > best_confidence and confidence >= self.policy.min_match_confidence:
                best_prior = pl
                best_confidence = confidence
                best_distance = distance
        
        # 매칭 결과 생성
        match = LesionMatch(
            current_id=curr_id,
            current_center_mm=curr_center,
            current_diameter_mm=curr_diameter,
            current_volume_mm3=curr_volume
        )
        
        if best_prior:
            # 매칭 성공
            best_prior.matched = True
            
            match.prior_id = best_prior.lesion_id
            match.prior_center_mm = best_prior.center_mm
            match.prior_diameter_mm = best_prior.diameter_mm
            match.prior_volume_mm3 = best_prior.volume_mm3
            match.prior_date = best_prior.study_date
            match.distance_mm = best_distance
            match.match_confidence = best_confidence
            
            # 변화 계산
            match.diameter_change_pct = self._calculate_change_pct(
                curr_diameter, best_prior.diameter_mm
            )
            match.volume_change_pct = self._calculate_change_pct(
                curr_volume, best_prior.volume_mm3
            )
            
            # 변화 유형 결정
            match.change_type = self._determine_change_type(match.diameter_change_pct)
        else:
            # 매칭 실패 = NEW
            match.change_type = ChangeType.NEW
        
        return match
    
    def _calculate_distance(
        self,
        center1: Tuple[float, float, float],
        center2: Tuple[float, float, float]
    ) -> float:
        """3D 유클리드 거리 (mm)"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(center1, center2)))
    
    def _calculate_match_confidence(
        self,
        distance: float,
        curr_diameter: float,
        prior_diameter: float
    ) -> float:
        """매칭 신뢰도 계산"""
        # 거리 점수 (가까울수록 높음)
        max_dist = self.policy.max_distance_mm
        distance_score = max(0, 1 - distance / max_dist)
        
        # 크기 유사도 (비슷할수록 높음)
        if prior_diameter > 0:
            size_ratio = min(curr_diameter, prior_diameter) / max(curr_diameter, prior_diameter)
        else:
            size_ratio = 0.5
        
        # 가중 평균
        confidence = (
            self.policy.distance_weight * distance_score +
            self.policy.size_weight * size_ratio
        )
        
        return confidence
    
    def _calculate_change_pct(self, current: float, prior: float) -> float:
        """변화율 계산 (%)"""
        if prior == 0:
            return 0.0
        return ((current - prior) / prior) * 100
    
    def _determine_change_type(self, change_pct: float) -> ChangeType:
        """변화 유형 결정"""
        threshold = self.policy.stable_threshold_pct
        
        if change_pct > threshold:
            return ChangeType.INCREASED
        elif change_pct < -threshold:
            return ChangeType.DECREASED
        else:
            return ChangeType.STABLE
    
    def create_comparison_table(
        self,
        matches: List[LesionMatch],
        include_resolved: bool = True
    ) -> List[Dict]:
        """
        Prior Comparison Table 데이터 생성
        """
        rows = []
        
        for match in matches:
            if match.change_type == ChangeType.RESOLVED and not include_resolved:
                continue
            
            rows.append(match.to_table_row())
        
        return rows


def load_prior_lesions_from_report(prior_report: Dict) -> List[PriorLesion]:
    """
    이전 리포트에서 병변 정보 추출
    
    Args:
        prior_report: 이전 리포트 JSON (measurements 포함)
        
    Returns:
        List of PriorLesion objects
    """
    lesions = []
    
    measurements = prior_report.get("measurements", [])
    study_date = prior_report.get("study_date", "unknown")
    
    for m in measurements:
        lesion = PriorLesion(
            lesion_id=m.get("lesion_id", "unknown"),
            center_mm=tuple(m.get("center_mm", (0, 0, 0))),
            diameter_mm=m.get("diameter_mm", 0.0),
            volume_mm3=m.get("volume_mm3", 0.0),
            study_date=study_date,
            confidence=m.get("confidence", 0.0)
        )
        lesions.append(lesion)
    
    return lesions


def load_prior_lesions_from_candidates(
    candidates: List["CandidateComponent"],
    study_date: str
) -> List[PriorLesion]:
    """
    CandidateComponent 리스트에서 Prior 정보 생성
    """
    from monai_pipeline.candidate_processor import CandidateComponent
    
    lesions = []
    
    for c in candidates:
        if c.status == "hidden":
            continue
        
        lesion = PriorLesion(
            lesion_id=c.candidate_id,
            center_mm=c.center_mm,
            diameter_mm=c.diameter_mm,
            volume_mm3=c.volume_mm3,
            study_date=study_date,
            confidence=c.confidence
        )
        lesions.append(lesion)
    
    return lesions
