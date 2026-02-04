#!/usr/bin/env python
# scripts/generate_synthetic_priors.py
"""
Synthetic Prior 생성 스크립트

Prior Comparison 테이블 검증을 위한 가상 시계열 데이터 생성
- 같은 patient_id, 다른 study_date
- lesion 크기 변화 (NEW / Stable / Increased / Decreased / Resolved)

사용:
    python scripts/generate_synthetic_priors.py --output data/synthetic_priors
"""
import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

# 프로젝트 루트를 path에 추가
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import logger


class SyntheticPriorGenerator:
    """
    Synthetic prior 데이터 생성기
    
    시나리오:
    1. NEW: 이전에 없던 결절이 새로 발견
    2. Stable: 크기 변화 ±10% 이내
    3. Increased: 크기 10% 이상 증가
    4. Decreased: 크기 10% 이상 감소
    5. Resolved: 이전에 있던 결절이 사라짐
    """
    
    CHANGE_TYPES = ["NEW", "Stable", "Increased", "Decreased", "Resolved"]
    
    LOCATION_CODES = ["RUL", "RML", "RLL", "LUL", "LLL"]
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
        logger.info(f"SyntheticPriorGenerator initialized (seed={seed})")
    
    def generate_patient_history(
        self,
        patient_id: str,
        num_studies: int = 3,
        base_date: Optional[datetime] = None,
        interval_days: int = 90
    ) -> List[Dict]:
        """
        단일 환자의 시계열 데이터 생성
        
        Args:
            patient_id: 환자 ID
            num_studies: 검사 횟수
            base_date: 첫 검사 날짜
            interval_days: 검사 간격 (일)
            
        Returns:
            시간순 정렬된 study 리스트
        """
        base_date = base_date or datetime(2025, 1, 1)
        
        studies = []
        active_lesions = {}  # lesion_id -> current state
        
        for study_idx in range(num_studies):
            study_date = base_date + timedelta(days=study_idx * interval_days)
            study_uid = f"1.2.3.{patient_id}.{study_idx+1}"
            
            lesions = []
            changes = []
            
            # 기존 병변 처리
            for lesion_id, state in list(active_lesions.items()):
                change_type = self._decide_change(state)
                
                if change_type == "Resolved":
                    changes.append({
                        "lesion_id": lesion_id,
                        "change": "Resolved",
                        "prior_size": f"{state['diameter_mm']:.1f} mm",
                        "current_size": "-"
                    })
                    del active_lesions[lesion_id]
                else:
                    new_diameter = self._apply_change(state["diameter_mm"], change_type)
                    
                    lesion = {
                        "lesion_id": lesion_id,
                        "location": state["location"],
                        "diameter_mm": round(new_diameter, 1),
                        "volume_mm3": round(self._diameter_to_volume(new_diameter), 1),
                        "confidence": round(random.uniform(0.75, 0.95), 2)
                    }
                    lesions.append(lesion)
                    
                    changes.append({
                        "lesion_id": lesion_id,
                        "change": change_type,
                        "prior_size": f"{state['diameter_mm']:.1f} mm",
                        "current_size": f"{new_diameter:.1f} mm"
                    })
                    
                    active_lesions[lesion_id]["diameter_mm"] = new_diameter
            
            # 새 병변 추가 (확률적)
            if study_idx == 0 or random.random() < 0.3:
                new_lesion = self._generate_new_lesion(
                    patient_id=patient_id,
                    lesion_num=len(active_lesions) + 1
                )
                lesions.append(new_lesion)
                
                active_lesions[new_lesion["lesion_id"]] = {
                    "location": new_lesion["location"],
                    "diameter_mm": new_lesion["diameter_mm"]
                }
                
                if study_idx > 0:
                    changes.append({
                        "lesion_id": new_lesion["lesion_id"],
                        "change": "NEW",
                        "prior_size": "-",
                        "current_size": f"{new_lesion['diameter_mm']:.1f} mm"
                    })
            
            study = {
                "patient_id": patient_id,
                "study_uid": study_uid,
                "study_date": study_date.strftime("%Y-%m-%d"),
                "acquisition_datetime": study_date.isoformat(),
                "lesions": lesions,
                "changes": changes if study_idx > 0 else [],
                "quality": {
                    "slice_thickness_mm": random.choice([1.0, 1.25, 2.5, 5.0]),
                    "coverage_score": round(random.uniform(0.85, 0.98), 2),
                    "artifact_score": round(random.uniform(0.0, 0.3), 2)
                }
            }
            
            studies.append(study)
        
        logger.info(
            f"Generated {num_studies} studies for patient {patient_id} "
            f"with {len(active_lesions)} active lesions"
        )
        
        return studies
    
    def _decide_change(self, state: Dict) -> str:
        """병변 변화 유형 결정"""
        weights = [0.0, 0.5, 0.2, 0.2, 0.1]  # NEW는 여기서 안 씀
        return random.choices(self.CHANGE_TYPES[1:], weights=weights[1:])[0]
    
    def _apply_change(self, diameter: float, change_type: str) -> float:
        """변화 유형에 따른 크기 조정"""
        if change_type == "Stable":
            return diameter * (1 + random.uniform(-0.08, 0.08))
        elif change_type == "Increased":
            return diameter * (1 + random.uniform(0.15, 0.40))
        elif change_type == "Decreased":
            return diameter * (1 - random.uniform(0.15, 0.40))
        return diameter
    
    def _generate_new_lesion(self, patient_id: str, lesion_num: int) -> Dict:
        """새 병변 생성"""
        diameter = random.uniform(4.0, 15.0)
        return {
            "lesion_id": f"lesion_{patient_id}_{lesion_num}",
            "location": random.choice(self.LOCATION_CODES),
            "diameter_mm": round(diameter, 1),
            "volume_mm3": round(self._diameter_to_volume(diameter), 1),
            "confidence": round(random.uniform(0.75, 0.95), 2)
        }
    
    def _diameter_to_volume(self, diameter_mm: float) -> float:
        """구체 가정 볼륨 계산"""
        radius = diameter_mm / 2
        return (4/3) * np.pi * (radius ** 3)
    
    def generate_dataset(
        self,
        num_patients: int = 10,
        studies_per_patient: int = 3,
        output_dir: Optional[Path] = None
    ) -> List[Dict]:
        """
        전체 synthetic 데이터셋 생성
        
        Args:
            num_patients: 환자 수
            studies_per_patient: 환자당 검사 수
            output_dir: 출력 디렉토리 (None이면 저장 안 함)
            
        Returns:
            모든 환자의 모든 study 리스트
        """
        all_studies = []
        patient_histories = {}
        
        for i in range(num_patients):
            patient_id = f"PT{i+1:05d}"
            
            history = self.generate_patient_history(
                patient_id=patient_id,
                num_studies=studies_per_patient,
                base_date=datetime(2025, 1, 1) + timedelta(days=random.randint(0, 30)),
                interval_days=random.choice([60, 90, 120, 180])
            )
            
            all_studies.extend(history)
            patient_histories[patient_id] = history
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 전체 데이터셋 저장
            with open(output_dir / "all_studies.json", "w") as f:
                json.dump(all_studies, f, indent=2)
            
            # 환자별 히스토리 저장
            histories_dir = output_dir / "patient_histories"
            histories_dir.mkdir(exist_ok=True)
            
            for patient_id, history in patient_histories.items():
                with open(histories_dir / f"{patient_id}.json", "w") as f:
                    json.dump(history, f, indent=2)
            
            # Prior comparison 테이블용 데이터 생성
            comparisons = self._generate_comparison_tables(patient_histories)
            with open(output_dir / "prior_comparisons.json", "w") as f:
                json.dump(comparisons, f, indent=2)
            
            # 통계 저장
            stats = self._compute_statistics(all_studies)
            with open(output_dir / "statistics.json", "w") as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Saved synthetic dataset to {output_dir}")
        
        return all_studies
    
    def _generate_comparison_tables(self, patient_histories: Dict) -> List[Dict]:
        """Prior comparison 테이블 데이터 생성"""
        comparisons = []
        
        for patient_id, studies in patient_histories.items():
            for i in range(1, len(studies)):
                current = studies[i]
                prior = studies[i-1]
                
                comparison = {
                    "patient_id": patient_id,
                    "current_study_uid": current["study_uid"],
                    "current_date": current["study_date"],
                    "prior_study_uid": prior["study_uid"],
                    "prior_date": prior["study_date"],
                    "changes": current["changes"],
                    "interval_days": (
                        datetime.fromisoformat(current["acquisition_datetime"]) -
                        datetime.fromisoformat(prior["acquisition_datetime"])
                    ).days
                }
                comparisons.append(comparison)
        
        return comparisons
    
    def _compute_statistics(self, all_studies: List[Dict]) -> Dict:
        """데이터셋 통계 계산"""
        total_lesions = sum(len(s["lesions"]) for s in all_studies)
        
        change_counts = {"NEW": 0, "Stable": 0, "Increased": 0, "Decreased": 0, "Resolved": 0}
        for study in all_studies:
            for change in study.get("changes", []):
                change_type = change["change"]
                if change_type in change_counts:
                    change_counts[change_type] += 1
        
        diameters = [
            l["diameter_mm"]
            for s in all_studies
            for l in s["lesions"]
        ]
        
        return {
            "num_studies": len(all_studies),
            "num_patients": len(set(s["patient_id"] for s in all_studies)),
            "total_lesions": total_lesions,
            "change_distribution": change_counts,
            "diameter_stats": {
                "min": min(diameters) if diameters else 0,
                "max": max(diameters) if diameters else 0,
                "mean": np.mean(diameters) if diameters else 0,
                "std": np.std(diameters) if diameters else 0
            }
        }


def convert_to_api_format(study: Dict) -> Dict:
    """
    Synthetic study를 API 요청 형식으로 변환
    
    StructuredAIResult 형식에 맞춤
    """
    from api.schemas import (
        StructuredAIResult, QualityMetrics, NoduleCandidate,
        VisionEvidence, StructuredFindings, FindingLabel, ModelVersioning
    )
    
    # Quality metrics
    quality = QualityMetrics(
        slice_thickness_mm=study["quality"]["slice_thickness_mm"],
        coverage_score=study["quality"]["coverage_score"],
        artifact_score=study["quality"]["artifact_score"]
    )
    
    # Nodules
    nodules = []
    for lesion in study["lesions"]:
        nodule = NoduleCandidate(
            id=lesion["lesion_id"],
            center_zyx=(50, 100, 120),  # Placeholder
            bbox_zyx=(45, 95, 115, 55, 105, 125),  # Placeholder
            diameter_mm=lesion["diameter_mm"],
            volume_mm3=lesion["volume_mm3"],
            confidence=lesion["confidence"],
            evidence=VisionEvidence(
                series_uid=f"series.{study['study_uid']}",
                instance_uids=[f"instance.{lesion['lesion_id']}"],
                slice_range=(40, 60),
                confidence=lesion["confidence"]
            ),
            location_code=lesion["location"]
        )
        nodules.append(nodule)
    
    # Default findings (absent)
    findings = StructuredFindings(
        pleural_effusion=FindingLabel(label="absent", probability=0.05),
        pneumothorax=FindingLabel(label="absent", probability=0.02),
        consolidation=FindingLabel(label="absent", probability=0.08),
        atelectasis=FindingLabel(label="absent", probability=0.10),
        emphysema=FindingLabel(label="absent", probability=0.15)
    )
    
    # Versioning
    versioning = ModelVersioning(
        model_version="synthetic-v1.0",
        pipeline_version="1.0.0",
        thresholds={"nodule_detection": 0.7, "nodule_reporting": 0.75}
    )
    
    return StructuredAIResult(
        study_uid=study["study_uid"],
        series_uid=f"series.{study['study_uid']}",
        acquisition_datetime=datetime.fromisoformat(study["acquisition_datetime"]),
        quality=quality,
        lung_volume_ml=5000.0,
        nodules=nodules,
        findings=findings,
        versioning=versioning,
        processing_time_seconds=1.0
    )


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic prior 데이터 생성"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/synthetic_priors"),
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--num-patients", "-n",
        type=int,
        default=10,
        help="생성할 환자 수"
    )
    parser.add_argument(
        "--studies-per-patient", "-s",
        type=int,
        default=3,
        help="환자당 검사 수"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드"
    )
    
    args = parser.parse_args()
    
    generator = SyntheticPriorGenerator(seed=args.seed)
    
    studies = generator.generate_dataset(
        num_patients=args.num_patients,
        studies_per_patient=args.studies_per_patient,
        output_dir=args.output
    )
    
    print(f"\n✅ Generated {len(studies)} studies for {args.num_patients} patients")
    print(f"📁 Output: {args.output}")
    print("\nFiles created:")
    print(f"  - all_studies.json")
    print(f"  - prior_comparisons.json")
    print(f"  - statistics.json")
    print(f"  - patient_histories/PT*.json")


if __name__ == "__main__":
    main()
