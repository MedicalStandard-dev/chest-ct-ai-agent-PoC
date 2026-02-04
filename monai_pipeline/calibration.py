# monai_pipeline/calibration.py
"""
Probability calibration for model outputs
실전 환경에서는 temperature scaling, Platt scaling 등 적용
"""
from typing import Dict, List
import numpy as np


class ProbabilityCalibrator:
    """확률 보정 인터페이스"""
    
    def __init__(self, calibration_params: Dict[str, float] = None):
        """
        Args:
            calibration_params: 모델별 보정 파라미터
                예: {"nodule_detector": {"temperature": 1.2}}
        """
        self.calibration_params = calibration_params or {}
    
    def calibrate_nodule_confidence(self, raw_confidence: float, model_name: str = "default") -> float:
        """
        Nodule detection confidence 보정
        
        실전에서는 validation set에서 학습한 calibration curve 적용
        """
        params = self.calibration_params.get(model_name, {})
        temperature = params.get("temperature", 1.0)
        
        # Simple temperature scaling
        calibrated = self._temperature_scale(raw_confidence, temperature)
        
        return float(np.clip(calibrated, 0.0, 1.0))
    
    def calibrate_finding_probability(self, raw_prob: float, finding_type: str) -> float:
        """
        Multi-label finding probability 보정
        """
        params = self.calibration_params.get(finding_type, {})
        temperature = params.get("temperature", 1.0)
        
        calibrated = self._temperature_scale(raw_prob, temperature)
        
        return float(np.clip(calibrated, 0.0, 1.0))
    
    def _temperature_scale(self, prob: float, temperature: float) -> float:
        """Temperature scaling for calibration"""
        # Convert to logits, scale, convert back
        epsilon = 1e-7
        prob = np.clip(prob, epsilon, 1 - epsilon)
        logit = np.log(prob / (1 - prob))
        scaled_logit = logit / temperature
        scaled_prob = 1 / (1 + np.exp(-scaled_logit))
        return scaled_prob
    
    @classmethod
    def from_validation_data(cls, validation_results: Dict) -> "ProbabilityCalibrator":
        """
        Validation data로부터 calibration 파라미터 학습
        
        TODO: 실제 구현 시 temperature/Platt scaling 학습
        """
        # Placeholder: 실제로는 validation set의 
        # (predictions, ground_truth)로 최적 temperature 계산
        calibration_params = {
            "nodule_detector": {"temperature": 1.0},
            "pleural_effusion": {"temperature": 1.0},
            "pneumothorax": {"temperature": 1.0}
        }
        
        return cls(calibration_params)


class ThresholdManager:
    """Confidence threshold 관리"""
    
    DEFAULT_THRESHOLDS = {
        "nodule_detection": 0.7,
        "nodule_reporting": 0.75,  # Findings에 포함할 최소 confidence
        "pleural_effusion": 0.6,
        "pneumothorax": 0.8,  # High specificity needed
        "consolidation": 0.6,
        "atelectasis": 0.6,
        "emphysema": 0.6
    }
    
    def __init__(self, custom_thresholds: Dict[str, float] = None):
        self.thresholds = {**self.DEFAULT_THRESHOLDS}
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)
    
    def get_threshold(self, item_type: str) -> float:
        """특정 항목의 threshold 반환"""
        return self.thresholds.get(item_type, 0.5)
    
    def should_report_nodule(self, confidence: float) -> bool:
        """Nodule을 Findings에 포함할지 결정"""
        return confidence >= self.thresholds["nodule_reporting"]
    
    def should_include_in_limitations(self, confidence: float) -> bool:
        """Low confidence nodule을 limitations에 언급할지"""
        detection_threshold = self.thresholds["nodule_detection"]
        reporting_threshold = self.thresholds["nodule_reporting"]
        
        return detection_threshold <= confidence < reporting_threshold
    
    def get_all_thresholds(self) -> Dict[str, float]:
        """모든 threshold 반환 (versioning용)"""
        return self.thresholds.copy()
