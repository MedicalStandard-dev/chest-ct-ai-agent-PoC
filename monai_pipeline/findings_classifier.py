# monai_pipeline/findings_classifier.py
"""
Multi-label findings classifier interface
실제 모델은 추후 교체 가능 (abstract)
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from api.schemas import StructuredFindings, FindingLabel, VisionEvidence
from utils.logger import logger


class FindingsClassifierInterface(ABC):
    """Findings classifier abstract interface"""
    
    @abstractmethod
    def predict(self, volume: torch.Tensor, metadata: Dict) -> StructuredFindings:
        """
        Multi-label classification
        
        Returns:
            StructuredFindings with evidence
        """
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        pass


class MockFindingsClassifier(FindingsClassifierInterface):
    """Mock classifier for testing"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.version = "mock-v1.0"
        np.random.seed(seed)
    
    def predict(self, volume: torch.Tensor, metadata: Dict) -> StructuredFindings:
        """Generate mock findings with evidence"""
        
        series_uid = metadata.get("series_uid", "mock.series.uid")
        
        # Mock probabilities
        findings_probs = {
            "pleural_effusion": np.random.random(),
            "pneumothorax": np.random.random(),
            "consolidation": np.random.random(),
            "atelectasis": np.random.random(),
            "emphysema": np.random.random()
        }
        
        # Convert to labels
        def _to_label(prob: float) -> str:
            if prob > 0.7:
                return "present"
            elif prob < 0.3:
                return "absent"
            else:
                return "uncertain"
        
        # Create evidence for positive findings
        def _create_evidence(finding_name: str, prob: float) -> VisionEvidence:
            if prob > 0.5:
                return VisionEvidence(
                    series_uid=series_uid,
                    instance_uids=[f"instance.{finding_name}.1"],
                    slice_range=(10, 50),
                    confidence=float(prob)
                )
            return None
        
        findings = StructuredFindings(
            pleural_effusion=FindingLabel(
                label=_to_label(findings_probs["pleural_effusion"]),
                probability=findings_probs["pleural_effusion"],
                evidence=_create_evidence("effusion", findings_probs["pleural_effusion"])
            ),
            pneumothorax=FindingLabel(
                label=_to_label(findings_probs["pneumothorax"]),
                probability=findings_probs["pneumothorax"],
                evidence=_create_evidence("pneumothorax", findings_probs["pneumothorax"])
            ),
            consolidation=FindingLabel(
                label=_to_label(findings_probs["consolidation"]),
                probability=findings_probs["consolidation"],
                evidence=_create_evidence("consolidation", findings_probs["consolidation"])
            ),
            atelectasis=FindingLabel(
                label=_to_label(findings_probs["atelectasis"]),
                probability=findings_probs["atelectasis"],
                evidence=_create_evidence("atelectasis", findings_probs["atelectasis"])
            ),
            emphysema=FindingLabel(
                label=_to_label(findings_probs["emphysema"]),
                probability=findings_probs["emphysema"],
                evidence=_create_evidence("emphysema", findings_probs["emphysema"])
            )
        )
        
        return findings
    
    def get_version(self) -> str:
        return self.version


class RuleBasedFindingsClassifier(FindingsClassifierInterface):
    """
    Rule-based findings classifier using CT volume intensity features.

    별도 학습 데이터 없이 HU 분포 + 결절 탐지 결과를 활용한 rule-based 판단.
    """

    PRESENT_THRESHOLD = 0.6
    ABSENT_THRESHOLD = 0.3

    def __init__(self):
        self.version = "rule-based-v1.0"

    def predict(
        self,
        volume: torch.Tensor,
        metadata: Dict,
        volume_hu: Optional[np.ndarray] = None,
        spacing: Optional[Tuple[float, float, float]] = None,
        lung_mask: Optional[np.ndarray] = None,
        nodule_candidates: Optional[List[Dict]] = None,
    ) -> StructuredFindings:
        """
        Rule-based multi-label classification.

        Args:
            volume: torch.Tensor (unused if volume_hu provided, kept for interface compat)
            metadata: dict with series_uid etc.
            volume_hu: numpy CT volume in HU scale. If None, derived from volume.
            spacing: voxel spacing (z, y, x) in mm
            lung_mask: binary lung mask, same shape as volume_hu
            nodule_candidates: list of nodule candidate dicts
        """
        # Obtain HU volume
        hu = self._ensure_hu(volume, volume_hu)

        if lung_mask is not None and lung_mask.shape != hu.shape:
            logger.warning("lung_mask shape mismatch, ignoring mask")
            lung_mask = None

        # Compute per-finding probabilities
        probs = {
            "pleural_effusion": self._detect_pleural_effusion(hu, lung_mask),
            "pneumothorax": self._detect_pneumothorax(hu, lung_mask),
            "consolidation": self._detect_consolidation(hu, lung_mask),
            "atelectasis": self._detect_atelectasis(hu, lung_mask),
            "emphysema": self._detect_emphysema(hu, lung_mask),
        }

        findings = StructuredFindings(
            pleural_effusion=FindingLabel(
                label=self._to_label(probs["pleural_effusion"]),
                probability=probs["pleural_effusion"],
                evidence=None,
            ),
            pneumothorax=FindingLabel(
                label=self._to_label(probs["pneumothorax"]),
                probability=probs["pneumothorax"],
                evidence=None,
            ),
            consolidation=FindingLabel(
                label=self._to_label(probs["consolidation"]),
                probability=probs["consolidation"],
                evidence=None,
            ),
            atelectasis=FindingLabel(
                label=self._to_label(probs["atelectasis"]),
                probability=probs["atelectasis"],
                evidence=None,
            ),
            emphysema=FindingLabel(
                label=self._to_label(probs["emphysema"]),
                probability=probs["emphysema"],
                evidence=None,
            ),
        )

        logger.info(
            f"RuleBasedFindings: "
            + ", ".join(f"{k}={v:.2f}({self._to_label(v)})" for k, v in probs.items())
        )

        return findings

    def get_version(self) -> str:
        return self.version

    # ── helpers ──────────────────────────────────────────────

    def _ensure_hu(
        self, volume: torch.Tensor, volume_hu: Optional[np.ndarray]
    ) -> np.ndarray:
        """Get HU-scale numpy volume. Convert from [0,1] if needed."""
        if volume_hu is not None:
            vol = np.asarray(volume_hu, dtype=np.float32)
        else:
            vol = volume.detach().cpu().numpy() if isinstance(volume, torch.Tensor) else np.asarray(volume)
            # squeeze batch/channel dims
            while vol.ndim > 3:
                vol = vol[0]
            vol = vol.astype(np.float32)

        v_min, v_max = float(vol.min()), float(vol.max())
        if v_min >= -1e-3 and v_max <= 1.0 + 1e-3:
            # normalized [0,1] → HU: val * 1400 - 1000
            vol = vol * 1400.0 - 1000.0
        return vol

    def _to_label(self, prob: float) -> str:
        if prob >= self.PRESENT_THRESHOLD:
            return "present"
        elif prob <= self.ABSENT_THRESHOLD:
            return "absent"
        return "uncertain"

    def _lung_voxels(self, hu: np.ndarray, lung_mask: Optional[np.ndarray]) -> np.ndarray:
        """Return HU values inside the lung region."""
        if lung_mask is not None:
            return hu[lung_mask > 0]
        # fallback: typical lung HU range
        return hu[(hu > -1050) & (hu < 0)]

    # ── per-finding rules ────────────────────────────────────

    def _detect_pleural_effusion(
        self, hu: np.ndarray, lung_mask: Optional[np.ndarray]
    ) -> float:
        """
        Pleural effusion: 폐 하부 의존부위에 액체 밀도(HU 0~30) 영역 비율.
        하부 25% 슬라이스에서 폐 마스크 내부/경계의 해당 HU 범위 비율을 계산.
        """
        depth = hu.shape[0]
        lower_start = int(depth * 0.75)
        lower_region = hu[lower_start:]

        if lung_mask is not None:
            lower_lung = lung_mask[lower_start:]
            # 폐 내부에서 fluid density 탐색
            lung_voxels = lower_lung > 0
            total_voxels = float(lung_voxels.sum())
            if total_voxels == 0:
                return 0.0
            fluid_mask = lung_voxels & (lower_region >= -10) & (lower_region <= 40)
            fluid_ratio = float(fluid_mask.sum()) / total_voxels
        else:
            # lung mask 없으면 approximate lung region (exclude bone/body)
            lung_approx = (lower_region > -1050) & (lower_region < 100)
            total_voxels = float(lung_approx.sum())
            if total_voxels == 0:
                return 0.0
            fluid_mask = lung_approx & (lower_region >= -10) & (lower_region <= 40)
            fluid_ratio = float(fluid_mask.sum()) / total_voxels

        # Map ratio to probability: 0.05 → 0.3, 0.15 → 0.7, 0.25+ → 0.9
        prob = np.clip((fluid_ratio - 0.02) / 0.20, 0.0, 1.0)
        return float(prob)

    def _detect_pneumothorax(
        self, hu: np.ndarray, lung_mask: Optional[np.ndarray]
    ) -> float:
        """
        Pneumothorax: 폐 상부에서 체내(body) 영역 중 폐 마스크 외부에
        매우 낮은 밀도(HU < -950, 공기) 영역이 존재하는 비율.
        배경 공기(body 외부)는 제외해야 오탐 방지.
        """
        depth = hu.shape[0]
        upper_end = max(1, int(depth * 0.35))
        upper_region = hu[:upper_end]

        very_low = upper_region < -950
        # body mask: HU > -900 (exclude background air)
        body_upper = upper_region > -900

        if lung_mask is not None:
            upper_lung = lung_mask[:upper_end]
            # pneumothorax: body 내부이면서 폐 마스크 바깥인 곳에 air
            body_outside_lung = body_upper & (upper_lung == 0)
            total = float(body_outside_lung.sum())
            if total == 0:
                return 0.0
            pneumo_voxels = very_low & body_outside_lung
            ratio = float(pneumo_voxels.sum()) / total
        else:
            # lung mask 없으면 body 내 매우 낮은 밀도 비율 (보수적)
            body_total = float(body_upper.sum())
            if body_total == 0:
                return 0.0
            # body 내에서 air pocket 탐색
            ratio = float((very_low & body_upper).sum()) / body_total

        # air-in-pleural-space가 보통 작은 비율이므로 민감하게 설정
        prob = np.clip((ratio - 0.01) / 0.08, 0.0, 1.0)
        return float(prob)

    def _detect_consolidation(
        self, hu: np.ndarray, lung_mask: Optional[np.ndarray]
    ) -> float:
        """
        Consolidation: 폐 실질 내 높은 밀도(HU > -100) 영역의 연결된 큰 덩어리.
        반드시 폐 마스크 내부에서만 평가해야 body/bone 오탐을 방지.
        """
        from scipy.ndimage import label as scipy_label

        if lung_mask is not None:
            # 폐 내부에서만 높은 밀도 탐색
            high_density = (lung_mask > 0) & (hu > -100)
            lung_total = float(lung_mask.sum())
        else:
            # Approximate: 폐 영역은 -1050 ~ -200 범위, consolidation은 그 안에서 > -100
            # 먼저 rough lung region을 구하고, 그 안에서 고밀도 탐색
            lung_approx = (hu > -1050) & (hu < -200)
            # erode 없이, lung 범위 내 고밀도를 찾기 위해 범위 확장
            high_density = (hu > -1050) & (hu > -100) & (hu < 100)
            lung_total = float(lung_approx.sum())

        if high_density.sum() == 0 or lung_total == 0:
            return 0.0

        labeled, n_components = scipy_label(high_density.astype(np.uint8))
        if n_components == 0:
            return 0.0

        comp_sizes = np.bincount(labeled.ravel())
        comp_sizes[0] = 0  # background
        max_component = float(comp_sizes.max())

        ratio = max_component / lung_total

        # Map: 0.02 → 0.3, 0.08 → 0.7, 0.15+ → 0.95
        prob = np.clip((ratio - 0.01) / 0.12, 0.0, 1.0)
        return float(prob)

    def _detect_atelectasis(
        self, hu: np.ndarray, lung_mask: Optional[np.ndarray]
    ) -> float:
        """
        Atelectasis: 좌우 폐 볼륨 비대칭(>20%) + 해당 영역 밀도 증가.
        """
        mid_x = hu.shape[2] // 2

        if lung_mask is not None:
            left_vol = float(lung_mask[:, :, :mid_x].sum())
            right_vol = float(lung_mask[:, :, mid_x:].sum())
        else:
            # Approximate lung region
            lung_approx = (hu > -1050) & (hu < -200)
            left_vol = float(lung_approx[:, :, :mid_x].sum())
            right_vol = float(lung_approx[:, :, mid_x:].sum())

        if left_vol + right_vol == 0:
            return 0.0

        total_vol = left_vol + right_vol
        asymmetry = abs(left_vol - right_vol) / total_vol

        # 비대칭 폐에서 밀도 증가 확인
        smaller_side = "left" if left_vol < right_vol else "right"
        if smaller_side == "left":
            side_hu = hu[:, :, :mid_x]
        else:
            side_hu = hu[:, :, mid_x:]

        # 밀도 증가: HU > -300 (정상 폐는 -700 ~ -500)
        dense_ratio = float((side_hu > -300).sum()) / max(1.0, float(side_hu.size))

        # 비대칭 + 밀도증가 결합
        if asymmetry < 0.15:
            return float(np.clip(asymmetry * dense_ratio * 5, 0.0, 0.25))

        combined = asymmetry * 0.6 + dense_ratio * 0.4
        prob = np.clip((combined - 0.10) / 0.30, 0.0, 1.0)
        return float(prob)

    def _detect_emphysema(
        self, hu: np.ndarray, lung_mask: Optional[np.ndarray]
    ) -> float:
        """
        Emphysema: LAA-950 (Low Attenuation Area).
        폐 내 HU < -950 voxel 비율. 15% 이상이면 present.
        """
        lung_vals = self._lung_voxels(hu, lung_mask)

        if lung_vals.size == 0:
            return 0.0

        laa950 = float((lung_vals < -950).sum()) / float(lung_vals.size)

        # Map: 5% → 0.2, 10% → 0.4, 15% → 0.65, 25%+ → 0.95
        prob = np.clip((laa950 - 0.03) / 0.22, 0.0, 1.0)
        return float(prob)


class ProductionFindingsClassifier(FindingsClassifierInterface):
    """Production classifier (placeholder for real model)"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model_path = model_path
        self.version = "production-v1.0"
        # TODO: Load actual model
        # self.model = load_model(model_path)
    
    def predict(self, volume: torch.Tensor, metadata: Dict) -> StructuredFindings:
        """
        Real model inference
        
        TODO: Implement actual model forward pass
        """
        raise NotImplementedError("Production model not yet implemented")
    
    def get_version(self) -> str:
        return self.version
