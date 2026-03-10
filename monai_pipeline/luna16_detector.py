# monai_pipeline/luna16_detector.py
"""
MONAI lung_nodule_ct_detection (LUNA16 pretrained RetinaNet) wrapper.

- 아키텍처: ResNet50 + FPN + RetinaNet (3D)
- 학습: LUNA16 (mAP=0.852 @ fold0)
- 입력: numpy (D,H,W) HU 값 + spacing
- 출력: List[NoduleCandidate]
"""
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="monai")

import numpy as np
import torch
from scipy.ndimage import zoom as scipy_zoom

from monai.networks.nets.resnet import resnet50
from monai.apps.detection.networks.retinanet_network import (
    resnet_fpn_feature_extractor,
    RetinaNet,
)
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape

from api.schemas import NoduleCandidate, VisionEvidence
from utils.logger import logger


class Luna16Detector:
    """
    MONAI lung_nodule_ct_detection pretrained RetinaNet wrapper.

    bundle_dir 예시:
        models/luna16_retinanet/lung_nodule_ct_detection
    """

    # 학습 시 사용한 voxel spacing (XYZ 순서)
    BUNDLE_SPACING_XYZ = (0.703125, 0.703125, 1.25)

    # HU 범위 (학습 설정과 동일)
    HU_MIN, HU_MAX = -1024.0, 300.0

    def __init__(
        self,
        bundle_dir: str,
        device: str = "auto",
        score_thresh: float = 0.02,
        nms_thresh: float = 0.22,
        amp: bool = True,
    ):
        self.bundle_dir = Path(bundle_dir)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.amp = amp

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Luna16Detector: initializing on device={self.device}, amp={self.amp}")

        self.detector = self._build_detector()
        self._load_weights()
        self.detector.eval()
        logger.info("Luna16Detector: ready")

    # ─── 모델 구성 ────────────────────────────────────────────────────────────

    def _build_detector(self) -> RetinaNetDetector:
        """inference.json과 동일한 설정으로 RetinaNet 구성"""
        backbone = resnet50(
            spatial_dims=3,
            n_input_channels=1,
            conv1_t_stride=[2, 2, 1],
            conv1_t_size=[7, 7, 7],
        )

        feature_extractor = resnet_fpn_feature_extractor(
            backbone=backbone,
            spatial_dims=3,
            pretrained_backbone=False,
            returned_layers=(1, 2),   # inference.json: [1, 2]
            trainable_backbone_layers=None,
        )

        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=[1, 2, 4],
            base_anchor_shapes=[[6, 8, 4], [8, 6, 5], [10, 10, 6]],
        )

        network = RetinaNet(
            spatial_dims=3,
            num_classes=1,
            num_anchors=3,
            feature_extractor=feature_extractor,
            size_divisible=[16, 16, 8],
        ).to(self.device)

        detector = RetinaNetDetector(
            network=network,
            anchor_generator=anchor_generator,
            debug=False,
            spatial_dims=3,
            num_classes=1,
            size_divisible=[16, 16, 8],
        )

        detector.set_target_keys(box_key="box", label_key="label")
        detector.set_box_selector_parameters(
            score_thresh=self.score_thresh,
            topk_candidates_per_level=1000,
            nms_thresh=self.nms_thresh,
            detections_per_img=300,
        )
        # sliding window device='cpu': 8GB VRAM 보호
        detector.set_sliding_window_inferer(
            roi_size=[512, 512, 192],
            overlap=0.25,
            sw_batch_size=1,
            mode="constant",
            device="cpu",
        )

        return detector

    def _load_weights(self):
        """사전학습 가중치 로드"""
        model_pt = self.bundle_dir / "models" / "model.pt"
        if not model_pt.exists():
            raise FileNotFoundError(f"Model weights not found: {model_pt}")

        checkpoint = torch.load(model_pt, map_location=self.device, weights_only=True)
        self.detector.network.load_state_dict(checkpoint)
        logger.info(f"Luna16Detector: weights loaded from {model_pt}")

    # ─── 전처리 ───────────────────────────────────────────────────────────────

    def _resample(
        self,
        volume: np.ndarray,
        spacing_zyx: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        볼륨을 BUNDLE_SPACING으로 리샘플.

        Returns:
            (resampled_volume, zoom_factors_zyx)
        """
        target_zyx = (
            self.BUNDLE_SPACING_XYZ[2],   # Z
            self.BUNDLE_SPACING_XYZ[1],   # Y
            self.BUNDLE_SPACING_XYZ[0],   # X
        )
        zoom_factors = tuple(s / t for s, t in zip(spacing_zyx, target_zyx))

        if all(abs(z - 1.0) < 0.01 for z in zoom_factors):
            return volume, (1.0, 1.0, 1.0)

        resampled = scipy_zoom(volume, zoom_factors, order=1, prefilter=False)
        logger.info(
            f"Luna16Detector: resampled {volume.shape} → {resampled.shape} "
            f"(zoom={tuple(f'{z:.3f}' for z in zoom_factors)})"
        )
        return resampled, zoom_factors

    def _preprocess(self, volume: np.ndarray) -> torch.Tensor:
        """HU 클리핑 → [0,1] 정규화 → tensor [1, D, H, W]

        이미 [0,1]로 정규화된 데이터(LIDC preprocessed)는 정규화 생략.
        raw HU 값(-1024~300 범위)이면 클리핑 + 정규화 적용.
        """
        if volume.min() >= -1.0 and volume.max() <= 1.0 + 1e-3:
            # 이미 정규화된 데이터 — 그대로 사용
            vol = volume.astype(np.float32)
        else:
            # Raw HU → [0, 1]
            vol = np.clip(volume, self.HU_MIN, self.HU_MAX)
            vol = (vol - self.HU_MIN) / (self.HU_MAX - self.HU_MIN)
            vol = vol.astype(np.float32)
        return torch.from_numpy(vol[np.newaxis]).float()

    # ─── 후처리 ───────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_location(center_zyx: Tuple[float, float, float], volume_shape: Tuple[int, int, int]) -> str:
        """ZYX 좌표 → 폐 구역 코드 (RUL/RML/RLL/LUL/LLL)"""
        d, h, w = volume_shape
        z, y, x = center_zyx
        side = "L" if x >= w / 2 else "R"
        z_ratio = z / d
        if z_ratio < 0.33:
            level = "LL"
        elif z_ratio < 0.66:
            level = "ML" if side == "R" else "UL"
        else:
            level = "UL"
        return f"{side}{level}"

    def _boxes_to_candidates(
        self,
        boxes: np.ndarray,          # [N, 6] xyzxyz (resampled voxel space)
        scores: np.ndarray,         # [N]
        zoom_factors_zyx: Tuple[float, float, float],
        orig_shape: Tuple[int, int, int],
        series_uid: str,
    ) -> List[NoduleCandidate]:
        """3D bounding box → NoduleCandidate 리스트"""
        zoom_z, zoom_y, zoom_x = zoom_factors_zyx
        candidates = []

        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, z1, x2, y2, z2 = box

            # 중심 (원본 ZYX 공간)
            orig_z = ((z1 + z2) / 2) / zoom_z
            orig_y = ((y1 + y2) / 2) / zoom_y
            orig_x = ((x1 + x2) / 2) / zoom_x

            # 크기 (mm)
            width_mm  = (x2 - x1) * self.BUNDLE_SPACING_XYZ[0]
            height_mm = (y2 - y1) * self.BUNDLE_SPACING_XYZ[1]
            depth_mm  = (z2 - z1) * self.BUNDLE_SPACING_XYZ[2]
            diameter_mm = float(max(width_mm, height_mm, depth_mm))
            volume_mm3  = float((np.pi / 6) * width_mm * height_mm * depth_mm)

            # Slice range (원본 Z)
            slice_z1 = max(0, int(z1 / zoom_z))
            slice_z2 = min(orig_shape[0] - 1, int(np.ceil(z2 / zoom_z)))

            # BBox (원본 ZYX)
            bz1 = max(0, int(z1 / zoom_z))
            bz2 = min(orig_shape[0], int(np.ceil(z2 / zoom_z)))
            by1 = max(0, int(y1 / zoom_y))
            by2 = min(orig_shape[1], int(np.ceil(y2 / zoom_y)))
            bx1 = max(0, int(x1 / zoom_x))
            bx2 = min(orig_shape[2], int(np.ceil(x2 / zoom_x)))

            location_code = self._estimate_location((orig_z, orig_y, orig_x), orig_shape)

            evidence = VisionEvidence(
                series_uid=series_uid,
                instance_uids=[],
                slice_range=(slice_z1, slice_z2),
                mask_path=None,
                confidence=float(score),
            )

            candidate = NoduleCandidate(
                id=f"N{i + 1}",
                center_zyx=(orig_z, orig_y, orig_x),
                bbox_zyx=(bz1, by1, bx1, bz2, by2, bx2),
                diameter_mm=round(diameter_mm, 1),
                volume_mm3=round(volume_mm3, 1),
                confidence=round(float(score), 3),
                evidence=evidence,
                location_code=location_code,
                characteristics=None,
            )
            candidates.append(candidate)

        return candidates

    # ─── 메인 인터페이스 ──────────────────────────────────────────────────────

    def detect(
        self,
        volume_zyx: np.ndarray,
        spacing_mm: Tuple[float, float, float],
        series_uid: str,
        lung_mask: Optional[np.ndarray] = None,
        score_thresh: Optional[float] = None,
    ) -> List[NoduleCandidate]:
        """
        CT 볼륨에서 결절 후보 검출.

        Args:
            volume_zyx: (D, H, W) HU 값 numpy array
            spacing_mm: (sz, sy, sx) voxel spacing
            series_uid: 시리즈 UID
            lung_mask: 폐 마스크 (optional, 현재 미사용)
            score_thresh: 점수 임계값 (None = self.score_thresh)

        Returns:
            List[NoduleCandidate] (score 내림차순)
        """
        thresh = score_thresh if score_thresh is not None else self.score_thresh
        orig_shape = volume_zyx.shape

        # 1. 리샘플
        resampled, zoom_factors = self._resample(volume_zyx, spacing_mm)

        # 2. 전처리
        tensor = self._preprocess(resampled)  # [1, D', H', W']

        # 3. 추론
        with torch.no_grad():
            if self.amp and self.device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    outputs = self.detector([tensor.to(self.device)], use_inferer=True)
            else:
                outputs = self.detector([tensor.to(self.device)], use_inferer=True)

        if not outputs:
            logger.info("Luna16Detector: no detections")
            return []

        pred = outputs[0]
        boxes  = pred.get("box",          torch.zeros(0, 6)).cpu().numpy()
        scores = pred.get("label_scores", torch.zeros(0)).cpu().numpy()

        # 4. Score 필터링
        mask   = scores >= thresh
        boxes  = boxes[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            logger.info(f"Luna16Detector: 0 candidates (score_thresh={thresh:.3f})")
            return []

        # 5. 정렬 (confidence 내림차순)
        order  = np.argsort(-scores)
        boxes  = boxes[order]
        scores = scores[order]

        logger.info(f"Luna16Detector: {len(boxes)} candidates (score_thresh={thresh:.3f})")

        # 6. NoduleCandidate 변환
        return self._boxes_to_candidates(boxes, scores, zoom_factors, orig_shape, series_uid)
