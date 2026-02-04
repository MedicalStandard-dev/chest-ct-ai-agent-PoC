# monai_pipeline/nodule_detection.py
"""
Nodule Detection Training/Inference (LIDC-IDRI)
Heatmap-based approach: GTX 1660 optimized
"""
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter, label as scipy_label
from scipy.ndimage import center_of_mass, find_objects

from monai.networks.nets import UNet, SegResNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    RandGaussianNoised, EnsureTyped
)
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import FocalLoss

from api.schemas import NoduleCandidate, VisionEvidence
from utils.logger import logger


class NoduleDetectionTrainer:
    """Nodule detection trainer (Heatmap-based, LIDC format)"""
    
    # GTX 1660 optimized defaults
    DEFAULT_CONFIG = {
        "roi_size": (96, 96, 96),
        "batch_size": 1,
        "num_workers": 2,
        "cache_rate": 0.1,
        "max_epochs": 100,
        "learning_rate": 1e-4,
        "amp_enabled": True,
        "target_spacing": (1.0, 1.0, 1.0),
        "intensity_range": (-1000, 400),
        "gaussian_sigma": 3.0,  # For heatmap generation
        "pos_sample_ratio": 0.7,  # Positive patch sampling ratio
        "early_stopping_patience": 15
    }
    
    def __init__(self, config: Optional[Dict] = None, model_type: str = "unet"):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        
        # Model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Loss (Focal for class imbalance)
        self.loss_fn = FocalLoss(gamma=2.0, reduction="mean")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"]
        )
        
        # AMP
        self.scaler = torch.cuda.amp.GradScaler() if self.config["amp_enabled"] else None
        
        logger.info(f"NoduleDetectionTrainer initialized (model: {model_type}, device: {self.device})")
    
    def _build_model(self) -> nn.Module:
        """Build detection model (outputs heatmap)"""
        if self.model_type == "segresnet":
            return SegResNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,  # Single channel heatmap
                init_filters=16,
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1]
            )
        else:
            return UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128),
                strides=(2, 2, 2),
                num_res_units=2
            )
    
    def _get_train_transforms(self) -> Compose:
        """Training transforms with augmentation"""
        return Compose([
            LoadImaged(keys=["image", "heatmap"]),
            EnsureChannelFirstd(keys=["image", "heatmap"]),
            Orientationd(keys=["image", "heatmap"], axcodes="RAS"),
            Spacingd(
                keys=["image", "heatmap"],
                pixdim=self.config["target_spacing"],
                mode=("bilinear", "nearest")
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.config["intensity_range"][0],
                a_max=self.config["intensity_range"][1],
                b_min=0.0, b_max=1.0,
                clip=True
            ),
            CropForegroundd(keys=["image", "heatmap"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "heatmap"],
                label_key="heatmap",
                spatial_size=self.config["roi_size"],
                pos=self.config["pos_sample_ratio"],
                neg=1 - self.config["pos_sample_ratio"],
                num_samples=2
            ),
            RandFlipd(keys=["image", "heatmap"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "heatmap"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "heatmap"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "heatmap"], prob=0.3, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
            EnsureTyped(keys=["image", "heatmap"])
        ])
    
    def _get_val_transforms(self) -> Compose:
        """Validation transforms (no augmentation)"""
        return Compose([
            LoadImaged(keys=["image", "heatmap"]),
            EnsureChannelFirstd(keys=["image", "heatmap"]),
            Orientationd(keys=["image", "heatmap"], axcodes="RAS"),
            Spacingd(
                keys=["image", "heatmap"],
                pixdim=self.config["target_spacing"],
                mode=("bilinear", "nearest")
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.config["intensity_range"][0],
                a_max=self.config["intensity_range"][1],
                b_min=0.0, b_max=1.0,
                clip=True
            ),
            CropForegroundd(keys=["image", "heatmap"], source_key="image"),
            EnsureTyped(keys=["image", "heatmap"])
        ])
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        for batch_data in train_loader:
            inputs = batch_data["image"].to(self.device)
            targets = batch_data["heatmap"].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.config["amp_enabled"]:
                with torch.cuda.amp.autocast():
                    outputs = torch.sigmoid(self.model(inputs))
                    loss = self.loss_fn(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = torch.sigmoid(self.model(inputs))
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)
    
    def train(
        self,
        train_files: List[Dict],
        val_files: List[Dict],
        output_dir: Path
    ) -> Dict:
        """Full training loop"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Datasets
        train_ds = CacheDataset(
            data=train_files,
            transform=self._get_train_transforms(),
            cache_rate=self.config["cache_rate"],
            num_workers=self.config["num_workers"]
        )
        
        val_ds = CacheDataset(
            data=val_files,
            transform=self._get_val_transforms(),
            cache_rate=self.config["cache_rate"],
            num_workers=self.config["num_workers"]
        ) if val_files else None
        
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"]
        )
        
        best_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": []}
        
        for epoch in range(self.config["max_epochs"]):
            train_loss = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
                torch.save(
                    self.model.state_dict(),
                    output_dir / "best_nodule_det_model.pth"
                )
                logger.info(f"Epoch {epoch+1}: New best loss={train_loss:.4f}")
            else:
                patience_counter += 1
            
            logger.info(
                f"Epoch {epoch+1}/{self.config['max_epochs']} | Loss: {train_loss:.4f}"
            )
            
            if patience_counter >= self.config["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        torch.save(
            self.model.state_dict(),
            output_dir / "final_nodule_det_model.pth"
        )
        
        return {"best_loss": best_loss, "history": history}


class NoduleDetectionInference:
    """Nodule detection inference with postprocessing"""
    
    def __init__(
        self,
        model_path: Path,
        model_type: str = "unet",
        device: Optional[str] = None,
        roi_size: Tuple[int, int, int] = (96, 96, 96),
        detection_threshold: float = 0.5,
        min_diameter_mm: float = 3.0
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.roi_size = roi_size
        self.detection_threshold = detection_threshold
        self.min_diameter_mm = min_diameter_mm
        self.model_type = model_type
        
        # Load model
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"NoduleDetectionInference loaded from {model_path}")
    
    def _build_model(self) -> nn.Module:
        if self.model_type == "segresnet":
            return SegResNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                init_filters=16,
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1]
            )
        else:
            return UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128),
                strides=(2, 2, 2),
                num_res_units=2
            )
    
    def predict_heatmap(self, volume: torch.Tensor) -> np.ndarray:
        """
        Predict nodule probability heatmap
        
        Args:
            volume: (1, 1, D, H, W) tensor, normalized [0, 1]
            
        Returns:
            heatmap: (D, H, W) numpy array, probability [0, 1]
        """
        with torch.no_grad():
            volume = volume.to(self.device)
            
            output = sliding_window_inference(
                volume,
                roi_size=self.roi_size,
                sw_batch_size=1,
                predictor=self.model
            )
            
            heatmap = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        return heatmap
    
    def extract_candidates(
        self,
        heatmap: np.ndarray,
        spacing_mm: Tuple[float, float, float],
        series_uid: str
    ) -> List[NoduleCandidate]:
        """
        Extract nodule candidates from heatmap
        
        Args:
            heatmap: (D, H, W) probability map
            spacing_mm: voxel spacing (z, y, x) in mm
            series_uid: DICOM series UID
            
        Returns:
            List of NoduleCandidate objects
        """
        # Threshold
        binary = (heatmap > self.detection_threshold).astype(np.uint8)
        
        # Connected components
        labeled, num_features = scipy_label(binary)
        
        candidates = []
        for i in range(1, num_features + 1):
            # Get component mask
            component_mask = (labeled == i)
            
            # Find bounding box
            slices = find_objects(labeled == i)
            if not slices:
                continue
            sl = slices[0]
            
            # Center of mass (voxel coordinates)
            com = center_of_mass(component_mask)
            center_zyx = (float(com[0]), float(com[1]), float(com[2]))
            
            # Bounding box (voxel coordinates)
            bbox_zyx = (
                sl[0].start, sl[1].start, sl[2].start,
                sl[0].stop, sl[1].stop, sl[2].stop
            )
            
            # Diameter estimation (longest axis in mm)
            extent_z = (sl[0].stop - sl[0].start) * spacing_mm[0]
            extent_y = (sl[1].stop - sl[1].start) * spacing_mm[1]
            extent_x = (sl[2].stop - sl[2].start) * spacing_mm[2]
            diameter_mm = max(extent_z, extent_y, extent_x)
            
            # Skip too small
            if diameter_mm < self.min_diameter_mm:
                continue
            
            # Volume estimation
            voxel_count = component_mask.sum()
            voxel_volume = spacing_mm[0] * spacing_mm[1] * spacing_mm[2]
            volume_mm3 = float(voxel_count * voxel_volume)
            
            # Confidence (max probability in component)
            confidence = float(heatmap[component_mask].max())
            
            # Slice range for evidence
            slice_range = (int(sl[0].start), int(sl[0].stop))
            
            candidate = NoduleCandidate(
                id=f"N{len(candidates)+1}",
                center_zyx=center_zyx,
                bbox_zyx=bbox_zyx,
                diameter_mm=round(diameter_mm, 1),
                volume_mm3=round(volume_mm3, 1),
                confidence=round(confidence, 3),
                evidence=VisionEvidence(
                    series_uid=series_uid,
                    instance_uids=[],  # TODO: map to actual instance UIDs
                    slice_range=slice_range,
                    confidence=round(confidence, 3)
                ),
                location_code=self._estimate_location(center_zyx, heatmap.shape)
            )
            
            candidates.append(candidate)
        
        # Sort by confidence
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Extracted {len(candidates)} nodule candidates")
        return candidates
    
    def _estimate_location(
        self,
        center_zyx: Tuple[float, float, float],
        volume_shape: Tuple[int, int, int]
    ) -> str:
        """
        Estimate lung location code (RUL/RML/RLL/LUL/LLL)
        
        Simple heuristic based on position:
        - Left/Right: x position (< 50% = Right, >= 50% = Left)
        - Upper/Middle/Lower: z position (thirds)
        """
        d, h, w = volume_shape
        z, y, x = center_zyx
        
        # Left/Right (assumes RAS orientation)
        side = "L" if x >= w / 2 else "R"
        
        # Upper/Middle/Lower
        z_ratio = z / d
        if z_ratio < 0.33:
            level = "LL"  # Lower lobe
        elif z_ratio < 0.66:
            level = "ML" if side == "R" else "UL"  # Middle (R only) or Upper
        else:
            level = "UL"  # Upper lobe
        
        return f"{side}{level}"
    
    def detect(
        self,
        volume: torch.Tensor,
        spacing_mm: Tuple[float, float, float],
        series_uid: str
    ) -> List[NoduleCandidate]:
        """
        Full detection pipeline
        
        Args:
            volume: (1, 1, D, H, W) normalized CT volume
            spacing_mm: voxel spacing in mm
            series_uid: DICOM series UID
            
        Returns:
            List of NoduleCandidate objects
        """
        heatmap = self.predict_heatmap(volume)
        return self.extract_candidates(heatmap, spacing_mm, series_uid)


def generate_heatmap_label(
    volume_shape: Tuple[int, int, int],
    nodule_centers: List[Tuple[float, float, float]],
    sigma: float = 3.0
) -> np.ndarray:
    """
    Generate 3D Gaussian heatmap label from nodule centers
    
    Args:
        volume_shape: (D, H, W) shape
        nodule_centers: List of (z, y, x) coordinates
        sigma: Gaussian sigma
        
    Returns:
        heatmap: (D, H, W) array with Gaussian blobs at centers
    """
    heatmap = np.zeros(volume_shape, dtype=np.float32)
    
    for center in nodule_centers:
        z, y, x = int(center[0]), int(center[1]), int(center[2])
        
        # Bounds check
        if not (0 <= z < volume_shape[0] and
                0 <= y < volume_shape[1] and
                0 <= x < volume_shape[2]):
            continue
        
        # Place point
        heatmap[z, y, x] = 1.0
    
    # Apply Gaussian filter
    if len(nodule_centers) > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        heatmap = heatmap / (heatmap.max() + 1e-8)  # Normalize to [0, 1]
    
    return heatmap
