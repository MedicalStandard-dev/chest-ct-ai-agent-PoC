# monai_pipeline/lung_segmentation.py
"""
Lung Segmentation Training/Inference (MSD Task06_Lung)
GTX 1660 optimized: patch-based, AMP, low batch size
"""
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import torch
import numpy as np

from monai.networks.nets import DynUNet
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    EnsureTyped
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference

from utils.logger import logger


class LungSegmentationTrainer:
    """Lung segmentation trainer (DynUNet, MSD format)"""
    
    # GTX 1660 optimized defaults
    DEFAULT_CONFIG = {
        "roi_size": (96, 96, 96),
        "batch_size": 1,
        "num_workers": 2,
        "cache_rate": 0.1,
        "max_epochs": 100,
        "learning_rate": 1e-4,
        "amp_enabled": True,
        "target_spacing": (1.5, 1.5, 2.0),
        "intensity_range": (-1000, 400),
        "early_stopping_patience": 10
    }
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Loss and optimizer (DiceFocalLoss for extreme class imbalance)
        self.loss_fn = DiceFocalLoss(
            to_onehot_y=True, 
            softmax=True,
            gamma=2.0,  # Focal loss gamma (higher = more focus on hard examples)
            lambda_dice=1.0,
            lambda_focal=1.0
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config["learning_rate"]
        )
        
        # Metrics
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        # AMP
        self.scaler = torch.cuda.amp.GradScaler() if self.config["amp_enabled"] else None
        
        logger.info(f"LungSegmentationTrainer initialized (device: {self.device})")
    
    def _build_model(self) -> DynUNet:
        """Build DynUNet for lung segmentation"""
        kernels = [[3, 3, 3]] * 5
        strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        
        return DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,  # background + lung
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",
            deep_supervision=False  # Simplified for training
        )
    
    def _get_train_transforms(self) -> Compose:
        """Training transforms with augmentation"""
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
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
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=self.config["roi_size"],
                pos=4, neg=1,  # 4:1 ratio - focus on foreground
                num_samples=4,  # More samples per volume
                allow_smaller=True  # Handle small tumors
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            EnsureTyped(keys=["image", "label"])
        ])
    
    def _get_val_transforms(self) -> Compose:
        """Validation transforms (no augmentation)"""
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
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
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"])
        ])
    
    def prepare_data(
        self,
        data_dir: Path,
        val_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare MSD format data loaders"""
        images_dir = data_dir / "imagesTr"
        labels_dir = data_dir / "labelsTr"
        
        # Build file list
        image_files = sorted(images_dir.glob("*.nii.gz"))
        data_dicts = []
        for img_path in image_files:
            label_path = labels_dir / img_path.name
            if label_path.exists():
                data_dicts.append({
                    "image": str(img_path),
                    "label": str(label_path)
                })
        
        logger.info(f"Found {len(data_dicts)} training samples")
        
        # Split
        n_val = int(len(data_dicts) * val_split)
        train_files = data_dicts[:-n_val] if n_val > 0 else data_dicts
        val_files = data_dicts[-n_val:] if n_val > 0 else []
        
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
        
        # Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"]
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.config["num_workers"]
        ) if val_ds else None
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        for batch_data in train_loader:
            inputs = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.config["amp_enabled"]:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate and return mean Dice"""
        self.model.eval()
        self.dice_metric.reset()
        
        with torch.no_grad():
            for batch_data in val_loader:
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                
                outputs = sliding_window_inference(
                    inputs,
                    roi_size=self.config["roi_size"],
                    sw_batch_size=1,
                    predictor=self.model
                )
                
                # Argmax for class prediction
                outputs = torch.argmax(outputs, dim=1, keepdim=True)
                
                self.dice_metric(y_pred=outputs, y=labels)
        
        return self.dice_metric.aggregate().item()
    
    def train(
        self,
        data_dir: Path,
        output_dir: Path,
        val_split: float = 0.2
    ) -> Dict:
        """Full training loop"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_loader, val_loader = self.prepare_data(data_dir, val_split)
        
        best_dice = 0.0
        patience_counter = 0
        history = {"train_loss": [], "val_dice": []}
        
        for epoch in range(self.config["max_epochs"]):
            train_loss = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            
            val_dice = 0.0
            if val_loader:
                val_dice = self.validate(val_loader)
                history["val_dice"].append(val_dice)
                
                if val_dice > best_dice:
                    best_dice = val_dice
                    patience_counter = 0
                    torch.save(
                        self.model.state_dict(),
                        output_dir / "best_lung_seg_model.pth"
                    )
                    logger.info(f"Epoch {epoch+1}: New best Dice={val_dice:.4f}")
                else:
                    patience_counter += 1
            
            logger.info(
                f"Epoch {epoch+1}/{self.config['max_epochs']} | "
                f"Loss: {train_loss:.4f} | Dice: {val_dice:.4f}"
            )
            
            # Early stopping
            if patience_counter >= self.config["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save final model
        torch.save(
            self.model.state_dict(),
            output_dir / "final_lung_seg_model.pth"
        )
        
        return {"best_dice": best_dice, "history": history}


class LungSegmentationInference:
    """Lung segmentation inference"""
    
    def __init__(
        self,
        model_path: Path,
        device: Optional[str] = None,
        roi_size: Tuple[int, int, int] = (96, 96, 96)
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.roi_size = roi_size
        
        # Load model
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"LungSegmentationInference loaded from {model_path}")
    
    def _build_model(self) -> DynUNet:
        kernels = [[3, 3, 3]] * 5
        strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        
        return DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",
            deep_supervision=False
        )
    
    def predict(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Predict lung mask from CT volume
        
        Args:
            volume: (1, 1, D, H, W) tensor, normalized [0, 1]
            
        Returns:
            mask: (1, 1, D, H, W) binary mask
        """
        with torch.no_grad():
            volume = volume.to(self.device)
            
            output = sliding_window_inference(
                volume,
                roi_size=self.roi_size,
                sw_batch_size=1,
                predictor=self.model
            )
            
            mask = torch.argmax(output, dim=1, keepdim=True)
        
        return mask.cpu()
    
    def compute_lung_volume_ml(
        self,
        mask: torch.Tensor,
        spacing_mm: Tuple[float, float, float]
    ) -> float:
        """Compute lung volume in mL from mask"""
        voxel_volume_mm3 = spacing_mm[0] * spacing_mm[1] * spacing_mm[2]
        lung_voxels = (mask > 0).sum().item()
        return lung_voxels * voxel_volume_mm3 / 1000.0  # mm³ → mL
