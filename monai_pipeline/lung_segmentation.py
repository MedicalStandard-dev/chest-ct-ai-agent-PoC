# monai_pipeline/lung_segmentation.py
"""
Lung Segmentation Training/Inference (MSD Task06_Lung)
RTX 3070 Ti (8GB) optimized: patch-based, AMP, DiceCELoss
Features: ETA display, checkpoint resume
"""
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import time
import json
import torch
import numpy as np
import torch.nn.functional as F

from monai.networks.nets import DynUNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, SpatialPadd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    RandGaussianNoised, EnsureTyped
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference

from utils.logger import logger


def _fmt_time(seconds: float) -> str:
    """Format seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m {seconds%60:.0f}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


class LungSegmentationTrainer:
    """Lung segmentation trainer (DynUNet, MSD format)"""

    # RTX 3070 Ti (8GB) optimized defaults — VRAM 85% target
    DEFAULT_CONFIG = {
        "roi_size": (128, 128, 128),
        "batch_size": 2,           # effective 4 patches (num_samples=2), ~8GB VRAM
        "num_workers": 0,          # WSL2 16GB: no fork to avoid RAM doubling
        "cache_rate": 1.0,         # all data in RAM (~5.7GB at 1.5mm spacing)
        "max_epochs": 150,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "amp_enabled": True,
        "target_spacing": (1.5, 1.5, 1.5),  # WSL2 16GB: 1.0→1.5 reduces volume ~3.4x
        "intensity_range": (-1000, 400),
        "early_stopping_patience": 30,
        "val_interval": 5,
        "checkpoint_interval": 10,  # Save checkpoint every N epochs
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.model = self._build_model()
        self.model.to(self.device)

        # Loss (DiceCELoss: Dice + CE, more stable than DiceFocalLoss)
        self.loss_fn = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            lambda_dice=1.0,
            lambda_ce=1.0
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["max_epochs"],
            eta_min=1e-6
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
            out_channels=2,  # background + lung tumor
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",
            deep_supervision=True
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
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=self.config["roi_size"],
                mode="constant"
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=self.config["roi_size"],
                pos=2, neg=1,
                num_samples=2,
                allow_smaller=False
            ),
            RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
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

        logger.info(f"Split: {len(train_files)} train, {len(val_files)} val")

        # Datasets (CacheDataset: all data in RAM, fast training)
        train_ds = CacheDataset(
            data=train_files,
            transform=self._get_train_transforms(),
            cache_rate=self.config["cache_rate"],
            num_workers=0,
        )

        val_ds = CacheDataset(
            data=val_files,
            transform=self._get_val_transforms(),
            cache_rate=1.0,
            num_workers=0,
        ) if val_files else None

        # Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=True
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=True
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
                with torch.amp.autocast("cuda"):
                    outputs = self.model(inputs)
                    if isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]
                    elif outputs.dim() == 6:  # deep supervision: [B, heads, C, D, H, W]
                        outputs = outputs[:, 0]
                    loss = self.loss_fn(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                elif outputs.dim() == 6:  # deep supervision: [B, heads, C, D, H, W]
                    outputs = outputs[:, 0]
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
                    sw_batch_size=2,
                    predictor=self.model,
                    overlap=0.5
                )

                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                elif outputs.dim() == 6:  # deep supervision
                    outputs = outputs[:, 0]

                # DiceMetric(include_background=False) expects channel-first class predictions.
                pred_idx = torch.argmax(outputs, dim=1).long()           # [B, D, H, W]
                label_idx = labels.squeeze(1).long()                     # [B, D, H, W]
                pred_1h = F.one_hot(pred_idx, num_classes=2).permute(0, 4, 1, 2, 3).float()
                label_1h = F.one_hot(label_idx, num_classes=2).permute(0, 4, 1, 2, 3).float()
                self.dice_metric(y_pred=pred_1h, y=label_1h)

        return self.dice_metric.aggregate().item()

    def save_checkpoint(self, path: Path, epoch: int, best_dice: float,
                        patience_counter: int, history: Dict):
        """Save training checkpoint for resume"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_dice": best_dice,
            "patience_counter": patience_counter,
            "history": history,
            "config": self.config,
        }
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> Dict:
        """Load training checkpoint to resume"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        logger.info(
            f"Resumed from epoch {checkpoint['epoch']+1}, "
            f"best_dice={checkpoint['best_dice']:.4f}"
        )
        return checkpoint

    def train(
        self,
        data_dir: Path,
        output_dir: Path,
        val_split: float = 0.2,
        resume: bool = False
    ) -> Dict:
        """Full training loop with ETA and checkpoint resume"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_loader, val_loader = self.prepare_data(data_dir, val_split)

        best_dice = 0.0
        patience_counter = 0
        start_epoch = 0
        val_interval = self.config["val_interval"]
        ckpt_interval = self.config["checkpoint_interval"]
        history = {"train_loss": [], "val_dice": []}

        # Resume from checkpoint
        ckpt_path = output_dir / "checkpoint.pth"
        if resume and ckpt_path.exists():
            ckpt = self.load_checkpoint(ckpt_path)
            start_epoch = ckpt["epoch"] + 1
            best_dice = ckpt["best_dice"]
            patience_counter = ckpt["patience_counter"]
            history = ckpt["history"]
            logger.info(f"Resuming from epoch {start_epoch+1}")
        elif resume:
            logger.warning("No checkpoint found, starting from scratch")

        max_epochs = self.config["max_epochs"]
        remaining = max_epochs - start_epoch
        epoch_times = []
        train_start = time.time()

        logger.info(f"Training epochs {start_epoch+1} → {max_epochs} ({remaining} remaining)")

        for epoch in range(start_epoch, max_epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)
            self.scheduler.step()
            history["train_loss"].append(train_loss)

            epoch_elapsed = time.time() - epoch_start
            epoch_times.append(epoch_elapsed)

            current_lr = self.optimizer.param_groups[0]["lr"]

            # ETA calculation (moving average of last 10 epochs)
            recent = epoch_times[-10:]
            avg_epoch_time = sum(recent) / len(recent)
            epochs_left = max_epochs - epoch - 1
            eta_seconds = avg_epoch_time * epochs_left
            total_elapsed = time.time() - train_start

            # Validate
            val_dice = 0.0
            if val_loader and (epoch + 1) % val_interval == 0:
                val_start = time.time()
                val_dice = self.validate(val_loader)
                val_elapsed = time.time() - val_start
                history["val_dice"].append(val_dice)

                if val_dice > best_dice:
                    best_dice = val_dice
                    patience_counter = 0
                    torch.save(
                        self.model.state_dict(),
                        output_dir / "best_lung_seg_model.pth"
                    )
                    logger.info(f"  ★ New best Dice={val_dice:.4f}")
                else:
                    patience_counter += 1

            # Log with ETA
            logger.info(
                f"Epoch {epoch+1}/{max_epochs} | "
                f"Loss: {train_loss:.4f} | Dice: {val_dice:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"{epoch_elapsed:.1f}s/epoch | "
                f"Elapsed: {_fmt_time(total_elapsed)} | "
                f"ETA: {_fmt_time(eta_seconds)}"
            )

            # Save checkpoint periodically
            if (epoch + 1) % ckpt_interval == 0:
                self.save_checkpoint(
                    ckpt_path, epoch, best_dice, patience_counter, history
                )

            # Early stopping
            if patience_counter >= self.config["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Save final
        total_time = time.time() - train_start
        torch.save(
            self.model.state_dict(),
            output_dir / "final_lung_seg_model.pth"
        )
        self.save_checkpoint(
            ckpt_path, epoch, best_dice, patience_counter, history
        )

        logger.info(f"Training complete in {_fmt_time(total_time)}")
        logger.info(f"Best Dice: {best_dice:.4f}")

        return {"best_dice": best_dice, "history": history, "total_time": total_time}


class LungSegmentationInference:
    """Lung segmentation inference"""

    def __init__(
        self,
        model_path: Path,
        device: Optional[str] = None,
        roi_size: Tuple[int, int, int] = (128, 128, 128)
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.roi_size = roi_size

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
            deep_supervision=True
        )

    def predict(self, volume: torch.Tensor) -> torch.Tensor:
        """
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
                sw_batch_size=2,
                predictor=self.model,
                overlap=0.5
            )
            if isinstance(output, (list, tuple)):
                output = output[0]
            elif output.dim() == 6:  # deep supervision: [B, heads, C, D, H, W]
                output = output[:, 0]
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
        return lung_voxels * voxel_volume_mm3 / 1000.0
