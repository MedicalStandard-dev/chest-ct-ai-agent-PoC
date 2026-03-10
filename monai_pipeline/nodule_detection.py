# monai_pipeline/nodule_detection.py
"""
Nodule Detection Training/Inference (LIDC-IDRI)
Heatmap-based approach: RTX 3070 Ti (8GB) optimized
Features: ETA display, checkpoint resume
"""
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter, label as scipy_label
from scipy.ndimage import center_of_mass, find_objects, binary_fill_holes

from monai.networks.nets import UNet, SegResNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, SpatialPadd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    RandGaussianNoised, RandSpatialCropSamplesd, EnsureTyped,
    RandomizableTransform, MapTransform
)
from monai.data import PersistentDataset, DataLoader
from monai.inferers import sliding_window_inference

from api.schemas import NoduleCandidate, VisionEvidence
from utils.logger import logger


class LoRAConv3d(nn.Module):
    """Conv3d adapter branch for parameter-efficient fine-tuning."""

    def __init__(self, base_conv: nn.Conv3d, rank: int = 8, alpha: int = 16):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.rank = rank
        self.scale = alpha / float(rank)
        base_weight = base_conv.weight.detach().clone()
        base_bias = (
            base_conv.bias.detach().clone()
            if base_conv.bias is not None
            else None
        )
        self.stride = base_conv.stride
        self.padding = base_conv.padding
        self.dilation = base_conv.dilation
        self.groups = base_conv.groups

        self.register_buffer("base_weight", base_weight)
        if base_bias is not None:
            self.register_buffer("base_bias", base_bias)
        else:
            self.base_bias = None

        self.down = nn.Conv3d(
            in_channels=base_conv.in_channels,
            out_channels=rank,
            kernel_size=base_conv.kernel_size,
            stride=base_conv.stride,
            padding=base_conv.padding,
            dilation=base_conv.dilation,
            groups=base_conv.groups,
            bias=False,
        )
        self.up = nn.Conv3d(
            in_channels=rank,
            out_channels=base_conv.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.down.weight, a=np.sqrt(5))
        nn.init.zeros_(self.up.weight)
        # Keep adapter branch on the same device/dtype as base.
        self.down = self.down.to(
            device=base_conv.weight.device,
            dtype=base_conv.weight.dtype,
        )
        self.up = self.up.to(
            device=base_conv.weight.device,
            dtype=base_conv.weight.dtype,
        )

    def _base_forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.base_weight.to(dtype=x.dtype)
        bias = self.base_bias.to(dtype=x.dtype) if self.base_bias is not None else None
        return F.conv3d(
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._base_forward(x) + self.scale * self.up(self.down(x))


class QLoRAConv3d(LoRAConv3d):
    """
    QLoRA-style Conv3d adapter:
    - base conv stored as low-bit quantized frozen weights
    - LoRA adapters are trainable
    """

    def __init__(self, base_conv: nn.Conv3d, rank: int = 8, alpha: int = 16, bits: int = 4):
        if bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {bits}")
        self.bits = bits
        super().__init__(base_conv=base_conv, rank=rank, alpha=alpha)
        self._quantize_base()

    def _quantize_base(self) -> None:
        with torch.no_grad():
            w = self.base_weight.float()
            # Per-output-channel symmetric quantization.
            out_channels = w.shape[0]
            w_flat = w.view(out_channels, -1)
            max_abs = w_flat.abs().amax(dim=1).clamp(min=1e-8)
            qmax = float((2 ** (self.bits - 1)) - 1)
            scale = max_abs / qmax
            q = torch.round(w / scale.view(-1, 1, 1, 1, 1)).clamp(-qmax, qmax)

            self.register_buffer("qweight", q.to(torch.int8))
            self.register_buffer("qscale", scale)
            if self.base_bias is not None:
                self.register_buffer("qbias", self.base_bias.float())
            else:
                self.qbias = None

        # Remove full-precision base from the state to actually reduce checkpoint footprint.
        del self.base_weight
        if hasattr(self, "base_bias"):
            del self.base_bias

    def _base_forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = (self.qweight.float() * self.qscale.view(-1, 1, 1, 1, 1)).to(dtype=x.dtype)
        bias = self.qbias.to(dtype=x.dtype) if self.qbias is not None else None
        return F.conv3d(
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


def _replace_conv_with_adapter(
    module: nn.Module,
    mode: str,
    rank: int,
    alpha: int,
    qlora_bits: int,
) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv3d):
            if mode == "lora":
                wrapped = LoRAConv3d(child, rank=rank, alpha=alpha)
            elif mode == "qlora":
                wrapped = QLoRAConv3d(child, rank=rank, alpha=alpha, bits=qlora_bits)
            else:
                raise ValueError(f"Unsupported adapter mode: {mode}")
            setattr(module, name, wrapped)
            replaced += 1
            continue
        replaced += _replace_conv_with_adapter(
            child,
            mode=mode,
            rank=rank,
            alpha=alpha,
            qlora_bits=qlora_bits,
        )
    return replaced


def apply_parameter_efficient_finetuning(
    model: nn.Module,
    mode: str,
    rank: int = 8,
    alpha: int = 16,
    qlora_bits: int = 4,
) -> int:
    """
    Apply LoRA/QLoRA adapters to all Conv3d layers.
    Returns number of replaced layers.
    """
    if mode not in {"lora", "qlora"}:
        raise ValueError(f"mode must be one of ['lora', 'qlora'], got {mode}")

    for p in model.parameters():
        p.requires_grad = False

    replaced = _replace_conv_with_adapter(
        model,
        mode=mode,
        rank=rank,
        alpha=alpha,
        qlora_bits=qlora_bits,
    )
    if replaced == 0:
        raise RuntimeError("No Conv3d layers were replaced by adapters.")
    return replaced


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class ForegroundWeightedMSELoss(nn.Module):
    """
    MSE loss with higher weight on foreground voxels to combat background bias.
    """

    def __init__(self, fg_weight: float = 10.0, fg_threshold: float = 0.01):
        super().__init__()
        self.fg_weight = fg_weight
        self.fg_threshold = fg_threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sq_diff = (pred - target) ** 2
        fg_mask = (target > self.fg_threshold).float()
        weight = 1.0 + (self.fg_weight - 1.0) * fg_mask
        return (sq_diff * weight).mean()


class SoftDiceLoss(nn.Module):
    """
    Soft Dice loss for continuous heatmap regression.

    Dice = 2 * sum(pred * target) / (sum(pred^2) + sum(target^2) + eps)

    Inherently handles class imbalance because it normalizes by the
    total predicted and target volumes, not by voxel count.
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        denominator = (pred_flat ** 2).sum(dim=1) + (target_flat ** 2).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return (1.0 - dice).mean()


class CombinedHeatmapLoss(nn.Module):
    """
    Combined loss: ForegroundWeightedMSE + alpha * SoftDice

    MSE handles per-voxel accuracy, Dice handles global overlap.
    Together they prevent background collapse while maintaining
    precise heatmap values.
    """

    def __init__(
        self,
        fg_weight: float = 50.0,
        fg_threshold: float = 0.01,
        dice_alpha: float = 1.0,
    ):
        super().__init__()
        self.mse = ForegroundWeightedMSELoss(fg_weight=fg_weight, fg_threshold=fg_threshold)
        self.dice = SoftDiceLoss()
        self.dice_alpha = dice_alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, target) + self.dice_alpha * self.dice(pred, target)


class _NoduleDetectionBase:
    """Nodule detection trainer (Heatmap-based, LIDC format)"""

    # RTX 3070 Ti (8GB) optimized defaults — VRAM 39% target
    DEFAULT_CONFIG = {
        "roi_size": (128, 128, 128),   # 96→128: wider context
        "batch_size": 4,               # effective 8 patches (num_samples=2), ~3.2GB VRAM
        "num_workers": 2,          # PersistentDataset: workers safe, no RAM bloat
        "train_num_samples": 1,    # low-RAM default to avoid host OOM on large manifests
        "pin_memory": False,       # pinned host memory can trigger OOM under WSL
        "val_sw_batch_size": 1,
        "max_epochs": 200,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "amp_enabled": True,
        "target_spacing": (1.0, 1.0, 1.0),
        "intensity_range": (-1000, 400),
        "gaussian_sigma": 3.0,
        "pos_sample_ratio": 0.7,
        "early_stopping_patience": 20,
        "val_interval": 5,
        "checkpoint_interval": 10,
        # Best-model selection:
        # - "val_loss": lower is better
        # - "detection_proxy": sensitivity - fp_weight * fp_per_scan
        "selection_metric": "detection_proxy",
        "detection_eval_threshold": 0.3,
        "detection_gt_threshold": 0.4,
        "detection_match_distance_mm": 6.0,
        "detection_fp_weight": 0.05,
        "detection_min_component_voxels": 16,
        # Combined loss: ForegroundWeightedMSE + SoftDice
        "loss_fg_weight": 5.0,
        "loss_fg_threshold": 0.01,
        "loss_dice_alpha": 1.0,
    }

    def __init__(self, config: Optional[Dict] = None, model_type: str = "unet"):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        # Model
        self.model = self._build_model()
        self.model.to(self.device)

        # Loss: ForegroundWeightedMSE + SoftDice to prevent background collapse
        self.loss_fn = CombinedHeatmapLoss(
            fg_weight=self.config["loss_fg_weight"],
            fg_threshold=self.config["loss_fg_threshold"],
            dice_alpha=self.config["loss_dice_alpha"],
        )

        # Optimizer
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

        # AMP
        self.scaler = torch.cuda.amp.GradScaler() if self.config["amp_enabled"] else None

        logger.info(f"NoduleDetectionTrainer initialized (model: {model_type}, device: {self.device})")

    def _build_model(self) -> nn.Module:
        """Build detection model (outputs heatmap)"""
        if self.model_type == "segresnet":
            return SegResNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                init_filters=32,
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1]
            )
        else:
            return UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=(64, 128, 256, 512),
                strides=(2, 2, 2),
                num_res_units=2
            )

    def reset_optimizer_for_current_trainable_params(
        self,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        t_max: Optional[int] = None,
    ) -> None:
        """Rebuild optimizer/scheduler after changing trainable parameters."""
        lr = self.config["learning_rate"] if learning_rate is None else learning_rate
        wd = self.config["weight_decay"] if weight_decay is None else weight_decay
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters found for optimizer reset.")

        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=t_max or self.config["max_epochs"],
            eta_min=1e-6,
        )


class ForegroundAwareRandCropd(RandomizableTransform):
    """
    Apply pos/neg label crop for positive heatmaps and random spatial crop for empty heatmaps.
    This avoids repeated warnings when a sample has no foreground voxels.
    """

    def __init__(
        self,
        keys: List[str],
        label_key: str,
        spatial_size: Tuple[int, int, int],
        pos: float,
        neg: float,
        num_samples: int,
        allow_smaller: bool = False,
    ):
        super().__init__()
        self.label_key = label_key
        self.pos_neg_crop = RandCropByPosNegLabeld(
            keys=keys,
            label_key=label_key,
            spatial_size=spatial_size,
            pos=pos,
            neg=neg,
            num_samples=num_samples,
            allow_smaller=allow_smaller,
        )
        self.neg_only_crop = RandSpatialCropSamplesd(
            keys=keys,
            roi_size=spatial_size,
            num_samples=num_samples,
            random_center=True,
            random_size=False,
        )

    def set_random_state(self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None):
        super().set_random_state(seed=seed, state=state)
        self.pos_neg_crop.set_random_state(seed=seed, state=state)
        self.neg_only_crop.set_random_state(seed=seed, state=state)
        return self

    def _has_foreground(self, label: Any) -> bool:
        if torch.is_tensor(label):
            return bool(torch.any(label > 0))
        return bool(np.any(np.asarray(label) > 0))

    def __call__(self, data: Dict[str, Any]):
        if self._has_foreground(data[self.label_key]):
            return self.pos_neg_crop(data)
        return self.neg_only_crop(data)


class AdaptiveNormalizeIntensityd(MapTransform):
    """
    Normalize CT image robustly across HU-space and pre-normalized [0,1] inputs.
    - If data already in [0,1], only clamp to [0,1].
    - Else apply HU clipping + scaling to [0,1].
    """

    def __init__(self, keys: List[str], a_min: float = -1000.0, a_max: float = 400.0):
        super().__init__(keys)
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self._eps = 1e-3

    def _is_pre_normalized(self, img: Any) -> bool:
        if torch.is_tensor(img):
            return bool((img.min() >= -self._eps) and (img.max() <= 1.0 + self._eps))
        arr = np.asarray(img)
        return bool((arr.min() >= -self._eps) and (arr.max() <= 1.0 + self._eps))

    def _normalize(self, img: Any) -> Any:
        if torch.is_tensor(img):
            if self._is_pre_normalized(img):
                return torch.clamp(img, 0.0, 1.0)
            out = torch.clamp(img, self.a_min, self.a_max)
            return (out - self.a_min) / (self.a_max - self.a_min)

        arr = np.asarray(img, dtype=np.float32)
        if self._is_pre_normalized(arr):
            return np.clip(arr, 0.0, 1.0)
        arr = np.clip(arr, self.a_min, self.a_max)
        return (arr - self.a_min) / (self.a_max - self.a_min)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.keys:
            d[key] = self._normalize(d[key])
        return d


class HeuristicLungMaskd(MapTransform):
    """
    Mask heatmap to lung region using intensity-based heuristic.

    Expects the image to be normalised to [0, 1] (HU clipped).
    Lung parenchyma: body (>0.03) AND not too dense (<0.65).
    Large connected components are kept as the lung mask.
    """

    def __init__(
        self,
        image_key: str = "image",
        heatmap_key: str = "heatmap",
        body_thr: float = 0.03,
        dense_thr: float = 0.65,
        min_component_voxels: int = 128,
    ):
        super().__init__(keys=[heatmap_key])
        self.image_key = image_key
        self.heatmap_key = heatmap_key
        self.body_thr = body_thr
        self.dense_thr = dense_thr
        self.min_component_voxels = min_component_voxels

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        img = d[self.image_key]
        if torch.is_tensor(img):
            arr = img[0].cpu().numpy()
        else:
            arr = np.asarray(img)[0]

        mask = np.logical_and(arr > self.body_thr, arr < self.dense_thr)
        mask = binary_fill_holes(mask)

        labeled, n = scipy_label(mask.astype(np.uint8))
        if n > 0:
            comp_sizes = np.bincount(labeled.ravel())
            comp_sizes[0] = 0
            keep = set(
                int(k) for k in np.where(comp_sizes >= self.min_component_voxels)[0]
                if k > 0
            )
            if keep:
                mask = np.isin(labeled, list(keep))

        mask_f = mask.astype(np.float32)
        hm = d[self.heatmap_key]
        if torch.is_tensor(hm):
            d[self.heatmap_key] = hm * torch.from_numpy(mask_f).unsqueeze(0).to(hm.device)
        else:
            d[self.heatmap_key] = np.asarray(hm) * mask_f[np.newaxis]
        return d


class NoduleDetectionTrainer(_NoduleDetectionBase):
    """Nodule detection trainer (Heatmap-based, LIDC format)"""

    def _get_train_transforms(self) -> Compose:
        """Training transforms with augmentation"""
        return Compose([
            LoadImaged(keys=["image", "heatmap"]),
            EnsureChannelFirstd(keys=["image", "heatmap"]),
            Orientationd(keys=["image", "heatmap"], axcodes="RAS"),
            Spacingd(
                keys=["image", "heatmap"],
                pixdim=self.config["target_spacing"],
                mode=("bilinear", "bilinear")
            ),
            AdaptiveNormalizeIntensityd(
                keys=["image"],
                a_min=self.config["intensity_range"][0],
                a_max=self.config["intensity_range"][1],
            ),
            HeuristicLungMaskd(image_key="image", heatmap_key="heatmap"),
            SpatialPadd(
                keys=["image", "heatmap"],
                spatial_size=self.config["roi_size"],
                mode="constant"
            ),
            ForegroundAwareRandCropd(
                keys=["image", "heatmap"],
                label_key="heatmap",
                spatial_size=self.config["roi_size"],
                pos=self.config["pos_sample_ratio"],
                neg=1 - self.config["pos_sample_ratio"],
                num_samples=self.config["train_num_samples"],
                allow_smaller=False
            ),
            RandFlipd(keys=["image", "heatmap"], prob=0.3, spatial_axis=0),
            RandFlipd(keys=["image", "heatmap"], prob=0.3, spatial_axis=1),
            RandFlipd(keys=["image", "heatmap"], prob=0.3, spatial_axis=2),
            RandRotate90d(keys=["image", "heatmap"], prob=0.3, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
            EnsureTyped(keys=["image", "heatmap"], track_meta=False)
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
                mode=("bilinear", "bilinear")
            ),
            AdaptiveNormalizeIntensityd(
                keys=["image"],
                a_min=self.config["intensity_range"][0],
                a_max=self.config["intensity_range"][1],
            ),
            HeuristicLungMaskd(image_key="image", heatmap_key="heatmap"),
            EnsureTyped(keys=["image", "heatmap"], track_meta=False)
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
                with torch.amp.autocast("cuda"):
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

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        best_loss: float,
        best_metric_value: float,
        patience_counter: int,
        history: Dict,
    ):
        """Save training checkpoint for resume"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": best_loss,
            "best_metric_value": best_metric_value,
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
            f"best_loss={checkpoint['best_loss']:.6f}"
        )
        return checkpoint

    def train(
        self,
        train_files: List[Dict],
        val_files: List[Dict],
        output_dir: Path,
        resume: bool = False
    ) -> Dict:
        """Full training loop with ETA and checkpoint resume"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Datasets (PersistentDataset: cache preprocessed data to disk, minimal RAM)
        cache_dir = output_dir / ".persistent_cache"
        train_cache = cache_dir / "train"
        train_cache.mkdir(parents=True, exist_ok=True)

        train_ds = PersistentDataset(
            data=train_files,
            transform=self._get_train_transforms(),
            cache_dir=train_cache,
        )

        if val_files:
            val_cache = cache_dir / "val"
            val_cache.mkdir(parents=True, exist_ok=True)
            val_ds = PersistentDataset(
                data=val_files,
                transform=self._get_val_transforms(),
                cache_dir=val_cache,
            )
        else:
            val_ds = None

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"]
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"]
        ) if val_ds else None

        selection_metric = self.config.get("selection_metric", "detection_proxy")
        if selection_metric not in {"val_loss", "detection_proxy"}:
            raise ValueError(f"Unsupported selection_metric: {selection_metric}")

        best_loss = float("inf")
        best_metric_value = float("inf") if selection_metric == "val_loss" else float("-inf")
        patience_counter = 0
        start_epoch = 0
        val_interval = self.config["val_interval"]
        ckpt_interval = self.config["checkpoint_interval"]
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_sensitivity": [],
            "val_fp_per_scan": [],
            "val_proxy_score": [],
        }

        # Resume from checkpoint
        ckpt_path = output_dir / "checkpoint.pth"
        if resume and ckpt_path.exists():
            ckpt = self.load_checkpoint(ckpt_path)
            start_epoch = ckpt["epoch"] + 1
            best_loss = ckpt["best_loss"]
            if "best_metric_value" in ckpt:
                best_metric_value = ckpt["best_metric_value"]
            elif selection_metric == "val_loss":
                best_metric_value = ckpt["best_loss"]
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

            train_loss = self.train_epoch(train_loader)
            self.scheduler.step()
            history["train_loss"].append(train_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Validate
            val_loss = 0.0
            val_sensitivity = 0.0
            val_fp_per_scan = 0.0
            val_proxy_score = float("-inf")
            if val_loader and (epoch + 1) % val_interval == 0:
                val_loss, val_sensitivity, val_fp_per_scan, val_proxy_score = self._validate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_sensitivity"].append(val_sensitivity)
                history["val_fp_per_scan"].append(val_fp_per_scan)
                history["val_proxy_score"].append(val_proxy_score)

                current_metric = val_loss if selection_metric == "val_loss" else val_proxy_score
                is_better = current_metric < best_metric_value if selection_metric == "val_loss" else current_metric > best_metric_value
                if is_better:
                    best_loss = val_loss
                    best_metric_value = current_metric
                    patience_counter = 0
                    torch.save(
                        self.model.state_dict(),
                        output_dir / "best_nodule_det_model.pth"
                    )
                    if selection_metric == "val_loss":
                        logger.info(f"  ★ New best val_loss={val_loss:.6f}")
                    else:
                        logger.info(
                            "  ★ New best detection_proxy="
                            f"{val_proxy_score:.6f} "
                            f"(sens={val_sensitivity:.4f}, fp/scan={val_fp_per_scan:.2f}, val_loss={val_loss:.6f})"
                        )
                else:
                    patience_counter += 1

            epoch_elapsed = time.time() - epoch_start
            epoch_times.append(epoch_elapsed)

            # ETA calculation
            recent = epoch_times[-10:]
            avg_epoch_time = sum(recent) / len(recent)
            epochs_left = max_epochs - epoch - 1
            eta_seconds = avg_epoch_time * epochs_left
            total_elapsed = time.time() - train_start

            def _fmt(s):
                total = int(round(float(s)))
                if total < 60:
                    return f"{total}s"
                if total < 3600:
                    minutes, seconds = divmod(total, 60)
                    return f"{minutes}m {seconds}s"
                hours, remain = divmod(total, 3600)
                minutes = remain // 60
                return f"{hours}h {minutes}m"

            logger.info(
                f"Epoch {epoch+1}/{max_epochs} | "
                f"Loss: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Sens: {val_sensitivity:.4f} | FP/scan: {val_fp_per_scan:.2f} | "
                f"LR: {current_lr:.2e} | "
                f"{epoch_elapsed:.1f}s/epoch | "
                f"Elapsed: {_fmt(total_elapsed)} | "
                f"ETA: {_fmt(eta_seconds)}"
            )

            # Save checkpoint periodically
            if (epoch + 1) % ckpt_interval == 0:
                self.save_checkpoint(
                    ckpt_path, epoch, best_loss, best_metric_value, patience_counter, history
                )

            if patience_counter >= self.config["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Save final
        total_time = time.time() - train_start
        torch.save(
            self.model.state_dict(),
            output_dir / "final_nodule_det_model.pth"
        )
        self.save_checkpoint(
            ckpt_path, epoch, best_loss, best_metric_value, patience_counter, history
        )

        def _fmt(s):
            total = int(round(float(s)))
            if total < 60:
                return f"{total}s"
            if total < 3600:
                minutes, seconds = divmod(total, 60)
                return f"{minutes}m {seconds}s"
            hours, remain = divmod(total, 3600)
            minutes = remain // 60
            return f"{hours}h {minutes}m"

        logger.info(f"Training complete in {_fmt(total_time)}")
        logger.info(f"Best val_loss: {best_loss:.6f}")
        if selection_metric == "val_loss":
            logger.info(f"Best selection metric(val_loss): {best_metric_value:.6f}")
        else:
            logger.info(f"Best selection metric(detection_proxy): {best_metric_value:.6f}")

        return {
            "best_loss": best_loss,
            "best_metric_value": best_metric_value,
            "selection_metric": selection_metric,
            "history": history,
            "total_time": total_time,
        }

    def _match_centers(
        self,
        pred_centers: List[np.ndarray],
        gt_centers: List[np.ndarray],
        spacing_mm: Tuple[float, float, float],
        match_distance_mm: float,
    ) -> Tuple[int, int, int]:
        if len(gt_centers) == 0:
            return 0, len(pred_centers), 0

        matched = set()
        tp = 0
        spacing = np.asarray(spacing_mm, dtype=np.float32)
        for pc in pred_centers:
            best_idx = -1
            best_dist = float("inf")
            for i, gc in enumerate(gt_centers):
                if i in matched:
                    continue
                dist_mm = float(np.linalg.norm((pc - gc) * spacing))
                if dist_mm <= match_distance_mm and dist_mm < best_dist:
                    best_idx = i
                    best_dist = dist_mm
            if best_idx >= 0:
                matched.add(best_idx)
                tp += 1
        fp = len(pred_centers) - tp
        fn = len(gt_centers) - tp
        return tp, fp, fn

    def _extract_component_centers(self, binary_map: np.ndarray) -> List[np.ndarray]:
        labeled, num = scipy_label(binary_map.astype(np.uint8))
        if num == 0:
            return []

        min_voxels = int(self.config.get("detection_min_component_voxels", 16))
        comp_sizes = np.bincount(labeled.ravel())
        if comp_sizes.size <= 1:
            return []

        valid_ids = np.where(comp_sizes >= min_voxels)[0]
        valid_ids = valid_ids[valid_ids > 0]
        if valid_ids.size == 0:
            return []

        # Vectorized center-of-mass for selected components.
        mass = np.ones_like(labeled, dtype=np.float32)
        coms = center_of_mass(mass, labels=labeled, index=valid_ids.tolist())
        centers: List[np.ndarray] = []
        for com in coms:
            if com is None:
                continue
            if np.isnan(com[0]) or np.isnan(com[1]) or np.isnan(com[2]):
                continue
            centers.append(np.asarray(com, dtype=np.float32))
        return centers

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float, float, float]:
        """Validation pass"""
        self.model.eval()
        total_loss = 0.0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        scans = 0

        pred_thr = float(self.config.get("detection_eval_threshold", 0.9))
        gt_thr = float(self.config.get("detection_gt_threshold", 0.4))
        match_distance_mm = float(self.config.get("detection_match_distance_mm", 6.0))
        fp_weight = float(self.config.get("detection_fp_weight", 0.05))
        spacing_mm = tuple(float(x) for x in self.config.get("target_spacing", (1.5, 1.5, 1.5)))

        with torch.no_grad():
            for batch_data in val_loader:
                inputs = batch_data["image"].to(self.device)
                targets = batch_data["heatmap"].to(self.device)

                outputs = sliding_window_inference(
                    inputs,
                    roi_size=self.config["roi_size"],
                    sw_batch_size=self.config["val_sw_batch_size"],
                    predictor=self.model,
                    overlap=0.5
                )
                outputs = torch.sigmoid(outputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                scans += int(outputs.shape[0])

                pred_np = outputs.detach().cpu().numpy()
                target_np = targets.detach().cpu().numpy()
                for b in range(pred_np.shape[0]):
                    pred_binary = pred_np[b, 0] > pred_thr
                    gt_binary = target_np[b, 0] > gt_thr
                    pred_centers = self._extract_component_centers(pred_binary)
                    gt_centers = self._extract_component_centers(gt_binary)
                    tp, fp, fn = self._match_centers(
                        pred_centers=pred_centers,
                        gt_centers=gt_centers,
                        spacing_mm=spacing_mm,
                        match_distance_mm=match_distance_mm,
                    )
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

        val_loss = total_loss / len(val_loader)
        val_sensitivity = (
            float(total_tp) / float(total_tp + total_fn)
            if (total_tp + total_fn) > 0 else 0.0
        )
        val_fp_per_scan = float(total_fp) / float(max(1, scans))
        val_proxy_score = val_sensitivity - fp_weight * val_fp_per_scan
        return val_loss, val_sensitivity, val_fp_per_scan, val_proxy_score


class NoduleDetectionInference:
    """Nodule detection inference with postprocessing"""

    def __init__(
        self,
        model_path: Path,
        model_type: str = "unet",
        device: Optional[str] = None,
        roi_size: Tuple[int, int, int] = (96, 96, 96),
        detection_threshold: float = 0.5,
        min_diameter_mm: float = 3.0,
        max_diameter_mm: Optional[float] = None,
        max_volume_mm3: Optional[float] = None,
        restrict_to_lung_mask: bool = False,
        min_lung_overlap_ratio: float = 0.3,
        lung_seg_model_path: Optional[Path] = None,
        finetune_mode: str = "full",
        lora_rank: int = 8,
        lora_alpha: int = 16,
        qlora_bits: int = 4,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.roi_size = roi_size
        self.detection_threshold = detection_threshold
        self.min_diameter_mm = min_diameter_mm
        self.max_diameter_mm = max_diameter_mm
        self.max_volume_mm3 = max_volume_mm3
        self.restrict_to_lung_mask = restrict_to_lung_mask
        self.min_lung_overlap_ratio = min_lung_overlap_ratio
        self.model_type = model_type
        self.finetune_mode = finetune_mode
        self.lung_seg_model_path = Path(lung_seg_model_path) if lung_seg_model_path else None
        self._lung_seg_inference = None

        # Load model
        self.model = self._build_model()
        if finetune_mode in {"lora", "qlora"}:
            replaced = apply_parameter_efficient_finetuning(
                model=self.model,
                mode=finetune_mode,
                rank=lora_rank,
                alpha=lora_alpha,
                qlora_bits=qlora_bits,
            )
            logger.info(
                f"Inference adapter mode={finetune_mode}, replaced Conv3d layers={replaced}"
            )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        if self.restrict_to_lung_mask and self.lung_seg_model_path is not None:
            self._load_lung_seg_inference()

        logger.info(f"NoduleDetectionInference loaded from {model_path}")
        logger.info(
            "Postprocess config: "
            f"th={self.detection_threshold}, "
            f"min_diam={self.min_diameter_mm}, "
            f"max_diam={self.max_diameter_mm}, "
            f"max_vol={self.max_volume_mm3}, "
            f"restrict_to_lung_mask={self.restrict_to_lung_mask}, "
            f"min_lung_overlap_ratio={self.min_lung_overlap_ratio}"
        )

    def _load_lung_seg_inference(self) -> None:
        """Lazy-load lung segmentation model for mask-constrained detection."""
        if self._lung_seg_inference is not None:
            return
        if self.lung_seg_model_path is None:
            return
        try:
            from monai_pipeline.lung_segmentation import LungSegmentationInference

            self._lung_seg_inference = LungSegmentationInference(
                model_path=self.lung_seg_model_path,
                device=str(self.device),
                roi_size=self.roi_size,
            )
            logger.info(f"Lung mask model loaded: {self.lung_seg_model_path}")
        except Exception as e:
            logger.warning(f"Failed to load lung segmentation model: {e}")
            self._lung_seg_inference = None

    def _estimate_lung_mask(self, volume: torch.Tensor) -> np.ndarray:
        """
        Estimate lung mask from input volume.
        Uses segmentation model when available, otherwise a simple intensity heuristic.
        """
        if self._lung_seg_inference is not None:
            try:
                v = volume.detach().to(dtype=torch.float32).cpu()
                # Model expects [0, 1] style normalization.
                v_min, v_max = float(v.min()), float(v.max())
                if v_max > v_min:
                    v = (v - v_min) / (v_max - v_min)
                mask = self._lung_seg_inference.predict(v).squeeze().numpy().astype(np.uint8)
                return mask
            except Exception as e:
                logger.warning(f"Lung mask inference failed, fallback to heuristic: {e}")

        # Heuristic fallback for normalized CT ([0,1] from clipped HU).
        arr = volume.detach().to(dtype=torch.float32).cpu().numpy()[0, 0]
        body_mask = arr > 0.03
        lung_like = arr < 0.65
        mask = np.logical_and(body_mask, lung_like)
        mask = binary_fill_holes(mask)

        labeled, n = scipy_label(mask.astype(np.uint8))
        if n > 0:
            comp_sizes = np.bincount(labeled.ravel())
            comp_sizes[0] = 0
            keep = np.argsort(comp_sizes)[-4:]
            keep = set(int(k) for k in keep if k > 0 and comp_sizes[k] >= 128)
            if keep:
                mask = np.isin(labeled, list(keep))
        return mask.astype(np.uint8)

    def _build_model(self) -> nn.Module:
        if self.model_type == "segresnet":
            return SegResNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                init_filters=32,
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1]
            )
        else:
            return UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=(64, 128, 256, 512),
                strides=(2, 2, 2),
                num_res_units=2
            )

    def _normalize_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """Normalize inference volume to [0,1] with HU-aware fallback."""
        v = volume.to(dtype=torch.float32)
        v_min = float(v.min().item())
        v_max = float(v.max().item())
        if v_min >= -1e-3 and v_max <= 1.0 + 1e-3:
            return torch.clamp(v, 0.0, 1.0)
        v = torch.clamp(v, -1000.0, 400.0)
        return (v + 1000.0) / 1400.0

    def predict_heatmap(self, volume: torch.Tensor) -> np.ndarray:
        """
        Predict nodule probability heatmap

        Args:
            volume: (1, 1, D, H, W) tensor, normalized [0, 1]

        Returns:
            heatmap: (D, H, W) numpy array, probability [0, 1]
        """
        with torch.no_grad():
            volume = self._normalize_volume(volume).to(self.device)

            output = sliding_window_inference(
                volume,
                roi_size=self.roi_size,
                sw_batch_size=2,
                predictor=self.model,
                overlap=0.5
            )

            heatmap = torch.sigmoid(output).cpu().numpy()[0, 0]

        return heatmap

    def extract_candidates(
        self,
        heatmap: np.ndarray,
        spacing_mm: Tuple[float, float, float],
        series_uid: str,
        lung_mask: Optional[np.ndarray] = None,
    ) -> List[NoduleCandidate]:
        """
        Extract nodule candidates from heatmap

        Args:
            heatmap: (D, H, W) probability map
            spacing_mm: voxel spacing (z, y, x) in mm
            series_uid: DICOM series UID
            lung_mask: optional (D,H,W) mask. When provided and
                restrict_to_lung_mask=True, candidates are limited to lung area.

        Returns:
            List of NoduleCandidate objects
        """
        # Threshold
        binary = (heatmap > self.detection_threshold).astype(np.uint8)

        # Connected components
        labeled, num_features = scipy_label(binary)
        valid_lung_mask = None
        if lung_mask is not None:
            if lung_mask.shape == heatmap.shape:
                valid_lung_mask = lung_mask.astype(bool)
            else:
                logger.warning(
                    f"Lung mask shape mismatch: mask={lung_mask.shape}, heatmap={heatmap.shape}. "
                    "Skip lung-mask filtering for this volume."
                )

        candidates = []
        skipped_small = 0
        skipped_large_diameter = 0
        skipped_large_volume = 0
        skipped_outside_lung = 0
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
                skipped_small += 1
                continue

            # Skip too large
            if self.max_diameter_mm is not None and diameter_mm > self.max_diameter_mm:
                skipped_large_diameter += 1
                continue

            # Volume estimation
            voxel_count = component_mask.sum()
            voxel_volume = spacing_mm[0] * spacing_mm[1] * spacing_mm[2]
            volume_mm3 = float(voxel_count * voxel_volume)

            if self.max_volume_mm3 is not None and volume_mm3 > self.max_volume_mm3:
                skipped_large_volume += 1
                continue

            if self.restrict_to_lung_mask and valid_lung_mask is not None:
                inside_ratio = float(valid_lung_mask[component_mask].mean())
                if inside_ratio < self.min_lung_overlap_ratio:
                    skipped_outside_lung += 1
                    continue

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
                    instance_uids=[],
                    slice_range=slice_range,
                    confidence=round(confidence, 3)
                ),
                location_code=self._estimate_location(center_zyx, heatmap.shape)
            )

            candidates.append(candidate)

        # Sort by confidence
        candidates.sort(key=lambda x: x.confidence, reverse=True)

        logger.info(
            f"Extracted {len(candidates)} nodule candidates "
            f"(components={num_features}, "
            f"skip_small={skipped_small}, "
            f"skip_large_diameter={skipped_large_diameter}, "
            f"skip_large_volume={skipped_large_volume}, "
            f"skip_outside_lung={skipped_outside_lung})"
        )
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
        series_uid: str,
        lung_mask: Optional[np.ndarray] = None,
    ) -> List[NoduleCandidate]:
        """
        Full detection pipeline

        Args:
            volume: (1, 1, D, H, W) normalized CT volume
            spacing_mm: voxel spacing in mm
            series_uid: DICOM series UID
            lung_mask: optional (D,H,W) lung mask

        Returns:
            List of NoduleCandidate objects
        """
        heatmap = self.predict_heatmap(volume)
        final_lung_mask = lung_mask
        if self.restrict_to_lung_mask and final_lung_mask is None:
            final_lung_mask = self._estimate_lung_mask(volume)
        return self.extract_candidates(
            heatmap,
            spacing_mm,
            series_uid,
            lung_mask=final_lung_mask,
        )


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
