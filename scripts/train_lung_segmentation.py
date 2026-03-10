#!/usr/bin/env python
# scripts/train_lung_segmentation.py
"""
Lung Segmentation 학습 스크립트 (MSD Task06_Lung)

RTX 3070 Ti (8GB) 최적화:
- roi_size: (128, 128, 128)
- batch_size: 2
- AMP: enabled
- cache_rate: 0.5
- spacing: isotropic (1.0, 1.0, 1.0)
- loss: DiceCELoss
- scheduler: CosineAnnealingLR

사용:
    # 새로 학습
    python scripts/train_lung_segmentation.py \
        --data-dir data/Task06_Lung \
        --output-dir models/lung_seg

    # 이어서 학습 (끊긴 경우)
    python scripts/train_lung_segmentation.py \
        --data-dir data/Task06_Lung \
        --output-dir models/lung_seg \
        --resume
"""
import argparse
from pathlib import Path
import sys

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from monai_pipeline.lung_segmentation import LungSegmentationTrainer, LungSegmentationInference
from utils.logger import logger


def train(args):
    """학습 실행"""
    logger.info("=" * 60)
    logger.info("LUNG SEGMENTATION TRAINING (RTX 3070 Ti)")
    logger.info("=" * 60)

    # Config
    config = {
        "roi_size": tuple(args.roi_size),
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "amp_enabled": args.amp,
        "num_workers": args.num_workers,
        "early_stopping_patience": args.patience,
        "val_interval": args.val_interval,
    }

    logger.info(f"Config: {config}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")

    # Initialize trainer
    trainer = LungSegmentationTrainer(config=config)

    # Train
    result = trainer.train(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        val_split=args.val_split,
        resume=args.resume
    )

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Best Dice: {result['best_dice']:.4f}")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("=" * 60)

    return result


def test_inference(args):
    """추론 테스트"""
    import torch

    logger.info("Testing inference...")

    model_path = Path(args.output_dir) / "best_lung_seg_model.pth"
    if not model_path.exists():
        model_path = Path(args.output_dir) / "final_lung_seg_model.pth"

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    # Load inference model
    inference = LungSegmentationInference(
        model_path=model_path,
        roi_size=tuple(args.roi_size)
    )

    # Create dummy input
    dummy_input = torch.randn(1, 1, 128, 128, 128)

    # Predict
    mask = inference.predict(dummy_input)

    logger.info(f"Input shape: {dummy_input.shape}")
    logger.info(f"Output shape: {mask.shape}")
    logger.info(f"Unique values: {torch.unique(mask).tolist()}")

    # Compute volume
    spacing = (1.0, 1.0, 1.0)
    volume_ml = inference.compute_lung_volume_ml(mask, spacing)
    logger.info(f"Lung volume: {volume_ml:.1f} mL")


def main():
    parser = argparse.ArgumentParser(
        description="Lung segmentation 학습 (MSD Task06_Lung, RTX 3070 Ti)"
    )

    # Data paths
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        required=True,
        help="MSD Task06_Lung 데이터 디렉토리"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("models/lung_seg"),
        help="모델 저장 디렉토리"
    )

    # Training config (RTX 3070 Ti defaults)
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=150,
        help="최대 에폭 수 (default: 150)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=2,
        help="배치 크기 (RTX 3070 Ti: 2 권장)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="AdamW weight decay"
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="ROI 크기 (D H W)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split 비율"
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=5,
        help="Validation 주기 (epoch 단위)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers (PersistentDataset: 2 권장)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience (val_interval 단위)"
    )

    # Options
    parser.add_argument(
        "--resume",
        action="store_true",
        help="마지막 checkpoint에서 이어서 학습"
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="AMP 비활성화"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="추론 테스트만 실행"
    )

    args = parser.parse_args()
    args.amp = not args.no_amp

    if args.test_only:
        test_inference(args)
    else:
        train(args)
        test_inference(args)


if __name__ == "__main__":
    main()
