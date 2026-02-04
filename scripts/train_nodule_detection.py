#!/usr/bin/env python
# scripts/train_nodule_detection.py
"""
Nodule Detection 학습 스크립트 (LIDC-IDRI)

Heatmap 기반 approach:
- 3D UNet / SegResNet
- Gaussian blob labels
- Focal loss for class imbalance

GTX 1660 최적화:
- patch_size: (96, 96, 96)
- batch_size: 1
- AMP: enabled

사용:
    # 1. LIDC 전처리 먼저 실행
    python scripts/preprocess_lidc.py --lidc-root data/LIDC-IDRI --output data/processed_lidc
    
    # 2. 학습 실행
    python scripts/train_nodule_detection.py \
        --data-dir data/processed_lidc \
        --output-dir models/nodule_det \
        --epochs 100
"""
import argparse
from pathlib import Path
import sys

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from monai_pipeline.nodule_detection import NoduleDetectionTrainer, NoduleDetectionInference
from monai_pipeline.data_processing import create_training_split
from utils.logger import logger


def train(args):
    """학습 실행"""
    logger.info("=" * 60)
    logger.info("NODULE DETECTION TRAINING")
    logger.info("=" * 60)
    
    # Load data
    manifest_path = Path(args.data_dir) / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        logger.error("Run preprocess_lidc.py first!")
        return None
    
    train_files, val_files = create_training_split(
        manifest_path=manifest_path,
        train_ratio=1 - args.val_split,
        seed=args.seed
    )
    
    logger.info(f"Train samples: {len(train_files)}")
    logger.info(f"Val samples: {len(val_files)}")
    
    # Config
    config = {
        "roi_size": tuple(args.roi_size),
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "learning_rate": args.lr,
        "amp_enabled": args.amp,
        "cache_rate": args.cache_rate,
        "num_workers": args.num_workers,
        "pos_sample_ratio": args.pos_ratio,
        "early_stopping_patience": args.patience
    }
    
    logger.info(f"Config: {config}")
    logger.info(f"Model type: {args.model_type}")
    
    # Initialize trainer
    trainer = NoduleDetectionTrainer(
        config=config,
        model_type=args.model_type
    )
    
    # Train
    result = trainer.train(
        train_files=train_files,
        val_files=val_files,
        output_dir=Path(args.output_dir)
    )
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Best loss: {result['best_loss']:.4f}")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("=" * 60)
    
    return result


def test_inference(args):
    """추론 테스트"""
    import torch
    
    logger.info("Testing inference...")
    
    model_path = Path(args.output_dir) / "best_nodule_det_model.pth"
    if not model_path.exists():
        model_path = Path(args.output_dir) / "final_nodule_det_model.pth"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    # Load inference model
    inference = NoduleDetectionInference(
        model_path=model_path,
        model_type=args.model_type,
        roi_size=tuple(args.roi_size),
        detection_threshold=args.threshold,
        min_diameter_mm=args.min_diameter
    )
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, 128, 128, 128)
    
    # Detect
    spacing = (1.0, 1.0, 1.0)
    candidates = inference.detect(
        volume=dummy_input,
        spacing_mm=spacing,
        series_uid="test.series.uid"
    )
    
    logger.info(f"Input shape: {dummy_input.shape}")
    logger.info(f"Candidates found: {len(candidates)}")
    
    for c in candidates[:5]:  # Show top 5
        logger.info(
            f"  - {c.id}: {c.location_code}, "
            f"{c.diameter_mm:.1f}mm, conf={c.confidence:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Nodule detection 학습 (LIDC-IDRI)"
    )
    
    # Data paths
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        required=True,
        help="전처리된 LIDC 데이터 디렉토리 (manifest.json 포함)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("models/nodule_det"),
        help="모델 저장 디렉토리"
    )
    
    # Model config
    parser.add_argument(
        "--model-type", "-m",
        type=str,
        choices=["unet", "segresnet"],
        default="unet",
        help="모델 유형"
    )
    
    # Training config
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="최대 에폭 수"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="배치 크기 (GTX 1660: 1 권장)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        default=[96, 96, 96],
        help="ROI 크기 (D H W)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split 비율"
    )
    parser.add_argument(
        "--pos-ratio",
        type=float,
        default=0.7,
        help="Positive sample 비율"
    )
    parser.add_argument(
        "--cache-rate",
        type=float,
        default=0.1,
        help="Cache rate (메모리 부족 시 낮추기)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # Inference config
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold"
    )
    parser.add_argument(
        "--min-diameter",
        type=float,
        default=3.0,
        help="Minimum nodule diameter (mm)"
    )
    
    # Options
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
