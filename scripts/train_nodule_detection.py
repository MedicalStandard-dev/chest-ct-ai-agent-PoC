#!/usr/bin/env python
# scripts/train_nodule_detection.py
"""
Nodule Detection 학습 스크립트 (LIDC-IDRI)

Heatmap 기반 approach:
- 3D UNet / SegResNet
- Gaussian blob labels
- MSE loss for heatmap regression

RTX 3070 Ti (8GB) 최적화:
- patch_size: (96, 96, 96)
- batch_size: 2
- AMP: enabled
- cache_rate: 0.5

사용:
    # 1. LIDC 전처리 먼저 실행
    python scripts/preprocess_lidc.py --lidc-root data/LIDC-IDRI --output data/LIDC-preprocessed

    # 2. 새로 학습
    python scripts/train_nodule_detection.py \
        --data-dir data/LIDC-preprocessed \
        --output-dir models/nodule_det

    # 3. 이어서 학습 (끊긴 경우)
    python scripts/train_nodule_detection.py \
        --data-dir data/LIDC-preprocessed \
        --output-dir models/nodule_det \
        --resume
"""
import argparse
import os
from pathlib import Path
import sys

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from monai_pipeline.nodule_detection import (
    NoduleDetectionTrainer,
    NoduleDetectionInference,
    apply_parameter_efficient_finetuning,
    count_parameters,
)
from monai_pipeline.data_processing import create_training_split
from utils.logger import logger


def _pid_exists(pid: int) -> bool:
    """Return True if a process with given pid exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _acquire_train_lock(output_dir: Path) -> Path:
    """
    Prevent concurrent training runs in the same output directory.
    Stale lock files are cleaned up automatically.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".train.lock"

    if lock_path.exists():
        try:
            existing_pid = int(lock_path.read_text().strip() or "0")
        except (ValueError, OSError):
            existing_pid = 0

        if existing_pid > 0 and _pid_exists(existing_pid):
            raise RuntimeError(
                f"Another training process is already running "
                f"(pid={existing_pid}, output_dir={output_dir})."
            )
        lock_path.unlink(missing_ok=True)

    lock_path.write_text(str(os.getpid()))
    return lock_path


def train(args):
    """학습 실행"""
    logger.info("=" * 60)
    logger.info("NODULE DETECTION TRAINING (RTX 3070 Ti)")
    logger.info("=" * 60)

    # Load data
    manifest_path = (
        Path(args.manifest_path)
        if args.manifest_path is not None
        else Path(args.data_dir) / "manifest.json"
    )
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        logger.error("Run preprocess_lidc.py first!")
        return None

    train_files, val_files = create_training_split(
        manifest_path=manifest_path,
        train_ratio=1 - args.val_split,
        seed=args.seed,
        require_nodules=not args.include_negatives,
    )

    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Include negatives: {args.include_negatives}")

    logger.info(f"Train samples: {len(train_files)}")
    logger.info(f"Val samples: {len(val_files)}")

    # Config
    config = {
        "roi_size": tuple(args.roi_size),
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "amp_enabled": args.amp,
        "num_workers": args.num_workers,
        "train_num_samples": args.train_num_samples,
        "pin_memory": args.pin_memory,
        "val_sw_batch_size": args.val_sw_batch_size,
        "pos_sample_ratio": args.pos_ratio,
        "early_stopping_patience": args.patience,
        "val_interval": args.val_interval,
        "selection_metric": args.selection_metric,
        "detection_eval_threshold": args.detection_eval_threshold,
        "detection_gt_threshold": args.detection_gt_threshold,
        "detection_match_distance_mm": args.detection_match_distance_mm,
        "detection_fp_weight": args.detection_fp_weight,
    }

    logger.info(f"Config: {config}")
    logger.info(f"Model type: {args.model_type}")

    lock_path = _acquire_train_lock(Path(args.output_dir))
    try:
        # Initialize trainer
        trainer = NoduleDetectionTrainer(
            config=config,
            model_type=args.model_type
        )

        # Finetuning mode (full / LoRA / QLoRA)
        replaced_layers = 0
        if args.finetune_mode in {"lora", "qlora"}:
            replaced_layers = apply_parameter_efficient_finetuning(
                model=trainer.model,
                mode=args.finetune_mode,
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                qlora_bits=args.qlora_bits,
            )
            trainer.reset_optimizer_for_current_trainable_params(
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                t_max=args.epochs,
            )
        elif args.finetune_mode == "full":
            for p in trainer.model.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unsupported finetune_mode: {args.finetune_mode}")

        total_params, trainable_params = count_parameters(trainer.model)
        logger.info(
            f"Finetune mode={args.finetune_mode} | "
            f"trainable/total={trainable_params:,}/{total_params:,} "
            f"({trainable_params / total_params * 100:.2f}%)"
        )
        if replaced_layers:
            logger.info(f"Adapter replaced Conv3d layers: {replaced_layers}")

        # Train
        result = trainer.train(
            train_files=train_files,
            val_files=val_files,
            output_dir=Path(args.output_dir),
            resume=args.resume
        )

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED")
        logger.info(f"Best loss: {result['best_loss']:.6f}")
        if "best_metric_value" in result:
            logger.info(
                f"Best {result.get('selection_metric', 'metric')}: "
                f"{result['best_metric_value']:.6f}"
            )
        logger.info(f"Model saved to: {args.output_dir}")
        logger.info("=" * 60)

        return result
    finally:
        lock_path.unlink(missing_ok=True)


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
        min_diameter_mm=args.min_diameter,
        max_diameter_mm=(None if args.max_diameter <= 0 else args.max_diameter),
        max_volume_mm3=(None if args.max_volume <= 0 else args.max_volume),
        restrict_to_lung_mask=args.restrict_to_lung_mask,
        lung_seg_model_path=args.lung_seg_model_path,
        finetune_mode=args.finetune_mode,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        qlora_bits=args.qlora_bits,
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
        description="Nodule detection 학습 (LIDC-IDRI, RTX 3070 Ti)"
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
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="학습 manifest 경로 (기본값: --data-dir/manifest.json)",
    )

    # Model config
    parser.add_argument(
        "--model-type", "-m",
        type=str,
        choices=["unet", "segresnet"],
        default="unet",
        help="모델 유형"
    )
    parser.add_argument(
        "--finetune-mode",
        type=str,
        choices=["full", "lora", "qlora"],
        default="full",
        help="파인튜닝 방식: full / lora / qlora",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA/QLoRA adapter rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA/QLoRA adapter alpha",
    )
    parser.add_argument(
        "--qlora-bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="QLoRA base weight quantization bits",
    )

    # Training config (RTX 3070 Ti defaults)
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=200,
        help="최대 에폭 수 (default: 200)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=2,
        help="배치 크기 (OOM 방지 기본값: 2)"
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
        "--pos-ratio",
        type=float,
        default=0.7,
        help="Positive sample 비율"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (OOM 방지 기본값: 0)"
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=1,
        help="한 volume에서 추출할 crop 개수 (OOM 방지 기본값: 1)"
    )
    parser.add_argument(
        "--val-sw-batch-size",
        type=int,
        default=1,
        help="validation sliding-window batch size"
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="DataLoader pin_memory 활성화 (기본: 비활성)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (val_interval 단위)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--include-negatives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="num_nodules=0(음성) 케이스도 학습에 포함 (기본: 포함)",
    )

    # Inference config
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold"
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=["val_loss", "detection_proxy"],
        default="detection_proxy",
        help="Best 모델 선택 기준",
    )
    parser.add_argument(
        "--detection-eval-threshold",
        type=float,
        default=0.9,
        help="detection_proxy 계산 시 prediction threshold",
    )
    parser.add_argument(
        "--detection-gt-threshold",
        type=float,
        default=0.4,
        help="detection_proxy 계산 시 GT heatmap threshold",
    )
    parser.add_argument(
        "--detection-match-distance-mm",
        type=float,
        default=6.0,
        help="detection_proxy 계산 시 매칭 거리(mm)",
    )
    parser.add_argument(
        "--detection-fp-weight",
        type=float,
        default=0.05,
        help="detection_proxy = sensitivity - weight * fp_per_scan",
    )
    parser.add_argument(
        "--min-diameter",
        type=float,
        default=3.0,
        help="Minimum nodule diameter (mm)"
    )
    parser.add_argument(
        "--max-diameter",
        type=float,
        default=0.0,
        help="Maximum nodule diameter (mm), 0이면 비활성화",
    )
    parser.add_argument(
        "--max-volume",
        type=float,
        default=0.0,
        help="Maximum nodule volume (mm3), 0이면 비활성화",
    )
    parser.add_argument(
        "--restrict-to-lung-mask",
        action="store_true",
        help="폐 마스크 내부 후보만 유지",
    )
    parser.add_argument(
        "--lung-seg-model-path",
        type=Path,
        default=None,
        help="폐 분할 모델 경로 (없으면 intensity heuristic 사용)",
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
