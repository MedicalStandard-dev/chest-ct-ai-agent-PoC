#!/usr/bin/env python
# scripts/run_full_pipeline.py
"""
전체 파이프라인 실행 예제

단계:
1. Synthetic prior 생성 (Prior comparison 검증용)
2. LIDC 전처리 (실제 데이터 있을 때)
3. Lung segmentation 학습
4. Nodule detection 학습
5. API 서버 테스트

사용:
    # Dry run (synthetic만)
    python scripts/run_full_pipeline.py --dry-run
    
    # 전체 실행
    python scripts/run_full_pipeline.py \
        --lidc-root data/LIDC-IDRI \
        --msd-root data/Task06_Lung
"""
import argparse
import subprocess
from pathlib import Path
import sys

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import logger


def run_command(cmd: list, description: str, dry_run: bool = False):
    """명령 실행"""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {description}")
    logger.info(f"{'='*60}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        logger.info("[DRY RUN] Skipping execution")
        return True
    
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            check=True,
            capture_output=False
        )
        logger.info(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="전체 파이프라인 실행"
    )
    
    # Data paths
    parser.add_argument(
        "--lidc-root",
        type=Path,
        default=None,
        help="LIDC-IDRI 루트 디렉토리"
    )
    parser.add_argument(
        "--msd-root",
        type=Path,
        default=None,
        help="MSD Task06_Lung 디렉토리"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="출력 디렉토리"
    )
    
    # Options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="명령만 출력 (실행 안 함)"
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Synthetic prior 생성 스킵"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="모델 학습 스킵"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=10,
        help="LIDC 전처리 최대 케이스 수"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="학습 에폭 수 (테스트용 10)"
    )
    
    args = parser.parse_args()
    
    # Output directories
    output_dir = Path(args.output_dir)
    synthetic_dir = output_dir / "synthetic_priors"
    processed_lidc_dir = output_dir / "processed_lidc"
    lung_seg_model_dir = output_dir / "models" / "lung_seg"
    nodule_det_model_dir = output_dir / "models" / "nodule_det"
    
    logger.info("=" * 60)
    logger.info("FULL PIPELINE EXECUTION")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    
    steps_completed = []
    
    # Step 1: Synthetic prior generation
    if not args.skip_synthetic:
        success = run_command(
            [
                sys.executable, "scripts/generate_synthetic_priors.py",
                "--output", str(synthetic_dir),
                "--num-patients", "10",
                "--studies-per-patient", "3"
            ],
            "Synthetic prior generation",
            args.dry_run
        )
        if success:
            steps_completed.append("synthetic_priors")
    
    # Step 2: LIDC preprocessing
    if args.lidc_root and args.lidc_root.exists():
        success = run_command(
            [
                sys.executable, "scripts/preprocess_lidc.py",
                "--lidc-root", str(args.lidc_root),
                "--output", str(processed_lidc_dir),
                "--max-cases", str(args.max_cases)
            ],
            "LIDC preprocessing",
            args.dry_run
        )
        if success:
            steps_completed.append("lidc_preprocessing")
    else:
        logger.info("\n⏭️ Skipping LIDC preprocessing (no data path provided)")
    
    # Step 3: Lung segmentation training
    if not args.skip_training and args.msd_root and args.msd_root.exists():
        success = run_command(
            [
                sys.executable, "scripts/train_lung_segmentation.py",
                "--data-dir", str(args.msd_root),
                "--output-dir", str(lung_seg_model_dir),
                "--epochs", str(args.epochs),
                "--batch-size", "1"
            ],
            "Lung segmentation training",
            args.dry_run
        )
        if success:
            steps_completed.append("lung_seg_training")
    else:
        logger.info("\n⏭️ Skipping lung segmentation training")
    
    # Step 4: Nodule detection training
    if not args.skip_training and (processed_lidc_dir / "manifest.json").exists():
        success = run_command(
            [
                sys.executable, "scripts/train_nodule_detection.py",
                "--data-dir", str(processed_lidc_dir),
                "--output-dir", str(nodule_det_model_dir),
                "--epochs", str(args.epochs),
                "--batch-size", "1"
            ],
            "Nodule detection training",
            args.dry_run
        )
        if success:
            steps_completed.append("nodule_det_training")
    else:
        logger.info("\n⏭️ Skipping nodule detection training")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Steps completed: {', '.join(steps_completed) or 'None'}")
    logger.info(f"\nOutput files:")
    
    if "synthetic_priors" in steps_completed or args.dry_run:
        logger.info(f"  📁 {synthetic_dir}/")
        logger.info(f"     - all_studies.json")
        logger.info(f"     - prior_comparisons.json")
        logger.info(f"     - patient_histories/")
    
    if "lidc_preprocessing" in steps_completed or args.dry_run:
        logger.info(f"  📁 {processed_lidc_dir}/")
        logger.info(f"     - images/")
        logger.info(f"     - heatmaps/")
        logger.info(f"     - manifest.json")
    
    if "lung_seg_training" in steps_completed or args.dry_run:
        logger.info(f"  📁 {lung_seg_model_dir}/")
        logger.info(f"     - best_lung_seg_model.pth")
    
    if "nodule_det_training" in steps_completed or args.dry_run:
        logger.info(f"  📁 {nodule_det_model_dir}/")
        logger.info(f"     - best_nodule_det_model.pth")
    
    # Next steps
    logger.info("\n" + "=" * 60)
    logger.info("NEXT STEPS")
    logger.info("=" * 60)
    logger.info("1. Start API server:")
    logger.info("   python api/main.py")
    logger.info("")
    logger.info("2. Test API with synthetic data:")
    logger.info("   curl -X POST http://localhost:8000/api/v1/analyze \\")
    logger.info('     -H "Content-Type: application/json" \\')
    logger.info('     -d \'{"patient_id": "PT00001", "study_uid": "1.2.3.4.5", "include_report": true}\'')
    logger.info("")
    logger.info("3. View rendered report:")
    logger.info("   Check draft_report.rendered_report in response")


if __name__ == "__main__":
    main()
