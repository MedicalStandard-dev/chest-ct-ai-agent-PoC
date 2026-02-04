#!/usr/bin/env python
# scripts/preprocess_lidc.py
"""
LIDC-IDRI 전처리 스크립트

DICOM + XML annotations → NIfTI + Gaussian heatmap labels

사용:
    python scripts/preprocess_lidc.py \
        --lidc-root data/LIDC-IDRI \
        --output data/processed_lidc \
        --max-cases 100
"""
import argparse
from pathlib import Path
import sys

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from monai_pipeline.data_processing import LIDCPreprocessor
from utils.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="LIDC-IDRI 전처리 (DICOM → NIfTI + heatmap)"
    )
    
    parser.add_argument(
        "--lidc-root", "-l",
        type=Path,
        required=True,
        help="LIDC-IDRI 루트 디렉토리"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/processed_lidc"),
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--max-cases", "-n",
        type=int,
        default=None,
        help="최대 케이스 수 (테스트용)"
    )
    
    # Preprocessing config
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Target spacing (Z Y X) in mm"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Gaussian sigma for heatmap"
    )
    parser.add_argument(
        "--agreement",
        type=int,
        default=2,
        help="Minimum radiologist agreement for nodule"
    )
    parser.add_argument(
        "--min-diameter",
        type=float,
        default=3.0,
        help="Minimum nodule diameter (mm)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("LIDC-IDRI PREPROCESSING")
    logger.info("=" * 60)
    logger.info(f"Input: {args.lidc_root}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Max cases: {args.max_cases or 'All'}")
    
    # Config
    config = {
        "target_spacing": tuple(args.spacing),
        "gaussian_sigma": args.sigma,
        "nodule_agreement_threshold": args.agreement,
        "min_diameter_mm": args.min_diameter
    }
    
    logger.info(f"Config: {config}")
    
    # Initialize preprocessor
    preprocessor = LIDCPreprocessor(config=config)
    
    # Process dataset
    results = preprocessor.process_dataset(
        lidc_root=Path(args.lidc_root),
        output_dir=Path(args.output),
        max_cases=args.max_cases
    )
    
    # Summary
    total_nodules = sum(r["num_nodules"] for r in results)
    
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETED")
    logger.info(f"Cases processed: {len(results)}")
    logger.info(f"Total nodules: {total_nodules}")
    logger.info(f"Output directory: {args.output}")
    logger.info("=" * 60)
    
    print(f"\n✅ Processed {len(results)} cases with {total_nodules} nodules")
    print(f"📁 Output: {args.output}")
    print("\nNext step:")
    print(f"  python scripts/train_nodule_detection.py --data-dir {args.output}")


if __name__ == "__main__":
    main()
