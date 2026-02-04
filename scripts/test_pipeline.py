#!/usr/bin/env python
# scripts/test_pipeline.py
"""
Production Pipeline 테스트

학습된 Nodule Detection 모델로 전체 파이프라인 테스트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import nibabel as nib
import json

from monai_pipeline.production_pipeline import ProductionPipeline, create_pipeline
from utils.logger import logger


def test_with_real_data():
    """실제 데이터로 테스트"""
    logger.info("=" * 60)
    logger.info("PRODUCTION PIPELINE TEST")
    logger.info("=" * 60)
    
    # 경로 설정
    model_path = Path("models/nodule_det/best_nodule_det_model.pth")
    data_dir = Path("data/processed/lidc")
    output_dir = Path("outputs/pipeline_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 모델 존재 확인
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        logger.info("Running with mock heatmap...")
        model_path = None
    else:
        logger.info(f"Model: {model_path}")
    
    # 테스트 데이터 로드
    test_images = sorted((data_dir / "images").glob("*.nii.gz"))[:3]
    
    if not test_images:
        logger.error("No test images found!")
        return
    
    logger.info(f"Test images: {len(test_images)}")
    
    # 파이프라인 생성
    pipeline = create_pipeline(
        nodule_model_path=str(model_path) if model_path else None,
        output_dir=str(output_dir)
    )
    
    # 각 이미지 처리
    results_summary = []
    
    for img_path in test_images:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing: {img_path.name}")
        logger.info(f"{'='*40}")
        
        # 이미지 로드
        nii = nib.load(img_path)
        volume = nii.get_fdata().astype(np.float32)
        spacing = tuple(nii.header.get_zooms()[:3])
        
        logger.info(f"Volume shape: {volume.shape}")
        logger.info(f"Spacing: {spacing}")
        
        # Annotation 로드 (있으면)
        ann_path = data_dir / "annotations" / f"{img_path.stem.replace('.nii', '')}.json"
        gt_nodules = 0
        if ann_path.exists():
            with open(ann_path) as f:
                ann = json.load(f)
                gt_nodules = len(ann.get("nodules", []))
        
        # 파이프라인 실행
        result = pipeline.process_volume(
            volume=volume,
            spacing_mm=spacing,
            series_uid=img_path.stem,
            patient_id=img_path.stem.split("_")[0] if "_" in img_path.stem else img_path.stem
        )
        
        # 결과 출력
        logger.info(f"\n--- Results ---")
        logger.info(f"Ground truth nodules: {gt_nodules}")
        logger.info(f"Detected candidates: {result.total_candidates}")
        logger.info(f"  - Findings: {result.findings_count}")
        logger.info(f"  - Limitations: {result.limitations_count}")
        logger.info(f"Processing time: {result.processing_time_ms:.0f}ms")
        
        # Findings Table
        if result.findings_table:
            logger.info(f"\nFINDINGS TABLE:")
            for row in result.findings_table[:5]:
                logger.info(f"  {row['type']} | {row['location']} | "
                           f"conf={row['confidence']:.2f} | {row['status']}")
        
        # Measurements Table
        if result.measurements_table:
            logger.info(f"\nMEASUREMENTS TABLE:")
            for row in result.measurements_table[:5]:
                logger.info(f"  {row['lesion_id']} | {row['location']} | "
                           f"{row['diameter_mm']:.1f}mm | {row['volume_mm3']:.0f}mm³")
        
        # Key Flags
        logger.info(f"\nKEY FLAGS: {result.key_flags}")
        
        # 결과 저장
        case_output = output_dir / img_path.stem
        case_output.mkdir(exist_ok=True)
        
        with open(case_output / "result.json", "w") as f:
            json.dump(result.to_structured_result(), f, indent=2, default=str)
        
        results_summary.append({
            "case": img_path.stem,
            "gt_nodules": gt_nodules,
            "detected": result.total_candidates,
            "findings": result.findings_count,
            "time_ms": result.processing_time_ms
        })
    
    # 전체 요약
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    
    total_gt = sum(r["gt_nodules"] for r in results_summary)
    total_detected = sum(r["detected"] for r in results_summary)
    total_findings = sum(r["findings"] for r in results_summary)
    avg_time = np.mean([r["time_ms"] for r in results_summary])
    
    logger.info(f"Cases processed: {len(results_summary)}")
    logger.info(f"Total GT nodules: {total_gt}")
    logger.info(f"Total detected: {total_detected}")
    logger.info(f"Total findings: {total_findings}")
    logger.info(f"Avg processing time: {avg_time:.0f}ms")
    
    # 요약 저장
    with open(output_dir / "summary.json", "w") as f:
        json.dump({
            "cases": results_summary,
            "total_gt": total_gt,
            "total_detected": total_detected,
            "total_findings": total_findings,
            "avg_time_ms": avg_time
        }, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_dir}")


def test_mock():
    """Mock 데이터로 빠른 테스트"""
    logger.info("Running mock test...")
    
    # Mock 볼륨 생성
    volume = np.random.randn(128, 256, 256).astype(np.float32) * 0.1
    spacing = (1.0, 1.0, 1.0)
    
    pipeline = create_pipeline()
    
    result = pipeline.process_volume(
        volume=volume,
        spacing_mm=spacing,
        series_uid="mock_test_001"
    )
    
    logger.info(f"Candidates: {result.total_candidates}")
    logger.info(f"Findings: {result.findings_count}")
    logger.info(f"Time: {result.processing_time_ms:.0f}ms")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run mock test only")
    args = parser.parse_args()
    
    if args.mock:
        test_mock()
    else:
        test_with_real_data()
