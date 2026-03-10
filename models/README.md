# Models Directory

```
models/
├── nodule_det/                  # Custom UNet Heatmap 모델 (보조)
│   ├── best_nodule_det_model.pth
│   ├── final_nodule_det_model.pth
│   └── checkpoint.pth
│
├── lung_seg/                    # Custom DynUNet 폐 분할 모델
│   ├── best_lung_seg_model.pth
│   ├── final_lung_seg_model.pth
│   └── checkpoint.pth
│
└── luna16_retinanet/            # MONAI LUNA16 Pretrained RetinaNet (주 검출기)
    ├── lung_nodule_ct_detection/     # 압축 해제된 MONAI bundle
    └── lung_nodule_ct_detection_v0.6.8.zip
```

---

## 모델 설명

### 1. LUNA16 RetinaNet (주 검출기)
- **아키텍처**: ResNet50 + FPN + RetinaNet (3D)
- **출처**: MONAI Model Zoo pretrained
- **성능**: mAP ~0.852 @ LUNA16 fold0
- **사용 방법**: `models/luna16_retinanet/lung_nodule_ct_detection/` 경로가 존재하면 자동 활성화

### 2. UNet Heatmap (보조/대체)
- **아키텍처**: MONAI UNet
- **학습 데이터**: LIDC-IDRI + auxneg 보조 음성 샘플
- **상태**: 학습됨 (성능 개선 필요)
- **사용 조건**: luna16_retinanet 없을 때 대체 사용

### 3. DynUNet (폐 분할)
- **아키텍처**: MONAI DynUNet
- **학습 데이터**: MSD Task06_Lung
- **상태**: 학습됨 (Best Dice ~0.23, 개선 필요)
- **역할**: 결절 위치 판정 (폐엽: RUL/RML/RLL/LUL/LLL)

---

## 모델 학습

### Lung Segmentation 재학습
```bash
python scripts/train_lung_segmentation.py \
    --data-dir data/Task06_Lung \
    --output-dir models/lung_seg \
    --epochs 100
```

### Nodule Detection 재학습
```bash
python scripts/train_nodule_detection.py \
    --data-dir data/LIDC-preprocessed-v2 \
    --output-dir models/nodule_det \
    --epochs 100
```

---

## 추론 사용법

### LUNA16 RetinaNet
```python
from monai_pipeline.luna16_detector import Luna16Detector

detector = Luna16Detector(
    bundle_dir="models/luna16_retinanet/lung_nodule_ct_detection"
)
candidates = detector.detect(volume_np, spacing, series_uid)
```

### Lung Segmentation
```python
from monai_pipeline.lung_segmentation import LungSegmentationInference

inference = LungSegmentationInference(
    model_path="models/lung_seg/best_lung_seg_model.pth"
)
mask = inference.predict(volume)
```

---

## 주의사항

- 모델 파일(`.pth`)은 Git에 포함하지 않음 (`.gitignore`)
- LUNA16 bundle은 MONAI Model Zoo에서 다운로드 후 `models/luna16_retinanet/`에 배치
- `USE_MOCK_VISION=true` 설정 시 실제 모델 없이도 mock으로 동작 가능
