# Models Directory

```
models/
├── lung_seg/           # Lung Segmentation 모델
│   ├── best_lung_seg_model.pth
│   └── final_lung_seg_model.pth
│
├── nodule_det/         # Nodule Detection 모델
│   ├── best_nodule_det_model.pth
│   └── final_nodule_det_model.pth
│
└── findings/           # Findings Classifier 모델 (선택)
    └── findings_classifier.pth
```

## 모델 학습

### Lung Segmentation
```bash
python scripts/train_lung_segmentation.py \
    --data-dir data/raw/Task06_Lung \
    --output-dir models/lung_seg \
    --epochs 100
```

### Nodule Detection
```bash
python scripts/train_nodule_detection.py \
    --data-dir data/processed/lidc \
    --output-dir models/nodule_det \
    --epochs 100
```

## 모델 사용

### Lung Segmentation Inference
```python
from monai_pipeline.lung_segmentation import LungSegmentationInference

inference = LungSegmentationInference(
    model_path="models/lung_seg/best_lung_seg_model.pth"
)
mask = inference.predict(volume)
```

### Nodule Detection Inference
```python
from monai_pipeline.nodule_detection import NoduleDetectionInference

inference = NoduleDetectionInference(
    model_path="models/nodule_det/best_nodule_det_model.pth"
)
candidates = inference.detect(volume, spacing, series_uid)
```

## 주의사항

- 모델 파일(.pth)은 Git에 포함하지 않음 (.gitignore)
- MONAI pretrained 모델 사용 시 처음 실행 시 자동 다운로드됨
