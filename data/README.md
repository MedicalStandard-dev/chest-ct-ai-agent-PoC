# Data Directory Structure

```
data/
├── raw/                      # 원본 데이터 (다운로드)
│   ├── LIDC-IDRI/           # LIDC-IDRI 원본 DICOM + XML
│   └── Task06_Lung/         # MSD Task06_Lung 원본 NIfTI
│
├── processed/               # 전처리된 데이터
│   ├── lidc/               # LIDC 전처리 결과 (NIfTI + heatmap)
│   └── synthetic_priors/   # Synthetic prior 데이터
│
├── dicom_storage/          # DICOM 입력 저장소
├── dicom_output/           # DICOM 출력 저장소
└── chroma_db/              # ChromaDB 벡터 데이터베이스
```

## 데이터 다운로드

### LIDC-IDRI (Nodule Detection)
```bash
# 수동 다운로드 필요
# https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
# 다운로드 후 data/raw/LIDC-IDRI/ 에 배치
```

### MSD Task06_Lung (Lung Segmentation)
```bash
# 수동 다운로드 필요
# https://medicaldecathlon.com/
# 다운로드 후 data/raw/Task06_Lung/ 에 배치
```

## 전처리 실행

### LIDC 전처리
```bash
python scripts/preprocess_lidc.py \
    --lidc-root data/raw/LIDC-IDRI \
    --output data/processed/lidc
```

### Synthetic Prior 생성
```bash
python scripts/generate_synthetic_priors.py \
    --output data/processed/synthetic_priors
```

## 주의사항

- `raw/` 폴더의 원본 데이터는 Git에 포함하지 않음 (.gitignore)
- `processed/` 폴더의 전처리 결과도 용량이 커서 Git에 포함하지 않음
- `chroma_db/`는 런타임에 자동 생성됨
