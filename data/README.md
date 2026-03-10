# Data Directory Structure

```
data/
├── LIDC-IDRI/                  # LIDC 원본 데이터 (DICOM + XML)
│
├── LIDC-preprocessed-v2/       # LIDC 전처리 결과 (현행, 학습에 사용)
│   └── {case_id}/
│       ├── image.nii.gz        # CT volume (1mm isotropic)
│       └── label.nii.gz        # 결절 heatmap GT
│
├── auxneg-preprocessed/        # 보조 음성 샘플 (거짓 양성 억제용)
│   └── {case_id}/
│       ├── image.nii.gz
│       └── label.nii.gz        # 음성 레이블 (결절 없음)
│
├── Task06_Lung/                # MSD 폐 분할 학습 데이터
│   ├── imagesTr/               # 학습 CT volumes
│   ├── labelsTr/               # 폐 분할 GT masks
│   └── imagesTs/               # 테스트 CT volumes
│
├── lunit_clean/                # Lunit 외부 검증 데이터
│
├── dicom_clean/                # 전처리된 DICOM (CT only, clean)
├── dicom_clean_ct_unknown/     # CT 여부 불명확한 DICOM
├── dicom_clean_full/           # 전체 전처리된 DICOM
├── dicom_data/                 # 원본 DICOM 데이터
│
├── dicom_storage/              # API 업로드 임시 저장소 (런타임)
├── dicom_output/               # 출력 DICOM 저장소 (런타임)
└── chroma_db/                  # ChromaDB 벡터 DB (런타임 자동 생성)
```

---

## 데이터셋 소개

### LIDC-IDRI (폐결절 검출)
- 출처: [Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
- 구성: ~1000 케이스 CT + 4명 방사선과 의사 annotation
- 사용 범위: ~200 케이스 서브셋 (학습/검증/테스트)
- 분할: patient-level, fixed seed

### MSD Task06_Lung (폐 분할)
- 출처: [Medical Segmentation Decathlon](https://medicaldecathlon.com/)
- 구성: 64 케이스 (CT + 폐 분할 mask)
- 용도: Lung Segmentation 학습 전용

### Auxiliary Negatives (거짓 양성 억제)
- LUNA16 등에서 추출한 결절 없는 CT 패치
- 학습 시 음성 샘플 균형 조정 목적

---

## 전처리 실행

### LIDC 전처리
```bash
python scripts/preprocess_lidc.py \
    --lidc-root data/LIDC-IDRI \
    --output data/LIDC-preprocessed-v2
```

### Synthetic Prior 생성 (RAG 테스트용)
```bash
python scripts/generate_synthetic_priors.py \
    --output data/processed/synthetic_priors
```

---

## 주의사항

- 원본 데이터(`LIDC-IDRI/`, `Task06_Lung/`)는 용량이 크며 Git에 포함하지 않음
- `chroma_db/`는 서버 실행 시 자동 생성됨
- `dicom_storage/`, `dicom_output/`는 런타임 임시 디렉토리
- `LIDC-preprocessed/` (v1, 구버전)은 `to_delete_later/`로 이동 예정
