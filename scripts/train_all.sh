#!/bin/bash
cd /home/jiwoonkim/jiwoon/medical-ai-pacs
source venv/bin/activate

echo "=== Starting Nodule Detection Training ==="
python scripts/train_nodule_detection.py \
    --data-dir data/processed/lidc \
    --output-dir models/nodule_det \
    --epochs 50 \
    --batch-size 1

echo "=== Starting Lung Segmentation Training ==="
python scripts/train_lung_segmentation.py \
    --data-dir data/raw/Task06_Lung \
    --output-dir models/lung_seg \
    --epochs 50 \
    --batch-size 1

echo "=== All Training Complete ==="
