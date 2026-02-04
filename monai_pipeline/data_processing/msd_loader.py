# monai_pipeline/data_processing/msd_loader.py
"""
MSD (Medical Segmentation Decathlon) Data Loader
Task06_Lung (Lung tumor segmentation) / Task09_Spleen / etc.
"""
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json

from utils.logger import logger


class MSDLoader:
    """
    MSD (Medical Segmentation Decathlon) 데이터 로더
    
    MSD 구조:
    - Task06_Lung/
      - imagesTr/
        - lung_001.nii.gz
        - ...
      - labelsTr/
        - lung_001.nii.gz
        - ...
      - imagesTs/ (optional)
      - dataset.json
    
    사용:
    ```python
    loader = MSDLoader()
    train_files, val_files = loader.load_dataset(
        data_dir="path/to/Task06_Lung",
        val_split=0.2
    )
    ```
    """
    
    SUPPORTED_TASKS = {
        "Task06_Lung": {
            "name": "Lung Tumor Segmentation",
            "modality": "CT",
            "labels": {"0": "background", "1": "lung tumor"}
        },
        "Task09_Spleen": {
            "name": "Spleen Segmentation",
            "modality": "CT",
            "labels": {"0": "background", "1": "spleen"}
        },
        "Task03_Liver": {
            "name": "Liver and Tumor Segmentation",
            "modality": "CT",
            "labels": {"0": "background", "1": "liver", "2": "tumor"}
        }
    }
    
    def __init__(self):
        logger.info("MSDLoader initialized")
    
    def load_dataset(
        self,
        data_dir: Path,
        val_split: float = 0.2,
        seed: int = 42
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Load MSD dataset and split into train/val
        
        Args:
            data_dir: Path to MSD task directory (e.g., Task06_Lung)
            val_split: Validation split ratio
            seed: Random seed for reproducibility
            
        Returns:
            (train_files, val_files) in MONAI format:
            [{"image": "path/to/image.nii.gz", "label": "path/to/label.nii.gz"}, ...]
        """
        import numpy as np
        np.random.seed(seed)
        
        data_dir = Path(data_dir)
        
        # Load dataset.json if exists
        dataset_json = data_dir / "dataset.json"
        if dataset_json.exists():
            with open(dataset_json) as f:
                metadata = json.load(f)
            logger.info(f"Loaded dataset: {metadata.get('name', 'Unknown')}")
        else:
            metadata = {}
        
        # Find training images and labels
        images_dir = data_dir / "imagesTr"
        labels_dir = data_dir / "labelsTr"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Build file list
        data_dicts = []
        
        # Method 1: From dataset.json training list
        if "training" in metadata:
            for item in metadata["training"]:
                image_path = data_dir / item["image"].lstrip("./")
                label_path = data_dir / item["label"].lstrip("./")
                
                if image_path.exists() and label_path.exists():
                    data_dicts.append({
                        "image": str(image_path),
                        "label": str(label_path)
                    })
        
        # Method 2: Direct directory scan
        if not data_dicts:
            image_files = sorted(images_dir.glob("*.nii.gz"))
            
            for img_path in image_files:
                label_path = labels_dir / img_path.name
                if label_path.exists():
                    data_dicts.append({
                        "image": str(img_path),
                        "label": str(label_path)
                    })
        
        if not data_dicts:
            raise ValueError(f"No valid image-label pairs found in {data_dir}")
        
        logger.info(f"Found {len(data_dicts)} training samples")
        
        # Shuffle and split
        np.random.shuffle(data_dicts)
        
        n_val = int(len(data_dicts) * val_split)
        train_files = data_dicts[:-n_val] if n_val > 0 else data_dicts
        val_files = data_dicts[-n_val:] if n_val > 0 else []
        
        logger.info(f"Split: {len(train_files)} train, {len(val_files)} val")
        
        return train_files, val_files
    
    def load_test_data(self, data_dir: Path) -> List[Dict]:
        """
        Load MSD test data (images only, no labels)
        
        Args:
            data_dir: Path to MSD task directory
            
        Returns:
            List of {"image": "path/to/image.nii.gz"}
        """
        data_dir = Path(data_dir)
        images_dir = data_dir / "imagesTs"
        
        if not images_dir.exists():
            logger.warning(f"Test directory not found: {images_dir}")
            return []
        
        test_files = [
            {"image": str(f)}
            for f in sorted(images_dir.glob("*.nii.gz"))
        ]
        
        logger.info(f"Found {len(test_files)} test samples")
        return test_files
    
    def get_dataset_info(self, data_dir: Path) -> Dict:
        """Get dataset metadata"""
        data_dir = Path(data_dir)
        dataset_json = data_dir / "dataset.json"
        
        if dataset_json.exists():
            with open(dataset_json) as f:
                return json.load(f)
        
        # Infer from directory name
        task_name = data_dir.name
        if task_name in self.SUPPORTED_TASKS:
            return self.SUPPORTED_TASKS[task_name]
        
        return {"name": task_name, "modality": "Unknown", "labels": {}}


class LungSegmentationDataset:
    """
    Lung segmentation 전용 데이터셋 래퍼
    
    MSD Task06_Lung 또는 유사 데이터셋 지원
    """
    
    def __init__(self, data_dir: Path, val_split: float = 0.2, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.loader = MSDLoader()
        
        self.train_files, self.val_files = self.loader.load_dataset(
            data_dir=data_dir,
            val_split=val_split,
            seed=seed
        )
        
        self.info = self.loader.get_dataset_info(data_dir)
    
    def get_train_files(self) -> List[Dict]:
        """Get training file list"""
        return self.train_files
    
    def get_val_files(self) -> List[Dict]:
        """Get validation file list"""
        return self.val_files
    
    def get_all_files(self) -> List[Dict]:
        """Get all training files (train + val)"""
        return self.train_files + self.val_files
    
    def __len__(self) -> int:
        return len(self.train_files) + len(self.val_files)
    
    def __repr__(self) -> str:
        return (
            f"LungSegmentationDataset("
            f"train={len(self.train_files)}, "
            f"val={len(self.val_files)}, "
            f"path={self.data_dir})"
        )


def download_msd_task(
    task_name: str,
    output_dir: Path,
    extract: bool = True
) -> Path:
    """
    Download MSD task from official source
    
    NOTE: This is a placeholder - actual download requires
    manual registration and download from:
    https://medicaldecathlon.com/
    
    Args:
        task_name: e.g., "Task06_Lung"
        output_dir: Directory to save dataset
        extract: Whether to extract archive
        
    Returns:
        Path to extracted dataset
    """
    logger.info(
        f"MSD datasets must be downloaded manually from https://medicaldecathlon.com/\n"
        f"Please download {task_name} and extract to {output_dir}"
    )
    
    return output_dir / task_name
