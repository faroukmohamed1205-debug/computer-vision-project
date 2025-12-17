"""
Enhanced Brain Tumor Segmentation Dataset - PyTorch Version
Integrates advanced preprocessing and augmentation
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional
import cv2
from PIL import Image
import logging

from src.preprocessing import ImagePreprocessor, MaskPreprocessor
from src.augmentation import ImageAugmentor

logger = logging.getLogger(__name__)


def extract_patient_id(filename: str) -> str:
    """
    Extract patient ID from filename.
    Example: 'TCGA_CS_4941_19960909_12.tif' -> 'TCGA_CS_4941_19960909'
    """
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) >= 4:
        return '_'.join(parts[:4])
    return basename.split('.')[0]


def load_data_paths(data_dir: str) -> Tuple[List[str], List[str], Dict[str, List[int]]]:
    """
    Load all image and mask paths from the dataset directory.
    
    Args:
        data_dir: Root directory of LGG-MRI dataset
        
    Returns:
        image_paths, mask_paths, patient_groups
    """
    try:
        possible_patterns = [
            os.path.join(data_dir, '*', '*_mask.tif'),
            os.path.join(data_dir, 'lgg-mri-segmentation', 'kaggle_3m', '*', '*_mask.tif'),
            os.path.join(data_dir, 'kaggle_3m', '*', '*_mask.tif'),
        ]
        
        mask_paths = []
        for pattern in possible_patterns:
            mask_paths = sorted(glob.glob(pattern))
            if mask_paths:
                break
        
        if not mask_paths:
            raise FileNotFoundError(f"No mask files found in {data_dir}")
        
        image_paths = [p.replace('_mask.tif', '.tif') for p in mask_paths]
        
        missing = [p for p in image_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} image files")
        
        patient_groups = {}
        for idx, img_path in enumerate(image_paths):
            patient_id = extract_patient_id(img_path)
            if patient_id not in patient_groups:
                patient_groups[patient_id] = []
            patient_groups[patient_id].append(idx)
        
        print(f"✓ Loaded {len(image_paths)} image-mask pairs")
        print(f"✓ Found {len(patient_groups)} unique patients")
        print(f"✓ Average {len(image_paths)/len(patient_groups):.1f} slices per patient")
        
        return image_paths, mask_paths, patient_groups
        
    except Exception as e:
        raise RuntimeError(f"Error loading data: {str(e)}")


def split_patients(
    patient_groups: Dict[str, List[int]], 
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[List[int], List[int], List[int], List[str], List[str], List[str]]:
    """
    Split patients into train/val/test sets (NO DATA LEAKAGE).
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"
    
    patient_ids = list(patient_groups.keys())
    
    train_patients, temp_patients = train_test_split(
        patient_ids, train_size=train_size, random_state=random_state
    )
    
    val_ratio = val_size / (val_size + test_size)
    val_patients, test_patients = train_test_split(
        temp_patients, train_size=val_ratio, random_state=random_state
    )
    
    train_indices = [idx for pid in train_patients for idx in patient_groups[pid]]
    val_indices = [idx for pid in val_patients for idx in patient_groups[pid]]
    test_indices = [idx for pid in test_patients for idx in patient_groups[pid]]
    
    print(f"\n📊 Patient-wise Split:")
    print(f"  Train: {len(train_patients)} patients, {len(train_indices)} slices")
    print(f"  Val:   {len(val_patients)} patients, {len(val_indices)} slices")
    print(f"  Test:  {len(test_patients)} patients, {len(test_indices)} slices")
    
    return train_indices, val_indices, test_indices, train_patients, val_patients, test_patients


class EnhancedBrainTumorDataset(Dataset):
    """
    Enhanced PyTorch Dataset with advanced preprocessing and augmentation.
    """
    
    def __init__(
        self, 
        image_paths: List[str],
        mask_paths: List[str],
        indices: List[int],
        target_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
        augment_mode: str = 'light',
        use_preprocessing: bool = True,
        use_zscore: bool = True,
        use_bias_correction: bool = True,
        use_clahe: bool = True
    ):
        """
        Args:
            image_paths: Full list of image paths
            mask_paths: Full list of mask paths
            indices: Indices to select
            target_size: Target size for resizing
            augment: Whether to apply augmentation
            augment_mode: 'light' or 'heavy'
            use_preprocessing: Use advanced preprocessing
            use_zscore: Use Z-score normalization
            use_bias_correction: Apply bias field correction
            use_clahe: Apply CLAHE
        """
        self.image_paths = [image_paths[i] for i in indices]
        self.mask_paths = [mask_paths[i] for i in indices]
        self.target_size = target_size
        self.augment = augment
        self.use_preprocessing = use_preprocessing
        
        # Initialize preprocessors
        if use_preprocessing:
            self.image_preprocessor = ImagePreprocessor(
                target_size=target_size,
                use_zscore=use_zscore,
                use_bias_correction=use_bias_correction,
                use_clahe=use_clahe
            )
            self.mask_preprocessor = MaskPreprocessor(target_size=target_size)
        else:
            self.image_preprocessor = None
            self.mask_preprocessor = None
        
        # Initialize augmentor
        if augment:
            self.augmentor = ImageAugmentor(mode=augment_mode)
        else:
            self.augmentor = None
        
        logger.info(f"EnhancedBrainTumorDataset initialized:")
        logger.info(f"  - Samples: {len(self.image_paths)}")
        logger.info(f"  - Augmentation: {augment} (mode: {augment_mode if augment else 'N/A'})")
        logger.info(f"  - Advanced preprocessing: {use_preprocessing}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and preprocess
        if self.use_preprocessing:
            # Use advanced preprocessing
            image = self.image_preprocessor.preprocess_from_path(self.image_paths[idx])
            mask = self.mask_preprocessor.preprocess_from_path(self.mask_paths[idx])
        else:
            # Basic preprocessing
            image = cv2.imread(self.image_paths[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.target_size)
            image = image.astype(np.float32) / 255.0
            
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)
        
        # Apply augmentation
        if self.augment and self.augmentor is not None:
            image, mask = self.augmentor.augment(image, mask)
        
        # Convert to PyTorch tensors [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        
        return image, mask


def create_enhanced_data_loaders(
    image_paths: List[str],
    mask_paths: List[str],
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    batch_size: int = 16,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (256, 256),
    augment_mode: str = 'light',
    use_preprocessing: bool = True,
    use_zscore: bool = True,
    use_bias_correction: bool = True,
    use_clahe: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create enhanced PyTorch DataLoaders with advanced preprocessing.
    
    Args:
        image_paths: Full list of image paths
        mask_paths: Full list of mask paths
        train_indices: Training indices
        val_indices: Validation indices
        test_indices: Test indices
        batch_size: Batch size
        num_workers: Number of workers
        target_size: Target image size
        augment_mode: 'light' or 'heavy'
        use_preprocessing: Use advanced preprocessing
        use_zscore: Use Z-score normalization
        use_bias_correction: Apply bias field correction
        use_clahe: Apply CLAHE
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = EnhancedBrainTumorDataset(
        image_paths, mask_paths, train_indices,
        target_size=target_size,
        augment=True,
        augment_mode=augment_mode,
        use_preprocessing=use_preprocessing,
        use_zscore=use_zscore,
        use_bias_correction=use_bias_correction,
        use_clahe=use_clahe
    )
    
    val_dataset = EnhancedBrainTumorDataset(
        image_paths, mask_paths, val_indices,
        target_size=target_size,
        augment=False,
        use_preprocessing=use_preprocessing,
        use_zscore=use_zscore,
        use_bias_correction=use_bias_correction,
        use_clahe=use_clahe
    )
    
    test_dataset = EnhancedBrainTumorDataset(
        image_paths, mask_paths, test_indices,
        target_size=target_size,
        augment=False,
        use_preprocessing=use_preprocessing,
        use_zscore=use_zscore,
        use_bias_correction=use_bias_correction,
        use_clahe=use_clahe
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✓ Enhanced Data Loaders Created:")
    print(f"  - Train: {len(train_loader)} batches")
    print(f"  - Val: {len(val_loader)} batches")
    print(f"  - Test: {len(test_loader)} batches")
    print(f"  - Augmentation mode: {augment_mode}")
    print(f"  - Advanced preprocessing: {use_preprocessing}")
    
    return train_loader, val_loader, test_loader


def analyze_dataset_distribution(mask_paths: List[str], indices: List[int]) -> Dict[str, int]:
    """
    Analyze the distribution of tumor vs non-tumor slices.
    """
    tumor_count = 0
    non_tumor_count = 0
    
    for idx in indices:
        mask = cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            if mask.max() > 0:
                tumor_count += 1
            else:
                non_tumor_count += 1
    
    return {
        'tumor': tumor_count,
        'non_tumor': non_tumor_count,
        'total': len(indices)
    }
