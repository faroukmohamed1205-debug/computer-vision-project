"""
Brain Tumor Segmentation Dataset Utilities - PyTorch Version
Handles patient-wise splitting to prevent data leakage
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


def extract_patient_id(filename: str) -> str:
    """
    Extract patient ID from filename.
    Example: 'TCGA_CS_4941_19960909_12.tif' -> 'TCGA_CS_4941_19960909'
    """
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) >= 4:
        return '_'.join(parts[:4])  # TCGA_CS_4941_19960909
    return basename.split('.')[0]


def load_data_paths(data_dir: str) -> Tuple[List[str], List[str], Dict[str, List[int]]]:
    """
    Load all image and mask paths from the dataset directory.
    
    Args:
        data_dir: Root directory of LGG-MRI dataset
        
    Returns:
        image_paths: List of image file paths
        mask_paths: List of corresponding mask file paths
        patient_groups: Dict mapping patient_id to list of indices
    """
    try:
        # Try different possible directory structures
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
            raise FileNotFoundError(f"No mask files found in {data_dir}. Check directory structure.")
        
        # Create corresponding image paths
        image_paths = [p.replace('_mask.tif', '.tif') for p in mask_paths]
        
        # Verify all image files exist
        missing = [p for p in image_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} image files")
        
        # Group by patient ID
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


def split_patients(patient_groups: Dict[str, List[int]], 
                   train_size: float = 0.7,
                   val_size: float = 0.15,
                   test_size: float = 0.15,
                   random_state: int = 42) -> Tuple[List[int], List[int], List[int], List[str], List[str], List[str]]:
    """
    Split patients into train/val/test sets (NO DATA LEAKAGE).
    
    Args:
        patient_groups: Dict mapping patient_id to list of slice indices
        train_size: Proportion for training
        val_size: Proportion for validation
        test_size: Proportion for testing
        random_state: Random seed for reproducibility
        
    Returns:
        train_indices, val_indices, test_indices, train_patients, val_patients, test_patients
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"
    
    patient_ids = list(patient_groups.keys())
    
    # First split: train vs (val + test)
    train_patients, temp_patients = train_test_split(
        patient_ids, 
        train_size=train_size,
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    val_patients, test_patients = train_test_split(
        temp_patients,
        train_size=val_ratio,
        random_state=random_state
    )
    
    # Collect all indices for each split
    train_indices = [idx for pid in train_patients for idx in patient_groups[pid]]
    val_indices = [idx for pid in val_patients for idx in patient_groups[pid]]
    test_indices = [idx for pid in test_patients for idx in patient_groups[pid]]
    
    print(f"\n📊 Patient-wise Split:")
    print(f"  Train: {len(train_patients)} patients, {len(train_indices)} slices")
    print(f"  Val:   {len(val_patients)} patients, {len(val_indices)} slices")
    print(f"  Test:  {len(test_patients)} patients, {len(test_indices)} slices")
    
    return train_indices, val_indices, test_indices, train_patients, val_patients, test_patients


class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for Brain Tumor Segmentation.
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 mask_paths: List[str],
                 indices: List[int],
                 target_size: Tuple[int, int] = (256, 256),
                 augment: bool = False):
        """
        Args:
            image_paths: Full list of image paths
            mask_paths: Full list of mask paths
            indices: Indices to select from the full lists
            target_size: Target size for resizing
            augment: Whether to apply data augmentation
        """
        self.image_paths = [image_paths[i] for i in indices]
        self.mask_paths = [mask_paths[i] for i in indices]
        self.target_size = target_size
        self.augment = augment
        
        # Transforms
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(target_size)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        
        # Resize
        image = self.resize(image)
        mask = self.resize(mask)
        
        # Apply augmentation if training
        if self.augment:
            image, mask = self._augment(image, mask)
        
        # Convert to tensor
        image = self.to_tensor(image)  # [3, H, W] in [0, 1]
        mask = self.to_tensor(mask)    # [1, H, W] in [0, 1]
        
        # Binarize mask
        mask = (mask > 0.5).float()
        
        return image, mask
    
    def _augment(self, image, mask):
        """
        Apply random augmentations to image and mask.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Random rotation (0, 90, 180, 270)
        if random.random() > 0.5:
            angle = random.choice([0, 90, 180, 270])
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        
        # Color jitter (only for image)
        if random.random() > 0.5:
            jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)
            image = jitter(image)
        
        return image, mask


def create_data_loaders(image_paths: List[str],
                        mask_paths: List[str],
                        train_indices: List[int],
                        val_indices: List[int],
                        test_indices: List[int],
                        batch_size: int = 16,
                        num_workers: int = 4,
                        target_size: Tuple[int, int] = (256, 256)) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        image_paths: Full list of image paths
        mask_paths: Full list of mask paths
        train_indices: Training indices
        val_indices: Validation indices
        test_indices: Test indices
        batch_size: Batch size
        num_workers: Number of workers for data loading
        target_size: Target image size
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = BrainTumorDataset(
        image_paths, mask_paths, train_indices,
        target_size=target_size, augment=True
    )
    
    val_dataset = BrainTumorDataset(
        image_paths, mask_paths, val_indices,
        target_size=target_size, augment=False
    )
    
    test_dataset = BrainTumorDataset(
        image_paths, mask_paths, test_indices,
        target_size=target_size, augment=False
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
    
    print(f"\n✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader)} batches")
    print(f"✓ Test loader: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def analyze_dataset_distribution(mask_paths: List[str], indices: List[int]) -> Dict[str, int]:
    """
    Analyze the distribution of tumor vs non-tumor slices.
    
    Args:
        mask_paths: List of mask file paths
        indices: Indices to analyze
        
    Returns:
        Dictionary with counts
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
