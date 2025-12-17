"""
Data augmentation utilities - PyTorch Compatible
Supports 'light' and 'heavy' modes for optimized training.
"""

import numpy as np
from scipy import ndimage
import logging
from typing import Tuple
import torch

logger = logging.getLogger(__name__)


class ImageAugmentor:
    """
    Data augmentation for images and masks.
    
    Modes:
    - 'light': Flip, Rotation (Standard)
    - 'heavy': Elastic Transform, Gaussian Noise, Brightness/Contrast (Advanced)
    """
    
    def __init__(self, seed: int = 42, mode: str = 'light'):
        """
        Args:
            seed: Random seed for reproducibility.
            mode: 'light' or 'heavy'.
        """
        self.rng = np.random.RandomState(seed)
        self.mode = mode
        logger.info(f"ImageAugmentor initialized in '{mode}' mode")

    def add_gaussian_noise(self, image: np.ndarray, var: float = 0.01) -> np.ndarray:
        """
        Add random Gaussian noise to the image (Simulates MRI noise).
        
        Args:
            image: Input image [H, W, C]
            var: Variance of the noise
            
        Returns:
            Noisy image
        """
        if self.mode == 'light': 
            return image
        
        noise = self.rng.normal(0, var ** 0.5, image.shape)
        # Note: We don't clip aggressively here to support Z-score output
        return image + noise

    def elastic_transform(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        alpha: float = 120, 
        sigma: float = 6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Elastic Transform to simulate tissue deformation.
        Crucial for medical imaging generalization.
        
        Args:
            image: Input image [H, W, C]
            mask: Input mask [H, W, 1] or [H, W]
            alpha: Intensity of deformation
            sigma: Smoothness of deformation
            
        Returns:
            Transformed image and mask
        """
        if self.mode == 'light':
            return image, mask
        
        shape = image.shape[:2]
        
        # Generate random displacement fields
        dx = ndimage.gaussian_filter((self.rng.rand(*shape) * 2 - 1), sigma) * alpha
        dy = ndimage.gaussian_filter((self.rng.rand(*shape) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

        # Map coordinates for Image (Bilinear interpolation)
        dist_img = np.zeros_like(image)
        if image.ndim == 3:
            for i in range(image.shape[2]):
                dist_img[:,:,i] = ndimage.map_coordinates(
                    image[:,:,i], indices, order=1, mode='reflect'
                ).reshape(shape)
        else:
            dist_img = ndimage.map_coordinates(
                image, indices, order=1, mode='reflect'
            ).reshape(shape)
            
        # Map coordinates for Mask (Nearest Neighbor to keep binary values)
        if mask.ndim == 3:
            dist_mask = ndimage.map_coordinates(
                mask[:,:,0], indices, order=0, mode='reflect'
            ).reshape(shape)
            dist_mask = np.expand_dims(dist_mask, axis=-1)
        else:
            dist_mask = ndimage.map_coordinates(
                mask, indices, order=0, mode='reflect'
            ).reshape(shape)
            dist_mask = np.expand_dims(dist_mask, axis=-1)

        return dist_img, dist_mask

    def augment(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation pipeline based on selected mode.
        
        Args:
            image: Input image [H, W, C]
            mask: Input mask [H, W, 1]
            
        Returns:
            Augmented image and mask
        """
        # Ensure proper dimensions
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)
        
        # --- 1. Basic Augmentations (Applied in both modes) ---
        
        # Horizontal Flip
        if self.rng.rand() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        
        # Vertical Flip (Less frequent)
        if self.rng.rand() < 0.2:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        # Random Rotation (+- 15 degrees)
        if self.rng.rand() < 0.5:
            angle = self.rng.randint(-15, 15)
            # Order 1 for image, Order 0 for mask
            image = ndimage.rotate(image, angle, reshape=False, mode='reflect', order=1)
            mask = ndimage.rotate(mask, angle, reshape=False, mode='reflect', order=0)

        # --- 2. Heavy Augmentations (Only if mode='heavy') ---
        if self.mode == 'heavy':
            # Elastic Transform (30% probability)
            if self.rng.rand() < 0.3:
                image, mask = self.elastic_transform(image, mask)
            
            # Gaussian Noise (20% probability)
            if self.rng.rand() < 0.2:
                image = self.add_gaussian_noise(image)
                
            # Brightness & Contrast adjustment
            if self.rng.rand() < 0.3:
                alpha = 1.0 + self.rng.uniform(-0.2, 0.2)  # Contrast factor
                beta = self.rng.uniform(-0.1, 0.1)         # Brightness bias
                image = image * alpha + beta

        # Ensure mask stays binary
        mask = (mask > 0.5).astype(np.float32)
        
        return image, mask


class TorchAugmentor:
    """
    PyTorch-compatible augmentor that works with tensors.
    Wraps ImageAugmentor for use in PyTorch Datasets.
    """
    
    def __init__(self, seed: int = 42, mode: str = 'light'):
        """
        Args:
            seed: Random seed
            mode: 'light' or 'heavy'
        """
        self.augmentor = ImageAugmentor(seed=seed, mode=mode)
        self.mode = mode
    
    def __call__(
        self, 
        image: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentation to PyTorch tensors.
        
        Args:
            image: Image tensor [C, H, W]
            mask: Mask tensor [1, H, W]
            
        Returns:
            Augmented image and mask tensors
        """
        # Convert to numpy [H, W, C]
        image_np = image.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.permute(1, 2, 0).cpu().numpy()
        
        # Apply augmentation
        image_aug, mask_aug = self.augmentor.augment(image_np, mask_np)
        
        # Convert back to tensor [C, H, W]
        image_tensor = torch.from_numpy(image_aug).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_aug).permute(2, 0, 1).float()
        
        return image_tensor, mask_tensor


def visualize_augmentations(
    image: np.ndarray,
    mask: np.ndarray,
    mode: str = 'heavy',
    num_samples: int = 6,
    save_path: str = 'augmentation_examples.png'
):
    """
    Visualize different augmentation results.
    
    Args:
        image: Original image [H, W, C]
        mask: Original mask [H, W, 1]
        mode: Augmentation mode
        num_samples: Number of augmented samples to generate
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    augmentor = ImageAugmentor(mode=mode)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(num_samples):
        # Generate augmented version
        img_aug, mask_aug = augmentor.augment(image.copy(), mask.copy())
        
        # Handle Z-score normalized images for visualization
        img_display = image.copy()
        img_aug_display = img_aug.copy()
        
        # Normalize to [0, 1] for display if needed
        if img_display.min() < 0 or img_display.max() > 1.5:
            img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        if img_aug_display.min() < 0 or img_aug_display.max() > 1.5:
            img_aug_display = (img_aug_display - img_aug_display.min()) / (img_aug_display.max() - img_aug_display.min())
        
        # Original Image
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title(f'Original {i+1}', fontsize=10, weight='bold')
        axes[i, 0].axis('off')
        
        # Augmented Image
        axes[i, 1].imshow(img_aug_display)
        axes[i, 1].set_title(f'Augmented {i+1}', fontsize=10, weight='bold')
        axes[i, 1].axis('off')
        
        # Augmented Mask Overlay
        axes[i, 2].imshow(img_aug_display)
        axes[i, 2].imshow(mask_aug[:,:,0], alpha=0.5, cmap='Reds')
        axes[i, 2].set_title(f'Aug + Mask {i+1}', fontsize=10, weight='bold')
        axes[i, 2].axis('off')
    
    plt.suptitle(f'Augmentation Examples (Mode: {mode})', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved augmentation examples to {save_path}")
    plt.close()


def compare_augmentation_modes(
    image: np.ndarray,
    mask: np.ndarray,
    save_path: str = 'augmentation_mode_comparison.png'
):
    """
    Compare light vs heavy augmentation modes.
    
    Args:
        image: Original image [H, W, C]
        mask: Original mask [H, W, 1]
        save_path: Path to save comparison figure
    """
    import matplotlib.pyplot as plt
    
    augmentor_light = ImageAugmentor(mode='light')
    augmentor_heavy = ImageAugmentor(mode='heavy')
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Original
    img_display = image.copy()
    if img_display.min() < 0 or img_display.max() > 1.5:
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
    
    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title('Original Image', fontsize=11, weight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_display)
    axes[0, 1].imshow(mask[:,:,0], alpha=0.5, cmap='Reds')
    axes[0, 1].set_title('Original Mask', fontsize=11, weight='bold')
    axes[0, 1].axis('off')
    
    # Empty cells
    axes[0, 2].axis('off')
    axes[0, 3].axis('off')
    
    # Light mode samples
    for i in range(2):
        img_aug, mask_aug = augmentor_light.augment(image.copy(), mask.copy())
        
        img_aug_display = img_aug.copy()
        if img_aug_display.min() < 0 or img_aug_display.max() > 1.5:
            img_aug_display = (img_aug_display - img_aug_display.min()) / (img_aug_display.max() - img_aug_display.min())
        
        axes[1, i*2].imshow(img_aug_display)
        axes[1, i*2].set_title(f'Light Aug {i+1}', fontsize=11, weight='bold', color='#3498db')
        axes[1, i*2].axis('off')
        
        axes[1, i*2+1].imshow(img_aug_display)
        axes[1, i*2+1].imshow(mask_aug[:,:,0], alpha=0.5, cmap='Reds')
        axes[1, i*2+1].set_title(f'Light Mask {i+1}', fontsize=11, weight='bold', color='#3498db')
        axes[1, i*2+1].axis('off')
    
    # Heavy mode samples
    for i in range(2):
        img_aug, mask_aug = augmentor_heavy.augment(image.copy(), mask.copy())
        
        img_aug_display = img_aug.copy()
        if img_aug_display.min() < 0 or img_aug_display.max() > 1.5:
            img_aug_display = (img_aug_display - img_aug_display.min()) / (img_aug_display.max() - img_aug_display.min())
        
        axes[2, i*2].imshow(img_aug_display)
        axes[2, i*2].set_title(f'Heavy Aug {i+1}', fontsize=11, weight='bold', color='#e74c3c')
        axes[2, i*2].axis('off')
        
        axes[2, i*2+1].imshow(img_aug_display)
        axes[2, i*2+1].imshow(mask_aug[:,:,0], alpha=0.5, cmap='Reds')
        axes[2, i*2+1].set_title(f'Heavy Mask {i+1}', fontsize=11, weight='bold', color='#e74c3c')
        axes[2, i*2+1].axis('off')
    
    plt.suptitle('Augmentation Mode Comparison: Light vs Heavy', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved augmentation mode comparison to {save_path}")
    plt.close()
