"""
Data preprocessing utilities for brain tumor segmentation - PyTorch Compatible

Advanced Pipeline:
1. Load image & mask.
2. Resize to target size (256x256).
3. Apply Bias Field Correction (Using LAB space to preserve details).
4. Apply CLAHE for contrast enhancement.
5. Apply Normalization (Z-Score or Min-Max).
6. Convert to PyTorch tensors.
"""

import numpy as np
import cv2
import logging
from typing import Tuple, Optional
from PIL import Image
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Advanced Image Preprocessor including:
    - Resize
    - Bias Field Correction (Uses LAB space)
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Z-Score Normalization
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        use_zscore: bool = True,
        use_bias_correction: bool = True,
        use_clahe: bool = True
    ):
        """
        Args:
            target_size: Target image size (H, W)
            use_zscore: Use Z-score normalization instead of min-max
            use_bias_correction: Apply bias field correction
            use_clahe: Apply CLAHE for contrast enhancement
        """
        self.target_size = target_size
        self.use_zscore = use_zscore
        self.use_bias_correction = use_bias_correction
        self.use_clahe = use_clahe
        
        # Initialize CLAHE with clip limit 2.0 (standard for MRI)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        logger.info(f"ImagePreprocessor initialized:")
        logger.info(f"  - Target size: {target_size}")
        logger.info(f"  - Z-score normalization: {use_zscore}")
        logger.info(f"  - Bias correction: {use_bias_correction}")
        logger.info(f"  - CLAHE: {use_clahe}")

    def apply_bias_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Simulate N4 Bias Field Correction using OpenCV in LAB color space.
        This preserves color information by operating only on the Luminance channel.
        
        Args:
            image: RGB image as numpy array [H, W, 3]
            
        Returns:
            Bias-corrected image [H, W, 3]
        """
        # Ensure uint8 for color conversion
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        # 1. Convert to LAB color space
        # L = Lightness, A/B = Color components
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # 2. Estimate bias field on the L channel only
        # Use float32 for precision during division
        l_float = l.astype(np.float32)
        
        # Estimate the low-frequency bias field using Gaussian blur
        bias_field = cv2.GaussianBlur(l_float, (0, 0), sigmaX=50, sigmaY=50)
        
        # 3. Correct the L channel: I_corr = I_orig / Bias
        epsilon = 1e-5
        l_corrected = l_float / (bias_field + epsilon)
        
        # 4. Normalize back to 0-255 range
        l_corrected = cv2.normalize(l_corrected, None, 0, 255, cv2.NORM_MINMAX)
        l_corrected = l_corrected.astype(np.uint8)
        
        # 5. Merge back with original A and B channels
        lab_corrected = cv2.merge((l_corrected, a, b))
        
        # 6. Convert back to RGB
        return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2RGB)

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE on the L channel of LAB space.
        
        Args:
            image: RGB image as numpy array [H, W, 3]
            
        Returns:
            CLAHE-enhanced image [H, W, 3]
        """
        # Ensure uint8
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_enhanced = self.clahe.apply(l)
        
        # Merge and convert back
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    def apply_z_score(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Z-Score Normalization (Standardization).
        (Pixel - Mean) / Std_Dev
        
        Args:
            image: Image as numpy array
            
        Returns:
            Normalized image
        """
        image = image.astype(np.float32)
        mean = np.mean(image)
        std = np.std(image)
        
        if std > 0:
            return (image - mean) / std
        return image

    def preprocess_from_path(self, image_path: str) -> np.ndarray:
        """
        Run the full preprocessing pipeline on a single image path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image as numpy array [H, W, 3]
        """
        try:
            # 1. Load Image
            image = cv2.imread(str(image_path))
            if image is None:
                raise IOError(f"Could not read image: {image_path}")
            
            # Ensure RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 2. Resize
            if image.shape[:2] != self.target_size:
                image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

            # 3. Bias Field Correction (Uses LAB to preserve colors)
            if self.use_bias_correction:
                image = self.apply_bias_correction(image)

            # 4. CLAHE (Applied on L channel of LAB space)
            if self.use_clahe:
                image = self.apply_clahe(image)

            # 5. Normalization
            if self.use_zscore:
                image = self.apply_z_score(image)
            else:
                # Fallback to Min-Max (0-1)
                image = image.astype(np.float32) / 255.0

            return image

        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            # Return zero array on failure to prevent crash
            if self.use_zscore:
                return np.zeros((*self.target_size, 3), dtype=np.float32)
            else:
                return np.zeros((*self.target_size, 3), dtype=np.float32)

    def preprocess_from_pil(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess a PIL Image (used in PyTorch Dataset).
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed numpy array [H, W, 3]
        """
        try:
            # Convert PIL to numpy
            image = np.array(image)
            
            # Ensure RGB
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # Resize if needed
            if image.shape[:2] != self.target_size:
                image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

            # Bias Field Correction
            if self.use_bias_correction:
                image = self.apply_bias_correction(image)

            # CLAHE
            if self.use_clahe:
                image = self.apply_clahe(image)

            # Normalization
            if self.use_zscore:
                image = self.apply_z_score(image)
            else:
                image = image.astype(np.float32) / 255.0

            return image

        except Exception as e:
            logger.error(f"Error preprocessing PIL image: {e}")
            if self.use_zscore:
                return np.zeros((*self.target_size, 3), dtype=np.float32)
            else:
                return np.zeros((*self.target_size, 3), dtype=np.float32)


class MaskPreprocessor:
    """
    Preprocessing utilities for segmentation masks.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        """
        Args:
            target_size: Target mask size (H, W)
        """
        self.target_size = target_size
        logger.info(f"MaskPreprocessor initialized with target size: {target_size}")

    def preprocess_from_path(self, mask_path: str) -> np.ndarray:
        """
        Load and preprocess mask from path.
        
        Args:
            mask_path: Path to mask file
            
        Returns:
            Binary mask as numpy array [H, W, 1]
        """
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise IOError(f"Could not read mask: {mask_path}")

            if mask.shape != self.target_size:
                mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

            # Binarize: 0 or 1
            mask = (mask > 127).astype(np.float32)
            
            # Expand dims to [H, W, 1]
            return np.expand_dims(mask, axis=-1)
            
        except Exception as e:
            logger.error(f"Error preprocessing mask {mask_path}: {e}")
            return np.zeros((*self.target_size, 1), dtype=np.float32)

    def preprocess_from_pil(self, mask: Image.Image) -> np.ndarray:
        """
        Preprocess a PIL mask image.
        
        Args:
            mask: PIL Image (grayscale)
            
        Returns:
            Binary mask as numpy array [H, W, 1]
        """
        try:
            # Convert to grayscale numpy
            mask = np.array(mask.convert('L'))
            
            # Resize if needed
            if mask.shape != self.target_size:
                mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            
            # Binarize
            mask = (mask > 127).astype(np.float32)
            
            # Add channel dimension
            return np.expand_dims(mask, axis=-1)
            
        except Exception as e:
            logger.error(f"Error preprocessing PIL mask: {e}")
            return np.zeros((*self.target_size, 1), dtype=np.float32)


def visualize_preprocessing_steps(image_path: str, save_path: str = 'preprocessing_comparison.png'):
    """
    Visualize the effect of each preprocessing step.
    
    Args:
        image_path: Path to sample image
        save_path: Path to save comparison figure
    """
    import matplotlib.pyplot as plt
    
    # Load original
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (256, 256))
    
    # Create preprocessors with different configurations
    prep_basic = ImagePreprocessor(use_zscore=False, use_bias_correction=False, use_clahe=False)
    prep_bias = ImagePreprocessor(use_zscore=False, use_bias_correction=True, use_clahe=False)
    prep_clahe = ImagePreprocessor(use_zscore=False, use_bias_correction=True, use_clahe=True)
    prep_full = ImagePreprocessor(use_zscore=True, use_bias_correction=True, use_clahe=True)
    
    # Process
    img_basic = prep_basic.preprocess_from_path(image_path)
    img_bias = prep_bias.preprocess_from_path(image_path)
    img_clahe = prep_clahe.preprocess_from_path(image_path)
    img_full = prep_full.preprocess_from_path(image_path)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original', fontsize=12, weight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_basic)
    axes[0, 1].set_title('Min-Max Normalized', fontsize=12, weight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_bias)
    axes[0, 2].set_title('+ Bias Correction', fontsize=12, weight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(img_clahe)
    axes[1, 0].set_title('+ CLAHE', fontsize=12, weight='bold')
    axes[1, 0].axis('off')
    
    # Z-score needs special handling for visualization
    img_full_vis = (img_full - img_full.min()) / (img_full.max() - img_full.min())
    axes[1, 1].imshow(img_full_vis)
    axes[1, 1].set_title('+ Z-Score (Full Pipeline)', fontsize=12, weight='bold')
    axes[1, 1].axis('off')
    
    # Hide last subplot
    axes[1, 2].axis('off')
    
    plt.suptitle('Preprocessing Pipeline Visualization', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved preprocessing comparison to {save_path}")
    plt.close()
