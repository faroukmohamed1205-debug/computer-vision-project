"""
Configuration File for Brain Tumor Segmentation Project
Modify these settings to customize the experiment
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Base directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data' / 'kaggle_3m'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'

# Create directories if they don't exist
for directory in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA SETTINGS
# ============================================================================

# Image preprocessing
TARGET_SIZE = (256, 256)  # Resize all images to this size
NORMALIZE = True          # Normalize images to [0, 1]
BINARIZE_MASKS = True     # Convert masks to binary (0 or 1)

# Data split ratios (must sum to 1.0)
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# TRAINING SETTINGS
# ============================================================================

# Training hyperparameters
BATCH_SIZE = 16           # Reduce if GPU memory issues
EPOCHS = 5               # Maximum epochs to train
LEARNING_RATE = 1e-4      # Initial learning rate for Adam optimizer

# Data augmentation
USE_AUGMENTATION = True   # Apply augmentation to training data
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'vertical_flip': True,
    'brightness_range': 0.2,
    'rotation_range': 0,      # Degrees (0 = disabled)
    'zoom_range': 0.0,        # (0.0 = disabled)
}

# Early stopping
EARLY_STOPPING_PATIENCE = 10    # Stop if no improvement for N epochs
REDUCE_LR_PATIENCE = 5          # Reduce LR if no improvement for N epochs
MIN_LEARNING_RATE = 1e-7        # Minimum learning rate

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

# ResUNet configuration
INPUT_SHAPE = (*TARGET_SIZE, 3)  # (H, W, C)
FILTERS_BASE = 16                 # Base number of filters (doubles each level)
USE_BATCH_NORM = True             # Use batch normalization
USE_DROPOUT = False               # Use dropout (not implemented by default)
DROPOUT_RATE = 0.2                # Dropout rate if enabled

# ============================================================================
# LOSS FUNCTION - BASELINE MODEL
# ============================================================================

# Tversky Loss parameters for baseline
BASELINE_ALPHA = 0.5  # Weight for False Negatives (missing tumors)
BASELINE_BETA = 0.5   # Weight for False Positives (false alarms)

# When alpha > beta: Penalize missing tumors more (higher recall)
# When beta > alpha: Penalize false alarms more (higher precision)
# When alpha = beta = 0.5: Equivalent to Dice Loss

# ============================================================================
# OPTUNA OPTIMIZATION SETTINGS
# ============================================================================

# Hyperparameter search space
OPTUNA_CONFIG = {
    'n_trials': 5,              # Number of optimization trials
    'timeout': None,              # Timeout in seconds (None = no limit)
    'n_jobs': 1,                  # Number of parallel jobs
    
    # Search ranges
    'alpha_range': (0.1, 0.9),    # Tversky alpha range
    'beta_range': (0.1, 0.9),     # Tversky beta range
    'lr_range': (1e-5, 1e-3),     # Learning rate range (log scale)
    
    # Optimization settings
    'sampler': 'TPE',             # TPE or RandomSampler
    'pruner': None,               # MedianPruner or None
    'direction': 'maximize',      # Maximize validation Dice
    'metric': 'val_dice_coefficient',  # Metric to optimize
    
    # Training during optimization
    'optimization_epochs': 5,    # Epochs per trial (reduced for speed)
    'optimization_patience': 5,   # Early stopping patience during optimization
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Figure settings
FIG_DPI = 300                    # Resolution for saved figures
FIG_FORMAT = 'png'               # Figure format (png, pdf, svg)
STYLE = 'seaborn-v0_8-paper'    # Matplotlib style

# Contest visualization
CONTEST_NUM_SAMPLES = 5          # Number of samples to show
CONTEST_ONLY_TUMOR = True        # Show only slices with tumors

# Colors for plots
COLOR_BASELINE = '#3498db'       # Blue
COLOR_OPTIMIZED = '#e74c3c'      # Red
COLOR_VALIDATION = '#2ecc71'     # Green

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

# Prediction threshold
PREDICTION_THRESHOLD = 0.5       # Threshold for binary segmentation

# Metrics to track
METRICS = [
    'dice_coefficient',
    'iou_coefficient',
    'binary_accuracy'
]

# Statistical tests
RUN_STATISTICAL_TESTS = True     # Run paired t-test on test results
SIGNIFICANCE_LEVEL = 0.05        # Alpha for statistical significance

# ============================================================================
# LOGGING AND SAVING
# ============================================================================

# Model checkpointing
SAVE_BEST_ONLY = True            # Save only best model or all checkpoints
MONITOR_METRIC = 'val_dice_coefficient'  # Metric to monitor for saving
SAVE_WEIGHTS_ONLY = False        # Save entire model or weights only

# Logging
VERBOSE = 1                      # Verbosity level (0=silent, 1=progress, 2=one line per epoch)
LOG_TO_FILE = True               # Save training logs to file

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# TensorFlow settings
TF_CONFIG = {
    'allow_memory_growth': True,  # Enable GPU memory growth
    'mixed_precision': False,     # Use mixed precision training (faster on modern GPUs)
    'xla': False,                 # Enable XLA compilation (experimental)
}

# Reproducibility
SET_TF_SEED = True               # Set TensorFlow random seed
SET_NUMPY_SEED = True            # Set NumPy random seed
SET_PYTHON_SEED = True           # Set Python random seed

# Debug mode
DEBUG_MODE = False               # Enable debug logging
LIMIT_SAMPLES = None             # Limit samples for testing (None = use all)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_name(model_type: str, alpha: float = None, beta: float = None) -> str:
    """
    Generate model name based on type and parameters.
    
    Args:
        model_type: 'baseline' or 'optimized'
        alpha: Tversky alpha parameter
        beta: Tversky beta parameter
    
    Returns:
        Model filename
    """
    if model_type == 'baseline':
        return f'baseline_model_a{BASELINE_ALPHA}_b{BASELINE_BETA}_best.keras'
    else:
        if alpha is not None and beta is not None:
            return f'optimized_model_a{alpha:.3f}_b{beta:.3f}_best.keras'
        return 'optimized_model_best.keras'

def print_config():
    """Print current configuration."""
    print("=" * 70)
    print("CURRENT CONFIGURATION")
    print("=" * 70)
    print(f"\nData Directory: {DATA_DIR}")
    print(f"Image Size: {TARGET_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"\nBaseline Tversky Loss: α={BASELINE_ALPHA}, β={BASELINE_BETA}")
    print(f"\nOptuna Trials: {OPTUNA_CONFIG['n_trials']}")
    print(f"Optuna Alpha Range: {OPTUNA_CONFIG['alpha_range']}")
    print(f"Optuna Beta Range: {OPTUNA_CONFIG['beta_range']}")
    print(f"\nContest Samples: {CONTEST_NUM_SAMPLES}")
    print(f"Figure DPI: {FIG_DPI}")
    print("=" * 70)

# ============================================================================
# VALIDATE CONFIGURATION
# ============================================================================

def validate_config():
    """Validate configuration settings."""
    errors = []
    
    # Check split ratios
    if abs(TRAIN_SIZE + VAL_SIZE + TEST_SIZE - 1.0) > 1e-6:
        errors.append(f"Split ratios must sum to 1.0 (current: {TRAIN_SIZE + VAL_SIZE + TEST_SIZE})")
    
    # Check paths
    if not DATA_DIR.exists():
        errors.append(f"Data directory not found: {DATA_DIR}")
    
    # Check Tversky parameters
    if not (0 < BASELINE_ALPHA < 1):
        errors.append(f"Baseline alpha must be in (0, 1), got {BASELINE_ALPHA}")
    if not (0 < BASELINE_BETA < 1):
        errors.append(f"Baseline beta must be in (0, 1), got {BASELINE_BETA}")
    
    # Check batch size
    if BATCH_SIZE < 1:
        errors.append(f"Batch size must be positive, got {BATCH_SIZE}")
    
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✓ Configuration validated successfully")
    return True

if __name__ == "__main__":
    print_config()
    validate_config()
