"""
Publication-Ready Visualization Functions
High-quality plots for research papers and presentations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple
import os
import cv2

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14


def ensure_dir(directory: str):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def plot_patient_split(train_ids: List[str], 
                       val_ids: List[str], 
                       test_ids: List[str],
                       save_path: str = 'results/figures/patient_split.png'):
    """
    Plot pie chart showing patient distribution across splits.
    
    Args:
        train_ids: Training patient IDs
        val_ids: Validation patient IDs
        test_ids: Test patient IDs
        save_path: Path to save figure
    """
    ensure_dir(os.path.dirname(save_path))
    
    counts = [len(train_ids), len(val_ids), len(test_ids)]
    labels = ['Train', 'Validation', 'Test']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    wedges, texts, autotexts = ax.pie(
        counts, 
        labels=labels, 
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    
    # Make percentage text white
    for autotext in autotexts:
        autotext.set_color('white')
    
    ax.set_title('Patient Distribution Across Splits\n(No Data Leakage)', 
                 fontsize=14, weight='bold', pad=20)
    
    # Add count annotations
    legend_labels = [f'{labels[i]}: {counts[i]} patients' for i in range(3)]
    ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✓ Saved patient split plot to {save_path}")
    plt.close()


def plot_slice_distribution(train_dist: Dict, val_dist: Dict, test_dist: Dict,
                            save_path: str = 'results/figures/slice_distribution.png'):
    """
    Plot bar chart showing tumor vs non-tumor slice distribution.
    
    Args:
        train_dist: Training distribution dict
        val_dist: Validation distribution dict
        test_dist: Test distribution dict
        save_path: Path to save figure
    """
    ensure_dir(os.path.dirname(save_path))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = [train_dist, val_dist, test_dist]
    titles = ['Training Set', 'Validation Set', 'Test Set']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for ax, dist, title, color in zip(axes, datasets, titles, colors):
        categories = ['Tumor', 'Non-Tumor']
        values = [dist['tumor'], dist['non_tumor']]
        
        bars = ax.bar(categories, values, color=[color, '#95a5a6'], alpha=0.8, edgecolor='black')
        ax.set_ylabel('Number of Slices', fontsize=11, weight='bold')
        ax.set_title(title, fontsize=12, weight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, weight='bold')
        
        # Add percentage
        total = sum(values)
        tumor_pct = (values[0] / total) * 100
        ax.text(0.5, 0.95, f'Tumor: {tumor_pct:.1f}%', 
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Distribution of Tumor vs Non-Tumor Slices', 
                fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✓ Saved slice distribution plot to {save_path}")
    plt.close()


def plot_training_history(baseline_history: Dict, 
                          optimized_history: Dict,
                          save_path: str = 'results/figures/training_curves.png'):
    """
    Plot training and validation curves for both models.
    
    Args:
        baseline_history: Training history from baseline model (PyTorch format)
        optimized_history: Training history from optimized model (PyTorch format)
        save_path: Path to save figure
    """
    ensure_dir(os.path.dirname(save_path))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(baseline_history['train_loss'], label='Baseline Train', 
           color='#3498db', linewidth=2, alpha=0.8)
    ax.plot(baseline_history['val_loss'], label='Baseline Val', 
           color='#3498db', linewidth=2, linestyle='--', alpha=0.8)
    ax.plot(optimized_history['train_loss'], label='Optimized Train', 
           color='#e74c3c', linewidth=2, alpha=0.8)
    ax.plot(optimized_history['val_loss'], label='Optimized Val', 
           color='#e74c3c', linewidth=2, linestyle='--', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11, weight='bold')
    ax.set_ylabel('Loss', fontsize=11, weight='bold')
    ax.set_title('Training & Validation Loss', fontsize=12, weight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Dice coefficient curves
    ax = axes[0, 1]
    ax.plot(baseline_history['train_dice'], label='Baseline Train', 
           color='#3498db', linewidth=2, alpha=0.8)
    ax.plot(baseline_history['val_dice'], label='Baseline Val', 
           color='#3498db', linewidth=2, linestyle='--', alpha=0.8)
    ax.plot(optimized_history['train_dice'], label='Optimized Train', 
           color='#e74c3c', linewidth=2, alpha=0.8)
    ax.plot(optimized_history['val_dice'], label='Optimized Val', 
           color='#e74c3c', linewidth=2, linestyle='--', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11, weight='bold')
    ax.set_ylabel('Dice Coefficient', fontsize=11, weight='bold')
    ax.set_title('Dice Coefficient', fontsize=12, weight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # IoU curves
    ax = axes[1, 0]
    ax.plot(baseline_history['train_iou'], label='Baseline Train', 
           color='#3498db', linewidth=2, alpha=0.8)
    ax.plot(baseline_history['val_iou'], label='Baseline Val', 
           color='#3498db', linewidth=2, linestyle='--', alpha=0.8)
    ax.plot(optimized_history['train_iou'], label='Optimized Train', 
           color='#e74c3c', linewidth=2, alpha=0.8)
    ax.plot(optimized_history['val_iou'], label='Optimized Val', 
           color='#e74c3c', linewidth=2, linestyle='--', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11, weight='bold')
    ax.set_ylabel('IoU Coefficient', fontsize=11, weight='bold')
    ax.set_title('Intersection over Union', fontsize=12, weight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Summary statistics plot
    ax = axes[1, 1]
    
    # Calculate final metrics
    baseline_final_dice = baseline_history['val_dice'][-1]
    optimized_final_dice = optimized_history['val_dice'][-1]
    baseline_best_dice = max(baseline_history['val_dice'])
    optimized_best_dice = max(optimized_history['val_dice'])
    
    metrics = ['Final Val\nDice', 'Best Val\nDice']
    baseline_vals = [baseline_final_dice, baseline_best_dice]
    optimized_vals = [optimized_final_dice, optimized_best_dice]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', 
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, optimized_vals, width, label='Optimized',
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Dice Coefficient', fontsize=11, weight='bold')
    ax.set_title('Summary Metrics', fontsize=12, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Training Progress: Baseline vs Optimized ResUNet', 
                fontsize=14, weight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✓ Saved training curves to {save_path}")
    plt.close()


def plot_model_comparison_boxplot(baseline_scores: List[float], 
                                  optimized_scores: List[float],
                                  save_path: str = 'results/figures/model_comparison.png'):
    """
    Plot box plot comparing test set Dice scores.
    
    Args:
        baseline_scores: List of Dice scores from baseline model
        optimized_scores: List of Dice scores from optimized model
        save_path: Path to save figure
    """
    ensure_dir(os.path.dirname(save_path))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    data = [baseline_scores, optimized_scores]
    labels = ['Baseline\n(α=0.5, β=0.5)', 'Optimized\n(Optuna Tuned)']
    colors = ['#3498db', '#e74c3c']
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    widths=0.6, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='gold', 
                                  markeredgecolor='black', markersize=8))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add statistics
    baseline_mean = np.mean(baseline_scores)
    baseline_median = np.median(baseline_scores)
    optimized_mean = np.mean(optimized_scores)
    optimized_median = np.median(optimized_scores)
    
    # Add statistical annotations
    y_pos = max(max(baseline_scores), max(optimized_scores)) * 1.02
    
    stats_text = (
        f"Baseline:    Mean={baseline_mean:.4f}, Median={baseline_median:.4f}\n"
        f"Optimized: Mean={optimized_mean:.4f}, Median={optimized_median:.4f}\n"
        f"Improvement: {((optimized_mean - baseline_mean)/baseline_mean * 100):+.2f}%"
    )
    
    ax.text(0.5, 0.02, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom', horizontalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_ylabel('Dice Coefficient', fontsize=12, weight='bold')
    ax.set_title('Test Set Performance: Baseline vs Optimized ResUNet', 
                fontsize=14, weight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    
    # Add legend for mean marker
    ax.legend([mpatches.Patch(color='gold', label='Mean')], 
             ['Mean (Diamond)'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✓ Saved model comparison boxplot to {save_path}")
    plt.close()


def visualize_segmentation_contest(model_baseline, 
                                   model_optimized, 
                                   test_loader,
                                   device,
                                   num_samples: int = 5,
                                   save_path: str = 'results/figures/contest_results.png'):
    """
    Create side-by-side comparison of predictions from both models.
    Shows: [Original MRI | Ground Truth | Baseline Prediction | Optimized Prediction]
    
    Args:
        model_baseline: Trained baseline model
        model_optimized: Trained optimized model
        test_loader: Test DataLoader
        device: PyTorch device
        num_samples: Number of samples to visualize
        save_path: Path to save figure
    """
    import torch
    ensure_dir(os.path.dirname(save_path))
    
    model_baseline.eval()
    model_optimized.eval()
    
    # Collect samples with tumors
    images_with_tumor = []
    masks_with_tumor = []
    
    with torch.no_grad():
        for batch_images, batch_masks in test_loader:
            for img, mask in zip(batch_images, batch_masks):
                if mask.max() > 0:  # Has tumor
                    images_with_tumor.append(img.cpu().numpy())
                    masks_with_tumor.append(mask.cpu().numpy())
                    if len(images_with_tumor) >= num_samples:
                        break
            if len(images_with_tumor) >= num_samples:
                break
    
    if len(images_with_tumor) < num_samples:
        print(f"⚠ Warning: Only found {len(images_with_tumor)} samples with tumors")
        num_samples = len(images_with_tumor)
    
    # Get predictions
    images_array = np.array(images_with_tumor[:num_samples])
    masks_array = np.array(masks_with_tumor[:num_samples])
    
    # Convert to torch tensors and get predictions
    images_tensor = torch.from_numpy(images_array).to(device)
    
    with torch.no_grad():
        pred_baseline = model_baseline(images_tensor).cpu().numpy()
        pred_optimized = model_optimized(images_tensor).cpu().numpy()
    
    # Calculate Dice scores per sample
    dice_baseline = []
    dice_optimized = []
    
    for i in range(num_samples):
        mask = masks_array[i]
        pred_b = (pred_baseline[i] > 0.5).astype(np.float32)
        pred_o = (pred_optimized[i] > 0.5).astype(np.float32)
        
        # Calculate Dice
        intersection_b = (mask * pred_b).sum()
        dice_b = (2. * intersection_b + 1e-6) / (mask.sum() + pred_b.sum() + 1e-6)
        
        intersection_o = (mask * pred_o).sum()
        dice_o = (2. * intersection_o + 1e-6) / (mask.sum() + pred_o.sum() + 1e-6)
        
        dice_baseline.append(dice_b)
        dice_optimized.append(dice_o)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Convert from CHW to HWC for matplotlib
        img = np.transpose(images_array[i], (1, 2, 0))
        mask = masks_array[i][0]  # Remove channel dimension
        pred_b = (pred_baseline[i][0] > 0.5).astype(np.float32)
        pred_o = (pred_optimized[i][0] > 0.5).astype(np.float32)
        
        # Original Image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Sample {i+1}: Original MRI', fontsize=11, weight='bold')
        axes[i, 0].axis('off')
        
        # Ground Truth
        axes[i, 1].imshow(img)
        axes[i, 1].imshow(mask, alpha=0.5, cmap='Reds')
        axes[i, 1].set_title('Ground Truth', fontsize=11, weight='bold')
        axes[i, 1].axis('off')
        
        # Baseline Prediction
        axes[i, 2].imshow(img)
        axes[i, 2].imshow(pred_b, alpha=0.5, cmap='Blues')
        axes[i, 2].set_title(f'Baseline\nDice: {dice_baseline[i]:.4f}', 
                            fontsize=11, weight='bold', color='#3498db')
        axes[i, 2].axis('off')
        
        # Optimized Prediction
        axes[i, 3].imshow(img)
        axes[i, 3].imshow(pred_o, alpha=0.5, cmap='Greens')
        axes[i, 3].set_title(f'Optimized\nDice: {dice_optimized[i]:.4f}', 
                            fontsize=11, weight='bold', color='#e74c3c')
        axes[i, 3].axis('off')
        
        # Add improvement indicator
        improvement = dice_optimized[i] - dice_baseline[i]
        if improvement > 0:
            axes[i, 3].text(0.5, -0.05, f'↑ +{improvement:.4f}', 
                          transform=axes[i, 3].transAxes,
                          ha='center', fontsize=10, weight='bold', color='green')
        elif improvement < 0:
            axes[i, 3].text(0.5, -0.05, f'↓ {improvement:.4f}', 
                          transform=axes[i, 3].transAxes,
                          ha='center', fontsize=10, weight='bold', color='red')
    
    plt.suptitle('Segmentation Contest: Baseline vs Optimized ResUNet', 
                fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✓ Saved contest visualization to {save_path}")
    plt.close()
    
    return dice_baseline, dice_optimized


def plot_optuna_optimization(study, save_path: str = 'results/figures/optuna_optimization.png'):
    """
    Plot Optuna optimization results.
    
    Args:
        study: Optuna study object
        save_path: Path to save figure
    """
    ensure_dir(os.path.dirname(save_path))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Optimization history
    ax = axes[0]
    trials = study.trials
    values = [t.value for t in trials if t.value is not None]
    ax.plot(values, marker='o', linestyle='-', color='#3498db', linewidth=2)
    ax.axhline(y=max(values), color='r', linestyle='--', label=f'Best: {max(values):.4f}')
    ax.set_xlabel('Trial', fontsize=11, weight='bold')
    ax.set_ylabel('Validation Dice Score', fontsize=11, weight='bold')
    ax.set_title('Optuna Optimization History', fontsize=12, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Parameter importance
    ax = axes[1]
    best_params = study.best_params
    param_names = list(best_params.keys())
    param_values = list(best_params.values())
    
    bars = ax.barh(param_names, param_values, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Parameter Value', fontsize=11, weight='bold')
    ax.set_title('Best Parameters Found', fontsize=12, weight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (name, value) in enumerate(zip(param_names, param_values)):
        ax.text(value, i, f' {value:.3f}', va='center', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✓ Saved Optuna optimization plot to {save_path}")
    plt.close()
