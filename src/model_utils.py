"""
Brain Tumor Segmentation Model Utilities - PyTorch Version
ResUNet architecture with Tversky Loss and Metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import numpy as np


class TverskyLoss(nn.Module):
    """
    Tversky Loss function for imbalanced segmentation.
    
    Tversky Index = TP / (TP + alpha*FN + beta*FP)
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6):
        """
        Args:
            alpha: Weight for False Negatives (higher = penalize missing tumors more)
            beta: Weight for False Positives (higher = penalize false alarms more)
            smooth: Smoothing factor to avoid division by zero
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Predictions [B, 1, H, W]
            y_true: Ground truth [B, 1, H, W]
        """
        # Flatten
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)
        
        # Calculate components
        TP = (y_true_flat * y_pred_flat).sum()
        FN = (y_true_flat * (1 - y_pred_flat)).sum()
        FP = ((1 - y_true_flat) * y_pred_flat).sum()
        
        # Tversky index
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        
        return 1 - tversky_index


def dice_coefficient(y_pred, y_true, smooth: float = 1e-6):
    """
    Dice Coefficient metric for segmentation.
    
    Args:
        y_pred: Predictions [B, 1, H, W]
        y_true: Ground truth [B, 1, H, W]
    """
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    
    intersection = (y_true_flat * y_pred_flat).sum()
    dice = (2. * intersection + smooth) / (y_true_flat.sum() + y_pred_flat.sum() + smooth)
    
    return dice


def iou_coefficient(y_pred, y_true, smooth: float = 1e-6):
    """
    Intersection over Union (IoU) metric.
    
    Args:
        y_pred: Predictions [B, 1, H, W]
        y_true: Ground truth [B, 1, H, W]
    """
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    
    intersection = (y_true_flat * y_pred_flat).sum()
    union = y_true_flat.sum() + y_pred_flat.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out


class ResUNet(nn.Module):
    """
    ResUNet architecture for medical image segmentation.
    
    Architecture:
    - Encoder: 4 residual blocks with downsampling
    - Bridge: 1 residual block
    - Decoder: 4 residual blocks with upsampling and skip connections
    - Output: Sigmoid activation for binary segmentation
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1, filters_base: int = 16):
        super(ResUNet, self).__init__()
        
        # Encoder
        self.enc1 = ResidualBlock(in_channels, filters_base, stride=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = ResidualBlock(filters_base, filters_base * 2, stride=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = ResidualBlock(filters_base * 2, filters_base * 4, stride=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = ResidualBlock(filters_base * 4, filters_base * 8, stride=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bridge
        self.bridge = ResidualBlock(filters_base * 8, filters_base * 16, stride=1)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(filters_base * 16, filters_base * 8, 
                                      kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(filters_base * 16, filters_base * 8, stride=1)
        
        self.up3 = nn.ConvTranspose2d(filters_base * 8, filters_base * 4,
                                      kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(filters_base * 8, filters_base * 4, stride=1)
        
        self.up2 = nn.ConvTranspose2d(filters_base * 4, filters_base * 2,
                                      kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(filters_base * 4, filters_base * 2, stride=1)
        
        self.up1 = nn.ConvTranspose2d(filters_base * 2, filters_base,
                                      kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(filters_base * 2, filters_base, stride=1)
        
        # Output
        self.output = nn.Conv2d(filters_base, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bridge
        bridge = self.bridge(self.pool4(enc4))
        
        # Decoder with skip connections
        dec4 = self.up4(bridge)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        out = self.output(dec1)
        out = self.sigmoid(out)
        
        return out


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model: nn.Module, 
                dataloader, 
                criterion, 
                optimizer, 
                device: torch.device) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        Dictionary with average metrics
    """
    model.train()
    
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            dice = dice_coefficient(outputs, masks)
            iou = iou_coefficient(outputs, masks)
        
        total_loss += loss.item()
        total_dice += dice.item()
        total_iou += iou.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'iou': total_iou / num_batches
    }


def validate(model: nn.Module, 
             dataloader, 
             criterion, 
             device: torch.device) -> Dict[str, float]:
    """
    Validate the model.
    
    Returns:
        Dictionary with average metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            dice = dice_coefficient(outputs, masks)
            iou = iou_coefficient(outputs, masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
            total_iou += iou.item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'iou': total_iou / num_batches
    }


class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0, mode: str = 'max'):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


def train_model(model: nn.Module,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                device: torch.device,
                num_epochs: int = 30,
                patience: int = 10,
                save_path: str = 'best_model.pth') -> Dict[str, List[float]]:
    """
    Complete training loop with early stopping.
    
    Returns:
        Dictionary with training history
    """
    history = {
        'train_loss': [], 'train_dice': [], 'train_iou': [],
        'val_loss': [], 'val_dice': [], 'val_iou': []
    }
    
    early_stopping = EarlyStopping(patience=patience, mode='max')
    best_dice = 0.0
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_dice'].append(train_metrics['dice'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        
        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': best_dice,
            }, save_path)
            print(f"  ✓ Saved best model (Dice: {best_dice:.4f})")
        
        # Early stopping
        if early_stopping(val_metrics['dice']):
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    return history


def calculate_dice_per_sample(model: nn.Module, 
                               dataloader, 
                               device: torch.device,
                               threshold: float = 0.5) -> List[float]:
    """
    Calculate Dice coefficient for each sample in the dataloader.
    
    Returns:
        List of Dice scores
    """
    model.eval()
    dice_scores = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            # Calculate per-sample Dice
            for i in range(len(images)):
                pred = (outputs[i] > threshold).float()
                true = masks[i]
                dice = dice_coefficient(pred.unsqueeze(0), true.unsqueeze(0))
                dice_scores.append(dice.item())
    
    return dice_scores
