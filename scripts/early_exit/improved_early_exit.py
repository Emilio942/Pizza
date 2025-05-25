#!/usr/bin/env python3
"""
Improved MicroPizzaNet with Early Exit and Enhanced Training

This implementation adds class weights, more regularization, and better
early exit training mechanisms to improve model accuracy and energy efficiency.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path
import logging
import argparse
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.pizza_detector import (
    MicroPizzaNet, RP2040Config, MemoryEstimator,
    create_optimized_dataloaders, PizzaDatasetAnalysis
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("improved_early_exit.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImprovedMicroPizzaNetWithEarlyExit(nn.Module):
    """
    Enhanced MicroPizzaNet with improved Early-Exit functionality.
    Features stronger regularization and better designed exit paths.
    """
    def __init__(self, num_classes=6, dropout_rate=0.3, confidence_threshold=0.5):
        super(ImprovedMicroPizzaNetWithEarlyExit, self).__init__()
        
        # Confidence threshold for early exit
        self.confidence_threshold = confidence_threshold
        
        # Block 1: Standard convolution and pooling (3 -> 8 channels)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 8x12x12
        )
        
        # Block 2: Depthwise Separable convolution (8 -> 16 channels)
        self.block2 = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=8, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            # Pointwise convolution (1x1) for channel expansion
            nn.Conv2d(8, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 16x6x6
        )
        
        # Early Exit Classifier after Block 2
        # Improved with stronger capacity
        self.early_exit_pooling = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.early_exit_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 32),  # Additional layer for more capacity
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )
        
        # Block 3: Second Depthwise Separable convolution (16 -> 32 channels)
        self.block3 = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Pointwise convolution (1x1) for channel expansion
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32x3x3
        )
        
        # Main classifier (after Block 3)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )
        
        # Weight initialization for better convergence
        self._initialize_weights()
    
    def forward(self, x, use_early_exit=True, forced_exit=False):
        """
        Forward pass with optional early exit and forced early exit control
        
        Args:
            x: Input tensor
            use_early_exit: Whether early exit should be activated
            forced_exit: Force the model to use early exit regardless of confidence
            
        Returns:
            tuple: (outputs, early_exit_used)
                - outputs: Logits of prediction
                - early_exit_used: Boolean indicating if early exit was used
        """
        # First feature extraction
        x = self.block1(x)
        x = self.block2(x)
        
        # Early Exit after Block 2, if activated
        early_exit_used = False
        
        # Calculate early exit output in any case for later use
        early_features = self.early_exit_pooling(x)
        early_exit_output = self.early_exit_classifier(early_features)
        
        if forced_exit:
            # Force early exit regardless of confidence
            early_exit_used = True
            return early_exit_output, early_exit_used
        
        if use_early_exit:
            # Try early exit
            # Calculate confidence for early exit
            early_probs = F.softmax(early_exit_output, dim=1)
            
            # Get maximum confidence
            max_confidence, _ = torch.max(early_probs, dim=1)
            
            # If in inference mode (no gradients) and confidence is high enough
            if not self.training and torch.all(max_confidence >= self.confidence_threshold):
                early_exit_used = True
                return early_exit_output, early_exit_used
        
        # If early exit not used, continue normally
        x = self.block3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x, early_exit_used
    
    def _initialize_weights(self):
        """Optimized weight initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def count_parameters(self):
        """Count trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_improved_early_exit_model(model, train_loader, val_loader, config, class_names, 
                           epochs=50, early_stopping_patience=10, 
                           lambda_ee=0.5, model_name="improved_early_exit",
                           class_weights=None, output_dir=None):
    """
    Train the MicroPizzaNet model with improved early exit branch
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Configuration object
        class_names: List of class names
        epochs: Number of training epochs
        early_stopping_patience: Number of epochs without improvement before stopping
        lambda_ee: Weight of early exit loss (0-1)
        model_name: Name for saved model
        class_weights: Optional tensor of class weights for imbalanced data
        output_dir: Directory to save results
        
    Returns:
        tuple: (history, model)
    """
    if output_dir is None:
        output_dir = os.path.join(project_root, "output", "model_optimization", "improved_early_exit")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting training of {model_name} with early exit (λ={lambda_ee})...")
    logger.info(f"Using class weights: {class_weights is not None}")
    
    # Set model path
    model_dir = os.path.join(project_root, "models_optimized")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    
    # Set device
    device = config.DEVICE
    model = model.to(device)
    
    # Loss function (with class weights if provided)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Using weighted CrossEntropyLoss with weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    
    # Learning Rate Scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
    )
    
    # Initialize early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_weights = None
    
    # Training history
    history = {
        'train_loss': [],
        'train_loss_main': [],
        'train_loss_early': [],
        'val_loss': [],
        'train_acc': [],
        'train_acc_early': [],
        'val_acc': [],
        'val_acc_early': [],
        'early_exit_rate': [],
        'lr': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_loss_main = 0.0
        running_loss_early = 0.0
        correct_main = 0
        correct_early = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass (disable early exit for training, we want to train both paths)
            main_output, _ = model(inputs, use_early_exit=False)
            
            # Calculate early exit output separately (for training)
            early_features = model.early_exit_pooling(model.block2(model.block1(inputs)))
            early_output = model.early_exit_classifier(early_features)
            
            # Calculate losses
            loss_main = criterion(main_output, labels)
            loss_early = criterion(early_output, labels)
            
            # Combined loss with weighting
            loss = (1 - lambda_ee) * loss_main + lambda_ee * loss_early
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            running_loss_main += loss_main.item() * inputs.size(0)
            running_loss_early += loss_early.item() * inputs.size(0)
            
            _, preds_main = torch.max(main_output, 1)
            _, preds_early = torch.max(early_output, 1)
            
            total += labels.size(0)
            correct_main += (preds_main == labels).sum().item()
            correct_early += (preds_early == labels).sum().item()
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': loss.item(),
                'acc_main': 100 * correct_main / total,
                'acc_early': 100 * correct_early / total,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Calculate training metrics
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_loss_main = running_loss_main / len(train_loader.dataset)
        epoch_train_loss_early = running_loss_early / len(train_loader.dataset)
        epoch_train_acc = 100.0 * correct_main / total
        epoch_train_acc_early = 100.0 * correct_early / total
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_main = 0
        correct_early = 0
        early_exit_used_count = 0
        total = 0
        
        # Store per-class accuracy
        class_correct_main = [0] * len(class_names)
        class_correct_early = [0] * len(class_names)
        class_total = [0] * len(class_names)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass with activated early exit
                outputs, early_exit_used = model(inputs, use_early_exit=True)
                
                # If early exit was used, outputs is already the early exit output
                if early_exit_used:
                    early_exit_used_count += inputs.size(0)
                    # Loss of early exit output
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    # For per-class accuracy of early exit
                    for i, label in enumerate(labels):
                        label_idx = label.item()
                        class_total[label_idx] += 1
                        if preds[i] == label:
                            class_correct_early[label_idx] += 1
                else:
                    # Calculate early exit part manually
                    early_features = model.early_exit_pooling(model.block2(model.block1(inputs)))
                    early_output = model.early_exit_classifier(early_features)
                    
                    # Main loss
                    loss_main = criterion(outputs, labels)
                    loss_early = criterion(early_output, labels)
                    loss = (1 - lambda_ee) * loss_main + lambda_ee * loss_early
                    
                    # Predictions
                    _, preds_main = torch.max(outputs, 1)
                    _, preds_early = torch.max(early_output, 1)
                    
                    # For statistics
                    preds = preds_main
                    correct_early += (preds_early == labels).sum().item()
                    
                    # Per-class accuracy
                    for i, label in enumerate(labels):
                        label_idx = label.item()
                        class_total[label_idx] += 1
                        if preds_main[i] == label:
                            class_correct_main[label_idx] += 1
                        if preds_early[i] == label:
                            class_correct_early[label_idx] += 1
                
                # Update statistics
                running_val_loss += loss.item() * inputs.size(0)
                total += labels.size(0)
                correct_main += (preds == labels).sum().item()
                
                # Update progress bar
                val_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100 * correct_main / total,
                    'early_exit_rate': 100 * early_exit_used_count / total
                })
        
        # Calculate validation metrics
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = 100.0 * correct_main / total
        epoch_val_acc_early = 100.0 * correct_early / total
        early_exit_rate = 100.0 * early_exit_used_count / total
        
        # Calculate per-class validation accuracy
        per_class_acc_main = {}
        per_class_acc_early = {}
        
        for i in range(len(class_names)):
            if class_total[i] > 0:
                per_class_acc_main[class_names[i]] = 100.0 * class_correct_main[i] / class_total[i]
                per_class_acc_early[class_names[i]] = 100.0 * class_correct_early[i] / class_total[i]
            else:
                per_class_acc_main[class_names[i]] = 0.0
                per_class_acc_early[class_names[i]] = 0.0
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['train_loss_main'].append(epoch_train_loss_main)
        history['train_loss_early'].append(epoch_train_loss_early)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['train_acc_early'].append(epoch_train_acc_early)
        history['val_acc'].append(epoch_val_acc)
        history['val_acc_early'].append(epoch_val_acc_early)
        history['early_exit_rate'].append(early_exit_rate)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Output
        logger.info(f"Epoch {epoch+1}/{epochs}:")
        logger.info(f"  Train Loss: {epoch_train_loss:.4f} (Main: {epoch_train_loss_main:.4f}, Early: {epoch_train_loss_early:.4f})")
        logger.info(f"  Train Acc: {epoch_train_acc:.2f}% (Early Exit: {epoch_train_acc_early:.2f}%)")
        logger.info(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        logger.info(f"  Early Exit Rate: {early_exit_rate:.2f}%")
        
        # Output per-class accuracy
        logger.info("  Per-Class Accuracies:")
        for cls_name in class_names:
            logger.info(f"    {cls_name}: Main={per_class_acc_main[cls_name]:.2f}%, Early={per_class_acc_early[cls_name]:.2f}%")
        
        # Early stopping and model saving
        if epoch_val_loss < best_val_loss:
            logger.info(f"  Validation loss improved: {best_val_loss:.4f} -> {epoch_val_loss:.4f}")
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # Save checkpoint
            torch.save(model.state_dict(), model_path)
            logger.info(f"  Model saved: {model_path}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement. Early stopping counter: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping in epoch {epoch+1}. Best epoch was {best_epoch+1}.")
                break
    
    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
    
    # Save training history
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        serializable_history = {}
        for key, values in history.items():
            serializable_history[key] = [float(val) for val in values]
        json.dump(serializable_history, f, indent=2)
    
    # Create training visualization
    visualize_training(history, output_dir, model_name)
    
    return history, model

def visualize_training(history, output_dir, model_name):
    """
    Create visualizations of training history
    
    Args:
        history: Training history dictionary
        output_dir: Directory to save visualizations
        model_name: Name of the model
    """
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot losses
    axs[0, 0].plot(history['train_loss'], label='Train Loss')
    axs[0, 0].plot(history['train_loss_main'], label='Main Path Loss')
    axs[0, 0].plot(history['train_loss_early'], label='Early Exit Loss')
    axs[0, 0].plot(history['val_loss'], label='Validation Loss')
    axs[0, 0].set_title('Training and Validation Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot accuracies
    axs[0, 1].plot(history['train_acc'], label='Train Acc (Main)')
    axs[0, 1].plot(history['train_acc_early'], label='Train Acc (Early)')
    axs[0, 1].plot(history['val_acc'], label='Val Acc')
    axs[0, 1].plot(history['val_acc_early'], label='Val Acc (Early)')
    axs[0, 1].set_title('Training and Validation Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy (%)')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot early exit rate
    axs[1, 0].plot(history['early_exit_rate'], 'o-')
    axs[1, 0].set_title('Early Exit Rate')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Rate (%)')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot learning rate
    axs[1, 1].plot(history['lr'], 'o-')
    axs[1, 1].set_title('Learning Rate')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Learning Rate')
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_training.png"))
    plt.close()
    
    logger.info(f"Training visualization saved to {os.path.join(output_dir, f'{model_name}_training.png')}")

if __name__ == "__main__":
    print("This module provides improved early exit functionality for MicroPizzaNet.")
    print("It should be imported and used in other scripts.")
