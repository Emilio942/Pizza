#!/usr/bin/env python3
"""
Example script demonstrating how to integrate the standard augmentation pipeline
into the training process for the pizza classification model.

This script shows how to:
1. Import and configure the standard augmentation pipeline
2. Apply it to a dataset
3. Use it in a training loop

Usage:
    python scripts/train_with_augmentation.py
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random

# Import the standard augmentation pipeline
from scripts.standard_augmentation import get_standard_augmentation_pipeline

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example dataset class
class PizzaDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train'):
        """
        Pizza dataset loader
        
        Args:
            data_dir (str): Directory with class subdirectories
            transform (callable, optional): Transform to apply to images
            split (str): 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Collect image paths and labels
        self.samples = []
        self.classes = []
        
        # Find class directories
        for class_name in sorted(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                self.classes.append(class_name)
                
                # Get image files in this class directory
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        label = len(self.classes) - 1
                        self.samples.append((img_path, label))
        
        print(f"Found {len(self.samples)} images in {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            
            # Apply transformations
            if self.transform:
                img = self.transform(img)
            
            return img, label

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, num_epochs=10, save_path=None):
    """
    Train a model with the provided data loaders
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        num_epochs (int): Number of training epochs
        save_path (str, optional): Path to save the best model
        
    Returns:
        model: Trained model
        history: Training history
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Backward + optimize
                loss.backward()
                optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        
        print(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc.item())
        
        # Save best model
        if save_path and epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with validation accuracy: {best_val_acc:.4f}")
    
    return model, history

# Plot training history
def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss and accuracy
    
    Args:
        history (dict): Training history dictionary
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train pizza classification model with standard augmentation pipeline")
    parser.add_argument('--data-dir', type=str, default="data/classified",
                        help="Directory containing the dataset (with class subdirectories)")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--image-size', type=int, default=224,
                        help="Input image size")
    parser.add_argument('--aug-intensity', type=str, default='medium', choices=['low', 'medium', 'high'],
                        help="Augmentation intensity")
    parser.add_argument('--output-dir', type=str, default="output/training",
                        help="Directory to save output files")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create standard augmentation pipeline
    aug_pipeline = get_standard_augmentation_pipeline(
        image_size=args.image_size,
        intensity=args.aug_intensity
    )
    
    # Get transforms
    train_transform = aug_pipeline.get_transforms()
    
    # For validation/test, we only need to resize and normalize
    val_transform = torch.nn.Sequential(
        torch.nn.Resize((args.image_size, args.image_size)),
        torch.nn.ToTensor(),
        torch.nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    
    # Create datasets
    train_dataset = PizzaDataset(
        os.path.join(args.data_dir, 'train'),
        transform=train_transform,
        split='train'
    )
    
    val_dataset = PizzaDataset(
        os.path.join(args.data_dir, 'val'),
        transform=val_transform,
        split='val'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    num_classes = len(train_dataset.classes)
    print(f"Training model for {num_classes} classes: {train_dataset.classes}")
    
    model = models.resnet50(weights='IMAGENET1K_V2')  # Pre-trained on ImageNet
    
    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=args.epochs,
        save_path=os.path.join(args.output_dir, 'best_model.pth')
    )
    
    # Plot and save training history
    plot_training_history(
        history,
        save_path=os.path.join(args.output_dir, 'training_history.png')
    )
    
    print(f"Training completed. Model saved to {os.path.join(args.output_dir, 'best_model.pth')}")

if __name__ == '__main__':
    main()
