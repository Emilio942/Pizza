#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Fine-tuning Script for Pizza Classification using Vision Models

This script tests the fine-tuning pipeline with a more accessible vision model.
"""

import os
import sys
import json
import time
import torch
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

from transformers import (
    AutoTokenizer, 
    AutoModelForVision2Seq, 
    AutoProcessor,
    get_linear_schedule_with_warmup,
    BlipProcessor,
    BlipForConditionalGeneration
)

from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support
)

from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Pizza classes
PIZZA_CLASSES = [
    "margherita", "pepperoni", "hawaiian", "veggie", "meat",
    "seafood", "bbq", "white", "supreme", "custom"
]

# Simplified config using a more accessible model
TRAINING_CONFIG = {
    "model_name": "Salesforce/blip-image-captioning-base",  # More accessible model
    "max_length": 77,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "num_epochs": 3,  # Reduced for testing
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "early_stopping_patience": 2,
    "fp16": True,
    "dataloader_num_workers": 0,
    "seed": 42
}


class SimplePizzaDataset(Dataset):
    """Simple dataset for pizza images."""
    
    def __init__(self, image_paths: List[Path], labels: List[int], processor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            label = self.labels[idx]
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Process image
            encoding = self.processor(image, return_tensors="pt", padding=True)
            
            return {
                'pixel_values': encoding['pixel_values'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.long),
                'image_path': str(image_path)
            }
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a dummy sample
            dummy_image = Image.new('RGB', (224, 224), color='white')
            encoding = self.processor(dummy_image, return_tensors="pt", padding=True)
            return {
                'pixel_values': encoding['pixel_values'].squeeze(0),
                'label': torch.tensor(0, dtype=torch.long),
                'image_path': str(image_path)
            }


class SimplePizzaClassifier(nn.Module):
    """Simple pizza classifier using BLIP as backbone."""
    
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load the vision model
        self.vision_model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        # Freeze most of the model
        for param in self.vision_model.parameters():
            param.requires_grad = False
        
        # Get the dimension of vision features
        vision_dim = self.vision_model.config.vision_config.hidden_size
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.Linear(vision_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, pixel_values):
        # Get vision features
        vision_outputs = self.vision_model.vision_model(pixel_values=pixel_values)
        
        # Pool the features (use CLS token or mean pooling)
        pooled_features = vision_outputs.pooler_output
        
        # Classify
        logits = self.classifier(pooled_features)
        
        return logits


def create_data_loaders(data_dir: str, processor, config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    
    data_path = Path(data_dir)
    logger.info(f"Looking for data in: {data_path}")
    
    # Collect all images and labels
    image_paths = []
    labels = []
    
    if not data_path.exists():
        raise ValueError(f"Data directory {data_path} does not exist")
    
    # Look for images in subdirectories (class-based structure)
    for class_idx, class_name in enumerate(PIZZA_CLASSES):
        class_dir = data_path / class_name
        if class_dir.exists():
            for img_path in class_dir.glob("*.jpg"):
                image_paths.append(img_path)
                labels.append(class_idx)
            for img_path in class_dir.glob("*.png"):
                image_paths.append(img_path)
                labels.append(class_idx)
    
    # If no class structure, look for all images and assign random labels for testing
    if not image_paths:
        logger.warning("No class-based structure found, using all images with random labels")
        all_images = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png"))
        if all_images:
            image_paths = all_images[:20]  # Limit for testing
            labels = [np.random.randint(0, len(PIZZA_CLASSES)) for _ in image_paths]
    
    if not image_paths:
        raise ValueError(f"No images found in {data_path}")
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Create dataset
    dataset = SimplePizzaDataset(image_paths, labels, processor)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['dataloader_num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['dataloader_num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_samples = 0
    
    for batch_idx, batch in enumerate(train_loader):
        try:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            
            with autocast(enabled=config['fp16']):
                outputs = model(pixel_values)
                loss = F.cross_entropy(outputs, labels)
                loss = loss / config['gradient_accumulation_steps']
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * config['gradient_accumulation_steps']
            num_samples += labels.size(0)
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                
        except Exception as e:
            logger.error(f"Error in training batch {batch_idx}: {e}")
            continue
    
    return total_loss / len(train_loader) if len(train_loader) > 0 else 0


def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(pixel_values)
                loss = F.cross_entropy(outputs, labels)
                
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            except Exception as e:
                logger.error(f"Error in validation batch: {e}")
                continue
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0
    
    return avg_loss, accuracy


def main():
    """Main function."""
    
    # Simple argument handling
    data_dir = "augmented_pizza"
    output_dir = "models/spatial_mllm"
    
    config = TRAINING_CONFIG.copy()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Starting Test Fine-tuning for Pizza Classification")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_path}")
    
    try:
        # Load processor
        logger.info(f"Loading processor from {config['model_name']}...")
        processor = BlipProcessor.from_pretrained(config['model_name'])
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_data_loaders(data_dir, processor, config)
        
        # Initialize model
        logger.info("Initializing model...")
        model = SimplePizzaClassifier(
            model_name=config['model_name'],
            num_classes=len(PIZZA_CLASSES)
        )
        model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        total_steps = len(train_loader) * config['num_epochs'] // config['gradient_accumulation_steps']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * config['warmup_ratio']),
            num_training_steps=total_steps
        )
        
        scaler = GradScaler(enabled=config['fp16'])
        
        # Training loop
        logger.info("Starting training...")
        start_time = time.time()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config['num_epochs']):
            logger.info(f"\\nEpoch {epoch + 1}/{config['num_epochs']}")
            
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, config)
            
            # Validate
            val_loss, val_accuracy = validate(model, val_loader, device)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save model
                model_path = output_path / "pizza_finetuned_v1.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'classes': PIZZA_CLASSES,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'epoch': epoch
                }, model_path)
                
                logger.info(f"✅ New best model saved to {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= config['early_stopping_patience']:
                    logger.info("Early stopping triggered")
                    break
        
        # Final metrics
        final_metrics = {
            'training_time': time.time() - start_time,
            'best_val_loss': best_val_loss,
            'final_val_accuracy': val_accuracy,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': len(PIZZA_CLASSES),
            'classes': PIZZA_CLASSES
        }
        
        with open(output_path / "final_metrics.json", 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        logger.info(f"\\n✅ Training completed successfully!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Model saved to: {output_path / 'pizza_finetuned_v1.pth'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
