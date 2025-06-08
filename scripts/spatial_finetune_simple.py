#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Spatial-MLLM Fine-Tuning Script for Pizza Classification

This script implements a simplified Transfer Learning approach from the pretrained 
Spatial-MLLM to pizza classification, focusing on getting the basic pipeline working.

SPATIAL-2.3 Implementation (Simplified)
Author: GitHub Copilot (2025-06-06)
"""

import os
import sys
import json
import time
import torch
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from PIL import Image

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

from transformers import (
    AutoTokenizer, 
    AutoModelForVision2Seq, 
    AutoProcessor,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report,
    confusion_matrix
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configuration
PIZZA_CLASSES = ['basic', 'burnt']  # Start with binary classification
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(PIZZA_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for cls_name, idx in CLASS_TO_IDX.items()}

# Training parameters - optimized for pizza classification
TRAINING_CONFIG = {
    "model_name": "Diankun/Spatial-MLLM-subset-sft",
    "max_length": 512,
    "learning_rate": 5e-5,  # Slightly higher for simpler approach
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "num_epochs": 5,
    "batch_size": 1,  # Small batch size due to large model
    "gradient_accumulation_steps": 8,  # Effective batch size = 8
    "max_grad_norm": 1.0,
    "early_stopping_patience": 3,
    "fp16": True,
    "dataloader_num_workers": 0,  # Reduced to avoid issues
    "seed": 42
}


class SimplePizzaDataset(Dataset):
    """
    Simplified dataset that uses regular pizza images instead of spatial data.
    """
    
    def __init__(
        self, 
        data_dir: Union[str, Path],
        processor: AutoProcessor,
        class_to_idx: Dict[str, int],
        max_length: int = 512,
        split: str = "train"
    ):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.class_to_idx = class_to_idx
        self.max_length = max_length
        self.split = split
        
        # Look for pizza images in parent directory
        self.image_dir = self.data_dir.parent / "augmented_pizza"
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        logger.info(f"Class distribution: {self._get_class_distribution()}")
    
    def _load_samples(self) -> List[Tuple[Path, str, int]]:
        """Load pizza images from augmented_pizza directory."""
        samples = []
        
        for class_name in self.class_to_idx.keys():
            class_dir = self.image_dir / class_name
            if class_dir.exists():
                for img_file in class_dir.glob("*.jpg"):
                    label_idx = self.class_to_idx[class_name]
                    samples.append((img_file, class_name, label_idx))
        
        # Limit to a small subset for demonstration
        samples = samples[:30]  # Use only 30 images total
        
        if not samples:
            raise ValueError(f"No valid pizza images found in {self.image_dir}")
        
        return samples
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution statistics."""
        distribution = {}
        for _, class_name, _ in self.samples:
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution
    
    def _create_classification_prompt(self, class_name: str) -> str:
        """Create a prompt for pizza classification."""
        if self.split == "train":
            if class_name == "basic":
                return "This is a properly cooked pizza with golden brown color. It is not burnt."
            else:  # burnt
                return "This is a burnt pizza with dark, overcooked areas."
        else:
            return "Look at this pizza image. Is this pizza burnt or properly cooked (basic)? Answer with just 'burnt' or 'basic':"
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path, class_name, label_idx = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(file_path).convert("RGB")
            
            # Create prompt
            prompt = self._create_classification_prompt(class_name)
            
            # Create messages in the expected format for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Process with the processor
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'image_grid_thw': inputs.get('image_grid_thw', torch.tensor([[1, 1, 1]])).squeeze(0),
                'labels': torch.tensor(label_idx, dtype=torch.long),
                'class_name': class_name,
                'file_path': str(file_path)
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {file_path}: {e}")
            # Return a dummy sample
            dummy_image = Image.new('RGB', (224, 224), color='red')
            dummy_text = "This is a pizza."
            
            inputs = self.processor(
                text=[dummy_text],
                images=[dummy_image],
                padding=True,
                return_tensors="pt"
            )
            
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'image_grid_thw': inputs.get('image_grid_thw', torch.tensor([[1, 1, 1]])).squeeze(0),
                'labels': torch.tensor(0, dtype=torch.long),
                'class_name': 'basic',
                'file_path': str(file_path)
            }


class SimplePizzaClassifier(nn.Module):
    """
    Simplified wrapper around Spatial-MLLM for pizza classification.
    """
    
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
        # Load pretrained model
        print("Loading Spatial-MLLM model...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Get hidden size
        hidden_size = getattr(self.model.config, 'hidden_size', 2048)
        
        # Add simple classification head
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        logger.info(f"Initialized SimplePizzaClassifier with {num_classes} classes")
    
    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw=None, labels=None):
        """Forward pass through the model."""
        
        # Forward through the base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True
        )
        
        # Extract last hidden state for classification
        if hasattr(outputs, 'last_hidden_state'):
            # Use the last token's representation
            sequence_output = outputs.last_hidden_state[:, -1, :]
        else:
            # Fallback: use mean pooling of logits
            sequence_output = torch.mean(outputs.logits, dim=1)
        
        # Classification
        logits = self.classifier(sequence_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }


def create_data_loaders(data_dir: str, processor: AutoProcessor, config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    
    # Create full dataset
    full_dataset = SimplePizzaDataset(
        data_dir=data_dir,
        processor=processor,
        class_to_idx=CLASS_TO_IDX,
        max_length=config['max_length'],
        split="train"
    )
    
    # Split into train and validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['dataloader_num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['dataloader_num_workers'],
        pin_memory=True
    )
    
    logger.info(f"Created data loaders: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    return train_loader, val_loader


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
                scheduler: optim.lr_scheduler.LRScheduler, scaler: GradScaler,
                device: torch.device, config: Dict[str, Any]) -> Dict[str, float]:
    """Train for one epoch."""
    
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        image_grid_thw = batch['image_grid_thw'].to(device) if 'image_grid_thw' in batch else None
        labels = batch['labels'].to(device)
        
        # Forward pass with mixed precision
        with autocast(enabled=config['fp16']):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels
            )
            
            loss = outputs['loss'] / config['gradient_accumulation_steps']
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Calculate accuracy
        with torch.no_grad():
            predictions = torch.argmax(outputs['logits'], dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
        
        # Gradient accumulation
        if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
        
        total_loss += loss.item() * config['gradient_accumulation_steps']
        
        if batch_idx % 5 == 0:
            logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    accuracy = correct_predictions / max(total_predictions, 1)
    avg_loss = total_loss / len(train_loader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def evaluate_model(model: nn.Module, val_loader: DataLoader, device: torch.device,
                  config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the model."""
    
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            image_grid_thw = batch['image_grid_thw'].to(device) if 'image_grid_thw' in batch else None
            labels = batch['labels'].to(device)
            
            # Forward pass
            with autocast(enabled=config['fp16']):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=labels
                )
            
            predictions = torch.argmax(outputs['logits'], dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += outputs['loss'].item()
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(description="Simple fine-tuning of Spatial-MLLM for pizza classification")
    parser.add_argument("--data_dir", type=str, default="data/spatial_processed",
                       help="Directory containing data (will look for ../augmented_pizza)")
    parser.add_argument("--output_dir", type=str, default="models/spatial_mllm",
                       help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG['num_epochs'],
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=TRAINING_CONFIG['batch_size'],
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=TRAINING_CONFIG['learning_rate'],
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = TRAINING_CONFIG.copy()
    config.update({
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
    })
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Starting Simplified Spatial-MLLM Fine-tuning for Pizza Classification")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Training configuration: {config}")
    
    try:
        # Load processor
        logger.info(f"Loading processor from {config['model_name']}...")
        processor = AutoProcessor.from_pretrained(config['model_name'], trust_remote_code=True)
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_data_loaders(args.data_dir, processor, config)
        
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
        warmup_steps = int(total_steps * config['warmup_ratio'])
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Mixed precision scaler
        scaler = GradScaler(enabled=config['fp16'])
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
        best_val_loss = float('inf')
        start_time = time.time()
        
        # Training loop
        logger.info("Starting training...")
        for epoch in range(config['num_epochs']):
            logger.info(f"\\nEpoch {epoch + 1}/{config['num_epochs']}")
            logger.info("-" * 50)
            
            # Train
            train_metrics = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, config)
            
            # Evaluate
            val_metrics = evaluate_model(model, val_loader, device, config)
            
            # Log results
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                
                # Save final model
                final_model_path = output_dir / "pizza_finetuned_v1.pth"
                torch.save(model.state_dict(), final_model_path)
                logger.info(f"New best model saved: {final_model_path}")
        
        # Save training history and metrics
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        final_metrics = {
            'training_time': time.time() - start_time,
            'best_val_loss': best_val_loss,
            'final_val_metrics': val_metrics,
            'model_info': {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_name': config['model_name'],
                'num_classes': len(PIZZA_CLASSES),
                'classes': PIZZA_CLASSES
            }
        }
        
        with open(output_dir / "final_metrics.json", 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        logger.info(f"\\n✅ Training completed successfully!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Final model saved to: {final_model_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
