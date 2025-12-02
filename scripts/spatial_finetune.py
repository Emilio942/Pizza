#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial-MLLM Fine-Tuning Script for Pizza Classification

This script implements Transfer Learning from the pretrained Spatial-MLLM
to the pizza classification task. It includes:

1. Custom dataset loader for spatial-processed pizza data
2. Fine-tuning pipeline with proper loss functions and metrics
3. Evaluation metrics specific to spatial features
4. Model saving and monitoring capabilities

SPATIAL-2.3 Implementation
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

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
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

import matplotlib.pyplot as plt
import seaborn as sns

# Setup project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append("/home/emilio/Documents/ai/Spatial-MLLM")

try:
    from src.models import Qwen2_5_VL_VGGTForConditionalGeneration, Qwen2_5_VLProcessor
    from qwen_vl_utils import process_vision_info
    SPATIAL_MLLM_AVAILABLE = True
    print("✅ Spatial-MLLM modules imported successfully")
except ImportError as e:
    print(f"⚠️  Warning: Could not import Spatial-MLLM modules: {e}")
    print("Proceeding with standard transformers modules...")
    SPATIAL_MLLM_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
PIZZA_CLASSES = ['basic', 'burnt']  # Start with binary classification
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(PIZZA_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for cls_name, idx in CLASS_TO_IDX.items()}

# Training parameters - optimized for pizza classification
TRAINING_CONFIG = {
    "model_name": "Diankun/Spatial-MLLM-subset-sft",
    "max_length": 128,
    "learning_rate": 2e-5,  # Conservative LR for fine-tuning
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "num_epochs": 10,
    "batch_size": 4,  # Small batch size due to large model
    "gradient_accumulation_steps": 4,  # Effective batch size = 16
    "max_grad_norm": 1.0,
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch",
    "early_stopping_patience": 3,
    "fp16": True,  # Mixed precision training
    "dataloader_num_workers": 2,
    "seed": 42
}


class SpatialPizzaDataset(Dataset):
    """
    Dataset class for spatial-processed pizza data.
    Loads preprocessed spatial data and creates prompts for classification.
    """
    
    def __init__(
        self, 
        data_dir: Union[str, Path],
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        class_to_idx: Dict[str, int],
        max_length: int = 128,
        split: str = "train"
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.processor = processor
        self.class_to_idx = class_to_idx
        self.max_length = max_length
        self.split = split
        
        # Load spatial data files
        self.samples = self._load_samples()
        
        # Calculate class weights for balanced sampling
        self.class_weights = self._calculate_class_weights()
        self.sample_weights = self._get_sample_weights()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        logger.info(f"Class distribution: {self._get_class_distribution()}")
    
    def _load_samples(self) -> List[Tuple[Path, str, int]]:
        """Load all spatial data files and extract labels from filenames."""
        samples = []
        
        for file_path in self.data_dir.glob("*.pt"):
            if "spatial" in file_path.name:
                # Extract class from filename (e.g., pizza_0_basic_2_0_spatial.pt -> basic)
                parts = file_path.stem.split("_")
                if len(parts) >= 3:
                    class_name = parts[2]  # Should be 'basic' or 'burnt'
                    if class_name in self.class_to_idx:
                        label_idx = self.class_to_idx[class_name]
                        samples.append((file_path, class_name, label_idx))
        
        if not samples:
            raise ValueError(f"No valid spatial data files found in {self.data_dir}")
        
        return samples
    
    def _calculate_class_weights(self) -> Dict[str, float]:
        """Calculate class weights for balanced training."""
        class_counts = {}
        for _, class_name, _ in self.samples:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        total_samples = len(self.samples)
        class_weights = {}
        for class_name, count in class_counts.items():
            class_weights[class_name] = total_samples / (len(class_counts) * count)
        
        return class_weights
    
    def _get_sample_weights(self) -> List[float]:
        """Get sample weights for WeightedRandomSampler."""
        return [self.class_weights[class_name] for _, class_name, _ in self.samples]
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution statistics."""
        distribution = {}
        for _, class_name, _ in self.samples:
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution
    
    def _create_classification_prompt(self, class_name: str) -> str:
        """Create a prompt for pizza classification."""
        prompts = {
            "basic": "Analyze this pizza image. Is this pizza basic (not burnt)? Answer: Yes, this pizza appears basic and properly cooked.",
            "burnt": "Analyze this pizza image. Is this pizza burnt? Answer: Yes, this pizza appears burnt with dark areas."
        }
        
        # For training, use the target answer
        return prompts.get(class_name, "Analyze this pizza image and classify its cooking state.")
    
    def _create_evaluation_prompt(self) -> str:
        """Create a prompt for evaluation/inference."""
        return """Analyze this pizza image and classify its cooking state. 
        
Consider the following spatial and visual characteristics:
- Surface texture and color variations
- Burning patterns and distribution
- Overall cooking state and appearance
        
Please classify this pizza as either:
- basic: properly cooked, golden brown, not burnt
- burnt: overcooked with dark/black areas

Answer with just the classification: """
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path, class_name, label_idx = self.samples[idx]
        
        try:
            # Load spatial data
            spatial_data = torch.load(file_path, weights_only=False)
            
            # Extract visual and spatial inputs
            visual_input = spatial_data['visual_input']  # (1, 1, 3, 518, 518)
            spatial_input = spatial_data['spatial_input']  # (1, 1, 4, 518, 518)
            
            # Create prompt based on split
            if self.split == "train":
                prompt = self._create_classification_prompt(class_name)
            else:
                prompt = self._create_evaluation_prompt()
            
            # Convert visual input to PIL Image for processor
            # Remove batch and frame dimensions: (1, 1, 3, 518, 518) -> (3, 518, 518)
            visual_tensor = visual_input.squeeze(0).squeeze(0)  # (3, 518, 518)
            
            # Convert to PIL Image (processor expects PIL)
            from torchvision.transforms.functional import to_pil_image
            visual_pil = to_pil_image(visual_tensor)
            
            # Create messages in the expected format for Spatial-MLLM
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": visual_pil},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Process with the processor
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            )
            
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'visual_input': visual_input.squeeze(0),  # Remove batch dim: (1, 3, 518, 518)
                'spatial_input': spatial_input.squeeze(0),  # Remove batch dim: (1, 4, 518, 518)
                'labels': torch.tensor(label_idx, dtype=torch.long),
                'class_name': class_name,
                'file_path': str(file_path)
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {file_path}: {e}")
            # Return a dummy sample
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'visual_input': torch.zeros(1, 3, 518, 518),
                'spatial_input': torch.zeros(1, 4, 518, 518),
                'labels': torch.tensor(0, dtype=torch.long),
                'class_name': 'basic',
                'file_path': str(file_path)
            }


class SpatialPizzaClassifier(nn.Module):
    """
    Wrapper around Spatial-MLLM for pizza classification.
    Adds a classification head on top of the pretrained model.
    """
    
    def __init__(self, model_name: str, num_classes: int, freeze_base: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_base = freeze_base
        
        # Load pretrained Spatial-MLLM
        if SPATIAL_MLLM_AVAILABLE:
            try:
                self.spatial_mllm = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map=None,  # We'll handle device placement manually
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                print("✅ Using Spatial-MLLM architecture")
            except Exception as e:
                print(f"⚠️  Failed to load Spatial-MLLM: {e}")
                print("Falling back to standard transformers...")
                self.spatial_mllm = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map=None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
        else:
            # Fallback to standard transformers
            print("Using standard transformers model as fallback")
            self.spatial_mllm = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        # Freeze base model if requested
        if freeze_base:
            for param in self.spatial_mllm.parameters():
                param.requires_grad = False
        
        # Get hidden size from the model
        if hasattr(self.spatial_mllm.config, 'hidden_size'):
            hidden_size = self.spatial_mllm.config.hidden_size
        else:
            hidden_size = 2048  # Default for Qwen2.5-VL
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        logger.info(f"Initialized SpatialPizzaClassifier with {num_classes} classes")
        logger.info(f"Base model frozen: {freeze_base}")
    
    def forward(self, input_ids, attention_mask, visual_input, spatial_input, labels=None):
        """Forward pass through the model."""
        
        # Prepare inputs for the model
        batch_size = input_ids.size(0)
        
        # Create pixel_values from visual_input
        # visual_input shape: (batch_size, 1, 3, 518, 518)
        pixel_values = visual_input.squeeze(1)  # Remove frame dimension
        
        # For Qwen2.5-VL, we need to provide image_grid_thw
        # This represents the temporal, height, width grid for each image
        # For single images: [1, height_patches, width_patches]
        image_size = pixel_values.shape[-1]  # 518
        patch_size = 14  # Standard patch size for vision transformers
        grid_size = image_size // patch_size  # Number of patches per dimension
        
        # Create grid_thw for each image in the batch
        # Format: [(temporal, height, width), ...]
        image_grid_thw = torch.tensor(
            [[1, grid_size, grid_size]] * batch_size, 
            device=pixel_values.device, 
            dtype=torch.long
        )
        
        try:
            outputs = self.spatial_mllm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                return_dict=True
            )
        except Exception as e:
            # Fallback without image_grid_thw if it fails
            print(f"⚠️  Trying fallback without image_grid_thw: {e}")
            outputs = self.spatial_mllm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_dict=True
            )
        
        # Extract hidden states from the last layer
        # For classification, we'll use the pooled output or last hidden state
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
            # Use the last token's hidden state for classification
            pooled_output = hidden_states[:, -1, :]  # (batch_size, hidden_size)
        else:
            # Fallback: use the model's hidden states or logits
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                pooled_output = outputs.hidden_states[-1][:, -1, :]
            elif hasattr(outputs, 'logits'):
                # Use the last token from logits
                pooled_output = outputs.logits[:, -1, :]
            else:
                # Last resort: create a pooled representation
                pooled_output = torch.mean(outputs.logits, dim=1) if hasattr(outputs, 'logits') else torch.zeros(batch_size, self.classifier[1].in_features, device=pixel_values.device)
        
        # Pass through classification head
        logits = self.classifier(pooled_output)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': pooled_output
        }


class SpatialMetricsCalculator:
    """Calculate evaluation metrics specific to spatial features."""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.true_labels = []
        self.confidence_scores = []
        self.spatial_features = []
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor, 
               confidence_scores: torch.Tensor, spatial_info: Optional[Dict] = None):
        """Update metrics with new batch."""
        self.predictions.extend(predictions.cpu().numpy())
        self.true_labels.extend(labels.cpu().numpy())
        self.confidence_scores.extend(confidence_scores.cpu().numpy())
        
        if spatial_info:
            self.spatial_features.append(spatial_info)
    
    def compute(self) -> Dict[str, Any]:
        """Compute all metrics."""
        predictions = np.array(self.predictions)
        true_labels = np.array(self.true_labels)
        confidence_scores = np.array(self.confidence_scores)
        
        # Basic classification metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = \
            precision_recall_fscore_support(true_labels, predictions, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Confidence statistics
        confidence_stats = {
            'mean_confidence': np.mean(confidence_scores),
            'std_confidence': np.std(confidence_scores),
            'min_confidence': np.min(confidence_scores),
            'max_confidence': np.max(confidence_scores)
        }
        
        # Spatial-specific metrics (if available)
        spatial_metrics = self._compute_spatial_metrics()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'per_class_metrics': {
                self.class_names[i]: {
                    'precision': precision_per_class[i],
                    'recall': recall_per_class[i],
                    'f1_score': f1_per_class[i],
                    'support': support[i]
                } for i in range(len(self.class_names))
            },
            'confusion_matrix': cm.tolist(),
            'confidence_stats': confidence_stats,
            'spatial_metrics': spatial_metrics,
            'classification_report': classification_report(
                true_labels, predictions, target_names=self.class_names, output_dict=True
            )
        }
    
    def _compute_spatial_metrics(self) -> Dict[str, Any]:
        """Compute spatial-specific metrics."""
        # Placeholder for spatial-specific metrics
        # These would be computed based on the spatial features
        return {
            'spatial_consistency': 0.0,  # How consistent spatial features are
            'depth_utilization': 0.0,    # How well depth information is used
            'surface_analysis_quality': 0.0  # Quality of surface analysis
        }


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.patience_counter = 0
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            self.should_stop = True
        
        return self.should_stop


def create_data_loaders(data_dir: str, tokenizer: AutoTokenizer, processor: AutoProcessor,
                       config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    
    # Create full dataset
    full_dataset = SpatialPizzaDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
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
    
    # Update split for validation dataset
    for i in range(len(val_dataset)):
        val_dataset.dataset.split = "val"
    
    # Create weighted sampler for training
    train_weights = [full_dataset.sample_weights[idx] for idx in train_dataset.indices]
    sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=config['dataloader_num_workers'],
        pin_memory=True,
        drop_last=True
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
    num_batches = 0
    
    metrics_calc = SpatialMetricsCalculator(PIZZA_CLASSES)
    
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        visual_input = batch['visual_input'].to(device)
        spatial_input = batch['spatial_input'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass with mixed precision
        with autocast(enabled=config['fp16']):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_input=visual_input,
                spatial_input=spatial_input,
                labels=labels
            )
            
            loss = outputs['loss'] / config['gradient_accumulation_steps']
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Update metrics
        with torch.no_grad():
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
            confidence_scores = torch.softmax(logits, dim=-1).max(dim=-1)[0]
            
            metrics_calc.update(predictions, labels, confidence_scores)
        
        # Gradient accumulation
        if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if scheduler:
                scheduler.step()
        
        total_loss += loss.item() * config['gradient_accumulation_steps']
        num_batches += 1
        
        # Log progress
        if batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    # Compute epoch metrics
    epoch_metrics = metrics_calc.compute()
    epoch_metrics['average_loss'] = total_loss / num_batches
    
    return epoch_metrics


def evaluate_model(model: nn.Module, val_loader: DataLoader, device: torch.device,
                  config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the model on validation set."""
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    metrics_calc = SpatialMetricsCalculator(PIZZA_CLASSES)
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            visual_input = batch['visual_input'].to(device)
            spatial_input = batch['spatial_input'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            with autocast(enabled=config['fp16']):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    visual_input=visual_input,
                    spatial_input=spatial_input,
                    labels=labels
                )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Update metrics
            predictions = torch.argmax(logits, dim=-1)
            confidence_scores = torch.softmax(logits, dim=-1).max(dim=-1)[0]
            
            metrics_calc.update(predictions, labels, confidence_scores)
            
            total_loss += loss.item()
            num_batches += 1
    
    # Compute evaluation metrics
    eval_metrics = metrics_calc.compute()
    eval_metrics['average_loss'] = total_loss / num_batches
    
    return eval_metrics


def save_model(model: nn.Module, tokenizer: AutoTokenizer, processor: AutoProcessor,
               save_dir: Path, epoch: int, metrics: Dict[str, Any]):
    """Save model checkpoint with metadata."""
    
    checkpoint_dir = save_dir / f"checkpoint-epoch-{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), checkpoint_dir / "model.pth")
    
    # Save tokenizer and processor
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
    processor.save_pretrained(checkpoint_dir / "processor")
    
    # Save metrics and config
    with open(checkpoint_dir / "metrics.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, (np.int64, np.float64)):
                serializable_metrics[key] = value.item()
            else:
                serializable_metrics[key] = value
        
        json.dump(serializable_metrics, f, indent=2)
    
    # Save training config
    with open(checkpoint_dir / "config.json", 'w') as f:
        json.dump(TRAINING_CONFIG, f, indent=2)
    
    logger.info(f"Model checkpoint saved to {checkpoint_dir}")


def plot_training_history(history: Dict[str, List], save_path: Path):
    """Plot and save training history."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(history['train_accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score plot
    axes[1, 0].plot(history['train_f1'], label='Train F1')
    axes[1, 0].plot(history['val_f1'], label='Validation F1')
    axes[1, 0].set_title('Training and Validation F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate plot
    if 'learning_rate' in history:
        axes[1, 1].plot(history['learning_rate'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Tracked', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training history plot saved to {save_path}")


def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(description="Fine-tune Spatial-MLLM for pizza classification")
    parser.add_argument("--data_dir", type=str, default="data/spatial_processed",
                       help="Directory containing spatial-processed pizza data")
    parser.add_argument("--output_dir", type=str, default="models/spatial_mllm",
                       help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG['num_epochs'],
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=TRAINING_CONFIG['batch_size'],
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=TRAINING_CONFIG['learning_rate'],
                       help="Learning rate")
    parser.add_argument("--freeze_base", action="store_true",
                       help="Freeze the base Spatial-MLLM model")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = TRAINING_CONFIG.copy()
    config.update({
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
    })
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Starting Spatial-MLLM Fine-tuning for Pizza Classification")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Training configuration: {config}")
    
    try:
        # Load tokenizer and processor
        logger.info(f"Loading tokenizer and processor from {config['model_name']}...")
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(config['model_name'], trust_remote_code=True)
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_data_loaders(
            args.data_dir, tokenizer, processor, config
        )
        
        # Initialize model
        logger.info("Initializing Spatial-MLLM model...")
        model = SpatialPizzaClassifier(
            model_name=config['model_name'],
            num_classes=len(PIZZA_CLASSES),
            freeze_base=args.freeze_base
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
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'],
            mode='min'  # Monitor validation loss
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rate': []
        }
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        # Training loop
        logger.info("Starting training...")
        for epoch in range(config['num_epochs']):
            epoch_start_time = time.time()
            
            logger.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
            logger.info("-" * 50)
            
            # Train
            train_metrics = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, config)
            
            # Evaluate
            val_metrics = evaluate_model(model, val_loader, device, config)
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            logger.info(f"Train - Loss: {train_metrics['average_loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_score']:.4f}")
            logger.info(f"Val   - Loss: {val_metrics['average_loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
            
            # Update history
            history['train_loss'].append(train_metrics['average_loss'])
            history['val_loss'].append(val_metrics['average_loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['train_f1'].append(train_metrics['f1_score'])
            history['val_f1'].append(val_metrics['f1_score'])
            
            # Save best model
            if val_metrics['average_loss'] < best_val_loss:
                best_val_loss = val_metrics['average_loss']
                save_model(model, tokenizer, processor, output_dir, epoch + 1, val_metrics)
                logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
            
            # Early stopping check
            if early_stopping(val_metrics['average_loss']):
                logger.info(f"Early stopping triggered after epoch {epoch + 1}")
                break
        
        # Save final model
        final_model_path = output_dir / "pizza_finetuned_v1.pth"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        # Save training history
        history_path = output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot training history
        plot_path = output_dir / "training_history.png"
        plot_training_history(history, plot_path)
        
        # Save final metrics
        metrics_path = output_dir / "final_metrics.json"
        final_metrics = {
            'training_time': time.time() - start_time,
            'best_val_loss': best_val_loss,
            'total_epochs': len(history['train_loss']),
            'final_train_metrics': train_metrics,
            'final_val_metrics': val_metrics,
            'model_info': {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_name': config['model_name'],
                'num_classes': len(PIZZA_CLASSES),
                'classes': PIZZA_CLASSES
            }
        }
        
        with open(metrics_path, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.float64)):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            final_metrics = convert_numpy(final_metrics)
            json.dump(final_metrics, f, indent=2)
        
        training_time = time.time() - start_time
        logger.info(f"\n✅ Training completed successfully!")
        logger.info(f"Total training time: {training_time:.2f}s")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Final model saved to: {final_model_path}")
        logger.info(f"Training metrics saved to: {metrics_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
