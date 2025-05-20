#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pizza Diffusion Model Integration Training Pipeline

This script integrates the advanced diffusion-generated synthetic data
with the existing pizza recognition training pipeline. It provides:

1. Balanced dataset creation with both real and synthetic data
2. Comparative evaluation of model performance with and without synthetic data
3. Automated hyperparameter search for optimal synthetic/real data ratio
4. Visualization and reporting of improvements from synthetic data

Author: GitHub Copilot (2025-05-10)
"""

import os
import sys
import time
import json
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset, random_split
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from PIL import Image

# Setup project paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import pizza diffusion generator
from src.augmentation.diffusion_pizza_generator import PizzaDiffusionGenerator, PIZZA_STAGES
from src.augmentation.advanced_pizza_diffusion_control import AdvancedPizzaDiffusionControl

# Try to import project modules
try:
    from src.pizza_detector import PizzaModel, train_model, evaluate_model
    from src.utils.utils import setup_logging, save_config
    PROJECT_MODULES_AVAILABLE = True
except ImportError:
    PROJECT_MODULES_AVAILABLE = False
    print("Warning: Project modules not found. Using simplified implementation.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('diffusion_training.log')
    ]
)
logger = logging.getLogger(__name__)

# Default training parameters
DEFAULT_PARAMS = {
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "synthetic_ratio": 0.5,  # Ratio of synthetic data to use
    "augment_real": True,    # Whether to augment real data
    "early_stopping": 5,     # Patience for early stopping
    "mixup_alpha": 0.2,      # Alpha parameter for mixup
    "optimizer": "adam",     # Optimizer to use
    "scheduler": "cosine",   # Learning rate scheduler
    "experiment_name": "diffusion_training_" + datetime.now().strftime("%Y%m%d_%H%M%S")
}

class PizzaDataset(Dataset):
    """
    Dataset for pizza cooking stages with support for both real and synthetic data.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        transform=None,
        is_synthetic: bool = False,
        class_map: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the data
            transform: Optional torchvision transforms to apply
            is_synthetic: Whether this is synthetic data
            class_map: Optional mapping from class names to indices
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_synthetic = is_synthetic
        
        # Scan for images
        self.images = []
        self.labels = []
        self.metadata = []
        
        # Use provided class map or load from PIZZA_STAGES
        if class_map is not None:
            self.class_map = class_map
        else:
            self.class_map = {stage: i for i, stage in enumerate(PIZZA_STAGES.keys())}
        
        self.classes = list(self.class_map.keys())
        
        # Scan directories for images
        self._scan_directories()
        
        logger.info(f"Loaded {'synthetic' if is_synthetic else 'real'} dataset with {len(self.images)} images")
        logger.info(f"Class distribution: {self._get_class_distribution()}")
    
    def _scan_directories(self):
        """Scan directories for images and assign labels"""
        # We can scan for specific class directories or use image filenames to determine classes
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            
            # If the class directory exists, use that
            if class_dir.exists() and class_dir.is_dir():
                for img_path in class_dir.glob("*.*"):
                    if self._is_image_file(img_path):
                        self.images.append(img_path)
                        self.labels.append(self.class_map[class_name])
                        self.metadata.append({"class": class_name, "path": str(img_path), "synthetic": self.is_synthetic})
            
            # Otherwise, scan for images with class name in the filename
            else:
                for img_path in self.data_dir.glob(f"**/*{class_name}*.*"):
                    if self._is_image_file(img_path):
                        self.images.append(img_path)
                        self.labels.append(self.class_map[class_name])
                        self.metadata.append({"class": class_name, "path": str(img_path), "synthetic": self.is_synthetic})
    
    def _is_image_file(self, path: Path) -> bool:
        """Check if a file is an image file"""
        return path.suffix.lower() in [".jpg", ".jpeg", ".png"] and not path.stem.endswith("_mask")
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset"""
        distribution = {class_name: 0 for class_name in self.classes}
        for idx, label in enumerate(self.labels):
            class_name = self.classes[label]
            distribution[class_name] += 1
        return distribution
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        img_path = self.images[idx]
        label = self.labels[idx]
        metadata = self.metadata[idx]
        
        try:
            # Load image
            img = Image.open(img_path).convert("RGB")
            
            # Apply transforms
            if self.transform:
                img = self.transform(img)
            
            return img, label, metadata
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            if self.transform:
                blank = torch.zeros((3, 224, 224))
            else:
                blank = Image.new("RGB", (224, 224), (0, 0, 0))
            return blank, label, metadata


class PizzaDiffusionTrainer:
    """
    Trainer class for pizza recognition model with diffusion-generated data integration.
    """
    
    def __init__(
        self,
        real_data_dir: str,
        synthetic_data_dir: str,
        output_dir: str,
        params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            real_data_dir: Directory with real data
            synthetic_data_dir: Directory with synthetic data
            output_dir: Directory to save outputs
            params: Training parameters
            device: Device to use for training
        """
        self.real_data_dir = Path(real_data_dir)
        self.synthetic_data_dir = Path(synthetic_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use provided params or defaults
        self.params = params or DEFAULT_PARAMS.copy()
        
        # Setup device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Set up storage for results and statistics
        self.results = {}
        self.history = []
        
        logger.info("Initialized Pizza Diffusion Trainer")
        logger.info(f"  Real data: {self.real_data_dir}")
        logger.info(f"  Synthetic data: {self.synthetic_data_dir}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Device: {self.device}")
    
    def load_datasets(self):
        """
        Load real and synthetic datasets.
        """
        logger.info("Loading datasets...")
        
        # Determine class mapping
        self.class_map = {stage: i for i, stage in enumerate(PIZZA_STAGES.keys())}
        
        # Load real dataset
        try:
            self.real_dataset = PizzaDataset(
                data_dir=self.real_data_dir,
                transform=self.train_transform,
                is_synthetic=False,
                class_map=self.class_map
            )
            logger.info(f"Loaded real dataset with {len(self.real_dataset)} images")
        except Exception as e:
            logger.error(f"Error loading real dataset: {e}")
            self.real_dataset = None
        
        # Load synthetic dataset
        try:
            self.synthetic_dataset = PizzaDataset(
                data_dir=self.synthetic_data_dir,
                transform=self.train_transform,
                is_synthetic=True,
                class_map=self.class_map
            )
            logger.info(f"Loaded synthetic dataset with {len(self.synthetic_dataset)} images")
        except Exception as e:
            logger.error(f"Error loading synthetic dataset: {e}")
            self.synthetic_dataset = None
    
    def prepare_mixed_dataset(self, synthetic_ratio: float = None):
        """
        Prepare a mixed dataset with both real and synthetic data.
        
        Args:
            synthetic_ratio: Ratio of synthetic data to use (0.0 to 1.0)
        """
        if synthetic_ratio is None:
            synthetic_ratio = self.params["synthetic_ratio"]
        
        if not hasattr(self, "real_dataset") or not hasattr(self, "synthetic_dataset"):
            self.load_datasets()
        
        if self.real_dataset is None:
            logger.error("Real dataset not available")
            return
        
        if self.synthetic_dataset is None:
            logger.warning("Synthetic dataset not available, using only real data")
            synthetic_ratio = 0.0
        
        # Calculate dataset sizes
        real_size = len(self.real_dataset)
        synthetic_size = len(self.synthetic_dataset) if self.synthetic_dataset else 0
        
        if synthetic_ratio > 0 and synthetic_size > 0:
            # Calculate how many synthetic samples to use
            target_synthetic_size = int(real_size * synthetic_ratio / (1 - synthetic_ratio))
            
            # Cap to available synthetic samples
            actual_synthetic_size = min(target_synthetic_size, synthetic_size)
            
            # If we don't have enough synthetic data, adjust the ratio
            actual_ratio = actual_synthetic_size / (real_size + actual_synthetic_size)
            if abs(actual_ratio - synthetic_ratio) > 0.05:
                logger.warning(f"Requested synthetic ratio {synthetic_ratio:.2f} adjusted to {actual_ratio:.2f} due to dataset size constraints")
            
            # Create mixed dataset
            if actual_synthetic_size > 0:
                # If we don't need all synthetic data, select a random subset
                if actual_synthetic_size < synthetic_size:
                    synthetic_indices = torch.randperm(synthetic_size)[:actual_synthetic_size].tolist()
                    synthetic_subset = Subset(self.synthetic_dataset, synthetic_indices)
                    self.mixed_dataset = ConcatDataset([self.real_dataset, synthetic_subset])
                else:
                    self.mixed_dataset = ConcatDataset([self.real_dataset, self.synthetic_dataset])
                
                logger.info(f"Created mixed dataset with {real_size} real and {actual_synthetic_size} synthetic images (ratio: {actual_ratio:.2f})")
            else:
                self.mixed_dataset = self.real_dataset
                logger.info(f"Using only real dataset with {real_size} images")
        else:
            # Use only real data
            self.mixed_dataset = self.real_dataset
            logger.info(f"Using only real dataset with {real_size} images")
        
        # Create train/val split
        dataset_size = len(self.mixed_dataset)
        val_size = int(dataset_size * 0.2)  # 20% for validation
        train_size = dataset_size - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            self.mixed_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
        )
        
        logger.info(f"Split dataset into {train_size} training and {val_size} validation samples")
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == "cuda" else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == "cuda" else False
        )
    
    def train_with_real_data_only(self):
        """
        Train a model using only real data for comparison.
        """
        logger.info("Training model with real data only...")
        
        # Set up output directory
        exp_dir = self.output_dir / "real_only"
        exp_dir.mkdir(exist_ok=True)
        
        # Save a copy of parameters
        params_copy = self.params.copy()
        params_copy["synthetic_ratio"] = 0.0
        params_copy["experiment_name"] = "real_only"
        with open(exp_dir / "params.json", "w") as f:
            json.dump(params_copy, f, indent=2)
        
        # Prepare dataset with 0% synthetic data
        self.prepare_mixed_dataset(synthetic_ratio=0.0)
        
        # Create and train model
        if PROJECT_MODULES_AVAILABLE:
            # Use project's existing model and training code
            model = PizzaModel(num_classes=len(self.class_map))
            result = train_model(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                num_epochs=self.params["epochs"],
                learning_rate=self.params["learning_rate"],
                weight_decay=self.params["weight_decay"],
                device=self.device,
                output_dir=exp_dir
            )
        else:
            # Use a simple training implementation
            result = self._simple_training_implementation(output_dir=exp_dir)
        
        # Store results
        self.results["real_only"] = result
        
        logger.info(f"Completed training with real data only. Best validation accuracy: {result['best_val_accuracy']:.4f}")
        return result
    
    def train_with_mixed_data(self, synthetic_ratio: float = None):
        """
        Train a model using mixed real and synthetic data.
        
        Args:
            synthetic_ratio: Ratio of synthetic data to use
        """
        if synthetic_ratio is None:
            synthetic_ratio = self.params["synthetic_ratio"]
        
        logger.info(f"Training model with mixed data (synthetic ratio: {synthetic_ratio:.2f})...")
        
        # Set up output directory
        exp_name = f"mixed_ratio_{synthetic_ratio:.2f}"
        exp_dir = self.output_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        # Save a copy of parameters
        params_copy = self.params.copy()
        params_copy["synthetic_ratio"] = synthetic_ratio
        params_copy["experiment_name"] = exp_name
        with open(exp_dir / "params.json", "w") as f:
            json.dump(params_copy, f, indent=2)
        
        # Prepare dataset with specified synthetic ratio
        self.prepare_mixed_dataset(synthetic_ratio=synthetic_ratio)
        
        # Create and train model
        if PROJECT_MODULES_AVAILABLE:
            # Use project's existing model and training code
            model = PizzaModel(num_classes=len(self.class_map))
            result = train_model(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                num_epochs=self.params["epochs"],
                learning_rate=self.params["learning_rate"],
                weight_decay=self.params["weight_decay"],
                device=self.device,
                output_dir=exp_dir
            )
        else:
            # Use a simple training implementation
            result = self._simple_training_implementation(output_dir=exp_dir)
        
        # Store results
        self.results[exp_name] = result
        
        logger.info(f"Completed training with mixed data. Best validation accuracy: {result['best_val_accuracy']:.4f}")
        return result
    
    def _simple_training_implementation(self, output_dir: Path) -> Dict[str, Any]:
        """
        Simple training implementation for when project modules are not available.
        
        Args:
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with training results
        """
        # Create a simple CNN model
        model = self._create_simple_model(num_classes=len(self.class_map))
        model.to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        
        if self.params["optimizer"].lower() == "adam":
            optimizer = optim.Adam(
                model.parameters(), 
                lr=self.params["learning_rate"],
                weight_decay=self.params["weight_decay"]
            )
        else:
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.params["learning_rate"],
                momentum=0.9,
                weight_decay=self.params["weight_decay"]
            )
        
        # Learning rate scheduler
        if self.params["scheduler"].lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.params["epochs"]
            )
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max', 
                factor=0.1, 
                patience=3
            )
        
        # Training loop
        best_val_acc = 0.0
        best_epoch = 0
        history = []
        
        for epoch in range(self.params["epochs"]):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, targets, _ in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward + optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets, _ in self.val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
            # Update scheduler
            if self.params["scheduler"].lower() == "cosine":
                scheduler.step()
            else:
                scheduler.step(val_acc)
            
            # Save checkpoint if this is the best model so far
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                
                # Save model
                torch.save(model.state_dict(), output_dir / "best_model.pth")
                
                # Reset early stopping counter
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{self.params['epochs']} - "
                       f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} - "
                       f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
            
            # Record history
            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"]
            })
            
            # Early stopping
            if no_improve_epochs >= self.params["early_stopping"]:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save history
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        # Plot and save learning curves
        self._plot_learning_curves(history, output_dir)
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(output_dir / "best_model.pth"))
        
        # Final evaluation on validation set
        model.eval()
        val_correct = 0
        val_total = 0
        
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, targets, _ in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        final_val_acc = val_correct / val_total
        
        # Calculate confusion matrix
        confusion = np.zeros((len(self.class_map), len(self.class_map)), dtype=np.int64)
        for t, p in zip(all_targets, all_predictions):
            confusion[t, p] += 1
        
        # Save confusion matrix
        np.save(output_dir / "confusion_matrix.npy", confusion)
        
        # Return results
        result = {
            "best_val_accuracy": best_val_acc,
            "best_epoch": best_epoch,
            "final_val_accuracy": final_val_acc,
            "confusion_matrix": confusion,
            "history": history
        }
        
        # Save results
        with open(output_dir / "results.json", "w") as f:
            json.dump({k: v for k, v in result.items() if k != "confusion_matrix"}, f, indent=2)
        
        return result
    
    def _create_simple_model(self, num_classes: int) -> nn.Module:
        """
        Create a simple CNN model when project modules are not available.
        
        Args:
            num_classes: Number of classes
            
        Returns:
            PyTorch model
        """
        # Simple model based on MobileNetV2
        from torchvision.models import mobilenet_v2
        
        model = mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        return model
    
    def _plot_learning_curves(self, history: List[Dict[str, float]], output_dir: Path):
        """
        Plot and save learning curves.
        
        Args:
            history: Training history
            output_dir: Directory to save plots
        """
        # Extract data
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        train_acc = [h["train_acc"] for h in history]
        val_acc = [h["val_acc"] for h in history]
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
        ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy')
        ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_dir / "learning_curves.png", dpi=300)
        plt.close()
    
    def find_optimal_synthetic_ratio(self, ratios: List[float] = None):
        """
        Find the optimal synthetic/real data ratio through grid search.
        
        Args:
            ratios: List of ratios to try
        """
        if ratios is None:
            ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        
        logger.info(f"Finding optimal synthetic ratio from {ratios}...")
        
        # Set up output directory
        exp_dir = self.output_dir / "ratio_search"
        exp_dir.mkdir(exist_ok=True)
        
        # Save search parameters
        with open(exp_dir / "search_params.json", "w") as f:
            json.dump({
                "ratios": ratios,
                "base_params": self.params
            }, f, indent=2)
        
        # Train models with different ratios
        ratio_results = {}
        
        for ratio in ratios:
            logger.info(f"Training with synthetic ratio {ratio:.2f}")
            result = self.train_with_mixed_data(synthetic_ratio=ratio)
            ratio_results[ratio] = result["best_val_accuracy"]
        
        # Find best ratio
        best_ratio = max(ratio_results.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Best synthetic ratio: {best_ratio:.2f} with validation accuracy {ratio_results[best_ratio]:.4f}")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(list(ratio_results.keys()), list(ratio_results.values()), 'o-')
        plt.axvline(x=best_ratio, color='r', linestyle='--', label=f'Best ratio: {best_ratio:.2f}')
        plt.xlabel('Synthetic Data Ratio')
        plt.ylabel('Validation Accuracy')
        plt.title('Effect of Synthetic Data Ratio on Model Performance')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(exp_dir / "ratio_search_results.png", dpi=300)
        plt.close()
        
        # Save results
        with open(exp_dir / "ratio_search_results.json", "w") as f:
            json.dump({
                "ratio_results": ratio_results,
                "best_ratio": best_ratio,
                "best_accuracy": ratio_results[best_ratio]
            }, f, indent=2)
        
        return best_ratio, ratio_results
    
    def compare_and_report(self):
        """
        Compare different training approaches and generate a report.
        """
        logger.info("Generating comparison report...")
        
        if not self.results:
            logger.warning("No results available for comparison")
            return
        
        # Create report directory
        report_dir = self.output_dir / "report"
        report_dir.mkdir(exist_ok=True)
        
        # Extract results for comparison
        real_only_result = self.results.get("real_only", {})
        
        # Find best mixed result
        mixed_results = {k: v for k, v in self.results.items() if k.startswith("mixed")}
        best_mixed_key = max(mixed_results.items(), key=lambda x: x[1].get("best_val_accuracy", 0))[0] if mixed_results else None
        best_mixed_result = mixed_results.get(best_mixed_key, {}) if best_mixed_key else {}
        
        # Compare results
        if real_only_result and best_mixed_result:
            real_acc = real_only_result.get("best_val_accuracy", 0)
            mixed_acc = best_mixed_result.get("best_val_accuracy", 0)
            
            improvement = (mixed_acc - real_acc) / real_acc * 100 if real_acc > 0 else 0
            
            logger.info(f"Comparison:")
            logger.info(f"  Real only accuracy: {real_acc:.4f}")
            logger.info(f"  Best mixed accuracy ({best_mixed_key}): {mixed_acc:.4f}")
            logger.info(f"  Improvement: {improvement:.2f}%")
            
            # Create comparison plot
            plt.figure(figsize=(10, 6))
            plt.bar(["Real data only", "With synthetic data"], [real_acc, mixed_acc])
            plt.ylabel("Validation Accuracy")
            plt.title("Performance Comparison: Real vs. Mixed Training Data")
            for i, v in enumerate([real_acc, mixed_acc]):
                plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
            plt.tight_layout()
            plt.savefig(report_dir / "accuracy_comparison.png", dpi=300)
            plt.close()
            
            # Generate confusion matrix comparison if available
            if "confusion_matrix" in real_only_result and "confusion_matrix" in best_mixed_result:
                real_cm = real_only_result["confusion_matrix"]
                mixed_cm = best_mixed_result["confusion_matrix"]
                
                # Normalize confusion matrices
                real_cm_norm = real_cm / real_cm.sum(axis=1, keepdims=True)
                mixed_cm_norm = mixed_cm / mixed_cm.sum(axis=1, keepdims=True)
                
                # Create figure with 2 subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Plot real data confusion matrix
                im1 = ax1.imshow(real_cm_norm, cmap="Blues")
                ax1.set_title("Real Data Only")
                ax1.set_xlabel("Predicted")
                ax1.set_ylabel("True")
                fig.colorbar(im1, ax=ax1)
                
                # Plot mixed data confusion matrix
                im2 = ax2.imshow(mixed_cm_norm, cmap="Blues")
                ax2.set_title("With Synthetic Data")
                ax2.set_xlabel("Predicted")
                ax2.set_ylabel("True")
                fig.colorbar(im2, ax=ax2)
                
                # Set labels
                classes = list(self.class_map.keys())
                ax1.set_xticks(range(len(classes)))
                ax1.set_yticks(range(len(classes)))
                ax1.set_xticklabels(classes, rotation=45, ha="right")
                ax1.set_yticklabels(classes)
                
                ax2.set_xticks(range(len(classes)))
                ax2.set_yticks(range(len(classes)))
                ax2.set_xticklabels(classes, rotation=45, ha="right")
                ax2.set_yticklabels(classes)
                
                plt.tight_layout()
                plt.savefig(report_dir / "confusion_matrix_comparison.png", dpi=300)
                plt.close()
            
            # Generate HTML report
            with open(report_dir / "report.html", "w") as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Pizza Recognition - Diffusion Model Impact Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
                        h1, h2 {{ color: #333; }}
                        .container {{ max-width: 1200px; margin: 0 auto; }}
                        .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                        .comparison {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
                        .comparison > div {{ flex: 1; padding: 15px; }}
                        table {{ width: 100%; border-collapse: collapse; }}
                        table, th, td {{ border: 1px solid #ddd; }}
                        th, td {{ padding: 12px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        img {{ max-width: 100%; height: auto; }}
                        .highlight {{ font-weight: bold; color: {('#4CAF50' if improvement > 0 else '#FF5252')}; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Pizza Recognition - Diffusion Model Impact Report</h1>
                        
                        <div class="summary">
                            <h2>Summary</h2>
                            <p>This report compares the performance of pizza recognition models trained with and without diffusion-generated synthetic data.</p>
                            <p>Best performance was achieved with <span class="highlight">{best_mixed_key}</span>, showing a 
                            <span class="highlight">{improvement:.2f}%</span> improvement over using only real data.</p>
                        </div>
                        
                        <h2>Accuracy Comparison</h2>
                        <img src="accuracy_comparison.png" alt="Accuracy Comparison Chart">
                        
                        <h2>Confusion Matrix Comparison</h2>
                        <img src="confusion_matrix_comparison.png" alt="Confusion Matrix Comparison">
                        
                        <h2>Detailed Results</h2>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Real Data Only</th>
                                <th>With Synthetic Data</th>
                                <th>Difference</th>
                            </tr>
                            <tr>
                                <td>Best Validation Accuracy</td>
                                <td>{real_acc:.4f}</td>
                                <td>{mixed_acc:.4f}</td>
                                <td class="highlight">{mixed_acc - real_acc:.4f} ({improvement:.2f}%)</td>
                            </tr>
                            <tr>
                                <td>Best Epoch</td>
                                <td>{real_only_result.get('best_epoch', 'N/A')}</td>
                                <td>{best_mixed_result.get('best_epoch', 'N/A')}</td>
                                <td>-</td>
                            </tr>
                        </table>
                        
                        <h2>Conclusion</h2>
                        <p>
                            The integration of diffusion-generated synthetic data has {
                            'significantly improved' if improvement > 5 else 
                            'improved' if improvement > 0 else 
                            'not improved'} the performance of the pizza recognition model.
                            {'This demonstrates the value of using high-quality synthetic data for training.' if improvement > 0 else
                             'Further optimization of the synthetic data generation may be needed to see benefits.'}
                        </p>
                        
                        <h2>Next Steps</h2>
                        <ul>
                            <li>Fine-tune the synthetic data generation process based on these findings</li>
                            <li>Test with different diffusion models and prompt strategies</li>
                            <li>Further optimize the synthetic/real data ratio</li>
                            <li>Evaluate model performance on additional test datasets</li>
                        </ul>
                    </div>
                </body>
                </html>
                """)
            
            logger.info(f"Comparison report generated in {report_dir}")
        else:
            logger.warning("Insufficient results for comparison")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Pizza Diffusion Model Integration for Training")
    parser.add_argument("--real_dir", type=str, default="data/processed",
                        help="Directory with real training data")
    parser.add_argument("--synthetic_dir", type=str, default="data/synthetic",
                        help="Directory with synthetic training data")
    parser.add_argument("--output_dir", type=str, default="output/diffusion_training",
                        help="Directory to save outputs")
    parser.add_argument("--synthetic_ratio", type=float, default=0.5,
                        help="Ratio of synthetic to real data (0.0 to 1.0)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for training")
    parser.add_argument("--find_optimal_ratio", action="store_true",
                        help="Find optimal synthetic/real data ratio")
    parser.add_argument("--real_only", action="store_true",
                        help="Train with real data only")
    parser.add_argument("--mixed_only", action="store_true",
                        help="Train with mixed data only")
    parser.add_argument("--compare", action="store_true",
                        help="Train with both real-only and mixed data and compare")
    
    args = parser.parse_args()
    
    # Set up parameters
    params = DEFAULT_PARAMS.copy()
    params.update({
        "synthetic_ratio": args.synthetic_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate
    })
    
    # Create trainer
    trainer = PizzaDiffusionTrainer(
        real_data_dir=args.real_dir,
        synthetic_data_dir=args.synthetic_dir,
        output_dir=args.output_dir,
        params=params
    )
    
    # Load datasets
    trainer.load_datasets()
    
    # Run requested operations
    if args.find_optimal_ratio:
        # Find optimal ratio
        best_ratio, ratio_results = trainer.find_optimal_synthetic_ratio()
        print(f"Best synthetic ratio: {best_ratio:.2f} with accuracy {ratio_results[best_ratio]:.4f}")
        
    elif args.real_only:
        # Train with real data only
        result = trainer.train_with_real_data_only()
        print(f"Training complete. Best validation accuracy: {result['best_val_accuracy']:.4f}")
        
    elif args.mixed_only:
        # Train with mixed data
        result = trainer.train_with_mixed_data()
        print(f"Training complete. Best validation accuracy: {result['best_val_accuracy']:.4f}")
        
    elif args.compare:
        # Train with both and compare
        trainer.train_with_real_data_only()
        trainer.train_with_mixed_data()
        trainer.compare_and_report()
        
    else:
        # Default to comparison
        print("No specific action requested, performing full comparison")
        trainer.train_with_real_data_only()
        trainer.train_with_mixed_data()
        trainer.compare_and_report()


if __name__ == "__main__":
    main()
