#!/usr/bin/env python3
"""
Metrics Module for Pizza Detection System

This module provides metrics calculation and evaluation tools for the
Pizza Detection System. It includes performance metrics, resource usage
tracking, and model comparison utilities.
"""

import os
import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt

class ModelMetrics:
    """
    Calculates and tracks various metrics for model evaluation including:
    - Accuracy, precision, recall, F1 score
    - Inference time
    - Memory usage
    - Model size
    """
    
    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize the metrics tracker
        
        Args:
            model_name: Name of the model being evaluated
            device: The device used for evaluation ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "inference_time_ms": 0.0,
            "memory_usage_mb": 0.0,
            "model_size_kb": 0.0,
            "class_accuracies": {}
        }
        self.confusion_matrix = None
        self.class_names = []
    
    def calculate_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate prediction accuracy"""
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        return correct / total if total > 0 else 0.0
    
    def measure_inference_time(self, model: torch.nn.Module, input_tensor: torch.Tensor, 
                              num_runs: int = 10) -> float:
        """Measure average inference time over multiple runs"""
        model.eval()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_tensor)
        
        # Timed runs
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) * 1000 / num_runs
        return avg_time_ms
    
    def calculate_model_size(self, model: torch.nn.Module) -> float:
        """Calculate model size in KB"""
        model_size_bytes = 0
        for param in model.parameters():
            model_size_bytes += param.nelement() * param.element_size()
        return model_size_bytes / 1024  # Convert to KB
    
    def update_metrics(self, new_metrics: Dict[str, Any]) -> None:
        """Update stored metrics with new values"""
        for key, value in new_metrics.items():
            if key in self.metrics:
                self.metrics[key] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all calculated metrics"""
        return self.metrics
    
    def save_metrics(self, output_dir: str) -> None:
        """Save metrics to a file"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{self.model_name}_metrics.json")
        
        import json
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot_metrics(self, output_dir: str) -> None:
        """Create visualizations of metrics"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Accuracy plot
        plt.figure(figsize=(10, 6))
        plt.bar(self.class_names, 
                [self.metrics["class_accuracies"].get(c, 0) for c in self.class_names])
        plt.title(f"{self.model_name} - Class Accuracies")
        plt.xlabel("Class")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.savefig(os.path.join(output_dir, f"{self.model_name}_class_accuracies.png"))
        plt.close()
        
        # If confusion matrix is available
        if self.confusion_matrix is not None:
            plt.figure(figsize=(10, 8))
            plt.imshow(self.confusion_matrix, cmap='Blues')
            plt.title(f"{self.model_name} - Confusion Matrix")
            plt.colorbar()
            
            # Add labels and ticks
            tick_marks = np.arange(len(self.class_names))
            plt.xticks(tick_marks, self.class_names, rotation=45)
            plt.yticks(tick_marks, self.class_names)
            
            # Add text annotations
            thresh = self.confusion_matrix.max() / 2
            for i in range(self.confusion_matrix.shape[0]):
                for j in range(self.confusion_matrix.shape[1]):
                    plt.text(j, i, format(self.confusion_matrix[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if self.confusion_matrix[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig(os.path.join(output_dir, f"{self.model_name}_confusion_matrix.png"))
            plt.close()
    
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names=None) -> dict:
    """
    Calculate performance metrics for model evaluation.
    
    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        class_names: Optional list of class names
        
    Returns:
        Dictionary with metrics including accuracy, precision, recall, f1_score, and confusion_matrix
    """
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    
    # Error handling for empty arrays
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    # Error handling for mismatched array lengths
    if len(y_true) != len(y_pred):
        raise ValueError(f"Input arrays must have same length: {len(y_true)} != {len(y_pred)}")
    
    # Get number of classes
    num_classes = max(max(y_true), max(y_pred)) + 1 if len(y_true) > 0 else 0
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    accuracy = (y_true == y_pred).mean()
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': conf_matrix
    }

def visualize_confusion_matrix(confusion_matrix: np.ndarray, class_names=None, output_path=None, normalize=True):
    """
    Visualize confusion matrix as a heatmap.
    
    Args:
        confusion_matrix: The confusion matrix to visualize
        class_names: Optional list of class names
        output_path: Path to save the visualization
        normalize: Whether to normalize the confusion matrix
        
    Returns:
        None (displays or saves the visualization)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Normalize if requested
    if normalize:
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        vmin, vmax = 0, 1
    else:
        cm = confusion_matrix
        title = 'Confusion Matrix'
        vmin, vmax = None, None
    
    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    
    # Set axis labels
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
    else:
        tick_marks = np.arange(cm.shape[0])
        plt.xticks(tick_marks)
        plt.yticks(tick_marks)
    
    # Add labels
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        if normalize:
            plt.text(j, i, f"{cm[i, j]:.2f}",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, f"{cm[i, j]}",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
