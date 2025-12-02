#!/usr/bin/env python3
"""
Metrics Module for Pizza Detection System

This module provides metrics calculation and evaluation tools for the
Pizza Detection System. It includes performance metrics, resource usage
tracking, and model comparison utilities.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from . import constants
from .types import InferenceResult


def _get_class_names() -> Tuple[str, ...]:
    """Return the authoritative class-name tuple (respects monkeypatching)."""

    names = getattr(constants, "CLASS_NAMES", ())
    if isinstance(names, (list, tuple)) and names:
        return tuple(names)
    raise ValueError("CLASS_NAMES is not defined or empty in src.constants")

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
        self.class_names = list(_get_class_names())
    
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

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot_metrics(self, output_dir: str) -> None:
        """Create visualizations of metrics"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Accuracy plot
        plt.figure(figsize=(10, 6))
        plt.bar(
            self.class_names,
            [self.metrics["class_accuracies"].get(c, 0) for c in self.class_names],
        )
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


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Calculate accuracy-focused metrics for classification predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional")

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Input arrays cannot be empty")

    if y_true.size != y_pred.size:
        raise ValueError(f"Input arrays must have same length: {y_true.size} != {y_pred.size}")

    if np.min(y_true) < 0 or np.min(y_pred) < 0:
        raise ValueError("Labels must be non-negative")

    effective_class_names: Sequence[str]
    if class_names is None:
        effective_class_names = _get_class_names()
    else:
        effective_class_names = tuple(class_names)

    if len(effective_class_names) == 0:
        raise ValueError("At least one class name is required")

    num_classes = len(effective_class_names)

    if np.max(y_true) >= num_classes or np.max(y_pred) >= num_classes:
        raise ValueError("Labels exceed number of provided class names")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(num_classes), average='macro', zero_division=0
    )

    conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    accuracy = float((y_true == y_pred).sum() / y_true.size)

    return {
        'accuracy': accuracy,
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': conf_matrix.astype(int),
        'class_names': list(effective_class_names[:num_classes])
    }


def _resolve_output_dir(custom_dir: Optional[Union[str, Path]]) -> Path:
    if custom_dir is not None:
        return Path(custom_dir)
    # Always pull the latest value from ``src.constants`` so monkeypatches in tests
    # are respected.
    return Path(getattr(constants, "OUTPUT_DIR")) / "evaluation"


def save_metrics(metrics: Dict[str, Any], model_name: str, output_dir: Optional[Union[str, Path]] = None) -> Path:
    """Persist metrics to disk as JSON and return the corresponding path."""

    base_dir = _resolve_output_dir(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    save_path = base_dir / f"{model_name}_metrics.json"

    serializable_metrics = dict(metrics)
    conf_matrix = serializable_metrics.get('confusion_matrix')
    if isinstance(conf_matrix, np.ndarray):
        serializable_metrics['confusion_matrix'] = conf_matrix.tolist()

    with open(save_path, 'w', encoding='utf-8') as fp:
        json.dump(serializable_metrics, fp, indent=2)

    return save_path


def load_metrics(model_name: str, output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load metrics JSON generated by ``save_metrics`` and return it as dict."""

    base_dir = _resolve_output_dir(output_dir)
    load_path = base_dir / f"{model_name}_metrics.json"

    if not load_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {load_path}")

    with open(load_path, 'r', encoding='utf-8') as fp:
        metrics = json.load(fp)

    if 'confusion_matrix' in metrics:
        metrics['confusion_matrix'] = np.array(metrics['confusion_matrix'])

    return metrics


def format_inference_result(logits: torch.Tensor) -> InferenceResult:
    """Convert raw logits to structured inference output."""

    if logits.ndim != 1:
        logits = logits.squeeze()

    probabilities = torch.softmax(logits, dim=0).detach().cpu().numpy()
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction])

    class_names = list(_get_class_names())
    if len(class_names) < len(probabilities):
        class_names.extend([f"class_{idx}" for idx in range(len(class_names), len(probabilities))])
    class_names = class_names[:len(probabilities)]

    prob_map = {
        class_name: float(prob)
        for class_name, prob in zip(class_names, probabilities)
    }

    return InferenceResult(
        prediction=prediction,
        confidence=confidence,
        class_name=class_names[prediction],
        probabilities=prob_map
    )


def get_error_analysis(metrics: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Generate error statistics per class from a confusion matrix."""

    conf_matrix = metrics.get('confusion_matrix')
    if conf_matrix is None:
        raise ValueError("Metrics dictionary must include a 'confusion_matrix'")

    if isinstance(conf_matrix, list):
        conf_matrix = np.array(conf_matrix)

    error_analysis: Dict[str, Dict[str, float]] = {}

    class_names = metrics.get('class_names')
    if class_names is None:
        class_names = _get_class_names()[:conf_matrix.shape[0]]
    else:
        class_names = list(class_names)
        if len(class_names) < conf_matrix.shape[0]:
            class_names.extend(
                [f"class_{idx}" for idx in range(len(class_names), conf_matrix.shape[0])]
            )

    for idx, class_name in enumerate(class_names[:conf_matrix.shape[0]]):
        tp = int(conf_matrix[idx, idx])
        fp = int(conf_matrix[:, idx].sum() - tp)
        fn = int(conf_matrix[idx, :].sum() - tp)
        total = tp + fp + fn

        error_rate = ((fp + fn) / total) if total > 0 else 0.0

        error_analysis[class_name] = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'error_rate': error_rate
        }

    return error_analysis

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
