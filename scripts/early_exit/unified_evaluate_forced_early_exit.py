#!/usr/bin/env python3
"""
Unified Early Exit Evaluation Script
Works with both the original and improved early exit models
by using the model adapter for compatibility.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import json
from collections import Counter
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.pizza_detector import (
    create_optimized_dataloaders, RP2040Config
)
from scripts.early_exit.model_adapter import load_model_with_compatibility

def evaluate_baseline(model, loader, device):
    """Evaluate the model with normal inference (no early exit)"""
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    
    # Store per-class accuracies
    class_correct = Counter()
    class_total = Counter()
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass with early exit disabled
            outputs, _ = model(inputs, use_early_exit=False)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Update statistics
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            # Update per-class statistics
            for i, label in enumerate(labels):
                label_idx = label.item()
                class_total[label_idx] += 1
                if preds[i] == label:
                    class_correct[label_idx] += 1
    
    end_time = time.time()
    
    # Calculate metrics
    accuracy = 100 * correct / total
    inference_time = (end_time - start_time) / total
    
    # Calculate per-class accuracies
    per_class_accuracy = {}
    for class_idx in class_total:
        if class_total[class_idx] > 0:
            per_class_accuracy[class_idx] = 100 * class_correct[class_idx] / class_total[class_idx]
        else:
            per_class_accuracy[class_idx] = 0
    
    return accuracy, inference_time, per_class_accuracy

def evaluate_with_forced_early_exit(model, loader, device, forced_exit_rate=0.0):
    """
    Evaluate the model with a specified forced early exit rate
    
    Args:
        model: The model to evaluate
        loader: DataLoader for evaluation
        device: Device to use
        forced_exit_rate: Percentage of samples to force through early exit path (0.0-1.0)
        
    Returns:
        tuple: (accuracy, avg_time, early_exit_count, total_samples, per_class_accuracy)
    """
    model.eval()
    correct = 0
    total = 0
    early_exit_count = 0
    
    # Store per-class accuracies
    class_correct = Counter()
    class_total = Counter()
    
    # For timing
    total_time = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = labels.size(0)
            
            # Determine how many samples should use early exit
            num_early_exit = int(batch_size * forced_exit_rate)
            
            # Forced early exit for selected samples
            if num_early_exit > 0:
                # Process samples that will use early exit
                early_inputs = inputs[:num_early_exit]
                early_labels = labels[:num_early_exit]
                
                # Time measurement for early exit path
                start_time = time.time()
                
                # Check if model has forced_exit parameter
                if 'forced_exit' in model.forward.__code__.co_varnames:
                    early_outputs, _ = model(early_inputs, use_early_exit=True, forced_exit=True)
                else:
                    # For original model, manually extract after block2
                    # Process feature extraction up to block2
                    x = model.block1(early_inputs)
                    x = model.block2(x)
                    
                    # Apply early exit classifier
                    early_features = model.early_exit_pooling(x)
                    early_outputs = model.early_exit_classifier(early_features)
                
                end_time = time.time()
                total_time += (end_time - start_time)
                
                # Get predictions
                _, early_preds = torch.max(early_outputs, 1)
                
                # Update early exit statistics
                early_exit_count += num_early_exit
                
                # Update accuracy statistics
                correct += (early_preds == early_labels).sum().item()
                
                # Update per-class statistics for early exit samples
                for i, label in enumerate(early_labels):
                    label_idx = label.item()
                    class_total[label_idx] += 1
                    if early_preds[i] == label:
                        class_correct[label_idx] += 1
            
            # Regular path for remaining samples
            if num_early_exit < batch_size:
                # Process samples that will use full network
                late_inputs = inputs[num_early_exit:]
                late_labels = labels[num_early_exit:]
                
                # Time measurement for full path
                start_time = time.time()
                late_outputs, _ = model(late_inputs, use_early_exit=False)
                end_time = time.time()
                total_time += (end_time - start_time)
                
                # Get predictions
                _, late_preds = torch.max(late_outputs, 1)
                
                # Update accuracy statistics
                correct += (late_preds == late_labels).sum().item()
                
                # Update per-class statistics for full network samples
                for i, label in enumerate(late_labels):
                    label_idx = label.item()
                    class_total[label_idx] += 1
                    if late_preds[i] == label:
                        class_correct[label_idx] += 1
            
            # Update total count
            total += batch_size
    
    # Calculate metrics
    accuracy = 100 * correct / total if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0
    
    # Calculate per-class accuracies
    per_class_accuracy = {}
    for class_idx in class_total:
        if class_total[class_idx] > 0:
            per_class_accuracy[class_idx] = 100 * class_correct[class_idx] / class_total[class_idx]
        else:
            per_class_accuracy[class_idx] = 0
    
    return accuracy, avg_time, early_exit_count, total, per_class_accuracy

def run_forced_evaluation(model_path, data_dir="data/augmented", batch_size=16, device="cuda", output_dir=None):
    """
    Run evaluation on the early exit model, forcing different proportions of samples
    through the early exit path to demonstrate potential energy savings.
    """
    # Configuration and data loading
    logger.info(f"Loading configuration with data directory: {data_dir}")
    config = RP2040Config(data_dir=data_dir)
    config.BATCH_SIZE = batch_size
    
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
        logger.info("CUDA not available, using CPU")
    
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    # Create output directory if needed
    if output_dir is None:
        output_dir = os.path.join(project_root, "output", "model_optimization", "early_exit", "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    _, val_loader, class_names, _ = create_optimized_dataloaders(config)
    logger.info(f"Created loaders with {len(class_names)} classes: {class_names}")
    
    # Load the model using the adapter
    logger.info(f"Loading model from {model_path}")
    model = load_model_with_compatibility(model_path, num_classes=len(class_names), device=device)
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Dictionary to store results for different early exit rates
    results = {}
    
    # Baseline evaluation (normal inference, no early exit)
    logger.info("Running baseline evaluation...")
    baseline_acc, baseline_time, baseline_per_class = evaluate_baseline(model, val_loader, device)
    
    logger.info(f"Baseline accuracy: {baseline_acc:.2f}%")
    logger.info(f"Baseline inference time: {baseline_time:.6f} seconds")
    logger.info(f"Per-class accuracies: {baseline_per_class}")
    
    # Test different forced early exit rates
    exit_rates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    for exit_rate in exit_rates:
        logger.info(f"\nTesting with forced early exit rate: {exit_rate:.1f}")
        
        # Run evaluation with forced early exit
        accuracy, avg_time, early_exit_count, total_samples, per_class_accuracy = evaluate_with_forced_early_exit(
            model, val_loader, device, forced_exit_rate=exit_rate
        )
        
        time_saved_percent = 100 * (1 - avg_time / baseline_time)
        
        actual_exit_rate = 100 * early_exit_count / total_samples
        
        results[exit_rate] = {
            'accuracy': accuracy,
            'time_per_sample': avg_time,
            'time_saved_percent': time_saved_percent,
            'actual_exit_rate': actual_exit_rate,
            'per_class_accuracy': per_class_accuracy
        }
        
        logger.info(f"Accuracy: {accuracy:.2f}%")
        logger.info(f"Average time per sample: {avg_time:.6f} seconds")
        logger.info(f"Time saved: {time_saved_percent:.2f}%")
        logger.info(f"Per-class accuracies: {per_class_accuracy}")
    
    # Save results
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    results_file = os.path.join(output_dir, f"{model_name}_forced_exit_results.json")
    
    logger.info(f"Saving results to {results_file}")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create visualization
    create_early_exit_visualizations(results, model_name, output_dir)
    
    return results

def create_early_exit_visualizations(results, model_name, output_dir):
    """Create visualizations of early exit performance"""
    exit_rates = list(results.keys())
    accuracies = [results[rate]['accuracy'] for rate in exit_rates]
    time_savings = [results[rate]['time_saved_percent'] for rate in exit_rates]
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Accuracy line (primary y-axis)
    color = 'tab:blue'
    ax1.set_xlabel('Forced Early Exit Rate')
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(exit_rates, accuracies, 'o-', color=color, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Time saved line (secondary y-axis)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Time Saved (%)', color=color)
    ax2.plot(exit_rates, time_savings, 's-', color=color, label='Time Saved')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Title and styling
    plt.title(f'Early Exit Performance - {model_name}')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, f"{model_name}_early_exit_performance.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved performance visualization to {plot_file}")
    
    # Create a second plot for per-class accuracy
    if 0.0 in results and 1.0 in results:
        # Get class indices
        class_indices = results[0.0]['per_class_accuracy'].keys()
        
        # Extract per-class accuracies for full model (exit_rate=0.0) and early exit only (exit_rate=1.0)
        full_accs = [results[0.0]['per_class_accuracy'].get(str(idx), 0) for idx in class_indices]
        early_accs = [results[1.0]['per_class_accuracy'].get(str(idx), 0) for idx in class_indices]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(class_indices))
        width = 0.35
        
        ax.bar(x - width/2, full_accs, width, label='Full Model')
        ax.bar(x + width/2, early_accs, width, label='Early Exit Only')
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_xlabel('Class Index')
        ax.set_title(f'Per-Class Accuracy Comparison - {model_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(class_indices)
        ax.legend()
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(output_dir, f"{model_name}_per_class_accuracy.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved per-class accuracy visualization to {plot_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate early exit model with forced exit rates")
    
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the model weights file")
    parser.add_argument("--data-dir", type=str, default="data/augmented",
                       help="Directory containing the dataset")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for evaluation (cuda or cpu)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save results and visualizations")
    
    args = parser.parse_args()
    run_forced_evaluation(**vars(args))
