#!/usr/bin/env python3
"""
Improved Early Exit Evaluation Script
Evaluates the model with different forced early exit rates 
and provides detailed analysis of energy-accuracy tradeoffs.
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
from scripts.early_exit.improved_early_exit import ImprovedMicroPizzaNetWithEarlyExit

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
    inference_times = []
    
    # Store per-class accuracies
    class_correct = Counter()
    class_total = Counter()
    
    # Process by batch to maintain proper statistical distribution
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)
            
            # Calculate how many samples to force through early exit
            num_early_exit = int(batch_size * forced_exit_rate)
            
            # Indices to use early exit
            early_exit_indices = torch.randperm(batch_size)[:num_early_exit]
            
            # Process each sample in the batch
            for i in range(batch_size):
                sample_input = inputs[i:i+1]  # Get a single sample
                sample_label = labels[i:i+1]  # Get corresponding label
                
                # Decide whether to force early exit
                use_early_exit = i in early_exit_indices
                
                # Timing
                start_time = time.time()
                
                # Forward pass
                if use_early_exit:
                    outputs, _ = model(sample_input, use_early_exit=True, forced_exit=True)
                    early_exit_count += 1
                else:
                    outputs, is_early_exit = model(sample_input, use_early_exit=True)
                    if is_early_exit:
                        early_exit_count += 1
                
                end_time = time.time()
                inference_times.append(end_time - start_time)
                
                # Get prediction
                _, preds = torch.max(outputs, 1)
                
                # Update statistics
                total += 1
                correct += (preds == sample_label).sum().item()
                
                # Update per-class statistics
                label_idx = sample_label.item()
                class_total[label_idx] += 1
                if preds[0] == sample_label[0]:
                    class_correct[label_idx] += 1
    
    # Calculate metrics
    accuracy = 100 * correct / total if total > 0 else 0
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
    
    # Calculate per-class accuracies
    per_class_accuracy = {}
    for class_idx in class_total:
        if class_total[class_idx] > 0:
            per_class_accuracy[class_idx] = 100 * class_correct[class_idx] / class_total[class_idx]
        else:
            per_class_accuracy[class_idx] = 0
    
    return accuracy, avg_time, early_exit_count, total, per_class_accuracy

def run_forced_evaluation(model_path, data_dir="data/augmented", batch_size=16, device="cuda"):
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
    else:
        device = torch.device(device)
        logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    _, val_loader, class_names, _ = create_optimized_dataloaders(config)
    logger.info(f"Created loaders with {len(class_names)} classes: {class_names}")
    
    # Load the model
    logger.info(f"Loading model from {model_path}")
    model = ImprovedMicroPizzaNetWithEarlyExit(num_classes=len(class_names), confidence_threshold=0.1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(model_path), "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store results for different early exit rates
    results = {}
    
    # Baseline evaluation (normal inference, no early exit)
    logger.info("Running baseline evaluation...")
    baseline_acc, baseline_time, baseline_per_class_acc = evaluate_baseline(model, val_loader, device)
    
    logger.info(f"Baseline accuracy: {baseline_acc:.2f}%")
    logger.info(f"Baseline inference time: {baseline_time:.6f} seconds")
    logger.info("Baseline per-class accuracy:")
    for class_idx, acc in baseline_per_class_acc.items():
        logger.info(f"  Class {class_names[class_idx]}: {acc:.2f}%")
    
    # Test different forced early exit rates
    exit_rates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    for exit_rate in exit_rates:
        logger.info(f"\nTesting with forced early exit rate: {exit_rate:.1f}")
        
        # Run evaluation with forced early exit
        accuracy, avg_time, early_exit_count, total_samples, per_class_acc = evaluate_with_forced_early_exit(
            model, val_loader, device, forced_exit_rate=exit_rate
        )
        
        time_saved_percent = 100 * (1 - avg_time / baseline_time)
        
        actual_exit_rate = 100 * early_exit_count / total_samples
        
        results[exit_rate] = {
            'accuracy': accuracy,
            'time_per_sample': avg_time,
            'time_saved_percent': time_saved_percent,
            'actual_exit_rate': actual_exit_rate,
            'per_class_accuracy': {class_names[idx]: acc for idx, acc in per_class_acc.items()}
        }
        
        logger.info(f"Accuracy: {accuracy:.2f}%")
        logger.info(f"Average time per sample: {avg_time:.6f} seconds")
        logger.info(f"Time saved: {time_saved_percent:.2f}%")
        logger.info(f"Actual early exit rate: {actual_exit_rate:.2f}%")
        logger.info("Per-class accuracy:")
        for class_idx, acc in per_class_acc.items():
            logger.info(f"  Class {class_names[class_idx]}: {acc:.2f}%")
    
    # Store overall evaluation results
    evaluation_results = {
        'baseline_accuracy': baseline_acc,
        'baseline_time': baseline_time,
        'baseline_per_class_accuracy': {class_names[idx]: acc for idx, acc in baseline_per_class_acc.items()},
        'forced_exit_results': results
    }
    
    # Save results to JSON
    results_path = os.path.join(output_dir, "forced_exit_evaluation.json")
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_path}")
    
    # Create visualizations
    # 1. Accuracy vs. Time Saved
    create_accuracy_time_visualization(exit_rates, results, baseline_acc, output_dir)
    
    # 2. Per-Class Accuracy
    create_per_class_visualization(exit_rates, results, class_names, baseline_per_class_acc, output_dir)

    # 3. Generate metrics table
    create_metrics_table(exit_rates, results, baseline_acc, baseline_time, output_dir)
    
    return evaluation_results

def create_accuracy_time_visualization(exit_rates, results, baseline_acc, output_dir):
    """Create visualization of accuracy vs time savings"""
    accuracies = [results[rate]['accuracy'] for rate in exit_rates]
    time_savings = [results[rate]['time_saved_percent'] for rate in exit_rates]
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # First axis: Accuracy
    color = 'tab:blue'
    ax1.set_xlabel('Forced Early Exit Rate')
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(exit_rates, accuracies, 'o-', color=color, label='Accuracy')
    ax1.axhline(y=baseline_acc, color=color, linestyle='--', alpha=0.7, label='Baseline Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Second axis: Time/Energy Savings
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Time Saved (%)', color=color)
    ax2.plot(exit_rates, time_savings, 's-', color=color, label='Time Saved')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add title and legend
    plt.title('Early Exit Performance vs. Energy Savings')
    
    # Create a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "improved_early_exit_energy_savings.png"))
    plt.close()

def create_per_class_visualization(exit_rates, results, class_names, baseline_per_class_acc, output_dir):
    """Create visualization of per-class accuracy at different exit rates"""
    plt.figure(figsize=(12, 8))
    
    # Plot one line for each class
    for class_idx, class_name in enumerate(class_names):
        # Get accuracy for this class at each exit rate
        class_accuracies = []
        for rate in exit_rates:
            if class_name in results[rate]['per_class_accuracy']:
                class_accuracies.append(results[rate]['per_class_accuracy'][class_name])
            else:
                class_accuracies.append(0)
        
        # Plot this class line
        plt.plot(exit_rates, class_accuracies, 'o-', label=f'{class_name}')
    
    # Add baseline horizontal lines
    for class_name, acc in baseline_per_class_acc.items():
        plt.axhline(y=acc, linestyle='--', alpha=0.3, color='gray')
    
    plt.title('Per-Class Accuracy at Different Early Exit Rates')
    plt.xlabel('Forced Early Exit Rate')
    plt.ylabel('Accuracy (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "improved_early_exit_per_class.png"))
    plt.close()

def create_metrics_table(exit_rates, results, baseline_acc, baseline_time, output_dir):
    """Create a table visualization of metrics"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.axis('tight')
    
    # Table data
    table_data = [
        ['Exit Rate', 'Accuracy (%)', 'Time/Sample (s)', 'Time Saved (%)', 'Actual Exit Rate (%)'],
    ]
    
    for rate in exit_rates:
        row = [
            f"{rate:.1f}",
            f"{results[rate]['accuracy']:.2f}",
            f"{results[rate]['time_per_sample']:.6f}",
            f"{results[rate]['time_saved_percent']:.2f}",
            f"{results[rate]['actual_exit_rate']:.2f}"
        ]
        table_data.append(row)
    
    # Add baseline
    table_data.append(['Baseline', f"{baseline_acc:.2f}", f"{baseline_time:.6f}", "0.00", "0.00"])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    plt.title('Improved Early Exit Performance Metrics')
    plt.tight_layout()
    
    # Save table figure
    plt.savefig(os.path.join(output_dir, "improved_early_exit_metrics_table.png"))
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate improved early exit model with forced exit rates")
    parser.add_argument("--model-path", type=str, 
                        default="models_optimized/improved_early_exit.pth",
                        help="Path to the trained early exit model")
    parser.add_argument("--data-dir", type=str, 
                        default="data/augmented",
                        help="Path to the dataset directory")
    parser.add_argument("--batch-size", type=int, default=12,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for evaluation ('cuda' or 'cpu')")
    
    args = parser.parse_args()
    
    run_forced_evaluation(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=args.device
    )
